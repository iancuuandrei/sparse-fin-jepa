"""Manifest-driven orchestration for the locked historical paper stages."""

from __future__ import annotations

import gc
import os
import tempfile
from collections.abc import Iterator
from dataclasses import asdict
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from execsim.data.paper.corporate_action_manifest import (
    ingest_corporate_actions,
    write_corporate_action_manifest,
)
from execsim.data.paper.formation import (
    build_formation_candidates_from_corpus,
    ingest_constituent_snapshot,
    write_formation_receipts,
)
from execsim.data.paper.formation_v2 import (
    build_daily_formation_candidates,
    select_v2_universe,
    write_v2_universe_manifest,
)
from execsim.data.paper.identity import resolve_provider_symbol, validate_symbol_history
from execsim.data.paper.manifests import file_sha256, read_json, stable_hash, write_json_atomic
from execsim.data.paper.resolution_quality import assess_session_resolution_quality
from execsim.data.paper.schemas import InstrumentSymbolInterval
from execsim.data.paper.universe import select_frozen_universe, write_universe_manifest
from execsim.data.paper.validation import validate_exact_xnys_session
from execsim.ml.paper.configs import PaperRunConfig, PaperRuntimeApproval
from execsim.ml.paper.evaluation_execution import (
    evaluation_report_root,
    evaluation_root,
    representation_root,
    verify_evaluation_execution,
)
from execsim.ml.sequences.corpus import (
    build_fold_sequence_corpus,
    build_fold_sequence_corpus_from_root,
    load_corpus_instrument,
)

if TYPE_CHECKING:
    from execsim.ml.models.lightgbm_adapter import LightGBMExecutionOptions


def build_universe_stage(config: PaperRunConfig) -> dict[str, object]:
    """Produce candidate statistics, receipts, and the frozen sourced-identity universe."""
    if config.paper_run_id == "sparse-jepa-v2":
        return _build_v2_universe_stage(config)
    snapshot_path = Path(config.data["constituent_snapshot"])
    formation_root = Path(config.data["formation_corpus_root"])
    ticker_path = Path(config.data["ticker_history"])
    for path in (snapshot_path, formation_root, ticker_path):
        if not path.exists():
            raise RuntimeError(f"BLOCKED: required formation input is unavailable: {path}")
    snapshot = ingest_constituent_snapshot(snapshot_path)
    import exchange_calendars as xcals

    calendar = xcals.get_calendar("XNYS")
    formation_start, formation_end = config.data["formation_period"]
    expected = _expected_primary_session_count(calendar, formation_start, formation_end)
    candidates, exclusions = build_formation_candidates_from_corpus(
        snapshot, formation_root, expected_session_count=expected
    )
    artifact_root = config.artifact_root / "formation"
    artifact_root.mkdir(parents=True, exist_ok=True)
    candidates_path = artifact_root / "candidates.parquet"
    candidates.to_parquet(candidates_path, index=False)
    receipts_path = write_formation_receipts(
        artifact_root / "receipts.json",
        snapshot_path=snapshot_path,
        candidates=candidates,
        exclusions=exclusions,
        paper_config_hash=config.config_hash,
    )
    members = select_frozen_universe(candidates)
    history_frame = pd.read_parquet(ticker_path)
    intervals = _symbol_intervals(history_frame)
    validate_symbol_history(intervals)
    member_ids = {member.instrument_id for member in members}
    member_ids.add(str(config.data["spy_instrument_id"]))
    retained = tuple(item for item in intervals if item.instrument_id in member_ids)
    missing = member_ids.difference(item.instrument_id for item in retained)
    if missing:
        raise RuntimeError(
            f"BLOCKED: sourced ticker history is missing instruments: {sorted(missing)}"
        )
    output = Path(config.data["universe_manifest"])
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = write_universe_manifest(
        members,
        output=str(output),
        source_hashes=(
            file_sha256(snapshot_path),
            file_sha256(ticker_path),
            file_sha256(candidates_path),
        ),
        symbol_history=tuple(
            {
                "instrument_id": item.instrument_id,
                "symbol": item.symbol,
                "start": item.start.isoformat(),
                "end": item.end.isoformat(),
                "source": item.source,
            }
            for item in retained
        ),
        paper_config_hash=config.config_hash,
    )
    return {
        "status": "SOFTWARE READY",
        "members": len(members),
        "candidates": str(candidates_path),
        "receipts": str(receipts_path),
        "manifest": manifest,
    }


def _build_v2_universe_stage(config: PaperRunConfig) -> dict[str, object]:
    """Build the v2 universe only from direct daily formation observations."""
    snapshot_path = Path(config.data["constituent_snapshot"])
    daily_path = Path(config.data["formation_daily_corpus"])
    ticker_path = Path(config.data["ticker_history"])
    for path in (snapshot_path, daily_path, ticker_path):
        if not path.is_file():
            raise RuntimeError(f"BLOCKED: required v2 formation input is unavailable: {path}")
    snapshot = ingest_constituent_snapshot(snapshot_path)
    daily = pd.read_parquet(daily_path)
    import exchange_calendars as xcals

    formation_start, formation_end = config.data["formation_period"]
    expected = tuple(
        value.date()
        for value in xcals.get_calendar("XNYS").sessions_in_range(formation_start, formation_end)
    )
    candidates = build_daily_formation_candidates(
        snapshot,
        daily,
        expected_session_dates=expected,
        identity_source_hash=file_sha256(ticker_path),
    )
    artifact_root = config.artifact_root / "formation"
    artifact_root.mkdir(parents=True, exist_ok=True)
    candidates_path = artifact_root / "candidates-v2.parquet"
    candidates.to_parquet(candidates_path, index=False)
    eligible = (
        (candidates["security_type"] == "ordinary_common_stock")
        & candidates["in_sp500_on_formation_date"].astype(bool)
        & (candidates["median_daily_price"] >= 5.0)
        & (
            candidates["daily_completeness"]
            >= float(config.data["formation_daily_completeness_minimum"])
        )
        & (candidates["median_daily_dollar_volume"] > 0)
        & candidates["instrument_id"].astype(str).str.len().gt(0)
    )
    eligibility_path = artifact_root / "eligibility-v2.json"
    write_json_atomic(
        eligibility_path,
        {
            "schema_version": "paper-v2-formation-eligibility-v1",
            "protocol_id": config.paper_run_id,
            "paper_config_hash": config.config_hash,
            "candidate_table_sha256": file_sha256(candidates_path),
            "candidate_count": len(candidates),
            "eligible_count": int(eligible.sum()),
            "exclusions": [
                {
                    "instrument_id": str(row.instrument_id),
                    "formation_symbol": str(row.formation_symbol),
                    "reasons": str(row.exclusion_reasons),
                }
                for row in candidates.loc[~eligible].itertuples(index=False)
            ],
        },
    )
    members = select_v2_universe(candidates, size=int(config.data["universe_size"]))
    intervals = _symbol_intervals(pd.read_parquet(ticker_path))
    validate_symbol_history(intervals)
    member_ids = {member.instrument_id for member in members}
    member_ids.add(str(config.data["spy_instrument_id"]))
    retained = tuple(item for item in intervals if item.instrument_id in member_ids)
    missing = member_ids.difference(item.instrument_id for item in retained)
    if missing:
        raise RuntimeError(
            f"BLOCKED: sourced ticker history is missing instruments: {sorted(missing)}"
        )
    output = Path(config.data["universe_manifest"])
    manifest_path = write_v2_universe_manifest(
        members,
        output=output,
        source_hashes=(
            file_sha256(snapshot_path),
            file_sha256(ticker_path),
            file_sha256(daily_path),
            file_sha256(candidates_path),
        ),
        symbol_history=tuple(
            {
                "instrument_id": item.instrument_id,
                "symbol": item.symbol,
                "start": item.start.isoformat(),
                "end": item.end.isoformat(),
                "source": item.source,
            }
            for item in retained
        ),
        paper_config_hash=config.config_hash,
    )
    return {
        "status": "SOFTWARE READY",
        "members": len(members),
        "candidates": str(candidates_path),
        "eligibility_receipt": str(eligibility_path),
        "manifest": str(manifest_path),
    }


def download_data_stage(
    config: PaperRunConfig,
    *,
    cli_enabled: bool,
    runtime_approval: PaperRuntimeApproval | None,
) -> dict[str, object]:
    """Acquire formation candidates, freeze the universe, then acquire target bars."""
    from execsim.data.paper.acquisition import (
        acquire_chunk,
        create_alpaca_sip_fetcher,
        monthly_chunks,
        probe_alpaca_sip_entitlement,
    )
    from execsim.data.paper.planning import build_acquisition_plan
    from execsim.data.paper.schemas import PaperDataConfig
    from execsim.data.paper.sources import acquire_constituent_identity_sources

    config.authorize("target_acquisition", approval=runtime_approval, cli_enabled=cli_enabled)
    snapshot_path = Path(config.data["constituent_snapshot"])
    ticker_path = Path(config.data["ticker_history"])
    formation_start = _as_date(config.data["formation_period"][0])
    formation_end = _as_date(config.data["formation_period"][1])
    target_start = _as_date(config.data["target_period"][0])
    target_end = _as_date(config.data["target_period"][1])
    if not snapshot_path.is_file() or not ticker_path.is_file():
        acquire_constituent_identity_sources(
            formation_date=formation_start,
            target_end=target_end,
            snapshot_output=snapshot_path,
            ticker_history_output=ticker_path,
            receipt_output=config.artifact_root / "acquisition" / "formation-source.json",
            spy_instrument_id=str(config.data["spy_instrument_id"]),
        )
    snapshot = ingest_constituent_snapshot(snapshot_path)
    intervals = _symbol_intervals(pd.read_parquet(ticker_path))
    validate_symbol_history(intervals)
    data = PaperDataConfig(
        provider=cast(Any, config.data["provider"]),
        feed=cast(Any, config.data["feed"]),
        frequency=cast(Any, config.data["frequency"]),
        timezone=cast(Any, config.data["timezone"]),
        adjustment=cast(Any, config.data["adjustment"]),
        extended_hours=bool(config.data["extended_hours"]),
        formation_start=formation_start,
        formation_end=formation_end,
        target_start=target_start,
        target_end=target_end,
        allow_network=True,
        paper_config_hash=config.config_hash,
    )
    acquisition_root = config.artifact_root / "acquisition"
    plan = build_acquisition_plan(
        snapshot=snapshot,
        intervals=intervals,
        formation_start=formation_start,
        formation_end=formation_end,
        target_start=target_start,
        target_end=target_end,
        target_universe_size=int(config.data["universe_size"]),
        spy_instrument_id=str(config.data["spy_instrument_id"]),
        output_directory=acquisition_root,
        paper_config_hash=config.config_hash,
        protocol_id=config.paper_run_id,
        formation_frequency=str(config.data.get("formation_frequency", "1min")),
    )
    probe_path = acquisition_root / "alpaca-sip-probe.json"
    probe = (
        read_json(probe_path)
        if probe_path.is_file()
        else probe_alpaca_sip_entitlement(data, cli_enabled=True, output=probe_path)
    )
    if probe.get("paper_config_hash") != config.config_hash or probe.get("status") != "PASS":
        raise ValueError("Existing Alpaca SIP probe is incompatible with this paper run.")
    fetcher = create_alpaca_sip_fetcher()
    spy_id = str(config.data["spy_instrument_id"])
    if config.paper_run_id == "sparse-jepa-v2":
        from execsim.data.paper.daily_acquisition import acquire_formation_daily_bars

        if _has_frozen_v2_formation_evidence(config):
            formation_chunks: int | dict[str, object] = {
                "status": "reused_frozen_v2_formation_evidence"
            }
        else:
            formation_chunks = acquire_formation_daily_bars(
                snapshot,
                formation_start=formation_start,
                formation_end=formation_end,
                spy_instrument_id=spy_id,
                output_path=Path(config.data["formation_daily_corpus"]),
                receipt_path=Path(config.data["formation_daily_receipt"]),
                paper_config_hash=config.config_hash,
                cli_enabled=True,
                config_enabled=True,
            )
        formation_output = str(config.data["formation_daily_corpus"])
    else:
        formation_ids = tuple(dict.fromkeys((*snapshot["instrument_id"].astype(str), spy_id)))
        formation_chunks = _acquire_period(
            formation_ids,
            intervals,
            start=data.formation_start,
            end=data.formation_end,
            output=Path(config.data["formation_corpus_root"]),
            fetcher=fetcher,
            data=data,
            acquire_chunk=acquire_chunk,
            monthly_chunks=monthly_chunks,
        )
        formation_output = str(config.data["formation_corpus_root"])
    universe_path = Path(config.data["universe_manifest"])
    universe_result: dict[str, object] | str = "reused"
    if not _is_frozen_universe(universe_path, config_hash=config.config_hash):
        universe_result = build_universe_stage(config)
    universe = read_json(universe_path)
    target_ids = tuple(
        dict.fromkeys((*[str(member["instrument_id"]) for member in universe["members"]], spy_id))
    )
    from execsim.data.paper.corporate_action_acquisition import acquire_split_actions

    action_source = Path(config.data["corporate_action_source"])
    corporate_actions = acquire_split_actions(
        intervals,
        target_ids,
        start=formation_start,
        end=target_end,
        output_path=action_source,
        raw_output_path=action_source.with_suffix(".raw.json"),
        receipt_path=action_source.with_name("acquisition-receipt.json"),
        paper_config_hash=config.config_hash,
        config=data,
        cli_enabled=True,
    )
    target_chunks = _acquire_period(
        target_ids,
        intervals,
        start=data.target_start,
        end=data.target_end,
        output=Path(config.data["target_corpus_root"]),
        fetcher=fetcher,
        data=data,
        acquire_chunk=acquire_chunk,
        monthly_chunks=monthly_chunks,
    )
    target_audit = _audit_acquisition_period(
        target_ids,
        intervals,
        start=data.target_start,
        end=data.target_end,
        output=Path(config.data["target_corpus_root"]),
        monthly_chunks=monthly_chunks,
        paper_config_hash=config.config_hash,
    )
    return {
        "status": "SOFTWARE READY",
        "formation_chunks": formation_chunks,
        "target_chunks": target_chunks,
        "target_audit": target_audit,
        "acquisition_plan": plan,
        "provider_probe": probe,
        "corporate_actions": corporate_actions,
        "universe": universe_result,
        "formation_output": formation_output,
        "target_output": str(config.data["target_corpus_root"]),
    }


def _expected_primary_session_count(calendar: Any, start: object, end: object) -> int:
    """Count only full 390-minute XNYS sessions eligible for the primary corpus."""
    return sum(
        len(calendar.session_minutes(session)) == 390
        for session in calendar.sessions_in_range(start, end)
    )


def validate_data_stage(config: PaperRunConfig, source: Path | None = None) -> dict[str, object]:
    """Validate target sessions under the configured representation-quality protocol."""
    root = source or Path(config.data["target_corpus_root"])
    universe = read_json(Path(config.data["universe_manifest"]))
    symbol_intervals = _paper_symbol_intervals(config, universe)
    allowed_instruments = {
        *(str(member["instrument_id"]) for member in universe.get("members", ())),
        str(config.data["spy_instrument_id"]),
    }
    protocol = str(config.sequences.get("quality_protocol", "exact-minute-v1"))
    errors = []
    quality_rows = []
    valid = 0
    if root.is_dir():

        def iter_frames() -> Iterator[pd.DataFrame]:
            for instrument_id in sorted(allowed_instruments):
                try:
                    yield load_corpus_instrument(root, instrument_id)
                except FileNotFoundError as exc:
                    raise RuntimeError(
                        f"BLOCKED: target corpus is missing instrument {instrument_id}."
                    ) from exc

        frames = iter_frames()
    else:
        frames = iter((_load_parquet_corpus(root),))
    for frame in frames:
        timestamps = pd.to_datetime(frame["timestamp"])
        dates = timestamps.dt.tz_convert("America/New_York").dt.date
        for (instrument, session_date), session in frame.groupby(
            [frame["instrument_id"].astype(str), dates], sort=True
        ):
            identity_errors: list[str] = []
            if instrument not in allowed_instruments:
                identity_errors.append("instrument is not in the frozen universe or SPY")
            observed_symbols = tuple(session["symbol"].astype(str).str.upper().drop_duplicates())
            if len(observed_symbols) != 1:
                identity_errors.append("session must contain one observed symbol")
            else:
                try:
                    expected_symbol = resolve_provider_symbol(
                        symbol_intervals, instrument, session_date
                    )
                except RuntimeError as exc:
                    identity_errors.append(str(exc))
                else:
                    if observed_symbols[0] != expected_symbol:
                        identity_errors.append(
                            f"observed symbol {observed_symbols[0]} does not match "
                            f"sourced symbol {expected_symbol}"
                        )
            session_errors: tuple[str, ...]
            if protocol == "resolution-aware-v2":
                quality = assess_session_resolution_quality(session)
                quality_rows.append(quality.to_dict())
                session_errors = (
                    () if quality.token_valid_full_session else (quality.invalid_token_reason,)
                )
            elif protocol == "exact-minute-v1":
                session_errors = validate_exact_xnys_session(session)
            else:
                raise ValueError(f"Unknown paper quality protocol: {protocol}")
            session_errors = (*identity_errors, *session_errors)
            if session_errors:
                errors.append(
                    {
                        "instrument_id": instrument,
                        "session_date": session_date.isoformat(),
                        "errors": session_errors,
                    }
                )
            else:
                valid += 1
    return {
        "valid": not errors,
        "quality_protocol": protocol,
        "valid_sessions": valid,
        "invalid_sessions": errors,
        "session_quality": quality_rows,
    }


def build_sequences_stage(config: PaperRunConfig, source: Path | None = None) -> dict[str, object]:
    """Build complete train/validation/test stores for every locked fold."""
    universe_path = Path(config.data["universe_manifest"])
    action_source = Path(config.data["corporate_action_source"])
    if not universe_path.is_file() or not action_source.is_file():
        raise RuntimeError("BLOCKED: universe or sourced corporate-action input is unavailable.")
    universe = read_json(universe_path)
    if universe.get("paper_config_hash") != config.config_hash:
        raise ValueError("Universe manifest was built under a different paper configuration.")
    actions = ingest_corporate_actions(action_source)
    action_manifest_path = Path(config.data["corporate_action_manifest"])
    action_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    write_corporate_action_manifest(
        action_source, actions, action_manifest_path, paper_config_hash=config.config_hash
    )
    corpus_source = source or Path(config.data["target_corpus_root"])
    bars = None if corpus_source.is_dir() else _load_parquet_corpus(corpus_source)
    manifests = []
    for fold in config.evaluation["folds"]:
        kwargs = {
            "universe_members": tuple(universe["members"]),
            "corporate_actions": actions,
            "fold_id": str(fold["id"]),
            "output_root": config.artifact_root / "sequences",
            "universe_manifest_hash": file_sha256(universe_path),
            "corporate_action_manifest_hash": file_sha256(action_manifest_path),
            "config_hash": config.config_hash,
            "spy_instrument_id": str(config.data["spy_instrument_id"]),
            "data_classification": "historical",
            "quality_protocol": str(config.sequences["quality_protocol"]),
            "symbol_history": tuple(
                {
                    "instrument_id": item.instrument_id,
                    "symbol": item.symbol,
                    "start": item.start.isoformat(),
                    "end": item.end.isoformat(),
                    "source": item.source,
                }
                for item in _paper_symbol_intervals(config, universe)
            ),
        }
        built = (
            build_fold_sequence_corpus_from_root(corpus_source, **kwargs)
            if bars is None
            else build_fold_sequence_corpus(bars, **kwargs)
        )
        manifests.append(asdict(built))
    return {"status": "SOFTWARE READY", "folds": manifests}


def validate_sequences_stage(config: PaperRunConfig) -> dict[str, object]:
    """Verify all fold manifests, indexes, partitions, and upstream config identity."""
    from execsim.ml.sequences.streaming import PaperSequenceDataset

    rows = []
    for fold in config.evaluation["folds"]:
        path = config.artifact_root / "sequences" / str(fold["id"]) / "sequence-manifest.json"
        payload = read_json(path)
        if payload["config_hash"] != config.config_hash:
            raise ValueError(f"Sequence manifest config mismatch: {path}")
        counts = {}
        for partition in ("train", "validation", "test"):
            dataset = PaperSequenceDataset(path, partition=partition, seed=13)
            counts[partition] = len(dataset)
        rows.append({"fold_id": fold["id"], "samples": counts, "manifest": str(path)})
    return {"valid": True, "folds": rows}


def select_rdm_lambda_stage(
    config: PaperRunConfig,
    *,
    training_cli_enabled: bool,
    runtime_approval: PaperRuntimeApproval | None,
    trusted_local_resume: bool = False,
) -> dict[str, object]:
    """Run the six predeclared Fold 1 candidates and freeze one common coefficient."""
    config.authorize(
        "historical_training",
        approval=runtime_approval,
        cli_enabled=training_cli_enabled,
    )
    import torch

    from execsim.ml.representations.checkpoints import load_checkpoint
    from execsim.ml.representations.historical_trainer import (
        HistoricalTrainerOptions,
        HistoricalTrainingIdentity,
        HistoricalTrainingRejected,
        train_historical_representation,
    )
    from execsim.ml.representations.jepa import PredictiveRepresentationModel
    from execsim.ml.representations.schemas import CheckpointCompatibility, RepresentationConfig
    from execsim.ml.representations.selection import (
        CommonLambdaCandidate,
        select_common_rdm_lambda,
        streaming_observable_probe_error,
    )
    from execsim.ml.sequences.streaming import PaperSequenceDataset, build_sequence_dataloader

    values = config.representation
    fold = next(item for item in config.evaluation["folds"] if item["id"] == "fold-1")
    sequence_path = config.artifact_root / "sequences" / "fold-1" / "sequence-manifest.json"
    sequence = read_json(sequence_path)
    options = HistoricalTrainerOptions(
        batch_size=int(values["batch_size"]),
        num_workers=int(config.sequences["num_workers"]),
        prefetch_factor=int(config.sequences["prefetch_factor"]),
        cache_size=int(config.sequences["session_cache_size"]),
        max_epochs=int(values["max_epochs"]),
        patience=int(values["early_stopping_patience"]),
        learning_rate=float(values["learning_rate"]),
        weight_decay=float(values["weight_decay"]),
        warmup_fraction=float(values["warmup_fraction"]),
        gradient_clip=float(values["gradient_clip"]),
        checkpoint_interval_steps=int(values["checkpoint_interval_steps"]),
        diagnostic_sample_rows=int(values["rdm_diagnostic_sample_rows"]),
    )
    identity = HistoricalTrainingIdentity(
        fold_id="fold-1",
        cutoff=str(fold["train"][1]),
        universe_manifest_hash=str(sequence["universe_manifest_hash"]),
        dataset_manifest_hash=stable_hash(sequence["raw_hashes"]),
        normalization_hash=stable_hash(sequence["normalization"]),
        architecture_hash=stable_hash(
            {"observed_dynamic_conditioning_latent_context": [18, 13, 5, 128, 8]}
        ),
        config_hash=config.config_hash,
        code_commit=_git_head(),
    )
    candidates = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for rdm_lambda in values["rdm_lambda_candidates"]:
        for geometry in ("dense", "sparse"):
            target = values[f"{geometry}_target"]
            representation = RepresentationConfig(
                geometry,
                generalized_gaussian_p=float(target["p"]),
                generalized_gaussian_mu=float(target["mu"]),
                generalized_gaussian_sigma=float(target["sigma"]),
                rdm_projections_train=int(values["rdm_projections_train"]),
                rdm_projections_evaluation=int(values["rdm_projections_evaluation"]),
                seed=13,
            )
            root = config.artifact_root / "selection" / f"lambda={rdm_lambda}" / geometry
            final_manifest = root / "final" / "manifest.json"
            failure_path = root / "training-failure.json"
            if not final_manifest.is_file() and not failure_path.is_file():
                resume_from = _latest_periodic_checkpoint(root)
                if resume_from is not None and not trusted_local_resume:
                    raise RuntimeError(
                        "BLOCKED: a checksummed local periodic resume exists; pass "
                        "--trust-local-resume to load its pickle state."
                    )
                try:
                    train_historical_representation(
                        sequence_path,
                        representation=representation,
                        identity=identity,
                        output_root=root,
                        allow_historical_training=True,
                        options=options,
                        rdm_lambda=float(rdm_lambda),
                        resume_from=resume_from,
                        trusted_resume=trusted_local_resume,
                    )
                except HistoricalTrainingRejected:
                    if not failure_path.is_file():
                        raise RuntimeError(
                            "Rejected historical candidate did not write its failure receipt."
                        ) from None
            if not final_manifest.is_file():
                failure = read_json(failure_path)
                expected_failure = {
                    "schema_version": "historical-training-rejection-v1",
                    "status": "REJECTED_BY_COLLAPSE_GATE",
                    "fold_id": "fold-1",
                    "geometry": geometry,
                    "seed": 13,
                    "rdm_lambda": float(rdm_lambda),
                    "paper_config_hash": config.config_hash,
                    "sequence_manifest_hash": file_sha256(sequence_path),
                    "training_config_hash": stable_hash(
                        {
                            "representation": asdict(representation),
                            "trainer": asdict(options),
                            "common_rdm_lambda": float(rdm_lambda),
                        }
                    ),
                    "code_commit": _git_head(),
                }
                if any(failure.get(key) != value for key, value in expected_failure.items()):
                    raise ValueError("Rejected RDM candidate receipt is incompatible.")
                reasons = tuple(str(value) for value in failure.get("collapse_gate_failures", ()))
                if not reasons:
                    raise ValueError("Rejected RDM candidate receipt has no collapse-gate reason.")
                candidates.append(
                    CommonLambdaCandidate(
                        float(rdm_lambda),
                        geometry,
                        "fold-1",
                        13,
                        0.0,
                        "FAIL",
                        "",
                        "; ".join(reasons),
                    )
                )
                continue
            expected = CheckpointCompatibility(**read_json(root / "compatibility.json"))
            _assert_representation_reuse(
                expected,
                representation,
                identity,
                sequence_path,
                options,
                float(rdm_lambda),
            )
            model = PredictiveRepresentationModel(representation).to(device)
            manifest = load_checkpoint(model, root / "final", expected=expected)
            train_data = PaperSequenceDataset(
                sequence_path,
                partition="train",
                seed=13,
                cache_size=options.cache_size,
                sample_train_positions=False,
            )
            valid_data = PaperSequenceDataset(
                sequence_path,
                partition="validation",
                seed=13,
                cache_size=options.cache_size,
            )
            error = streaming_observable_probe_error(
                model,
                build_sequence_dataloader(
                    train_data,
                    batch_size=options.batch_size,
                    num_workers=options.num_workers,
                    device=device,
                    prefetch_factor=options.prefetch_factor,
                ),
                build_sequence_dataloader(
                    valid_data,
                    batch_size=options.batch_size,
                    num_workers=options.num_workers,
                    device=device,
                    prefetch_factor=options.prefetch_factor,
                ),
                device=device,
            )
            candidates.append(
                CommonLambdaCandidate(
                    float(rdm_lambda),
                    geometry,
                    "fold-1",
                    13,
                    error,
                    manifest.collapse_gate_status,
                    manifest.weights_sha256,
                )
            )
    output = config.artifact_root / "selection" / "rdm-lambda.json"
    selected = select_common_rdm_lambda(
        tuple(candidates), output=output, paper_config_hash=config.config_hash
    )
    freeze = _freeze_representation_parameters(config)
    return {
        "status": "SOFTWARE READY",
        "selected_rdm_lambda": selected,
        "receipt": str(output),
        "parameter_freeze": str(freeze["path"]),
    }


def train_representations_stage(
    config: PaperRunConfig,
    *,
    training_cli_enabled: bool,
    runtime_approval: PaperRuntimeApproval | None,
    trusted_local_resume: bool = False,
) -> dict[str, object]:
    """Train the locked fold/seed/dense-sparse matrix using streaming historical loaders."""
    from execsim.ml.representations.checkpoints import load_checkpoint
    from execsim.ml.representations.historical_trainer import (
        HistoricalTrainerOptions,
        HistoricalTrainingIdentity,
        train_historical_representation,
    )
    from execsim.ml.representations.jepa import PredictiveRepresentationModel
    from execsim.ml.representations.schemas import CheckpointCompatibility, RepresentationConfig

    config.authorize(
        "historical_training",
        approval=runtime_approval,
        cli_enabled=training_cli_enabled,
    )
    values = config.representation
    options = HistoricalTrainerOptions(
        batch_size=int(values["batch_size"]),
        num_workers=int(config.sequences["num_workers"]),
        prefetch_factor=int(config.sequences["prefetch_factor"]),
        cache_size=int(config.sequences["session_cache_size"]),
        max_epochs=int(values["max_epochs"]),
        patience=int(values["early_stopping_patience"]),
        learning_rate=float(values["learning_rate"]),
        weight_decay=float(values["weight_decay"]),
        warmup_fraction=float(values["warmup_fraction"]),
        gradient_clip=float(values["gradient_clip"]),
        checkpoint_interval_steps=int(values["checkpoint_interval_steps"]),
        diagnostic_sample_rows=int(values["rdm_diagnostic_sample_rows"]),
    )
    selection_path = config.artifact_root / "selection" / "rdm-lambda.json"
    if not selection_path.is_file():
        select_rdm_lambda_stage(
            config,
            training_cli_enabled=training_cli_enabled,
            runtime_approval=runtime_approval,
            trusted_local_resume=trusted_local_resume,
        )
    representation_freeze = (
        config.artifact_root / "selection" / "representation-parameter-freeze-v1.json"
    )
    if not representation_freeze.is_file():
        _freeze_representation_parameters(config)
    _require_representation_parameter_freeze(config)
    selection = _load_common_lambda_receipt(config)
    common_rdm_lambda = float(cast(Any, selection["selected_rdm_lambda"]))
    results = []
    for fold in config.evaluation["folds"]:
        fold_id = str(fold["id"])
        sequence_path = config.artifact_root / "sequences" / fold_id / "sequence-manifest.json"
        sequence = read_json(sequence_path)
        identity = HistoricalTrainingIdentity(
            fold_id=fold_id,
            cutoff=str(fold["train"][1]),
            universe_manifest_hash=str(sequence["universe_manifest_hash"]),
            dataset_manifest_hash=stable_hash(sequence["raw_hashes"]),
            normalization_hash=stable_hash(sequence["normalization"]),
            architecture_hash=stable_hash(
                {
                    "encoder": "linear-layernorm-gelu-linear",
                    "observed_dynamic_conditioning_latent_context": [18, 13, 5, 128, 8],
                }
            ),
            config_hash=config.config_hash,
            code_commit=_git_head(),
        )
        for geometry in values["geometries"]:
            target = values[f"{geometry}_target"]
            for seed in values["seeds"]:
                representation = RepresentationConfig(
                    str(geometry),  # type: ignore[arg-type]
                    predictor_family=str(values["predictor_family"]),  # type: ignore[arg-type]
                    generalized_gaussian_p=float(target["p"]),
                    generalized_gaussian_mu=float(target["mu"]),
                    generalized_gaussian_sigma=float(target["sigma"]),
                    rdm_projections_train=int(values["rdm_projections_train"]),
                    rdm_projections_evaluation=int(values["rdm_projections_evaluation"]),
                    seed=int(seed),
                )
                output = (
                    config.artifact_root / "representations" / fold_id / str(geometry) / str(seed)
                )
                if (output / "final" / "manifest.json").is_file():
                    expected = CheckpointCompatibility(**read_json(output / "compatibility.json"))
                    _assert_representation_reuse(
                        expected,
                        representation,
                        identity,
                        sequence_path,
                        options,
                        common_rdm_lambda,
                    )
                    loaded = load_checkpoint(
                        PredictiveRepresentationModel(representation),
                        output / "final",
                        expected=expected,
                    )
                    if loaded.seed != int(seed):
                        raise ValueError("Reusable checkpoint seed does not match the run.")
                    results.append(
                        {"fold_id": fold_id, "geometry": geometry, "seed": seed, "status": "reused"}
                    )
                    continue
                trained = train_historical_representation(
                    sequence_path,
                    representation=representation,
                    identity=identity,
                    output_root=output,
                    allow_historical_training=True,
                    options=options,
                    rdm_lambda=common_rdm_lambda,
                    resume_from=_latest_periodic_checkpoint(output),
                    trusted_resume=trusted_local_resume,
                )
                results.append(asdict(trained))
    return {
        "status": "SOFTWARE READY",
        "runs": results,
        "future_difficulty_adaptation": "EXCLUDED FROM PAPER MATRIX",
    }


def export_embeddings_stage(config: PaperRunConfig) -> dict[str, object]:
    """Export every fold/seed/geometry checkpoint through the batched historical path."""
    import torch

    from execsim.ml.representations.embedding_pipeline import export_embedding_corpus
    from execsim.ml.representations.jepa import PredictiveRepresentationModel
    from execsim.ml.representations.schemas import CheckpointCompatibility, RepresentationConfig

    results = []
    for fold in config.evaluation["folds"]:
        fold_id = str(fold["id"])
        sequence_path = config.artifact_root / "sequences" / fold_id / "sequence-manifest.json"
        export_variants = (("dense", "none"), ("sparse", "none"))
        for storage_geometry, adaptation in export_variants:
            for seed in config.representation["seeds"]:
                run_root = (
                    config.artifact_root
                    / "representations"
                    / fold_id
                    / storage_geometry
                    / str(seed)
                )
                compatibility_path = run_root / "compatibility.json"
                if not compatibility_path.is_file():
                    raise RuntimeError(
                        f"BLOCKED: representation compatibility is missing: {run_root}"
                    )
                expected = CheckpointCompatibility(**read_json(compatibility_path))
                representation = RepresentationConfig(
                    expected.geometry,
                    predictor_family=expected.predictor_family,
                    generalized_gaussian_p=expected.generalized_gaussian_p,
                    generalized_gaussian_mu=expected.generalized_gaussian_mu,
                    generalized_gaussian_sigma=expected.generalized_gaussian_sigma,
                    rdm_projections_train=expected.rdm_projections,
                    rdm_projections_evaluation=int(
                        config.representation["rdm_projections_evaluation"]
                    ),
                    seed=int(seed),
                )
                output = (
                    config.artifact_root / "embeddings" / fold_id / storage_geometry / str(seed)
                )
                if (output / "manifest.json").is_file():
                    _validate_embedding_reuse(
                        output,
                        expected=expected,
                        sequence_path=sequence_path,
                        checkpoint_directory=run_root / "final",
                        seed=int(seed),
                        geometry=expected.geometry,
                        adaptation=adaptation,
                    )
                    results.append(
                        {
                            "fold_id": fold_id,
                            "geometry": storage_geometry,
                            "seed": seed,
                            "status": "reused",
                        }
                    )
                    continue
                manifest = export_embedding_corpus(
                    PredictiveRepresentationModel(representation),
                    checkpoint_directory=run_root / "final",
                    expected_checkpoint=expected,
                    sequence_manifest_path=sequence_path,
                    output_root=output,
                    seed=int(seed),
                    geometry=expected.geometry,
                    adaptation=adaptation,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    batch_size=int(config.representation["batch_size"]),
                    num_workers=int(config.sequences["num_workers"]),
                    cache_size=int(config.sequences["session_cache_size"]),
                )
                results.append(
                    {
                        "fold_id": fold_id,
                        "geometry": storage_geometry,
                        "seed": seed,
                        "manifest": str(manifest),
                    }
                )
    return {"status": "SOFTWARE READY", "exports": results}


def train_volume_models_stage(
    config: PaperRunConfig,
    *,
    training_cli_enabled: bool,
    runtime_approval: PaperRuntimeApproval | None,
    execution: LightGBMExecutionOptions | None = None,
    fold_id: str | None = None,
    input_identity: dict[str, object] | None = None,
) -> dict[str, object]:
    """Train the exact validation-only LightGBM grid for all locked feature rows."""
    config.authorize(
        "historical_training",
        approval=runtime_approval,
        cli_enabled=training_cli_enabled,
    )
    from execsim.data.paper.manifests import write_json_atomic
    from execsim.ml.models.lightgbm_adapter import (
        LightGBMConfig,
        LightGBMExecutionOptions,
        LightGBMVolumeModel,
        run_lightgbm_grid,
    )
    from execsim.ml.paper.lightgbm_data import (
        attach_lightgbm_embeddings,
        cached_lightgbm_base_frames,
    )

    execution_options = execution or LightGBMExecutionOptions()
    selected_folds = [
        fold for fold in config.evaluation["folds"] if fold_id is None or str(fold["id"]) == fold_id
    ]
    if not selected_folds:
        raise ValueError(f"Unknown LightGBM fold: {fold_id}")
    requested_fold = fold_id
    source_commit = _git_head()
    operational_receipts: dict[str, object] = {}
    if execution_options.device_type == "gpu":
        qualification_path = config.cache_root / "lightgbm-gpu" / "qualification-receipt.json"
        build_path = config.cache_root / "lightgbm-gpu" / "build-provenance.json"
        for name, path in (("qualification", qualification_path), ("build", build_path)):
            if not path.is_file():
                raise RuntimeError(f"BLOCKED: LightGBM GPU {name} receipt is missing: {path}")
        qualification = read_json(qualification_path)
        build = read_json(build_path)
        if (
            qualification.get("status") != "PASS"
            or qualification.get("source_commit") != source_commit
            or qualification.get("source_tree") != _git_tree()
            or qualification.get("paper_config_hash") != config.config_hash
            or qualification.get("execution")
            != {
                "device_type": execution_options.device_type,
                "gpu_platform_id": execution_options.gpu_platform_id,
                "gpu_device_id": execution_options.gpu_device_id,
                "gpu_use_dp": execution_options.gpu_use_dp,
                "selected_num_threads": execution_options.num_threads,
            }
            or build.get("status") != "PASS"
            or build.get("lightgbm_version") != "4.7.0"
        ):
            raise ValueError("LightGBM GPU qualification/build identity is incompatible.")
        operational_receipts = {
            "qualification_receipt_sha256": file_sha256(qualification_path),
            "build_provenance_sha256": file_sha256(build_path),
            "build_provenance": build,
        }
    execution_receipt_path = config.lightgbm_root
    if requested_fold is not None:
        execution_receipt_path /= requested_fold
    execution_receipt_path /= "execution-receipt.json"
    execution_receipt = {
        "schema_version": "paper-lightgbm-execution-v1",
        "paper_config_hash": config.config_hash,
        "git_commit": source_commit,
        "lightgbm_execution": execution_options.identity(),
        "cpu_deterministic_controls_applied": execution_options.device_type == "cpu",
        "determinism_policy": (
            "lightgbm-cpu-deterministic"
            if execution_options.device_type == "cpu"
            else "seeded-opencl-without-cpu-deterministic-guarantee"
        ),
        "scientific_grid_changed": False,
        "locked_test_or_tca_used": False,
        **operational_receipts,
        "fold_id": requested_fold,
        "input_identity": input_identity,
        "qualification": qualification if execution_options.device_type == "gpu" else None,
    }
    if execution_receipt_path.is_file():
        if read_json(execution_receipt_path) != execution_receipt:
            raise ValueError("Reusable LightGBM execution receipt is incompatible.")
    else:
        write_json_atomic(execution_receipt_path, execution_receipt)

    universe = read_json(config.data_path("universe_manifest"))
    liquidity = {
        str(member["instrument_id"]): int(member["liquidity_group"])
        for member in universe["members"]
    }
    results = []
    lightgbm_candidates = tuple(
        LightGBMConfig(
            num_leaves=int(leaves),
            min_child_samples=int(child),
            reg_lambda=float(l2),
            learning_rate=float(config.lightgbm["learning_rate"]),
            n_estimators=int(config.lightgbm["n_estimators"]),
            early_stopping_rounds=int(config.lightgbm["early_stopping_rounds"]),
            feature_fraction=float(config.lightgbm["feature_fraction"]),
            bagging_fraction=float(config.lightgbm["bagging_fraction"]),
            bagging_freq=int(config.lightgbm["bagging_freq"]),
        )
        for leaves in config.lightgbm["num_leaves"]
        for child in config.lightgbm["min_child_samples"]
        for l2 in config.lightgbm["reg_lambda"]
    )
    for fold in selected_folds:
        fold_id = str(fold["id"])
        sequence = config.sequence_root / fold_id / "sequence-manifest.json"
        # The raw causal rows, targets, weights, and identities are invariant across
        # representation variants. Build them once per fold/partition; only the
        # active coordinate receives a temporary 644-column representation block.
        base_training = cached_lightgbm_base_frames(
            sequence,
            partition="train",
            liquidity_groups=liquidity,
            cache_directory=config.cache_root / "lightgbm-base" / fold_id / "train",
            source_commit=source_commit,
            config_hash=config.config_hash,
        )
        base_validation = cached_lightgbm_base_frames(
            sequence,
            partition="validation",
            liquidity_groups=liquidity,
            cache_directory=config.cache_root / "lightgbm-base" / fold_id / "validation",
            source_commit=source_commit,
            config_hash=config.config_hash,
        )
        variants: list[tuple[str, int | None, Path | None]] = [("raw", None, None)]
        variants.append(("untrained_neural", None, None))
        for name in ("dense", "sparse"):
            variants.extend(
                (name, int(seed), config.embedding_root / fold_id / name / str(seed))
                for seed in config.representation["seeds"]
            )
        for method, seed, embedding_root in variants:
            output = config.lightgbm_root / fold_id / method / str(seed or "shared")
            if (output / "manifest.json").is_file():
                _, metadata = LightGBMVolumeModel.load_native(
                    output, expected_execution=execution_options
                )
                expected_metadata = {
                    "fold_id": fold_id,
                    "paper_config_hash": config.config_hash,
                    "sequence_manifest_hash": file_sha256(sequence),
                    "method": method,
                    "seed": seed,
                    "git_commit": source_commit,
                    "lightgbm_execution": execution_options.identity(),
                }
                mismatches = [
                    name for name, value in expected_metadata.items() if metadata.get(name) != value
                ]
                if mismatches:
                    raise ValueError(f"Reusable LightGBM identity mismatch: {sorted(mismatches)}")
                if file_sha256(output / "grid-results.json") != metadata.get("grid_results_sha256"):
                    raise ValueError("Reusable LightGBM grid checksum mismatch.")
                results.append(
                    {"fold_id": fold_id, "method": method, "seed": seed, "status": "reused"}
                )
                continue
            train_embedding = (
                embedding_root / "partition=train" / "embeddings.parquet"
                if embedding_root is not None
                else None
            )
            validation_embedding = (
                embedding_root / "partition=validation" / "embeddings.parquet"
                if embedding_root is not None
                else None
            )
            training_frames = base_training
            validation_frames = base_validation
            if train_embedding is not None and validation_embedding is not None:
                training_frames = attach_lightgbm_embeddings(
                    base_training, embedding_path=train_embedding
                )
                validation_frames = attach_lightgbm_embeddings(
                    base_validation, embedding_path=validation_embedding
                )
            training = training_frames.as_tuple()
            validation = validation_frames.as_tuple()
            if method == "untrained_neural":
                training = _append_untrained_control(training, fold_seed=13)
                validation = _append_untrained_control(validation, fold_seed=13)
            candidate_cache = (
                config.cache_root
                / "lightgbm-grid-cache"
                / config.config_hash
                / source_commit
                / fold_id
                / method
                / str(seed or "shared")
            )
            embedding_identity = {
                "train": file_sha256(train_embedding) if train_embedding is not None else None,
                "validation": (
                    file_sha256(validation_embedding) if validation_embedding is not None else None
                ),
            }
            model, candidates = run_lightgbm_grid(
                training,
                validation,
                categorical_features=tuple(config.lightgbm["categorical_features"]),
                seed=int(seed or 13),
                execution=execution_options,
                candidate_configs=tuple(
                    LightGBMConfig(**{**asdict(item), "seed": int(seed or 13)})
                    for item in lightgbm_candidates
                ),
                resume_directory=candidate_cache,
                resume_identity={
                    "schema_version": "paper-lightgbm-grid-resume-v1",
                    "training_cutoff": str(fold["train"][1]),
                    "validation_range": [str(value) for value in fold["validation"]],
                    "paper_config_hash": config.config_hash,
                    "git_commit": source_commit,
                    "fold_id": fold_id,
                    "sequence_manifest_hash": file_sha256(sequence),
                    "method": method,
                    "seed": seed,
                    "embedding_sha256": embedding_identity,
                    "execution_receipt_sha256": file_sha256(execution_receipt_path),
                },
            )
            metadata = {
                "fold_id": fold_id,
                "feature_schema_version": "paper-lgbm-residual-long-shape-v2",
                "training_cutoff": str(fold["train"][1]),
                "validation_range": [str(value) for value in fold["validation"]],
                "categorical_features": config.lightgbm["categorical_features"],
                "paper_config_hash": config.config_hash,
                "sequence_manifest_hash": file_sha256(sequence),
                "method": method,
                "seed": seed,
                "git_commit": source_commit,
                "embedding_sha256": embedding_identity,
                "execution_receipt_sha256": file_sha256(execution_receipt_path),
                "input_identity": input_identity,
            }
            output.parent.mkdir(parents=True, exist_ok=True)
            staging_parent = Path(tempfile.mkdtemp(prefix=".coordinate-", dir=output.parent))
            staging = staging_parent / "complete"
            model.save_native(staging, metadata)
            write_json_atomic(
                staging / "grid-results.json",
                {
                    "candidates": [
                        {
                            **asdict(item),
                            "config": asdict(item.config),
                        }
                        for item in candidates
                    ],
                    "selected_scale_config": asdict(model.scale_config),
                    "selected_shape_config": asdict(model.shape_config),
                    "lightgbm_execution": execution_options.identity(),
                    "selection_data": "validation_only",
                },
            )
            completed_manifest = read_json(staging / "manifest.json")
            completed_manifest["grid_results_sha256"] = file_sha256(staging / "grid-results.json")
            write_json_atomic(staging / "manifest.json", completed_manifest)
            os.replace(staging, output)
            staging_parent.rmdir()
            results.append(
                {"fold_id": fold_id, "method": method, "seed": seed, "artifact": str(output)}
            )
            # Each coordinate owns several multi-gigabyte historical frames. Release
            # them before constructing the next coordinate so peak memory does not
            # include both the completed and incoming feature matrices.
            del training, validation, training_frames, validation_frames, model, candidates
            gc.collect()
        del base_training, base_validation
        gc.collect()
    if requested_fold is not None:
        return {
            "status": "FOLD_TRAINING_COMPLETE",
            "fold_id": requested_fold,
            "models": results,
            "parameter_freeze_created": False,
            "locked_test_opened": False,
        }
    selection_receipt = config.artifact_root / "selection" / "rdm-lambda.json"
    model_manifests = sorted((config.artifact_root / "lightgbm").glob("*/*/*/manifest.json"))
    expected_model_count = len(config.evaluation["folds"]) * (
        2 + 2 * len(config.representation["seeds"])
    )
    if len(model_manifests) != expected_model_count:
        raise RuntimeError(
            "BLOCKED: parameter freeze requires the complete validation-selected LightGBM matrix."
        )
    freeze_path = config.artifact_root / "selection" / "parameter-freeze-v1.json"
    execution_receipt_file = config.artifact_root / "lightgbm" / "execution-receipt.json"
    if not execution_receipt_file.is_file():
        raise RuntimeError("BLOCKED: LightGBM execution receipt is missing from parameter freeze.")
    representation_manifests = sorted(
        (config.artifact_root / "representations").glob("*/*/*/final/manifest.json")
    )
    representation_commits = {
        str(read_json(path).get("code_commit")) for path in representation_manifests
    }
    if len(representation_manifests) != 18 or len(representation_commits) != 1:
        raise RuntimeError("BLOCKED: immutable representation source identity is incomplete.")
    freeze_payload = {
        "schema_version": "paper-parameter-selection-freeze-v1",
        "status": "PARAMETERS_FROZEN",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_head(),
        "git_tree": _git_tree(),
        "representation_source_commit": representation_commits.pop(),
        "paper_config_hash": config.config_hash,
        "lightgbm_execution_receipt_sha256": file_sha256(execution_receipt_file),
        "rdm_lambda_receipt_sha256": file_sha256(selection_receipt),
        "selected_rdm_lambda": read_json(selection_receipt)["selected_rdm_lambda"],
        "lightgbm_manifests": [
            {
                "path": str(path.relative_to(config.artifact_root)).replace("\\", "/"),
                "sha256": file_sha256(path),
            }
            for path in model_manifests
        ],
        "test_or_tca_used": False,
    }
    if freeze_path.is_file():
        existing_freeze = read_json(freeze_path)
        comparable = {
            name: value for name, value in freeze_payload.items() if name != "frozen_at_utc"
        }
        existing_comparable = {
            name: value for name, value in existing_freeze.items() if name != "frozen_at_utc"
        }
        if existing_comparable != comparable:
            raise ValueError(
                "Existing parameter-selection freeze does not match current artifacts."
            )
    else:
        write_json_atomic(freeze_path, freeze_payload)
    return {
        "status": "SOFTWARE READY",
        "models": results,
        "lightgbm_execution": execution_options.identity(),
        "execution_receipt": str(execution_receipt_path),
        "parameter_freeze": str(freeze_path),
    }


def _learned_ledger_identity(
    config: PaperRunConfig,
    fold_id: str,
    method: str,
    seed: int | None,
) -> dict[str, Any]:
    """Bind reused predictions to the same immutable inputs as their original call."""
    root = config.artifact_root
    embedding = (
        (
            root
            / "embeddings"
            / fold_id
            / method
            / str(seed)
            / "partition=test"
            / "embeddings.parquet"
        )
        if seed is not None
        else None
    )
    return {
        "schema_version": "paper-forecast-ledger-v2",
        "fold_id": fold_id,
        "method": method,
        "seed": seed,
        "paper_config_hash": config.config_hash,
        "source_commit": _git_head(),
        "source_tree": _git_tree(),
        "parameter_freeze_sha256": file_sha256(root / "selection" / "parameter-freeze-v1.json"),
        "model_manifest_sha256": file_sha256(
            root / "lightgbm" / fold_id / method / str(seed or "shared") / "manifest.json"
        ),
        "base_manifest_sha256": file_sha256(
            evaluation_root(config) / "evaluation-v2" / "bases" / fold_id / "manifest.json"
        ),
        "embedding_sha256": file_sha256(embedding) if embedding is not None else None,
    }


def evaluate_forecasts_stage(
    config: PaperRunConfig,
    *,
    full_run_cli_enabled: bool,
    runtime_approval: PaperRuntimeApproval | None,
) -> dict[str, object]:
    """Evaluate frozen LightGBM artifacts on locked test rows without model selection."""
    config.authorize(
        "locked_result_evaluation",
        approval=runtime_approval,
        cli_enabled=full_run_cli_enabled,
    )
    _require_parameter_freeze(config)
    _require_locked_test_opened(config)
    from execsim.ml.models.lightgbm_adapter import LightGBMVolumeModel
    from execsim.ml.paper.evaluation_artifacts import (
        evaluation_base,
        forecast_metric_frame,
        prediction_batches,
        publish_frames,
        verify_artifact,
    )
    from execsim.ml.paper.evaluation_workers import (
        EWMAWork,
        compact_profile_corpus,
        instrument_key,
        run_ewma_workers,
    )

    universe_path = _verify_frozen_universe_manifest(config)
    universe = read_json(universe_path)
    liquidity = {
        str(member["instrument_id"]): int(member["liquidity_group"])
        for member in universe["members"]
    }
    execution_identity = {
        "source_commit": _git_head(),
        "source_tree": _git_tree(),
        "paper_config_hash": config.config_hash,
        "parameter_freeze_sha256": file_sha256(
            config.artifact_root / "selection" / "parameter-freeze-v1.json"
        ),
    }
    market_inputs = compact_profile_corpus(
        config.data_path("target_corpus_root"),
        evaluation_root(config) / "evaluation-v2" / "profile-corpus",
        identity=execution_identity,
    )
    output_parts: list[Path] = []
    for fold in config.evaluation["folds"]:
        fold_id = str(fold["id"])
        sequence = config.artifact_root / "sequences" / fold_id / "sequence-manifest.json"
        variants: list[tuple[str, int | None, Path | None]] = [
            ("raw", None, None),
            ("untrained_neural", None, None),
        ]
        for name in ("dense", "sparse"):
            variants.extend(
                (name, int(seed), config.artifact_root / "embeddings" / fold_id / name / str(seed))
                for seed in config.representation["seeds"]
            )
        base = evaluation_base(
            sequence,
            partition="test",
            liquidity_groups=liquidity,
            directory=evaluation_root(config) / "evaluation-v2" / "bases" / fold_id,
            execution_identity={
                "source_commit": _git_head(),
                "source_tree": _git_tree(),
                "paper_config_hash": config.config_hash,
                "parameter_freeze_sha256": file_sha256(
                    config.artifact_root / "selection" / "parameter-freeze-v1.json"
                ),
            },
        )
        for method, seed, embedding_root in variants:
            embedding = (
                embedding_root / "partition=test" / "embeddings.parquet"
                if embedding_root is not None
                else None
            )
            model_root = (
                config.artifact_root / "lightgbm" / fold_id / method / str(seed or "shared")
            )
            ledger_root = (
                evaluation_root(config)
                / "evaluation-v2"
                / "forecasts"
                / fold_id
                / method
                / str(seed or "shared")
            )
            ledger_identity = _learned_ledger_identity(config, fold_id, method, seed)
            ledger_files = ("scale.parquet", "shape.parquet", "metrics.parquet")
            if ledger_root.exists():
                verify_artifact(ledger_root, identity=ledger_identity, names=ledger_files)
                output_parts.append(ledger_root / "metrics.parquet")
                continue
            model, metadata = LightGBMVolumeModel.load_native(model_root)
            if metadata.get("paper_config_hash") != config.config_hash:
                raise ValueError(f"LightGBM config identity mismatch: {model_root}")
            total_batches = []
            shape_batches = []
            for batch in prediction_batches(
                base, embedding_path=embedding, untrained_control=method == "untrained_neural"
            ):
                batch_totals, batch_shape = model.predict_frames(
                    batch.scale, batch.shape, group_columns=("case_id",)
                )
                total_batches.append(batch_totals)
                shape_batches.append(batch_shape)
                del batch
            totals = np.concatenate(total_batches)
            predicted_shape = pd.concat(shape_batches, ignore_index=True)
            del total_batches, shape_batches
            metrics = forecast_metric_frame(
                base, totals, predicted_shape, fold_id=fold_id, method=method, seed=seed
            )
            ledger_scale = base.scale.loc[
                :,
                [
                    "sample_id",
                    "fold_id",
                    "instrument_id",
                    "symbol",
                    "session_date",
                    "as_of",
                    "training_cutoff",
                    "market_information_as_of",
                    "feature_history_end",
                ],
            ].copy()
            ledger_scale["predicted_remaining_volume"] = totals
            publish_frames(
                ledger_root,
                identity=ledger_identity,
                frames={
                    "scale.parquet": ledger_scale,
                    "shape.parquet": predicted_shape.sort_values(
                        ["case_id", "target_bucket"], kind="stable"
                    ).reset_index(drop=True),
                    "metrics.parquet": metrics,
                },
            )
            output_parts.append(ledger_root / "metrics.parquet")
            del model, predicted_shape, ledger_scale
            gc.collect()
        ewma_tasks = [
            EWMAWork(
                evaluation_root(config) / "evaluation-v2" / "bases" / fold_id,
                market_inputs[str(instrument)],
                evaluation_root(config)
                / "evaluation-v2"
                / "forecasts"
                / fold_id
                / "ewma"
                / instrument_key(str(instrument)),
                str(instrument),
                {**execution_identity, "fold_id": fold_id},
            )
            for instrument in sorted(base.scale["instrument_id"].astype(str).unique())
        ]
        for completed in run_ewma_workers(ewma_tasks):
            output_parts.append(completed / "metrics.parquet")
        del base
        gc.collect()
    from execsim.ml.paper.evaluation_artifacts import merge_result_shards

    destination = evaluation_root(config) / "evaluation" / "forecast-results.parquet"
    merged = merge_result_shards(
        destination,
        sources={
            str(path.relative_to(config.artifact_root)): (
                path,
                read_json(path.parent / "manifest.json")["files"][path.name]["sha256"],
            )
            for path in output_parts
        },
        keys=(
            "fold_id",
            "method",
            "seed",
            "instrument_id",
            "session_date",
            "as_of_token",
            "sample_id",
        ),
        identity=execution_identity,
        schema_version="paper-forecast-evaluation-v1",
    )
    missing_parts = [
        path.parent / "unavailable.parquet"
        for path in output_parts
        if (path.parent / "unavailable.parquet").is_file()
    ]
    if missing_parts:
        merge_result_shards(
            destination.with_name("forecast-unavailable.parquet"),
            sources={
                str(path.relative_to(config.artifact_root)): (
                    path,
                    read_json(path.parent / "manifest.json")["files"][path.name]["sha256"],
                )
                for path in missing_parts
            },
            keys=(
                "fold_id",
                "instrument_id",
                "session_date",
                "as_of",
                "end_token",
                "generated_at",
                "sample_id",
            ),
            identity=execution_identity,
            schema_version="paper-forecast-unavailable-v1",
        )
    return {"status": "SOFTWARE READY", "rows": merged["rows"], "artifact": str(destination)}


def evaluate_representations_stage(
    config: PaperRunConfig,
    *,
    full_run_cli_enabled: bool,
    runtime_approval: PaperRuntimeApproval | None,
) -> dict[str, object]:
    """Run the frozen capacity ladder, observable probe, and exploratory support analysis."""
    config.authorize(
        "locked_result_evaluation",
        approval=runtime_approval,
        cli_enabled=full_run_cli_enabled,
    )
    _require_parameter_freeze(config)
    _require_locked_test_opened(config)
    import json

    import torch

    from execsim.ml.paper.evaluation_artifacts import (
        merge_result_shards,
        publish_frames,
        verify_artifact,
    )
    from execsim.ml.paper.lightgbm_data import build_historical_baseline_regime_frame
    from execsim.ml.paper.regimes import (
        fit_unusual_session_thresholds,
        label_unusual_sessions,
    )
    from execsim.ml.representations.checkpoints import load_checkpoint
    from execsim.ml.representations.frozen_evaluation import (
        FrozenProbeOptions,
        evaluate_frozen_capacity_streaming,
    )
    from execsim.ml.representations.jepa import PredictiveRepresentationModel
    from execsim.ml.representations.schemas import CheckpointCompatibility, RepresentationConfig
    from execsim.ml.sequences.streaming import PaperSequenceDataset, build_sequence_dataloader

    device = "cuda" if torch.cuda.is_available() else "cpu"
    execution_identity = {
        "source_commit": _git_head(),
        "source_tree": _git_tree(),
        "paper_config_hash": config.config_hash,
        "parameter_freeze_sha256": file_sha256(
            config.artifact_root / "selection" / "parameter-freeze-v1.json"
        ),
    }
    parts: dict[str, dict[str, tuple[Path, str]]] = {
        "accessibility": {},
        "date-metrics": {},
        "support": {},
    }
    for fold in config.evaluation["folds"]:
        fold_id = str(fold["id"])
        sequence = config.artifact_root / "sequences" / fold_id / "sequence-manifest.json"
        test_states: pd.DataFrame | None = None
        for method in ("dense", "sparse"):
            for seed in config.representation["seeds"]:
                checkpoint_root = representation_root(config) / fold_id / method / str(seed)
                embedding_root = config.artifact_root / "embeddings" / fold_id / method / str(seed)
                coordinate = f"{fold_id}/{method}/{seed}"
                destination = (
                    evaluation_root(config) / "evaluation-v2" / "representations" / coordinate
                )
                identity = {
                    **execution_identity,
                    "schema_version": "paper-representation-coordinate-v1",
                    "coordinate": coordinate,
                    "sequence_sha256": file_sha256(sequence),
                    "checkpoint_sha256": file_sha256(checkpoint_root / "final" / "manifest.json"),
                    "compatibility_sha256": file_sha256(checkpoint_root / "compatibility.json"),
                    "embedding_manifest_sha256": file_sha256(embedding_root / "manifest.json"),
                }
                names = ["accessibility.parquet", "date-metrics.parquet"]
                if method == "sparse":
                    names.append("support.parquet")
                if destination.exists():
                    receipt = verify_artifact(destination, identity=identity, names=names)
                    for name in names:
                        parts[Path(name).stem][coordinate] = (
                            destination / name,
                            receipt["files"][name]["sha256"],
                        )
                    continue
                if test_states is None:
                    training_states = build_historical_baseline_regime_frame(
                        sequence, partition="train"
                    )
                    test_states = label_unusual_sessions(
                        build_historical_baseline_regime_frame(sequence, partition="test"),
                        fit_unusual_session_thresholds(training_states),
                    )
                    del training_states
                accessibility_rows: list[dict[str, object]] = []
                date_metric_rows: list[dict[str, object]] = []
                support_rows: list[dict[str, object]] = []
                expected = CheckpointCompatibility(
                    **read_json(checkpoint_root / "compatibility.json")
                )
                representation = RepresentationConfig(
                    expected.geometry,
                    predictor_family=expected.predictor_family,
                    generalized_gaussian_p=expected.generalized_gaussian_p,
                    generalized_gaussian_mu=expected.generalized_gaussian_mu,
                    generalized_gaussian_sigma=expected.generalized_gaussian_sigma,
                    rdm_projections_train=expected.rdm_projections,
                    rdm_projections_evaluation=int(
                        config.representation["rdm_projections_evaluation"]
                    ),
                    seed=int(seed),
                )
                model = PredictiveRepresentationModel(representation).to(device)
                load_checkpoint(model, checkpoint_root / "final", expected=expected)

                def loader(
                    partition: str,
                    *,
                    manifest_path: Path = sequence,
                    run_seed: int = int(seed),
                ) -> Any:
                    dataset = PaperSequenceDataset(
                        manifest_path,
                        partition=partition,
                        seed=run_seed,
                        cache_size=int(config.sequences["session_cache_size"]),
                        sample_train_positions=False,
                    )
                    return build_sequence_dataloader(
                        dataset,
                        batch_size=int(config.representation["batch_size"]),
                        num_workers=int(config.sequences["num_workers"]),
                        device=device,
                        prefetch_factor=int(config.sequences["prefetch_factor"]),
                    )

                capacity, observable, dated = evaluate_frozen_capacity_streaming(
                    model,
                    loader("train"),
                    loader("validation"),
                    loader("test"),
                    device=device,
                    seed=int(seed),
                    options=FrozenProbeOptions(
                        ridge_alphas=tuple(
                            float(value) for value in config.representation["probe_ridge_alphas"]
                        ),
                        mlp_epochs=int(config.representation["probe_mlp_epochs"]),
                    ),
                )
                observable_by_capacity_horizon = {
                    (str(row["probe_capacity"]), int(row["horizon"])): row for row in observable
                }
                date_metric_rows.extend(
                    {
                        "fold_id": fold_id,
                        "geometry": method,
                        "seed": int(seed),
                        **row,
                    }
                    for row in dated
                )
                diagnostics, transitions, regime_counts = _stream_embedding_diagnostics(
                    embedding_root / "partition=test" / "embeddings.parquet",
                    test_states,
                )
                for row in capacity:
                    horizon = int(row["horizon"])
                    accessibility_rows.append(
                        {
                            "fold_id": fold_id,
                            "geometry": method,
                            "seed": int(seed),
                            **row,
                            **observable_by_capacity_horizon[(str(row["probe_capacity"]), horizon)],
                            "zero_fraction": diagnostics["zero_fraction"],
                            "mean_active_dimensions": diagnostics["mean_active_dimensions"],
                        }
                    )
                if method == "sparse":
                    support_rows.append(
                        {
                            "fold_id": fold_id,
                            "geometry": method,
                            "seed": int(seed),
                            **diagnostics,
                            **{
                                name: json.dumps(value, sort_keys=True)
                                if isinstance(value, dict)
                                else value
                                for name, value in transitions.items()
                            },
                            "ordinary_rows": regime_counts.get("ordinary", 0),
                            "unusual_rows": regime_counts.get("unusual", 0),
                        }
                    )
                frames = {
                    "accessibility.parquet": pd.DataFrame(accessibility_rows),
                    "date-metrics.parquet": pd.DataFrame(date_metric_rows),
                }
                if method == "sparse":
                    frames["support.parquet"] = pd.DataFrame(support_rows)
                receipt = publish_frames(destination, identity=identity, frames=frames)
                for name in names:
                    parts[Path(name).stem][coordinate] = (
                        destination / name,
                        receipt["files"][name]["sha256"],
                    )
                del model, frames, capacity, observable, dated
                gc.collect()
    output_root = evaluation_root(config) / "evaluation"
    output_root.mkdir(parents=True, exist_ok=True)
    accessibility_path = output_root / "representation-accessibility.parquet"
    date_metrics_path = output_root / "representation-date-metrics.parquet"
    support_path = output_root / "support-regimes.parquet"
    for kind, path, extra_keys in (
        ("accessibility", accessibility_path, ["probe_capacity", "horizon"]),
        ("date-metrics", date_metrics_path, ["probe_capacity", "horizon", "date"]),
        ("support", support_path, []),
    ):
        merge_result_shards(
            path,
            sources=parts[kind],
            keys=["fold_id", "geometry", "seed", *extra_keys],
            identity=execution_identity,
            schema_version="paper-representation-result-v1",
        )
    from execsim.data.paper.manifests import write_json_atomic

    write_json_atomic(
        output_root / "representation-evaluation-manifest.json",
        {
            "schema_version": "paper-representation-evaluation-v2",
            "paper_config_hash": config.config_hash,
            "accessibility_sha256": file_sha256(accessibility_path),
            "date_metrics_sha256": file_sha256(date_metrics_path),
            "support_regimes_sha256": file_sha256(support_path),
        },
    )
    return {
        "status": "SOFTWARE READY",
        "accessibility": str(accessibility_path),
        "date_metrics": str(date_metrics_path),
        "support_regimes": str(support_path),
    }


def run_tca_stage(
    config: PaperRunConfig,
    source: Path | None = None,
    *,
    full_run_cli_enabled: bool,
    runtime_approval: PaperRuntimeApproval | None,
) -> dict[str, object]:
    """Run independent matched date shards using the frozen forecast ledgers."""
    config.authorize(
        "locked_result_evaluation",
        approval=runtime_approval,
        cli_enabled=full_run_cli_enabled,
    )
    _require_parameter_freeze(config)
    _require_locked_test_opened(config)
    from execsim.ml.paper.evaluation_artifacts import publish_frames, verify_artifact
    from execsim.ml.paper.evaluation_workers import (
        EWMAWork,
        compact_profile_corpus,
        ewma_ledger_identity,
        instrument_key,
    )
    from execsim.ml.paper.tca import select_liquidity_spaced_instruments
    from execsim.ml.paper.tca_inputs import (
        filter_tca_window_exact,
        prepare_tca_history,
        read_tca_date,
        tca_eligible_instrument_ids,
        validate_tca_adv20,
    )
    from execsim.ml.paper.tca_workers import (
        TCAWork,
        preflight_tca_ledgers,
        run_tca_workers,
    )

    execution = {
        "source_commit": _git_head(),
        "source_tree": _git_tree(),
        "paper_config_hash": config.config_hash,
        "parameter_freeze_sha256": file_sha256(
            config.artifact_root / "selection" / "parameter-freeze-v1.json"
        ),
    }
    profile_root = evaluation_root(config) / "evaluation-v2" / "profile-corpus"
    profile_receipt = read_json(profile_root / "manifest.json")
    profile_identity = profile_receipt["identity"]
    if any(profile_identity.get(key) != value for key, value in execution.items()):
        raise ValueError("TCA profile corpus belongs to another evaluation execution.")
    verify_artifact(profile_root, identity=profile_identity, names=tuple(profile_receipt["files"]))
    market_profiles = {
        key: profile_root / name for key, name in profile_receipt["instruments"].items()
    }

    def fold_context(fold: dict[str, Any]) -> dict[str, Any]:
        """Resolve immutable ledger paths and fold bounds once for both passes."""
        fold_id = str(fold["id"])
        start, end = (pd.Timestamp(value).date() for value in fold["test"])
        variants = [
            ("raw", None),
            ("untrained_neural", None),
            *(
                (geometry, int(seed))
                for geometry in ("dense", "sparse")
                for seed in config.representation["seeds"]
            ),
        ]
        ledgers = tuple(
            (
                method,
                seed,
                evaluation_root(config)
                / "evaluation-v2"
                / "forecasts"
                / fold_id
                / method
                / str(seed or "shared"),
                _learned_ledger_identity(config, fold_id, method, seed),
            )
            for method, seed in variants
        )
        ewma_records = {}
        for instrument in sorted(instruments):
            work = EWMAWork(
                evaluation_root(config) / "evaluation-v2" / "bases" / fold_id,
                market_profiles[instrument],
                evaluation_root(config)
                / "evaluation-v2"
                / "forecasts"
                / fold_id
                / "ewma"
                / instrument_key(instrument),
                instrument,
                {**execution, "fold_id": fold_id},
            )
            ewma_records[instrument] = (work.output_directory, ewma_ledger_identity(work))
        return {
            "fold_id": fold_id,
            "start": start,
            "end": end,
            "sequence": config.artifact_root / "sequences" / fold_id / "sequence-manifest.json",
            "ledgers": ledgers,
            "ewma_records": ewma_records,
            "cutoff": pd.Timestamp(fold["train"][1]).date(),
        }

    universe_path = _verify_frozen_universe_manifest(config)
    universe = pd.DataFrame(read_json(universe_path)["members"])
    instruments = set(
        select_liquidity_spaced_instruments(universe, size=int(config.tca["universe_size"]))
    ) | set(
        select_liquidity_spaced_instruments(
            universe, size=int(config.tca["sensitivity_universe_size"])
        )
    )
    market = compact_profile_corpus(
        source or config.data_path("target_corpus_root"),
        evaluation_root(config) / "evaluation-v2" / "tca-market",
        identity=execution,
        include_market_bars=True,
        selected_instruments=tuple(sorted(instruments)),
    )
    market_manifest = evaluation_root(config) / "evaluation-v2" / "tca-market" / "manifest.json"
    if (
        read_json(market_manifest)["identity"]["source_inventory_sha256"]
        != profile_identity["source_inventory_sha256"]
    ):
        raise ValueError("TCA and forecast ledgers require the same immutable source corpus.")
    histories = {
        instrument: prepare_tca_history(
            market[instrument],
            evaluation_root(config) / "evaluation-v2" / "tca-history" / instrument_key(instrument),
            instrument_id=instrument,
            cutoffs={
                str(fold["id"]): pd.Timestamp(fold["train"][1]).date()
                for fold in config.evaluation["folds"]
            },
            identity=execution,
        )
        for instrument in sorted(instruments)
    }
    dates = sorted(
        {
            pd.Timestamp(day).date()
            for history in histories.values()
            for day in pd.read_parquet(history / "sessions.parquet")["session_date"]
        }
    )
    fold_contexts = {str(fold["id"]): fold_context(fold) for fold in config.evaluation["folds"]}
    # Resolve the scientific population and validate every required ledger before
    # constructing or launching any expensive date worker.
    eligible_by_fold: dict[str, dict[date, tuple[str, ...]]] = {}
    for fold in config.evaluation["folds"]:
        context = fold_contexts[str(fold["id"])]
        fold_id = str(context["fold_id"])
        start = context["start"]
        end = context["end"]
        eligible_cases: dict[date, tuple[str, ...]] = {}
        for day in dates:
            if not start <= day <= end:
                continue
            date_bars, date_adv = read_tca_date(histories, day)
            eligible = tca_eligible_instrument_ids(date_bars, instruments)
            if eligible:
                # Validate required derived evidence before ledger preflight or
                # construction/launch of any TCA worker.
                validate_tca_adv20(date_adv, {day: eligible})
                eligible_cases[day] = eligible
        preflight_tca_ledgers(
            ledger_records=context["ledgers"],
            ewma_records=context["ewma_records"],
            eligible_cases=eligible_cases,
            training_cutoff=context["cutoff"],
            tca_config=config.tca,
        )
        eligible_by_fold[fold_id] = eligible_cases

    main_outputs, sensitivity_outputs = [], []
    for fold in config.evaluation["folds"]:
        context = fold_contexts[str(fold["id"])]
        fold_id = str(context["fold_id"])
        start = context["start"]
        end = context["end"]
        sequence = context["sequence"]
        ledgers = context["ledgers"]
        ewma_records = context["ewma_records"]
        cutoff = context["cutoff"]
        profiles = pd.concat(
            [
                pd.read_parquet(path / "profiles.parquet", filters=[("fold_id", "==", fold_id)])
                for path in histories.values()
            ],
            ignore_index=True,
        ).drop(columns="fold_id")
        tasks = []
        for day in dates:
            if not start <= day <= end or day not in eligible_by_fold[fold_id]:
                continue
            date_bars, date_adv = read_tca_date(histories, day)
            eligible_instruments = set(eligible_by_fold[fold_id][day])
            date_bars = filter_tca_window_exact(date_bars, eligible_instruments)
            date_adv = date_adv.loc[
                date_adv["instrument_id"].astype(str).isin(eligible_instruments)
            ].reset_index(drop=True)
            if date_bars.empty:
                continue
            identity = {
                **execution,
                "fold_id": fold_id,
                "session_date": str(day),
                "profile_corpus_manifest_sha256": file_sha256(profile_root / "manifest.json"),
            }
            input_directory = (
                evaluation_root(config) / "evaluation-v2" / "tca-inputs" / fold_id / str(day)
            )
            publish_frames(
                input_directory,
                identity=identity,
                frames={
                    "bars.parquet": date_bars.reset_index(drop=True),
                    "adv.parquet": date_adv.reset_index(drop=True),
                    "profiles.parquet": profiles,
                    "universe.parquet": universe,
                },
            )
            del date_bars, date_adv
            tasks.append(
                TCAWork(
                    input_directory,
                    evaluation_root(config) / "evaluation-v2" / "tca-shards" / fold_id / str(day),
                    ledgers,
                    ewma_records,
                    cutoff,
                    file_sha256(sequence),
                    dict(config.tca),
                    identity,
                )
            )
        for completed in run_tca_workers(tasks):
            main_outputs.append(completed / "main.parquet")
            sensitivity_outputs.append(completed / "sensitivity.parquet")
        del profiles, tasks
        gc.collect()
    output_root = evaluation_root(config) / "tca"
    output_root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, parts in (("main", main_outputs), ("sensitivity", sensitivity_outputs)):
        path = output_root / f"{name}.parquet"
        from execsim.ml.paper.evaluation_artifacts import merge_result_shards

        merge_result_shards(
            path,
            sources={
                str(part.relative_to(config.artifact_root)): (
                    part,
                    read_json(part.parent / "manifest.json")["files"][part.name]["sha256"],
                )
                for part in parts
            },
            keys=("fold_id", "date", "instrument_id", "method", "order_fraction_adv20"),
            identity=execution,
            schema_version="paper-tca-merged-v3",
        )
        paths[name] = str(path)
    write_json_atomic(
        output_root / "manifest.json",
        {
            "schema_version": "paper-tca-v1",
            "paper_config_hash": config.config_hash,
            "evaluation_identity": execution,
            "files": {
                name: {"path": path, "sha256": file_sha256(Path(path))}
                for name, path in paths.items()
            },
        },
    )
    return {"status": "SOFTWARE READY", **paths}


def report_stage(
    config: PaperRunConfig,
    *,
    full_run_cli_enabled: bool,
    runtime_approval: PaperRuntimeApproval | None,
) -> dict[str, object]:
    """Publish the full report atomically and verify all output bytes on resume."""
    config.authorize(
        "locked_result_evaluation", approval=runtime_approval, cli_enabled=full_run_cli_enabled
    )
    _require_parameter_freeze(config)
    _require_locked_test_opened(config)
    from execsim.ml.paper.evaluation_artifacts import publish_bundle

    root = evaluation_root(config)
    input_names: tuple[str, ...] = (
        "evaluation/forecast-results.parquet",
        "evaluation/representation-accessibility.parquet",
        "evaluation/representation-date-metrics.parquet",
        "evaluation/support-regimes.parquet",
        "tca/main.parquet",
        "tca/sensitivity.parquet",
    )
    if (root / "evaluation/forecast-unavailable.parquet").is_file():
        input_names += ("evaluation/forecast-unavailable.parquet",)
    inputs = {}
    for name in input_names:
        path = root / name
        digest = file_sha256(path)
        if config.runtime_evaluation_root is not None:
            receipt = read_json(path.with_suffix(".manifest.json"))
            if (
                receipt.get("parquet_sha256") != digest
                or receipt.get("paper_config_hash") != config.config_hash
                or receipt.get("merge_identity", {}).get("source_commit") != _git_head()
            ):
                raise ValueError("Report input merge checksum or source identity mismatch.")
        inputs[name] = digest
    identity = {
        "source_commit": _git_head(),
        "source_tree": _git_tree(),
        "paper_config_hash": config.config_hash,
        "parameter_freeze_sha256": file_sha256(
            config.artifact_root / "selection/parameter-freeze-v1.json"
        ),
        "input_sha256": inputs,
    }
    destination = evaluation_report_root(config) / config.paper_run_id

    def build(staging: Path) -> None:
        _build_report_stage(config, output_root=staging)
        built = staging / config.paper_run_id
        for path in list(built.iterdir()):
            os.replace(path, staging / path.name)
        built.rmdir()

    reused = destination.exists()
    publish_bundle(destination, identity=identity, build=build)
    return {
        "status": "SOFTWARE READY",
        "output": str(destination),
        "reuse": "validated" if reused else "created",
    }


def _build_report_stage(config: PaperRunConfig, *, output_root: Path) -> dict[str, object]:
    """Construct named historical tables, matched inference, and the real report bundle."""
    from execsim.ml.paper.reports import (
        write_historical_paper_bundle,
    )
    from execsim.ml.paper.statistics import (
        build_confirmatory_inference,
        construct_complete_case_differences,
        moving_block_bootstrap,
    )

    report_inputs = {
        "forecast": evaluation_root(config) / "evaluation" / "forecast-results.parquet",
        "tca": evaluation_root(config) / "tca" / "main.parquet",
        "accessibility": evaluation_root(config)
        / "evaluation"
        / "representation-accessibility.parquet",
        "representation_dates": evaluation_root(config)
        / "evaluation"
        / "representation-date-metrics.parquet",
        "support": evaluation_root(config) / "evaluation" / "support-regimes.parquet",
        "tca_sensitivity": evaluation_root(config) / "tca" / "sensitivity.parquet",
    }
    if missing := [name for name, path in report_inputs.items() if not path.is_file()]:
        raise RuntimeError(f"BLOCKED: historical report inputs are missing: {missing}")
    forecast = pd.read_parquet(report_inputs["forecast"])
    tca = pd.read_parquet(report_inputs["tca"])
    accessibility = pd.read_parquet(report_inputs["accessibility"])
    representation_dates = pd.read_parquet(report_inputs["representation_dates"])
    support = pd.read_parquet(report_inputs["support"])
    tca_sensitivity = pd.read_parquet(report_inputs["tca_sensitivity"])
    identity = (
        "fold_id",
        "date",
        "instrument_id",
        "order_fraction_adv20",
        "parent_quantity",
        "side",
        "start",
        "end",
        "planned_participation",
        "hard_participation",
        "risk_aversion",
        "tracking_penalty",
    )
    execution_rows = []
    bootstrap_sensitivity_rows = []
    unavailable_comparisons = []
    for candidate in sorted(set(tca["method"].astype(str))):
        if candidate.startswith("raw_sparse_jepa_seed_"):
            baseline = candidate.replace("raw_sparse_jepa", "raw_dense_jepa")
        else:
            baseline = "lightgbm_raw"
        paired = construct_complete_case_differences(
            tca,
            baseline=baseline,
            candidate=candidate,
            value_column="normalized_allocation_regret",
            identity_columns=identity,
        )
        if not paired.matched_rows:
            unavailable_comparisons.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "status": "NO_MATCHED_CASES",
                    "dropped_baseline": paired.dropped_baseline_rows,
                    "dropped_candidate": paired.dropped_candidate_rows,
                }
            )
            continue
        by_date = paired.paired_rows.groupby(["fold_id", "date"], sort=True, as_index=False)[
            "difference"
        ].mean()
        sensitivity_results = {}
        for block_length in (
            int(config.evaluation["bootstrap_block_dates"]),
            *(int(value) for value in config.evaluation["bootstrap_block_sensitivity_dates"]),
        ):
            block_result = moving_block_bootstrap(
                by_date,
                block_length=block_length,
                repetitions=int(config.evaluation["bootstrap_repetitions"]),
                confidence=float(config.evaluation["confidence"]),
            )
            sensitivity_results[block_length] = block_result
            bootstrap_sensitivity_rows.append(
                {
                    "candidate": candidate,
                    "baseline": baseline,
                    "block_length_dates": block_length,
                    "mean_difference": block_result.mean_difference,
                    "ci_lower": block_result.confidence_interval[0],
                    "ci_upper": block_result.confidence_interval[1],
                    "paired_dates": block_result.paired_dates,
                    "raw_p_value": block_result.raw_p_value,
                }
            )
        result = sensitivity_results[int(config.evaluation["bootstrap_block_dates"])]
        candidate_rows = tca.loc[tca["method"] == candidate]
        seed_text = candidate.rsplit("_seed_", maxsplit=1)
        seed = int(seed_text[1]) if len(seed_text) == 2 else -1
        execution_rows.append(
            {
                "method": candidate,
                "comparison_baseline": baseline,
                "seed": seed,
                "normalized_allocation_regret": float(
                    candidate_rows["normalized_allocation_regret"].mean()
                ),
                "absolute_modeled_impact_cost": float(
                    candidate_rows["absolute_modeled_impact_cost"].mean()
                ),
                "completion_rate": float(candidate_rows["completion_rate"].mean()),
                "implementation_shortfall_bps": float(
                    candidate_rows["implementation_shortfall_bps"].mean()
                ),
                "mean_difference": result.mean_difference,
                "ci_lower": result.confidence_interval[0],
                "ci_upper": result.confidence_interval[1],
                "raw_p_value": result.raw_p_value,
                "matched_cases": paired.matched_rows,
                "dropped_baseline": paired.dropped_baseline_rows,
                "dropped_candidate": paired.dropped_candidate_rows,
            }
        )
    forecast = forecast.copy()
    forecast["method_key"] = np.where(
        forecast["seed"].isna(),
        forecast["method"].astype(str),
        forecast["method"].astype(str) + "_seed_" + forecast["seed"].astype("Int64").astype(str),
    )
    if forecast.duplicated(["method_key", "fold_id", "sample_id"]).any():
        raise ValueError("Forecast evaluation duplicates a method/case identity.")
    case_sets = [
        set(group["fold_id"].astype(str) + "|" + group["sample_id"].astype(str))
        for _, group in forecast.groupby("method_key", sort=True)
    ]
    if not case_sets:
        raise ValueError("Forecast evaluation contains no method rows.")
    common_cases = set.intersection(*case_sets)
    forecast["case_key"] = forecast["fold_id"].astype(str) + "|" + forecast["sample_id"].astype(str)
    matched_forecast = forecast.loc[forecast["case_key"].isin(common_cases)]
    forecast_by_asof = (
        matched_forecast.groupby(["method", "seed", "as_of_token"], dropna=False, as_index=False)
        .agg(
            log_remaining_volume_mae=("log_remaining_volume_absolute_error", "mean"),
            conditional_curve_error=("conditional_curve_wasserstein", "mean"),
            matched_cases=("sample_id", "size"),
            causal_baseline_remaining_volume=("causal_baseline_remaining_volume", "mean"),
        )
        .sort_values(["method", "seed", "as_of_token"], kind="stable")
    )
    forecast_performance = (
        matched_forecast.groupby(["method", "seed"], dropna=False, as_index=False)
        .agg(
            log_remaining_volume_mae=("log_remaining_volume_absolute_error", "mean"),
            conditional_curve_error=("conditional_curve_wasserstein", "mean"),
            matched_cases=("sample_id", "size"),
        )
        .sort_values(["method", "seed"], kind="stable")
    )
    confirmatory, confirmatory_seeds, confirmatory_sensitivity = build_confirmatory_inference(
        representation_dates,
        forecast,
        definitions=tuple(
            {str(name): str(value) for name, value in definition.items()}
            for definition in config.evaluation["confirmatory_contrast_definitions"]
        ),
        block_length=int(config.evaluation["bootstrap_block_dates"]),
        sensitivity_block_lengths=tuple(
            int(value) for value in config.evaluation["bootstrap_block_sensitivity_dates"]
        ),
        repetitions=int(config.evaluation["bootstrap_repetitions"]),
        confidence=float(config.evaluation["confidence"]),
    )
    dataset_rows = []
    for fold in config.evaluation["folds"]:
        manifest = read_json(
            config.artifact_root / "sequences" / str(fold["id"]) / "sequence-manifest.json"
        )
        for partition, included in manifest["partition_counts"].items():
            dataset_rows.append(
                {
                    "fold_id": fold["id"],
                    "partition": partition,
                    "included": int(included),
                    "excluded": len(manifest["exclusions"]),
                }
            )
    jepa_diagnostics = (
        accessibility.groupby(["fold_id", "geometry", "seed"], sort=True, as_index=False)
        .agg(
            zero_fraction=("zero_fraction", "first"),
            mean_active_dimensions=("mean_active_dimensions", "first"),
        )
        .sort_values(["fold_id", "geometry", "seed"], kind="stable")
    )
    representation_accessibility = accessibility.loc[
        :,
        [
            "fold_id",
            "geometry",
            "seed",
            "horizon",
            "probe_capacity",
            "parameter_count",
            "approximate_macs",
            "inference_seconds",
            "normalized_latent_error",
            "zero_baseline",
            "train_mean_baseline",
            "persistence_baseline",
            "test_rows",
        ],
    ].copy()
    observable_accessibility = accessibility.loc[
        :,
        [
            "fold_id",
            "geometry",
            "seed",
            "horizon",
            "probe_capacity",
            "observable_volume_probe_mae",
            "observable_volume_probe_rmse",
            "observable_parameter_count",
            "observable_approximate_macs",
            "observable_inference_seconds",
            "observable_test_rows",
        ],
    ].copy()
    lightgbm_parameters = []
    for manifest_path in sorted((config.artifact_root / "lightgbm").glob("*/*/*/manifest.json")):
        payload = read_json(manifest_path)
        scale = payload["selected_scale_config"]
        shape = payload["selected_shape_config"]
        iterations = payload["selected_iterations"]
        lightgbm_parameters.append(
            {
                "fold_id": payload["fold_id"],
                "method": payload["method"],
                "seed": payload["seed"],
                "scale_num_leaves": scale["num_leaves"],
                "scale_min_child_samples": scale["min_child_samples"],
                "scale_reg_lambda": scale["reg_lambda"],
                "scale_best_iteration": iterations["scale"],
                "shape_num_leaves": shape["num_leaves"],
                "shape_min_child_samples": shape["min_child_samples"],
                "shape_reg_lambda": shape["reg_lambda"],
                "shape_best_iteration": iterations["shape"],
            }
        )
    appendix_frames = [
        pd.DataFrame(bootstrap_sensitivity_rows).assign(analysis="tca_block_length"),
        tca_sensitivity.assign(analysis="tca_order_size"),
        confirmatory_seeds.assign(analysis="confirmatory_seed_effect"),
        confirmatory_sensitivity.assign(analysis="confirmatory_block_length"),
    ]
    tables = {
        "dataset_folds_exclusions": pd.DataFrame(dataset_rows),
        "jepa_representation_diagnostics": jepa_diagnostics,
        "representation_accessibility": representation_accessibility,
        "observable_financial_accessibility": observable_accessibility,
        "forecast_performance": forecast_performance,
        "forecast_by_asof": forecast_by_asof,
        "lightgbm_selected_parameters": pd.DataFrame(lightgbm_parameters),
        "tca_execution": pd.DataFrame(execution_rows),
        "confirmatory_statistics": confirmatory,
        "support_regime_diagnostics": support,
        "appendix_sensitivities": pd.concat(appendix_frames, ignore_index=True, sort=False),
    }
    output = write_historical_paper_bundle(
        output_root,
        paper_run_id=config.paper_run_id,
        tables=tables,
        provenance={
            "data_classification": "historical",
            "paper_config_hash": config.config_hash,
            "network_acquisition": "completed before this reporting stage",
            "historical_training": "completed before this reporting stage",
            "empirical_claim": "not automatically generated",
        },
    )
    appendix = output / "appendix"
    appendix.mkdir()
    pd.DataFrame(bootstrap_sensitivity_rows).to_parquet(
        appendix / "bootstrap-block-sensitivity.parquet", index=False
    )
    support.to_parquet(appendix / "support-regimes.parquet", index=False)
    pd.DataFrame(
        unavailable_comparisons,
        columns=[
            "candidate",
            "baseline",
            "status",
            "dropped_baseline",
            "dropped_candidate",
        ],
    ).to_parquet(appendix / "unavailable-comparisons.parquet", index=False)
    for name, frame in (("main", tca), ("sensitivity", tca_sensitivity)):
        if "status" in frame:
            frame.loc[frame["status"] == "EWMA_UNAVAILABLE"].to_parquet(
                appendix / f"tca-{name}-unavailable.parquet",
                index=False,
            )
    missing_path = evaluation_root(config) / "evaluation" / "forecast-unavailable.parquet"
    if missing_path.is_file():
        import shutil

        shutil.copyfile(missing_path, appendix / "forecast-unavailable.parquet")
    pd.DataFrame(
        [
            {
                "method_key": key,
                "available_cases": len(group),
                "all_method_matched_cases": int(group["case_key"].isin(common_cases).sum()),
                "dropped_from_all_method_summary": int(
                    (~group["case_key"].isin(common_cases)).sum()
                ),
            }
            for key, group in forecast.groupby("method_key", sort=True)
        ]
    ).to_parquet(appendix / "forecast-case-coverage.parquet", index=False)
    return {"status": "SOFTWARE READY", "output": str(output)}


def _adapt_sparse_stage(config: PaperRunConfig, options: Any) -> list[dict[str, object]]:
    from dataclasses import replace
    from functools import partial

    import torch

    from execsim.data.paper.manifests import write_json_atomic
    from execsim.ml.representations.checkpoints import load_checkpoint, save_checkpoint
    from execsim.ml.representations.diagnostics import sparse_acceptance
    from execsim.ml.representations.difficulty_pipeline import build_difficulty_ledger
    from execsim.ml.representations.historical_trainer import (
        _loader,
        _validate,
        adapt_with_difficulty_loader,
    )
    from execsim.ml.representations.jepa import PredictiveRepresentationModel
    from execsim.ml.representations.schemas import CheckpointCompatibility, RepresentationConfig
    from execsim.ml.sequences.streaming import PaperSequenceDataset

    results = []
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for fold in config.evaluation["folds"]:
        fold_id = str(fold["id"])
        sequence_path = config.artifact_root / "sequences" / fold_id / "sequence-manifest.json"
        ledger_path = config.artifact_root / "difficulty" / fold_id / "training.parquet"
        ledger = (
            pd.read_parquet(ledger_path)
            if ledger_path.is_file()
            else build_difficulty_ledger(sequence_path, ledger_path)
        )
        if (
            ledger.empty
            or ledger["sample_id"].duplicated().any()
            or set(ledger["fold_id"].astype(str)) != {fold_id}
            or set(ledger["paper_config_hash"].astype(str)) != {config.config_hash}
            or set(ledger["sequence_manifest_hash"].astype(str)) != {file_sha256(sequence_path)}
        ):
            raise ValueError("Reusable difficulty ledger identity is incompatible.")
        weights = dict(zip(ledger["sample_id"].astype(str), ledger["weight"], strict=True))
        for seed in config.representation["seeds"]:
            base = config.artifact_root / "representations" / fold_id / "sparse" / str(seed)
            adapted = (
                config.artifact_root / "representations" / fold_id / "sparse_adapted" / str(seed)
            )
            expected = CheckpointCompatibility(**read_json(base / "compatibility.json"))
            model_config = RepresentationConfig(
                "sparse",
                predictor_family=expected.predictor_family,
                generalized_gaussian_p=expected.generalized_gaussian_p,
                generalized_gaussian_mu=expected.generalized_gaussian_mu,
                generalized_gaussian_sigma=expected.generalized_gaussian_sigma,
                rdm_projections_train=expected.rdm_projections,
                rdm_projections_evaluation=int(config.representation["rdm_projections_evaluation"]),
                seed=int(seed),
            )
            model = PredictiveRepresentationModel(model_config).to(device)
            adapted_hash = stable_hash(
                {
                    "base_training_config_hash": expected.training_config_hash,
                    "adaptation": "difficulty-v1",
                }
            )
            adapted_expected = replace(
                expected, training_config_hash=adapted_hash, adaptation="difficulty-v1"
            )
            if (adapted / "final" / "manifest.json").is_file():
                recorded = CheckpointCompatibility(**read_json(adapted / "compatibility.json"))
                if recorded != adapted_expected:
                    raise ValueError("Reusable adapted checkpoint identity mismatch.")
                loaded = load_checkpoint(model, adapted / "final", expected=adapted_expected)
                if loaded.seed != int(seed):
                    raise ValueError("Reusable adapted checkpoint seed does not match the run.")
                results.append({"fold_id": fold_id, "seed": seed, "status": "reused"})
                continue
            base_manifest = load_checkpoint(model, base / "final", expected=expected)
            dataset = PaperSequenceDataset(
                sequence_path, partition="train", seed=int(seed), cache_size=options.cache_size
            )
            training_result = read_json(base / "training-result.json")
            loader_factory = partial(_loader, dataset, options, device)
            steps = adapt_with_difficulty_loader(
                model,
                loader_factory,
                weights,
                actual_base_training_steps=int(training_result["global_steps"]),
                rdm_lambda=expected.calibrated_rdm_lambda,
                seed=int(seed) + 4_000_001,
                device=device,
            )
            validation = PaperSequenceDataset(
                sequence_path,
                partition="validation",
                seed=int(seed),
                cache_size=options.cache_size,
            )
            _, diagnostics = _validate(model, validation, options, device)
            failures = sparse_acceptance(
                diagnostics, target_zero_fraction=model_config.target_zero_fraction
            )
            if failures:
                raise RuntimeError(f"Adapted sparse checkpoint failed collapse gates: {failures}")
            adapted.mkdir(parents=True, exist_ok=False)
            write_json_atomic(adapted / "compatibility.json", asdict(adapted_expected))
            for role in ("latest", "best", "final"):
                manifest = replace(
                    base_manifest,
                    checkpoint_id=f"{fold_id}-sparse-adapted-{seed}-{role}",
                    weights_sha256="",
                    checkpoint_role=role,
                    adaptation="difficulty-v1",
                    training_config_hash=adapted_hash,
                    validation_diagnostics=tuple(sorted(diagnostics.items())),
                )
                save_checkpoint(model, adapted / role, manifest)
            results.append({"fold_id": fold_id, "seed": seed, "steps": steps})
    return results


def run_authorized_stages(
    config: PaperRunConfig,
    *,
    network_cli_enabled: bool,
    training_cli_enabled: bool,
    full_run_cli_enabled: bool,
    runtime_approval: PaperRuntimeApproval | None = None,
    trusted_local_resume: bool = False,
) -> dict[str, object]:
    """Resume idempotently through stages whose separate authorizations are present."""
    results: dict[str, object] = {}
    universe = Path(config.data["universe_manifest"])
    target_root = Path(config.data["target_corpus_root"])
    network_requested = network_cli_enabled or (
        runtime_approval is not None and runtime_approval.approves("target_acquisition")
    )
    training_requested = training_cli_enabled or (
        runtime_approval is not None and runtime_approval.approves("historical_training")
    )
    evaluation_requested = full_run_cli_enabled or (
        runtime_approval is not None and runtime_approval.approves("locked_result_evaluation")
    )
    network_enabled = config.authorization_granted(
        "target_acquisition", approval=runtime_approval, cli_enabled=network_cli_enabled
    )
    training_enabled = config.authorization_granted(
        "historical_training", approval=runtime_approval, cli_enabled=training_cli_enabled
    )
    evaluation_enabled = config.authorization_granted(
        "locked_result_evaluation",
        approval=runtime_approval,
        cli_enabled=full_run_cli_enabled,
    )
    if network_requested and not network_enabled:
        config.authorize(
            "target_acquisition", approval=runtime_approval, cli_enabled=network_cli_enabled
        )
    if training_requested and not training_enabled:
        config.authorize(
            "historical_training", approval=runtime_approval, cli_enabled=training_cli_enabled
        )
    if evaluation_requested and not evaluation_enabled:
        config.authorize(
            "locked_result_evaluation",
            approval=runtime_approval,
            cli_enabled=full_run_cli_enabled,
        )
    frozen_universe = _is_frozen_universe(universe, config_hash=config.config_hash)
    if not frozen_universe and not _formation_artifacts_ready(config):
        if network_enabled:
            results["download_data"] = download_data_stage(
                config,
                cli_enabled=network_cli_enabled,
                runtime_approval=runtime_approval,
            )
        else:
            results["download_data"] = "DATA NOT ACQUIRED"
            return results
    if not _is_frozen_universe(universe, config_hash=config.config_hash):
        results["build_universe"] = build_universe_stage(config)
    else:
        results["build_universe"] = "reused"
    if network_enabled and "download_data" not in results:
        results["download_data"] = download_data_stage(
            config,
            cli_enabled=network_cli_enabled,
            runtime_approval=runtime_approval,
        )
    elif not _has_parquet_corpus(target_root):
        results["download_data"] = "DATA NOT ACQUIRED"
        return results
    else:
        from execsim.data.paper.acquisition import monthly_chunks

        universe_payload = read_json(universe)
        target_ids = tuple(
            dict.fromkeys(
                (
                    *(str(member["instrument_id"]) for member in universe_payload["members"]),
                    str(config.data["spy_instrument_id"]),
                )
            )
        )
        intervals = _symbol_intervals(pd.DataFrame(universe_payload["symbol_history"]))
        try:
            audit = _audit_acquisition_period(
                target_ids,
                intervals,
                start=_as_date(config.data["target_period"][0]),
                end=_as_date(config.data["target_period"][1]),
                output=target_root,
                monthly_chunks=monthly_chunks,
                paper_config_hash=config.config_hash,
            )
        except RuntimeError as exc:
            if "receipt set is incomplete" not in str(exc):
                raise
            results["download_data"] = "DATA ACQUISITION INCOMPLETE"
            return results
        results["download_data"] = {"status": "reused", "target_audit": audit}
    results["validate_data"] = validate_data_stage(config)
    sequence_root = config.artifact_root / "sequences"
    expected_manifests = [
        sequence_root / str(fold["id"]) / "sequence-manifest.json"
        for fold in config.evaluation["folds"]
    ]
    if not all(path.is_file() for path in expected_manifests):
        results["build_sequences"] = build_sequences_stage(config)
    else:
        results["build_sequences"] = "reused"
    results["validate_sequences"] = validate_sequences_stage(config)
    from execsim.ml.paper.benchmark import estimate_manifest_resources

    results["resource_plan"] = estimate_manifest_resources(
        tuple(expected_manifests),
        batch_size=int(config.representation["batch_size"]),
        max_epochs=int(config.representation["max_epochs"]),
        bounds={
            str(name): int(value)
            for name, value in config.representation["safe_resource_bounds"].items()
        },
    )
    if training_enabled:
        selection_receipt = config.artifact_root / "selection" / "rdm-lambda.json"
        if not selection_receipt.is_file():
            results["select_rdm_lambda"] = select_rdm_lambda_stage(
                config,
                training_cli_enabled=training_cli_enabled,
                runtime_approval=runtime_approval,
                trusted_local_resume=trusted_local_resume,
            )
        else:
            results["select_rdm_lambda"] = "reused"
        results["train_representations"] = train_representations_stage(
            config,
            training_cli_enabled=training_cli_enabled,
            runtime_approval=runtime_approval,
            trusted_local_resume=trusted_local_resume,
        )
        results["export_embeddings"] = export_embeddings_stage(config)
        results["train_volume_models"] = train_volume_models_stage(
            config,
            training_cli_enabled=training_cli_enabled,
            runtime_approval=runtime_approval,
        )
    else:
        results["training"] = "TRAINING NOT RUN"
    if evaluation_enabled:
        results["locked_test"] = open_locked_test(
            config,
            full_run_cli_enabled=full_run_cli_enabled,
            runtime_approval=runtime_approval,
        )
        results["evaluate_forecast"] = evaluate_forecasts_stage(
            config,
            full_run_cli_enabled=full_run_cli_enabled,
            runtime_approval=runtime_approval,
        )
        results["evaluate_representation"] = evaluate_representations_stage(
            config,
            full_run_cli_enabled=full_run_cli_enabled,
            runtime_approval=runtime_approval,
        )
        results["run_tca"] = run_tca_stage(
            config,
            full_run_cli_enabled=full_run_cli_enabled,
            runtime_approval=runtime_approval,
        )
        results["report"] = report_stage(
            config,
            full_run_cli_enabled=full_run_cli_enabled,
            runtime_approval=runtime_approval,
        )
        results["final_result_freeze"] = write_final_result_freeze(config)
    else:
        results["evaluation"] = "EMPIRICAL RESULT NOT AVAILABLE"
    return results


def _formation_artifacts_ready(config: PaperRunConfig) -> bool:
    """Resolve formation readiness using the selected protocol's own data contract."""
    if config.paper_run_id == "sparse-jepa-v2":
        return Path(config.data["formation_daily_corpus"]).is_file()
    return _has_parquet_corpus(Path(config.data["formation_corpus_root"]))


def _has_frozen_v2_formation_evidence(config: PaperRunConfig) -> bool:
    """Accept only the exact v2 formation artifacts named by the design freeze."""
    if config.paper_run_id != "sparse-jepa-v2":
        return False
    evidence = config.design_freeze.get("formation_evidence")
    if not isinstance(evidence, dict) or evidence.get("status") != "COMPLETE":
        return False
    paths = {
        "daily_corpus_sha256": Path(config.data["formation_daily_corpus"]),
        "daily_receipt_sha256": Path(config.data["formation_daily_receipt"]),
        "universe_manifest_sha256": Path(config.data["universe_manifest"]),
    }
    return _is_frozen_universe(
        paths["universe_manifest_sha256"], config_hash=config.config_hash
    ) and all(
        path.is_file() and file_sha256(path) == evidence.get(field) for field, path in paths.items()
    )


def _assert_representation_reuse(
    expected: Any,
    representation: Any,
    identity: Any,
    sequence_path: Path,
    options: Any,
    common_rdm_lambda: float,
) -> None:
    """Reject a completed representation directory whose current run identity changed."""
    import torch

    training_hash = stable_hash(
        {
            "representation": asdict(representation),
            "trainer": asdict(options),
            "common_rdm_lambda": common_rdm_lambda,
        }
    )
    p, mu, sigma = representation.target_parameters
    exact = {
        "geometry": representation.geometry,
        "predictor_family": representation.predictor_family,
        "fold_id": identity.fold_id,
        "cutoff": identity.cutoff,
        "universe_manifest_hash": identity.universe_manifest_hash,
        "dataset_manifest_hash": identity.dataset_manifest_hash,
        "sequence_manifest_hash": file_sha256(sequence_path),
        "normalization_hash": identity.normalization_hash,
        "architecture_hash": identity.architecture_hash,
        "training_config_hash": training_hash,
        "paper_config_hash": identity.config_hash,
        "rdm_projections": representation.rdm_projections_train,
        "calibrated_rdm_lambda": common_rdm_lambda,
        "adaptation": "none",
    }
    mismatches = [name for name, value in exact.items() if getattr(expected, name) != value]
    floating = {
        "generalized_gaussian_p": p,
        "generalized_gaussian_mu": mu,
        "generalized_gaussian_sigma": sigma,
        "target_rms": representation.target_rms,
        "target_zero_fraction": representation.target_zero_fraction,
    }
    mismatches.extend(
        name
        for name, value in floating.items()
        if not np.isclose(getattr(expected, name), value, rtol=0, atol=1e-12)
    )
    current_torch = str(torch.__version__)
    if expected.torch_compatibility == "exact":
        torch_matches = expected.torch_version == current_torch
    else:
        torch_matches = expected.torch_version.split(".")[:2] == current_torch.split(".")[:2]
    if not torch_matches:
        mismatches.append("torch_version")
    if mismatches:
        raise ValueError(f"Reusable representation identity mismatch: {sorted(mismatches)}")


def _validate_embedding_reuse(
    output: Path,
    *,
    expected: Any,
    sequence_path: Path,
    checkpoint_directory: Path,
    seed: int,
    geometry: str,
    adaptation: str,
) -> None:
    """Validate every identity and checksum before reusing an embedding corpus."""
    checkpoint = read_json(checkpoint_directory / "manifest.json")
    payload = read_json(output / "manifest.json")
    exact = {
        "fold_id": expected.fold_id,
        "seed": seed,
        "geometry": geometry,
        "adaptation": adaptation,
        "checkpoint_hash": checkpoint["weights_sha256"],
        "checkpoint_manifest_hash": file_sha256(checkpoint_directory / "manifest.json"),
        "sequence_manifest_hash": file_sha256(sequence_path),
        "normalization_hash": expected.normalization_hash,
        "paper_config_hash": expected.paper_config_hash,
        "training_cutoff": expected.cutoff,
        "pytorch_compatibility": expected.torch_compatibility,
    }
    mismatches = [name for name, value in exact.items() if payload.get(name) != value]
    files = payload.get("files")
    if not isinstance(files, list) or {item.get("partition") for item in files} != {
        "train",
        "validation",
        "test",
    }:
        mismatches.append("files")
    else:
        for item in files:
            path = output / str(item["path"])
            if not path.is_file() or file_sha256(path) != item.get("sha256"):
                mismatches.append(f"file:{item.get('partition')}")
    if mismatches:
        raise ValueError(f"Reusable embedding identity mismatch: {sorted(mismatches)}")


def _acquire_period(
    instrument_ids: tuple[str, ...],
    intervals: tuple[InstrumentSymbolInterval, ...],
    *,
    start: date,
    end: date,
    output: Path,
    fetcher: Any,
    data: Any,
    acquire_chunk: Any,
    monthly_chunks: Any,
) -> int:
    """Acquire every sourced symbol interval after proving trading-session coverage."""
    import exchange_calendars as xcals

    calendar = xcals.get_calendar("XNYS")
    sessions = tuple(value.date() for value in calendar.sessions_in_range(start, end))
    completed = 0
    for instrument_id in instrument_ids:
        for session_date in sessions:
            resolve_provider_symbol(intervals, instrument_id, session_date)
        history = sorted(
            (item for item in intervals if item.instrument_id == instrument_id),
            key=lambda item: item.start,
        )
        for interval in history:
            interval_start = max(interval.start, start)
            interval_end = min(interval.end, end)
            if interval_start > interval_end:
                continue
            for chunk in monthly_chunks(
                instrument_id, interval.symbol, interval_start, interval_end
            ):
                try:
                    acquire_chunk(
                        chunk,
                        output_directory=output,
                        fetch=fetcher,
                        config=data,
                        cli_enabled=True,
                    )
                    completed += 1
                except RuntimeError as exc:
                    causes = []
                    current: BaseException | None = exc
                    while current is not None:
                        causes.append(str(current))
                        current = current.__cause__
                    if not any("row count is zero" in message for message in causes):
                        raise
    return completed


def _audit_acquisition_period(
    instrument_ids: tuple[str, ...],
    intervals: tuple[InstrumentSymbolInterval, ...],
    *,
    start: date,
    end: date,
    output: Path,
    monthly_chunks: Any,
    paper_config_hash: str,
) -> dict[str, object]:
    """Require one compatible terminal receipt for every planned monthly chunk."""
    expected = {
        chunk.identity: chunk
        for instrument_id in instrument_ids
        for interval in sorted(
            (item for item in intervals if item.instrument_id == instrument_id),
            key=lambda item: (item.start, item.end, item.symbol),
        )
        if max(interval.start, start) <= min(interval.end, end)
        for chunk in monthly_chunks(
            instrument_id,
            interval.symbol,
            max(interval.start, start),
            min(interval.end, end),
        )
    }
    receipt_paths = {path.stem: path for path in output.glob("*.json")}
    missing = sorted(set(expected).difference(receipt_paths))
    unexpected = sorted(set(receipt_paths).difference(expected))
    if missing or unexpected:
        raise RuntimeError(
            "BLOCKED: target acquisition receipt set is incomplete or unexpected: "
            f"missing={missing[:10]}, unexpected={unexpected[:10]}"
        )
    complete = 0
    zero_row_exclusions = 0
    for identity, chunk in expected.items():
        receipt = read_json(receipt_paths[identity])
        mismatches = []
        expected_fields = {
            "chunk_identity": identity,
            "instrument_id": chunk.instrument_id,
            "symbol": chunk.symbol,
            "requested_start": chunk.start.isoformat(),
            "requested_end": chunk.end.isoformat(),
            "feed": chunk.feed,
            "adjustment": chunk.adjustment,
            "paper_config_hash": paper_config_hash,
        }
        for field, value in expected_fields.items():
            if receipt.get(field) != value:
                mismatches.append(field)
        status = receipt.get("status")
        response_path = output / f"{identity}.response"
        if status == "complete":
            if (
                not response_path.is_file()
                or receipt.get("response_sha256") != file_sha256(response_path)
                or int(receipt.get("row_count", 0)) <= 0
            ):
                mismatches.append("complete_response")
            complete += 1
        elif status == "failed" and "row count is zero" in str(receipt.get("error", "")):
            if response_path.exists():
                mismatches.append("failed_response_present")
            zero_row_exclusions += 1
        else:
            mismatches.append("terminal_status")
        if mismatches:
            raise ValueError(
                f"Target acquisition receipt {identity} is incompatible: {sorted(mismatches)}"
            )
    return {
        "status": ("complete" if zero_row_exclusions == 0 else "complete_with_zero_row_exclusions"),
        "expected_chunks": len(expected),
        "complete_chunks": complete,
        "zero_row_exclusion_chunks": zero_row_exclusions,
        "missing_chunks": 0,
        "unexpected_chunks": 0,
        "paper_config_hash": paper_config_hash,
    }


def _load_parquet_corpus(path: Path) -> pd.DataFrame:
    if path.is_file():
        return pd.read_parquet(path)
    files = sorted(path.rglob("*.parquet")) + sorted(path.rglob("*.response"))
    if not files:
        raise RuntimeError(f"BLOCKED: no Parquet corpus files found under {path}")
    frames = []
    for file in files:
        try:
            frames.append(pd.read_parquet(file))
        except Exception as exc:
            raise ValueError(f"Corpus artifact is not valid Parquet: {file}") from exc
    return pd.concat(frames, ignore_index=True)


def _has_parquet_corpus(path: Path) -> bool:
    """Return whether a file/root contains at least one candidate corpus artifact."""
    if path.is_file():
        return True
    return path.is_dir() and (
        next(path.rglob("*.parquet"), None) is not None
        or next(path.rglob("*.response"), None) is not None
    )


def _is_frozen_universe(path: Path, *, config_hash: str) -> bool:
    """Reject the tracked NOT RUN placeholder and incompatible empirical manifests."""
    if not path.is_file():
        return False
    payload = read_json(path)
    return (
        payload.get("status") == "complete"
        and payload.get("paper_config_hash") == config_hash
        and isinstance(payload.get("members"), list)
        and len(payload["members"]) == 100
    )


def _verify_frozen_universe_manifest(config: PaperRunConfig) -> Path:
    """Bind relocated runtime universe bytes to every fold's frozen sequence identity.

    The universe controls liquidity groups and therefore the locked evaluation
    population.  Sequence manifests are immutable upstream evidence; a
    relocated evaluator must consume exactly the byte-identical manifest they
    reference, never a similarly named repository-relative file.
    """
    universe_path = config.data_path("universe_manifest")
    if not universe_path.is_file():
        raise RuntimeError(f"BLOCKED: runtime universe manifest is unavailable: {universe_path}")
    expected_hashes: list[str] = []
    for fold in config.evaluation["folds"]:
        fold_id = str(fold["id"])
        sequence_path = config.artifact_root / "sequences" / fold_id / "sequence-manifest.json"
        if not sequence_path.is_file():
            raise RuntimeError(f"BLOCKED: frozen sequence manifest is unavailable: {sequence_path}")
        sequence = read_json(sequence_path)
        expected = sequence.get("universe_manifest_hash")
        if not isinstance(expected, str) or not expected:
            raise ValueError(f"Sequence manifest has no universe identity: {sequence_path}")
        expected_hashes.append(expected)
    if not expected_hashes or len(set(expected_hashes)) != 1:
        raise ValueError("Frozen sequence manifests do not share one universe manifest identity.")
    actual = file_sha256(universe_path)
    if actual != expected_hashes[0]:
        raise ValueError(
            "Runtime universe manifest checksum does not match the frozen sequence identity."
        )
    return universe_path


def _as_date(value: object) -> date:
    """Normalize YAML date scalars and ISO strings."""
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value))


def _symbol_intervals(frame: pd.DataFrame) -> tuple[InstrumentSymbolInterval, ...]:
    required = {"instrument_id", "symbol", "start", "end", "source"}
    if missing := required.difference(frame.columns):
        raise ValueError(f"Ticker-history source missing columns: {sorted(missing)}")
    return tuple(
        InstrumentSymbolInterval(
            str(row.instrument_id),
            str(row.symbol).upper(),
            date.fromisoformat(str(row.start)[:10]),
            date.fromisoformat(str(row.end)[:10]),
            str(row.source),
        )
        for row in frame.itertuples(index=False)
    )


def _paper_symbol_intervals(
    config: PaperRunConfig, universe: dict[str, Any]
) -> tuple[InstrumentSymbolInterval, ...]:
    """Resolve stock history plus the separately configured SPY benchmark identity."""
    intervals = _symbol_intervals(pd.DataFrame(universe.get("symbol_history", ())))
    spy_id = str(config.data["spy_instrument_id"])
    if not any(item.instrument_id == spy_id for item in intervals):
        source_path = Path(config.data["ticker_history"])
        if not source_path.is_file():
            raise RuntimeError("BLOCKED: sourced SPY ticker history is unavailable.")
        source = _symbol_intervals(pd.read_parquet(source_path))
        spy_intervals = tuple(item for item in source if item.instrument_id == spy_id)
        if not spy_intervals:
            raise RuntimeError("BLOCKED: sourced SPY ticker history is missing.")
        intervals = (*intervals, *spy_intervals)
    validate_symbol_history(intervals)
    return intervals


def _git_head() -> str:
    import subprocess

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _git_tree() -> str:
    """Return the exact tracked source tree used by historical training."""
    import subprocess

    completed = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"], check=True, capture_output=True, text=True
    )
    return completed.stdout.strip()


def _git_tracked_worktree_clean() -> bool:
    """Return whether tracked files exactly match the current source commit."""
    import subprocess

    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        check=True,
        capture_output=True,
        text=True,
    )
    return not completed.stdout.strip()


def _freeze_representation_parameters(config: PaperRunConfig) -> dict[str, object]:
    """Freeze validation-only RDM selection before the final training matrix."""
    if not _git_tracked_worktree_clean():
        raise RuntimeError("BLOCKED: representation parameters require a clean source tree.")
    selection_path = config.artifact_root / "selection" / "rdm-lambda.json"
    selection = _load_common_lambda_receipt(config)
    identity = {
        "schema_version": "paper-representation-parameter-freeze-v1",
        "status": "REPRESENTATION_PARAMETERS_FROZEN",
        "git_commit": _git_head(),
        "git_tree": _git_tree(),
        "paper_config_hash": config.config_hash,
        "rdm_lambda_receipt_sha256": file_sha256(selection_path),
        "selected_rdm_lambda": selection["selected_rdm_lambda"],
        "selection_partition": "fold-1/validation",
        "selection_seed": 13,
        "test_or_tca_used": False,
    }
    path = config.artifact_root / "selection" / "representation-parameter-freeze-v1.json"
    if path.is_file():
        existing = read_json(path)
        stable_existing = {key: existing.get(key) for key in identity}
        if stable_existing != identity:
            raise ValueError("Existing representation parameter freeze is incompatible.")
        return {**existing, "path": str(path)}
    payload = {**identity, "frozen_at_utc": datetime.now(UTC).isoformat()}
    write_json_atomic(path, payload)
    return {**payload, "path": str(path)}


def _require_representation_parameter_freeze(config: PaperRunConfig) -> dict[str, object]:
    """Reject final representation training before validation parameters are frozen."""
    path = config.artifact_root / "selection" / "representation-parameter-freeze-v1.json"
    if not path.is_file():
        raise RuntimeError("BLOCKED: representation parameter freeze is missing.")
    payload = read_json(path)
    selection_path = config.artifact_root / "selection" / "rdm-lambda.json"
    selection = _load_common_lambda_receipt(config)
    if (
        payload.get("schema_version") != "paper-representation-parameter-freeze-v1"
        or payload.get("status") != "REPRESENTATION_PARAMETERS_FROZEN"
        or payload.get("git_commit") != _git_head()
        or payload.get("git_tree") != _git_tree()
        or payload.get("paper_config_hash") != config.config_hash
        or payload.get("rdm_lambda_receipt_sha256") != file_sha256(selection_path)
        or payload.get("selected_rdm_lambda") != selection["selected_rdm_lambda"]
        or payload.get("selection_partition") != "fold-1/validation"
        or payload.get("selection_seed") != 13
        or payload.get("test_or_tca_used") is not False
    ):
        raise ValueError("Representation parameter freeze is incompatible.")
    return payload


def _require_parameter_freeze(config: PaperRunConfig) -> dict[str, object]:
    """Fail closed before any locked-test stage unless validation selections are frozen."""
    path = config.artifact_root / "selection" / "parameter-freeze-v1.json"
    if not path.is_file():
        raise RuntimeError(
            "BLOCKED: parameter-selection freeze is missing; locked test artifacts cannot be read."
        )
    payload = read_json(path)
    parameter_source = {"commit": _git_head(), "tree": _git_tree()}
    if getattr(config, "runtime_evaluation_root", None) is not None:
        execution = verify_evaluation_execution(
            config, source_commit=_git_head(), source_tree=_git_tree()
        )
        parameter_source = execution["upstream_parameter_source"]
    if (
        payload.get("status") != "PARAMETERS_FROZEN"
        or payload.get("paper_config_hash") != config.config_hash
        or payload.get("test_or_tca_used") is not False
        or payload.get("git_commit") != parameter_source["commit"]
        or payload.get("git_tree") != parameter_source["tree"]
    ):
        raise ValueError("Parameter-selection freeze is incompatible with this paper run.")
    receipt = config.artifact_root / "selection" / "rdm-lambda.json"
    if not receipt.is_file() or file_sha256(receipt) != payload.get("rdm_lambda_receipt_sha256"):
        raise ValueError("Parameter-selection freeze RDM receipt checksum mismatch.")
    selection = _load_common_lambda_receipt(config)
    if (
        selection.get("paper_config_hash") != config.config_hash
        or selection.get("selected_rdm_lambda") != payload.get("selected_rdm_lambda")
        or selection.get("test_or_tca_used") is not False
    ):
        raise ValueError("Parameter-selection freeze RDM receipt identity mismatch.")
    execution_receipt = config.artifact_root / "lightgbm" / "execution-receipt.json"
    if not execution_receipt.is_file() or payload.get(
        "lightgbm_execution_receipt_sha256"
    ) != file_sha256(execution_receipt):
        raise ValueError("Parameter-selection freeze LightGBM execution identity mismatch.")
    records = payload.get("lightgbm_manifests", [])
    expected_paths = {
        (
            Path("lightgbm")
            / str(fold["id"])
            / method
            / str(seed if seed is not None else "shared")
            / "manifest.json"
        ).as_posix()
        for fold in config.evaluation["folds"]
        for method, seed in (
            ("raw", None),
            ("untrained_neural", None),
            *(
                (geometry, int(seed))
                for geometry in ("dense", "sparse")
                for seed in config.representation["seeds"]
            ),
        )
    }
    recorded_paths = [str(record.get("path")) for record in records]
    if len(recorded_paths) != len(expected_paths) or set(recorded_paths) != expected_paths:
        raise ValueError("Parameter-selection freeze LightGBM matrix is incomplete or duplicated.")
    for record in records:
        artifact = config.artifact_root / str(record["path"])
        if not artifact.is_file() or file_sha256(artifact) != record.get("sha256"):
            raise ValueError("Parameter-selection freeze LightGBM checksum mismatch.")
    return payload


def write_prelock_amendment_receipt(config: PaperRunConfig) -> dict[str, object]:
    """Freeze the secondary observable analysis before TEST effectiveness is opened."""
    if not _git_tracked_worktree_clean():
        raise RuntimeError("BLOCKED: the pre-lock amendment requires a clean source tree.")
    specification = Path("docs/SECONDARY_OBSERVABLE_CAPACITY_EXTENSION.md")
    if not specification.is_file():
        raise RuntimeError("BLOCKED: the pre-lock observable-capacity specification is missing.")
    payload: dict[str, object] = {
        "schema_version": "paper-prelock-amendment-v1",
        "status": "OBSERVABLE_CAPACITY_EXTENSION_FROZEN",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "downstream_git_commit": _git_head(),
        "downstream_git_tree": _git_tree(),
        "paper_config_hash": config.config_hash,
        "specification": specification.as_posix(),
        "specification_sha256": file_sha256(specification),
        "target": (
            "log1p(actual future bucket volume) - "
            "log1p(causal historical baseline future bucket volume)"
        ),
        "horizons": [1, 2, 4, 8],
        "capacities": ["Affine", "MLP-64", "MLP-256"],
        "roles": {"train": "fit", "validation": "select ridge alpha", "test": "score once"},
        "classification": "SECONDARY_EXPLORATORY_MECHANISM_ANALYSIS",
        "locked_test_effectiveness_inspection": "NOT RUN",
    }
    path = config.artifact_root / "selection" / "prelock-observable-amendment-v1.json"
    _write_or_verify_timestamped_receipt(path, payload, timestamp_field="frozen_at_utc")
    return {**read_json(path), "path": str(path), "sha256": file_sha256(path)}


def write_locked_test_ready_receipt(config: PaperRunConfig) -> dict[str, object]:
    """Prove that code and validation-selected artifacts are frozen before TEST opens."""
    if not _git_tracked_worktree_clean():
        raise RuntimeError("BLOCKED: LOCKED-TEST-READY requires a clean source tree.")
    parameter_path = config.artifact_root / "selection" / "parameter-freeze-v1.json"
    parameter = _require_parameter_freeze(config)
    amendment = write_prelock_amendment_receipt(config)
    embedding_path = config.artifact_root / "embeddings" / "completion-receipt.json"
    execution_path = config.artifact_root / "lightgbm" / "execution-receipt.json"
    for required in (embedding_path, execution_path):
        if not required.is_file():
            raise RuntimeError(f"BLOCKED: LOCKED-TEST-READY input is missing: {required}")
    forbidden = (
        evaluation_root(config) / "evaluation" / "forecast-results.parquet",
        evaluation_root(config) / "evaluation" / "representation-accessibility.parquet",
        evaluation_root(config) / "tca" / "main.parquet",
    )
    if any(path.exists() for path in forbidden):
        raise RuntimeError("BLOCKED: locked result artifacts exist before LOCKED-TEST-READY.")
    payload = {
        "schema_version": "paper-locked-test-ready-v1",
        "status": "LOCKED-TEST-READY",
        "ready_at_utc": datetime.now(UTC).isoformat(),
        "downstream_git_commit": _git_head(),
        "downstream_git_tree": _git_tree(),
        "paper_config_hash": config.config_hash,
        "parameter_freeze_sha256": file_sha256(parameter_path),
        "embedding_completion_receipt_sha256": file_sha256(embedding_path),
        "lightgbm_execution_receipt_sha256": file_sha256(execution_path),
        "prelock_amendment_receipt_sha256": amendment["sha256"],
        "selected_rdm_lambda": parameter["selected_rdm_lambda"],
        "firewall": {
            "jepa_test_effectiveness": "NOT INSPECTED",
            "forecast_test_effectiveness": "NOT INSPECTED",
            "historical_tca": "NOT RUN",
            "bootstrap_inference": "NOT RUN",
            "confirmatory_p_values": "NOT COMPUTED",
        },
    }
    path = config.artifact_root / "selection" / "locked-test-ready-v1.json"
    _write_or_verify_timestamped_receipt(path, payload, timestamp_field="ready_at_utc")
    return {**read_json(path), "path": str(path), "sha256": file_sha256(path)}


def open_locked_test(
    config: PaperRunConfig,
    *,
    full_run_cli_enabled: bool,
    runtime_approval: PaperRuntimeApproval | None,
) -> dict[str, object]:
    """Open TEST once, only after authorization and the complete pre-lock firewall."""
    config.authorize(
        "locked_result_evaluation",
        approval=runtime_approval,
        cli_enabled=full_run_cli_enabled,
    )
    ready = write_locked_test_ready_receipt(config)
    parameter_path = config.artifact_root / "selection" / "parameter-freeze-v1.json"
    embedding_path = config.artifact_root / "embeddings" / "completion-receipt.json"
    execution_path = config.artifact_root / "lightgbm" / "execution-receipt.json"
    representation_completion = config.artifact_root / "representations" / "completion-receipt.json"
    payload = {
        "schema_version": "paper-locked-test-open-v1",
        "status": "LOCKED-TEST-OPENED",
        "opened_at_utc": datetime.now(UTC).isoformat(),
        "evaluation_git_commit": _git_head(),
        "evaluation_git_tree": _git_tree(),
        "paper_config_hash": config.config_hash,
        "locked_test_ready_receipt_sha256": ready["sha256"],
        "parameter_freeze_sha256": file_sha256(parameter_path),
        "embedding_completion_receipt_sha256": file_sha256(embedding_path),
        "lightgbm_execution_receipt_sha256": file_sha256(execution_path),
        "representation_completion_receipt_sha256": (
            file_sha256(representation_completion) if representation_completion.is_file() else None
        ),
        "confirmatory_contrasts": config.evaluation["confirmatory_contrast_definitions"],
        "secondary_analyses": ["observable-capacity", "support-regime", "sensitivities"],
    }
    path = config.artifact_root / "selection" / "locked-test-opened-v1.json"
    _write_or_verify_timestamped_receipt(path, payload, timestamp_field="opened_at_utc")
    return {**read_json(path), "path": str(path), "sha256": file_sha256(path)}


def _require_locked_test_opened(config: PaperRunConfig) -> dict[str, object]:
    path = config.artifact_root / "selection" / "locked-test-opened-v1.json"
    if not path.is_file():
        raise RuntimeError("BLOCKED: locked TEST has not been durably opened.")
    payload = read_json(path)
    ready_path = config.artifact_root / "selection" / "locked-test-ready-v1.json"
    opened_source = {"commit": _git_head(), "tree": _git_tree()}
    if getattr(config, "runtime_evaluation_root", None) is not None:
        execution = verify_evaluation_execution(
            config, source_commit=_git_head(), source_tree=_git_tree()
        )
        opened_source = execution["previous_evaluation_source"]
    if (
        payload.get("status") != "LOCKED-TEST-OPENED"
        or payload.get("evaluation_git_commit") != opened_source["commit"]
        or payload.get("evaluation_git_tree") != opened_source["tree"]
        or payload.get("paper_config_hash") != config.config_hash
        or not ready_path.is_file()
        or payload.get("locked_test_ready_receipt_sha256") != file_sha256(ready_path)
    ):
        raise ValueError("Locked-TEST-open receipt is incompatible with this execution.")
    _require_parameter_freeze(config)
    return payload


def _write_or_verify_timestamped_receipt(
    path: Path, payload: dict[str, object], *, timestamp_field: str
) -> None:
    if path.is_file():
        existing = read_json(path)
        comparable = {name: value for name, value in payload.items() if name != timestamp_field}
        existing_comparable = {
            name: value for name, value in existing.items() if name != timestamp_field
        }
        if existing_comparable != comparable:
            raise ValueError(f"Existing receipt is incompatible: {path}")
        return
    write_json_atomic(path, payload)


def write_final_result_freeze(config: PaperRunConfig) -> dict[str, object]:
    """Hash the complete locked result bundle after every declared stage finishes."""
    _require_parameter_freeze(config)
    opened_path = config.artifact_root / "selection" / "locked-test-opened-v1.json"
    _require_locked_test_opened(config)
    required = {
        "parameter_freeze": config.artifact_root / "selection" / "parameter-freeze-v1.json",
        "representation_evaluation": evaluation_root(config)
        / "evaluation"
        / "representation-evaluation-manifest.json",
        "forecast_evaluation": evaluation_root(config)
        / "evaluation"
        / "forecast-results.manifest.json",
        "tca": evaluation_root(config) / "tca" / "manifest.json",
        "report_provenance": evaluation_report_root(config)
        / config.paper_run_id
        / "provenance.json",
    }
    if getattr(config, "runtime_evaluation_root", None) is not None:
        required["evaluation_execution"] = evaluation_root(config) / "execution.json"
        required["report_completion"] = (
            evaluation_report_root(config) / config.paper_run_id / "completion.json"
        )
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise RuntimeError(f"BLOCKED: final result-freeze inputs are missing: {missing}")
    result_root = evaluation_report_root(config) / config.paper_run_id
    if getattr(config, "runtime_evaluation_root", None) is not None:
        from execsim.ml.paper.evaluation_artifacts import publish_bundle

        completion = read_json(required["report_completion"])
        identity = completion["identity"]
        if (
            identity.get("source_commit") != _git_head()
            or identity.get("source_tree") != _git_tree()
            or identity.get("paper_config_hash") != config.config_hash
            or identity.get("parameter_freeze_sha256") != file_sha256(required["parameter_freeze"])
        ):
            raise ValueError("Final report completion identity mismatch.")
        publish_bundle(result_root, identity=identity, build=lambda _: None)
        for name, digest in identity["input_sha256"].items():
            if file_sha256(evaluation_root(config) / name) != digest:
                raise ValueError("Final report numerical input checksum mismatch.")
    result_files = sorted(
        path
        for path in result_root.rglob("*")
        if path.is_file() and path.name != "final-result-freeze-v1.json"
    )
    if not result_files:
        raise RuntimeError("BLOCKED: final historical result bundle is empty.")
    representation_manifests = sorted(representation_root(config).glob("*/*/*/final/manifest.json"))
    representation_commits = [
        read_json(path).get("code_commit") for path in representation_manifests
    ]
    normalized_representation_commits = [
        commit.strip()
        for commit in representation_commits
        if isinstance(commit, str) and commit.strip()
    ]
    if (
        len(representation_manifests) != 18
        or len(normalized_representation_commits) != len(representation_manifests)
        or len(set(normalized_representation_commits)) != 1
    ):
        raise RuntimeError("BLOCKED: representation source identity is not uniquely frozen.")
    representation_source_commit = normalized_representation_commits[0]
    lightgbm_manifests = sorted((config.artifact_root / "lightgbm").glob("*/*/*/manifest.json"))
    if len(lightgbm_manifests) != 24:
        raise RuntimeError("BLOCKED: final result freeze requires 24 LightGBM manifests.")
    payload = {
        "schema_version": "paper-final-result-freeze-v1",
        "status": "FINAL-RESULTS-FROZEN",
        "frozen_at_utc": datetime.now(UTC).isoformat(),
        "representation_source_commit": representation_source_commit,
        "downstream_evaluation_commit": _git_head(),
        "downstream_evaluation_tree": _git_tree(),
        "paper_config_hash": config.config_hash,
        "selected_rdm_lambda": 10.0,
        "locked_test_open_receipt_sha256": file_sha256(opened_path),
        "stage_receipts": {
            name: {"path": str(path), "sha256": file_sha256(path)}
            for name, path in required.items()
        },
        "lightgbm_manifests": [
            {
                "path": str(path.relative_to(config.artifact_root)).replace("\\", "/"),
                "sha256": file_sha256(path),
            }
            for path in lightgbm_manifests
        ],
        "result_bundle": str(result_root),
        "result_files": [
            {
                "path": str(path.relative_to(result_root)).replace("\\", "/"),
                "sha256": file_sha256(path),
            }
            for path in result_files
        ],
    }
    path = (
        evaluation_root(config) / "selection" / "final-result-freeze-v1.json"
        if getattr(config, "runtime_evaluation_root", None) is not None
        else result_root / "final-result-freeze-v1.json"
    )
    _write_or_verify_timestamped_receipt(path, payload, timestamp_field="frozen_at_utc")
    return {**read_json(path), "path": str(path), "sha256": file_sha256(path)}


def _load_common_lambda_receipt(config: PaperRunConfig) -> dict[str, object]:
    """Recompute and verify the complete validation-only common-lambda decision."""
    from execsim.ml.representations.selection import (
        CommonLambdaCandidate,
        select_common_rdm_lambda,
    )

    path = config.artifact_root / "selection" / "rdm-lambda.json"
    payload = read_json(path)
    if (
        payload.get("schema_version") != "paper-rdm-lambda-selection-v1"
        or payload.get("paper_config_hash") != config.config_hash
        or payload.get("selection_partition") != "fold-1/validation"
        or payload.get("seed") != 13
        or payload.get("test_or_tca_used") is not False
    ):
        raise ValueError("Common RDM lambda receipt identity is incompatible.")
    try:
        candidates = tuple(CommonLambdaCandidate(**row) for row in payload["candidates"])
    except (KeyError, TypeError) as exc:
        raise ValueError("Common RDM lambda receipt candidate matrix is malformed.") from exc
    selected = select_common_rdm_lambda(candidates)
    if float(payload.get("selected_rdm_lambda", -1)) != selected:
        raise ValueError("Common RDM lambda receipt does not reproduce its selection.")
    return payload


def _latest_periodic_checkpoint(output_root: Path) -> Path | None:
    """Return the highest complete periodic step for deterministic local continuation."""
    periodic = output_root / "periodic"
    if not periodic.is_dir():
        return None
    candidates = [
        path
        for path in sorted(periodic.glob("step=*"))
        if (path / "weights" / "manifest.json").is_file()
        and (path / "weights" / "model.safetensors").is_file()
        and (path / "resume.pt").is_file()
        and (path / "resume.sha256").is_file()
    ]
    return candidates[-1] if candidates else None


def _append_untrained_control(
    values: tuple[pd.DataFrame, Any, pd.DataFrame, Any], *, fold_seed: int
) -> tuple[pd.DataFrame, Any, pd.DataFrame, Any]:
    """Append the frozen nonlinear, target-free neural placebo by stable case identity."""
    from execsim.ml.paper.features import append_untrained_neural_control_frames

    return append_untrained_neural_control_frames(values, fold_seed=fold_seed)


def _grouped_support_transitions(
    group_ids: np.ndarray, latents: np.ndarray, labels: np.ndarray
) -> dict[str, object]:
    from execsim.ml.representations.diagnostics import support_transition_diagnostics

    results = []
    for group_id in np.unique(group_ids):
        selected = group_ids == group_id
        if selected.sum() >= 2:
            results.append(support_transition_diagnostics(latents[selected], labels[selected]))
    if not results:
        raise ValueError("Support diagnostics require consecutive rows within an instrument.")
    regime_values: dict[str, list[float]] = {}
    matrix: dict[str, dict[str, int]] = {}
    support_state_matrix = {
        "inactive_to_inactive": 0,
        "inactive_to_active": 0,
        "active_to_inactive": 0,
        "active_to_active": 0,
    }
    for result in results:
        per_regime = result["per_regime_support_jaccard"]
        transitions = result["regime_transition_matrix"]
        if not isinstance(per_regime, dict) or not isinstance(transitions, dict):
            raise TypeError("Support transition diagnostics have an invalid mapping shape.")
        for regime, value in per_regime.items():
            regime_values.setdefault(regime, []).append(float(value))
        for source, targets in transitions.items():
            if not isinstance(targets, dict):
                raise TypeError("Regime transition row must be a mapping.")
            for target, count in targets.items():
                matrix.setdefault(source, {})[target] = matrix.setdefault(source, {}).get(
                    target, 0
                ) + int(count)
        dimension_transitions = result["support_state_transition_matrix"]
        if not isinstance(dimension_transitions, dict):
            raise TypeError("Support state transition matrix must be a mapping.")
        for name in support_state_matrix:
            support_state_matrix[name] += int(dimension_transitions[name])
    return {
        "mean_consecutive_support_jaccard": float(
            np.mean([result["mean_consecutive_support_jaccard"] for result in results])
        ),
        "support_transition_rate": float(
            np.mean([result["support_transition_rate"] for result in results])
        ),
        "chance_support_jaccard": float(
            np.mean([result["chance_support_jaccard"] for result in results])
        ),
        "support_state_transition_matrix": support_state_matrix,
        "per_regime_support_jaccard": {
            regime: float(np.mean(values)) for regime, values in regime_values.items()
        },
        "regime_transition_matrix": matrix,
    }


def _stream_embedding_diagnostics(
    path: Path,
    states: pd.DataFrame,
    *,
    latent_width: int = 128,
    batch_size: int = 4096,
) -> tuple[dict[str, float], dict[str, object], dict[str, int]]:
    """Compute TEST support diagnostics without materializing the embedding matrix."""
    import pyarrow.parquet as pq

    from execsim.ml.representations.diagnostics import support_transition_diagnostics

    required = {"sample_id", "session_id", "instrument_id", "session_date", "as_of_token", "regime"}
    missing = required.difference(states.columns)
    if missing:
        raise ValueError(f"Regime state frame is missing columns: {sorted(missing)}")
    ordered = states.sort_values(
        ["instrument_id", "session_date", "as_of_token"], kind="stable"
    ).reset_index(drop=True)
    ordered["sample_id"] = ordered["sample_id"].astype(str)
    if ordered["sample_id"].duplicated().any():
        raise ValueError("Regime state frame duplicates TEST sample identities.")
    metadata = ordered.set_index("sample_id")
    if not metadata.index.is_unique:
        raise ValueError("Regime state frame duplicates TEST sample identities.")
    expected_ids = ordered["sample_id"].astype(str).tolist()
    seen: set[str] = set()
    count = 0
    value_sum = np.zeros(latent_width, dtype=np.float64)
    gram = np.zeros((latent_width, latent_width), dtype=np.float64)
    activation_count = np.zeros(latent_width, dtype=np.int64)
    active_counts: list[int] = []
    hoyer_sum = 0.0
    all_finite = True
    current_session: str | None = None
    session_latents: list[np.ndarray] = []
    session_labels: list[str] = []
    transition_results: list[dict[str, object]] = []
    regime_counts: dict[str, int] = {}

    def finish_session() -> None:
        if len(session_latents) >= 2:
            transition_results.append(
                support_transition_diagnostics(
                    np.stack(session_latents), np.asarray(session_labels, dtype=object)
                )
            )
        session_latents.clear()
        session_labels.clear()

    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=batch_size, columns=["sample_id", "embedding"]):
        frame = batch.to_pandas()
        for sample_id, embedding in zip(frame["sample_id"], frame["embedding"], strict=True):
            key = str(sample_id)
            if key in seen or key not in metadata.index:
                raise ValueError(f"Embedding TEST identity is duplicate or unexpected: {key}")
            if count >= len(expected_ids) or key != expected_ids[count]:
                raise ValueError(
                    "Embedding TEST rows do not follow the frozen sequence-index order."
                )
            seen.add(key)
            row = metadata.loc[key]
            latent = np.asarray(embedding, dtype=np.float64)[:latent_width]
            if latent.shape != (latent_width,):
                raise ValueError("Embedding latent width is incompatible with diagnostics.")
            finite = bool(np.isfinite(latent).all())
            all_finite = all_finite and finite
            safe = np.nan_to_num(latent)
            support = np.abs(safe) > 0.0
            value_sum += safe
            gram += np.outer(safe, safe)
            activation_count += support
            active = int(support.sum())
            active_counts.append(active)
            l2 = float(np.linalg.norm(safe, ord=2))
            hoyer_sum += float(
                (np.sqrt(latent_width) - np.linalg.norm(safe, ord=1) / max(l2, 1e-12))
                / (np.sqrt(latent_width) - 1)
            )
            session_id = str(row["session_id"])
            if current_session is not None and session_id != current_session:
                finish_session()
            current_session = session_id
            label = str(row["regime"])
            session_latents.append(safe)
            session_labels.append(label)
            regime_counts[label] = regime_counts.get(label, 0) + 1
            count += 1
    finish_session()
    if count == 0 or count != len(expected_ids) or len(seen) != len(expected_ids):
        raise ValueError("Embedding TEST rows do not exactly match the frozen sequence index.")
    mean = value_sum / count
    centered_gram = gram - count * np.outer(mean, mean)
    eigenvalues = np.clip(np.linalg.eigvalsh(centered_gram), 0.0, None)
    probabilities = eigenvalues / max(float(eigenvalues.sum()), 1e-12)
    effective_rank = float(np.exp(-np.sum(probabilities * np.log(probabilities + 1e-12))))
    activation = activation_count / count
    entropy = -activation * np.log(activation + 1e-12) - (1 - activation) * np.log(
        1 - activation + 1e-12
    )
    active_values = np.asarray(active_counts)
    diagnostics = {
        "finite": float(all_finite),
        "mean_variance": float(np.trace(centered_gram) / count / latent_width),
        "effective_rank": effective_rank,
        "zero_fraction": float(1.0 - activation.mean()),
        "active_dimension_fraction": float((activation > 0).mean()),
        "dead_dimension_fraction": float((activation == 0).mean()),
        "always_on_dimension_fraction": float((activation == 1).mean()),
        "mean_hoyer_sparsity": hoyer_sum / count,
        "mean_support_entropy": float(entropy.mean()),
        "mean_active_dimensions": float(active_values.mean()),
        "median_active_dimensions": float(np.median(active_values)),
        "p95_active_dimensions": float(np.quantile(active_values, 0.95)),
        "activation_frequency_q05": float(np.quantile(activation, 0.05)),
        "activation_frequency_q50": float(np.quantile(activation, 0.50)),
        "activation_frequency_q95": float(np.quantile(activation, 0.95)),
    }
    if not transition_results:
        raise ValueError("Support diagnostics require consecutive rows within a session.")
    transitions = _combine_support_transition_results(transition_results)
    return diagnostics, transitions, regime_counts


def _combine_support_transition_results(results: list[dict[str, object]]) -> dict[str, object]:
    """Combine session-local transition summaries with the frozen equal-session estimator."""
    regime_values: dict[str, list[float]] = {}
    matrix: dict[str, dict[str, int]] = {}
    support_state_matrix = {
        "inactive_to_inactive": 0,
        "inactive_to_active": 0,
        "active_to_inactive": 0,
        "active_to_active": 0,
    }
    for result in results:
        per_regime = cast(dict[str, float], result["per_regime_support_jaccard"])
        transitions = cast(dict[str, dict[str, int]], result["regime_transition_matrix"])
        for regime, value in per_regime.items():
            regime_values.setdefault(regime, []).append(float(value))
        for source, targets in transitions.items():
            for target, value in targets.items():
                matrix.setdefault(source, {})[target] = matrix.setdefault(source, {}).get(
                    target, 0
                ) + int(value)
        dimension_transitions = cast(dict[str, int], result["support_state_transition_matrix"])
        for name in support_state_matrix:
            support_state_matrix[name] += int(dimension_transitions[name])
    return {
        "mean_consecutive_support_jaccard": float(
            np.mean([cast(float, row["mean_consecutive_support_jaccard"]) for row in results])
        ),
        "support_transition_rate": float(
            np.mean([cast(float, row["support_transition_rate"]) for row in results])
        ),
        "chance_support_jaccard": float(
            np.mean([cast(float, row["chance_support_jaccard"]) for row in results])
        ),
        "support_state_transition_matrix": support_state_matrix,
        "per_regime_support_jaccard": {
            name: float(np.mean(values)) for name, values in regime_values.items()
        },
        "regime_transition_matrix": matrix,
    }


def _feature_resolver(scale: pd.DataFrame, shape: pd.DataFrame, instrument_id: str) -> Any:
    def resolve(
        symbol: str,
        session_date: date,
        generated_at: pd.Timestamp,
        observations: pd.DataFrame | None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        del symbol, observations
        local = generated_at.tz_convert("America/New_York")
        as_of = ((local.hour * 60 + local.minute) - 570) // 15
        selected = scale.loc[
            (scale["instrument_id"].astype(str) == instrument_id)
            & (scale["session_date"].astype(str) == session_date.isoformat())
            & (scale["as_of"].astype(int) == as_of)
        ]
        if len(selected) != 1:
            raise ValueError(
                f"No unique frozen LightGBM row for {instrument_id}/{session_date}/{as_of}."
            )
        case_id = str(selected["sample_id"].iloc[0])
        selected_shape = shape.loc[shape["case_id"].astype(str) == case_id]
        if selected_shape.empty:
            raise ValueError(f"No frozen shape rows for LightGBM case {case_id}.")
        return selected, selected_shape

    return resolve
