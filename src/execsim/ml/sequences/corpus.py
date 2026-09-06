"""Corpus-wide, fold-safe sequence construction from validated minute bars."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable
from dataclasses import replace
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from execsim.data.paper.corporate_actions import apply_point_in_time_split_adjustment
from execsim.data.paper.manifests import file_sha256
from execsim.data.paper.partitions import fold_training_cutoff, resolve_fold_partition
from execsim.data.paper.resolution_quality import (
    aggregate_observed_tokens,
    assess_session_resolution_quality,
)
from execsim.data.paper.validation import validate_exact_xnys_session
from execsim.ml.sequences.builder import _aggregate_tokens, build_session_sequence
from execsim.ml.sequences.index import build_sample_index, write_sample_index
from execsim.ml.sequences.manifests import (
    SequenceManifest,
    write_sequence_manifest,
    write_sequence_record,
)
from execsim.ml.sequences.normalization import RobustFoldNormalizer
from execsim.ml.sequences.schemas import SequenceRecord


def build_fold_sequence_corpus(
    bars: pd.DataFrame,
    *,
    universe_members: tuple[dict[str, Any], ...],
    corporate_actions: pd.DataFrame,
    fold_id: str,
    output_root: Path,
    universe_manifest_hash: str,
    corporate_action_manifest_hash: str,
    config_hash: str,
    spy_instrument_id: str,
    data_classification: str,
    quality_protocol: str = "exact-minute-v1",
    symbol_history: tuple[dict[str, Any], ...] = (),
) -> SequenceManifest:
    """Build, normalize, index, and manifest every valid session in one fold."""
    instruments = tuple(str(member["instrument_id"]) for member in universe_members)
    if spy_instrument_id in instruments:
        raise ValueError("SPY benchmark identity must not be an execution-universe member.")
    available = set(bars["instrument_id"].astype(str).unique())
    if spy_instrument_id not in available:
        raise ValueError("SPY corpus is required for every paper sequence build.")
    missing = set(instruments).difference(available)
    if missing:
        raise ValueError(f"Universe instruments missing from raw corpus: {sorted(missing)}")
    if data_classification != "synthetic_fixture" and not symbol_history:
        raise ValueError("Historical sequence builds require sourced symbol history.")

    def load_sessions(
        instrument_id: str,
        active_fold_id: str,
        exclusions: list[dict[str, str]] | None,
    ) -> list[tuple[date, pd.DataFrame]]:
        return _validated_sessions(
            bars,
            instrument_id,
            fold_id=active_fold_id,
            exclusions=exclusions,
            quality_protocol=quality_protocol,
        )

    return _build_fold_sequence_corpus_from_sessions(
        load_sessions,
        universe_members=universe_members,
        corporate_actions=corporate_actions,
        fold_id=fold_id,
        output_root=output_root,
        universe_manifest_hash=universe_manifest_hash,
        corporate_action_manifest_hash=corporate_action_manifest_hash,
        config_hash=config_hash,
        spy_instrument_id=spy_instrument_id,
        data_classification=data_classification,
        quality_protocol=quality_protocol,
        symbol_history=symbol_history,
    )


def build_fold_sequence_corpus_from_root(
    corpus_root: Path,
    *,
    universe_members: tuple[dict[str, Any], ...],
    corporate_actions: pd.DataFrame,
    fold_id: str,
    output_root: Path,
    universe_manifest_hash: str,
    corporate_action_manifest_hash: str,
    config_hash: str,
    spy_instrument_id: str,
    data_classification: str,
    quality_protocol: str = "exact-minute-v1",
    symbol_history: tuple[dict[str, Any], ...] = (),
) -> SequenceManifest:
    """Build one fold while holding only one instrument corpus in memory."""
    if not corpus_root.is_dir():
        raise FileNotFoundError(f"Paper corpus root does not exist: {corpus_root}")
    if data_classification != "synthetic_fixture" and not symbol_history:
        raise ValueError("Historical sequence builds require sourced symbol history.")

    def load_sessions(
        instrument_id: str,
        active_fold_id: str,
        exclusions: list[dict[str, str]] | None,
    ) -> list[tuple[date, pd.DataFrame]]:
        bars = load_corpus_instrument(corpus_root, instrument_id)
        return _validated_sessions(
            bars,
            instrument_id,
            fold_id=active_fold_id,
            exclusions=exclusions,
            quality_protocol=quality_protocol,
        )

    return _build_fold_sequence_corpus_from_sessions(
        load_sessions,
        universe_members=universe_members,
        corporate_actions=corporate_actions,
        fold_id=fold_id,
        output_root=output_root,
        universe_manifest_hash=universe_manifest_hash,
        corporate_action_manifest_hash=corporate_action_manifest_hash,
        config_hash=config_hash,
        spy_instrument_id=spy_instrument_id,
        data_classification=data_classification,
        quality_protocol=quality_protocol,
        symbol_history=symbol_history,
    )


SessionLoader = Callable[[str, str, list[dict[str, str]] | None], list[tuple[date, pd.DataFrame]]]


def _build_fold_sequence_corpus_from_sessions(
    load_sessions: SessionLoader,
    *,
    universe_members: tuple[dict[str, Any], ...],
    corporate_actions: pd.DataFrame,
    fold_id: str,
    output_root: Path,
    universe_manifest_hash: str,
    corporate_action_manifest_hash: str,
    config_hash: str,
    spy_instrument_id: str,
    data_classification: str,
    quality_protocol: str,
    symbol_history: tuple[dict[str, Any], ...],
) -> SequenceManifest:
    """Build one fold from a bounded session-loader boundary."""
    cutoff = fold_training_cutoff(fold_id)
    fold_root = output_root / fold_id
    instruments = tuple(str(member["instrument_id"]) for member in universe_members)
    if spy_instrument_id in instruments:
        raise ValueError("SPY benchmark identity must not be an execution-universe member.")
    spy_sessions = load_sessions(spy_instrument_id, "unresolved", None)
    if not spy_sessions:
        raise ValueError("SPY corpus is required for every paper sequence build.")
    spy_dates = [item[0] for item in spy_sessions]
    spy_by_date = dict(spy_sessions)
    records: list[tuple[str, SequenceRecord]] = []
    exclusions: list[dict[str, str]] = []
    raw_hashes: list[str] = []
    for member in universe_members:
        instrument_id = str(member["instrument_id"])
        sessions = load_sessions(instrument_id, fold_id, exclusions)
        history: list[tuple[date, pd.DataFrame]] = []
        for session_date, session in sessions:
            try:
                partition = resolve_fold_partition(fold_id, session_date)
            except ValueError:
                continue
            prior = history[-20:]
            spy_index = bisect_left(spy_dates, session_date)
            spy_prior = spy_sessions[max(0, spy_index - 20) : spy_index]
            prior_spy_session = spy_by_date.get(session_date)
            if not prior or not spy_prior or prior_spy_session is None:
                exclusions.append(
                    {
                        "fold_id": fold_id,
                        "instrument_id": instrument_id,
                        "session_date": session_date.isoformat(),
                        "reason": "insufficient causal stock or SPY history",
                    }
                )
                history.append((session_date, session))
                continue
            symbol = _session_symbol(session)
            if symbol_history:
                _verify_sourced_symbol(
                    symbol_history,
                    instrument_id=instrument_id,
                    session_date=session_date,
                    observed_symbol=symbol,
                )
                _verify_sourced_symbol(
                    symbol_history,
                    instrument_id=spy_instrument_id,
                    session_date=session_date,
                    observed_symbol=_session_symbol(prior_spy_session),
                )
            record_cutoff = prior[-1][0]
            adjusted = _adjust_for_cutoff(
                session,
                corporate_actions,
                instrument_id=instrument_id,
                cutoff=record_cutoff,
            )
            adjusted_prior = [
                (
                    prior_date,
                    _adjust_for_cutoff(
                        prior_session,
                        corporate_actions,
                        instrument_id=instrument_id,
                        cutoff=record_cutoff,
                    ),
                )
                for prior_date, prior_session in prior
            ]
            previous_close = float(adjusted_prior[-1][1]["close"].iloc[-1])
            stock_seasonal = _seasonal_frame(adjusted_prior, quality_protocol=quality_protocol)
            spy_seasonal = _seasonal_frame(spy_prior, quality_protocol=quality_protocol)
            source_hash = _frame_hash(session)
            record = build_session_sequence(
                adjusted,
                instrument_id=instrument_id,
                symbol=symbol,
                source_sha256=source_hash,
                cutoff=record_cutoff.isoformat(),
                seasonal=stock_seasonal,
                spy_bars=prior_spy_session,
                spy_seasonal=spy_seasonal,
                previous_close=previous_close,
                data_classification=data_classification,
                training_cutoff=cutoff.isoformat(),
                quality_protocol=quality_protocol,
            )
            records.append((partition, record))
            raw_hashes.append(source_hash)
            history.append((session_date, session))
    training = [record for partition, record in records if partition == "train"]
    if not training:
        raise ValueError(f"No valid training sequences were produced for {fold_id}.")
    training_values = np.stack([record.features for record in training])
    training_masks = np.stack([record.token_mask for record in training])
    normalizer = RobustFoldNormalizer.fit(training_values, training_masks)
    sequence_files: list[str] = []
    index_files: list[str] = []
    counts = {"train": 0, "validation": 0, "test": 0}
    for partition, record in records:
        normalized = replace(
            record,
            features=normalizer.transform(record.features[None, ...], record.token_mask[None, ...])[
                0
            ],
        )
        sequence_path = write_sequence_record(normalized, fold_root / "sessions" / partition)
        relative_sequence = str(sequence_path.relative_to(fold_root)).replace("\\", "/")
        sequence_hash = file_sha256(sequence_path)
        samples = build_sample_index(
            normalized,
            fold_id=fold_id,
            partition=partition,
            source_sequence_hash=sequence_hash,
        )
        index_path = fold_root / "indexes" / partition / f"{record.session_id}.parquet"
        write_sample_index(samples, index_path)
        sequence_files.append(relative_sequence)
        index_files.append(str(index_path.relative_to(fold_root)).replace("\\", "/"))
        counts[partition] += 1
    return write_sequence_manifest(
        fold_root / "sequence-manifest.json",
        fold_id=fold_id,
        cutoff=cutoff.isoformat(),
        raw_hashes=tuple(raw_hashes),
        sequence_files=tuple(sequence_files),
        normalizer=normalizer,
        universe_manifest_hash=universe_manifest_hash,
        corporate_action_manifest_hash=corporate_action_manifest_hash,
        config_hash=config_hash,
        index_files=tuple(index_files),
        partition_counts=counts,
        exclusions=tuple(exclusions),
        quality_protocol=quality_protocol,
    )


def load_corpus_instrument(root: Path, instrument_id: str) -> pd.DataFrame:
    """Load only one instrument's bounded Parquet partitions from a large corpus."""
    files = sorted(
        {
            *root.glob(f"{instrument_id}-*.response"),
            *root.glob(f"{instrument_id}-*.parquet"),
            *root.glob(f"**/instrument_id={instrument_id}/**/*.parquet"),
        }
    )
    if not files:
        raise FileNotFoundError(f"No corpus partitions for instrument {instrument_id}.")
    frame = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    observed = set(frame["instrument_id"].astype(str).unique())
    if observed != {instrument_id}:
        raise ValueError(
            f"Corpus partitions for {instrument_id} contain incompatible identities: {observed}"
        )
    return frame


def _validated_sessions(
    bars: pd.DataFrame,
    instrument_id: str,
    *,
    fold_id: str = "unresolved",
    exclusions: list[dict[str, str]] | None = None,
    quality_protocol: str = "exact-minute-v1",
) -> list[tuple[date, pd.DataFrame]]:
    selected = bars.loc[bars["instrument_id"].astype(str) == instrument_id].copy()
    if selected.empty:
        return []
    timestamps = pd.to_datetime(selected["timestamp"])
    local_dates = timestamps.dt.tz_convert("America/New_York").dt.date
    sessions: list[tuple[date, pd.DataFrame]] = []
    for session_date, session in selected.groupby(local_dates, sort=True):
        session = session.sort_values("timestamp", kind="stable").reset_index(drop=True)
        errors: tuple[str, ...]
        if quality_protocol == "resolution-aware-v2":
            quality = assess_session_resolution_quality(session)
            errors = () if quality.token_valid_full_session else (quality.invalid_token_reason,)
        elif quality_protocol == "exact-minute-v1":
            errors = validate_exact_xnys_session(session)
        else:
            raise ValueError(f"Unknown paper quality protocol: {quality_protocol}")
        if errors:
            if exclusions is not None:
                exclusions.append(
                    {
                        "fold_id": fold_id,
                        "instrument_id": instrument_id,
                        "session_date": session_date.isoformat(),
                        "reason": "; ".join(errors),
                    }
                )
            continue
        sessions.append((session_date, session))
    return sessions


def _seasonal_frame(
    history: list[tuple[date, pd.DataFrame]], *, quality_protocol: str
) -> pd.DataFrame:
    rows = []
    for session_date, session in history[-20:]:
        tokens = (
            aggregate_observed_tokens(session)
            if quality_protocol == "resolution-aware-v2"
            else _aggregate_tokens(session)
        )
        for bucket_index, token in enumerate(tokens.itertuples(index=False)):
            rows.append(
                {
                    "session_date": session_date,
                    "bucket_index": bucket_index,
                    "volume": float(token.volume),
                    "dollar_volume": float(token.vwap * token.volume),
                    "trade_count": float(token.trade_count),
                }
            )
    return pd.DataFrame(rows)


def _session_symbol(session: pd.DataFrame) -> str:
    symbols = tuple(session["symbol"].astype(str).str.upper().drop_duplicates())
    if len(symbols) != 1 or not symbols[0]:
        raise ValueError("Each sequence session must contain one non-empty observed symbol.")
    return symbols[0]


def _verify_sourced_symbol(
    symbol_history: tuple[dict[str, Any], ...],
    *,
    instrument_id: str,
    session_date: date,
    observed_symbol: str,
) -> None:
    matches = [
        str(item["symbol"]).upper()
        for item in symbol_history
        if str(item["instrument_id"]) == instrument_id
        and date.fromisoformat(str(item["start"])[:10])
        <= session_date
        <= date.fromisoformat(str(item["end"])[:10])
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"BLOCKED: no unique sourced symbol for {instrument_id} on {session_date}."
        )
    if observed_symbol != matches[0]:
        raise ValueError(
            f"Observed symbol {observed_symbol} does not match sourced symbol {matches[0]} "
            f"for {instrument_id} on {session_date}."
        )


def _adjust_for_cutoff(
    session: pd.DataFrame,
    actions: pd.DataFrame,
    *,
    instrument_id: str,
    cutoff: date,
) -> pd.DataFrame:
    del cutoff  # model-training cutoff is distinct from market-information as-of
    if actions.empty:
        return session.copy()
    from execsim.data.paper.corporate_actions import point_in_time_split_factor

    observation_at = pd.Timestamp(session["timestamp"].iloc[-1])
    factor = point_in_time_split_factor(
        actions,
        instrument_id=instrument_id,
        observation_at=observation_at,
        market_information_as_of=observation_at,
    )
    return apply_point_in_time_split_adjustment(
        session, pd.Series(np.full(len(session), factor), index=session.index)
    )


def _frame_hash(frame: pd.DataFrame) -> str:
    import hashlib

    values = frame.sort_values("timestamp", kind="stable").to_json(
        orient="records", date_format="iso", double_precision=15
    )
    return hashlib.sha256(values.encode()).hexdigest()
