"""Independent date-level TCA tasks preserving the frozen matched replay."""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date, time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from execsim.data.paper.manifests import file_sha256
from execsim.ml.paper.evaluation_artifacts import VerifiedArtifact, publish_frames, verify_artifact
from execsim.ml.paper.evaluation_workers import (
    EWMA_FILES,
    NATIVE_THREAD_VARIABLES,
    configure_evaluation_worker,
)
from execsim.ml.paper.forecast_ledger import (
    EWMAForecastLedgerProvider,
    ForecastLedgerDate,
    PaperForecastLedgerProvider,
)
from execsim.ml.paper.tca import run_historical_tca

_VERIFIED_LEDGERS: dict[Path, VerifiedArtifact] = {}

LEARNED_LEDGER_FILES = ("scale.parquet", "shape.parquet", "metrics.parquet")


def _verified_ledger(
    path: Path, expected: dict[str, Any], names: tuple[str, ...]
) -> VerifiedArtifact:
    """Hash immutable ledgers once per worker, then reject identity or file-state changes."""
    key = path.resolve()
    artifact = _VERIFIED_LEDGERS.get(key)
    if artifact is None:
        artifact = VerifiedArtifact(path, identity=expected, names=names)
        _VERIFIED_LEDGERS[key] = artifact
    else:
        artifact.check(path, expected)
    return artifact


@dataclass(frozen=True, slots=True)
class TCAWork:
    """File-only work message for all matched methods on one trading date."""

    input_directory: Path
    output_directory: Path
    ledger_records: tuple[tuple[str, int | None, Path, dict[str, Any]], ...]
    ewma_records: dict[str, tuple[Path, dict[str, Any]]]
    training_cutoff: date
    sequence_hash: str
    tca_config: dict[str, Any]
    identity: dict[str, Any]


def _tca_as_of_origins(tca_config: Mapping[str, Any]) -> tuple[int, ...]:
    """Return the 15-minute origins consumed by the locked TCA window."""
    start = time.fromisoformat(str(tca_config["window"][0]))
    end = time.fromisoformat(str(tca_config["window"][1]))
    start_offset = (start.hour * 60 + start.minute) - (9 * 60 + 30)
    end_offset = (end.hour * 60 + end.minute) - (9 * 60 + 30)
    if start_offset < 0 or end_offset <= start_offset or start_offset % 15 or end_offset % 15:
        raise ValueError("TCA window must align to the 15-minute sequence grid.")
    return tuple(range(start_offset // 15, end_offset // 15))


def _require_integer_key(frame: pd.DataFrame, column: str, *, ledger: str) -> None:
    """Reject null, floating-point, and string keys before any integer coercion."""
    values = frame[column]
    if (
        values.isna().any()
        or not pd.api.types.is_integer_dtype(values.dtype)
        or pd.api.types.is_bool_dtype(values.dtype)
    ):
        raise ValueError(f"TCA preflight found an invalid non-null integer {ledger} key: {column}.")


def _validate_learned_case(
    directory: Path,
    *,
    instrument_id: str,
    session_date: date,
    fold_id: str,
    training_cutoff: date,
    origins: tuple[int, ...],
    scale_frame: pd.DataFrame | None = None,
    shape_frame: pd.DataFrame | None = None,
) -> None:
    """Verify every learned ledger row and future bucket needed by TCA."""
    scale = (
        scale_frame.copy()
        if scale_frame is not None
        else pd.read_parquet(
            directory / "scale.parquet", filters=[("instrument_id", "==", instrument_id)]
        )
    )
    required = {
        "sample_id",
        "fold_id",
        "instrument_id",
        "symbol",
        "session_date",
        "as_of",
        "training_cutoff",
    }
    if missing := required.difference(scale.columns):
        raise ValueError(f"TCA learned scale is missing columns: {sorted(missing)}")
    scale_dates = pd.to_datetime(scale["session_date"], errors="coerce")
    cutoff_dates = pd.to_datetime(scale["training_cutoff"], errors="coerce")
    if scale_dates.isna().any() or cutoff_dates.isna().any():
        raise ValueError("TCA preflight found invalid learned date identities.")
    scale = scale.loc[scale_dates.dt.date == session_date].copy()
    _require_integer_key(scale, "as_of", ledger="learned")
    if scale.empty or scale["sample_id"].duplicated().any() or scale["as_of"].duplicated().any():
        raise ValueError("TCA preflight found missing or duplicate learned as-of rows.")
    if not scale["instrument_id"].astype(str).eq(instrument_id).all():
        raise ValueError("TCA preflight found a learned instrument identity mismatch.")
    if scale["symbol"].astype(str).nunique() != 1:
        raise ValueError("TCA preflight found an ambiguous learned symbol identity.")
    if not set(origins).issubset(set(scale["as_of"].astype(int))):
        raise ValueError("TCA preflight found an incomplete learned as-of grid.")
    if not scale["fold_id"].astype(str).eq(fold_id).all():
        raise ValueError("TCA preflight found a learned ledger fold mismatch.")
    if not (pd.to_datetime(scale["training_cutoff"]).dt.date == training_cutoff).all():
        raise ValueError("TCA preflight found a learned ledger cutoff mismatch.")
    sample_ids = scale["sample_id"].astype(str).tolist()
    shape_required = {"case_id", "target_bucket", "conditional_share"}
    if shape_frame is not None and shape_required.difference(shape_frame.columns):
        missing = shape_required.difference(shape_frame.columns)
        raise ValueError(f"TCA learned shape is missing columns: {sorted(missing)}")
    shape = (
        shape_frame.loc[shape_frame["case_id"].astype(str).isin(sample_ids)].copy()
        if shape_frame is not None
        else pd.read_parquet(directory / "shape.parquet", filters=[("case_id", "in", sample_ids)])
    )
    if missing := shape_required.difference(shape.columns):
        raise ValueError(f"TCA learned shape is missing columns: {sorted(missing)}")
    _require_integer_key(shape, "target_bucket", ledger="learned")
    if shape.empty or shape[["case_id", "target_bucket"]].astype(str).duplicated().any():
        raise ValueError("TCA preflight found missing or duplicate learned shape rows.")
    shape_positions = shape.groupby(shape["case_id"].astype(str), sort=False).indices
    for row in scale.itertuples(index=False):
        sample_shape = shape.iloc[shape_positions.get(str(row.sample_id), [])]
        expected = np.arange(int(row.as_of), 26)
        shares = sample_shape["conditional_share"].to_numpy(dtype=float)
        if (
            not np.array_equal(np.sort(sample_shape["target_bucket"].to_numpy(dtype=int)), expected)
            or not np.isfinite(shares).all()
            or (shares < 0).any()
            or not np.isclose(shares.sum(), 1)
        ):
            raise ValueError("TCA preflight found an incomplete learned future-bucket grid.")


def _validate_ewma_case(
    directory: Path,
    *,
    instrument_id: str,
    session_date: date,
    fold_id: str,
    origins: tuple[int, ...],
    scale_frame: pd.DataFrame | None = None,
    available_frame: pd.DataFrame | None = None,
    unavailable_frame: pd.DataFrame | None = None,
) -> None:
    """Verify every exact EWMA minute request is available or explicitly unavailable."""
    del fold_id  # The artifact manifest is the authoritative fold identity.
    scale = (
        scale_frame.copy()
        if scale_frame is not None
        else pd.read_parquet(directory / "scale.parquet")
    )
    required = {"sample_id", "instrument_id", "symbol", "session_date", "as_of"}
    if missing := required.difference(scale.columns):
        raise ValueError(f"TCA EWMA scale is missing columns: {sorted(missing)}")
    scale_dates = pd.to_datetime(scale["session_date"], errors="coerce")
    if scale_dates.isna().any():
        raise ValueError("TCA preflight found invalid EWMA session-date identities.")
    scale = scale.loc[
        (scale["instrument_id"].astype(str) == instrument_id)
        & (scale_dates.dt.date == session_date)
    ]
    _require_integer_key(scale, "as_of", ledger="EWMA")
    if scale.empty or scale["sample_id"].duplicated().any() or scale["as_of"].duplicated().any():
        raise ValueError("TCA preflight found missing or duplicate EWMA as-of rows.")
    if not set(origins).issubset(set(scale["as_of"].astype(int))):
        raise ValueError("TCA preflight found an incomplete EWMA as-of grid.")
    if scale["symbol"].astype(str).nunique() != 1:
        raise ValueError("TCA preflight found an EWMA symbol identity mismatch.")
    opened = pd.Timestamp.combine(session_date, time(9, 30)).tz_localize("America/New_York")
    expected = {
        opened + pd.Timedelta(minutes=15 * origin + offset)
        for origin in origins
        for offset in range(15)
    }
    expected_sample_ids = {
        opened + pd.Timedelta(minutes=15 * int(row.as_of) + offset): str(row.sample_id)
        for row in scale.itertuples(index=False)
        if int(row.as_of) in origins
        for offset in range(15)
    }
    available = (
        available_frame.copy()
        if available_frame is not None
        else pd.read_parquet(
            directory / "minute-forecasts.parquet",
            columns=["sample_id", "generated_at", "end_token"],
        )
    )
    available_required = {"sample_id", "generated_at", "end_token"}
    if missing := available_required.difference(available.columns):
        raise ValueError(f"TCA EWMA minute forecasts are missing columns: {sorted(missing)}")
    available = available.loc[available["end_token"].eq(24)].copy()
    unavailable = (
        unavailable_frame.copy()
        if unavailable_frame is not None
        else pd.read_parquet(directory / "unavailable.parquet")
    )
    unavailable_required = {"sample_id", "generated_at", "end_token", "status", "session_date"}
    if missing := unavailable_required.difference(unavailable.columns):
        raise ValueError(f"TCA EWMA unavailable ledger is missing columns: {sorted(missing)}")
    unavailable_dates = pd.to_datetime(unavailable["session_date"], errors="coerce")
    if unavailable_dates.isna().any():
        raise ValueError("TCA preflight found invalid EWMA unavailable session dates.")
    unavailable = unavailable.loc[
        unavailable["end_token"].eq(24) & unavailable_dates.dt.date.eq(session_date)
    ].copy()
    if not available.empty:
        available_dates = pd.to_datetime(available["generated_at"], errors="coerce")
        if available_dates.isna().any() or available_dates.dt.tz is None:
            raise ValueError("TCA preflight found invalid EWMA generated-at timestamps.")
        available = available.loc[
            available_dates.dt.tz_convert("America/New_York").dt.date.eq(session_date)
        ].copy()
    available_generated = pd.to_datetime(available["generated_at"], errors="coerce")
    unavailable_generated = pd.to_datetime(unavailable["generated_at"], errors="coerce")
    if available_generated.isna().any() or unavailable_generated.isna().any():
        raise ValueError("TCA preflight found invalid EWMA generated-at timestamps.")
    if available_generated.duplicated().any() or unavailable_generated.duplicated().any():
        raise ValueError("TCA preflight found duplicate EWMA minute requests.")
    available_times = set(available_generated)
    unavailable_times = set(unavailable_generated)
    if available_times & unavailable_times or not expected.issubset(
        available_times | unavailable_times
    ):
        raise ValueError("TCA preflight found missing or conflicting EWMA minute requests.")
    sample_ids = set(scale["sample_id"].astype(str))
    if not set(available["sample_id"].astype(str)).issubset(sample_ids) or not set(
        unavailable["sample_id"].astype(str)
    ).issubset(sample_ids):
        raise ValueError("TCA preflight found an EWMA sample identity mismatch.")
    if not unavailable["status"].eq("EWMA_UNAVAILABLE").all():
        raise ValueError("TCA preflight found an invalid EWMA availability status.")
    for frame in (available, unavailable):
        for row in frame.itertuples(index=False):
            generated_at = pd.Timestamp(row.generated_at)
            expected_sample = expected_sample_ids.get(generated_at)
            if expected_sample is not None and str(row.sample_id) != expected_sample:
                raise ValueError("TCA preflight found an EWMA as-of/sample identity mismatch.")


def preflight_tca_ledgers(
    *,
    ledger_records: tuple[tuple[str, int | None, Path, dict[str, Any]], ...],
    ewma_records: Mapping[str, tuple[Path, dict[str, Any]]],
    eligible_cases: Mapping[date, tuple[str, ...]],
    training_cutoff: date,
    tca_config: Mapping[str, Any],
) -> None:
    """Fail closed on derived-artifact gaps before any replay worker starts."""
    if not eligible_cases:
        return
    origins = _tca_as_of_origins(tca_config)
    if not ledger_records:
        raise ValueError("TCA preflight requires learned forecast ledger records.")

    # Keep one bounded in-memory view per instrument while validating all of its
    # eligible dates.  The ledgers are immutable and checksum-bound below, so
    # rereading them for every date would add latency without adding assurance.
    eligible_by_instrument: dict[str, list[date]] = {}
    for session_date, instruments in sorted(eligible_cases.items()):
        for instrument_id in sorted(set(instruments)):
            eligible_by_instrument.setdefault(str(instrument_id), []).append(session_date)

    for _, _, directory, identity in ledger_records:
        verify_artifact(directory, identity=identity, names=LEARNED_LEDGER_FILES)
        for instrument_id, session_dates in eligible_by_instrument.items():
            scale = pd.read_parquet(
                directory / "scale.parquet", filters=[("instrument_id", "==", instrument_id)]
            )
            # Avoid an empty ``in`` predicate (which some Arrow versions reject)
            # while still letting the validator issue the authoritative error.
            if "sample_id" in scale.columns and not scale.empty:
                sample_ids = scale["sample_id"].astype(str).drop_duplicates().tolist()
                shape = pd.read_parquet(
                    directory / "shape.parquet", filters=[("case_id", "in", sample_ids)]
                )
            else:
                shape = pd.DataFrame(columns=["case_id", "target_bucket", "conditional_share"])
            # Index immutable instrument frames once. Preserve row order within
            # each case; all existing per-case contract checks still run below.
            if "session_date" not in scale or "training_cutoff" not in scale:
                raise ValueError("TCA learned scale is missing date identity columns.")
            scale_dates = pd.to_datetime(scale["session_date"], errors="coerce")
            cutoff_dates = pd.to_datetime(scale["training_cutoff"], errors="coerce")
            if scale_dates.isna().any() or cutoff_dates.isna().any():
                raise ValueError("TCA preflight found invalid learned date identities.")
            if "case_id" not in shape:
                raise ValueError("TCA learned shape is missing columns: ['case_id']")
            date_positions = scale.groupby(scale_dates.dt.date, sort=False).indices
            case_positions = shape.groupby(shape["case_id"].astype(str), sort=False).indices
            for session_date in session_dates:
                date_scale = scale.iloc[date_positions.get(session_date, [])]
                positions = [
                    int(position)
                    for sample_id in date_scale["sample_id"].astype(str).drop_duplicates()
                    for position in case_positions.get(sample_id, [])
                ]
                # Original filtering kept physical shape order, not scale order.
                date_shape = shape.iloc[sorted(positions)]
                _validate_learned_case(
                    directory,
                    instrument_id=instrument_id,
                    session_date=session_date,
                    fold_id=str(identity["fold_id"]),
                    training_cutoff=training_cutoff,
                    origins=origins,
                    scale_frame=date_scale,
                    shape_frame=date_shape,
                )

    verified_ewma: set[Path] = set()
    for instrument_id, session_dates in eligible_by_instrument.items():
        if instrument_id not in ewma_records:
            raise ValueError("TCA preflight is missing an EWMA ledger record.")
        directory, identity = ewma_records[instrument_id]
        resolved = directory.resolve()
        if resolved not in verified_ewma:
            verify_artifact(directory, identity=identity, names=EWMA_FILES)
            verified_ewma.add(resolved)
        scale = pd.read_parquet(directory / "scale.parquet")
        available = pd.read_parquet(
            directory / "minute-forecasts.parquet",
            columns=["sample_id", "generated_at", "end_token"],
        )
        unavailable = pd.read_parquet(directory / "unavailable.parquet")
        for session_date in session_dates:
            _validate_ewma_case(
                directory,
                instrument_id=instrument_id,
                session_date=session_date,
                fold_id=str(identity["fold_id"]),
                origins=origins,
                scale_frame=scale,
                available_frame=available,
                unavailable_frame=unavailable,
            )


def run_tca_work(work: TCAWork) -> Path:
    """Run main and sensitivity samples on the same date population without model fitting."""
    verify_artifact(
        work.input_directory,
        identity=work.identity,
        names=("bars.parquet", "adv.parquet", "profiles.parquet", "universe.parquet"),
    )
    identity = {
        **work.identity,
        "schema_version": "paper-tca-date-shard-v3",
        "input_manifest_sha256": file_sha256(work.input_directory / "manifest.json"),
        "training_cutoff": work.training_cutoff.isoformat(),
        "sequence_hash": work.sequence_hash,
        "tca_config": work.tca_config,
        "ledgers": [record[3] for record in work.ledger_records],
        "ewma_ledgers": {key: record[1] for key, record in sorted(work.ewma_records.items())},
    }
    names = ("main.parquet", "sensitivity.parquet")
    if work.output_directory.exists():
        verify_artifact(work.output_directory, identity=identity, names=names)
        return work.output_directory
    bars = pd.read_parquet(work.input_directory / "bars.parquet")
    dates = pd.to_datetime(bars["timestamp"]).dt.tz_convert("America/New_York").dt.date
    if dates.nunique() != 1 or str(dates.iloc[0]) != str(work.identity["session_date"]):
        raise ValueError("TCA shard must contain exactly its declared session date.")
    universe = pd.read_parquet(work.input_directory / "universe.parquet")
    adv = pd.read_parquet(work.input_directory / "adv.parquet")
    profile_frame = pd.read_parquet(work.input_directory / "profiles.parquet")
    profiles = {
        str(row.instrument_id): np.asarray(row.profile, dtype=float)
        for row in profile_frame.itertuples(index=False)
    }
    baselines: dict[tuple[str, date], EWMAForecastLedgerProvider] = {}

    def ewma(instrument: str, day: date) -> EWMAForecastLedgerProvider:
        key = (instrument, day)
        if key not in baselines:
            path, expected = work.ewma_records[instrument]
            symbols = bars.loc[bars["instrument_id"] == instrument, "symbol"].unique()
            if len(symbols) != 1:
                raise ValueError("TCA instrument/session has ambiguous symbol identity.")
            baselines[key] = EWMAForecastLedgerProvider(
                path,
                expected_identity=expected,
                symbol=str(symbols[0]),
                session_date=day,
                verified_artifact=_verified_ledger(
                    path,
                    expected,
                    (*EWMA_FILES,),
                ),
            )
        return baselines[key]

    providers: dict[str, Any] = {"ewma": ewma}
    for method, seed, path, expected in work.ledger_records:
        artifact = _verified_ledger(
            path, expected, ("scale.parquet", "shape.parquet", "metrics.parquet")
        )
        date_slice = ForecastLedgerDate.read(
            artifact, dates.iloc[0], tuple(sorted(bars["instrument_id"].unique()))
        )

        def learned(
            instrument: str,
            day: date,
            artifact: VerifiedArtifact = artifact,
            date_slice: ForecastLedgerDate = date_slice,
        ) -> Any:
            return PaperForecastLedgerProvider(
                artifact.directory,
                expected_identity=artifact.identity,
                instrument_id=instrument,
                session_date=day,
                within_token_profile=profiles[instrument],
                training_cutoff=work.training_cutoff,
                sequence_hash=work.sequence_hash,
                verified_artifact=artifact,
                date_slice=date_slice,
            )

        name = {"raw": "lightgbm_raw", "untrained_neural": "raw_untrained_neural"}.get(
            method, f"raw_{method}_jepa_seed_{seed}"
        )
        if name in providers:
            raise ValueError("TCA shard duplicates a forecasting method.")
        providers[name] = learned
    config = work.tca_config
    options: dict[str, Any] = dict(
        required_methods=tuple(providers),
        start=str(config["window"][0]),
        end=str(config["window"][1]),
        planned_participation=float(config["planned_participation_rate"]),
        hard_participation=float(config["hard_participation_rate"]),
        risk_aversion=float(config["risk_aversion"]),
        tracking_penalty=float(config["tracking_penalty"]),
        half_spread_arrival_fraction=float(config["half_spread_arrival_fraction"]),
        temporary_impact_arrival_fraction=float(config["temporary_impact_at_full_participation"]),
    )
    main = run_historical_tca(
        bars,
        universe,
        adv,
        providers,
        liquidity_size=int(config["universe_size"]),
        order_fraction=float(config["quantity_fraction_adv20"]),
        **options,
    )
    sensitivity = pd.concat(
        [
            run_historical_tca(
                bars,
                universe,
                adv,
                providers,
                liquidity_size=int(config["sensitivity_universe_size"]),
                order_fraction=float(fraction),
                **options,
            )
            for fraction in config["sensitivity_quantity_fraction_adv20"]
        ],
        ignore_index=True,
    )
    for frame in (main, sensitivity):
        frame.insert(0, "fold_id", work.identity["fold_id"])
        frame.sort_values(
            ["date", "instrument_id", "method", "order_fraction_adv20"],
            kind="stable",
            inplace=True,
            ignore_index=True,
        )
    publish_frames(
        work.output_directory,
        identity=identity,
        frames={"main.parquet": main, "sensitivity.parquet": sensitivity},
    )
    return work.output_directory


def run_tca_workers(work: list[TCAWork], *, workers: int | None = None) -> list[Path]:
    """Spawn bounded independent date tasks with one native numeric thread each."""
    count = int(os.environ.get("EXECSIM_EVALUATION_WORKERS", "16")) if workers is None else workers
    if count < 1 or len({item.output_directory for item in work}) != len(work):
        raise ValueError("TCA workers require positive concurrency and distinct output paths.")
    from execsim.ml.representations.probe_runtime import effective_cpu_capacity

    count = min(count, max(1, int(effective_cpu_capacity())))
    if count == 1:
        return [run_tca_work(item) for item in work]
    previous = {key: os.environ.get(key) for key in NATIVE_THREAD_VARIABLES}
    try:
        for key in NATIVE_THREAD_VARIABLES:
            os.environ[key] = "1"
        with ProcessPoolExecutor(
            max_workers=count,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=configure_evaluation_worker,
        ) as pool:
            return list(pool.map(run_tca_work, work, chunksize=1))
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
