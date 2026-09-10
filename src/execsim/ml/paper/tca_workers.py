"""Independent date-level TCA tasks preserving the frozen matched replay."""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from execsim.data.paper.manifests import file_sha256
from execsim.ml.paper.evaluation_artifacts import VerifiedArtifact, publish_frames, verify_artifact
from execsim.ml.paper.evaluation_workers import NATIVE_THREAD_VARIABLES
from execsim.ml.paper.forecast_ledger import EWMAForecastLedgerProvider, PaperForecastLedgerProvider
from execsim.ml.paper.tca import run_historical_tca

_VERIFIED_LEDGERS: dict[Path, VerifiedArtifact] = {}


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


def run_tca_work(work: TCAWork) -> Path:
    """Run main and sensitivity samples on the same date population without model fitting."""
    verify_artifact(
        work.input_directory,
        identity=work.identity,
        names=("bars.parquet", "adv.parquet", "profiles.parquet", "universe.parquet"),
    )
    identity = {
        **work.identity,
        "schema_version": "paper-tca-date-shard-v2",
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
                    (
                        "scale.parquet",
                        "shape.parquet",
                        "metrics.parquet",
                        "minute-forecasts.parquet",
                    ),
                ),
            )
        return baselines[key]

    providers: dict[str, Any] = {"ewma": ewma}
    for method, seed, path, expected in work.ledger_records:
        artifact = _verified_ledger(
            path, expected, ("scale.parquet", "shape.parquet", "metrics.parquet")
        )

        def learned(instrument: str, day: date, artifact: VerifiedArtifact = artifact) -> Any:
            return PaperForecastLedgerProvider(
                artifact.directory,
                expected_identity=artifact.identity,
                instrument_id=instrument,
                session_date=day,
                within_token_profile=profiles[instrument],
                training_cutoff=work.training_cutoff,
                sequence_hash=work.sequence_hash,
                verified_artifact=artifact,
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
    if count == 1:
        return [run_tca_work(item) for item in work]
    previous = {key: os.environ.get(key) for key in NATIVE_THREAD_VARIABLES}
    try:
        for key in NATIVE_THREAD_VARIABLES:
            os.environ[key] = "1"
        with ProcessPoolExecutor(
            max_workers=count, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            return list(pool.map(run_tca_work, work, chunksize=1))
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
