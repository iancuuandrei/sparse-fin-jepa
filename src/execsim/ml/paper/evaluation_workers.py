"""Bounded spawn-safe CPU evaluation with immutable file inputs and atomic outputs."""

from __future__ import annotations

import hashlib
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from execsim.data.paper.manifests import file_sha256, read_json, stable_hash, write_json_atomic
from execsim.forecasting import HistoricalProfileForecaster
from execsim.ml.paper.evaluation_artifacts import (
    forecast_metric_frame,
    publish_frames,
    verify_artifact,
)
from execsim.ml.paper.lightgbm_data import LightGBMFrames

NATIVE_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)


@dataclass(frozen=True, slots=True)
class EWMAWork:
    """Small process message; market and base rows are read from bounded files."""

    base_directory: Path
    market_path: Path
    output_directory: Path
    instrument_id: str
    identity: dict[str, Any]


def ewma_ledger_identity(work: EWMAWork) -> dict[str, Any]:
    """Derive compatibility from authoritative inputs, never the output receipt itself."""
    return {
        **work.identity,
        "instrument_id": work.instrument_id,
        "market_sha256": file_sha256(work.market_path),
        "base_manifest_sha256": file_sha256(work.base_directory / "manifest.json"),
        "schema_version": "paper-ewma-ledger-v3",
    }


def run_ewma_work(work: EWMAWork) -> Path:
    """Persist one instrument's forecast and exact-horizon TCA baseline evidence."""
    names = ("scale.parquet", "shape.parquet", "metrics.parquet", "minute-forecasts.parquet")
    identity = ewma_ledger_identity(work)
    if work.output_directory.exists():
        verify_artifact(work.output_directory, identity=identity, names=names)
        return work.output_directory
    filters = [("instrument_id", "==", work.instrument_id)]
    scale = pd.read_parquet(work.base_directory / "scale-base.parquet", filters=filters)
    shape = pd.read_parquet(work.base_directory / "shape-base.parquet", filters=filters)
    base = LightGBMFrames(
        scale,
        scale.pop("__evaluation_target").to_numpy(),
        shape,
        shape.pop("__evaluation_target").to_numpy(),
    )
    bars = pd.read_parquet(work.market_path)
    provider = HistoricalProfileForecaster(
        bars, estimator="ewma", lookback_sessions=20, data_manifest_hash=identity["market_sha256"]
    )
    totals = np.empty(len(scale), dtype=float)
    token_shapes = []
    minute_ids, ends, generated_times, volume_rows, cutoffs, warning_rows = [], [], [], [], [], []
    normalized_rows, remaining_totals = [], []
    for index, (sample_id, symbol, day_value, origin) in enumerate(
        scale.loc[:, ["sample_id", "symbol", "session_date", "as_of"]].itertuples(
            index=False, name=None
        )
    ):
        day = pd.Timestamp(day_value).date()
        opened = pd.Timestamp.combine(day, pd.Timestamp("09:30").time()).tz_localize(
            "America/New_York"
        )
        generated = opened + pd.Timedelta(minutes=15 * int(origin))
        # EWMA depends on the exact requested window. TCA ends at 15:30, so a
        # truncated full-session estimate is not generally the same estimator.
        requests = [(26, 0)]
        if int(origin) < 24:
            requests.extend((24, offset) for offset in range(15))
        for end_token, offset in requests:
            request_time = generated + pd.Timedelta(minutes=offset)
            minutes = tuple(
                pd.date_range(
                    request_time, opened + pd.Timedelta(minutes=15 * end_token - 1), freq="min"
                )
            )
            forecast = provider.forecast(
                symbol=str(symbol),
                session_date=day,
                generated_at=request_time,
                bucket_timestamps=minutes,
            )
            minute_ids.append(sample_id)
            ends.append(end_token)
            generated_times.append(request_time)
            volume_rows.append(forecast.expected_volumes)
            cutoffs.append(forecast.training_data_cutoff)
            warning_rows.append(forecast.warnings)
            normalized_rows.append(forecast.normalized_shares)
            remaining_totals.append(forecast.expected_remaining_volume)
            if end_token == 26:
                totals[index] = forecast.expected_remaining_volume
                token_volumes = np.asarray(forecast.expected_volumes).reshape(-1, 15).sum(axis=1)
                token_shapes.append(token_volumes / token_volumes.sum())
    predicted = shape.loc[:, ["case_id", "target_bucket"]].copy()
    expected_ids = np.repeat(scale["sample_id"].to_numpy(), 26 - scale["as_of"].to_numpy(dtype=int))
    if not np.array_equal(predicted["case_id"].to_numpy(), expected_ids):
        raise ValueError("EWMA base shape order differs from scale identity order.")
    predicted["conditional_share"] = np.concatenate(token_shapes)
    metrics = forecast_metric_frame(
        base, totals, predicted, fold_id=str(identity["fold_id"]), method="ewma", seed=None
    )
    minute_ledger = pd.DataFrame(
        {
            "sample_id": minute_ids,
            "end_token": ends,
            "generated_at": generated_times,
            "expected_volumes": volume_rows,
            "training_data_cutoff": cutoffs,
            "warnings": warning_rows,
            "normalized_shares": normalized_rows,
            "expected_remaining_volume": remaining_totals,
        }
    )
    published_scale = scale.loc[
        :, ["sample_id", "instrument_id", "symbol", "session_date", "as_of"]
    ].copy()
    published_scale["predicted_remaining_volume"] = totals
    publish_frames(
        work.output_directory,
        identity=identity,
        frames=dict(zip(names, (published_scale, predicted, metrics, minute_ledger), strict=True)),
    )
    return work.output_directory


def run_ewma_workers(work: list[EWMAWork], *, workers: int | None = None) -> list[Path]:
    """Run disjoint instrument tasks with one native thread per spawned process."""
    count = int(os.environ.get("EXECSIM_EVALUATION_WORKERS", "16")) if workers is None else workers
    if count < 1:
        raise ValueError("Evaluation worker count must be positive.")
    if len({item.output_directory for item in work}) != len(work):
        raise ValueError("Evaluation workers must own distinct output directories.")
    if count == 1:
        return [run_ewma_work(item) for item in work]
    previous = {key: os.environ.get(key) for key in NATIVE_THREAD_VARIABLES}
    try:
        for key in NATIVE_THREAD_VARIABLES:
            os.environ[key] = "1"
        with ProcessPoolExecutor(
            max_workers=count, mp_context=multiprocessing.get_context("spawn")
        ) as pool:
            return list(pool.map(run_ewma_work, work, chunksize=1))
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def instrument_key(instrument_id: str) -> str:
    """Return a safe deterministic path component without provider-symbol assumptions."""
    return hashlib.sha256(instrument_id.encode()).hexdigest()


def compact_profile_corpus(
    source: Path,
    directory: Path,
    *,
    identity: dict[str, Any],
    include_market_bars: bool = False,
    selected_instruments: tuple[str, ...] | None = None,
) -> dict[str, Path]:
    """Scan raw files once into bounded per-instrument profile inputs for all folds."""
    sources = (
        [source]
        if source.is_file()
        else sorted(source.rglob("*.parquet")) + sorted(source.rglob("*.response"))
    )
    if not sources:
        raise ValueError("Profile corpus has no Parquet inputs.")
    inventory = {
        str(path.relative_to(source) if source.is_dir() else path.name): file_sha256(path)
        for path in sources
    }
    bound = {
        **identity,
        "schema_version": "paper-market-corpus-v2"
        if include_market_bars
        else "paper-profile-corpus-v2",
        "source_inventory_sha256": stable_hash(inventory),
    }
    if include_market_bars or selected_instruments is not None:
        bound["selected_instruments"] = (
            sorted(selected_instruments) if selected_instruments is not None else None
        )
    columns = ["instrument_id", "symbol", "timestamp", "volume"]
    if include_market_bars:
        columns.extend(["open", "high", "low", "close", "trade_count", "vwap"])
    if directory.exists():
        receipt = read_json(directory / "manifest.json")
        verify_artifact(directory, identity=bound, names=tuple(receipt["files"]))
        return {key: directory / name for key, name in receipt["instruments"].items()}
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".profile-", dir=directory.parent) as temporary:
        staging = Path(temporary) / "complete"
        staging.mkdir()
        writers: dict[str, pq.ParquetWriter] = {}
        instruments: dict[str, str] = {}
        try:
            for path in sources:
                with pq.ParquetFile(path) as parquet:
                    for batch in parquet.iter_batches(
                        batch_size=65_536,
                        columns=columns,
                    ):
                        frame = batch.to_pandas()
                        if frame["instrument_id"].isna().any():
                            raise ValueError("Profile corpus has null instrument identity.")
                        if selected_instruments is not None:
                            frame = frame.loc[
                                frame["instrument_id"].astype(str).isin(selected_instruments)
                            ]
                        for instrument, group in frame.groupby(
                            "instrument_id", sort=True, observed=True
                        ):
                            key = str(instrument)
                            name = instrument_key(key) + ".parquet"
                            instruments[key] = name
                            # Canonical types avoid dictionary/schema drift across monthly files.
                            table = pa.Table.from_pandas(
                                pd.DataFrame(
                                    {
                                        "symbol": group["symbol"].astype(str),
                                        "timestamp": pd.to_datetime(
                                            group["timestamp"]
                                        ).dt.tz_convert("America/New_York"),
                                        "volume": group["volume"].astype(float),
                                    }
                                ),
                                preserve_index=False,
                            )
                            if include_market_bars:
                                full = group.copy()
                                full["instrument_id"] = full["instrument_id"].astype(str)
                                full["symbol"] = full["symbol"].astype(str)
                                full["timestamp"] = pd.to_datetime(full["timestamp"]).dt.tz_convert(
                                    "America/New_York"
                                )
                                table = pa.Table.from_pandas(full, preserve_index=False)
                            if name not in writers:
                                writers[name] = pq.ParquetWriter(staging / name, table.schema)
                            writers[name].write_table(table, row_group_size=65_536)
        finally:
            for writer in writers.values():
                writer.close()
        files = {}
        for name in sorted(writers):
            path = staging / name
            with path.open("r+b") as stream:
                os.fsync(stream.fileno())
            with pq.ParquetFile(path) as parquet:
                files[name] = {
                    "sha256": file_sha256(path),
                    "rows": parquet.metadata.num_rows,
                    "schema": str(parquet.schema_arrow),
                }
        write_json_atomic(
            staging / "manifest.json",
            {
                "identity": bound,
                "files": files,
                "instruments": instruments,
                "source_inventory": inventory,
            },
        )
        with (staging / "manifest.json").open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(staging, directory)
    return {key: directory / name for key, name in instruments.items()}
