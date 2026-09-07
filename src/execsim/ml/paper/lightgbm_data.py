"""Construct fold-safe scale and long-form shape LightGBM rows from sequence stores."""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from execsim.data.paper.manifests import read_json
from execsim.ml.paper.features import append_embedding, build_raw_feature_frame
from execsim.ml.sequences.dataset import extract_window
from execsim.ml.sequences.manifests import read_sequence_record
from execsim.ml.sequences.schemas import SequenceRecord, SequenceSample
from execsim.ml.sequences.streaming import _sample_from_row


def build_lightgbm_frames(
    sequence_manifest_path: Path,
    *,
    partition: str,
    liquidity_groups: dict[str, int],
    embedding_path: Path | None = None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Build one scale row/case and one shape row/valid future token."""
    manifest = read_json(sequence_manifest_path)
    root = sequence_manifest_path.parent
    sequence_paths = {
        path.stem: path
        for value in manifest["sequence_files"]
        if f"sessions/{partition}/" in str(value).replace("\\", "/")
        for path in (root / str(value),)
    }
    index_paths = [
        root / str(value)
        for value in manifest["index_files"]
        if f"indexes/{partition}/" in str(value).replace("\\", "/")
    ]
    embedding_by_sample: dict[str, np.ndarray] = {}
    if embedding_path is not None:
        embeddings = pd.read_parquet(embedding_path)
        if embeddings["sample_id"].duplicated().any():
            raise ValueError("Embedding corpus duplicates sample identity.")
        embedding_by_sample = {
            str(row.sample_id): np.asarray(row.embedding, dtype=float)
            for row in embeddings.itertuples(index=False)
        }
    scale_frames: list[pd.DataFrame] = []
    scale_chunks: list[pd.DataFrame] = []
    scale_targets = []
    shape_frames: list[pd.DataFrame] = []
    shape_chunks: list[pd.DataFrame] = []
    shape_targets = []
    samples = [
        _sample_from_row(row)
        for index_path in sorted(index_paths)
        for row in pd.read_parquet(index_path).itertuples(index=False)
    ]
    shape_probabilities = (
        _shape_origin_probabilities(samples)
        if partition == "train"
        else {sample.sample_id: 1.0 for sample in samples}
    )

    @lru_cache(maxsize=512)
    def cached_record(session_id: str) -> SequenceRecord:
        return read_sequence_record(sequence_paths[session_id])

    def flush_frames(*, force: bool = False) -> None:
        if scale_frames and (force or len(scale_frames) >= 1_024):
            scale_chunks.append(pd.concat(scale_frames, ignore_index=True))
            scale_frames.clear()
        if shape_frames and (force or len(shape_frames) >= 1_024):
            shape_chunks.append(pd.concat(shape_frames, ignore_index=True))
            shape_frames.clear()

    for sample in samples:
        record = cached_record(sample.session_id)
        window = extract_window(record, sample)
        timestamp = pd.Timestamp(sample.as_of_ns, tz="UTC").tz_convert("America/New_York")
        metadata = pd.DataFrame(
            {
                "as_of_bucket": [sample.as_of_token],
                "target_bucket": [-1],
                "horizon_offset": [-1],
                "minutes_remaining": [(26 - sample.as_of_token) * 15],
                "weekday": [timestamp.weekday()],
                "month": [timestamp.month],
                "is_month_end": [timestamp.is_month_end],
                "is_quarter_end": [timestamp.is_quarter_end],
                "symbol": [record.symbol],
                "liquidity_group": [liquidity_groups[record.instrument_id]],
            }
        )
        raw = build_raw_feature_frame(
            window["context"][None, ...], window["context_mask"][None, ...], metadata
        ).drop(columns=["target_bucket", "horizon_offset"])
        raw.insert(0, "sample_id", sample.sample_id)
        raw.insert(1, "fold_id", sample.fold_id)
        raw.insert(2, "instrument_id", record.instrument_id)
        raw.insert(3, "session_date", record.session_date)
        raw.insert(4, "as_of", sample.as_of_token)
        raw.insert(5, "training_cutoff", sample.training_cutoff)
        raw.insert(6, "market_information_as_of", sample.market_information_as_of)
        raw.insert(7, "feature_history_end", sample.feature_history_end)
        if embedding_path is not None:
            try:
                raw = append_embedding(raw, embedding_by_sample[sample.sample_id][None, :])
            except KeyError as exc:
                raise ValueError(f"Missing embedding for sample {sample.sample_id}") from exc
        future = record.raw_volume[sample.as_of_token :]
        total = float(future.sum())
        baseline_remaining = float(record.causal_baseline_volume[sample.as_of_token :].sum())
        if total <= 0:
            continue
        raw["baseline_remaining_volume"] = baseline_remaining
        scale_frames.append(raw)
        scale_targets.append(total)
        if sample.sample_id not in shape_probabilities:
            continue
        repeated = raw.loc[raw.index.repeat(len(future))].reset_index(drop=True)
        repeated["target_bucket"] = np.arange(sample.as_of_token, 26)
        repeated["horizon_offset"] = np.arange(1, len(future) + 1)
        repeated["target_valid"] = True
        repeated["case_id"] = sample.sample_id
        baseline_future = record.causal_baseline_volume[sample.as_of_token :]
        baseline_total = float(baseline_future.sum())
        if baseline_total <= 0:
            raise ValueError("Causal baseline future volume must be positive for shape rows.")
        repeated["baseline_conditional_share"] = baseline_future / baseline_total
        inclusion_probability = shape_probabilities[sample.sample_id]
        repeated["shape_origin_inclusion_probability"] = inclusion_probability
        repeated["shape_case_weight"] = 1.0 / inclusion_probability
        repeated["shape_row_weight"] = 1.0 / inclusion_probability / len(future)
        repeated["sample_weight"] = repeated["shape_row_weight"]
        shape_frames.append(repeated)
        shape_targets.extend((future / total).tolist())
        flush_frames()
    flush_frames(force=True)
    if not scale_chunks or not shape_chunks:
        raise ValueError(f"No valid LightGBM rows were produced for {partition}.")
    return (
        pd.concat(scale_chunks, ignore_index=True),
        np.asarray(scale_targets, dtype=float),
        pd.concat(shape_chunks, ignore_index=True),
        np.asarray(shape_targets, dtype=float),
    )


def _shape_origin_probabilities(samples: list[SequenceSample]) -> dict[str, float]:
    """Select one deterministic TRAIN origin per session and predeclared time band."""
    bands = ((4, 9), (10, 15), (16, 20), (21, 25))
    grouped: dict[tuple[str, int], list[SequenceSample]] = {}
    for sample in samples:
        for band_index, (start, end) in enumerate(bands):
            if start <= sample.as_of_token <= end:
                grouped.setdefault((sample.session_id, band_index), []).append(sample)
                break
    selected: dict[str, float] = {}
    for candidates in grouped.values():
        ordered = sorted(
            candidates,
            key=lambda item: hashlib.sha256(item.sample_id.encode("utf-8")).hexdigest(),
        )
        selected[ordered[0].sample_id] = 1.0 / len(ordered)
    return selected


def build_historical_baseline_regime_frame(
    sequence_manifest_path: Path, *, partition: str
) -> pd.DataFrame:
    """Build the identical TRAIN/held-out unusual-session statistic without wide expansion."""
    manifest = read_json(sequence_manifest_path)
    root = sequence_manifest_path.parent
    sequence_paths = {
        path.stem: path
        for value in manifest["sequence_files"]
        if f"sessions/{partition}/" in str(value).replace("\\", "/")
        for path in (root / str(value),)
    }
    index_paths = [
        root / str(value)
        for value in manifest["index_files"]
        if f"indexes/{partition}/" in str(value).replace("\\", "/")
    ]
    records = {name: read_sequence_record(path) for name, path in sequence_paths.items()}
    rows = []
    for index_path in sorted(index_paths):
        for row in pd.read_parquet(index_path).itertuples(index=False):
            sample = _sample_from_row(row)
            record = records[sample.session_id]
            actual = record.raw_volume[sample.as_of_token :].astype(float)
            baseline = record.causal_baseline_volume[sample.as_of_token :].astype(float)
            if actual.sum() <= 0 or baseline.sum() <= 0:
                continue
            actual_share = actual / actual.sum()
            baseline_share = baseline / baseline.sum()
            curve_error = float(
                np.mean(np.abs(np.cumsum(actual_share) - np.cumsum(baseline_share)))
            )
            current = record.features[sample.as_of_token - 1]
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "instrument_id": record.instrument_id,
                    "session_date": record.session_date,
                    "as_of_token": sample.as_of_token,
                    "volume_surprise": float(current[4]),
                    "realized_volatility": float(current[3]),
                    "historical_baseline_curve_error": curve_error,
                }
            )
    result = pd.DataFrame(rows)
    if result.empty or result["sample_id"].duplicated().any():
        raise ValueError(f"Regime frame is empty or duplicates sample IDs for {partition}.")
    return result
