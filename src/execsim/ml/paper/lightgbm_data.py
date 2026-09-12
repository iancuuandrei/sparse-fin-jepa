"""Construct fold-safe scale and long-form shape LightGBM rows from sequence stores."""

from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd

from execsim.data.paper.manifests import file_sha256, read_json, write_json_atomic
from execsim.ml.paper.features import append_embedding, build_raw_feature_frame
from execsim.ml.sequences.dataset import extract_window
from execsim.ml.sequences.manifests import read_sequence_record
from execsim.ml.sequences.schemas import SequenceRecord, SequenceSample
from execsim.ml.sequences.streaming import _load_cached_index_frame, _sample_from_row


@dataclass(frozen=True, slots=True)
class LightGBMFrames:
    """One immutable common scale/shape frame set for a fold partition."""

    scale: pd.DataFrame
    scale_target: np.ndarray
    shape: pd.DataFrame
    shape_target: np.ndarray

    def as_tuple(self) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
        """Expose the established adapter boundary without copying frame blocks."""
        return self.scale, self.scale_target, self.shape, self.shape_target


def cached_lightgbm_base_frames(
    sequence_manifest_path: Path,
    *,
    partition: str,
    liquidity_groups: dict[str, int],
    cache_directory: Path,
    source_commit: str,
    config_hash: str,
) -> LightGBMFrames:
    """Publish and reuse checksummed TRAIN/VALIDATION cores without rereading sessions."""
    if partition not in {"train", "validation"}:
        raise ValueError("Training base cache accepts TRAIN/VALIDATION only.")
    identity = {
        "schema": "lightgbm-base-cache-v1",
        "sequence_manifest_sha256": file_sha256(sequence_manifest_path),
        "partition": partition,
        "liquidity_groups": liquidity_groups,
        "source_commit": source_commit,
        "config_hash": config_hash,
    }
    if cache_directory.exists():
        receipt = read_json(cache_directory / "manifest.json")
        if receipt["identity"] != identity:
            raise ValueError("LightGBM base cache identity mismatch.")
        for name in ("scale.parquet", "shape.parquet", "scale.npy", "shape.npy"):
            if file_sha256(cache_directory / name) != receipt["files"][name]:
                raise ValueError(f"LightGBM base cache checksum mismatch: {name}")
        return LightGBMFrames(
            pd.read_parquet(cache_directory / "scale.parquet"),
            np.load(cache_directory / "scale.npy", allow_pickle=False),
            pd.read_parquet(cache_directory / "shape.parquet"),
            np.load(cache_directory / "shape.npy", allow_pickle=False),
        )
    base = build_lightgbm_base_frames(
        sequence_manifest_path, partition=partition, liquidity_groups=liquidity_groups
    )
    cache_directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".base-", dir=cache_directory.parent) as temporary:
        staging = Path(temporary) / "complete"
        staging.mkdir()
        base.scale.to_parquet(staging / "scale.parquet", index=False)
        base.shape.to_parquet(staging / "shape.parquet", index=False)
        np.save(staging / "scale.npy", base.scale_target, allow_pickle=False)
        np.save(staging / "shape.npy", base.shape_target, allow_pickle=False)
        write_json_atomic(
            staging / "manifest.json",
            {
                "identity": identity,
                "files": {
                    name: file_sha256(staging / name)
                    for name in ("scale.parquet", "shape.parquet", "scale.npy", "shape.npy")
                },
            },
        )
        os.replace(staging, cache_directory)
    return base


def build_lightgbm_frames(
    sequence_manifest_path: Path,
    *,
    partition: str,
    liquidity_groups: dict[str, int],
    embedding_path: Path | None = None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray]:
    """Compatibility builder for one complete raw or hybrid frame set."""
    base = build_lightgbm_base_frames(
        sequence_manifest_path,
        partition=partition,
        liquidity_groups=liquidity_groups,
    )
    if embedding_path is not None:
        base = attach_lightgbm_embeddings(base, embedding_path=embedding_path)
    return base.as_tuple()


def build_lightgbm_base_frames(
    sequence_manifest_path: Path,
    *,
    partition: str,
    liquidity_groups: dict[str, int],
) -> LightGBMFrames:
    """Build common raw rows, targets, weights, and identities exactly once."""
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
    scale_frames: list[pd.DataFrame] = []
    scale_chunks: list[pd.DataFrame] = []
    scale_targets = []
    shape_frames: list[pd.DataFrame] = []
    shape_chunks: list[pd.DataFrame] = []
    shape_targets = []
    samples = [
        _sample_from_row(row)
        for row in _load_cached_index_frame(
            sequence_manifest_path,
            partition=partition,
            index_paths=tuple(sorted(index_paths)),
        ).itertuples(index=False)
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

    def raw_rows():
        # Group only consecutive rows: no sorting or population changes. A session
        # has at most 22 origins, so feature materialization stays bounded.
        for session_id, session_samples in groupby(samples, key=lambda sample: sample.session_id):
            group = list(session_samples)
            record = cached_record(session_id)
            windows = [extract_window(record, sample) for sample in group]
            timestamps = [
                pd.Timestamp(sample.as_of_ns, tz="UTC").tz_convert("America/New_York")
                for sample in group
            ]
            metadata = pd.DataFrame(
                {
                    "as_of_bucket": [sample.as_of_token for sample in group],
                    "target_bucket": -1,
                    "horizon_offset": -1,
                    "minutes_remaining": [(26 - sample.as_of_token) * 15 for sample in group],
                    "weekday": [timestamp.weekday() for timestamp in timestamps],
                    "month": [timestamp.month for timestamp in timestamps],
                    "is_month_end": [timestamp.is_month_end for timestamp in timestamps],
                    "is_quarter_end": [timestamp.is_quarter_end for timestamp in timestamps],
                    "symbol": record.symbol,
                    "liquidity_group": liquidity_groups[record.instrument_id],
                }
            )
            frame = build_raw_feature_frame(
                np.stack([window["context"] for window in windows]),
                np.stack([window["context_mask"] for window in windows]),
                metadata,
            ).drop(columns=["target_bucket", "horizon_offset"])
            for index, sample in enumerate(group):
                yield sample, record, frame.iloc[[index]].reset_index(drop=True)

    for sample, record, raw in raw_rows():
        raw.insert(0, "sample_id", sample.sample_id)
        raw.insert(1, "fold_id", sample.fold_id)
        raw.insert(2, "instrument_id", record.instrument_id)
        raw.insert(3, "session_date", record.session_date)
        raw.insert(4, "as_of", sample.as_of_token)
        raw.insert(5, "training_cutoff", sample.training_cutoff)
        raw.insert(6, "market_information_as_of", sample.market_information_as_of)
        raw.insert(7, "feature_history_end", sample.feature_history_end)
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
    return LightGBMFrames(
        scale=pd.concat(scale_chunks, ignore_index=True),
        scale_target=np.asarray(scale_targets, dtype=float),
        shape=pd.concat(shape_chunks, ignore_index=True),
        shape_target=np.asarray(shape_targets, dtype=float),
    )


def attach_lightgbm_embeddings(base: LightGBMFrames, *, embedding_path: Path) -> LightGBMFrames:
    """Attach one frozen representation by exact sample identity for one coordinate."""
    embeddings = pd.read_parquet(embedding_path, columns=["sample_id", "embedding"])
    return attach_lightgbm_embedding_frame(base, embeddings=embeddings)


def attach_lightgbm_embedding_frame(
    base: LightGBMFrames, *, embeddings: pd.DataFrame
) -> LightGBMFrames:
    """Attach a verified compact embedding slice without rereading its source file."""
    embedding_ids = pd.Index(embeddings["sample_id"].astype(str))
    if embedding_ids.has_duplicates:
        raise ValueError("Embedding corpus duplicates sample identity.")
    values = np.stack(
        [np.asarray(value, dtype=np.float32) for value in embeddings["embedding"]], axis=0
    )
    if values.shape != (len(embeddings), 644) or not np.isfinite(values).all():
        raise ValueError("Embedding corpus must contain finite 644-value rows.")

    scale_ids = pd.Index(base.scale["sample_id"].astype(str))
    if scale_ids.has_duplicates:
        raise ValueError("LightGBM scale base duplicates sample identity.")
    scale_positions = embedding_ids.get_indexer(scale_ids)
    if np.any(scale_positions < 0):
        missing = scale_ids[scale_positions < 0][0]
        raise ValueError(f"Missing embedding for sample {missing}")
    shape_positions = scale_ids.get_indexer(base.shape["sample_id"].astype(str))
    if np.any(shape_positions < 0):
        raise ValueError("LightGBM shape rows lack matching scale sample identity.")

    scale = append_embedding(base.scale, values[scale_positions])
    shape = append_embedding(base.shape, values[scale_positions][shape_positions])
    # Preserve the original feature order, including LightGBM's column sampling order.
    for frame, raw in ((scale, base.scale), (shape, base.shape)):
        columns = list(raw.columns)
        offset = columns.index("baseline_remaining_volume")
        columns[offset:offset] = [f"embedding_{index:03d}" for index in range(644)]
        if frame is scale:
            scale = frame.loc[:, columns]
        else:
            shape = frame.loc[:, columns]
    return LightGBMFrames(scale, base.scale_target, shape, base.shape_target)


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

    @lru_cache(maxsize=32)
    def record_for(session_id: str) -> SequenceRecord:
        return read_sequence_record(sequence_paths[session_id])

    rows = []
    index_frame = _load_cached_index_frame(
        sequence_manifest_path, partition=partition, index_paths=tuple(sorted(index_paths))
    )
    for row in index_frame.itertuples(index=False):
        sample = _sample_from_row(row)
        record = record_for(sample.session_id)
        actual = record.raw_volume[sample.as_of_token :].astype(float)
        baseline = record.causal_baseline_volume[sample.as_of_token :].astype(float)
        if actual.sum() <= 0 or baseline.sum() <= 0:
            continue
        actual_share = actual / actual.sum()
        baseline_share = baseline / baseline.sum()
        curve_error = float(np.mean(np.abs(np.cumsum(actual_share) - np.cumsum(baseline_share))))
        current = record.features[sample.as_of_token - 1]
        rows.append(
            {
                "sample_id": sample.sample_id,
                "session_id": sample.session_id,
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
