"""Atomic, identity-bound derived artifacts for the resealed evaluator."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from execsim.data.paper.manifests import file_sha256, read_json, write_json_atomic
from execsim.ml.paper.lightgbm_data import (
    LightGBMFrames,
    attach_lightgbm_embedding_frame,
    build_lightgbm_base_frames,
)

BASE_SCHEMA = "paper-evaluation-base-v2"
ROW_GROUP_SIZE = 65_536
_TARGET = "__evaluation_target"


def prediction_batches(
    base: LightGBMFrames,
    *,
    embedding_path: Path | None = None,
    untrained_control: bool = False,
    batch_samples: int = 2048,
) -> Iterator[LightGBMFrames]:
    """Materialize one wide prediction batch; never split a conditional horizon."""
    if batch_samples < 1 or (untrained_control and embedding_path is not None):
        raise ValueError("Invalid prediction batch size or representation selection.")
    scale_ids = pd.Index(base.scale["sample_id"].astype(str))
    if scale_ids.has_duplicates:
        raise ValueError("Prediction scale sample identities must be unique.")
    positions = scale_ids.get_indexer(base.shape["sample_id"].astype(str))
    if np.any(positions < 0) or np.any(np.diff(positions) < 0):
        raise ValueError("Prediction shape rows must follow canonical scale sample order.")
    embeddings = None
    if embedding_path is not None:
        embeddings = pd.read_parquet(embedding_path, columns=["sample_id", "embedding"])
        embeddings.index = pd.Index(embeddings["sample_id"].astype(str))
        if embeddings.index.has_duplicates or not scale_ids.isin(embeddings.index).all():
            raise ValueError("Prediction embedding sample identities are missing or duplicated.")
    for start in range(0, len(base.scale), batch_samples):
        end = min(start + batch_samples, len(base.scale))
        left, right = np.searchsorted(positions, [start, end])
        batch = LightGBMFrames(
            base.scale.iloc[start:end].reset_index(drop=True),
            base.scale_target[start:end],
            base.shape.iloc[left:right].reset_index(drop=True),
            base.shape_target[left:right],
        )
        if embeddings is not None:
            batch = attach_lightgbm_embedding_frame(
                batch, embeddings=embeddings.loc[scale_ids[start:end]].reset_index(drop=True)
            )
        if untrained_control:
            from execsim.ml.paper.features import append_untrained_neural_control_frames

            batch = LightGBMFrames(
                *append_untrained_neural_control_frames(batch.as_tuple(), fold_seed=13)
            )
        yield batch


def publish_bundle(
    destination: Path,
    *,
    identity: Mapping[str, Any],
    build: Callable[[Path], None],
) -> dict[str, Any]:
    """Publish a complete heterogeneous report tree, or verify it without rebuilding."""
    marker = "completion.json"

    def inventory(root: Path) -> dict[str, str]:
        return {
            path.relative_to(root).as_posix(): file_sha256(path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and path != root / marker
        }

    if destination.exists():
        receipt = read_json(destination / marker)
        if (
            receipt.get("schema_version") != "paper-evaluation-bundle-v1"
            or receipt.get("identity") != dict(identity)
            or not receipt.get("files")
            or receipt["files"] != inventory(destination)
        ):
            raise ValueError("Evaluation bundle identity or file inventory/checksum mismatch.")
        return receipt
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".report-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "complete"
        staging.mkdir()
        build(staging)
        files = inventory(staging)
        if not files or (staging / marker).exists():
            raise ValueError("Report builder must produce files without a completion marker.")
        for name in files:
            with (staging / name).open("r+b") as handle:
                os.fsync(handle.fileno())
        receipt = {
            "schema_version": "paper-evaluation-bundle-v1",
            "identity": dict(identity),
            "files": files,
        }
        write_json_atomic(staging / marker, receipt)
        with (staging / marker).open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(staging, destination)
    return receipt


def merge_result_shards(
    destination: Path,
    *,
    sources: Mapping[str, tuple[Path, str]],
    keys: Sequence[str],
    identity: Mapping[str, Any],
    schema_version: str,
) -> dict[str, Any]:
    """Sort one independent shard at a time, then stream disjoint key ranges.

    Callers supply the complete expected shard inventory and authoritative hashes.
    Overlapping ranges are rejected, not silently concatenated or deduplicated.
    Memory is bounded by one result shard plus one Parquet write batch.
    """
    if not sources or not keys:
        raise ValueError("Result merge requires expected shards and canonical keys.")
    if len({path.resolve() for path, _ in sources.values()}) != len(sources):
        raise ValueError("Result merge duplicates an input artifact.")
    binding = {
        **identity,
        "sources": {name: digest for name, (_, digest) in sorted(sources.items())},
        "sort_keys": list(keys),
    }
    receipt_path = destination.with_suffix(".manifest.json")
    for path, digest in sources.values():
        if file_sha256(path) != digest:
            raise ValueError("Result merge input checksum mismatch.")
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if (
            receipt.get("merge_identity") != binding
            or receipt.get("schema_version") != schema_version
            or file_sha256(destination) != receipt.get("parquet_sha256")
        ):
            raise ValueError("Result merge completion identity or checksum mismatch.")
        with pq.ParquetFile(destination) as parquet:
            if parquet.metadata.num_rows != receipt.get("rows") or str(
                parquet.schema_arrow
            ) != receipt.get("schema"):
                raise ValueError("Result merge completion row count or schema mismatch.")
        return receipt
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".merge-", dir=destination.parent) as temporary:
        staging = Path(temporary)
        ranges = []
        schemas = []
        total = 0
        for index, (_, (path, _)) in enumerate(sorted(sources.items())):
            frame = pd.read_parquet(path)
            if frame.duplicated(list(keys)).any():
                raise ValueError("Result shard duplicates canonical row identity.")
            frame = frame.sort_values(list(keys), kind="stable", na_position="last")
            table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(None)
            schemas.append(table.schema)
            if frame.empty:
                continue

            first, last = (
                tuple((1, None) if pd.isna(value) else (0, value) for value in values)
                for values in frame.iloc[[0, -1]][list(keys)].itertuples(index=False, name=None)
            )
            shard = staging / f"{index}.parquet"
            pq.write_table(table, shard, row_group_size=ROW_GROUP_SIZE)
            ranges.append((first, last, shard))
            total += len(frame)
            del frame, table
        schema = pa.unify_schemas(schemas)
        ranges.sort(key=lambda item: item[0])
        for previous, current in pairwise(ranges):
            if previous[1] >= current[0]:
                raise ValueError("Result shards overlap canonical identity ranges.")
        output = staging / "result.parquet"
        with pq.ParquetWriter(output, schema) as writer:
            for _, _, path in ranges:
                with pq.ParquetFile(path) as parquet:
                    for batch in parquet.iter_batches(batch_size=ROW_GROUP_SIZE):
                        writer.write_table(
                            pa.Table.from_batches([batch]).cast(schema),
                            row_group_size=ROW_GROUP_SIZE,
                        )
        with output.open("r+b") as handle:
            os.fsync(handle.fileno())
        receipt = {
            "schema_version": schema_version,
            "paper_config_hash": identity["paper_config_hash"],
            "merge_identity": binding,
            "rows": total,
            "parquet_sha256": file_sha256(output),
            "schema": str(schema),
        }
        os.replace(output, destination)
        write_json_atomic(receipt_path, receipt)
    return receipt


class VerifiedArtifact:
    """Verify immutable files once per worker; detect subsequent filesystem changes."""

    def __init__(
        self, directory: Path, *, identity: Mapping[str, Any], names: Sequence[str]
    ) -> None:
        self.directory = directory.resolve()
        self.identity = dict(identity)
        verify_artifact(self.directory, identity=self.identity, names=names)
        self._states = {name: self._state(name) for name in (*names, "manifest.json")}

    def _state(self, name: str) -> tuple[int, int, int, int]:
        stat = (self.directory / name).stat()
        return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino

    def check(self, directory: Path, identity: Mapping[str, Any]) -> None:
        """Reject changed identity or files during this immutable artifact lifetime."""
        if directory.resolve() != self.directory or dict(identity) != self.identity:
            raise ValueError("Verified evaluation artifact identity mismatch.")
        if any(self._state(name) != state for name, state in self._states.items()):
            raise ValueError("Verified evaluation artifact changed during execution.")


def forecast_metric_frame(
    base: LightGBMFrames,
    totals: np.ndarray,
    predicted_shape: pd.DataFrame,
    *,
    fold_id: str,
    method: str,
    seed: int | None,
) -> pd.DataFrame:
    """Compute named case-level errors using vectorized complete keyed alignment."""
    validate_base(base)
    if len(totals) != len(base.scale) or not np.isfinite(totals).all() or np.any(totals < 0):
        raise ValueError("Forecast totals must be finite nonnegative aligned predictions.")
    actual = base.shape.loc[:, ["case_id", "target_bucket"]].copy()
    actual["actual_share"] = base.shape_target
    joined = actual.merge(
        predicted_shape.loc[:, ["case_id", "target_bucket", "conditional_share"]],
        on=["case_id", "target_bucket"],
        validate="one_to_one",
        how="outer",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("Forecast shape population differs from the declared base.")
    shares = joined["conditional_share"].to_numpy(dtype=float)
    if not np.isfinite(shares).all() or np.any(shares < 0):
        raise ValueError("Forecast shares must be finite and nonnegative.")
    sums = joined.groupby("case_id", sort=False)["conditional_share"].sum()
    if not np.allclose(sums.to_numpy(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Forecast conditional shares must sum to one.")
    joined = joined.sort_values(["case_id", "target_bucket"], kind="stable")
    cumulative = joined.groupby("case_id", sort=False)[
        ["actual_share", "conditional_share"]
    ].cumsum()
    distances = (cumulative["actual_share"] - cumulative["conditional_share"]).abs()
    errors = distances.groupby(joined["case_id"], sort=False).mean()
    scale = base.scale.reset_index(drop=True)
    return pd.DataFrame(
        {
            "fold_id": fold_id,
            "method": method,
            "seed": seed,
            "sample_id": scale["sample_id"],
            "instrument_id": scale["instrument_id"],
            "session_date": scale["session_date"],
            "as_of_token": scale["as_of"].astype(int),
            "actual_remaining_volume": base.scale_target,
            "causal_baseline_remaining_volume": scale["baseline_remaining_volume"],
            "predicted_remaining_volume": totals,
            "log_remaining_volume_absolute_error": np.abs(
                np.log1p(totals) - np.log1p(base.scale_target)
            ),
            "conditional_curve_wasserstein": scale["sample_id"].map(errors),
        }
    )


def verify_artifact(
    directory: Path, *, identity: Mapping[str, Any], names: Sequence[str]
) -> dict[str, Any]:
    """Reject incomplete, changed, or incompatible artifacts before reading rows."""
    receipt = read_json(directory / "manifest.json")
    if receipt.get("identity") != dict(identity):
        raise ValueError("Evaluation artifact identity mismatch.")
    files = receipt.get("files", {})
    if set(files) != set(names):
        raise ValueError("Evaluation artifact file inventory mismatch.")
    for name in names:
        path = directory / name
        metadata = files[name]
        if not path.is_file() or file_sha256(path) != metadata["sha256"]:
            raise ValueError(f"Evaluation artifact checksum mismatch: {name}")
        parquet = pq.ParquetFile(path)
        if parquet.metadata.num_rows != metadata["rows"]:
            raise ValueError(f"Evaluation artifact row count mismatch: {name}")
        if str(parquet.schema_arrow) != metadata["schema"]:
            raise ValueError(f"Evaluation artifact schema mismatch: {name}")
    return receipt


def publish_frames(
    directory: Path,
    *,
    identity: Mapping[str, Any],
    frames: Mapping[str, pd.DataFrame],
    row_group_size: int = ROW_GROUP_SIZE,
) -> dict[str, Any]:
    """Publish a directory only after every data file and its manifest are durable.

    Callers own disjoint destinations. Existing artifacts are verified rather than
    overwritten. Temporary directories are never valid completion markers.
    """
    if directory.exists():
        return verify_artifact(directory, identity=identity, names=tuple(frames))
    if not frames or any(Path(name).name != name for name in frames):
        raise ValueError("Artifact requires a nonempty inventory of local file names.")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".evaluation-", dir=directory.parent) as temporary:
        staging = Path(temporary) / "complete"
        staging.mkdir()
        files = {}
        for name, frame in frames.items():
            path = staging / name
            table = pa.Table.from_pandas(frame, preserve_index=False)
            pq.write_table(table, path, row_group_size=row_group_size)
            with path.open("r+b") as stream:
                os.fsync(stream.fileno())
            parquet = pq.ParquetFile(path)
            files[name] = {
                "sha256": file_sha256(path),
                "rows": parquet.metadata.num_rows,
                "schema": str(parquet.schema_arrow),
            }
            parquet.close()
        receipt = {"identity": dict(identity), "files": files}
        write_json_atomic(staging / "manifest.json", receipt)
        with (staging / "manifest.json").open("r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(staging, directory)
    return receipt


def validate_base(base: LightGBMFrames) -> None:
    """Validate one scale row and the complete ordered future shape per sample."""
    scale, totals, shape, shares = base.as_tuple()
    if len(scale) != len(totals) or len(shape) != len(shares):
        raise ValueError("Evaluation base target alignment mismatch.")
    if scale.empty or shape.empty or scale["sample_id"].duplicated().any():
        raise ValueError("Evaluation base requires unique nonempty scale identities.")
    if scale["sample_id"].isna().any() or shape["sample_id"].isna().any():
        raise ValueError("Evaluation base has null sample identity.")
    if shape.duplicated(["sample_id", "target_bucket"]).any():
        raise ValueError("Evaluation shape duplicates sample/bucket identity.")
    if not (shape["case_id"].astype(str) == shape["sample_id"].astype(str)).all():
        raise ValueError("Evaluation shape case identity mismatch.")
    for values in (totals, shares):
        if not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError("Evaluation base requires finite nonnegative targets.")
    if np.any(totals <= 0):
        raise ValueError("Evaluation scale requires positive totals.")
    expected = scale.set_index("sample_id")["as_of"]
    if not shape["sample_id"].isin(expected.index).all():
        raise ValueError("Evaluation shape lacks scale identity.")
    origins = shape["sample_id"].map(expected).to_numpy(dtype=int)
    buckets = shape["target_bucket"].to_numpy(dtype=int)
    if np.any(buckets < origins) or np.any(buckets >= 26):
        raise ValueError("Evaluation shape contains unavailable future buckets.")
    counts = shape.groupby("sample_id", sort=False).size().reindex(expected.index)
    if not np.array_equal(counts.to_numpy(), 26 - expected.to_numpy()):
        raise ValueError("Evaluation shape must cover every future bucket.")
    actual = pd.Series(shares, index=shape.index).groupby(shape["sample_id"]).sum()
    if not np.allclose(actual.to_numpy(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Evaluation conditional targets must sum to one.")


def evaluation_base(
    sequence_manifest: Path,
    *,
    partition: str,
    liquidity_groups: dict[str, int],
    directory: Path,
    execution_identity: Mapping[str, Any],
    builder: Callable[..., LightGBMFrames] = build_lightgbm_base_frames,
) -> LightGBMFrames:
    """Build one compact partition base, then load verified immutable derived rows.

    VALIDATION is supported for semantic qualification. TEST callers must
    perform authorization before invoking this low-level artifact operation.
    """
    if partition not in {"validation", "test"}:
        raise ValueError("Evaluation cache requires VALIDATION or TEST complete grids.")
    required = {"parameter_freeze_sha256", "source_commit", "source_tree", "paper_config_hash"}
    if required.difference(execution_identity) or any(
        not execution_identity[key] for key in required
    ):
        raise ValueError("Evaluation execution identity is incomplete.")
    identity = {
        **dict(execution_identity),
        "schema_version": BASE_SCHEMA,
        "sequence_manifest_sha256": file_sha256(sequence_manifest),
        "partition": partition,
        "liquidity_groups": liquidity_groups,
    }
    names = ("scale-base.parquet", "shape-base.parquet")
    if not directory.exists():
        base = builder(sequence_manifest, partition=partition, liquidity_groups=liquidity_groups)
        validate_base(base)
        scale = base.scale.assign(**{_TARGET: base.scale_target})
        shape = base.shape.assign(**{_TARGET: base.shape_target})
        keys = ["instrument_id", "session_date", "as_of", "sample_id"]
        scale = scale.sort_values(keys, kind="stable").reset_index(drop=True)
        shape = shape.sort_values([*keys, "target_bucket"], kind="stable").reset_index(drop=True)
        publish_frames(
            directory, identity=identity, frames=dict(zip(names, (scale, shape), strict=True))
        )
    verify_artifact(directory, identity=identity, names=names)
    scale = pq.read_table(directory / names[0], memory_map=True).to_pandas()
    shape = pq.read_table(directory / names[1], memory_map=True).to_pandas()
    base = LightGBMFrames(
        scale, scale.pop(_TARGET).to_numpy(), shape, shape.pop(_TARGET).to_numpy()
    )
    validate_base(base)
    return base
