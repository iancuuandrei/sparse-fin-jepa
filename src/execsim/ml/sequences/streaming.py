"""Bounded, deterministic PyTorch dataset over partitioned session Parquet."""

from __future__ import annotations

import random
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

from execsim.data.paper.manifests import file_sha256, read_json, write_json_atomic
from execsim.data.paper.partitions import resolve_fold_partition
from execsim.ml.sequences.dataset import extract_window
from execsim.ml.sequences.index import sample_training_positions
from execsim.ml.sequences.manifests import read_sequence_record
from execsim.ml.sequences.schemas import SequenceRecord, SequenceSample


class PaperSequenceDataset(Dataset[dict[str, Any]]):
    """Generate contexts and targets on demand with a bounded session cache."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        partition: str,
        seed: int,
        cache_size: int = 32,
        sample_train_positions: bool = True,
    ) -> None:
        if partition not in {"train", "validation", "test"} or cache_size <= 0:
            raise ValueError("Streaming dataset partition or cache size is invalid.")
        payload = read_json(manifest_path)
        self.root = manifest_path.parent
        self.fold_id = str(payload["fold_id"])
        self.partition = partition
        self.seed = seed
        self.cache_size = cache_size
        self.sample_train_positions = sample_train_positions
        self.epoch = 0
        sequence_paths = [self.root / str(value) for value in payload["sequence_files"]]
        self._sessions = {
            path.stem: path
            for path in sequence_paths
            if f"sessions/{partition}/" in str(path).replace("\\", "/")
        }
        index_paths = [
            self.root / str(value)
            for value in payload["index_files"]
            if f"indexes/{partition}/" in str(value).replace("\\", "/")
        ]
        if not self._sessions or not index_paths:
            raise ValueError(f"Sequence manifest has no {partition} sessions or indexes.")
        index_frame = _load_cached_index_frame(
            manifest_path,
            partition=partition,
            index_paths=tuple(sorted(index_paths)),
        )
        self._all_samples = tuple(
            _sample_from_row(row) for row in index_frame.itertuples(index=False)
        )
        for sample in self._all_samples:
            if sample.fold_id != self.fold_id or sample.partition != partition:
                raise ValueError("Sequence index contradicts manifest fold or partition.")
        self._cache: OrderedDict[str, SequenceRecord] = OrderedDict()
        self._samples: tuple[SequenceSample, ...] = ()
        self.set_epoch(0)

    def set_epoch(self, epoch: int) -> None:
        """Select exactly two deterministic train positions per session and epoch."""
        if epoch < 0:
            raise ValueError("Dataset epoch must be non-negative.")
        self.epoch = epoch
        grouped: dict[str, list[SequenceSample]] = {}
        for sample in self._all_samples:
            grouped.setdefault(sample.session_id, []).append(sample)
        if self.partition == "train" and self.sample_train_positions:
            selected = [
                sample
                for session_id in sorted(grouped)
                for sample in sample_training_positions(
                    tuple(sorted(grouped[session_id], key=lambda item: item.as_of_token)),
                    epoch=epoch,
                    seed=self.seed,
                )
            ]
        else:
            selected = [
                sample
                for session_id in sorted(grouped)
                for sample in sorted(grouped[session_id], key=lambda item: item.as_of_token)
            ]
        self._samples = tuple(selected)

    def state_dict(self) -> dict[str, int]:
        """Expose sampler continuation state."""
        return {"epoch": self.epoch}

    def load_state_dict(self, payload: dict[str, int]) -> None:
        """Restore deterministic sampler epoch."""
        self.set_epoch(int(payload["epoch"]))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._samples[index]
        record = self._record(sample.session_id)
        session_date = pd.Timestamp(record.session_date).date()
        if resolve_fold_partition(self.fold_id, session_date) != self.partition:
            raise ValueError("Session date contradicts streaming dataset partition.")
        window = extract_window(record, sample)
        raw_target_volume = np.zeros(4, dtype=np.float32)
        causal_target_volume = np.zeros(4, dtype=np.float32)
        for horizon_index, (target_index, available) in enumerate(
            zip(sample.target_indices, sample.target_mask, strict=True)
        ):
            if available:
                raw_target_volume[horizon_index] = record.raw_volume[target_index]
                causal_target_volume[horizon_index] = record.causal_baseline_volume[target_index]
        return {
            "context": torch.as_tensor(window["context"], dtype=torch.float32),
            "context_mask": torch.as_tensor(window["context_mask"], dtype=torch.bool),
            "targets": torch.as_tensor(window["targets"], dtype=torch.float32),
            "target_mask": torch.as_tensor(window["target_mask"], dtype=torch.bool),
            "raw_target_volume": torch.from_numpy(raw_target_volume),
            "causal_target_volume": torch.from_numpy(causal_target_volume),
            "sample_id": sample.sample_id,
            "session_id": sample.session_id,
            "instrument_id": record.instrument_id,
            "symbol": record.symbol,
            "session_date": record.session_date,
            "as_of_token": sample.as_of_token,
            "as_of_ns": sample.as_of_ns,
            "sequence_hash": sample.source_sequence_hash,
            "cutoff": sample.cutoff,
            "training_cutoff": sample.training_cutoff,
            "market_information_as_of": sample.market_information_as_of,
            "feature_history_end": sample.feature_history_end,
        }

    def _record(self, session_id: str) -> SequenceRecord:
        if session_id in self._cache:
            record = self._cache.pop(session_id)
            self._cache[session_id] = record
            return record
        path = self._sessions.get(session_id)
        if path is None:
            raise ValueError(f"Index references unknown sequence session: {session_id}")
        record = read_sequence_record(path)
        self._cache[session_id] = record
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return record


def build_sequence_dataloader(
    dataset: PaperSequenceDataset,
    *,
    batch_size: int,
    num_workers: int,
    device: str,
    prefetch_factor: int = 2,
) -> DataLoader[dict[str, Any]]:
    """Build a Windows-spawn-safe loader with bounded deterministic prefetch."""
    if batch_size <= 0 or num_workers < 0 or prefetch_factor <= 0:
        raise ValueError("DataLoader batch, worker, and prefetch settings must be positive.")
    arguments: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": device.startswith("cuda"),
        "worker_init_fn": _seed_worker,
        "generator": torch.Generator().manual_seed(dataset.seed + dataset.epoch),
        "persistent_workers": False,
    }
    if num_workers:
        arguments["prefetch_factor"] = prefetch_factor
    return DataLoader(**arguments)


def _seed_worker(worker_id: int) -> None:
    seed = int(torch.initial_seed() % 2**32) + worker_id
    np.random.seed(seed)
    random.seed(seed)


def _sample_from_row(row: Any) -> SequenceSample:
    return SequenceSample(
        sample_id=str(row.sample_id),
        session_id=str(row.session_id),
        fold_id=str(row.fold_id),
        partition=str(row.partition),
        as_of_token=int(row.as_of_token),
        context_start=int(row.context_start),
        context_end=int(row.context_end),
        target_indices=tuple(int(value) for value in row.target_indices),
        target_mask=tuple(bool(value) for value in row.target_mask),
        as_of_ns=int(row.as_of_ns),
        source_sequence_hash=str(row.source_sequence_hash),
        cutoff=str(row.cutoff),
        training_cutoff=str(row.training_cutoff),
        market_information_as_of=str(row.market_information_as_of),
        feature_history_end=str(row.feature_history_end),
    )


def _load_cached_index_frame(
    manifest_path: Path,
    *,
    partition: str,
    index_paths: tuple[Path, ...],
) -> pd.DataFrame:
    """Load one validated manifest-keyed cache instead of many tiny index files."""
    manifest_hash = file_sha256(manifest_path)
    cache_root = manifest_path.parent / ".index-cache"
    cache_path = cache_root / f"{partition}-{manifest_hash}.parquet"
    receipt_path = cache_path.with_suffix(".json")
    receipt_identity = {
        "schema_version": "paper-sequence-index-cache-v1",
        "manifest_sha256": manifest_hash,
        "partition": partition,
        "source_index_count": len(index_paths),
    }
    if cache_path.is_file() and receipt_path.is_file():
        receipt = read_json(receipt_path)
        if any(receipt.get(key) != value for key, value in receipt_identity.items()):
            raise ValueError("Sequence index cache receipt is incompatible with its manifest.")
        if receipt.get("cache_sha256") != file_sha256(cache_path):
            raise ValueError("Sequence index cache checksum does not match its receipt.")
        frame = pd.read_parquet(cache_path)
        if int(receipt.get("row_count", -1)) != len(frame):
            raise ValueError("Sequence index cache row count does not match its receipt.")
        return frame

    cache_root.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".parquet.partial")
    if temporary.exists():
        temporary.unlink()
    writer: pq.ParquetWriter | None = None
    row_count = 0
    try:
        for start in range(0, len(index_paths), 512):
            frame = pd.concat(
                (pd.read_parquet(path) for path in index_paths[start : start + 512]),
                ignore_index=True,
            )
            table = pa.Table.from_pandas(frame, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
            row_count += len(frame)
    finally:
        if writer is not None:
            writer.close()
    if writer is None or row_count <= 0:
        raise ValueError(f"Sequence manifest has no {partition} index rows.")
    temporary.replace(cache_path)
    write_json_atomic(
        receipt_path,
        {
            **receipt_identity,
            "row_count": row_count,
            "cache_sha256": file_sha256(cache_path),
        },
    )
    return pd.read_parquet(cache_path)
