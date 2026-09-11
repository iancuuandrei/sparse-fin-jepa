"""Partitioned, compatibility-checked embedding export over sequence stores."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from execsim.data.paper.manifests import file_sha256, read_json, write_json_atomic
from execsim.ml.representations.checkpoints import load_checkpoint
from execsim.ml.representations.embeddings import export_frozen_embedding_batch
from execsim.ml.representations.schemas import CheckpointCompatibility
from execsim.ml.sequences.streaming import PaperSequenceDataset, build_sequence_dataloader


def export_embedding_corpus(
    model: Any,
    *,
    checkpoint_directory: Path,
    expected_checkpoint: CheckpointCompatibility,
    sequence_manifest_path: Path,
    output_root: Path,
    seed: int,
    geometry: str,
    adaptation: str,
    device: str,
    batch_size: int = 512,
    num_workers: int = 0,
    cache_size: int = 32,
) -> Path:
    """Export all valid train/validation/test rows in three partitioned Parquet files."""
    import torch

    checkpoint = load_checkpoint(model, checkpoint_directory, expected=expected_checkpoint)
    model.to(device)
    checkpoint_manifest_hash = file_sha256(checkpoint_directory / "manifest.json")
    sequence_hash = file_sha256(sequence_manifest_path)
    sequence_manifest = read_json(sequence_manifest_path)
    output_root.mkdir(parents=True, exist_ok=False)
    files = []
    total_rows = 0
    for partition in ("train", "validation", "test"):
        dataset = PaperSequenceDataset(
            sequence_manifest_path,
            partition=partition,
            seed=seed,
            cache_size=cache_size,
            sample_train_positions=False,
        )
        loader = build_sequence_dataloader(
            dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            device=device,
        )
        destination = output_root / f"partition={partition}" / "embeddings.parquet"
        destination.parent.mkdir(parents=True)
        writer: pq.ParquetWriter | None = None
        partition_rows = 0
        try:
            for batch in loader:
                context = batch["context"].to(device, non_blocking=device.startswith("cuda"))
                mask = batch["context_mask"].to(device, non_blocking=device.startswith("cuda"))
                horizon_mask = batch["target_mask"].to(
                    device, non_blocking=device.startswith("cuda")
                )
                embeddings = export_frozen_embedding_batch(model, context, mask, horizon_mask)
                batch_rows = []
                for index, values in enumerate(embeddings):
                    batch_rows.append(
                        {
                            "sample_id": batch["sample_id"][index],
                            "session_id": batch["session_id"][index],
                            "instrument_id": batch["instrument_id"][index],
                            "symbol": batch["symbol"][index],
                            "session_date": batch["session_date"][index],
                            "as_of_ns": int(batch["as_of_ns"][index]),
                            "fold_id": checkpoint.fold_id,
                            "partition": partition,
                            "seed": seed,
                            "geometry": geometry,
                            "adaptation": adaptation,
                            "checkpoint_hash": checkpoint.weights_sha256,
                            "sequence_hash": batch["sequence_hash"][index],
                            "cutoff": batch["cutoff"][index],
                            "training_cutoff": batch["training_cutoff"][index],
                            "market_information_as_of": batch["market_information_as_of"][index],
                            "feature_history_end": batch["feature_history_end"][index],
                            "embedding": values.tolist(),
                        }
                    )
                table = pa.Table.from_pylist(
                    batch_rows,
                    schema=None if writer is None else writer.schema,
                )
                if writer is None:
                    writer = pq.ParquetWriter(destination, table.schema, compression="zstd")
                writer.write_table(table)
                partition_rows += len(batch_rows)
        except BaseException:
            if writer is not None:
                writer.close()
            destination.unlink(missing_ok=True)
            raise
        else:
            if writer is not None:
                writer.close()
        if partition_rows == 0:
            raise ValueError(f"Embedding export found no rows for {partition}.")
        files.append(
            {
                "partition": partition,
                "path": str(destination.relative_to(output_root)).replace("\\", "/"),
                "rows": partition_rows,
                "sha256": file_sha256(destination),
            }
        )
        total_rows += partition_rows
    payload = {
        "schema_version": "paper-embedding-corpus-v2",
        "fold_id": checkpoint.fold_id,
        "seed": seed,
        "geometry": geometry,
        "adaptation": adaptation,
        "components": ["current", "h1", "h2", "h4", "h8", "horizon_availability"],
        "rows": total_rows,
        "checkpoint_hash": checkpoint.weights_sha256,
        "checkpoint_manifest_hash": checkpoint_manifest_hash,
        "sequence_manifest_hash": sequence_hash,
        "normalization_hash": expected_checkpoint.normalization_hash,
        "paper_config_hash": expected_checkpoint.paper_config_hash,
        "training_cutoff": checkpoint.cutoff,
        "torch_version": torch.__version__,
        "pytorch_compatibility": expected_checkpoint.torch_compatibility,
        "files": files,
        "source_sequence_manifest_id": sequence_manifest["manifest_id"],
    }
    write_json_atomic(output_root / "manifest.json", payload)
    return output_root / "manifest.json"
