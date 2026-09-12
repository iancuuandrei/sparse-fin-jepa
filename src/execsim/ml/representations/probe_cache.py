"""Atomic, bounded storage of frozen probe tensors with original batch boundaries."""

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from execsim.data.paper.manifests import file_sha256, read_json, write_json_atomic

FIELDS = ("features", "targets", "observable", "complete")


def discard_completed_probe_cache(root: Path, *, identity: dict[str, Any]) -> None:
    """Remove only verified disposable tensors after the caller seals a coordinate."""
    if not root.exists():
        return
    if root.is_symlink() or any(
        p.name not in {"train", "validation", "test"} for p in root.iterdir()
    ):
        raise ValueError("Unexpected completed probe cache directory.")
    files: list[Path] = []
    directories = list(root.iterdir())
    for partition in directories:
        if partition.is_symlink() or any(p.is_symlink() for p in partition.iterdir()):
            raise ValueError("Probe cache cleanup rejects symbolic links.")
        recorded = read_json(partition / "manifest.json")["identity"]
        if any(recorded.get(key) != value for key, value in identity.items()):
            raise ValueError("Completed probe cache identity mismatch.")
        EncodedProbeBatches(partition, recorded, [])
        files.extend(partition.iterdir())
    for path in files:
        path.unlink()
    for partition in directories:
        partition.rmdir()
    root.rmdir()


class EncodedProbeBatches:
    """Replay exact encoded batches from four contiguous files, never a full RAM copy."""

    def __init__(self, root: Path, identity: dict[str, Any], loader: Any) -> None:
        self.root = root
        self.loader = loader
        self.receipt = read_json(root / "manifest.json")
        if self.receipt["identity"] != identity:
            raise ValueError("Encoded probe cache identity mismatch.")
        expected_files = {"manifest.json", "batches.jsonl", *(f"{key}.bin" for key in FIELDS)}
        if {p.name for p in root.iterdir()} != expected_files:
            raise ValueError("Encoded probe cache file inventory mismatch.")
        if set(self.receipt["sha256"]) != expected_files - {"manifest.json"}:
            raise ValueError("Encoded probe cache checksum inventory mismatch.")
        if set(self.receipt["arrays"]) != set(FIELDS):
            raise ValueError("Encoded probe cache tensor inventory mismatch.")
        for name, digest in self.receipt["sha256"].items():
            if file_sha256(root / name) != digest:
                raise ValueError("Encoded probe cache checksum mismatch.")
        row_counts = set()
        for key, spec in self.receipt["arrays"].items():
            shape = spec["shape"]
            if not shape or any(not isinstance(n, int) or n <= 0 for n in shape):
                raise ValueError("Encoded probe cache shape is invalid.")
            expected_bytes = int(np.prod(shape)) * np.dtype(spec["dtype"]).itemsize
            if (root / f"{key}.bin").stat().st_size != expected_bytes:
                raise ValueError("Encoded probe cache tensor size mismatch.")
            row_counts.add(shape[0])
        if len(row_counts) != 1:
            raise ValueError("Encoded probe cache tensors are not aligned.")

    def __iter__(self) -> Iterator[dict[str, Any]]:
        import torch
        from torch.utils.data import DataLoader

        # DataLoader iterator creation consumes one base seed even with zero
        # workers. Preserve that RNG transition on every replay, without I/O.
        if isinstance(self.loader, DataLoader):
            torch.empty((), dtype=torch.int64).random_(generator=self.loader.generator)
        arrays = {
            key: np.memmap(
                self.root / f"{key}.bin",
                mode="r",
                dtype=spec["dtype"],
                shape=tuple(spec["shape"]),
            )
            for key, spec in self.receipt["arrays"].items()
        }
        with (self.root / "batches.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                batch = json.loads(line)
                start, end = batch.pop("start"), batch.pop("end")
                batch["encoded_probe"] = tuple(
                    torch.from_numpy(np.array(arrays[key][start:end], copy=True)) for key in FIELDS
                )
                yield batch


def materialize_probe_batches(
    root: Path,
    *,
    identity: dict[str, Any],
    loader: Iterable[dict[str, Any]],
    encode: Callable[[dict[str, Any]], tuple[Any, Any, Any, Any]],
) -> EncodedProbeBatches:
    """Encode once, preserve dtype/order/masks, and publish only a complete cache."""
    import torch

    if root.exists():
        return EncodedProbeBatches(root, identity, loader)
    root.parent.mkdir(parents=True, exist_ok=True)
    generator = getattr(loader, "generator", None)
    generator_state = generator.get_state() if generator is not None else None
    try:
        # Materialization is an operational prepass, not a new scientific RNG step.
        with (
            torch.random.fork_rng(),
            tempfile.TemporaryDirectory(prefix=".probe-", dir=root.parent) as temporary,
        ):
            staging = Path(temporary) / "cache"
            staging.mkdir()
            rows = 0
            arrays: dict[str, Any] = {}
            with (staging / "batches.jsonl").open("w", encoding="utf-8") as metadata:
                for batch in loader:
                    values = encode(batch)
                    count = len(values[0])
                    if count == 0:
                        raise ValueError("Encoded probe cache cannot contain empty batches.")
                    for key, tensor in zip(FIELDS, values, strict=True):
                        array = tensor.detach().cpu().numpy()
                        if len(array) != count or not np.isfinite(array).all():
                            raise ValueError("Encoded probe tensors must be finite and aligned.")
                        spec = {"dtype": array.dtype.str, "tail": list(array.shape[1:])}
                        if key in arrays and arrays[key] != spec:
                            raise ValueError("Encoded probe tensor schema changed between batches.")
                        arrays[key] = spec
                        with (staging / f"{key}.bin").open("ab") as output:
                            output.write(array.tobytes(order="C"))
                    metadata.write(
                        json.dumps(
                            {
                                "start": rows,
                                "end": rows + count,
                                "sample_id": list(batch["sample_id"]),
                                "session_date": list(batch["session_date"]),
                                "as_of_token": np.asarray(batch["as_of_token"]).tolist(),
                            }
                        )
                        + "\n"
                    )
                    rows += count
            if not rows:
                raise ValueError("Encoded probe cache has no rows.")
            receipt = {
                "identity": identity,
                "arrays": {
                    key: {"dtype": spec["dtype"], "shape": [rows, *spec["tail"]]}
                    for key, spec in arrays.items()
                },
                "sha256": {p.name: file_sha256(p) for p in sorted(staging.iterdir())},
            }
            write_json_atomic(staging / "manifest.json", receipt)
            staging.rename(root)
    finally:
        if generator is not None and generator_state is not None:
            generator.set_state(generator_state)
    return EncodedProbeBatches(root, identity, loader)
