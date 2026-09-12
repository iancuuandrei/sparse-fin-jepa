"""Reference-versus-cached probe trajectories and atomic cache failures."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from execsim.ml.representations.frozen_evaluation import (
    FrozenProbeOptions,
    _encoded_batch,
    evaluate_frozen_capacity_streaming,
)
from execsim.ml.representations.jepa import PredictiveRepresentationModel
from execsim.ml.representations.probe_cache import materialize_probe_batches
from execsim.ml.representations.schemas import RepresentationConfig


def batches(partition: str) -> list[dict]:
    generator = torch.Generator().manual_seed(51)
    result = []
    for index, size in enumerate((5, 3)):
        result.append(
            {
                "context": torch.randn(size, 8, 18, generator=generator),
                "context_mask": torch.ones(size, 8, dtype=torch.bool),
                "targets": torch.randn(size, 4, 18, generator=generator),
                "raw_target_volume": torch.rand(size, 4, generator=generator) * 100,
                "causal_target_volume": torch.rand(size, 4, generator=generator) * 90,
                "as_of_token": torch.tensor([4, 12, 18, 20, 23][:size]),
                "sample_id": [f"{partition}-{index}-{row}" for row in range(size)],
                "session_date": ["2024-10-01"] * size,
            }
        )
    return result


@pytest.mark.parametrize("geometry", ["dense", "sparse"])
def test_cached_probe_outputs_match_reference(tmp_path: Path, geometry: str) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(13)
    model = PredictiveRepresentationModel(RepresentationConfig(geometry)).eval()
    state = copy.deepcopy(model.state_dict())
    loaders = [batches(partition) for partition in ("train", "validation", "test")]
    options = FrozenProbeOptions(mlp_epochs=2)
    coordinate_identity = {
        "schema_version": "paper-representation-coordinate-v1",
        "coordinate": f"fold-1-{geometry}-13",
        "checkpoint": "checkpoint-sha",
        "sequence": "sequence-sha",
    }
    cache_base_identity = {**coordinate_identity, "batch_size": 5, "num_workers": 0}
    reference = evaluate_frozen_capacity_streaming(
        model,
        *loaders,
        device="cpu",
        seed=13,
        options=options,
    )
    cached = evaluate_frozen_capacity_streaming(
        model,
        *loaders,
        device="cpu",
        seed=13,
        options=options,
        cache_root=tmp_path / "probe",
        cache_identity=cache_base_identity,
    )
    import pandas as pd

    from execsim.ml.paper.evaluation_artifacts import publish_frames
    from execsim.ml.representations.probe_cache import discard_completed_probe_cache

    publish_frames(
        tmp_path / "published-coordinate",
        identity=coordinate_identity,
        frames={"capacity.parquet": pd.DataFrame(cached[0])},
    )
    discard_completed_probe_cache(tmp_path / "probe", identity=cache_base_identity, device="cpu")
    assert not (tmp_path / "probe").exists()
    assert (tmp_path / "published-coordinate" / "manifest.json").is_file()
    for before_rows, after_rows in zip(reference, cached, strict=True):
        assert len(before_rows) == len(after_rows)
        for before, after in zip(before_rows, after_rows, strict=True):
            assert before.keys() == after.keys()
            for key in before:
                if "seconds" not in key:
                    assert before[key] == after[key], key
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state[key], rtol=0, atol=0)


@pytest.mark.parametrize("geometry", ["dense", "sparse"])
@pytest.mark.parametrize("partition", ["train", "validation", "test"])
def test_encoded_values_and_batch_boundaries(tmp_path: Path, geometry: str, partition: str) -> None:
    model = PredictiveRepresentationModel(RepresentationConfig(geometry)).eval()
    source = batches(partition)
    cache = materialize_probe_batches(
        tmp_path / partition,
        identity={"geometry": geometry, "partition": partition},
        loader=source,
        encode=lambda batch: _encoded_batch(model, batch, "cpu"),
    )
    for before, after in zip(source, cache, strict=True):
        assert before["sample_id"] == after["sample_id"]
        assert before["session_date"] == after["session_date"]
        np.testing.assert_array_equal(before["as_of_token"], after["as_of_token"])
        for expected, actual in zip(
            _encoded_batch(model, before, "cpu"),
            _encoded_batch(model, after, "cpu"),
            strict=True,
        ):
            torch.testing.assert_close(expected, actual, rtol=0, atol=0)


def test_cache_interrupt_identity_and_checksum(tmp_path: Path) -> None:
    model = PredictiveRepresentationModel(RepresentationConfig("dense")).eval()
    root = tmp_path / "cache"

    def interrupted():
        yield batches("train")[0]
        raise RuntimeError("fixture interruption")

    def encode(batch):
        return _encoded_batch(model, batch, "cpu")

    with pytest.raises(RuntimeError, match="interruption"):
        materialize_probe_batches(
            root, identity={"source": "A"}, loader=interrupted(), encode=encode
        )
    assert not root.exists()
    materialize_probe_batches(
        root, identity={"source": "A"}, loader=batches("train"), encode=encode
    )
    with pytest.raises(ValueError, match="identity"):
        materialize_probe_batches(root, identity={"source": "B"}, loader=[], encode=encode)
    with (root / "features.bin").open("r+b") as handle:
        handle.write(b"bad!")
    with pytest.raises(ValueError, match="checksum"):
        materialize_probe_batches(root, identity={"source": "A"}, loader=[], encode=encode)


def test_cache_preserves_loader_generator_and_reuses_without_encoding(tmp_path: Path) -> None:
    from torch.utils.data import DataLoader

    # Batch-size None makes the fixture dictionaries the already-collated items.
    generator = torch.Generator().manual_seed(47)
    loader = DataLoader(batches("train"), batch_size=None, generator=generator)
    before = generator.get_state().clone()
    model = PredictiveRepresentationModel(RepresentationConfig("dense")).eval()
    root = tmp_path / "cache"
    calls = []

    def encode(batch):
        calls.append(1)
        return _encoded_batch(model, batch, "cpu")

    cache = materialize_probe_batches(root, identity={"source": "A"}, loader=loader, encode=encode)
    assert len(calls) == 2
    assert torch.equal(generator.get_state(), before)
    list(loader)
    expected = generator.get_state().clone()
    generator.set_state(before)
    list(cache)
    assert torch.equal(generator.get_state(), expected)
    materialize_probe_batches(root, identity={"source": "A"}, loader=loader, encode=encode)
    assert len(calls) == 2


def test_completed_cache_cleanup_rejects_wrong_identity(tmp_path: Path) -> None:
    from execsim.ml.representations.probe_cache import (
        discard_completed_probe_cache,
        encoded_probe_identity,
    )

    model = PredictiveRepresentationModel(RepresentationConfig("dense")).eval()
    root = tmp_path / "coordinate"
    materialize_probe_batches(
        root / "train",
        identity=encoded_probe_identity({"source": "A"}, partition="train", device="cpu"),
        loader=batches("train"),
        encode=lambda batch: _encoded_batch(model, batch, "cpu"),
    )
    with pytest.raises(ValueError, match="identity"):
        discard_completed_probe_cache(root, identity={"source": "B"}, device="cpu")
    assert root.exists()
    discard_completed_probe_cache(root, identity={"source": "A"}, device="cpu")
    assert not root.exists()


@pytest.mark.parametrize(
    "field",
    [
        "coordinate",
        "checkpoint",
        "sequence",
        "batch_size",
        "num_workers",
        "schema_version",
        "partition",
        "device",
        "torch_version",
    ],
)
def test_cleanup_checks_exact_partition_identity(tmp_path: Path, field: str) -> None:
    from execsim.data.paper.manifests import read_json, write_json_atomic
    from execsim.ml.representations.probe_cache import (
        discard_completed_probe_cache,
        encoded_probe_identity,
    )

    base = {
        "schema_version": "paper-representation-coordinate-v1",
        "coordinate": "fold-1-dense-13",
        "checkpoint": "sha-checkpoint",
        "sequence": "sha-sequence",
        "batch_size": 5,
        "num_workers": 0,
    }
    model = PredictiveRepresentationModel(RepresentationConfig("dense")).eval()
    root = tmp_path / "cache"
    for partition in ("train", "validation", "test"):
        materialize_probe_batches(
            root / partition,
            identity=encoded_probe_identity(base, partition=partition, device="cpu"),
            loader=batches(partition),
            encode=lambda batch: _encoded_batch(model, batch, "cpu"),
        )
    manifest = root / "test" / "manifest.json"
    receipt = read_json(manifest)
    receipt["identity"][field] = "wrong"
    write_json_atomic(manifest, receipt)
    with pytest.raises(ValueError, match="identity"):
        discard_completed_probe_cache(root, identity=base, device="cpu")
    # All partitions must be validated before any deletion occurs.
    assert all((root / p / "features.bin").is_file() for p in ("train", "validation", "test"))
