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
        cache_identity={"fixture": geometry},
    )
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
    from execsim.ml.representations.probe_cache import discard_completed_probe_cache

    model = PredictiveRepresentationModel(RepresentationConfig("dense")).eval()
    root = tmp_path / "coordinate"
    materialize_probe_batches(
        root / "train",
        identity={"source": "A"},
        loader=batches("train"),
        encode=lambda batch: _encoded_batch(model, batch, "cpu"),
    )
    with pytest.raises(ValueError, match="identity"):
        discard_completed_probe_cache(root, identity={"source": "B"})
    assert root.exists()
    discard_completed_probe_cache(root, identity={"source": "A"})
    assert not root.exists()
