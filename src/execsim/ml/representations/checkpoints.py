"""Safe safetensors checkpoints with checksummed manifests."""

from __future__ import annotations

import hashlib
import io
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from torch import Tensor

from execsim.data.paper.manifests import file_sha256, write_json_atomic
from execsim.ml.representations.schemas import CheckpointCompatibility, CheckpointManifest


class StateDictModel(Protocol):
    """Minimal torch state-dictionary boundary used by safe checkpoints."""

    def state_dict(self) -> dict[str, Tensor]: ...

    def load_state_dict(self, state_dict: dict[str, Tensor]) -> object: ...


class OptimizerState(Protocol):
    """Minimal optimizer-state boundary for trusted local continuation."""

    def state_dict(self) -> dict[str, object]: ...

    def load_state_dict(self, state_dict: dict[str, object]) -> None: ...


@dataclass(frozen=True, slots=True)
class ResumeProgress:
    """Return deterministic continuation counters after trusted restoration."""

    epoch: int
    global_step: int
    sampler_epoch: int
    rdm_counter: int
    batch_index: int = 0
    trainer_state: dict[str, Any] | None = None


def save_checkpoint(model: StateDictModel, directory: Path, manifest: CheckpointManifest) -> Path:
    """Persist tensor weights without pickle and verify the declared checksum."""
    from safetensors.torch import save_file

    directory.mkdir(parents=True, exist_ok=False)
    weights = directory / "model.safetensors"
    state_dict = model.state_dict()
    save_file(
        {name: tensor.detach().cpu().contiguous() for name, tensor in state_dict.items()}, weights
    )
    checksum = file_sha256(weights)
    if manifest.weights_sha256 and manifest.weights_sha256 != checksum:
        raise ValueError("Checkpoint checksum does not match the saved safetensors file.")
    payload = asdict(manifest)
    payload["weights_sha256"] = checksum
    write_json_atomic(directory / "manifest.json", payload)
    return directory


def load_checkpoint(
    model: StateDictModel, directory: Path, *, expected: CheckpointCompatibility
) -> CheckpointManifest:
    """Validate the complete experiment identity before loading safe weights."""
    from safetensors.torch import load_file

    payload = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    manifest = CheckpointManifest(**payload)
    _validate_checkpoint_compatibility(manifest, expected)
    weights = directory / "model.safetensors"
    if file_sha256(weights) != manifest.weights_sha256:
        raise ValueError("Checkpoint weights checksum does not match its manifest.")
    model.load_state_dict(load_file(weights))
    return manifest


def save_trusted_resume_state(
    path: Path,
    optimizer: OptimizerState,
    *,
    epoch: int,
    global_step: int = 0,
    sampler_epoch: int | None = None,
    rdm_counter: int = 0,
    scheduler: OptimizerState | None = None,
    batch_index: int = 0,
    trainer_state: dict[str, Any] | None = None,
) -> str:
    """Save all optimizer, scheduler, RNG, sampler, and RDM continuation state."""
    import torch

    if min(epoch, global_step, rdm_counter, batch_index) < 0:
        raise ValueError("Resume counters must be non-negative.")
    sampler_epoch = epoch if sampler_epoch is None else sampler_epoch
    if sampler_epoch < 0:
        raise ValueError("Resume sampler epoch must be non-negative.")
    payload = {
        "trusted_local_only": True,
        "epoch": epoch,
        "global_step": global_step,
        "sampler_epoch": sampler_epoch,
        "rdm_counter": rdm_counter,
        "batch_index": batch_index,
        "trainer_state": trainer_state,
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_trusted_resume_state(
    path: Path,
    optimizer: OptimizerState,
    *,
    expected_sha256: str,
    trusted_local: bool,
    scheduler: OptimizerState | None = None,
) -> ResumeProgress:
    """Restore complete local continuation state after trust and checksum validation."""
    import torch

    if not trusted_local:
        raise PermissionError("Resume state uses pickle and requires explicit local trust.")
    content = path.read_bytes()
    if hashlib.sha256(content).hexdigest() != expected_sha256:
        raise ValueError("Resume-state checksum mismatch.")
    # Legacy continuation files contain NumPy's MT19937 uint32 state. Permit
    # only those reconstruction types, never arbitrary checkpoint globals.
    from numpy.core.multiarray import _reconstruct

    with torch.serialization.safe_globals(
        [
            (_reconstruct, "numpy.core.multiarray._reconstruct"),
            (_reconstruct, "numpy._core.multiarray._reconstruct"),
            np.ndarray,
            np.dtype,
            type(np.dtype(np.uint32)),
        ]
    ):
        payload = torch.load(io.BytesIO(content), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("trusted_local_only") is not True:
        raise ValueError("Resume state is not marked trusted-local-only.")
    optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        if payload.get("scheduler") is None:
            raise ValueError("Resume state does not contain scheduler state.")
        scheduler.load_state_dict(payload["scheduler"])
    torch.set_rng_state(payload["torch_rng"])
    if torch.cuda.is_available() and payload.get("cuda_rng"):
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    np.random.set_state(payload["numpy_rng"])
    random.setstate(payload["python_rng"])
    sampler_epoch_value = payload.get("sampler_epoch")
    if sampler_epoch_value is None:
        sampler_epoch_value = payload["epoch"]
    return ResumeProgress(
        epoch=int(payload["epoch"]),
        global_step=int(payload.get("global_step", 0)),
        sampler_epoch=int(sampler_epoch_value),
        rdm_counter=int(payload.get("rdm_counter", 0)),
        batch_index=int(payload.get("batch_index", 0)),
        trainer_state=payload.get("trainer_state"),
    )


def _validate_checkpoint_compatibility(
    manifest: CheckpointManifest, expected: CheckpointCompatibility
) -> None:
    exact_fields = (
        "geometry",
        "predictor_family",
        "fold_id",
        "cutoff",
        "universe_manifest_hash",
        "dataset_manifest_hash",
        "sequence_manifest_hash",
        "normalization_hash",
        "architecture_hash",
        "training_config_hash",
        "paper_config_hash",
        "rdm_projections",
        "adaptation",
    )
    mismatches = [
        name for name in exact_fields if getattr(manifest, name) != getattr(expected, name)
    ]
    float_fields = (
        "generalized_gaussian_p",
        "generalized_gaussian_mu",
        "generalized_gaussian_sigma",
        "target_rms",
        "target_zero_fraction",
        "calibrated_rdm_lambda",
    )
    mismatches.extend(
        name
        for name in float_fields
        if not np.isclose(getattr(manifest, name), getattr(expected, name), rtol=0, atol=1e-12)
    )
    if expected.torch_compatibility == "exact":
        torch_matches = manifest.torch_version == expected.torch_version
    else:
        torch_matches = _major_minor(manifest.torch_version) == _major_minor(expected.torch_version)
    if not torch_matches:
        mismatches.append("torch_version")
    if mismatches:
        raise ValueError(f"Checkpoint compatibility mismatch: {sorted(mismatches)}")


def _major_minor(value: str) -> tuple[int, int]:
    core = value.split("+", maxsplit=1)[0]
    parts = core.split(".")
    try:
        return int(parts[0]), int(parts[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid PyTorch version in checkpoint identity: {value}") from exc
