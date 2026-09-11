"""Device-aware streaming JEPA training over fold-safe sequence manifests."""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Iterator, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from execsim.data.paper.manifests import file_sha256, stable_hash, write_json_atomic
from execsim.ml.representations.checkpoints import (
    load_checkpoint,
    load_trusted_resume_state,
    save_checkpoint,
    save_trusted_resume_state,
)
from execsim.ml.representations.diagnostics import (
    representation_diagnostics,
    sparse_acceptance,
)
from execsim.ml.representations.schemas import (
    CheckpointCompatibility,
    CheckpointManifest,
    RepresentationConfig,
)
from execsim.ml.sequences.streaming import PaperSequenceDataset, build_sequence_dataloader


@dataclass(frozen=True, slots=True)
class HistoricalTrainingIdentity:
    """Bind representation training to all upstream paper identities."""

    fold_id: str
    cutoff: str
    universe_manifest_hash: str
    dataset_manifest_hash: str
    normalization_hash: str
    architecture_hash: str
    config_hash: str
    code_commit: str


@dataclass(frozen=True, slots=True)
class HistoricalTrainerOptions:
    """Configure bounded I/O and real device execution without changing paper defaults."""

    batch_size: int = 256
    num_workers: int = 0
    prefetch_factor: int = 2
    cache_size: int = 32
    device: str = "auto"
    max_epochs: int = 40
    patience: int = 6
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_fraction: float = 0.05
    gradient_clip: float = 1.0
    use_bfloat16: bool = True
    checkpoint_interval_steps: int = 500
    diagnostic_sample_rows: int = 2048


@dataclass(frozen=True, slots=True)
class HistoricalTrainingResult:
    """Report software execution and selected validation checkpoint evidence."""

    calibrated_rdm_lambda: float
    epochs: int
    global_steps: int
    best_validation_loss: float
    best_epoch: int
    device: str
    precision: str
    checkpoint_root: str
    diagnostics: tuple[tuple[str, float], ...]


CollapseGate = Callable[[dict[str, float]], tuple[str, ...]]


class HistoricalTrainingInterrupted(RuntimeError):
    """Signal a deliberate, test-only interruption after a durable periodic checkpoint."""


class HistoricalTrainingRejected(RuntimeError):
    """Signal that no validation checkpoint satisfied the frozen collapse gate."""


def train_historical_representation(
    manifest_path: Path,
    *,
    representation: RepresentationConfig,
    identity: HistoricalTrainingIdentity,
    output_root: Path,
    allow_historical_training: bool,
    rdm_lambda: float,
    options: HistoricalTrainerOptions | None = None,
    collapse_gate: CollapseGate | None = None,
    resume_from: Path | None = None,
    trusted_resume: bool = False,
    interrupt_after_steps: int | None = None,
) -> HistoricalTrainingResult:
    """Train from streaming loaders after explicit historical-training authorization."""
    if not allow_historical_training:
        raise PermissionError("Historical representation training is disabled.")
    import torch

    from execsim.ml.representations.jepa import PredictiveRepresentationModel

    options = options or HistoricalTrainerOptions()
    if interrupt_after_steps is not None and interrupt_after_steps <= 0:
        raise ValueError("The deliberate interruption step must be positive.")
    if (
        options.max_epochs > 40
        or options.patience != 6
        or options.warmup_fraction != 0.05
        or options.diagnostic_sample_rows <= 0
    ):
        raise ValueError(
            "Historical trainer contradicts the locked epoch/patience/warmup protocol."
        )
    device = _resolve_device(torch, options.device)
    _seed_everything(torch, representation.seed)
    train_data = PaperSequenceDataset(
        manifest_path,
        partition="train",
        seed=representation.seed,
        cache_size=options.cache_size,
    )
    validation_data = PaperSequenceDataset(
        manifest_path,
        partition="validation",
        seed=representation.seed,
        cache_size=options.cache_size,
    )
    model = PredictiveRepresentationModel(representation).to(device)
    if rdm_lambda not in {0.1, 1.0, 10.0}:
        raise ValueError("Historical RDM coefficient must be one predeclared common candidate.")
    calibrated_lambda = rdm_lambda
    train_data.set_epoch(0)
    batches_per_epoch = max(1, math.ceil(len(train_data) / options.batch_size))
    total_steps = options.max_epochs * batches_per_epoch
    warmup_steps = max(1, math.ceil(total_steps * options.warmup_fraction))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _warmup_cosine(step, warmup_steps, total_steps)
    )
    sequence_hash = file_sha256(manifest_path)
    training_hash = stable_hash(
        {
            "representation": asdict(representation),
            "trainer": asdict(options),
            "common_rdm_lambda": calibrated_lambda,
        }
    )
    p, mu, sigma = representation.target_parameters
    compatibility = CheckpointCompatibility(
        geometry=representation.geometry,
        predictor_family=representation.predictor_family,
        fold_id=identity.fold_id,
        cutoff=identity.cutoff,
        universe_manifest_hash=identity.universe_manifest_hash,
        dataset_manifest_hash=identity.dataset_manifest_hash,
        sequence_manifest_hash=sequence_hash,
        normalization_hash=identity.normalization_hash,
        architecture_hash=identity.architecture_hash,
        training_config_hash=training_hash,
        paper_config_hash=identity.config_hash,
        generalized_gaussian_p=p,
        generalized_gaussian_mu=mu,
        generalized_gaussian_sigma=sigma,
        target_rms=representation.target_rms,
        target_zero_fraction=representation.target_zero_fraction,
        rdm_projections=representation.rdm_projections_train,
        calibrated_rdm_lambda=calibrated_lambda,
        torch_version=torch.__version__,
    )
    start_epoch = 0
    resume_batch_index = 0
    global_step = 0
    best_loss = float("inf")
    best_epoch = -1
    best_state: dict[str, Any] | None = None
    latest_state: dict[str, Any] | None = None
    best_diagnostics: dict[str, float] = {}
    latest_validation_loss = float("inf")
    latest_diagnostics: dict[str, float] = {}
    latest_gate_failures: tuple[str, ...] = ()
    stale = 0
    if resume_from is not None:
        load_checkpoint(model, resume_from / "weights", expected=compatibility)
        checksum = (resume_from / "resume.sha256").read_text(encoding="utf-8").strip()
        progress = load_trusted_resume_state(
            resume_from / "resume.pt",
            optimizer,
            expected_sha256=checksum,
            trusted_local=trusted_resume,
            scheduler=scheduler,
        )
        start_epoch = progress.epoch
        resume_batch_index = progress.batch_index
        global_step = progress.global_step
        trainer_state = progress.trainer_state or {}
        best_loss = float(trainer_state.get("best_loss", float("inf")))
        best_epoch = int(trainer_state.get("best_epoch", -1))
        stale = int(trainer_state.get("stale", 0))
        best_diagnostics = {
            str(name): float(value)
            for name, value in dict(trainer_state.get("best_diagnostics", {})).items()
        }
        restored_best = trainer_state.get("best_state")
        if restored_best is not None:
            best_state = {
                str(name): tensor.detach().cpu().clone()
                for name, tensor in dict(restored_best).items()
            }
    precision = _precision_name(torch, device, options.use_bfloat16)
    gate = collapse_gate or (
        (
            lambda values: sparse_acceptance(
                values, target_zero_fraction=representation.target_zero_fraction
            )
        )
        if representation.geometry == "sparse"
        else _dense_gate
    )
    gate_name = "development-fixture-override" if collapse_gate is not None else "paper-required-v1"
    epochs_run = 0
    for epoch in range(start_epoch, options.max_epochs):
        epochs_run = epoch + 1
        train_data.set_epoch(epoch)
        model.train()
        for batch_index, batch in enumerate(_loader(train_data, options, device)):
            if epoch == start_epoch and batch_index < resume_batch_index:
                continue
            values = _to_device(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with _autocast(torch, device, precision):
                output = model.loss(
                    values["context"],
                    values["context_mask"],
                    values["targets"],
                    values["target_mask"],
                    rdm_lambda=calibrated_lambda,
                    rdm_seed=representation.seed + global_step,
                )
            loss = output["loss"]
            assert_finite_training_state(loss, model, gradients_required=False)
            loss.backward()
            assert_finite_training_state(loss, model, gradients_required=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), options.gradient_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1
            periodic_due = (
                options.checkpoint_interval_steps > 0
                and global_step % options.checkpoint_interval_steps == 0
            ) or global_step == interrupt_after_steps
            if periodic_due:
                _save_periodic_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    output_root=output_root,
                    config=representation,
                    identity=identity,
                    compatibility=compatibility,
                    sequence_hash=sequence_hash,
                    training_hash=training_hash,
                    calibrated_lambda=calibrated_lambda,
                    epoch=epoch,
                    batch_index=batch_index + 1,
                    global_step=global_step,
                    gate_name=gate_name,
                    trainer_state={
                        "best_loss": best_loss,
                        "best_epoch": best_epoch,
                        "best_state": best_state,
                        "best_diagnostics": best_diagnostics,
                        "stale": stale,
                    },
                )
            if global_step == interrupt_after_steps:
                raise HistoricalTrainingInterrupted(
                    f"Deliberate interruption after durable step {global_step}."
                )
        resume_batch_index = 0
        validation_loss, diagnostics = _validate(model, validation_data, options, device)
        failures = gate(diagnostics)
        latest_validation_loss = validation_loss
        latest_diagnostics = diagnostics
        latest_gate_failures = failures
        latest_state = _cpu_state(model)
        if not failures and validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = latest_state
            best_diagnostics = diagnostics
            stale = 0
        else:
            stale += 1
        if stale >= options.patience:
            break
    if latest_state is None:
        raise RuntimeError("Historical trainer executed no optimization steps.")
    if best_state is None:
        output_root.mkdir(parents=True, exist_ok=True)
        write_json_atomic(
            output_root / "training-failure.json",
            {
                "schema_version": "historical-training-rejection-v1",
                "status": "REJECTED_BY_COLLAPSE_GATE",
                "fold_id": identity.fold_id,
                "geometry": representation.geometry,
                "seed": representation.seed,
                "rdm_lambda": calibrated_lambda,
                "paper_config_hash": identity.config_hash,
                "sequence_manifest_hash": sequence_hash,
                "training_config_hash": training_hash,
                "code_commit": identity.code_commit,
                "epochs": epochs_run,
                "global_steps": global_step,
                "collapse_gate_name": gate_name,
                "collapse_gate_failures": latest_gate_failures,
                "latest_validation_loss": latest_validation_loss,
                "latest_diagnostics": latest_diagnostics,
            },
        )
        raise HistoricalTrainingRejected(
            "No validation checkpoint passed the required collapse gates."
        )
    output_root.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_root / "compatibility.json", asdict(compatibility))
    latest_model = PredictiveRepresentationModel(representation)
    latest_model.load_state_dict(latest_state)
    best_model = PredictiveRepresentationModel(representation)
    best_model.load_state_dict(best_state)
    checkpoint_models: tuple[tuple[Literal["latest", "best", "final"], Any], ...] = (
        ("latest", latest_model),
        ("best", best_model),
        ("final", best_model),
    )
    for role, saved_model in checkpoint_models:
        manifest = _checkpoint_manifest(
            representation,
            identity,
            sequence_hash=sequence_hash,
            training_hash=training_hash,
            calibrated_lambda=calibrated_lambda,
            diagnostics=best_diagnostics,
            role=role,
            gate_name=gate_name,
        )
        save_checkpoint(saved_model, output_root / role, manifest)
    resume_hash = save_trusted_resume_state(
        output_root / "latest.resume.pt",
        optimizer,
        scheduler=scheduler,
        epoch=epochs_run,
        global_step=global_step,
        sampler_epoch=train_data.epoch,
        rdm_counter=global_step,
        batch_index=0,
        trainer_state={
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "best_state": best_state,
            "best_diagnostics": best_diagnostics,
            "stale": stale,
        },
    )
    (output_root / "latest.resume.sha256").write_text(resume_hash + "\n", encoding="utf-8")
    result = HistoricalTrainingResult(
        calibrated_lambda,
        epochs_run,
        global_step,
        best_loss,
        best_epoch,
        device,
        precision,
        str(output_root),
        tuple(sorted(best_diagnostics.items())),
    )
    write_json_atomic(output_root / "training-result.json", asdict(result))
    return result


def adapt_with_difficulty_loader(
    model: Any,
    loader_factory: Callable[[], Any],
    weights: Mapping[str, float],
    *,
    actual_base_training_steps: int,
    rdm_lambda: float,
    seed: int,
    device: str,
) -> int:
    """Adapt over complete weighted loader passes for exactly ceil(10% base steps)."""
    import torch

    steps = math.ceil(0.10 * actual_base_training_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
    iterator: Iterator[dict[str, Any]] = iter(loader_factory())
    for step in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader_factory())
            batch = next(iterator)
        values = _to_device(batch, device)
        sample_weights = torch.as_tensor(
            [weights[str(sample_id)] for sample_id in batch["sample_id"]],
            dtype=values["context"].dtype,
            device=device,
        )
        optimizer.zero_grad(set_to_none=True)
        output = model.loss(
            values["context"],
            values["context_mask"],
            values["targets"],
            values["target_mask"],
            rdm_lambda=rdm_lambda,
            rdm_seed=seed + step,
            sample_weights=sample_weights,
        )
        if not bool(torch.isfinite(output["loss"])):
            raise FloatingPointError("Non-finite difficulty-adaptation loss.")
        output["loss"].backward()
        if any(
            parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ):
            raise FloatingPointError("Non-finite difficulty-adaptation gradient.")
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return steps


def _loader(dataset: PaperSequenceDataset, options: HistoricalTrainerOptions, device: str) -> Any:
    return build_sequence_dataloader(
        dataset,
        batch_size=options.batch_size,
        num_workers=options.num_workers,
        device=device,
        prefetch_factor=options.prefetch_factor,
    )


def _take_exactly(loader: Any, count: int) -> Iterator[dict[str, Any]]:
    iterator = iter(loader)
    for _ in range(count):
        try:
            yield next(iterator)
        except StopIteration:
            iterator = iter(loader)
            try:
                yield next(iterator)
            except StopIteration as exc:
                raise ValueError("Training loader is empty during RDMReg calibration.") from exc


def _to_device(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {
        name: value.to(device, non_blocking=device.startswith("cuda"))
        if hasattr(value, "to")
        else value
        for name, value in batch.items()
    }


def _validate(
    model: Any,
    dataset: PaperSequenceDataset,
    options: HistoricalTrainerOptions,
    device: str,
) -> tuple[float, dict[str, float]]:
    import torch

    model.eval()
    losses = []
    sample = np.empty((0, model.config.latent_dim), dtype=np.float32)
    priorities = np.empty(0, dtype=np.float64)
    generator = np.random.default_rng(model.config.seed + 7_000_003)
    with torch.no_grad():
        for batch in _loader(dataset, options, device):
            values = _to_device(batch, device)
            output = model(
                values["context"],
                values["context_mask"],
                values["targets"],
                values["target_mask"],
            )
            losses.append(float(output["prediction_loss"]))
            mask = values["target_mask"].bool()
            rows = output["target"][mask].detach().float().cpu().numpy()
            sample, priorities = _update_diagnostic_sample(
                sample,
                priorities,
                rows,
                generator=generator,
                maximum_rows=options.diagnostic_sample_rows,
            )
    if not losses or len(sample) < 2:
        raise ValueError("Complete validation grid is empty.")
    diagnostics = representation_diagnostics(sample)
    from execsim.ml.representations.rdmreg import sliced_wasserstein_distance

    p, mu, sigma = model.config.target_parameters
    diagnostics["rdmreg_sliced_w2"] = float(
        sliced_wasserstein_distance(
            torch.from_numpy(sample),
            p=p,
            mu=mu,
            sigma=sigma,
            projections=model.config.rdm_projections_evaluation,
            seed=model.config.seed + 8_000_003,
            rectify_target=model.config.geometry == "sparse",
            target_rms=model.config.target_rms,
        )
    )
    diagnostics["rdmreg_diagnostic_rows"] = float(len(sample))
    diagnostics["rdmreg_diagnostic_projections"] = float(model.config.rdm_projections_evaluation)
    loss = float(np.mean(losses))
    if not np.isfinite(loss) or diagnostics["finite"] != 1.0:
        raise FloatingPointError("Validation produced non-finite loss or representation output.")
    return loss, diagnostics


def _update_diagnostic_sample(
    sample: np.ndarray,
    priorities: np.ndarray,
    rows: np.ndarray,
    *,
    generator: np.random.Generator,
    maximum_rows: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep a deterministic uniform priority sample of actual valid latents."""
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2 or values.shape[1:] != sample.shape[1:]:
        raise ValueError("Diagnostic latents do not match the representation dimension.")
    combined_values = np.concatenate((sample, values), axis=0)
    combined_priorities = np.concatenate((priorities, generator.random(len(values))), axis=0)
    if len(combined_values) <= maximum_rows:
        return combined_values, combined_priorities
    selected = np.argpartition(combined_priorities, maximum_rows - 1)[:maximum_rows]
    order = np.argsort(combined_priorities[selected], kind="stable")
    selected = selected[order]
    return combined_values[selected], combined_priorities[selected]


def _warmup_cosine(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _resolve_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if requested != "cpu" and not requested.startswith("cuda"):
        raise ValueError(f"Unsupported training device: {requested}")
    return requested


def _precision_name(torch: Any, device: str, use_bfloat16: bool) -> str:
    if device.startswith("cuda") and use_bfloat16 and torch.cuda.is_bf16_supported():
        return "bf16"
    return "fp32"


def _autocast(torch: Any, device: str, precision: str) -> Any:
    return torch.autocast(
        device_type="cuda" if device.startswith("cuda") else "cpu",
        dtype=torch.bfloat16,
        enabled=precision == "bf16",
    )


def _seed_everything(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _cpu_state(model: Any) -> dict[str, Any]:
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _save_periodic_checkpoint(
    model: Any,
    optimizer: Any,
    scheduler: Any,
    *,
    output_root: Path,
    config: RepresentationConfig,
    identity: HistoricalTrainingIdentity,
    compatibility: CheckpointCompatibility,
    sequence_hash: str,
    training_hash: str,
    calibrated_lambda: float,
    epoch: int,
    batch_index: int,
    global_step: int,
    gate_name: str,
    trainer_state: dict[str, Any],
) -> None:
    """Persist safe periodic weights and exact trusted mid-epoch continuation state."""
    directory = output_root / "periodic" / f"step={global_step:09d}"
    directory.mkdir(parents=True, exist_ok=False)
    write_json_atomic(directory / "compatibility.json", asdict(compatibility))
    manifest = _checkpoint_manifest(
        config,
        identity,
        sequence_hash=sequence_hash,
        training_hash=training_hash,
        calibrated_lambda=calibrated_lambda,
        diagnostics={},
        role="latest",
        gate_name=gate_name,
        gate_status="NOT RUN",
    )
    save_checkpoint(model, directory / "weights", manifest)
    checksum = save_trusted_resume_state(
        directory / "resume.pt",
        optimizer,
        scheduler=scheduler,
        epoch=epoch,
        batch_index=batch_index,
        global_step=global_step,
        sampler_epoch=epoch,
        rdm_counter=global_step,
        trainer_state=trainer_state,
    )
    (directory / "resume.sha256").write_text(checksum + "\n", encoding="utf-8")


def _dense_gate(diagnostics: dict[str, float]) -> tuple[str, ...]:
    failures = []
    if diagnostics["finite"] != 1.0:
        failures.append("latents contain non-finite values")
    if diagnostics["effective_rank"] <= 1.0:
        failures.append("dense representation collapsed to rank one")
    return tuple(failures)


def assert_finite_training_state(loss: Any, model: Any, *, gradients_required: bool) -> None:
    """Stop immediately when loss or any realized gradient is non-finite."""
    import torch

    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("Non-finite JEPA loss; training stopped immediately.")
    if gradients_required and any(
        parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    ):
        raise FloatingPointError("Non-finite JEPA gradient; training stopped immediately.")


def _checkpoint_manifest(
    config: RepresentationConfig,
    identity: HistoricalTrainingIdentity,
    *,
    sequence_hash: str,
    training_hash: str,
    calibrated_lambda: float,
    diagnostics: dict[str, float],
    role: Literal["latest", "best", "final"],
    gate_name: str,
    gate_status: Literal["PASS", "FAIL", "NOT RUN", "BLOCKED"] = "PASS",
) -> CheckpointManifest:
    import torch

    p, mu, sigma = config.target_parameters
    target_mean, target_second = config.target_positive_moments
    return CheckpointManifest(
        checkpoint_id=f"{identity.fold_id}-{config.geometry}-{config.seed}-{role}",
        geometry=config.geometry,
        predictor_family=config.predictor_family,
        fold_id=identity.fold_id,
        seed=config.seed,
        sequence_manifest_hash=sequence_hash,
        normalization_hash=identity.normalization_hash,
        cutoff=identity.cutoff,
        architecture_hash=identity.architecture_hash,
        torch_version=torch.__version__,
        weights_sha256="",
        checkpoint_role=role,
        dataset_manifest_hash=identity.dataset_manifest_hash,
        universe_manifest_hash=identity.universe_manifest_hash,
        link="relu" if config.geometry == "sparse" else "identity",
        generalized_gaussian_p=p,
        generalized_gaussian_mu=mu,
        generalized_gaussian_sigma=sigma,
        target_rms=config.target_rms,
        target_positive_mean=target_mean,
        target_positive_second_moment=target_second,
        target_zero_fraction=config.target_zero_fraction,
        rdm_projections=config.rdm_projections_train,
        calibrated_rdm_lambda=calibrated_lambda,
        cuda_version=torch.version.cuda,
        cudnn_version=str(torch.backends.cudnn.version())
        if torch.backends.cudnn.is_available()
        else None,
        code_commit=identity.code_commit,
        training_config_hash=training_hash,
        paper_config_hash=identity.config_hash,
        validation_diagnostics=tuple(sorted(diagnostics.items())),
        collapse_gate_status=gate_status,
        collapse_gate_name=gate_name,
    )
