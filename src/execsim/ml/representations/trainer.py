"""Deterministic CPU-capable JEPA training for synthetic fixtures and authorized runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from execsim.ml.representations.rdmreg import calibrate_rdm_lambda
from execsim.ml.representations.schemas import RepresentationConfig


@dataclass(frozen=True, slots=True)
class RepresentationTrainingResult:
    """Return fixture training evidence without making model-quality claims."""

    initial_loss: float
    final_loss: float
    finite_gradients: bool
    zero_fraction: float
    steps: int


@dataclass(frozen=True, slots=True)
class RepresentationTrainingOptions:
    """Lock the primary optimizer, schedule, and early-stopping defaults."""

    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_fraction: float = 0.05
    gradient_clip: float = 1.0
    batch_size: int = 256
    max_epochs: int = 40
    patience: int = 6
    use_bfloat16: bool = True


def fit_representation_arrays(
    config: RepresentationConfig,
    training: dict[str, np.ndarray],
    validation: dict[str, np.ndarray],
    *,
    rdm_lambda: float,
    data_classification: str,
    allow_historical_training: bool = False,
    options: RepresentationTrainingOptions | None = None,
) -> tuple[Any, RepresentationTrainingResult]:
    """Fit authorized arrays with warmup, cosine decay, clipping, and validation stopping."""
    import math

    import torch

    from execsim.ml.representations.jepa import PredictiveRepresentationModel

    options = options or RepresentationTrainingOptions()
    if data_classification != "synthetic_fixture" and not allow_historical_training:
        raise PermissionError("Historical representation training is disabled.")
    if (
        not 0 <= options.warmup_fraction < 1
        or options.max_epochs <= 0
        or options.patience <= 0
        or options.batch_size <= 0
    ):
        raise ValueError("Representation training schedule is invalid.")
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    model = PredictiveRepresentationModel(config)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=options.learning_rate, weight_decay=options.weight_decay
    )
    train_tensors = _tensor_batch(training)
    validation_tensors = _tensor_batch(validation)
    batches_per_epoch = max(1, math.ceil(len(train_tensors[0]) / options.batch_size))
    total_steps = options.max_epochs * batches_per_epoch
    warmup_steps = max(1, int(total_steps * options.warmup_fraction))
    best_validation = float("inf")
    best_state = None
    stale_epochs = 0
    losses: list[float] = []
    finite = True
    last_target = None
    bfloat16_enabled = options.use_bfloat16 and _supports_cpu_bfloat16(torch)
    sampling_generator = torch.Generator().manual_seed(config.seed + 101)
    global_step = 0
    for _epoch in range(options.max_epochs):
        permutation = torch.randperm(len(train_tensors[0]), generator=sampling_generator)
        for start in range(0, len(permutation), options.batch_size):
            batch_indices = permutation[start : start + options.batch_size]
            if global_step < warmup_steps:
                multiplier = (global_step + 1) / warmup_steps
            else:
                progress = (global_step - warmup_steps) / max(total_steps - warmup_steps - 1, 1)
                multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
            for group in optimizer.param_groups:
                group["lr"] = options.learning_rate * multiplier
            optimizer.zero_grad(set_to_none=True)
            batch = tuple(tensor[batch_indices] for tensor in train_tensors)
            with torch.autocast(
                device_type="cpu",
                dtype=torch.bfloat16,
                enabled=bfloat16_enabled,
            ):
                output = model.loss(
                    *batch,
                    rdm_lambda=rdm_lambda,
                    rdm_seed=config.seed + global_step,
                )
            output["loss"].backward()
            finite = finite and all(
                parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
                for parameter in model.parameters()
            )
            torch.nn.utils.clip_grad_norm_(model.parameters(), options.gradient_clip)
            optimizer.step()
            losses.append(float(output["loss"].detach()))
            last_target = output["target"].detach()
            global_step += 1
        with torch.no_grad():
            validation_loss = float(model(*validation_tensors)["prediction_loss"])
        if validation_loss < best_validation - 1e-10:
            best_validation = validation_loss
            best_state = {
                name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= options.patience:
                break
    if best_state is None or last_target is None:
        raise RuntimeError("Representation training produced no validated epoch.")
    model.load_state_dict(best_state)
    model.eval()
    zero_fraction = float((last_target == 0).to(torch.float32).mean())
    return model, RepresentationTrainingResult(
        losses[0], losses[-1], finite, zero_fraction, len(losses)
    )


def train_synthetic_fixture(
    config: RepresentationConfig, *, steps: int = 8, batch_size: int = 12
) -> RepresentationTrainingResult:
    """Fit a tiny deterministic random fixture; never load historical data."""
    import torch

    from execsim.ml.representations.jepa import PredictiveRepresentationModel

    if steps <= 0 or batch_size < 2:
        raise ValueError("Synthetic training requires positive steps and batch_size >= 2.")
    torch.set_num_threads(1)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator().manual_seed(config.seed + 1)
    context = torch.randn((batch_size, 8, 18), generator=generator)
    mask = torch.ones((batch_size, 8), dtype=torch.bool)
    targets = 0.8 * context[:, -1:, :].repeat(1, 4, 1) + 0.05 * torch.randn(
        (batch_size, 4, 18), generator=generator
    )
    target_mask = torch.ones((batch_size, 4), dtype=torch.bool)
    model = PredictiveRepresentationModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    losses = []
    finite = True
    last_target = None
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model.loss(
            context,
            mask,
            targets,
            target_mask,
            rdm_lambda=0.01,
            rdm_projections=min(32, config.rdm_projections_train),
            rdm_seed=config.seed + 10_000 + step,
        )
        output["loss"].backward()
        finite = finite and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(output["loss"].detach()))
        last_target = output["target"].detach()
    if last_target is None:
        raise RuntimeError("Synthetic fixture did not execute a training step.")
    zero_fraction = float((last_target == 0).to(torch.float32).mean())
    return RepresentationTrainingResult(losses[0], losses[-1], finite, zero_fraction, steps)


def calibrate_from_training_batches(
    model: Any, batches: list[dict[str, Any]], *, seed: int
) -> float:
    """Calibrate RDMReg from exactly 32 deterministic training-only mini-batches."""
    import torch

    if len(batches) != 32:
        raise ValueError("RDMReg calibration requires exactly 32 training mini-batches.")
    prediction_norms: list[float] = []
    rdm_norms: list[float] = []
    for index, batch in enumerate(batches):
        output = model.loss(
            batch["context"],
            batch["context_mask"],
            batch["targets"],
            batch["target_mask"],
            rdm_lambda=1.0,
            rdm_seed=seed + index,
        )
        pre_link = (output["context_pre_link"], output["target_pre_link"])
        prediction_gradients = torch.autograd.grad(
            output["prediction_loss"], pre_link, retain_graph=True, allow_unused=True
        )
        rdm_gradients = torch.autograd.grad(
            output["rdm_loss"], pre_link, retain_graph=False, allow_unused=True
        )
        prediction_norms.append(_gradient_norm(prediction_gradients))
        rdm_norms.append(_gradient_norm(rdm_gradients))
    return calibrate_rdm_lambda(float(np.median(prediction_norms)), float(np.median(rdm_norms)))


def adapt_with_difficulty(
    model: Any,
    batch: dict[str, Any],
    weights: np.ndarray,
    *,
    primary_steps: int,
    rdm_lambda: float,
    seed: int,
) -> RepresentationTrainingResult:
    """Adapt a sparse checkpoint for 10% more steps at one-tenth the final learning rate."""
    import torch

    context = batch["context"]
    tensor_weights = torch.as_tensor(weights, dtype=context.dtype, device=context.device)
    if tensor_weights.shape != (len(context),):
        raise ValueError("Difficulty weights must match the training batch.")
    steps = max(1, int(primary_steps * 0.1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=1e-4)
    losses: list[float] = []
    finite = True
    output = None
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = model.loss(
            batch["context"],
            batch["context_mask"],
            batch["targets"],
            batch["target_mask"],
            rdm_lambda=rdm_lambda,
            rdm_seed=seed + step,
            sample_weights=tensor_weights,
        )
        output["loss"].backward()
        finite = finite and all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(output["loss"].detach()))
    if output is None:
        raise RuntimeError("Difficulty adaptation executed no steps.")
    zero_fraction = float((output["target"].detach() == 0).to(torch.float32).mean())
    return RepresentationTrainingResult(losses[0], losses[-1], finite, zero_fraction, steps)


def _gradient_norm(gradients: tuple[Any, ...]) -> float:
    squared = sum(
        float(gradient.detach().square().sum()) for gradient in gradients if gradient is not None
    )
    return float(np.sqrt(squared))


def _tensor_batch(values: dict[str, np.ndarray]) -> tuple[Any, Any, Any, Any]:
    import torch

    required = ("context", "context_mask", "targets", "target_mask")
    missing = set(required).difference(values)
    if missing:
        raise ValueError(f"Representation batch missing arrays: {sorted(missing)}")
    context = torch.as_tensor(values["context"], dtype=torch.float32)
    context_mask = torch.as_tensor(values["context_mask"], dtype=torch.bool)
    targets = torch.as_tensor(values["targets"], dtype=torch.float32)
    target_mask = torch.as_tensor(values["target_mask"], dtype=torch.bool)
    if context.ndim != 3 or context.shape[1:] != (8, 18) or len(context) < 2:
        raise ValueError("Representation context must have shape [batch, 8, 18].")
    if targets.shape != (len(context), 4, 18):
        raise ValueError("Representation targets must have shape [batch, 4, 18].")
    if context_mask.shape != (len(context), 8) or target_mask.shape != (len(context), 4):
        raise ValueError("Representation masks do not match their tensors.")
    return context, context_mask, targets, target_mask


def _supports_cpu_bfloat16(torch_module: Any) -> bool:
    try:
        with torch_module.autocast(device_type="cpu", dtype=torch_module.bfloat16):
            output = torch_module.ones((2, 2)) @ torch_module.ones((2, 2))
        return output.dtype == torch_module.bfloat16
    except RuntimeError:
        return False
