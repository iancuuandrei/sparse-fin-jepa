"""Validation-only selection rules for the shared RDMReg coefficient."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from execsim.data.paper.manifests import write_json_atomic


@dataclass(frozen=True, slots=True)
class CommonLambdaCandidate:
    """Record one geometry's fixed-observable validation result."""

    rdm_lambda: float
    geometry: str
    fold_id: str
    seed: int
    observable_probe_error: float
    collapse_gate_status: str
    checkpoint_hash: str
    failure_reason: str = ""


def select_common_rdm_lambda(
    candidates: tuple[CommonLambdaCandidate, ...],
    *,
    output: Path | None = None,
    paper_config_hash: str = "",
) -> float:
    """Select one coefficient by mean dense/sparse Fold 1 observable error."""
    expected_values = {0.1, 1.0, 10.0}
    coordinates = {(item.rdm_lambda, item.geometry) for item in candidates}
    expected_coordinates = {
        (value, geometry) for value in expected_values for geometry in ("dense", "sparse")
    }
    if coordinates != expected_coordinates or len(candidates) != 6:
        raise ValueError("Common-lambda receipt requires the complete six-run candidate matrix.")
    if any(item.fold_id != "fold-1" or item.seed != 13 for item in candidates):
        raise ValueError("Common-lambda selection is locked to Fold 1 and seed 13.")
    if any(item.collapse_gate_status not in {"PASS", "FAIL"} for item in candidates):
        raise ValueError("Common-lambda candidates require explicit PASS or FAIL gate status.")
    eligible = []
    for value in sorted(expected_values):
        pair = [item for item in candidates if item.rdm_lambda == value]
        if all(item.collapse_gate_status == "PASS" for item in pair):
            if any(
                not math.isfinite(item.observable_probe_error)
                or item.observable_probe_error < 0
                or not item.checkpoint_hash
                for item in pair
            ):
                raise ValueError("Gate-passing candidates require finite errors and checkpoints.")
            eligible.append((sum(item.observable_probe_error for item in pair) / 2.0, value))
    if not eligible:
        raise RuntimeError("No common RDM coefficient passed both geometry collapse gates.")
    selected = min(eligible, key=lambda item: (item[0], item[1]))[1]
    if output is not None:
        write_json_atomic(
            output,
            {
                "schema_version": "paper-rdm-lambda-selection-v1",
                "selection_partition": "fold-1/validation",
                "seed": 13,
                "criterion": "mean_fixed_observable_probe_error_across_geometries",
                "selected_rdm_lambda": selected,
                "paper_config_hash": paper_config_hash,
                "candidates": [asdict(item) for item in candidates],
                "test_or_tca_used": False,
            },
        )
    return selected


def streaming_observable_probe_error(
    model: Any,
    training_loader: Any,
    validation_loader: Any,
    *,
    device: str,
    ridge_alpha: float = 1.0,
) -> float:
    """Fit bounded ridge sufficient statistics and score future-volume surprise."""
    import numpy as np
    import torch
    from scipy.linalg import cho_factor, cho_solve

    from execsim.ml.sequences.schemas import CONTEXT_LENGTH

    dimension = CONTEXT_LENGTH * model.config.latent_dim + CONTEXT_LENGTH + 1
    gram = [np.zeros((dimension, dimension), dtype=np.float64) for _ in range(4)]
    cross = [np.zeros(dimension, dtype=np.float64) for _ in range(4)]
    counts = [0, 0, 0, 0]
    model.eval()
    with torch.no_grad():
        for batch in training_loader:
            features, target, mask = _observable_batch(model, batch, device)
            for horizon in range(4):
                selected = mask[:, horizon]
                if not selected.any():
                    continue
                x = np.column_stack((features[selected], np.ones(int(selected.sum()))))
                y = target[selected, horizon]
                gram[horizon] += x.T @ x
                cross[horizon] += x.T @ y
                counts[horizon] += len(x)
    coefficients = []
    penalty = np.eye(dimension) * ridge_alpha
    penalty[-1, -1] = 0.0
    factor = cho_factor(gram[0] + penalty, lower=True, check_finite=False)
    for horizon in range(4):
        if counts[horizon] == 0:
            raise ValueError("Observable probe has no TRAIN rows for a required horizon.")
        if horizon:
            factor = cho_factor(gram[horizon] + penalty, lower=True, check_finite=False)
        coefficients.append(cho_solve(factor, cross[horizon], check_finite=False))
    total_error = 0.0
    total_count = 0
    with torch.no_grad():
        for batch in validation_loader:
            features, target, mask = _observable_batch(model, batch, device)
            for horizon in range(4):
                selected = mask[:, horizon]
                if not selected.any():
                    continue
                x = np.column_stack((features[selected], np.ones(int(selected.sum()))))
                residual = x @ coefficients[horizon] - target[selected, horizon]
                total_error += float(np.square(residual).sum())
                total_count += len(residual)
    if total_count == 0:
        raise ValueError("Observable probe has no validation rows.")
    return total_error / total_count


def _observable_batch(model: Any, batch: dict[str, Any], device: str) -> tuple[Any, Any, Any]:
    import numpy as np

    from execsim.ml.sequences.schemas import ENCODER_FEATURE_INDICES

    values = {
        name: value.to(device) if hasattr(value, "to") else value for name, value in batch.items()
    }
    _, linked = model.encode(values["context"][..., ENCODER_FEATURE_INDICES])
    linked = linked * values["context_mask"].unsqueeze(-1).to(linked.dtype)
    features = np.concatenate(
        (
            linked.detach().float().cpu().numpy().reshape(len(linked), -1),
            values["context_mask"].detach().float().cpu().numpy(),
        ),
        axis=1,
    )
    raw = values["raw_target_volume"].detach().float().cpu().numpy()
    expected = values["causal_target_volume"].detach().float().cpu().numpy()
    target = np.log1p(raw) - np.log1p(expected)
    mask = values["target_mask"].detach().cpu().numpy().astype(bool)
    return features, target, mask
