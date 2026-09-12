"""Streaming frozen-representation capacity and information-retention evaluation."""

from __future__ import annotations

import hashlib
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from execsim.ml.representations.evaluation import complete_horizon_origin_mask
from execsim.ml.sequences.schemas import ENCODER_FEATURE_INDICES, HORIZONS


@dataclass(frozen=True, slots=True)
class FrozenProbeOptions:
    """Bounded, deterministic settings for the predeclared capacity ladder."""

    ridge_alphas: tuple[float, ...] = (0.1, 1.0, 10.0)
    mlp_epochs: int = 20
    learning_rate: float = 1e-3


def evaluate_frozen_capacity_streaming(
    model: Any,
    training_loader: Iterable[dict[str, Any]],
    validation_loader: Iterable[dict[str, Any]],
    test_loader: Iterable[dict[str, Any]],
    *,
    device: str,
    seed: int,
    options: FrozenProbeOptions | None = None,
    cache_root: Path | None = None,
    cache_identity: dict[str, Any] | None = None,
) -> tuple[
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
]:
    """Fit horizon-specific probes without loading the historical corpus into RAM."""
    import torch
    from torch import nn

    options = options or FrozenProbeOptions()
    if not options.ridge_alphas or min(options.ridge_alphas) <= 0 or options.mlp_epochs <= 0:
        raise ValueError("Frozen-probe settings must be positive and non-empty.")
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if cache_root is not None:
        from execsim.ml.representations.probe_cache import (
            encoded_probe_identity,
            materialize_probe_batches,
        )

        if not cache_identity:
            raise ValueError("Encoded probe caches require an explicit immutable identity.")
        cached = [
            materialize_probe_batches(
                cache_root / partition,
                identity=encoded_probe_identity(cache_identity, partition=partition, device=device),
                loader=loader,
                encode=lambda batch: _encoded_batch(model, batch, device),
            )
            for partition, loader in zip(
                ("train", "validation", "test"),
                (training_loader, validation_loader, test_loader),
                strict=True,
            )
        ]
        training_loader, validation_loader, test_loader = cached

    feature_dim = 8 * model.config.latent_dim + 8
    target_dim = model.config.latent_dim
    train_stats = _target_statistics(model, training_loader, device)
    ridge_stats = _ridge_statistics(model, training_loader, device, feature_dim, target_dim)
    ridge_models = _select_ridge_models(
        ridge_stats,
        model,
        validation_loader,
        device,
        options.ridge_alphas,
        train_stats,
    )
    observable_models = _select_observable_ridge_models(
        ridge_stats,
        model,
        validation_loader,
        device,
        options.ridge_alphas,
    )

    probes: dict[str, Any] = {"affine_ridge": ridge_models}
    observable_probes: dict[str, Any] = {"affine_ridge": observable_models}
    for hidden in (64, 256):
        with torch.random.fork_rng():
            torch.manual_seed(seed + hidden)
            networks = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(feature_dim, hidden),
                        nn.GELU(),
                        nn.Linear(hidden, target_dim),
                    )
                    for _ in HORIZONS
                ]
            ).to(device)
            torch.manual_seed(seed + hidden + 1_000)
            observable_networks = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(feature_dim, hidden),
                        nn.GELU(),
                        nn.Linear(hidden, 1),
                    )
                    for _ in HORIZONS
                ]
            ).to(device)
        optimizer = torch.optim.AdamW(networks.parameters(), lr=options.learning_rate)
        observable_optimizer = torch.optim.AdamW(
            observable_networks.parameters(), lr=options.learning_rate
        )
        for _ in range(options.mlp_epochs):
            networks.train()
            observable_networks.train()
            for batch in training_loader:
                features, targets, observable, complete = _encoded_batch(model, batch, device)
                if not bool(complete.any()):
                    continue
                optimizer.zero_grad(set_to_none=True)
                observable_optimizer.zero_grad(set_to_none=True)
                selected_x = features[complete]
                selected_y = targets[complete]
                selected_observable = observable[complete]
                losses = [
                    torch.mean((networks[index](selected_x) - selected_y[:, index]) ** 2)
                    for index in range(len(HORIZONS))
                ]
                observable_losses = [
                    torch.mean(
                        (
                            observable_networks[index](selected_x).squeeze(-1)
                            - selected_observable[:, index]
                        )
                        ** 2
                    )
                    for index in range(len(HORIZONS))
                ]
                loss = torch.stack(losses).mean()
                observable_loss = torch.stack(observable_losses).mean()
                if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(observable_loss)):
                    raise FloatingPointError("Frozen MLP probe produced non-finite loss.")
                loss.backward()
                observable_loss.backward()
                optimizer.step()
                observable_optimizer.step()
        probes[f"mlp_{hidden}"] = networks
        observable_probes[f"mlp_{hidden}"] = observable_networks

    rows: list[dict[str, float | int | str]] = []
    observable_rows: list[dict[str, float | int | str]] = []
    date_rows: dict[tuple[str, str, int], dict[str, float | int | str]] = {}
    date_identities = _date_identity_statistics(test_loader)
    for capacity, probe in probes.items():
        summary, dated = _score_probe(
            model,
            probe,
            test_loader,
            device=device,
            capacity=capacity,
            train_stats=train_stats,
            feature_dim=feature_dim,
            target_dim=target_dim,
        )
        rows.extend(summary)
        for item in dated:
            key = (capacity, str(item["date"]), int(item["horizon"]))
            date_rows[key] = item
    for capacity, probe in observable_probes.items():
        summary, dated = _score_observable(
            model,
            probe,
            test_loader,
            device=device,
            capacity=capacity,
            feature_dim=feature_dim,
        )
        observable_rows.extend(summary)
        for item in dated:
            key = (capacity, str(item["date"]), int(item["horizon"]))
            existing = date_rows.get(key)
            if existing is None or int(existing["row_count"]) != int(item["row_count"]):
                raise ValueError("Latent and observable date metrics do not share exact rows.")
            existing.update(item)
    for item in date_rows.values():
        identity = date_identities.get(str(item["date"]))
        if identity is None or int(identity["row_count"]) != int(item["row_count"]):
            raise ValueError("Date metric rows contradict the TEST sample identity ledger.")
        item.update(identity)
    return rows, observable_rows, list(date_rows.values())


def _date_identity_statistics(
    loader: Iterable[dict[str, Any]],
) -> dict[str, dict[str, int | str]]:
    """Hash exact complete-origin TEST sample identities within each date."""
    digests: dict[str, Any] = {}
    counts: dict[str, int] = {}
    seen: set[str] = set()
    for batch in loader:
        as_of = np.asarray(batch["as_of_token"], dtype=int)
        complete = complete_horizon_origin_mask(as_of)
        dates = np.asarray(batch["session_date"], dtype=str)[complete]
        sample_ids = np.asarray(batch["sample_id"], dtype=str)[complete]
        for date_value, sample_id in zip(dates, sample_ids, strict=True):
            if sample_id in seen:
                raise ValueError("Frozen TEST probe loader contains a duplicate sample ID.")
            seen.add(sample_id)
            digest = digests.setdefault(date_value, hashlib.sha256())
            encoded = sample_id.encode("utf-8")
            digest.update(len(encoded).to_bytes(4, "big"))
            digest.update(encoded)
            counts[date_value] = counts.get(date_value, 0) + 1
    if not counts:
        raise ValueError("Frozen TEST probe loader has no complete origins.")
    return {
        date_value: {
            "sample_identity_sha256": digests[date_value].hexdigest(),
            "row_count": count,
        }
        for date_value, count in sorted(counts.items())
    }


def _encoded_batch(model: Any, batch: dict[str, Any], device: str) -> tuple[Any, Any, Any, Any]:
    import torch

    if "encoded_probe" in batch:
        features, targets, observable, complete = batch["encoded_probe"]
        return features.to(device), targets.to(device), observable.to(device), complete.to(device)

    context = batch["context"].to(device)
    context_mask = batch["context_mask"].to(device)
    targets = batch["targets"].to(device)
    with torch.no_grad():
        _, linked_context = model.encode(context[..., ENCODER_FEATURE_INDICES])
        linked_context *= context_mask.unsqueeze(-1).to(linked_context.dtype)
        _, linked_targets = model.encode(targets[..., ENCODER_FEATURE_INDICES])
    features = torch.cat(
        (linked_context.flatten(1), context_mask.to(linked_context.dtype)), dim=1
    ).float()
    as_of = np.asarray(batch["as_of_token"], dtype=int)
    complete = torch.as_tensor(complete_horizon_origin_mask(as_of), dtype=torch.bool, device=device)
    raw = batch["raw_target_volume"].to(device).float()
    baseline = batch["causal_target_volume"].to(device).float()
    observable = torch.log1p(raw) - torch.log1p(baseline)
    return features, linked_targets.float(), observable, complete


def _target_statistics(model: Any, loader: Iterable[dict[str, Any]], device: str) -> dict[str, Any]:
    sums = np.zeros((4, model.config.latent_dim), dtype=np.float64)
    squares = np.zeros_like(sums)
    counts = np.zeros(4, dtype=np.int64)
    for batch in loader:
        _, targets, _, complete = _encoded_batch(model, batch, device)
        values = targets[complete].cpu().numpy().astype(np.float64)
        for index in range(4):
            sums[index] += values[:, index].sum(axis=0)
            squares[index] += np.square(values[:, index]).sum(axis=0)
            counts[index] += len(values)
    if (counts < 2).any():
        raise ValueError("Frozen evaluation requires two complete TRAIN origins per horizon.")
    means = sums / counts[:, None]
    traces = (squares - counts[:, None] * np.square(means)).sum(axis=1) / (counts - 1)
    if not np.isfinite(traces).all() or (traces <= 0).any():
        raise ValueError("TRAIN latent covariance traces must be finite and positive.")
    return {"means": means, "traces": traces, "counts": counts}


def _ridge_statistics(
    model: Any,
    loader: Iterable[dict[str, Any]],
    device: str,
    feature_dim: int,
    target_dim: int,
) -> dict[str, Any]:
    gram = np.zeros((feature_dim, feature_dim), dtype=np.float64)
    latent_cross = np.zeros((4, feature_dim, target_dim), dtype=np.float64)
    observable_cross = np.zeros((4, feature_dim), dtype=np.float64)
    feature_sum = np.zeros(feature_dim, dtype=np.float64)
    latent_sum = np.zeros((4, target_dim), dtype=np.float64)
    observable_sum = np.zeros(4, dtype=np.float64)
    retained: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = []
    rows = 0
    for batch in loader:
        features, targets, observable, complete = _encoded_batch(model, batch, device)
        if not bool(complete.any()):
            continue
        x = features[complete].cpu().numpy().astype(np.float64)
        target = targets[complete].cpu().numpy().astype(np.float64)
        observed = observable[complete].cpu().numpy().astype(np.float64)
        gram += x.T @ x
        feature_sum += x.sum(axis=0)
        for index in range(4):
            latent_cross[index] += x.T @ target[:, index]
            observable_cross[index] += x.T @ observed[:, index]
            latent_sum[index] += target[:, index].sum(axis=0)
            observable_sum[index] += observed[:, index].sum()
        rows += len(x)
        if retained is not None:
            retained.append((x, target, observed))
            if rows > 4096:
                retained = None
    if rows == 0:
        raise ValueError("Frozen ridge probe found no complete TRAIN origins.")
    return {
        "gram": gram,
        "latent_cross": latent_cross,
        "observable_cross": observable_cross,
        "feature_sum": feature_sum,
        "latent_sum": latent_sum,
        "observable_sum": observable_sum,
        "rows": rows,
        "retained": retained,
    }


def _ridge_coefficients(stats: dict[str, Any], alpha: float) -> np.ndarray:
    from scipy.linalg import cho_factor, cho_solve

    rows = int(stats["rows"])
    mean_x = stats["feature_sum"] / rows
    coefficients = []
    retained = stats["retained"]
    if retained is not None:
        x = np.concatenate([item[0] for item in retained])
        targets = np.concatenate([item[1] for item in retained])
        centered_x = x - mean_x
        dual = centered_x @ centered_x.T + alpha * np.eye(rows)
        factor = cho_factor(dual, lower=True, check_finite=False)
    else:
        centered_gram = stats["gram"] - rows * np.outer(mean_x, mean_x)
        matrix = centered_gram + alpha * np.eye(len(mean_x))
        factor = cho_factor(matrix, lower=True, check_finite=False)
    for index in range(4):
        mean_y = stats["latent_sum"][index] / rows
        if retained is not None:
            centered_y = targets[:, index] - mean_y
            slope = centered_x.T @ cho_solve(factor, centered_y, check_finite=False)
        else:
            centered_cross = stats["latent_cross"][index] - rows * np.outer(mean_x, mean_y)
            slope = cho_solve(factor, centered_cross, check_finite=False)
        intercept = mean_y - mean_x @ slope
        coefficients.append(np.vstack((slope, intercept)))
    return np.stack(coefficients)


def _select_ridge_models(
    stats: dict[str, Any],
    model: Any,
    validation_loader: Iterable[dict[str, Any]],
    device: str,
    alphas: tuple[float, ...],
    train_stats: dict[str, Any],
) -> np.ndarray:
    candidates = [(alpha, _ridge_coefficients(stats, alpha)) for alpha in alphas]
    scored = [
        (
            _score_ridge_nmse(model, coefficients, validation_loader, device, train_stats),
            alpha,
            coefficients,
        )
        for alpha, coefficients in candidates
    ]
    return min(scored, key=lambda value: (value[0], value[1]))[2]


def _score_ridge_nmse(
    model: Any,
    coefficients: np.ndarray,
    loader: Iterable[dict[str, Any]],
    device: str,
    train_stats: dict[str, Any],
) -> float:
    errors = np.zeros(4, dtype=np.float64)
    count = 0
    for batch in loader:
        features, targets, _, complete = _encoded_batch(model, batch, device)
        if not bool(complete.any()):
            continue
        x = np.column_stack((features[complete].cpu().numpy(), np.ones(int(complete.sum()))))
        actual = targets[complete].cpu().numpy()
        for index in range(4):
            errors[index] += np.square(x @ coefficients[index] - actual[:, index]).sum()
        count += len(x)
    if count == 0:
        raise ValueError("Frozen ridge validation has no complete origins.")
    return float(np.mean(errors / count / train_stats["traces"]))


def _fit_observable_ridge(stats: dict[str, Any], *, alpha: float) -> np.ndarray:
    from scipy.linalg import cho_factor, cho_solve

    rows = int(stats["rows"])
    mean_x = stats["feature_sum"] / rows
    retained = stats["retained"]
    if retained is not None:
        x = np.concatenate([item[0] for item in retained])
        observed = np.concatenate([item[2] for item in retained])
        centered_x = x - mean_x
        dual = centered_x @ centered_x.T + alpha * np.eye(rows)
        factor = cho_factor(dual, lower=True, check_finite=False)
    else:
        centered_gram = stats["gram"] - rows * np.outer(mean_x, mean_x)
        matrix = centered_gram + alpha * np.eye(len(mean_x))
        factor = cho_factor(matrix, lower=True, check_finite=False)
    coefficients = []
    for index in range(4):
        mean_y = stats["observable_sum"][index] / rows
        if retained is not None:
            slope = centered_x.T @ cho_solve(
                factor, observed[:, index] - mean_y, check_finite=False
            )
        else:
            centered_cross = stats["observable_cross"][index] - rows * mean_x * mean_y
            slope = cho_solve(factor, centered_cross, check_finite=False)
        coefficients.append(np.append(slope, mean_y - mean_x @ slope))
    return np.stack(coefficients)


def _select_observable_ridge_models(
    stats: dict[str, Any],
    model: Any,
    validation_loader: Iterable[dict[str, Any]],
    device: str,
    alphas: tuple[float, ...],
) -> np.ndarray:
    candidates = [(alpha, _fit_observable_ridge(stats, alpha=alpha)) for alpha in alphas]
    scored = [
        (
            _score_observable_ridge_mae(model, coefficients, validation_loader, device),
            alpha,
            coefficients,
        )
        for alpha, coefficients in candidates
    ]
    return min(scored, key=lambda value: (value[0], value[1]))[2]


def _score_observable_ridge_mae(
    model: Any,
    coefficients: np.ndarray,
    loader: Iterable[dict[str, Any]],
    device: str,
) -> float:
    absolute = np.zeros(4, dtype=np.float64)
    count = 0
    for batch in loader:
        features, _, observable, complete = _encoded_batch(model, batch, device)
        if not bool(complete.any()):
            continue
        x = np.column_stack((features[complete].cpu().numpy(), np.ones(int(complete.sum()))))
        actual = observable[complete].cpu().numpy()
        for index in range(4):
            absolute[index] += np.abs(x @ coefficients[index] - actual[:, index]).sum()
        count += len(x)
    if count == 0:
        raise ValueError("Observable ridge validation has no complete origins.")
    return float(np.mean(absolute / count))


def _score_probe(
    model: Any,
    probe: Any,
    loader: Iterable[dict[str, Any]],
    *,
    device: str,
    capacity: str,
    train_stats: dict[str, Any],
    feature_dim: int,
    target_dim: int,
) -> tuple[
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
]:
    import torch

    squared = np.zeros(4, dtype=np.float64)
    zero = np.zeros(4, dtype=np.float64)
    mean = np.zeros(4, dtype=np.float64)
    persistence = np.zeros(4, dtype=np.float64)
    dated_squared: dict[str, np.ndarray] = {}
    dated_counts: dict[str, int] = {}
    count = 0
    start = time.perf_counter()
    for batch in loader:
        features, targets, _, complete = _encoded_batch(model, batch, device)
        if not bool(complete.any()):
            continue
        x = features[complete]
        actual = targets[complete]
        current = x[:, -8 - target_dim : -8]
        if capacity == "affine_ridge":
            design = np.column_stack((x.cpu().numpy(), np.ones(len(x))))
            predicted = np.stack([design @ probe[index] for index in range(4)], axis=1)
        else:
            probe.eval()
            with torch.no_grad():
                predicted = torch.stack([network(x) for network in probe], dim=1).cpu().numpy()
        actual_np = actual.cpu().numpy()
        current_np = current.cpu().numpy()
        row_squared = np.square(predicted - actual_np).sum(axis=2)
        for index in range(4):
            squared[index] += row_squared[:, index].sum()
            zero[index] += np.square(actual_np[:, index]).sum()
            mean[index] += np.square(actual_np[:, index] - train_stats["means"][index]).sum()
            persistence[index] += np.square(actual_np[:, index] - current_np).sum()
        selected_dates = np.asarray(batch["session_date"], dtype=str)[
            complete.detach().cpu().numpy()
        ]
        _accumulate_date_vectors(row_squared, selected_dates, dated_squared, dated_counts)
        count += len(x)
    elapsed = time.perf_counter() - start
    if count == 0:
        raise ValueError("Frozen probe test grid has no complete origins.")
    if capacity == "affine_ridge":
        parameters = 4 * (feature_dim + 1) * target_dim
        macs = 4 * feature_dim * target_dim
    else:
        parameters = sum(parameter.numel() for parameter in probe.parameters())
        hidden = 64 if capacity == "mlp_64" else 256
        macs = 4 * (feature_dim * hidden + hidden * target_dim)
    summary: list[dict[str, float | int | str]] = [
        {
            "horizon": int(horizon),
            "probe_capacity": capacity,
            "normalized_latent_error": float(squared[index] / count / train_stats["traces"][index]),
            "zero_baseline": float(zero[index] / count / train_stats["traces"][index]),
            "train_mean_baseline": float(mean[index] / count / train_stats["traces"][index]),
            "persistence_baseline": float(
                persistence[index] / count / train_stats["traces"][index]
            ),
            "parameter_count": int(parameters),
            "approximate_macs": int(macs),
            "inference_seconds": float(elapsed),
            "test_rows": int(count),
        }
        for index, horizon in enumerate(HORIZONS)
    ]
    dated: list[dict[str, float | int | str]] = [
        {
            "date": date_value,
            "horizon": int(horizon),
            "probe_capacity": capacity,
            "normalized_latent_error": float(
                values[index] / dated_counts[date_value] / train_stats["traces"][index]
            ),
            "row_count": int(dated_counts[date_value]),
        }
        for date_value, values in sorted(dated_squared.items())
        for index, horizon in enumerate(HORIZONS)
    ]
    return summary, dated


def _score_observable(
    model: Any,
    probe: Any,
    loader: Iterable[dict[str, Any]],
    *,
    device: str,
    capacity: str,
    feature_dim: int,
) -> tuple[
    list[dict[str, float | int | str]],
    list[dict[str, float | int | str]],
]:
    import torch

    absolute = np.zeros(4, dtype=np.float64)
    squared = np.zeros(4, dtype=np.float64)
    dated_absolute: dict[str, np.ndarray] = {}
    dated_squared: dict[str, np.ndarray] = {}
    dated_counts: dict[str, int] = {}
    count = 0
    start = time.perf_counter()
    for batch in loader:
        features, _, observable, complete = _encoded_batch(model, batch, device)
        if not bool(complete.any()):
            continue
        selected_x = features[complete]
        actual = observable[complete].cpu().numpy()
        if capacity == "affine_ridge":
            design = np.column_stack((selected_x.cpu().numpy(), np.ones(int(complete.sum()))))
            predicted = np.stack([design @ probe[index] for index in range(4)], axis=1)
        else:
            probe.eval()
            with torch.no_grad():
                predicted = (
                    torch.stack([network(selected_x).squeeze(-1) for network in probe], dim=1)
                    .cpu()
                    .numpy()
                )
        residual = predicted - actual
        row_absolute = np.abs(residual)
        row_squared = np.square(residual)
        absolute += row_absolute.sum(axis=0)
        squared += row_squared.sum(axis=0)
        selected_dates = np.asarray(batch["session_date"], dtype=str)[
            complete.detach().cpu().numpy()
        ]
        _accumulate_date_vectors(row_absolute, selected_dates, dated_absolute, dated_counts)
        squared_counts: dict[str, int] = {}
        _accumulate_date_vectors(row_squared, selected_dates, dated_squared, squared_counts)
        count += len(selected_x)
    elapsed = time.perf_counter() - start
    if count == 0:
        raise ValueError("Observable probe test grid has no complete origins.")
    if capacity == "affine_ridge":
        parameters = 4 * (feature_dim + 1)
        macs = 4 * feature_dim
    else:
        parameters = sum(parameter.numel() for parameter in probe.parameters())
        hidden = 64 if capacity == "mlp_64" else 256
        macs = 4 * (feature_dim * hidden + hidden)
    summary: list[dict[str, float | int | str]] = [
        {
            "horizon": int(horizon),
            "probe_capacity": capacity,
            "observable_volume_probe_mae": float(absolute[index] / count),
            "observable_volume_probe_rmse": float(np.sqrt(squared[index] / count)),
            "observable_parameter_count": int(parameters),
            "observable_approximate_macs": int(macs),
            "observable_inference_seconds": float(elapsed),
            "observable_test_rows": int(count),
        }
        for index, horizon in enumerate(HORIZONS)
    ]
    dated: list[dict[str, float | int | str]] = [
        {
            "date": date_value,
            "horizon": int(horizon),
            "probe_capacity": capacity,
            "observable_volume_probe_mae": float(values[index] / dated_counts[date_value]),
            "observable_volume_probe_rmse": float(
                np.sqrt(dated_squared[date_value][index] / dated_counts[date_value])
            ),
            "row_count": int(dated_counts[date_value]),
        }
        for date_value, values in sorted(dated_absolute.items())
        for index, horizon in enumerate(HORIZONS)
    ]
    return summary, dated


def _accumulate_date_vectors(
    values: np.ndarray,
    dates: np.ndarray,
    totals: dict[str, np.ndarray],
    counts: dict[str, int],
) -> None:
    """Accumulate one finite metric vector per row into deterministic date groups."""
    if values.ndim != 2 or len(values) != len(dates) or not np.isfinite(values).all():
        raise ValueError("Date-level probe metrics must be finite aligned vectors.")
    for date_value in np.unique(dates):
        selected = values[dates == date_value]
        if date_value in totals:
            totals[date_value] += selected.sum(axis=0)
        else:
            totals[date_value] = selected.sum(axis=0)
        counts[date_value] = counts.get(date_value, 0) + len(selected)
