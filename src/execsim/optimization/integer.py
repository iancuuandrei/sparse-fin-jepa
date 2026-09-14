from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def project_to_integer_capacities(
    continuous: ArrayLike,
    capacities: ArrayLike,
    target_quantity: int,
    *,
    tolerance: float = 1e-7,
) -> NDArray[np.int64]:
    """Reconcile sanitized quantities with deterministic integer capacities.

    Solver feasibility must be checked before calling this boundary. ``tolerance``
    is the independent integer epsilon, never a solver-relative error budget.
    """

    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Integer tolerance must be finite and non-negative.")
    values = np.asarray(continuous, dtype=float)
    caps_float = np.asarray(capacities, dtype=float)
    if values.ndim != 1 or caps_float.ndim != 1 or values.shape != caps_float.shape:
        raise ValueError("continuous and capacities must be equal-length vectors.")
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(caps_float)):
        raise ValueError("continuous and capacities must be finite.")
    if np.any(caps_float < 0) or np.any(np.abs(caps_float - np.floor(caps_float)) > tolerance):
        raise ValueError("capacities must be non-negative integers.")
    if isinstance(target_quantity, bool) or not isinstance(target_quantity, int):
        raise TypeError("target_quantity must be an integer.")
    caps = caps_float.astype(np.int64)
    if target_quantity < 0 or target_quantity > int(caps.sum()):
        raise ValueError("target_quantity must be feasible under capacities.")
    if np.any(values < -tolerance) or np.any(values > caps + tolerance):
        raise ValueError("continuous solution violates capacity bounds.")

    clipped = np.clip(values, 0.0, caps.astype(float))
    result = np.floor(clipped + tolerance).astype(np.int64)
    result = np.minimum(result, caps)
    remaining = target_quantity - int(result.sum())
    if remaining < 0:
        # Undo only epsilon-induced upward rounding first. Large excess is not
        # a rounding artefact and remains an error, rather than silently changing
        # an infeasible input. Ordinary successful allocations are unchanged.
        floor = np.floor(clipped).astype(np.int64)
        if int(floor.sum()) > target_quantity:
            raise ValueError("Floored continuous solution exceeds target quantity.")
        promoted = result > floor
        order = np.lexsort((np.arange(len(values)), clipped - floor))
        for index in order:
            if remaining == 0:
                break
            if promoted[index]:
                result[index] -= 1
                remaining += 1

    fractions = clipped - np.floor(clipped)
    order = np.lexsort((np.arange(len(values)), -fractions))
    for index in order:
        if remaining == 0:
            break
        if result[index] < caps[index]:
            result[index] += 1
            remaining -= 1

    if remaining:
        for index in range(len(result)):
            available = int(caps[index] - result[index])
            addition = min(available, remaining)
            result[index] += addition
            remaining -= addition
            if remaining == 0:
                break

    if (
        remaining
        or int(result.sum()) != target_quantity
        or np.any(result > caps)
        or np.any(result < 0)
    ):
        raise RuntimeError("Integer projection failed to preserve feasibility.")
    return result
