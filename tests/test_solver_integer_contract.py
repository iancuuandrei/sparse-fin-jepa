"""Share-scale solver acceptance is independent of exact integer reconciliation."""

from dataclasses import replace

import numpy as np
import pytest

from execsim.optimization import (
    OptimalExecutionProblem,
    OptimalExecutionWorkspace,
    solve_optimal_execution,
)
from execsim.optimization.qp import _sanitize_solver_quantities


def problem(capacities: np.ndarray, quantity: int) -> OptimalExecutionProblem:
    horizon = len(capacities)
    return OptimalExecutionProblem(
        quantity=quantity,
        forecast_volumes=capacities.astype(float) * 10.0,
        forecast_volatilities=np.full(horizon, 0.01),
        half_spreads=np.full(horizon, 0.008),
        temporary_impacts=np.full(horizon, 0.16),
        max_participation_rate=0.1,
    )


@pytest.mark.parametrize("horizon", [1, 2, 15, 30, 60, 300])
@pytest.mark.parametrize("pattern", ["zero", "one", "equal", "skewed", "large"])
def test_cold_and_reused_workspace_integer_constraints(horizon: int, pattern: str) -> None:
    caps = np.full(horizon, 100, dtype=np.int64)
    if pattern == "zero":
        caps[:] = 0
    elif pattern == "one":
        caps[:] = 0
        caps[-1] = 1000
    elif pattern == "skewed":
        caps = np.maximum(1, np.geomspace(1, 10000, horizon).astype(np.int64))
    elif pattern == "large":
        caps = np.linspace(470, 1174, horizon).astype(np.int64)
    total = int(caps.sum())
    quantities = sorted({1, max(1, total // 3), max(1, total - 1), max(1, total), total + 100})
    workspace = OptimalExecutionWorkspace(horizon)
    for quantity in quantities:
        current = problem(caps, quantity)
        for solve in (solve_optimal_execution, workspace.solve):
            result = solve(current)
            assert np.issubdtype(result.integer_quantities.dtype, np.integer)
            assert int(result.integer_quantities.sum()) == min(quantity, total)
            assert np.all(result.integer_quantities >= 0)
            assert np.all(result.integer_quantities <= caps)


@pytest.mark.parametrize("caps,quantity", [([0, 0], 10), ([2, 3], 5), ([2, 3], 10), ([100], 7)])
def test_unique_feasible_cases_do_not_call_osqp(monkeypatch, caps, quantity) -> None:
    def forbidden(*args, **kwargs):
        raise AssertionError("A unique feasible point needs no numerical solver")

    monkeypatch.setattr(OptimalExecutionWorkspace, "_configure_solver", forbidden)
    result = OptimalExecutionWorkspace(len(caps)).solve(problem(np.array(caps), quantity))
    assert result.diagnostics.iterations == 0
    assert result.diagnostics.solve_time_seconds == 0
    assert int(result.integer_quantities.sum()) == min(quantity, sum(caps))


def test_box_acceptance_does_not_widen_integer_rounding() -> None:
    caps = np.array([100000, 100000])
    current = problem(caps, 150000)
    values = np.array([100000.007, 49999.993])
    sanitized = _sanitize_solver_quantities(values, caps, 150000, current)
    assert sanitized.tolist() == [100000, 49999.993]
    from execsim.optimization.integer import project_to_integer_capacities

    with pytest.raises(ValueError, match="capacity bounds"):
        project_to_integer_capacities(values, caps, 150000, tolerance=1e-5)
    projected = project_to_integer_capacities(sanitized, caps, 150000, tolerance=1e-5)
    assert projected.tolist() == [100000, 50000]


@pytest.mark.parametrize("values", [[75000, 75001], [-1, 150001], [100001, 49999], [np.nan, 1]])
def test_materially_invalid_solver_values_fail_closed(values) -> None:
    caps = np.array([100000, 100000])
    with pytest.raises(RuntimeError):
        _sanitize_solver_quantities(np.array(values), caps, 150000, problem(caps, 150000))


def test_completion_has_no_implicit_numpy_relative_tolerance() -> None:
    caps = np.array([100000, 100000])
    current = replace(problem(caps, 150000), relative_tolerance=0.0)
    with pytest.raises(RuntimeError, match="completion"):
        _sanitize_solver_quantities(np.array([75000.001, 75000]), caps, 150000, current)


def test_shrinking_workspace_preserves_exact_integer_inventory() -> None:
    workspace = OptimalExecutionWorkspace(300)
    remaining = 155641
    for horizon in (300, 240, 60, 30, 15, 2, 1):
        caps = np.linspace(470, 1174, horizon).astype(np.int64)
        current = problem(caps, max(1, remaining))
        result = workspace.solve(current)
        assert int(result.integer_quantities.sum()) == min(max(1, remaining), int(caps.sum()))
        assert np.all(result.integer_quantities >= 0)
        assert np.all(result.integer_quantities <= caps)
        remaining = max(0, remaining - int(result.integer_quantities[0]))


def test_integer_epsilon_promotions_cannot_overshoot_requested_total() -> None:
    from execsim.optimization.integer import project_to_integer_capacities

    result = project_to_integer_capacities([0.999999, 0.999998], [1, 1], 1, tolerance=1e-5)
    assert result.tolist() == [1, 0]
    with pytest.raises(ValueError, match="exceeds target"):
        project_to_integer_capacities([2.0, 2.0], [3, 3], 1)


def test_many_accepted_lower_residuals_cannot_break_integer_completion() -> None:
    from execsim.optimization.integer import project_to_integer_capacities

    caps = np.array([0] * 200 + [2000] * 100)
    values = np.array([-0.007] * 200 + [1560.0] * 99 + [1202.4])
    quantity = 155641
    assert np.isclose(values.sum(), quantity, atol=1e-8, rtol=0)
    assert np.floor(np.clip(values, 0, caps)).sum() == quantity + 1
    repaired = _sanitize_solver_quantities(values, caps, quantity, problem(caps, quantity))
    assert np.isclose(repaired.sum(), quantity, atol=1e-8, rtol=0)
    assert (repaired >= 0).all() and (repaired <= caps).all()
    projected = project_to_integer_capacities(repaired, caps, quantity, tolerance=1e-5)
    assert projected.sum() == quantity
    assert (projected >= 0).all() and (projected <= caps).all()

    values[0] = -0.016
    values[-1] += 0.009  # Preserve equality while exceeding the raw bound budget.
    with pytest.raises(RuntimeError, match="capacity bounds"):
        _sanitize_solver_quantities(values, caps, quantity, problem(caps, quantity))
