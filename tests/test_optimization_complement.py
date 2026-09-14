"""Capacity-complement recovery for a status-7 OSQP solve."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from osqp.interface import OSQPException

import execsim.optimization.qp as qp
from execsim.optimization import OptimalExecutionProblem, OptimalExecutionWorkspace
from execsim.optimization.qp import _build_qp_data, _solve_capacity_complement


def _problem(*, near_saturated: bool, horizon: int = 300) -> OptimalExecutionProblem:
    volumes = np.linspace(10_000.0, 15_000.0, horizon)
    capacities = np.floor(0.1 * volumes).astype(np.int64)
    quantity = int(capacities.sum() - 1) if near_saturated else 10
    return OptimalExecutionProblem(
        quantity=quantity,
        forecast_volumes=volumes,
        forecast_volatilities=np.full(horizon, 0.01),
        half_spreads=np.full(horizon, 0.01),
        temporary_impacts=np.full(horizon, 0.16),
        max_participation_rate=0.1,
        forecast_weights=np.full(horizon, 1.0 / horizon),
    )


class _StatusFailureSolver:
    def __init__(self, status: int) -> None:
        self.status = status
        self.warm_starts: list[np.ndarray] = []

    def warm_start(self, *, x: np.ndarray, y: np.ndarray | None = None) -> None:
        self.warm_starts.append(np.asarray(x, dtype=float))

    def solve(self, *, raise_error: bool) -> SimpleNamespace:
        assert raise_error is True
        raise OSQPException(self.status)


class _SolvedSolver:
    def __init__(self, problem: OptimalExecutionProblem) -> None:
        data = _build_qp_data(problem, "structural")
        self.x = data.capacities.astype(float)
        self.x[0] -= 1.0
        self.info = SimpleNamespace(
            status="solved",
            status_val=1,
            iter=1,
            prim_res=0.0,
            dual_res=0.0,
            obj_val=0.0,
        )

    def solve(self, *, raise_error: bool) -> SimpleNamespace:
        assert raise_error is True
        return SimpleNamespace(x=self.x, y=np.zeros(len(self.x) + 1), info=self.info)


def _solution(
    problem: OptimalExecutionProblem,
    *,
    x: np.ndarray,
    y: np.ndarray,
    status: str = "solved",
    status_value: int = 1,
) -> SimpleNamespace:
    return SimpleNamespace(
        x=x,
        y=y,
        info=SimpleNamespace(
            status=status,
            status_val=status_value,
            iter=1,
            prim_res=0.0,
            dual_res=0.0,
            obj_val=0.0,
        ),
    )


def _force_primary_failure(
    workspace: OptimalExecutionWorkspace,
    solver: _StatusFailureSolver,
) -> None:
    def configure(*args: object, **kwargs: object) -> tuple[object, float, float, bool]:
        return solver, 0.0, 0.0, False

    workspace._configure_solver = configure  # type: ignore[method-assign]


def test_status_seven_retries_near_saturated_problem_in_complement_coordinates() -> None:
    problem = _problem(near_saturated=True)
    data = _build_qp_data(problem, "structural")
    workspace = OptimalExecutionWorkspace(len(problem.forecast_volumes))
    _force_primary_failure(workspace, _StatusFailureSolver(7))

    result = workspace.solve(problem)

    assert result.diagnostics.status == "solved"
    assert result.diagnostics.coordinate_system == "capacity_complement"
    assert result.diagnostics.fallback_used is True
    assert result.diagnostics.workspace_reused is False
    assert result.diagnostics.iterations > 0
    assert result.continuous_quantities.sum() == pytest.approx(result.feasible_quantity, abs=1e-8)
    assert np.all(result.continuous_quantities >= 0)
    assert np.all(result.continuous_quantities <= result.capacities)
    assert int(result.integer_quantities.sum()) == result.feasible_quantity
    expected_objective = (
        0.5 * result.continuous_quantities @ data.matrix @ result.continuous_quantities
        + data.linear @ result.continuous_quantities
    )
    assert result.diagnostics.objective_value == pytest.approx(expected_objective, abs=1e-8)


@pytest.mark.parametrize("status", [1, 3, 5, 10])
def test_non_status_seven_failures_do_not_trigger_complement(status: int) -> None:
    problem = _problem(near_saturated=True)
    workspace = OptimalExecutionWorkspace(len(problem.forecast_volumes))
    _force_primary_failure(workspace, _StatusFailureSolver(status))

    with pytest.raises(RuntimeError, match="before producing an acceptable solution"):
        workspace.solve(problem)


def test_status_seven_does_not_trigger_when_original_target_is_smaller() -> None:
    problem = _problem(near_saturated=False)
    workspace = OptimalExecutionWorkspace(len(problem.forecast_volumes))
    _force_primary_failure(workspace, _StatusFailureSolver(7))

    with pytest.raises(RuntimeError, match="before producing an acceptable solution"):
        workspace.solve(problem)


def test_secondary_failure_remains_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    problem = _problem(near_saturated=True)
    workspace = OptimalExecutionWorkspace(len(problem.forecast_volumes))
    _force_primary_failure(workspace, _StatusFailureSolver(7))
    failure = RuntimeError("forced secondary failure")

    def fail_secondary(*args: object, **kwargs: object) -> tuple[object, ...]:
        raise failure

    monkeypatch.setattr(qp, "_solve_capacity_complement", fail_secondary)

    with pytest.raises(RuntimeError, match="both original and capacity-complement") as caught:
        workspace.solve(problem)
    assert caught.value.__cause__ is failure


def test_secondary_nonsolved_result_remains_fail_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    problem = _problem(near_saturated=True)
    data = _build_qp_data(problem, "structural")
    workspace = OptimalExecutionWorkspace(len(problem.forecast_volumes))
    _force_primary_failure(workspace, _StatusFailureSolver(7))
    x = data.capacities.astype(float)
    x[0] -= 1.0
    nonsolved = _solution(
        problem,
        x=np.zeros(len(x)),
        y=np.zeros(len(x) + 1),
        status="maximum iterations reached",
        status_value=7,
    )
    monkeypatch.setattr(
        qp,
        "_solve_capacity_complement",
        lambda *args, **kwargs: (nonsolved, object(), 0.0, 0.0),
    )

    with pytest.raises(RuntimeError, match="solve was unreliable"):
        workspace.solve(problem)


def test_secondary_nonfinite_quantities_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    problem = _problem(near_saturated=True)
    data = _build_qp_data(problem, "structural")
    workspace = OptimalExecutionWorkspace(len(problem.forecast_volumes))
    _force_primary_failure(workspace, _StatusFailureSolver(7))
    secondary = _solution(
        problem,
        x=np.full(len(data.capacities), np.inf),
        y=np.zeros(len(data.capacities) + 1),
    )
    monkeypatch.setattr(
        qp,
        "_solve_capacity_complement",
        lambda *args, **kwargs: (secondary, object(), 0.0, 0.0),
    )

    with pytest.raises(RuntimeError, match="malformed or non-finite"):
        workspace.solve(problem)


def test_secondary_malformed_duals_are_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    problem = _problem(near_saturated=True)
    data = _build_qp_data(problem, "structural")
    workspace = OptimalExecutionWorkspace(len(problem.forecast_volumes))
    _force_primary_failure(workspace, _StatusFailureSolver(7))
    x = data.capacities.astype(float)
    x[0] -= 1.0
    secondary = _solution(problem, x=x, y=np.zeros(len(x)))
    monkeypatch.setattr(
        qp,
        "_solve_capacity_complement",
        lambda *args, **kwargs: (secondary, object(), 0.0, 0.0),
    )

    with pytest.raises(RuntimeError, match="malformed dual quantities"):
        workspace.solve(problem)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_secondary_nonfinite_duals_are_rejected(
    monkeypatch: pytest.MonkeyPatch,
    value: float,
) -> None:
    problem = _problem(near_saturated=True)
    data = _build_qp_data(problem, "structural")
    workspace = OptimalExecutionWorkspace(len(problem.forecast_volumes))
    _force_primary_failure(workspace, _StatusFailureSolver(7))
    x = data.capacities.astype(float)
    x[0] -= 1.0
    secondary = _solution(problem, x=x, y=np.full(len(x) + 1, value))
    monkeypatch.setattr(
        qp,
        "_solve_capacity_complement",
        lambda *args, **kwargs: (secondary, object(), 0.0, 0.0),
    )

    with pytest.raises(RuntimeError, match="malformed dual quantities"):
        workspace.solve(problem)


def test_secondary_material_dual_violation_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    problem = _problem(near_saturated=True)
    data = _build_qp_data(problem, "structural")
    workspace = OptimalExecutionWorkspace(len(problem.forecast_volumes))
    _force_primary_failure(workspace, _StatusFailureSolver(7))
    x = data.capacities.astype(float)
    x[0] -= 1.0
    secondary = _solution(problem, x=x, y=np.ones(len(x) + 1))
    monkeypatch.setattr(
        qp,
        "_solve_capacity_complement",
        lambda *args, **kwargs: (secondary, object(), 0.0, 0.0),
    )

    with pytest.raises(RuntimeError, match="dual_tolerance"):
        workspace.solve(problem)


def test_primary_success_does_not_attempt_secondary(monkeypatch: pytest.MonkeyPatch) -> None:
    problem = _problem(near_saturated=True, horizon=8)
    workspace = OptimalExecutionWorkspace(len(problem.forecast_volumes))
    primary = _SolvedSolver(problem)

    def configure(*args: object, **kwargs: object) -> tuple[object, float, float, bool]:
        return primary, 0.0, 0.0, False

    workspace._configure_solver = configure  # type: ignore[method-assign]

    def forbidden(*args: object, **kwargs: object) -> tuple[object, ...]:
        raise AssertionError("primary success must not invoke complement recovery")

    monkeypatch.setattr(qp, "_solve_capacity_complement", forbidden)
    result = workspace.solve(problem)

    assert result.diagnostics.coordinate_system == "original"
    assert result.diagnostics.fallback_used is False


def test_fallback_does_not_cache_secondary_and_reused_original_remains_usable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    problem = _problem(near_saturated=True, horizon=8)
    import osqp

    original_solve = osqp.OSQP.solve
    calls = 0

    def fail_once(self: object, raise_error: bool | None = None) -> object:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSQPException(7)
        return original_solve(self, raise_error=raise_error)

    monkeypatch.setattr(osqp.OSQP, "solve", fail_once)
    workspace = OptimalExecutionWorkspace(len(problem.forecast_volumes))

    recovered = workspace.solve(problem)
    reused = workspace.solve(problem)

    assert calls == 3  # primary, fresh complement, then cached original
    assert len(workspace._solvers) == 1  # the fresh complement is not cached
    assert recovered.diagnostics.fallback_used is True
    assert reused.diagnostics.coordinate_system == "original"
    assert reused.diagnostics.fallback_used is False
    assert reused.diagnostics.workspace_reused is True


def test_complement_is_algebraically_equivalent_with_risk_tracking_and_varied_caps() -> None:
    volumes = np.array([101.0, 257.0, 83.0, 401.0, 179.0, 143.0])
    weights = np.array([0.05, 0.15, 0.1, 0.3, 0.25, 0.15])
    problem = OptimalExecutionProblem(
        quantity=300,
        forecast_volumes=volumes,
        forecast_volatilities=np.array([0.01, 0.03, 0.02, 0.04, 0.015, 0.025]),
        half_spreads=np.array([0.01, 0.02, 0.012, 0.018, 0.009, 0.015]),
        temporary_impacts=np.array([0.16, 0.23, 0.12, 0.31, 0.19, 0.27]),
        max_participation_rate=0.5,
        risk_aversion=0.03,
        tracking_penalty=0.2,
        forecast_weights=weights,
    )
    data = _build_qp_data(problem, "full")
    original = OptimalExecutionWorkspace(len(volumes), validation_level="full").solve(problem)
    complement, _, _, _ = _solve_capacity_complement(
        problem,
        data,
        len(volumes),
        warm_start=None,
        validation_level="full",
    )
    recovered = data.capacities.astype(float) - np.asarray(complement.x, dtype=float)

    assert recovered == pytest.approx(original.continuous_quantities, abs=1e-5)
    assert np.isclose(recovered.sum(), data.feasible, atol=1e-8, rtol=0.0)
    assert np.all(recovered >= 0) and np.all(recovered <= data.capacities)
    original_objective = 0.5 * recovered @ data.matrix @ recovered + data.linear @ recovered
    complement_objective = (
        0.5 * complement.x @ data.matrix @ complement.x
        + (-(data.matrix @ data.capacities.astype(float)) - data.linear) @ complement.x
    )
    constant = 0.5 * data.capacities.astype(float) @ data.matrix @ data.capacities.astype(
        float
    ) + data.linear @ data.capacities.astype(float)
    assert original_objective == pytest.approx(complement_objective + constant, abs=1e-8)
