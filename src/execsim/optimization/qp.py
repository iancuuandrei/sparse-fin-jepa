from __future__ import annotations

import math
from dataclasses import dataclass
from time import perf_counter
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from scipy import sparse

from execsim.optimization.integer import project_to_integer_capacities

ValidationLevel = Literal["full", "structural"]
INTEGER_ROUNDING_EPSILON = 1e-5


@dataclass(frozen=True, slots=True)
class OptimalExecutionProblem:
    quantity: int
    forecast_volumes: NDArray[np.float64]
    forecast_volatilities: NDArray[np.float64]
    half_spreads: NDArray[np.float64]
    temporary_impacts: NDArray[np.float64]
    max_participation_rate: float
    risk_aversion: float = 0.0
    delta_t: float = 1.0
    tracking_penalty: float = 0.0
    forecast_weights: NDArray[np.float64] | None = None
    epsilon_volume: float = 1.0
    absolute_tolerance: float = 1e-7
    relative_tolerance: float = 1e-7
    max_iterations: int = 20_000

    def __post_init__(self) -> None:
        vectors = (
            self.forecast_volumes,
            self.forecast_volatilities,
            self.half_spreads,
            self.temporary_impacts,
        )
        arrays = tuple(np.asarray(vector, dtype=float) for vector in vectors)
        if any(array.ndim != 1 for array in arrays) or not arrays[0].size:
            raise ValueError("Optimizer inputs must be non-empty vectors.")
        if len({array.size for array in arrays}) != 1:
            raise ValueError("Optimizer input vectors must have equal length.")
        if any(not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("Optimizer inputs must be finite.")
        if np.any(arrays[0] < 0) or any(np.any(array < 0) for array in arrays[1:]):
            raise ValueError("Volumes, volatilities, spreads, and impacts must be non-negative.")
        if np.any(arrays[3] <= 0):
            raise ValueError("temporary_impacts must be strictly positive for a stable QP.")
        if (
            isinstance(self.quantity, bool)
            or not isinstance(self.quantity, int)
            or self.quantity <= 0
        ):
            raise ValueError("quantity must be a positive integer.")
        if any(
            not math.isfinite(value) or value < 0
            for value in (self.risk_aversion, self.tracking_penalty)
        ):
            raise ValueError("Risk and tracking penalties must be finite and non-negative.")
        if not math.isfinite(self.max_participation_rate) or not (
            0 <= self.max_participation_rate <= 1
        ):
            raise ValueError("max_participation_rate must be in [0, 1].")
        if not math.isfinite(self.delta_t) or self.delta_t <= 0:
            raise ValueError("delta_t must be finite and positive.")
        if not math.isfinite(self.epsilon_volume) or self.epsilon_volume <= 0:
            raise ValueError("epsilon_volume must be finite and positive.")
        if (
            any(
                not math.isfinite(value) or value < 0
                for value in (self.absolute_tolerance, self.relative_tolerance)
            )
            or self.absolute_tolerance + self.relative_tolerance == 0
        ):
            raise ValueError("Solver tolerances must be finite, non-negative, and not both zero.")
        if self.forecast_weights is not None:
            weights = np.asarray(self.forecast_weights, dtype=float)
            if (
                weights.shape != arrays[0].shape
                or np.any(weights < 0)
                or not np.all(np.isfinite(weights))
            ):
                raise ValueError("forecast_weights must be a finite non-negative matching vector.")
            if not np.isclose(weights.sum(), 1.0, atol=1e-8):
                raise ValueError("forecast_weights must sum to one.")
            object.__setattr__(self, "forecast_weights", weights)
        object.__setattr__(self, "forecast_volumes", arrays[0])
        object.__setattr__(self, "forecast_volatilities", arrays[1])
        object.__setattr__(self, "half_spreads", arrays[2])
        object.__setattr__(self, "temporary_impacts", arrays[3])


@dataclass(frozen=True, slots=True)
class SolverDiagnostics:
    status: str
    status_value: int
    iterations: int
    primal_residual: float
    dual_residual: float
    solve_time_seconds: float
    objective_value: float
    minimum_eigenvalue: float
    warm_started: bool
    absolute_tolerance: float
    relative_tolerance: float
    matrix_construction_time_seconds: float = 0.0
    solver_setup_time_seconds: float = 0.0
    solver_update_time_seconds: float = 0.0
    eigenvalue_validation_time_seconds: float = 0.0
    integer_projection_time_seconds: float = 0.0
    workspace_reused: bool = False
    validation_level: ValidationLevel = "full"
    coordinate_system: Literal["original", "capacity_complement"] = "original"
    fallback_used: bool = False


@dataclass(frozen=True, slots=True)
class OptimizationResult:
    continuous_quantities: NDArray[np.float64]
    integer_quantities: NDArray[np.int64]
    capacities: NDArray[np.int64]
    feasible_quantity: int
    predicted_capacity_shortfall: int
    diagnostics: SolverDiagnostics


@dataclass(frozen=True, slots=True)
class _QpData:
    matrix: NDArray[np.float64]
    linear: NDArray[np.float64]
    capacities: NDArray[np.int64]
    feasible: int
    minimum_eigenvalue: float
    eigenvalue_validation_time_seconds: float


def build_qp_matrices(
    problem: OptimalExecutionProblem,
    *,
    validation_level: ValidationLevel = "full",
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64], int]:
    """Build the documented dense QP objective and capacity vectors."""
    data = _build_qp_data(problem, validation_level)
    return data.matrix, data.linear, data.capacities, data.feasible


def _build_qp_data(problem: OptimalExecutionProblem, validation_level: ValidationLevel) -> _QpData:
    if validation_level not in {"full", "structural"}:
        raise ValueError("validation_level must be 'full' or 'structural'.")
    volumes = problem.forecast_volumes
    n = len(volumes)
    capacities = np.floor(problem.max_participation_rate * volumes).astype(np.int64)
    feasible = min(problem.quantity, int(capacities.sum()))
    temporary_coefficients = problem.temporary_impacts / np.maximum(volumes, problem.epsilon_volume)
    matrix = np.diag(2.0 * temporary_coefficients)
    linear = problem.half_spreads.astype(float).copy()

    if problem.risk_aversion:
        risk_weights = problem.forecast_volatilities**2 * problem.delta_t
        tail_risk = np.cumsum(risk_weights[::-1])[::-1]
        index = np.arange(n)
        matrix += 2.0 * problem.risk_aversion * tail_risk[np.maximum.outer(index, index)]
        linear -= 2.0 * problem.risk_aversion * problem.quantity * tail_risk
    tracking_curvature = 0.0
    if problem.tracking_penalty:
        if problem.forecast_weights is None:
            raise ValueError("forecast_weights are required when tracking_penalty is positive.")
        target = max(feasible, 1)
        tracking_curvature = 2.0 * problem.tracking_penalty / target**2
        matrix.flat[:: n + 1] += tracking_curvature
        linear -= 2.0 * problem.tracking_penalty * problem.forecast_weights / target

    eigen_started = perf_counter()
    if validation_level == "full":
        if not np.allclose(matrix, matrix.T, atol=1e-12):
            raise ValueError("QP objective matrix is not symmetric.")
        minimum_eigenvalue = float(np.linalg.eigvalsh(matrix).min())
        if minimum_eigenvalue < -1e-10:
            raise ValueError(
                f"QP objective matrix is not positive semidefinite: {minimum_eigenvalue}"
            )
    else:
        # Each added term is positive semidefinite by construction. Strictly
        # positive temporary impact supplies this inexpensive lower bound.
        minimum_eigenvalue = float(2.0 * temporary_coefficients.min() + tracking_curvature)
    eigen_elapsed = perf_counter() - eigen_started
    return _QpData(
        matrix,
        linear,
        capacities,
        feasible,
        minimum_eigenvalue,
        eigen_elapsed,
    )


class OptimalExecutionWorkspace:
    """Reuse horizon-indexed OSQP setups across repeated MPC solves."""

    def __init__(
        self,
        maximum_horizon: int,
        *,
        validation_level: ValidationLevel = "structural",
    ) -> None:
        if maximum_horizon <= 0:
            raise ValueError("maximum_horizon must be positive.")
        if validation_level not in {"full", "structural"}:
            raise ValueError("validation_level must be 'full' or 'structural'.")
        self.maximum_horizon = maximum_horizon
        self.validation_level = validation_level
        self._solvers: dict[int, Any] = {}
        self._settings: dict[int, tuple[float, float, int]] = {}
        self._algebra: str | None = None

    @property
    def initialized(self) -> bool:
        """Return whether OSQP setup has completed for this workspace."""
        return bool(self._solvers)

    def solve(
        self,
        problem: OptimalExecutionProblem,
        *,
        warm_start: NDArray[np.float64] | None = None,
    ) -> OptimizationResult:
        """Solve one active prefix without changing its mathematical problem."""
        horizon = len(problem.forecast_volumes)
        if horizon > self.maximum_horizon:
            raise ValueError("Problem horizon exceeds workspace maximum_horizon.")
        matrix_started = perf_counter()
        data = _build_qp_data(problem, self.validation_level)
        matrix_elapsed = perf_counter() - matrix_started
        if data.feasible == 0:
            return self._no_capacity_result(problem, data, matrix_elapsed)
        if data.feasible == int(data.capacities.sum()) or horizon == 1:
            quantities = (
                data.capacities.copy()
                if data.feasible == int(data.capacities.sum())
                else np.array([data.feasible], dtype=np.int64)
            )
            return self._unique_feasible_result(problem, data, quantities, matrix_elapsed)

        lower_bounds = np.concatenate([np.zeros(horizon), [float(data.feasible)]])
        upper_bounds = np.concatenate([data.capacities.astype(float), [float(data.feasible)]])
        solver, setup_elapsed, update_elapsed, reused = self._configure_solver(
            problem, data.matrix, data.linear, lower_bounds, upper_bounds
        )
        did_warm_start = warm_start is not None
        prepared_warm_start: NDArray[np.float64] | None = None
        if warm_start is not None:
            warm = np.asarray(warm_start, dtype=float)
            if warm.shape != (horizon,) or not np.all(np.isfinite(warm)):
                raise ValueError("warm_start must be a finite vector matching the horizon.")
            prepared_warm_start = _feasible_warm_start(warm, data.capacities, data.feasible)
            solver.warm_start(x=prepared_warm_start)
        elif reused:
            solver.warm_start(x=np.zeros(horizon), y=np.zeros(horizon + 1))

        solve_started = perf_counter()
        coordinate_system: Literal["original", "capacity_complement"] = "original"
        fallback_used = False
        try:
            solution = solver.solve(raise_error=True)
        except Exception as exc:
            if not (
                _is_osqp_max_iterations_exception(exc) and _capacity_complement_is_smaller(data)
            ):
                raise RuntimeError(
                    "Optimal execution solver failed before producing an acceptable solution: "
                    f"max_iterations={problem.max_iterations}, "
                    f"absolute_tolerance={problem.absolute_tolerance}, "
                    f"relative_tolerance={problem.relative_tolerance}."
                ) from exc
            try:
                solution, solver, fallback_setup_elapsed, fallback_update_elapsed = (
                    _solve_capacity_complement(
                        problem,
                        data,
                        horizon,
                        warm_start=prepared_warm_start,
                        validation_level=self.validation_level,
                    )
                )
            except Exception as fallback_exc:
                raise RuntimeError(
                    "Optimal execution solver failed before producing an acceptable solution "
                    "in both original and capacity-complement coordinates: "
                    f"max_iterations={problem.max_iterations}, "
                    f"absolute_tolerance={problem.absolute_tolerance}, "
                    f"relative_tolerance={problem.relative_tolerance}."
                ) from fallback_exc
            setup_elapsed += fallback_setup_elapsed
            update_elapsed += fallback_update_elapsed
            coordinate_system = "capacity_complement"
            fallback_used = True
        solve_elapsed = perf_counter() - solve_started
        info = solution.info
        status = str(info.status).lower()
        if status != "solved" or solution.x is None:
            raise RuntimeError(
                "Optimal execution solve was unreliable: "
                f"status={info.status}, primal_residual={info.prim_res}, "
                f"dual_residual={info.dual_res}"
            )
        try:
            raw_solution = np.asarray(solution.x, dtype=float)
        except (TypeError, ValueError, OverflowError) as exc:
            raise RuntimeError(
                "Optimal execution solver returned malformed or non-finite quantities."
            ) from exc
        if raw_solution.shape != data.capacities.shape or not np.all(np.isfinite(raw_solution)):
            raise RuntimeError(
                "Optimal execution solver returned malformed or non-finite quantities."
            )
        continuous = (
            data.capacities.astype(float) - raw_solution
            if coordinate_system == "capacity_complement"
            else raw_solution
        )
        objective_value = float(info.obj_val)
        dual_residual = float(info.dual_res)
        if fallback_used:
            # OSQP reports the complement objective, which differs by a constant.
            # Keep the public diagnostic in the original problem's units.
            objective_value = float(
                0.5 * continuous @ data.matrix @ continuous + data.linear @ continuous
            )
            dual_residual = _validate_original_dual_residual(
                problem,
                data,
                continuous,
                solution.y,
            )
        projection_started = perf_counter()
        try:
            sanitized = _sanitize_solver_quantities(
                continuous, data.capacities, data.feasible, problem
            )
            integer = project_to_integer_capacities(
                sanitized, data.capacities, data.feasible, tolerance=INTEGER_ROUNDING_EPSILON
            )
        except (RuntimeError, ValueError) as exc:
            raise RuntimeError(
                "Optimal execution numerical acceptance failed: "
                f"horizon={horizon}, quantity={problem.quantity}, feasible={data.feasible}, "
                f"status={info.status}, iterations={info.iter}, "
                f"primal_residual={info.prim_res}, dual_residual={info.dual_res}, "
                f"eps_abs={problem.absolute_tolerance}, eps_rel={problem.relative_tolerance}."
            ) from exc
        projection_elapsed = perf_counter() - projection_started
        diagnostics = SolverDiagnostics(
            status=status,
            status_value=int(info.status_val),
            iterations=int(info.iter),
            primal_residual=float(info.prim_res),
            dual_residual=dual_residual,
            solve_time_seconds=solve_elapsed,
            objective_value=objective_value,
            minimum_eigenvalue=data.minimum_eigenvalue,
            warm_started=did_warm_start,
            absolute_tolerance=problem.absolute_tolerance,
            relative_tolerance=problem.relative_tolerance,
            matrix_construction_time_seconds=matrix_elapsed,
            solver_setup_time_seconds=setup_elapsed,
            solver_update_time_seconds=update_elapsed,
            eigenvalue_validation_time_seconds=data.eigenvalue_validation_time_seconds,
            integer_projection_time_seconds=projection_elapsed,
            workspace_reused=reused and not fallback_used,
            validation_level=self.validation_level,
            coordinate_system=coordinate_system,
            fallback_used=fallback_used,
        )
        return OptimizationResult(
            continuous,
            integer,
            data.capacities,
            data.feasible,
            problem.quantity - data.feasible,
            diagnostics,
        )

    def _configure_solver(
        self,
        problem: OptimalExecutionProblem,
        matrix: NDArray[np.float64],
        linear: NDArray[np.float64],
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
    ) -> tuple[Any, float, float, bool]:
        horizon = len(linear)
        objective = sparse.triu(sparse.csc_matrix(matrix), format="csc")
        settings = (
            problem.absolute_tolerance,
            problem.relative_tolerance,
            problem.max_iterations,
        )
        solver = self._solvers.get(horizon)
        if solver is None:
            try:
                import osqp
            except ImportError as exc:  # pragma: no cover - dependency error path
                raise RuntimeError(
                    "Install the 'optimization' extra to use optimal policies."
                ) from exc
            constraints = sparse.vstack(
                [sparse.eye(horizon, format="csc"), np.ones((1, horizon))],
                format="csc",
            )
            if self._algebra is None:
                self._algebra = str(osqp.default_algebra())
            solver = osqp.OSQP(algebra=self._algebra)
            started = perf_counter()
            solver.setup(
                P=objective,
                q=linear,
                A=constraints,
                l=lower_bounds,
                u=upper_bounds,
                verbose=False,
                eps_abs=problem.absolute_tolerance,
                eps_rel=problem.relative_tolerance,
                max_iter=problem.max_iterations,
                adaptive_rho=True,
                polishing=True,
                scaled_termination=True,
                check_termination=1,
            )
            elapsed = perf_counter() - started
            self._solvers[horizon] = solver
            self._settings[horizon] = settings
            return solver, elapsed, 0.0, False

        started = perf_counter()
        if settings != self._settings[horizon]:
            solver.update_settings(
                eps_abs=problem.absolute_tolerance,
                eps_rel=problem.relative_tolerance,
                max_iter=problem.max_iterations,
            )
            self._settings[horizon] = settings
        solver.update(
            Px=objective.data,
            q=linear,
            l=lower_bounds,
            u=upper_bounds,
        )
        return solver, 0.0, perf_counter() - started, True

    def _no_capacity_result(
        self,
        problem: OptimalExecutionProblem,
        data: _QpData,
        matrix_elapsed: float,
    ) -> OptimizationResult:
        horizon = len(problem.forecast_volumes)
        zeros = np.zeros(horizon, dtype=float)
        diagnostics = SolverDiagnostics(
            status="no_capacity",
            status_value=0,
            iterations=0,
            primal_residual=0.0,
            dual_residual=0.0,
            solve_time_seconds=0.0,
            objective_value=0.0,
            minimum_eigenvalue=data.minimum_eigenvalue,
            warm_started=False,
            absolute_tolerance=problem.absolute_tolerance,
            relative_tolerance=problem.relative_tolerance,
            matrix_construction_time_seconds=matrix_elapsed,
            eigenvalue_validation_time_seconds=data.eigenvalue_validation_time_seconds,
            workspace_reused=len(problem.forecast_volumes) in self._solvers,
            validation_level=self.validation_level,
        )
        return OptimizationResult(
            zeros,
            zeros.astype(np.int64),
            data.capacities,
            0,
            problem.quantity,
            diagnostics,
        )

    def _unique_feasible_result(
        self,
        problem: OptimalExecutionProblem,
        data: _QpData,
        quantities: NDArray[np.int64],
        matrix_elapsed: float,
    ) -> OptimizationResult:
        """Return the sole feasible point without introducing numerical box violations."""
        continuous = quantities.astype(float)
        diagnostics = SolverDiagnostics(
            status="unique_feasible",
            status_value=0,
            iterations=0,
            primal_residual=0.0,
            dual_residual=0.0,
            solve_time_seconds=0.0,
            objective_value=float(
                0.5 * continuous @ data.matrix @ continuous + data.linear @ continuous
            ),
            minimum_eigenvalue=data.minimum_eigenvalue,
            warm_started=False,
            absolute_tolerance=problem.absolute_tolerance,
            relative_tolerance=problem.relative_tolerance,
            matrix_construction_time_seconds=matrix_elapsed,
            eigenvalue_validation_time_seconds=data.eigenvalue_validation_time_seconds,
            validation_level=self.validation_level,
        )
        return OptimizationResult(
            continuous,
            quantities,
            data.capacities,
            data.feasible,
            problem.quantity - data.feasible,
            diagnostics,
        )


_OSQP_MAX_ITERATIONS_STATUS = 7


def _is_osqp_max_iterations_exception(exc: Exception) -> bool:
    """Identify OSQP's status-7 exception without broad exception matching."""
    try:
        import osqp
    except ImportError:  # pragma: no cover - the caller already imported OSQP
        return False
    if not isinstance(exc, osqp.OSQPException) or not exc.args:
        return False
    try:
        return int(exc.args[0]) == _OSQP_MAX_ITERATIONS_STATUS
    except (TypeError, ValueError):
        return False


def _capacity_complement_is_smaller(data: _QpData) -> bool:
    """Return whether y=c-x has the smaller equality target."""
    capacity_sum = int(data.capacities.sum())
    return data.feasible > capacity_sum - data.feasible


def _solve_capacity_complement(
    problem: OptimalExecutionProblem,
    data: _QpData,
    horizon: int,
    *,
    warm_start: NDArray[np.float64] | None,
    validation_level: ValidationLevel,
) -> tuple[Any, Any, float, float]:
    """Retry a near-saturated QP in equivalent capacity-complement coordinates."""
    capacities = data.capacities.astype(float)
    capacity_sum = int(data.capacities.sum())
    complement_target = capacity_sum - data.feasible
    if complement_target <= 0:
        raise ValueError("Capacity complement requires a positive target.")
    complement_linear = -(data.matrix @ capacities) - data.linear
    lower_bounds = np.concatenate([np.zeros(horizon), [float(complement_target)]])
    upper_bounds = np.concatenate([capacities, [float(complement_target)]])

    # A fresh workspace prevents a failed ADMM iterate/rho state from affecting
    # the retry. It is deliberately not cached as the caller's original solver.
    workspace = OptimalExecutionWorkspace(horizon, validation_level=validation_level)
    solver, setup_elapsed, update_elapsed, _ = workspace._configure_solver(
        problem,
        data.matrix,
        complement_linear,
        lower_bounds,
        upper_bounds,
    )
    if warm_start is not None:
        solver.warm_start(x=capacities - warm_start)
    solution = solver.solve(raise_error=True)
    return solution, solver, setup_elapsed, update_elapsed


def _validate_original_dual_residual(
    problem: OptimalExecutionProblem,
    data: _QpData,
    continuous: NDArray[np.float64],
    dual: object,
) -> float:
    """Validate complement duals after mapping them back to original coordinates."""
    try:
        values = np.asarray(dual, dtype=float)
    except (TypeError, ValueError, OverflowError) as exc:
        raise RuntimeError("Optimal execution solver returned malformed dual quantities.") from exc
    horizon = len(continuous)
    if values.shape != (horizon + 1,) or not np.all(np.isfinite(values)):
        raise RuntimeError("Optimal execution solver returned malformed dual quantities.")
    # y=c-x reverses the stationarity sign, so the original dual is -y.
    original_dual = -values
    with np.errstate(over="ignore", invalid="ignore"):
        primal_gradient = data.matrix @ continuous
        stationarity = (
            primal_gradient + data.linear + original_dual[:horizon] + original_dual[horizon]
        )
    if not np.all(np.isfinite(primal_gradient)) or not np.all(np.isfinite(stationarity)):
        raise RuntimeError("Optimal execution complement solution has a non-finite dual residual.")
    residual = float(np.max(np.abs(stationarity)))
    scale = max(
        float(np.max(np.abs(primal_gradient))),
        float(np.max(np.abs(original_dual[:horizon] + original_dual[horizon]))),
        float(np.max(np.abs(data.linear))),
    )
    tolerance = problem.absolute_tolerance + problem.relative_tolerance * scale
    if not math.isfinite(residual) or not math.isfinite(scale) or residual > tolerance:
        raise RuntimeError(
            "Optimal execution complement solution violates original-unit dual "
            f"acceptance: residual={residual}, dual_tolerance={tolerance}."
        )
    return residual


def _sanitize_solver_quantities(
    continuous: NDArray[np.float64],
    capacities: NDArray[np.int64],
    feasible: int,
    problem: OptimalExecutionProblem,
) -> NDArray[np.float64]:
    """Check raw share-unit constraints before clipping only admissible box residuals.

    For A=[I; 1^T], use max(||Ax||_inf, Q_f) as the original-unit constraint
    scale. This is an explicit application-side check, not reliance on OSQP's
    internal z or scaling. No NumPy-default relative tolerance or integer-rounding
    epsilon participates in acceptance.
    Scaled OSQP termination alone does not replace this unscaled boundary.
    """
    if continuous.shape != capacities.shape or not np.all(np.isfinite(continuous)):
        raise RuntimeError("Optimal execution solver returned malformed or non-finite quantities.")
    total = float(continuous.sum())
    scale = max(abs(total), float(np.max(np.abs(continuous))), float(feasible))
    tolerance = problem.absolute_tolerance + problem.relative_tolerance * scale
    if not np.isclose(total, feasible, atol=tolerance, rtol=0.0):
        raise RuntimeError(
            f"Optimal execution solution violates completion: residual={total - feasible}, "
            f"share_tolerance={tolerance}."
        )
    if np.any(continuous < -tolerance) or np.any(continuous > capacities + tolerance):
        raise RuntimeError(
            "Optimal execution solution violates capacity bounds beyond numerical acceptance: "
            f"lower={max(0.0, -float(continuous.min()))}, "
            f"upper={max(0.0, float((continuous - capacities).max()))}, "
            f"share_tolerance={tolerance}."
        )
    clipped = np.clip(continuous, 0.0, capacities.astype(float))
    if int(np.floor(clipped).sum()) > feasible:
        # Many accepted lower-bound residuals can accumulate after clipping.
        # Repair only this otherwise-failing branch. The closest nonnegative
        # vector with the target sum only decreases entries, so upper capacities
        # stay satisfied. All previously successful rounding paths are untouched.
        ordered = np.sort(clipped)[::-1]
        counts = np.arange(1, len(ordered) + 1)
        thresholds = (np.cumsum(ordered) - feasible) / counts
        active = np.flatnonzero(ordered > thresholds)
        threshold = thresholds[active[-1]]
        clipped = np.maximum(clipped - threshold, 0.0)
    return clipped


def _feasible_warm_start(
    warm_start: NDArray[np.float64], capacities: NDArray[np.int64], feasible: int
) -> NDArray[np.float64]:
    """Clip a solver hint to the active inventory and capacity bounds."""
    return np.clip(warm_start, 0.0, np.minimum(capacities, feasible).astype(float))


def solve_optimal_execution(
    problem: OptimalExecutionProblem,
    *,
    warm_start: NDArray[np.float64] | None = None,
) -> OptimizationResult:
    """Solve one optimal-execution problem with full matrix validation."""
    workspace = OptimalExecutionWorkspace(len(problem.forecast_volumes), validation_level="full")
    return workspace.solve(problem, warm_start=warm_start)
