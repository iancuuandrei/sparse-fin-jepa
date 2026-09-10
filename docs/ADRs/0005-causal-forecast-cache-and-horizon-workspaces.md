# ADR 0005: Cache causal forecasts and reuse horizon-indexed solver workspaces

- Status: Accepted
- Date: 2026-09-04
- Owners: ExecSim maintainers

The historical matrix cache decision is superseded by
[ADR 0022](0022-index-history-and-reseal-evaluation.md). The solver workspace
decision remains accepted; the original rationale below is preserved.

## Context

The 2026-09-04 baseline profile attributed 3.76 of 4.54 seconds in one synthetic MPC replay to forecast generation, primarily repeated timestamp formatting and pivot construction. Repeated experiment units also reconstructed identical historical providers and solver setups. Performance work must preserve the objective, constraints, information boundary, integer schedule, and public results.

## Decision

Precompute normalized historical indexing once per provider and cache a session-by-bucket matrix by symbol scope and target date. Derive every shrinking forecast window from that causally filtered matrix. Reuse providers and policies within one experiment run. Represent optimizer reuse with `OptimalExecutionWorkspace`, which caches one OSQP setup per exact horizon, caches the selected OSQP algebra backend, and updates only numeric objective and bound data when that horizon recurs. Construct the risk Hessian from tail-risk sums and use structural positive-semidefinite validation in adaptive MPC; retain full eigenvalue validation in static, standalone, and test solves.

## Rationale

The cache key includes every datum that changes the admissible historical set, so reuse cannot cross a target-date cutoff. Horizon-indexed solvers preserve the exact original subproblem dimension and integer schedule. Structural validation follows directly from positive temporary-impact curvature plus positive-semidefinite risk and tracking terms.

## Consequences

Provider caches consume memory proportional to requested symbol-date pairs and are scoped to the provider instance. A first encounter with each horizon still performs OSQP setup; repeated experiment units update that setup. Diagnostics separately report matrix construction, setup, update, eigenvalue validation, solve, and integer projection time.

## Alternatives considered

- A padded fixed-maximum-horizon OSQP workspace was implemented and rejected because small floating-point differences changed several deterministic integer allocations, despite mathematical equivalence.
- Caching final forecast objects alone was rejected because shrinking MPC windows are unique within a replay.
- Native extensions, GPU execution, and distributed infrastructure were rejected because the profile identified Python-level repeated preprocessing, not insufficient compute throughput.

## Verification

`scripts/profile_performance.py`, `scripts/benchmark.py`, `tests/test_forecasts.py`, `tests/test_optimization.py`, and `tests/test_experiments.py` verify the decision.
