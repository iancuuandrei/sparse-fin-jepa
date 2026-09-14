# ExecSim project context

## Objective

ExecSim is a mathematically explicit, efficient, ML-ready framework for offline research on single-asset intraday parent-order execution. It compares causal policies under the same bars, constraints, benchmarks, and transparent cost assumptions.

## V1 scope

V1 includes canonical minute bars, deterministic scenarios, point-in-time volume forecasts, TWAP, historical VWAP, POV, analytical Almgren–Chriss, constrained optimal, adaptive MPC, an explicit evaluation-only oracle, transaction-cost analysis, experiment grids, statistical reports, and ML data-to-artifact infrastructure.

The framework treats optimization as the decision layer. ML may forecast volume inputs but does not directly choose unconstrained trades. Historical model fitting and predictive-performance claims are outside this acceptance run.

The optional sparse predictive-representation paper layer supplies the frozen
`sparse-jepa-v2` protocol. Its separately authorized historical experiment is
complete; the [experiment record](EXPERIMENT_RESULTS.md) distinguishes its mixed
empirical results from software qualification. It does not expand the system
into live trading or direct learned execution. Licensed acquisition and
historical training remain separately authorized actions.

## Non-goals

V1 does not provide live execution, broker order submission, alpha prediction, an order book, queue position, counterfactual market response, multi-asset optimization, or multi-day parent orders. It is not production trading infrastructure.

## Evidence policy

Tests establish software and mathematical invariants. Synthetic scenarios establish controlled behavior. Historical replay demonstrates behavior on the bundled sample. None of these alone establishes out-of-sample predictive value or live strategy superiority.

## Authority

Use `AGENTS.md` and `repo_manifest.yaml` to resolve applicable documents and checks. `docs/standards/implementation.md` controls code and documentation practice. `docs/SPECIFICATIONS.md` controls implemented behavior.
