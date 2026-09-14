# ADR 0034: Separate solver acceptance from integer rounding

Status: accepted

## Context

The preserved 6402e40 evaluator reached TCA after completing forecast and
representation evaluation. Its first date failed in a 240-minute, near-saturated
QP: OSQP returned `solved`, with a 0.007727-share box residual, but the integer
projector required an absolute 0.00001-share bound. Completion used NumPy's
implicit relative tolerance instead. Three different numerical contracts were
therefore applied to the same returned vector. Replaying the original problem
on both the retained host and the local environment reproduced the failure.

## Decision

Validate the raw vector in original share units with the explicit budget
`eps_abs + eps_rel * max(abs(sum(x)), max(abs(x)), feasible_quantity)`.
Use that absolute budget for completion with `rtol=0` and for box feasibility.
Do not infer application feasibility from a scaled solver status alone.
Reject non-finite or materially infeasible vectors before clipping admissible
box residuals. Preserve the independent existing integer epsilon, `1e-5`, at
the QP-to-projector boundary. It must never become the solver-relative budget.

There is a second aggregate edge: many individually accepted negative residuals
can make the clipped strict floor exceed the target. Only in that otherwise
failing case, remove clipping excess with the closest nonnegative vector of the
required sum: `z_i=max(clipped_i-lambda, 0)`, where `lambda` enforces the sum.
The KKT conditions of squared-distance projection give this common threshold.
Every component decreases, so the already-satisfied upper capacities remain valid.
This is numerical reconciliation after raw acceptance, not a different QP or
permission to repair materially invalid raw output.

The integer projection retains exact total, non-negativity, integer dtype, and
integer capacities. If epsilon promotions alone exceed the requested total,
undo the least-supported promotions; the direct projector still rejects an
excessive strict floor. Successful existing rounding paths and their tie ordering
are unchanged.

Return mathematically unique feasible points directly: zero feasible quantity,
full capacity, and a one-bucket horizon. Record exact-path diagnostics, without
pretending OSQP ran. These are reductions of the existing constrained problem.

Attach immutable case and decision context to numerical TCA failures and preserve
the exception cause. The exception must survive spawned-process serialization.
No scientific effectiveness metrics are needed to diagnose a numerical failure.

## Alternatives considered

Increasing the projector's single tolerance would also change rounding. Using
NumPy's default relative tolerance conceals a second acceptance budget. Blindly
forcing every solve to absolute-only termination adds numerical work without
addressing the producer-to-consumer contract. Ignoring solver status or clipping
unbounded violations would silently repair materially invalid solutions.

## Consequences

Objectives, participation limits, costs, forecasts, order sizes, and populations
are unchanged. Only previously inconsistent numerical acceptance and exact
degenerate solves change. Tests cover original-problem replay, cold and reused
workspaces, shrinking horizons, near-bound vectors, and exact integer invariants.
Historical effectiveness is not inferred from software qualification.

Primary source: [OSQP convergence](https://osqp.org/docs/solver/index.html#convergence).
