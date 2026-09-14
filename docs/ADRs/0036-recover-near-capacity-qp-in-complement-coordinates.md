# ADR 0036: Recover near-capacity QPs in complement coordinates

Status: accepted

## Context

The e8d047b TCA execution stopped when OSQP reached its 20,000-iteration
limit. The retained problem requested 700,293 shares against 700,294 shares
of aggregate integer capacity over 300 minutes. Three isolated replays
returned the same non-solved status. This is a convergence failure, not the
solver-to-integer acceptance mismatch addressed by ADR 0034. Accepting the
last iterate, dropping the case, or widening tolerances would be incorrect.

## Decision

Leave every successful original solve unchanged. After a maximum-iteration
failure only, when unused capacity is smaller than requested completion,
allow one fresh OSQP solve in unused-capacity coordinates. Let the original
objective be `f(x)=0.5*x^T*P*x+a^T*x`, capacities `c`, and completion `Q_f`.
Substitution `y=c-x` gives the exact equivalent problem:

```text
minimize 0.5*y^T*P*y + (-P*c-a)^T*y
subject to 0 <= y <= c
           sum(y) = sum(c)-Q_f
x = c-y
```

The omitted constant is `0.5*c^T*P*c+a^T*c`; it cannot change the optimum.
This is a bijection of feasible sets, preserving the strictly convex
objective and the unique continuous optimum. Both variables are shares.
No forecast, capacity, objective coefficient, or integer projection changes.

Use the same solver, tolerances, per-attempt iteration bound, and settings.
This adds at most one bounded attempt, not an unbounded retry or a larger
iteration setting. Do not reuse the transformed workspace for ordinary solves.
Require a solved secondary result and validate recovered quantities in the
original share units before the unchanged integer projection. Preserve
failure for other statuses or a failed secondary solve. Diagnostics identify
the coordinate change and report the original objective, not the objective
with its constant removed.

## Evidence and alternatives

The retained numerical input solved in 447 iterations in complement coordinates
in the local diagnostic, with primal residual below `3e-15`, dual residual below
`3e-14`, and exactly 700,293 projected shares within capacities. This is numerical
qualification, not an effectiveness result. Tests and host qualification remain
mandatory before official execution.

Blind same-source retries reproduced the failure. Raising tolerances or accepting
`maximum iterations reached` is rejected. An integer shortcut for capacity minus
one is not equivalent to solving and projecting the continuous QP. Replacing all
successful solves would needlessly change numerical trajectories. A different
solver or specialized allocation estimator is unnecessary.

## Consequences

The correction affects only previously failed near-capacity problems. It requires
a new evaluator identity and fresh TCA outputs; existing outputs are never
relabeled. Frozen models, scientific configuration, populations, and estimands
remain unchanged. ADR 0034's original-unit acceptance contract remains in force.

Primary references: [OSQP problem and convergence](https://osqp.org/docs/solver/index.html)
and [OSQP status values](https://osqp.org/docs/interfaces/status_values.html).
