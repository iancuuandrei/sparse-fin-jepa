# Mathematical model

This document derives the models governed by `docs/standards/implementation.md`. Code, tests, and this derivation must evolve together.

## Notation and units

- `Q`: absolute parent quantity, shares.
- `q_k`: planned or executed shares in bucket `k`.
- `x_k`: shares remaining before bucket `k`; `x_0=Q`.
- `v_k`: market volume, shares per bucket.
- `rho`: participation fraction, dimensionless.
- `P_k`: reference price, currency per share.
- `h_k`: half-spread, currency per share.
- `eta_k`: linear temporary-impact coefficient, currency per share at 100% participation.
- `sigma_k`: forecast price volatility, currency per share per square-root bucket.
- `delta_t`: bucket duration in minutes or a consistently normalized fraction.
- `lambda_risk`: risk-aversion multiplier; its numerical scale depends on the chosen `sigma` and `delta_t` units.

## Realized cost model

For side sign `s` (`+1` buy, `-1` sell), realized participation `q_k/max(v_k,epsilon)`, and non-negative parameters:

```text
P_exec,k = P_k + s [h_k + eta_k q_k/max(v_k,epsilon)]
C_spread,k = h_k q_k
C_temp,k = eta_k q_k^2/max(v_k,epsilon)
```

Thus `s*q_k*(P_exec,k-P_k)=C_spread,k+C_temp,k` for every fill. The model is deterministic, crosses half a spread, and uses linear instantaneous price impact so its total temporary cost is convex quadratic. It does not identify queue position, transient resilience, adverse selection, or the endogenous market response from OHLCV bars. Each parameter therefore records whether it was measured, estimated, assumed, or externally supplied.

## Constrained planning problem

Let `L` be the lower-triangular matrix of ones, including its diagonal. Remaining inventory after each bucket is:

```text
x = Q * 1 - Lq
```

With `D=diag(sigma_k^2*delta_t)`, `A=diag(eta_k/max(v_hat_k,epsilon))`, spread vector `h`, forecast profile `w`, and tracking coefficient `lambda_track`, the objective is:

```text
J(q) = h^T q + q^T A q
     + lambda_risk (Q1-Lq)^T D (Q1-Lq)
     + lambda_track ||q/Q_target-w||^2
```

Expanding into OSQP form `0.5 q^T P q + a^T q` gives:

```text
P = 2A + 2 lambda_risk L^T D L
    + 2 lambda_track I/Q_target^2
a = h - 2 lambda_risk Q L^T D 1
    - 2 lambda_track w/Q_target
```

Every term in `P` is positive semidefinite when inputs are valid. The implementation constructs the risk element directly as `R_ij=sum(d_k for k>=max(i,j))`, where `d_k=sigma_k^2*delta_t`; it does not materialize diagonal and lower-triangular intermediates. Static, standalone, and debug solves validate symmetry and the minimum eigenvalue numerically. Adaptive MPC uses the analytical lower bound supplied by strictly positive temporary-impact curvature plus positive-semidefinite risk and tracking terms. Integer forecast capacities are `c_k=floor(rho_max*v_hat_k)`, feasible completion is `Q_f=min(Q,sum c_k)`, and constraints are `0<=q<=c` and `sum q=Q_f`. A capacity shortfall is a result, not solver infeasibility.

## Integer projection

Continuous `q` is clipped to numerical bounds, floored, and the residual `Q_f-sum floor(q)` is allocated to eligible buckets in descending fractional-remainder order with ascending bucket index as the tie-break. The projection is valid only if its result is non-negative, within integer capacities, and sums exactly to `Q_f`.

Before clipping, the solver boundary checks finite raw quantities in original
share units. With `s=max(abs(sum(q)), max(abs(q)), Q_f)`, the deliberately explicit
acceptance budget is `tau=eps_abs+eps_rel*s`. Both the completion residual and
each lower/upper box violation must be within `tau`; completion uses `rtol=0`.
This application check remains required when OSQP uses scaled termination.
Only admissible box violations are clipped. Clipping several components can
accumulate a total residual; deterministic integer reconciliation still enforces
the exact target. The independent QP rounding epsilon remains `1e-5`, not `tau`.
Epsilon-only excess promotions are undone from the smallest fractional remainder;
the direct projector still rejects a strict floor exceeding the target. If
accepted lower-bound clipping itself creates that excess, the sanitizer first
uses `z_i=max(clipped_i-lambda, 0)` with `sum(z)=Q_f`. This is the unique
Euclidean projection onto the target-sum nonnegative simplex; it only decreases
components and therefore preserves upper capacities. It is used exclusively on
the previously failing strict-floor branch, leaving successful allocations intact.

If `Q_f=0`, the exact vector is zero. If `Q_f=sum(c)`, the exact vector is `c`.
If there is one bucket, its exact value is `Q_f`. These unique feasible sets do
not require OSQP, regardless of the objective coefficients. Exact-path diagnostics
record zero solver iterations. See ADR 0034 and the
[OSQP convergence contract](https://osqp.org/docs/solver/index.html#convergence).

## Classical constant-parameter reference

For the simplified objective

```text
eta * sum(q_k^2) + lambda_risk * sigma^2 * sum(x_(k+1)^2)
```

with complete liquidation and no active capacity bound, the inventory recurrence yields

```text
cosh(kappa) = 1 + lambda_risk*sigma^2/(2*eta)
x_j = Q * sinh(kappa*(N-j))/sinh(kappa*N)
q_j = x_j - x_(j+1)
```

The `kappa -> 0` limit is `x_j=Q(N-j)/N`, hence equal-rate TWAP. Greater risk aversion raises `kappa` and front-loads liquidation. Linear permanent impact for a fixed one-directional total is schedule-independent in this reference setup and is not used to justify a schedule.

## Model-predictive control

At each bucket, MPC constructs a new point-in-time forecast over the remaining horizon, recomputes feasible capacity for realized remaining inventory, solves the same constrained problem, executes only the first action subject to actual capacity, and repeats. The workspace keeps one solver setup per exact horizon; repeated experiment units update its numeric values. A shifted warm start is clipped to the new inventory and capacity bounds. Warm starts improve numerical work but do not change the accepted solution tolerance. Every solve is retained in a decision trace.

## TCA attribution

For executed shares, signed arrival shortfall is `s*sum q_k(P_exec,k-P_arrival)`. Timing cost is `s*sum q_k(P_k-P_arrival)`. The exact filled-share identity is:

```text
executed implementation shortfall
= timing cost + spread cost + temporary impact cost + residual
```

The residual should be numerical noise for this cost model. Unfilled opportunity cost is reported separately using an explicitly selected terminal reference; it is not forced into the filled-share identity.

## Known limitations

Historical replay is not counterfactual: simulated orders do not change future bar prices or volumes. Bar VWAP is not a tradable quote, within-bar causality is coarse, forecast parameters are assumptions unless provenance says otherwise, and the quadratic optimizer is optimal only for its stated convex objective and forecasts.
