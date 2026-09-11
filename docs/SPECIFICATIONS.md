# ExecSim implementation specifications

This document defines implemented V1 behavior. The focused mathematical, point-in-time, ML, and writing contracts linked here are normative. Code and documentation changes must update the applicable contract in the same change.

## System boundary

ExecSim simulates one equity parent order on one local-market session using timezone-aware minute bars. It accepts policy decisions, enforces inventory and participation constraints, applies a cost model, and produces an execution log, decision trace, and summary. It does not submit orders, mutate source bars, predict returns, model an order book, or infer live performance.

## Public substitution boundaries

The framework exposes typed protocols:

| Protocol | Responsibility |
|---|---|
| `SchedulingPolicy` | Create one static integer schedule from a `ParentOrder` and `DecisionContext` |
| `AdaptiveExecutionPolicy` | Reset state and choose one current-bucket quantity per causal decision |
| `ExecutionCostModel` | Quote a side-aware execution price and cost attribution for one quantity |
| `ArtifactStore` | Save and compatibility-check a fitted model plus immutable metadata |
| `VolumeForecastProvider` | Generate a forecast using the declared point-in-time information set |
| `ForecastModel` | Fit and predict through a replaceable regression adapter |

Concrete registries reject unknown names. Oracle VWAP is `EVALUATION_ONLY` and requires explicit opt-in in experiments.

## Information and time contract

`DecisionContext.observations` contains timestamps strictly earlier than `current_timestamp`. Its forecast must be generated no later than the decision and cover exactly the declared future timestamps. Historical profiles use complete sessions whose dates precede the target session. A historical provider caches one immutable session-by-bucket matrix per symbol or pooled scope and binary-searches the strict date cutoff before selecting complete windows. It never caches date-prefix matrices. Adaptive MPC receives only elapsed observations and a point-in-time forecast.

[The data leakage contract](DATA_LEAKAGE_CONTRACT.md) defines static and dynamic sample semantics, feature availability, chronological split ordering, embargo, and prohibited inputs.

## Order, data, and fill contract

A `ParentOrder` contains an uppercase symbol, `buy` or `sell` side, positive integer shares, a date, and a timezone-naive half-open execution window `[start_time, end_time)`. Simulation timestamps must be timezone-aware, unique, and ordered after normalization. Prices must be finite and positive; volume must be finite and non-negative.

The reference price is bar VWAP when finite and otherwise the OHLC mean. Each bucket uses:

```text
actual_capacity = floor(hard_participation_rate * actual_market_volume)
executed_quantity = min(planned_quantity, remaining_inventory, actual_capacity)
```

Zero-volume bars have zero capacity. Static schedule misses are not silently redistributed. POV observes current bucket volume as it materializes; its trace names that convention. MPC re-solves over the remaining horizon and may warm-start OSQP. `OptimalExecutionWorkspace` caches OSQP setups by exact horizon and updates objective and bound values when a horizon recurs. Warm starts are shifted to the new dimension and clipped to inventory and per-bucket capacity bounds.

## Policy contract

The policy registry supports these deployable policies:

| Name | Behavior |
|---|---|
| `twap` | Equal bucket weights with deterministic earliest-bucket integer remainder |
| `vwap` | Historical point-in-time volume-profile weights |
| `pov` | Floor of target participation times observable current bucket volume |
| `almgren-chriss` | Analytical constant-parameter risk-impact schedule |
| `optimal` | One constrained convex quadratic program over the full horizon |
| `mpc` | Constrained quadratic program re-solved at every decision |

Schedules use non-negative integers, preserve deterministic timestamp order, and reconcile to their feasible planned quantity. Constrained policies report forecast capacity shortfall rather than treating an infeasible order as complete.

## Mathematical and cost contract

[The mathematical model](MATHEMATICAL_MODEL.md) defines notation, units, QP matrices, risk and tracking terms, linear-in-participation temporary impact, half-spread, feasibility relaxation, OSQP acceptance, integer projection, benchmarks, and cost reconciliation. Assumed parameters remain labeled `assumed`; code does not call them calibrated.

## Result and research-output contract

The execution log records order identity, plan, actual capacity, executed quantity, inventory transition, forecast and decision IDs, reference and execution prices, and spread, impact, and timing attribution for every bucket. `SimulationSummary` reports requested, feasible, filled, and unfilled shares; completion plus overall, average-bucket, and maximum participation; benchmarks and side-aware slippage; modeled costs; capacity shortfall; and optimizer telemetry.

Experiment run IDs hash the canonical specification. A run reuses historical providers and compatible policy workspaces within that run. It writes Parquet results, CSV aggregates and paired differences, JSON config and provenance, a Markdown report, and figures. Statistics include sample size, dispersion, quantiles, seeded bootstrap intervals, paired differences versus TWAP, and win rates. Wall-clock solver timing is nondeterministic telemetry. Optimizer traces separate matrix construction, setup, numeric update, eigenvalue validation, solve, and integer projection timing.

## ML infrastructure contract

[The ML design](ML_DESIGN.md) specifies static and dynamic point-in-time rows, targets, feature metadata, 1/5/15-minute buckets, calendar filtering, partitioned Parquet, checksummed manifests, walk-forward folds, model adapters, training-only preprocessing, validation-only selection, locked tests, downstream TCA hooks, and compatibility-checked artifacts.

Historical training is disabled unless `allow_historical_training=true`; V1 acceptance does not grant that authorization. `--dry-run` resolves manifests, schemas, folds, model grids, artifacts, warnings, and evaluation intent without fitting. Tests may fit tiny `synthetic_fixture` datasets.

## Sparse predictive-representation paper contract

[The sparse predictive-representation paper specification](PAPER_DESIGN.md) defines `sparse-jepa-v2`. V1 remains a separate immutable blocked protocol. V2 rebuilds all 505 formation candidates from provider-native Alpaca SIP daily bars and admits ordinary/common stocks with valid daily observations on at least 95% of expected 2021 XNYS sessions, median price of at least $5, and positive median daily dollar volume. If at least 100 qualify, the top 100 are frozen by median daily dollar volume with stable instrument ID as tie-break; the ranking is known only after formation and has no survivor replacement.

Representation quality uses 26 fixed 15-minute intervals. Each interval aggregates at least two chronological observed provider bars: first open, extrema, last close, summed volume and trade count, observed-volume-weighted VWAP, and the square root of summed squared consecutive observed-close log returns. No missing minute is inserted, interpolated, or assigned zero volume. All 26 tokens and required cross-token returns must be computable for a primary session; early closes are excluded. Exact full-session minutes remain a separate diagnostic. TCA independently requires the exact 300 valid minutes from 10:30 through 15:29 that the simulator consumes. SPY follows the same resolution-specific contracts.

Every quality artifact stores daily, exact-full-minute, token-full-session, and exact-TCA-window validity separately with early-close, provider-gap, observed-minute, valid-token, reason, per-token observed-bar-count, and per-token gap-count fields. The corpus-wide v2 path applies the same token rule to target validation, stock sessions, SPY sessions, and seasonal histories. Each stored sequence records `resolution-aware-v2`, all 26 observed-bar counts, and its total provider-gap count. Its symbol comes from the session bars and must match the sourced symbol interval for the stable instrument and session date. Target-period gaps remove cases, never frozen instruments. The formation-only run qualified 497 candidates, froze 100, and is now `AWAITING V2 FORMATION APPROVAL`. Target acquisition remains `NOT RUN`; historical fitting is `TRAINING NOT RUN`; locked-test and TCA results are `EMPIRICAL RESULT NOT AVAILABLE`.

All other paper contracts remain unchanged: corpus-wide 26-by-18 tensors with a 13-feature encoder and five predictor-conditioning fields, matched dense-Gaussian and sparse-rectified-Gaussian JEPA, exact RepReLU, derived rectified RDMReg moments, bounded FP32 diagnostics, streaming device-aware training, frozen probes, residual-scale and bounded long-shape LightGBM targets, compatibility-complete artifacts, locked folds, 15-minute committed execution, oracle regret, and matched fold-safe inference.

[The paper implementation report](PAPER_IMPLEMENTATION_REPORT.md) records the final local gate evidence, numerical fixtures, performance telemetry, and the explicit `NOT RUN` empirical boundary.

## Command contract

The CLI provides these command groups:

- data: `download-data`, `validate-data`, and `build-manifest`;
- policies: `simulate --strategy ...` and the `simulate-twap` compatibility alias;
- research: `experiment run`, `experiment report`, and `scenario`;
- ML: `build-dataset`, `validate-dataset`, `inspect-dataset`, `create-splits`, `training-plan`, `train --dry-run`, and the nested manifest-driven `paper` workflow, including `paper run`;
- diagnostics: `show-config` and `smoke`.

Commands return zero on success. Validation returns one for invalid data. Contract errors fail with a concise parser error. `--json` emits machine-readable simulation output.

## Determinism and compatibility

The [optimized evaluator artifact contract](EVALUATION_IMPLEMENTATION.md) specifies
compact derived bases, atomic publication, identity checks, and vectorized metrics.

Stable ordering, explicit seeds, canonical JSON hashing, checksummed sources and models, exact integer reconciliation, and named numerical tolerances make substantive outputs reproducible. Git-tracked normative text uses SHA-256 after CRLF and bare CR are canonicalized to LF, so checkout line-ending conversion cannot change protocol identity. Design freezes, sidecars, provider responses, receipts, manifests, fitted models, and empirical artifacts use byte-exact hashes. Build timestamps, Git commits, dependency versions, and timing telemetry are provenance, not model inputs. Artifact loading fails closed on checksum or feature schema, target schema, bucket size, or timezone mismatch.

## Verification contract

Run all local gates from the repository root:

```powershell
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m ruff format --check src tests scripts
.\.venv\Scripts\python.exe -m mypy src
.\.venv\Scripts\python.exe scripts/repo_context.py --check
.\.venv\Scripts\python.exe -m pytest -q
```

GitHub Actions repeats these checks on Python 3.11 and 3.13. Tests cover formulas, monotonicity, constraints, causal cutoffs, leakage rejection, diagnostics, integer invariants, buy/sell signs, partial fills, synthetic regimes, statistics, manifests, splits, preprocessing, artifacts, and CLI workflows.

## Known limitations

Minute-bar replay omits quotes, spread measurement, queue position, within-bar paths, permanent impact, market reaction, auctions, fees, multi-asset coupling, and multi-day orders. The historical sample is small. Cost and risk inputs are assumptions unless their provenance states otherwise. No real historical model fit or predictive performance claim is part of V1 acceptance.
