# ADR 0022: Index historical scope once and reseal evaluation

- Status: Accepted
- Date: 2026-09-10
- Supersedes: The historical matrix cache in [ADR 0005](0005-causal-forecast-cache-and-horizon-workspaces.md); solver workspace decisions remain accepted.

## Context

The historical provider rebuilds and retains a pivot for every requested symbol and
date. The locked forecast evaluator also rebuilds common feature frames for each
method. The user authorized superseding the partial TEST execution and replacing
downstream execution infrastructure while preserving model parameters and estimands.

## Decision

Build one lazy session-by-minute matrix per symbol, or one pooled matrix. Preserve
stable ordering by each session's earliest timestamp, including pooled ties. Store
read-only volume, finite-value, and ordered date arrays. Select rows strictly before
the request date using binary search, then apply requested-window completeness,
lookback, positive-volume filtering, and the existing estimator in that order.

Compact checksum-bound fold bases, forecast ledgers shared with TCA, deterministic
process shards, atomic shard manifests, and bounded result merging are implemented.
Bounded TCA input preparation is implemented using per-instrument histories and
date-filtered row groups, with the original ADV/profile aggregation performed before
replay sorting. An independent evaluator receipt binds the new execution namespace
to the unchanged parameter freeze, while original source and TEST-open receipts
remain immutable. Deployment requires semantic tests and complete software gates
before the new TEST execution.

EWMA replay consumes stored exact minute/window forecasts. A shorter requested
window changes the estimator's eligible history and normalization, so truncating
a full-session EWMA forecast is not an equivalent replacement. The derived ledger
therefore records the existing TCA requests at every minute, while learned-model
replay preserves the existing 15-minute update and truncation contract. Tests
require exact forecast equality at every intra-token minute. Date-level TCA tasks
preserve the full population used for balanced side assignment; instrument-only
tasks without preassigned sides would change that assignment and are not used.

## Alternatives considered

Per-date prefix caches retain redundant copies. Recursive EWMA would change the
finite-window estimator. Native extensions do not address duplicated work. Reusing
old partial TEST result rows would mix execution identities. These alternatives are
rejected.

## Scientific consequences

The indexed history can contain later dates internally, but every forecast selects
strictly earlier rows before completeness or estimation. Later columns cannot make
a missing historical window valid. This changes no estimator, sample rule, model,
or estimand. Tests compare all four estimators with the former prefix construction,
including pooled scope, missing buckets, zero totals, and lookback boundaries.

## Operational consequences

Historical matrix storage is proportional to scope history rather than the sum of
all requested date prefixes. The superseded run and its logs remain provenance.
The parameter freeze and fitted upstream artifacts remain immutable.

The original design freeze also binds four living implementation/navigation
documents. A separate reversible amendment binds their current hashes and exact
edits; reversing them must reproduce the original frozen hashes. Scientific YAML,
the paper specification, and the original design freeze are not amendable through
this bridge. This allows documentation to describe current code while retaining
the actual frozen scientific evidence and config identity.
