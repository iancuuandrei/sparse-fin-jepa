# ADR 0032: Enforce evaluation producer contracts

Status: accepted

## Context

The a9 evaluator failed in representation diagnostics because the regime builder
omitted `session_id`. Resume tests mocked both sides of that boundary, and legacy
parity preserved the omission. The canonical identity existed on `SequenceSample`.
The audit also found token-key coercion before TCA lookup, invalid assumptions in
report rendering, and incomplete report-completion checks at final freeze.

## Decision

Propagate canonical session identity and compare it with the immutable embedding
export before grouping support transitions. Prove consumer ordering against the
actual exporter. Require integer ledger keys before TCA preflight rather than
truncating malformed keys into an apparently valid grid.

Exercise actual producers, transformations, consumers, and publication using
bounded synthetic integration fixtures. Keep focused mocked resume tests, but do
not treat them as proof of the schema boundaries they replace. Compare scientific
legacy columns exactly and canonical metadata independently.

Permit the two descriptive all-method forecast summaries to be empty with their
declared schemas; pairwise confirmatory populations remain independent. Render
confidence intervals from their endpoints without assuming they contain the
estimate. Label synthetic report fixtures explicitly. Final freeze must verify
report completion and the exact numerical input inventory.

## Alternatives considered

Injecting metadata only into fixtures preserves the failure. Independently
reconstructing identities duplicates canonical state. Weakening validation or
changing estimators to suit rendering would violate the research contract. A
general schema framework is unnecessary for these concrete boundaries.

## Consequences

This enforces existing semantics without changing estimands, frozen configuration,
trained artifacts, probe optimization, populations, bootstrap, or execution
mathematics. Failed executions remain immutable. Corrected source requires a new
chained reseal and fresh result namespace. The
[contract audit](../EVALUATION_CONTRACT_AUDIT.md) records coverage and scope.
