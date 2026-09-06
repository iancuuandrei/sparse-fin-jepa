# ADR 0012: Isolate historical sequence members

- Status: Accepted
- Date: 2026-09-06
- Scope: historical paper sequence construction

## Context

The v2 sequence builder validates, tokenizes, hashes, and derives causal seasonal features for each instrument-session. A bounded production profile showed that this work is dominated by Python and pandas execution. Eight threads preserved bounded memory but consumed about one CPU core because the hot operations do not release the Python global interpreter lock consistently. At that rate, the three expanding folds delayed the separately authorized representation training even though the workstation had 32 logical CPUs and sufficient memory.

Universe members are independent during feature construction. They share only immutable fold identity, sourced symbol history, corporate actions, and validated SPY history. Fold normalization and artifact publication must still occur after all member results are available and in the frozen universe order.

## Decision

For historical corpora loaded from partitioned Parquet, construct universe members in at most eight `spawn`-isolated worker processes. Initialize each worker once with immutable fold and SPY state. A worker loads only the Parquet partitions for its assigned instrument, applies the existing validation and causal feature functions, and returns its ordered records, exclusions, and source hashes.

Consume process results in the order declared by the frozen universe. Fit the robust normalizer in the parent from train records only. Normalize and publish session records and sample indexes in that deterministic order. Keep the existing in-process threaded path for small synthetic fixtures.

Process count, scheduling, and completion order are operational details. They must not affect returned record bytes, exclusions, hashes, normalizer inputs, index order, or manifest identity.

## Alternatives considered

- Continue with threads. Rejected because the measured historical hot path remained bound to about one CPU core.
- Load the full corpus into shared memory. Rejected because it would violate the bounded-memory contract and complicate Windows support.
- Partition one instrument across workers. Rejected because causal seasonal history and split availability are naturally sequential within an instrument.
- Change token, seasonal, or hashing mathematics. Rejected because performance work must not alter the frozen scientific protocol.
- Add a distributed execution framework. Rejected because one workstation and one Python package do not justify new infrastructure.

## Consequences

- Independent members can use multiple CPU cores while each worker retains bounded instrument-local memory.
- Windows uses explicit spawn semantics; no fork-inherited mutable state can influence results.
- Immutable SPY state is copied once per worker, increasing bounded memory relative to the threaded path.
- Parent-side ordered collection may wait for a slower earlier member, but worker execution continues and deterministic publication is preserved.
- The decision does not change the estimand, data inclusion rules, folds, features, normalization, model, or any locked comparison.

## Verification

- The multi-session, multi-fold fixture continues to verify causal construction, sourced renames, malformed-session exclusion, SPY separation, and deterministic manifests through the unchanged fixture path.
- Historical execution initializes the same worker function under Windows spawn and fails closed if worker state is absent.
- Ruff, mypy, repository-context validation, and the complete test suite cover the refactored shared member constructor.
