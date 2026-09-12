# ADR 0025: Fail-closed derived evidence and evaluation input identities

- Status: Accepted
- Date: 2026-09-11
- Deciders: ExecSim maintainers

## Context

The locked TCA sample is defined by exact execution-window eligibility.  Each
retained instrument/session then requires a causal ADV20 value to construct the
frozen order quantity.  Treating a missing or malformed ADV20 row as an
exclusion silently changes the scientific population after eligibility has
already been established.

Resealed evaluators may also run with a relocated runtime data root.  Reading a
configured path directly can therefore select the wrong universe or corpus.
Finally, final-result freeze requires all primary JEPA final manifests to share
one training source commit; discovering that only after forecast, representation,
and TCA work wastes a complete evaluation attempt.

## Decision

1. Validate exactly one finite, positive ADV20 row with matching instrument and
   session identities for every exact-window eligible case before TCA workers
   are launched.  `run_historical_tca` repeats this validation as a defensive
   boundary and raises instead of dropping a case.
2. Locked forecast and TCA stages resolve runtime universe and target-corpus
   inputs through `PaperRunConfig.data_path`.  Before either stage consumes the
   universe, its byte hash must match the single `universe_manifest_hash`
   recorded by every fold sequence manifest.
3. Evaluation reseal validates the complete configured primary JEPA final
   inventory (18 coordinates in the frozen v2 configuration), requiring every
   manifest to carry a non-empty `code_commit` and exactly one unique value.

These checks run before derived TEST outputs or `execution.json` publication.
The evaluator source commit remains distinct from the immutable JEPA training
source commit.

## Consequences

- Missing derived evidence and identity mismatches fail closed instead of
  redefining the eligible sample or producing an invalid resealed namespace.
- A relocated evaluator is portable without accepting a byte-different
  universe that is merely scientifically similar or similarly named.
- Invalid mixed-source upstream inventories are rejected at reseal, while the
  final-result-freeze check remains defense in depth.
- No fold boundary, estimator, target, model weight, TCA eligibility rule, or
  report estimand changes.
