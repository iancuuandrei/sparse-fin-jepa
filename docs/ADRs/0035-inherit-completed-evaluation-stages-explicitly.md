# ADR 0035: Inherit completed evaluation stages explicitly

Status: accepted

## Context

The 6402e40 execution completed forecast and representation evaluation before a
TCA numerical-contract failure. Its corrected evaluator must retain an honest
source chain without spending another full run reproducing unaffected outputs.
The recovery authorization explicitly permits inheriting these two completed
stages; earlier restart instructions required all stages to be recomputed.

## Decision

Use a typed `paper-evaluation-stage-inheritance-v1` receipt and an execution v4
seal. The only permitted inherited set is forecast plus representation, with
restart frontier `run-tca`. Bind the root TEST authorization, parameter freeze,
scientific configuration, immediate predecessor receipt and namespace, producer
commit/tree, replacement commit/tree, and complete manifest-derived file hashes.
Validate the actual completed artifacts, not a caller's assertion of completion.

Keep the predecessor namespace immutable. Resolve forecast profile corpus,
bases, learned/EWMA ledgers, merged results, and representation coordinate/merged
outputs at their original paths. Do not copy or relabel result files. New market
and TCA preparation, TCA results, reports, and final freeze belong exclusively to
the new execution. Old partial TCA shards are never eligible for inheritance.

Report and final-freeze provenance record stage-specific producers plus the
inheritance receipt hash. Every consumer retains its schema, population, and
checksum checks against the declared producer. An explicit source resolver is
not a general compatibility bypass. Cache verification must detect changes to
the inheritance receipt, all inherited bytes, and recursive ancestors.

The detached supervisor verifies inherited completion before invoking only TCA,
report, and final freeze. Status distinguishes inherited validated stages from
stages completed here. No stage is marked newly completed because it was inherited.

## Alternatives considered

Copying old results into the new namespace would misstate provenance. Rewriting
old manifests would destroy evidence. Recomputing unaffected completed stages is
unnecessary under the explicit recovery authorization. A general workflow DAG,
arbitrary stage inheritance, or source-equivalence inference is outside scope.

## Consequences

This narrowly supersedes the blanket cross-source result-isolation rule for
these two explicitly authorized, verified stages. Existing v2/v3 receipts remain
readable without migration; their original semantics and bytes do not change.
The scientific configuration, model weights, estimands, and TEST authorization
remain unchanged. Any missing, changed, partial, or mismatched input fails closed.
