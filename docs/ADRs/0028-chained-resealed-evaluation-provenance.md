# ADR 0028: Preserve chained resealed-evaluation provenance

- Status: Accepted
- Date: 2026-09-12
- Complements: [ADR 0022](0022-index-history-and-reseal-evaluation.md), [ADR 0025](0025-fail-closed-evaluation-input-identities.md), [ADR 0026](0026-cross-link-resealed-evaluation-artifacts.md)

## Context

An optimized evaluator can supersede an earlier resealed execution without
changing the immutable TEST authorization or fitted artifacts. Reusing the
original pre-optimization status for a second reseal loses the immediate
predecessor and permits a late provenance failure.

## Decision

Keep `paper-evaluation-execution-v2` readable for first-generation evidence.
New chained executions use `paper-evaluation-execution-v3` and a typed
`paper-evaluation-supersession-v1` receipt. The receipt binds the protocol,
paper configuration, parameter freeze, original TEST-open receipt and source,
the immediately superseded `execution.json` (including its checksum and
namespace), its evaluator source/tree, a replacement source identity, and a
non-empty reason. Reseal validates the referenced prior receipt and its
immutable inventory before publishing a new empty namespace. Runtime
verification validates the chain and watches both receipts for mutation.

## Consequences

The original authorization source, immediate predecessor, and current
evaluator are distinct identities. Existing v2 executions remain immutable and
verifiable. A corrupted or populated destination, changed prior receipt,
changed freeze/configuration, or broken root chain fails before `execution.json`
publication. No old forecast, TCA, or report output is copied into a new
namespace.
