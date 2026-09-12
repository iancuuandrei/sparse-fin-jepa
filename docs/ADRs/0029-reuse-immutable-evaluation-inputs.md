# ADR 0029: Reuse immutable evaluation inputs

Status: proposed pending exact-head qualification

## Context

The `1e3435b` evaluator repeatedly read session Parquet and encoded the same
frozen representation during statistics, ridge selection, and forty MLP epochs
per coordinate. Its host exposed 256 CPUs but granted only 27.2 cgroup cores.
Native defaults therefore oversubscribed the allocation. The execution was
operator-aborted for performance, not rejected on scientific effectiveness.

Forecast input construction also repeated single-row feature construction.
TCA providers reopened the same learned ledger slices across order sizes.

## Decision

Encode each coordinate/partition once into four contiguous, dtype-preserving
files plus original batch metadata. Replay through bounded memory maps, keeping
all samples, masks, batch boundaries, and sample/date identities. Preserve the
DataLoader generator transition at each replay and restore RNG state after the
operational materialization pass. Probe algorithms and optimization order remain
unchanged. The uncached evaluator remains an executable reference.

Publish a cache only after all files are complete and checksummed. Bind it to
the coordinate's evaluator, checkpoint, sequence, and configuration identities,
plus partition, device, PyTorch version, and batching. Reject incompatible or
corrupt caches. Cache files are disposable operational inputs, not results.
Retire them only after the coordinate result is atomically published. Preserve
partial scientific executions and their provenance; never copy results between
evaluator identities.

Limit representation native pools using the minimum CPU affinity and cgroup
quota, with one PyTorch inter-op thread. The operational default must be supported
by frozen-input benchmarks before qualification; it is not a probe hyperparameter.

Build raw forecast features for consecutive session groups without reordering
samples. Reuse the existing manifest-bound sequence index cache. In a TCA worker,
read each learned ledger's relevant date once, then give each simulation a fresh
provider sharing only immutable input data. File-state and identity checks remain
active when constructing a provider.

## Alternatives

- Larger session caches do not eliminate repeated encoding or the full-corpus
  passes and can consume unbounded host memory.
- Fewer samples, epochs, or larger training batches would change the experiment
  and are prohibited.
- Retaining every encoded coordinate on disk would consume unnecessary storage.
- Sharing mutable forecast-provider state would risk coupling simulations.

## Consequences and verification

This is an execution implementation change, not a change of estimand, scientific
configuration, trained model, or authorization. Exact fixture output comparisons
and real bounded input benchmarks are required. Timing fields are telemetry and
are excluded from mathematical equivalence comparisons. Historical effectiveness
is not evidence for selecting these optimizations.

The next restart requires a new merged evaluator identity and namespace, chained
supersession of `1e3435b`, and complete qualification. Previous results remain
historical evidence only. This ADR does not authorize a restart by itself.
