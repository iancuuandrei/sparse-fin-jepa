# ADR 0014: Cache manifest-keyed sequence indexes

- Status: Accepted
- Date: 2026-09-06
- Scope: streaming paper dataset initialization

## Context

The sequence corpus stores one sample-index Parquet file beside each session artifact. This makes every session independently inspectable, but a historical fold contains tens of thousands of sessions. Initializing one streaming dataset therefore opens tens of thousands of tiny files before it can select two deterministic train positions per session. The six development candidates, their probes, and the final matrix would repeat that file-open cost even though the index contents are immutable for a given sequence manifest.

The session tensor path is already lazy and bounded. Only index metadata must be resident so deterministic sample ordering and per-epoch selection remain explicit.

## Decision

On the first dataset initialization for a manifest and partition, read source index files in deterministic path order and write a consolidated Parquet cache in bounded batches of 512 files. Key the cache filename by the byte-exact sequence-manifest SHA-256 and partition.

Write a receipt containing the schema version, manifest hash, partition, source-file count, row count, and cache checksum. Reuse the cache only when every receipt identity matches, its checksum is valid, and its observed row count matches. Build through a `.partial` file and atomically replace the cache before writing the receipt. A partial or unreceipted cache is never trusted.

Preserve source row order exactly. Continue validating every loaded sample's fold and partition identity. Keep session tensors lazy and bounded; the cache applies only to index metadata.

## Alternatives considered

- Reopen every per-session index for every run. Rejected because repeated filesystem overhead does not add scientific evidence.
- Replace inspectable per-session indexes in the canonical manifest. Rejected during the frozen run because it would change sequence-manifest identities and require rebuilding otherwise valid corpora.
- Store the cache as an unchecked pickle. Rejected because it would be unsafe to trust and would not provide portable schema inspection.
- Load all session tensors eagerly. Rejected because it violates the streaming memory contract.

## Consequences

- The first read retains the existing source-validation cost and creates a portable derived artifact.
- Later JEPA candidates and probes open one checksummed Parquet file per partition instead of tens of thousands of files.
- A manifest change selects a different cache path automatically.
- Corrupt receipted caches fail closed. Interrupted unreceipted output is rebuilt from canonical indexes.
- The decision changes no sample, train-position selection, order, fold, feature, target, model, or estimand.

## Verification

- The multi-session fixture requires a cache and receipt, then reconstructs the dataset through the cache and compares sample counts.
- Existing sampler tests continue to require exactly two deterministic train positions per session per epoch.
- Historical sequence validation provides the first full-corpus cache build and fold/partition identity check.
