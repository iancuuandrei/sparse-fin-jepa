# ADR 0020: Bound LightGBM coordinate memory

- Status: Accepted
- Date: 2026-09-08
- Owners: ExecSim maintainers

## Context

The first authorized historical LightGBM attempt completed `fold-1/raw/shared`, then failed while constructing `fold-1/untrained_neural/shared`. The completed coordinate's training and validation frames remained reachable while pandas consolidated the next 1,081,960-row scale matrix. Causal context and embedding values were also promoted from their native FP32 representation to FP64. The process reached the qualified 32 GB host limit and could not allocate the next 1.16 GiB consolidation block. The failure occurred before locked TEST effectiveness inspection.

ADR 0017 bounded intermediate frame object counts, but it did not bind the numeric storage precision, the untrained-control inference batch, or the lifetime of a completed coordinate.

## Decision

Store LightGBM causal context and 644-value representation/control columns as FP32. Generate the frozen untrained neural control in deterministic bounded batches, then map long-shape rows to scale embeddings through exact `sample_id` indexing. Reject duplicate scale identities or missing shape identities. Release each completed coordinate's frames, model references, and grid records before constructing the next coordinate, and request Python garbage collection at that boundary.

Preserve every sample, feature, horizon flag, target, inverse-probability weight, categorical value, grid candidate, seed, and validation-only selection rule. Preserve the failed attempt and its original model artifact as superseded operational evidence; do not reuse that artifact under a new downstream source identity.

## Rationale

The encoder and exported JEPA artifacts already produce FP32 values. Retaining FP32 in pandas avoids a scientifically unnecessary twofold allocation. Bounded inference prevents the untrained neural placebo from materializing all intermediate activations at once. Exact index mapping replaces a million-entry Python dictionary without weakening identity validation. Coordinate-scoped release prevents peak memory from including both a finished and an incoming experiment.

## Consequences

Historical feature values use the declared FP32 representation before LightGBM binning. Batch size and garbage-collection timing remain operational controls and do not enter the scientific grid. Final training and validation frames must still fit in host memory for one coordinate, so a measured historical rerun remains the acceptance evidence. Any artifact produced by the failed source commit is incompatible with the corrected downstream commit and must remain outside the active 24-coordinate manifest set.

This decision does not change the estimand or open locked TEST.

The v2 design freeze binds `docs/standards/implementation.md`, `docs/SPECIFICATIONS.md`, and `repo_manifest.yaml` byte-for-byte. This downstream pre-lock correction therefore records its direction here and in the unfrozen implementation report instead of rewriting those frozen normative bytes. A future protocol may incorporate the active direction into its normative documents at formation time.

## Alternatives considered

- Retrying unchanged was rejected because the observed peak deterministically exceeded available memory.
- Reducing rows, representation width, shape origins, or grid candidates was rejected because each option changes the frozen experiment.
- Using FP16 was rejected because it introduces an unnecessary additional precision change.
- Loading all untrained-control inputs on CUDA at once was rejected because it only moves the unbounded allocation to another device.
- Distributed training was rejected because it is outside the declared system and is not required for a single coordinate.

## Verification

- `tests/test_paper_pipeline.py::test_lightgbm_raw_hybrid_and_untrained_placebo_share_the_causal_context`
- `tests/test_paper_sequences.py`
- Historical rerun telemetry and the final 24-coordinate audit
- Ruff, mypy, repository-context validation, and the complete pytest suite
