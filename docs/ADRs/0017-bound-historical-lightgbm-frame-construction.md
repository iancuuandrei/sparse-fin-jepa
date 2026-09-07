# ADR 0017: Bound historical LightGBM frame construction

- Status: Accepted
- Date: 2026-09-07
- Owners: ExecSim maintainers

## Context

The historical LightGBM corpus contains more than one million scale cases in the first fold and several long-form shape rows per selected origin. The original builder retained one pandas `DataFrame` per scale case and per selected shape origin, and reread the same 26-token session for every as-of row. This representation made object overhead and repeated Parquet reads dominate before LightGBM training began. It did not change the estimand, but it made the frozen experiment operationally unsafe on the qualified 32 GB workstation.

## Decision

Construct the identical pandas training contract in bounded chunks. Cache a bounded number of immutable session records, concatenate intermediate frames after at most 1,024 cases, and release each pending list after consolidation. Preserve manifest order, sample identity, deterministic four-band shape-origin selection, inverse-probability weights, targets, categorical vocabulary, and the frozen LightGBM grid.

## Rationale

Chunked consolidation removes millions of Python container objects and redundant session reads without changing any observation, feature, target, weight, or model parameter. Native pandas categorical columns remain available to LightGBM, as required by the paper contract.

## Consequences

Historical frame construction uses bounded intermediate object counts, although the final pandas scale and shape matrices must still fit in host memory for native LightGBM training. The chunk size is operational and does not enter scientific configuration or artifact identity. Tests must exercise enough rows to cross a consolidation boundary and compare the same downstream pipeline behavior.

## Alternatives considered

- Retaining one frame per case was rejected because its object overhead scales with corpus rows.
- Replacing pandas categories with pre-encoded NumPy matrices was rejected because it would weaken the declared native categorical contract.
- Changing shape sampling or reducing representation width was rejected because either change would alter the frozen experiment.
- Distributed training was rejected because it is outside the project scope and unnecessary for the declared corpus.

## Verification

- `tests/test_paper_sequences.py::test_multisession_multifold_corpus_builder_includes_spy_and_records_corruption`
- `tests/test_paper_pipeline.py`
- Ruff and mypy checks for `src/execsim/ml/paper/lightgbm_data.py`
