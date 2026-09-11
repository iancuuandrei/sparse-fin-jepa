# ADR 0018: Stream historical embedding export

- Status: Accepted
- Date: 2026-09-07
- Owners: ExecSim maintainers

## Context

Each historical embedding partition can contain more than one million 644-value rows. The original exporter accumulated an entire partition as Python dictionaries before creating one Arrow table. The first authorized historical export exhausted host memory at about 19 GB and failed before producing a valid Parquet artifact.

## Decision

Convert and write each inference batch as a Parquet row group through one schema-stable `ParquetWriter`. Keep only the current batch in Python memory, count rows across batches, close the writer deterministically, and delete an incomplete destination file when export raises.

## Rationale

Streaming bounds host memory by inference batch size while preserving row order, values, schema, partition identity, checksums, and the frozen 644-dimensional embedding contract. It changes storage mechanics only and does not alter the scientific protocol or estimand.

## Consequences

Large partitions contain many row groups and remain readable by standard Arrow and pandas consumers. A failed partition does not masquerade as complete output. The output directory remains fail-closed and must be absent before a full coordinate starts.

## Alternatives considered

- Increasing RAM or virtual memory was rejected because memory use still scaled with corpus size.
- Reducing exported rows or embedding width was rejected because it would change the frozen experiment.
- Writing one file per inference batch was rejected because it would expand artifact identity and file-count overhead unnecessarily.

## Verification

- `tests/test_paper_sequences.py::test_multisession_multifold_corpus_builder_includes_spy_and_records_corruption`
- A completed historical `fold-1/dense/13` export with 1,348,006 rows and bounded process memory
- Ruff and mypy checks for `src/execsim/ml/representations/embedding_pipeline.py`
