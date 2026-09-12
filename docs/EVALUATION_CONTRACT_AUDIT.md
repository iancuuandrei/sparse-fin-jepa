# Historical evaluation production-contract audit

Baseline: `a9e97ba9de314e1f0ff82ff30e6a6427dec833b3`, tree
`09afcff2eacb26e40c7a00bc91eceb0e60b327f3`.

This is software qualification, not evidence of model effectiveness. The failed
a9 execution is immutable. A corrected evaluator must use a new source-qualified
namespace and rerun its own stages. No frozen scientific configuration, models,
sample populations, optimization, or statistical estimators are changed.

## Contract matrix

Paths below are relative to `src/execsim/ml/` unless prefixed with `tests/`.
Coverage names identify executable checks, not an assertion that unexecuted gates
have passed. Final local, hosted, and pod qualification are recorded separately.

| Boundary | Producer | Consumer | Required identity/schema | Existing coverage | Gap found | Fix/test |
|---|---|---|---|---|---|---|
| Sequence → regime frame | `sequences/streaming.py`, indexed `SequenceSample` | `paper/lightgbm_data.py::build_historical_baseline_regime_frame` | Canonical sample/session IDs, instrument, date, origin, causal statistics | Legacy builder parity | Production omitted session ID | Propagate `sample.session_id`; `test_representation_contracts.py` |
| Regime frame → labels | Historical regime builder | `paper/regimes.py` thresholds and labels | TRAIN-only thresholds; metadata preserved | Threshold unit tests | Producer boundary mocked in resume test | Actual corpus → TRAIN thresholds → TEST labels regression |
| Labels → embedding diagnostics | Labeled state frame | `paper/orchestration.py::_stream_embedding_diagnostics` | Unique ordered sample IDs; matching exported session grouping | Hand-built state fixture | Missing identity and unchecked forged session grouping | Compare exported session ID; missing/duplicate/reordered/extra sample adversaries |
| Sequence → embedding export | `PaperSequenceDataset` | `representations/embedding_pipeline.py` | Stable session/origin order, source/checkpoint/sequence identity | Export and cache tests | No exact cross-producer ordering proof | Actual dense/sparse export versus consumer-ordered regime IDs |
| Probe output → coordinate | Frozen capacity streaming evaluator | Representation stage publication | Capacity, observable, date and support schemas; coordinate identity | Mocked resume and probe equivalence | Diagnostics boundary skipped | Real stage test with probes, labels and diagnostics, injected publication failures |
| Coordinate → representation merge | Atomic coordinate shards | `paper/evaluation_artifacts.py::merge_result_shards` | Checksums, source identity, unique keys, complete inventory | Merge/resume tests | No demonstrated production defect | Real stage completion and changed-source/corruption rejection |
| Forecast ledgers → TCA preflight | Learned forecast and EWMA workers | `paper/tca_workers.py::preflight_tca_ledgers` | Manifest fold/cutoff, complete integer origins/buckets, availability | Indexed corruption and EWMA multi-session tests | Fractional keys truncated before lookup | Reject noninteger/null token keys before validation |
| TCA inputs → worker | TCA stage date inputs | `TCAWork`, `run_tca_work` | Exact-window case population, positive causal ADV, date/source identity | Date-slicing test mocked worker/preflight | Fragmented orchestration coverage | Production-shaped integration using real preflight, providers and replay |
| Learned ledger → learned provider | Published scale/shape | `PaperForecastLedgerProvider` | Exact keyed origin, shrinking bucket grid, cutoff; null seed allowed | Provider boundary/parity tests | No learned ledger through actual replay | Real learned + EWMA worker fixture |
| EWMA ledger → EWMA provider | `run_ewma_work` | `EWMAForecastLedgerProvider` | Manifest fold, per-session requests, `end_token=24`, typed unavailability | EWMA production and preflight tests | Worker integration was EWMA-only | Same combined production fixture; no invented scale fold column |
| TCA shard → merge | `run_historical_tca`, worker publication | TCA stage merge | Fold/date/instrument/method/order fraction uniqueness, complete-case fields | Generic merge tests | No actual learned worker rows through merge | Merge actual fixture worker output |
| Forecast merge → report | `forecast_metric_frame`, merged forecast | `_build_report_stage` | Nullable method seed; fold/date/instrument/origin, named errors | Mocked report builder | Empty all-method intersection rejected despite valid pairwise contrasts | Allow only two schema-valid descriptive summaries to be empty |
| Representation merge → report | Capacity/date/support artifacts | `_build_report_stage` | Geometry/seed/horizon/capacity, probe baselines, date identity, diagnostics | Writer-only fixtures | Real table transformations untested | Real representation producer and report-stage fixture |
| TCA merge → report | TCA main/sensitivity output | Complete-case pairing and report | Exact 12-field case identity, method/status and named metrics | Separate pairing/replay tests | No whole-chain report proof | Actual worker schemas; unavailable rows remain typed and excluded only by availability |
| Report tables → bundle | `_build_report_stage` | `write_historical_paper_bundle`, `publish_bundle` | Eleven named table schemas, figures, appendices, provenance | Builder mocked in bundle test | Valid percentile interval can exclude observed estimate | Render absolute endpoints and separate estimate; real synthetic-labeled report fixture |
| Bundle → final freeze | Atomic report completion | `write_final_result_freeze` | Exact numerical inputs, source/tree/config/freeze, bundle checksums, 18/24 upstream inventories | Mocked report tree | Default path skipped completion; declared input subset accepted | Unconditional completion and exact input inventory validation; corruption regressions |
| Supersession/execution → next reseal | Typed supersession and execution writers | Recursive execution verification | Original authorization distinct from immediate predecessor and current source; checksummed ancestors | Multi-generation and ancestor mutation tests | No new demonstrated schema defect | Retain checks; new execution must immediately supersede a9, never reuse its results |

## Ordering and identity argument

Sequence records define canonical session identity as instrument ID followed by
the ISO session date. The evaluation/export dataset orders by session ID and
origin. Diagnostics orders regime states by instrument ID, date and origin.
For these canonical records the orders coincide. The regime builder's initial
index traversal need not itself be in export order; the consumer's actual sort
is part of the tested contract. Tests compare exact ID sequences, not merely
whether each frame independently reports itself sorted.

The added session field contains no new market information. Legacy parity still
compares every existing scientific column exactly. Canonical metadata is asserted
against the real sample rather than a broken reference omission. Timing remains
non-scientific telemetry and is not converted into a deterministic estimator.

## Interruption and reuse

Incomplete probe-cache materialization is not a completed cache. Completed caches
retain checkpoint, sequence, source, partition, device, batch and Torch identity,
checksums, exact batch boundaries and RNG transitions. A valid coordinate is
verified before resume skips its model/probe work. Interruption after publication
but before cache retirement may leave disposable cache bytes; that is not a result
validity failure and does not justify changing scientific or checksum semantics.
Different evaluator source identity rejects both old coordinate and cache reuse.

## Real-artifact read-only evidence

The bounded baseline pod audit read metadata for all 18 immutable TEST exports and
318 forecast ledgers. Export sample/session order agreed with the frozen indexes
and diagnostic ordering in every coordinate. The merged forecast contained
4,798,110 rows; only schema and counts were inspected, not effectiveness metrics.
The a9 namespace file inventory was unchanged, and its execution receipt remained
`1f4f0b4b0999934a6a5b8de7ae5e55d67e23f48877a9b6c0b192b665961c7e2c`.
This establishes baseline metadata compatibility, not final-source qualification
or permission to reuse old results. Final-source qualification remains a separate
required gate before restart.

## Confirmed defects and scope

The original missing-session regression failed with the same error as production:
`Regime state frame is missing columns: ['session_id']`. Other changes enforce
identity keys or permit rendering of already-valid statistical output; none
changes an estimator to avoid an error. Empty descriptive intersections do not
replace the independently matched confirmatory population. Percentile interval
endpoints remain unchanged even when the observed estimate lies outside them.
Report fixtures are explicitly synthetic and do not claim acquisition/training.

Final-freeze regressions also cover aggregate manifest claims: representation
hashes must match all three merged outputs; TCA schema/config/execution identity,
canonical paths and hashes must match both merged outputs. A report checksum does
not establish that a separate manifest's assertions are true. New freeze receipts
live outside the completed report bundle in both default and relocated modes;
the default-path regression reproduced the former second-freeze inventory error
and now checks successful repeated freeze and report reuse.

Cache arithmetic, provider forecast mathematics, EWMA estimation, TCA population,
ADV share basis, replay, bootstrap and chained provenance were inspected and are
not redesigned. A passing synthetic integration does not establish historical
completion. The task requires full local gates, exact-head CI, committed-source
pod qualification and a verified detached fresh launch before completion.
