# Sparse-JEPA v2 experiment record

The historical experiment for **Do Sparse JEPAs Simplify Intraday Market
Dynamics?** completed on 14 September 2026. Its terminal state is
`FINAL-RESULTS-FROZEN`. This document is a reader-facing record, not a new
scientific artifact or an amendment to the frozen protocol.

## Scope and completion

The evidence chain tests representation accessibility, observable information
retention, volume forecasting, and execution allocation. The primary geometry
comparison is dense Gaussian versus sparse rectified Gaussian—not a primary
Gaussian-versus-Laplace comparison. Both use shared encoders without EMA or
stop-gradient, fixed forward horizons, matched initialization, and frozen folds.

The completed inventory contains 18 primary JEPA checkpoints, 18 embedding
exports, 24 selected LightGBM coordinates, forecast and representation results,
250 TCA date shards, and a historical report with tables, figures, and appendices.
The report/freeze supervisor finished successfully at
`2026-09-14T16:58:30.640209+00:00`. Software completion is not a positive
scientific finding.

## Frozen identity

These identities belong to the completed run. Later documentation commits must
not be presented as the source that generated these results.

| Identity | Value |
|---|---|
| Protocol | `sparse-jepa-v2` |
| Scientific configuration SHA-256 | `e0a926869b8882ff3a3f031bda046786ccaf9194ad9a6f1e4a5b7c9846609ed1` |
| Parameter freeze SHA-256 | `cec151b977cc571a306dc34b9758d9228e773eed63917e8a84d59023124f8b48` |
| Report/freeze source commit | `8d83f845a689cf09e04928e41a2c96c2c9926075` |
| Report/freeze source tree | `91754c5cc0a996f8a1f797df3fe18a5ca4dafb2b` |
| Execution namespace | `evaluation-executions/8d83f84` |
| Execution receipt SHA-256 | `b28091a7249e3f621d855b5fa203683d3bcca6c8007a7b76732c424915a7be7e` |
| Final result freeze SHA-256 | `d8ab5f088e12b746fdc7490e6a5e98a347fc7682d6237d384e6da3339edf5e21` |

Recovery preserves original stage authorship through typed inheritance receipts.
The final report does not claim that all stages ran under one evaluator revision.

| Stage | Producing commit | Manifest SHA-256 |
|---|---|---|
| Forecast | `6402e40d42d8ff5f5e09d77611e23f3646d46ca7` | `32a7350f44c8c4ed4b31801d09c7b08e9db2c38ddaccaf7d1b8c88a1ff6f42ce` |
| Representation evaluation | `6402e40d42d8ff5f5e09d77611e23f3646d46ca7` | `a325678a138a9e6fc715ba175f69b20b47e73aba7edb0f57b58dda9d4c39fcbb` |
| TCA | `d0f8877c8106f81263a4a999ba7b371176fa9009` | `fb434d28747a8056305e74ac4322256670e61845d458959ab0c9b925be255732` |

The report-only recovery is documented in [ADR 0038](ADRs/0038-inherit-completed-tca-for-report-only-recovery.md).
Earlier partial executions remain provenance, not interchangeable result sources.

## Confirmatory results

The following rounded values are transcribed from the frozen
`confirmatory_statistics` table. Differences are sparse minus baseline; all
listed endpoints are errors, so a positive difference is worse. Each contrast
uses 250 paired dates, fold-stratified five-day blocks, 10,000 bootstrap
replicates, and Holm adjustment across the five predeclared tests.

| Endpoint | Baseline | Mean difference | 95% interval | Holm-adjusted p | Direction for sparse |
|---|---|---:|---|---:|---|
| Affine normalized latent error | Dense | +0.10236331 | [0.09459493, 0.10936010] | 0.00049995 | Worse |
| Log remaining-volume MAE | Dense | +0.00071300 | [0.00025175, 0.00116841] | 0.00539946 | Worse |
| Conditional-curve error | Dense | −0.00003178 | [−0.00004812, −0.00001534] | 0.00119988 | Better |
| Log remaining-volume MAE | Raw | +0.00071300 | [0.00028371, 0.00109973] | 0.00079992 | Worse |
| Conditional-curve error | Raw | −0.00002531 | [−0.00004332, −0.00000340] | 0.01319868 | Better |

The result is mixed: small shape-error improvements do not establish simpler
latent dynamics or a general forecasting advantage. Rejecting all five nulls
does not mean five favorable findings; three directions are adverse to sparse.
Statistical significance does not establish practical value.

## Interpretation limits

Forecast coverage depends on the comparison. The all-method table uses 532,222
common cases, while the learned-model pairwise comparisons use 533,236 sample
cases per seed. The difference is 1,014 cases unavailable to EWMA; it must not
be described as a universal exclusion from every confirmatory comparison.
Seed-matched rows are not independent dates.

TCA is downstream simulated execution under assumed costs and exogenous bars.
Its sparse-versus-dense effects vary by seed. Completion rates and unavailable
cases must accompany cost and regret comparisons; a smaller absolute cost is
not evidence of better execution when less inventory is filled. TCA contrasts
are not additional members of the five-test confirmatory family.

This summary does not certify fold-by-fold robustness, liquidity-subgroup
robustness, every sensitivity analysis, or publication readiness. A complete
independent scientific audit and visual inspection of all figures remain
separate from the recorded software completion. Do not infer live profitability,
market impact calibration, or general sparse-JEPA superiority.

## Artifact access and verification

Git contains the implementation and protocol, not the licensed raw corpus,
model weights, private operational approvals, or generated historical payloads.
No public artifact download is currently provided by this document. Reproducing
the run requires access to those retained, checksum-bound inputs and the recorded
environment; installing the package alone is insufficient.

Under the retained protocol artifact root, the final paths are:

```text
evaluation-executions/8d83f84/execution.json
evaluation-executions/8d83f84/selection/final-result-freeze-v1.json
evaluation-executions/8d83f84/reports/sparse-jepa-v2/completion.json
evaluation-executions/8d83f84/reports/sparse-jepa-v2/provenance.json
```

Begin with the final freeze digest above, then validate its referenced result
files and original producer manifests. Follow execution and inheritance receipts
recursively; do not reconstruct ancestry from shortened commit names. Use the
[evaluation verification contract](EVALUATION_IMPLEMENTATION.md) with the original
source and correctly relocated runtime roots. Verification must not create a
new execution or rerun a historical stage.

The read-only results extraction checked 59 result files, seven stage receipts,
and 24 LightGBM manifest digests. That bounded check is not a fresh complete
recursive verification of all upstream bytes; the latter was not completed by
the subsequent interpretation audit. This documentation update performs no new
historical computation or model fitting.

## Software evidence

The report-only recovery candidate passed 613 tests, Ruff lint and format,
mypy, and repository-context validation before merge. GitHub CI for the merged
`8d83f84` source completed successfully. These are source-qualified software
checks, distinct from the historical result-freeze receipt and scientific
interpretation. See [PR 12](https://github.com/iancuuandrei/sparse-fin-jepa/pull/12)
for the recovery change and [implementation history](PAPER_IMPLEMENTATION_REPORT.md)
for earlier acceptance snapshots.
