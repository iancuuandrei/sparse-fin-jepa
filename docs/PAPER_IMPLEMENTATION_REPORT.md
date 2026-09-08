# Sparse-JEPA paper implementation report

Privileged stage execution follows the separate [paper runtime authorization specification](PAPER_RUNTIME_AUTHORIZATION.md). Runtime approval is external to the six scientifically frozen YAML files and requires a matching command-line opt-in; this operational mechanism does not alter the sparse-jepa-v2 protocol hash.

This report records the active `sparse-jepa-v2` corpus-protocol correction and
the inherited v1 paper software. It distinguishes executable software evidence
from formation data, target data, historical training, and empirical evidence.
It makes no claim about representation quality, forecast accuracy, execution
cost, or economic value.

## 2026-09-08 pre-lock execution addendum

The authorized v2 target corpus, fold-safe sequences, common-RDM selection at
`lambda = 10`, 18 primary JEPA checkpoints, and 18 embedding exports are now
complete. The consolidated embedding audit reports 18 of 18 valid coordinates,
including structural and provenance validation of TRAIN, VALIDATION, and sealed
TEST partitions. No TEST effectiveness metric has been inspected.

The Windows OpenCL LightGBM 4.7.0 build and NVIDIA RTX 5050 qualification pass.
The first historical TRAIN+VALIDATION attempt completed
`fold-1/raw/shared`, then failed before the second coordinate when the process
could not allocate a 1.16 GiB pandas consolidation block. The failure exposed
an operational memory-lifetime defect: completed coordinate frames remained
reachable while the next coordinate was built, and native FP32 context and
embedding values were promoted to FP64. ADR 0020 records the pre-lock
correction. The first model is preserved as superseded evidence and cannot be
reused under the corrected downstream source identity. Historical LightGBM
matrix status remains **INCOMPLETE**; parameter freeze, locked TEST evaluation,
historical TCA, and confirmatory inference remain **NOT RUN**.

## Status

| Evidence class | Status |
|---|---|
| Sparse-JEPA v1 formation protocol | **BLOCKED** - 62 of 505 eligible |
| Sparse-JEPA v2 protocol and formation software | **SOFTWARE READY** |
| V2 formation daily corpus and frozen universe | **FORMATION COMPLETE - AWAITING APPROVAL** |
| V2 target-period Alpaca SIP corpus | **DATA NOT ACQUIRED** |
| Historical JEPA and LightGBM fits | **TRAINING NOT RUN** |
| Locked-test representation, forecast, execution, and paper estimates | **EMPIRICAL RESULT NOT AVAILABLE** |

V1 remains frozen under its original exact-minute formation contract. Its
terminal evidence manifest is `configs/paper/sparse_jepa/v1-evidence-final.json`;
the status is `BLOCKED — INSUFFICIENT FORMATION UNIVERSE UNDER LOCKED
EXACT-MINUTE CRITERION`, the corrected standard-session denominator is 251,
and all target acquisition, training, locked-test inspection, and TCA stages
are `NOT RUN`.

V2 changes only data-quality resolution: provider-native daily formation,
observed-only fixed 15-minute representation tokens, and exact 300-minute TCA
windows. The unchanged 95% concept now measures valid expected daily sessions.
No absent provider minute is interpreted as zero activity. The formation result,
SPY 2021-05-05 token audit, v1/v2 bias diagnostic, resource evidence, and final
v2 terminal status are recorded in `V2_FORMATION_QUALITY_REPORT.md`.

## Provider semantics gate

**PASS.** Current Alpaca documentation supports the v2 interpretation. Stock
minute and daily bars are separately aggregated from trades using tape-,
condition-, and bar-type-specific field rules. A stock bar is emitted only when
all OHLCV fields are nonzero. Consequently, an absent minute aggregate is not
evidence that the minute had zero market activity or zero volume. Alpaca's
higher-interval rule aggregates observed source bars using first open, extrema,
last close, summed volume and trade count, and volume-weighted VWAP. Direct SIP
`1Day` retrieval is available. Source links and the observed-grid volatility
convention are recorded in `docs/RESEARCH_REFERENCES.md`.

## V2 evidence boundary

| Stage | Status |
|---|---|
| Official provider-semantics review | **PASS** |
| V2 protocol documents, ADR 0009, and final freeze | **PASS** |
| Daily formation acquisition | **PASS** - 126,461 rows, 506 symbols |
| Formation candidate rebuild and top-100 freeze | **PASS** - 497 eligible, 100 frozen |
| Formation token-quality and v1/v2 bias diagnostics | **PASS** - 45,464,276 minute rows scanned |
| Target acquisition | **NOT RUN** - prohibited before formation approval |
| Historical model training | **NOT RUN** |
| Locked test and historical TCA | **NOT RUN** |

The current terminal state is `AWAITING V2 FORMATION APPROVAL`. The next
privileged action, target-period acquisition, requires new explicit approval.

## V2 formation validation evidence

| Gate | Result |
|---|---|
| Full repository pytest | **PASS** - 159 tests in 479.10 s |
| V2 focused resolution/formation/report suite | **PASS** - 18 tests |
| Ruff lint | **PASS** |
| Ruff format | **PASS** - 180 files |
| mypy | **PASS** - 125 source files |
| Repository-context validation | **PASS** - 14 areas, manifest hash `e46e53dedff85f56c0888a42ba73a56d6462790fcbf1a3b8cfb8d42464a40de2` |
| Dependency-light smoke | **PASS** - 2 tests |
| V1 immutable artifact audit | **PASS** - original freeze, separately named safe-default receipt, five evidence hashes, and terminal manifest; no Git object database required |
| V2 formation freeze artifact audit | **PASS** - eleven evidence hashes and 100 unique members |
| V2 target corpus path | **NOT CREATED** |

The bounded formation token scan read 45,464,276 minute rows from 1.439 GiB
of monthly responses in 1,516.73 seconds: 29,975.13 rows/s and 2,167.76
token attempts/s. Quality output is 2.78 MiB. The scanner's terminal in-process
RSS field was unavailable due to an incorrect Windows counter binding that was
subsequently fixed and tested. An operating-system lifetime-peak poll during
the run observed 267,558,912 bytes (255.16 MiB), which is reported as a lower
bound rather than an exact terminal peak. The scientific scan was not repeated
solely to improve telemetry.

## Inherited v1 software evidence

The redirected work began from PR head
`74bca4403cd351579221a86736d77210821adde3`. Its baseline was 118 passing
tests, Ruff lint and format, mypy, and repository-context validation over 14
registered areas. The implementation preserves the ordinary
`VolumeForecast -> deterministic MPC -> TCA` boundary and changes no execution
cost or fill mathematics outside the paper subsystem.

## Blockers and exact corrections

| Blocker | Correction |
|---|---|
| The study mixed representation geometry, difficulty adaptation, forecast value, and execution value too early | The frozen evidence chain is now accessibility, observable retention, supervised forecast value, then decision value. Difficulty adaptation is outside the paper matrix. |
| Laplace was the primary sparse target | The matched primary pair is dense Gaussian versus sparse rectified Gaussian with `p=2`, `mu=-0.6744897501960817`, and `sigma=1`. Rectified Laplace is appendix-only. |
| The encoder saw exogenous clock and scale fields | Stored tensors remain 18-wide, but only 13 dynamic features enter the encoder. Five known-at-decision fields condition the predictor. Schema and artifact identities changed incompatibly. |
| Sparse RDM normalization was fixed at 0.5 | Linked first and second moments and target RMS are derived analytically from `(p, mu, sigma)`, persisted, and Monte Carlo checked. |
| Padding and sparse-only initialization could confound geometry | Linked padded latents are structurally zeroed before every predictor and regularizer. Dense and sparse use identical seeded encoder and predictor initialization distributions. |
| RDM strength was selected separately or through gradient calibration | A common coefficient is selected from `{0.1, 1, 10}` on Fold 1 validation, seed 13, using the mean fixed-observable ridge error across both geometries. The six-run receipt is checksummed and test/TCA-blind. |
| Representation evaluation reused JEPA predictors | Encoders are frozen. Horizon-specific affine ridge, MLP-64, and MLP-256 probes are trained independently on TRAIN and selected on validation. Actual future latents are evaluation targets only. |
| Latent error lacked a fixed normalization and baselines | Error is normalized by the TRAIN target-covariance trace, per horizon, and reported beside zero, TRAIN-mean, and persistence baselines. |
| The observable probe was not fixed | Every probe predicts the same causal future-volume-surprise target. Complete origins are tokens 4 through 18 inclusive, so all `{1,2,4,8}` horizons exist. |
| A linear random projection was a weak supervised placebo | The control is now an untrained nonlinear JEPA-shaped network with deterministic Fold 1 seed 13 weights and the same target-free inputs and 644-column interface. |
| LightGBM targets and origin populations were under-specified | Scale predicts residual log remaining volume over the causal historical baseline. Shape is long-form, trained only on bounded predeclared origins, and normalized over valid shrinking future buckets. |
| Historical feature identities could blur distinct clocks | Every artifact distinguishes training cutoff, market as-of time, and feature-history cutoff. Corporate actions carry known-at and effective-at semantics; rolling histories are TRAIN-derived and held out causally. |
| Sparse support diagnostics lacked a predeclared unusual-session statistic | The exact OR composite uses TRAIN 90th-percentile thresholds for volume surprise, realized volatility, and historical-baseline curve error. The same statistic is evaluated on TRAIN and held-out rows, with chance support overlap reported. |
| Minute-by-minute forecast refresh overstated the decision frequency | The paper runner commits each 15-minute segment. It compares against a realized-volume allocation oracle and reports normalized allocation regret, while the existing minute-level engine remains unchanged. |
| Seed forecasts were averaged in the primary study | Every seed is a primary replicate. Effects are paired within seed and averaged as metrics; a three-seed forecast ensemble is appendix-only. |
| Selection and locked-test stages were not separated by a durable freeze | After all validation-only representation and LightGBM selection completes, a parameter-freeze receipt records the config, Git head, common-lambda receipt, and every selected model checksum. Locked-test stages refuse to run without it. |
| Resume could lose trainer decision state | Periodic trusted state now includes optimizer, scheduler, Python/NumPy/CPU/CUDA RNGs, sampler/RDM counters, early-stopping state, best eligible weights, diagnostics, epoch, and global step. Safe weights remain separate safetensors. |
| Reporting could substitute synthetic or incomplete tables | Historical reporting requires four named nonempty schemas and measured intervals. Synthetic output remains explicitly synthetic. Missing historical artifacts fail closed. |
| Compute feasibility had no executable gate | The run matrix is capped at 29 representation runs. Completed sequence manifests produce step, row, and storage bounds before training; configured safety limits block oversized work. A bounded kernel profiler makes no GPU-hours promise without representative evidence. |

## Mathematical target contract

For the symmetric generalized-Gaussian parameterization, let
`a = p^(1/p) sigma`. For `mu <= 0`, define `t = |mu| / a`,
`G_k = Gamma((k+1)/p, t^p)`, and `D = 2 Gamma(1/p)`. Then

```text
E[ReLU(X)]   = (mu G_0 + a G_1) / D
E[ReLU(X)^2] = (mu^2 G_0 + 2 mu a G_1 + a^2 G_2) / D
target_rms   = sqrt(E[ReLU(X)^2])
```

For positive `mu`, the implementation subtracts the corresponding negative
tail moments from the full raw moments. The incomplete-gamma form avoids
numerical quadrature. Gaussian and Laplace 50%, 75%, and 87.5%-zero settings
are independently checked against deterministic Monte Carlo samples. The
primary sparse target has 75% expected zeros; its RMS is derived rather than
borrowed from the old Laplace target.

## Executable historical architecture

```text
constituent snapshot + formation bars
  -> formation statistics, eligibility, and exclusion receipts
  -> frozen 100-stock universe with stable instrument and sourced symbol history
  -> validated stock + SPY minute corpus and point-in-time action manifest
  -> v2 token-valid fold sequence stores with observed-bar diagnostics (18 observed = 13 dynamic + 5 conditioning)
  -> TRAIN-only normalizers and deterministic sample indexes
  -> common-lambda selection and streaming dense/sparse JEPA training
  -> frozen accessibility and observable probes
  -> compatibility-bound 640 latent + 4 availability embedding partitions
  -> raw/untrained/dense/sparse residual-scale and long-shape LightGBM models
  -> causal VolumeForecast providers
  -> 15-minute committed MPC and realized-volume allocation oracle
  -> exact complete-case contrasts and fold-stratified block bootstrap
  -> named historical tables, main figures, and appendix artifacts
```

The six YAML files form one validated configuration with one canonical hash.
Artifacts also store upstream manifest, fold, cutoff, normalization,
architecture, target, training, checkpoint, and framework identities. Existing
bytes never establish compatibility by themselves.

## Streaming trainer and reproducible resume

The historical trainer reads Parquet sessions lazily through a bounded cache.
Each train epoch selects exactly two deterministic origins per session;
validation and test use complete deterministic grids. Workers are Windows
spawn-safe, use bounded prefetch, and do not own leakage-bearing mutable state.
CUDA uses pinned memory and BF16 where supported; CPU and unsupported devices
fall back to FP32. Training uses AdamW, 5% warmup, cosine decay, clipping, at
most 40 epochs, and patience six.

The trainer fails immediately on non-finite loss, gradients, or RDM output. Its
2,048-projection FP32 diagnostic uses a deterministic uniform sample capped at
2,048 valid actual latents. It selects only validation checkpoints passing
collapse gates and writes latest, best, and final safetensors. A deterministic production-path fixture proves
that continuous training and a forced interruption followed by checksummed,
explicitly trusted resume produce exactly equal final tensors.

## Frozen probes and supervised targets

P0 is four independent horizon-specific affine ridge models. P1 and P2 are
separate MLP-64 and MLP-256 probes; none reuse a geometry-specific JEPA
predictor. Probe parameters, approximate MACs, and measured inference time are
persisted beside normalized error and fixed baselines.

The remaining-volume model has one row per `(instrument, session, as_of)` and
predicts the residual over the causal baseline in log space. The shape model
has one row per `(instrument, session, as_of, target_bucket)` and predicts
`log(conditional_share + 1e-6)`. A native pandas categorical vocabulary is fit
on TRAIN, persisted, reused unchanged, and checked on held-out data. The exact
eight-candidate LightGBM grid is identical across raw, untrained, dense, and
sparse rows. Validation selects scale and shape independently; test and TCA
never select models.

Embedding export includes only the current observed latent, predicted
`h1/h2/h4/h8` latents, and four explicit availability flags: 644 values total.
Unavailable horizons are zero-filled and marked unavailable. Actual future
target latents are never exported.

## Execution and inference contract

At each 15-minute boundary the learned provider resolves causal normalized
context and the matching frozen embedding, predicts residual scale and valid
future-bucket shape, disaggregates with a TRAIN-only within-bin profile, and
returns an ordinary `VolumeForecast`. The paper execution policy commits the
resulting schedule for the next 15-minute segment. The established provider
also supports causal truncation at +1, +7, and +14 minutes without restarting
the within-token profile.

The matched primary execution rows are EWMA, raw LightGBM, untrained nonlinear
control, dense JEPA, and sparse JEPA, with seed-specific JEPA outputs. Symbol,
date, order, side, market bars, constraints, cost assumptions, optimizer, and
window are held fixed. The realized-volume oracle is an allocation reference,
not a feasible live strategy. Normalized allocation regret is the principal
decision metric; completion and implementation shortfall are diagnostics.

## Statistical and reporting contract

Duplicate method/case rows are rejected, exact common cases are intersected,
and paired differences are formed before date aggregation. The primary
moving-block bootstrap uses five-trading-day blocks within fold, 10,000
replicates, and fold-stratified contribution. Block lengths one and ten are
appendix sensitivities. The five confirmatory contrasts are named in the
design freeze; if formal claims are made, their p-values use Holm adjustment.
No arbitrary success threshold can turn an empirical outcome into a software
gate.

Main historical tables are dataset/folds/exclusions, representation
accessibility, forecasting, and execution. Main figures are accessibility by
probe capacity, forecast performance by model, forecast error by as-of, and
paired allocation regret with measured intervals. Support/regime, sparsity,
Laplace, block-length, order-size, and optional ensemble material remains in
the appendix.

## Executed software evidence

The deterministic acceptance corpus contains four synthetic stocks plus SPY,
40 sessions spanning two paper folds, causal seasonal history, a known-before-
effective split, and a malformed excluded session. Tiny models and epochs
exercise the same corpus, streaming, checkpoint, probe, embedding, supervised,
provider, MPC, matching, bootstrap, and historical-schema paths intended for
the future historical run.

| Gate | Result |
|---|---|
| Pre-redirect complete pytest | `PASS` - 118 tests |
| Redirected multi-session production-path fixture | `PASS` - 1 test in 183.07 s; oracle/diagnostic rerun 2 tests in 185.26 s |
| Post-redirect complete pytest | `PASS` - 134 tests in 199.48 s |
| Ruff lint | `PASS` |
| Ruff format | `PASS` |
| mypy | `PASS` - 118 source files |
| Repository-context validation | `PASS` - 14 registered areas, context hash `75c2ad3276e5c434a7ccdd6d24e7616a74cbb32e3d38808327a4c58a9f9ded52` |
| Ordinary ExecSim without PyTorch/LightGBM | `PASS` - guarded subprocess regression in the complete suite |
| Real network request | `NOT RUN` |
| Historical representation or LightGBM fit | `NOT RUN` |
| Locked historical evaluation or TCA | `NOT RUN` |

A bounded local CPU profile, which is not a historical runtime estimate,
measured 8,141.60 streaming sequence rows/s over 616 fixture rows; 18,702.10 RDM
rows/s for a 256-row, 512-projection batch; 37,050.44 embedding rows/s; and
15.75 tiny TCA simulations/s. GPU peak memory and GPU hours are `NOT MEASURED`.
Historical sequence counts and storage remain unknown until the real corpus
manifests exist.

## Files changed by responsibility

- Protocol and governance: `docs/PAPER_DESIGN.md`, ADR 0008, the ADR index,
  leakage contract, implementation directions/log, navigation, research
  references, repository manifest, and all paper YAML/config-freeze files.
- Sequence and data identities: the paper corporate-action module and sequence
  schemas, builder, corpus, indexes, manifests, and streaming loader.
- Representation system: target schemas and moments, encoder/JEPA/predictors,
  checkpoints, historical trainer, diagnostics, selection, frozen evaluation,
  embeddings, and export pipeline.
- Forecast and execution system: LightGBM adapter/data frames, placebo features,
  forecast provider, orchestration/CLI, regime diagnostics, TCA, matching,
  statistics, compute planning, and reports.
- Executable evidence: paper data, sequence, representation, and pipeline test
  modules. Ordinary ExecSim modules and behavior remain outside this change.

## Privileged stages still not run

- Alpaca authentication, SIP entitlement, formation download, and target-period
  download: **DATA NOT ACQUIRED**.
- Real-universe freeze and historical corpus materialization: **DATA NOT
  ACQUIRED**.
- The 29-run ceiling is a plan, not executed work. Historical common-lambda,
  primary dense/sparse, and appendix representation fits: **TRAINING NOT RUN**.
- Historical LightGBM grids: **TRAINING NOT RUN**.
- Locked-test accessibility, forecast, regime, execution, bootstrap, and final
  paper estimates: **EMPIRICAL RESULT NOT AVAILABLE**.

No synthetic result is an empirical paper result. The inherited v1 software
evidence does not establish v2 formation readiness. V2 is not complete until
its daily candidate rebuild, frozen-universe decision, resolution diagnostics,
resource measurements, checksummed report bundle, and current repository gates
have executed.
