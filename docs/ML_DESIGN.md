# ML volume-forecasting design

ExecSim's ML boundary forecasts market volume and uncertainty. The deterministic optimizer remains responsible for inventory, participation, feasibility, impact, risk, and trades. No real historical model is trained or shipped in V1.

## Forecasting tasks

For session `d`, bucket `k`, volume `V_dk`, daily total `D_d=sum V_dk`, and shape `w_dk=V_dk/D_d`:

- Static pre-session: predict `log(D_d)` and the complete `w_d` using data available through session `d-1`.
- Dynamic: at bucket `k`, using data through `k-1`, predict remaining volume `R_dk=sum_(j>=k)V_dj` and conditional future shape `V_dj/R_dk`.

Unconstrained shape scores are converted by stable softmax; positive absolute forecasts are clipped at zero and normalized. Forecast objects retain cutoffs, hashes, schemas, warnings, and optional quantiles.

## Feature system

The registry records dtype, rationale, sources, lookback, transformation, earliest availability, static/dynamic role, missing rule, version, and leakage note. Initial features cover calendar position, lagged session volume/return/range/trade count, rolling ADV/median/volatility and profiles, and point-in-time cumulative/recent volume, returns, realized volatility, inventory, completion, required participation, and remaining time. Availability validation runs before persistence and training.

## Dataset and targets

Builders support 1-, 5-, and 15-minute buckets and create static session rows or dynamic as-of rows from canonical Parquet. PyArrow discovers symbols in bounded batches, projects required columns, and feeds one symbol at a time into feature engineering and immediate partition writes. CLI builds do not retain completed partitions in pandas. Standard, incomplete, and early-close sessions use exchange-calendar metadata. Exclusion or masking reasons are durable; absent buckets are not zero-volume observations.

Targets keep total-volume and compositional shape objectives separate. Manifests contain schema versions, filters, counts, symbols, date range, source hashes, build time, and Git commit.

## Walk-forward evaluation

Default serious-study folds use 252 train, 21 validation, 21 test sessions, 21-session steps, and configurable embargo. Small fixtures may use smaller explicit windows. Cutoffs are global across pooled symbols. Transform fitting occurs on train only; validation selects one configuration; test is evaluated once per locked fold.

## Planned model sequence

1. Rolling mean/median, EWMA, previous-session, and pooled deterministic profiles.
2. Ridge, with optional Elastic Net, as interpretable fitted baselines.
3. Scikit-learn histogram gradient boosting as the first nonlinear model.
4. Later quantile/probabilistic adapters using the same forecast contract.
5. Sequence/deep models only after data scale, stable walk-forward evidence, and downstream TCA benefit justify them.

The intended future corpus is at least 1-2 years, preferably 2-3 years, across at least 50 liquid US equities, preferably 100+, with point-in-time-consistent histories. This task does not download that corpus.

## Training runner and artifact promotion

The runner loads a dataset manifest and schemas, validates folds, instantiates adapters, fits preprocessing on train, selects parameters on validation, optionally refits on train+validation, evaluates locked test, emits out-of-sample forecasts, runs downstream execution evaluation, and writes a checksummed local artifact. Its dry-run reports these actions without fitting.

Artifacts record model/preprocessing, schema and source/split hashes, cutoff/ranges, seed, dependencies, package/Git versions, metrics, downstream TCA, creation time, and checksum. Loading rejects incompatible feature/target schema, horizon, bucket size, timezone, or package contract.

Paper LightGBM execution is operational configuration, separate from the
scientific candidate grid. `LightGBMExecutionOptions` selects CPU or the Windows
OpenCL `gpu` backend, explicit OpenCL platform and device IDs, GPU accumulation
precision, and thread count. CPU is the default. GPU requests must identify the
platform and device explicitly, and the same options apply to every scale and
shape fit across the eight-point grid. GPU execution omits LightGBM's CPU-only
`deterministic` and `force_col_wise` controls.

Native model manifests, grid results, and the stage execution receipt store the
complete execution identity. Loading or resuming rejects missing or different
identity, including CPU/GPU and OpenCL-device mismatches. The stage receipt also
records that GPU execution has no CPU deterministic guarantee. Select
`gpu_use_dp` and thread count using repeated, same-seed synthetic runs and a
synthetic runtime benchmark before historical fitting; do not use historical
validation accuracy for this operational qualification.

Promotion requires predefined leakage, forecast-quality, calibration, stability, and downstream economic gates. Better statistical error alone is insufficient.

## Metrics

Total volume: MAE/RMSE on log volume and robust percentage error. Shape: share MAE/RMSE, stabilized cross-entropy/KL, cumulative-curve error, and one-dimensional cumulative earth-mover distance. Dynamic evaluation adds remaining-volume error, time-of-day error, update improvement, and interval coverage. Economic evaluation adds VWAP tracking, implementation shortfall, modeled cost, completion, capacity shortfall, and regime results against identical deterministic baselines.

## Deliberate V1 non-action

Only tiny synthetic fits in tests verify adapter and training mechanics. V1 does not fit repository history, search hyperparameters on real data, create real fitted artifacts, or report ML performance.

## Paper research extension

The accepted sparse predictive-representation extension is specified separately in [the paper design](PAPER_DESIGN.md). It adds optional PyTorch and LightGBM research paths while preserving this document's forecast-only boundary. Its software and numerical kernels are fixture-qualified; its licensed corpus acquisition, historical model fitting, model selection, test evaluation, and paper claims remain `NOT RUN` without separate authorization.
