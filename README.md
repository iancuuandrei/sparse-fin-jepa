# Sparse Fin-JEPA

Research code for **Do Sparse JEPAs Simplify Intraday Market Dynamics?** The
`sparse-jepa-v2` historical experiment completed and its results were frozen on
14 September 2026. See the [experiment record](docs/EXPERIMENT_RESULTS.md) for
completion evidence, exact source and artifact identities, result interpretation,
and reproducibility limits.

The frozen comparisons show mixed evidence, not general sparse-JEPA superiority:
sparse representations have worse affine latent-accessibility and remaining-volume
errors, but slightly better conditional volume-curve error. These are historical
research results under fixed assumptions, not live trading or profitability claims.

## Research architecture

The study compares matched dense Gaussian and sparse rectified-Gaussian shared-encoder
JEPAs across three fixed folds and seeds 13, 29, and 47. Frozen probes test
representation accessibility and observable information retention. LightGBM tests
forecast value, and deterministic replay tests execution-allocation value.

```text
Point-in-time sequences → frozen JEPA representations → volume forecasts
                                                        ↓
                                  deterministic MPC → matched TCA
                                                        ↓
                                      paired statistics → frozen report
```

The matrix contains 18 primary trained JEPA coordinates, 18 embedding exports,
and 24 selected LightGBM coordinates. Acquisition, fitting, and locked evaluation
remain separately authorized operations; completing this study does not enable
them by default for a fresh checkout.

## Execution research framework

Sparse Fin-JEPA uses **ExecSim**, the underlying offline execution-research engine,
to compare intraday parent-order execution policies. The engine provides causal
volume forecasts, static and adaptive policies, a transparent spread-and-impact
model, constrained optimization, transaction-cost analysis (TCA), reproducible
experiments, and point-in-time ML infrastructure.

The project is research software, not a live trading system, broker, order router,
alpha model, or claim of production readiness. The package and CLI retain the
technical name `execsim`; commands and imports below deliberately use that name.

## Install the project

Create an isolated environment and install the development dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

Verify the installation:

```powershell
.\.venv\Scripts\python.exe -m execsim.cli smoke
.\.venv\Scripts\python.exe -m ruff check src tests scripts
.\.venv\Scripts\python.exe -m mypy src
.\.venv\Scripts\python.exe -m pytest -q
```

## Run a simulation

Run any deployable policy through one command:

```powershell
.\.venv\Scripts\python.exe -m execsim.cli simulate --strategy optimal --symbol AAPL --trade-date 2026-03-23 --quantity 5000 --start-time 10:00 --end-time 11:00 --json
```

Supported policies are `twap`, `vwap`, `pov`, `almgren-chriss`, `optimal`, and `mpc`. The original `simulate-twap` command remains a compatibility alias.

## Run a reproducible experiment

Review `configs/experiment.yaml`, then run the configured grid:

```powershell
.\.venv\Scripts\python.exe -m execsim.cli experiment run --config configs/experiment.yaml
```

Each stable run ID receives raw Parquet results, aggregate and paired CSV statistics, a configuration snapshot, provenance, figures, and a Markdown report under `reports/runs/`.

## Prepare ML research data

Build and validate point-in-time rows without fitting historical data:

```powershell
.\.venv\Scripts\python.exe -m execsim.cli ml build-dataset --mode static --bucket-minutes 5 --output-root data/ml
.\.venv\Scripts\python.exe -m execsim.cli ml validate-dataset --manifest data/ml/DATASET_ID/manifest.json
```

Replace `DATASET_ID` with the ID printed by the build command. Historical model fitting is disabled by default; the test suite fits only tiny synthetic fixtures.

## Inspect the frozen experiment protocol

Install the optional paper stack only for fixture or separately authorized research paths:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[paper]"
```

Expand the locked experiment without downloading data or fitting history:

```powershell
.\.venv\Scripts\python.exe -m execsim.cli ml paper plan --dry-run
```

The workflow requires separate runtime approval and command-line authorization for
network acquisition, historical fitting, and full evaluation. The
[paper design](docs/PAPER_DESIGN.md) defines the frozen scientific contracts.
The [implementation report](docs/PAPER_IMPLEMENTATION_REPORT.md) distinguishes
current completion from dated development evidence. A dry-run plan does not
download the licensed corpus, reproduce the historical study, or open TEST.

The Git repository contains code, configuration, specifications, and tests—not
the licensed market corpus, trained models, or full generated result bundle.
See [artifact access and verification](docs/EXPERIMENT_RESULTS.md#artifact-access-and-verification)
before attempting reproduction. Never substitute a fresh model or a different
data snapshot for a checksum-bound frozen input.

## Navigate the repository

Use the manifest-backed context selector to find ownership, specifications, and focused checks:

```powershell
.\.venv\Scripts\python.exe scripts/repo_context.py --list
.\.venv\Scripts\python.exe scripts/repo_context.py --path src/execsim/optimization/qp.py --json
```

Start with these documents:

- [Experiment results and provenance](docs/EXPERIMENT_RESULTS.md) records the completed study and its limitations.
- [Documentation index](docs/README.md) separates research, implementation, and operational guides.

- `docs/standards/implementation.md` defines code and documentation practice and records engineering directions.
- `docs/ADRs/README.md` indexes architectural choices, rationale, and rejected alternatives.
- `docs/SPECIFICATIONS.md` is the normative component contract.
- `docs/MATHEMATICAL_MODEL.md` defines formulas, units, constraints, and signs.
- `docs/DATA_LEAKAGE_CONTRACT.md` defines the point-in-time boundary.
- `docs/ML_DESIGN.md` defines ML data, splits, training, and artifacts.
- `docs/PAPER_DESIGN.md` defines the optional sparse predictive-representation study.
- `docs/PAPER_IMPLEMENTATION_REPORT.md` classifies its software acceptance gates and empirical limitations.
- `docs/NAVIGATION.md` maps repository areas.

## Understand the evidence boundary

Historical replay keeps market bars exogenous and applies assumed costs to simulated fills. Minute bars do not expose quotes, queue position, within-bar paths, or counterfactual market response. Reports describe only the selected bars and assumptions; they do not establish strategy superiority or expected live performance.
