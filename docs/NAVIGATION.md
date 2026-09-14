# Repository navigation

Sparse Fin-JEPA uses a small manifest-backed navigation system rather than Nx.
Its execution engine and Python package retain the name ExecSim (`execsim`).
The repository is a single Python package; adding a Node task graph would
duplicate packaging and dependency management without improving current
ownership or build performance.

## Start here

- Reader-oriented documentation index: [Documentation guide](README.md)
- Completed historical experiment, results, and artifact identities: [Experiment record](EXPERIMENT_RESULTS.md)

- Implementation directions and writing standard: `docs/standards/implementation.md`
- Architectural decisions and rationale: `docs/ADRs/README.md`
- Normative implementation behavior and invariants: `docs/SPECIFICATIONS.md`
- Mathematical derivations: `docs/MATHEMATICAL_MODEL.md`
- Point-in-time rules: `docs/DATA_LEAKAGE_CONTRACT.md`
- ML design and future training procedure: `docs/ML_DESIGN.md`
- Sparse-JEPA paper contracts and locked experiment: `docs/PAPER_DESIGN.md`
- Operational paper-stage authorization: `docs/PAPER_RUNTIME_AUTHORIZATION.md`
- Sparse-JEPA implementation evidence and gate classification: `docs/PAPER_IMPLEMENTATION_REPORT.md`
- Sparse-JEPA v2 formation evidence and approval gate: `V2_FORMATION_QUALITY_REPORT.md`
- Provider and research-method sources: `docs/RESEARCH_REFERENCES.md`
- Component ownership and validation: `repo_manifest.yaml`
- Project goals and non-goals: `docs/PROJECT_CONTEXT.md`
- Completed work and validation history: `docs/IMPLEMENTATION_LOG.md`

## Deterministic context selector

```powershell
.\.venv\Scripts\python.exe scripts/repo_context.py --list
.\.venv\Scripts\python.exe scripts/repo_context.py --area simulation --json
.\.venv\Scripts\python.exe scripts/repo_context.py --path src/execsim/simulator/core.py --json
.\.venv\Scripts\python.exe scripts/repo_context.py --check
```

The selector reports the owning area, relevant specifications, tests, and verification commands. Long-running agents should use it before a scoped change instead of loading unrelated repository content.

## Top-level map

| Path | Role |
|---|---|
| `src/execsim/data/` | Canonical minute bars, loading, cleaning, validation, scenarios |
| `src/execsim/policies/` | Static and adaptive execution decisions |
| `src/execsim/forecasting/` | Point-in-time deterministic forecast providers |
| `src/execsim/optimization/` | QP construction, analytical reference, integer projection |
| `src/execsim/costs/` | Realized execution-price and cost models |
| `src/execsim/simulator/` | Planning/fill orchestration and results |
| `src/execsim/experiments/` | Config-driven research grids and durable outputs |
| `src/execsim/reporting/` | TCA aggregation, statistics, figures, Markdown reports |
| `src/execsim/ml/` | Point-in-time datasets, splits, adapters, training, artifacts |
| `src/execsim/data/paper/` | V1 evidence, v2 daily formation, resolution-specific token/minute quality, stable identities, SIP acquisition, corporate actions, and manifests |
| `src/execsim/ml/sequences/` | Corpus-wide fold stores, fixed session tensors, streaming sample indexes, and train-only normalization |
| `src/execsim/ml/representations/` | Optional matched dense/rectified-sparse JEPA, streaming training and resume, frozen capacity/observable probes, embeddings, diagnostics, and safe checkpoints |
| `src/execsim/ml/paper/` | Unified configuration, resource planning, parameter freeze, historical orchestration, learned forecasts, matched TCA/statistics, and main/appendix reports |
| `configs/paper/sparse_jepa/` | Immutable blocked v1 protocol and terminal evidence |
| `configs/paper/sparse_jepa_v2/` | Active v2 daily/token/minute protocol, design freeze, and authorization defaults |
| `configs/` | Versioned assumptions and runnable workflows |
| `scripts/` | Repository checks, deterministic benchmarks, and profiling tools |
| `tests/` | Unit, invariant, integration, and CLI evidence |
| `docs/ADRs/` | Durable architectural decisions, alternatives, and consequences |

Paths may be introduced incrementally, but once present they must be registered and specified.
