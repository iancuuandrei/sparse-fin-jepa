# Implementation standard

This living standard records material implementation directions and defines how ExecSim code, specifications, command examples, research claims, and reference documentation are written. Update the direction record in the same change whenever the repository adopts or reverses an architectural, mathematical, dependency, data, or research-method decision.

## Direction record

The active implementation directions are:

| Decision | Direction | Reason |
|---|---|---|
| Repository navigation | Use `AGENTS.md`, `repo_manifest.yaml`, `docs/NAVIGATION.md`, and `scripts/repo_context.py`; do not add Nx | A single Python package does not justify a second Node dependency and task graph |
| Decision history | Record material choices in indexed ADRs and supersede rather than rewrite accepted records | Specifications define current behavior, while ADRs preserve the reason and rejected alternatives |
| Policy information boundary | Give policies a point-in-time `DecisionContext`, not an unrestricted target-session data frame | The boundary makes future-data access enforceable and testable |
| Optimization core | Use an explicit OSQP convex quadratic program, with a separate analytical Almgren–Chriss reference | The QP exposes feasibility, participation constraints, matrices, residuals, and deterministic integer projection |
| Realized cost model | Use half-spread plus linear-in-participation temporary price impact | The resulting total impact cost is transparent, convex, and consistent between planning and simulation |
| Historical replay | Keep the replayed market path exogenous | Minute bars cannot model counterfactual market response without unsupported assumptions |
| ML role | Forecast point-in-time volume inputs; keep optimization responsible for trades and constraints | This preserves interpretability and prevents an unrestricted learned trading policy |
| ML execution in V1 | Build and test the full pipeline on tiny synthetic fixtures, but do not fit repository history | The current task explicitly prohibits real-data training and performance claims |
| Research output | Use deterministic result fields and separately label wall-clock timing as nondeterministic telemetry | Runtime varies even when schedules and costs are reproducible |
| Command surface | Use one `simulate --strategy` entry point, grouped `experiment` and `ml` commands, and retain `simulate-twap` as a compatibility alias | A task-oriented hierarchy keeps research workflows discoverable without breaking the original command |
| Automation | Test Python 3.11 and 3.13 with Ruff, mypy, repository-contract validation, and pytest in GitHub Actions | The matrix covers the supported minimum and current development runtime |
| Dependency compatibility | Constrain NumPy below 2.4 while Python 3.11 remains supported | Newer NumPy type stubs require syntax beyond the repository's declared type-check target |
| QP scaling | Enable OSQP adaptive penalty updates while keeping fixed tolerances, iteration bounds, and acceptance checks | The full parameter grid exposed poor convergence with a fixed penalty on differently scaled risk and impact terms |
| Performance architecture | Index one immutable historical matrix per symbol or pooled scope; binary-search the strict date cutoff; reuse exact-horizon OSQP setups through `OptimalExecutionWorkspace` | Repeated date-prefix pivots duplicate work and memory; scope indexes preserve finite-window estimators and exact-horizon reuse preserves the original integer schedule |
| QP validation levels | Use structural convexity validation in adaptive MPC and full eigenvalue validation in static or standalone solves | Positive temporary curvature and positive-semidefinite added terms prove convexity without repeated cubic work in the hot path |
| ML dataset memory | Discover symbols with bounded Arrow batches and build one symbol partition at a time; CLI builds do not retain result rows | This prevents a future multi-symbol, multi-year universe from becoming one pandas DataFrame |
| Paper research boundary | Compare matched dense and sparse predictive representations only through causal volume forecasts and fixed deterministic MPC/TCA | This isolates representation geometry without allowing a learned model to choose trades or constraints |
| Paper v2 quality hierarchy | Use provider-native daily observations for formation, observed-only fixed 15-minute tokens throughout target validation, stock/SPY sequence and seasonal construction, and exact consumed-window minutes for TCA; persist per-token observation counts | Data validity must match the scientific task's resolution; absence of a provider minute aggregate is not proof of zero activity |
| Paper corpus | Freeze a formation-period 100-stock universe from 95% valid Alpaca SIP daily coverage and later acquire immutable SIP minute responses in resumable monthly chunks | Stable instruments, hashes, failure receipts, and no IEX fallback make the licensed corpus auditable without activity-based exact-minute universe selection |
| Paper sequence representation | Store one 26-by-18 tensor per regular session and derive indexed eight-token contexts with direct 1/2/4/8-token targets | Session storage avoids overlapping materialization and explicit indexes preserve point-in-time cutoffs |
| Paper representation hierarchy | Test frozen representation accessibility, then information retention, supervised forecast value, and finally fixed execution decision value | This ordering separates what the representation contains from whether a downstream learner or optimizer can use it |
| Paper feature boundary | Store 18 causal observations, encode 13 dynamic fields, and pass five current clock, ADV, and cumulative-volume fields only as predictor conditioning | Exogenous scale and clock state must not define the learned representation |
| Sparse representation | Use a shared encoder, exact-forward RepReLU, a primary rectified-Gaussian RDMReg target, and no EMA or stop-gradient target | Gaussian dense and sparse targets isolate non-negativity and support before the Laplace tail-shape appendix |
| Paper comparison fairness | Mask linked padding before every predictor and use identical primary dense/sparse initialization distributions | Encoder bias and sparse-only output bias otherwise create avoidable comparison confounds |
| Sparse target normalization | Derive rectified moments and RMS from each configured generalized-Gaussian target | The 0.5 RMS applies only to the 75%-zero Laplace setting |
| Paper RDM coefficient | Select one common value from 0.1, 1, and 10 on Fold 1 validation with seed 13 and freeze it for both geometries | Geometry-specific tuning would confound the matched comparison |
| Paper frozen probes | Reuse one selected encoder for horizon-specific affine, 64-unit MLP, and 256-unit MLP dynamics probes plus a fixed volume-surprise probe | Retraining encoders per capacity or using the JEPA predictor for retention would confound accessibility with representation learning |
| Paper distribution diagnostics | Compute the 2,048-projection FP32 RDM diagnostic over a deterministic bounded sample of actual valid encoder latents | Validation memory remains bounded without admitting padding or predictions into the distribution comparison |
| Paper controls and seeds | Use a frozen untrained nonlinear neural control and evaluate seed-specific metric effects; reserve forecast ensembling for an appendix | A linear projection is too weak a placebo, while primary ensembling hides training variability |
| Paper supervised targets | Predict remaining-volume residual over a causal baseline and train shape on weighted deterministic origins from four time bands | The baseline anchors scale and inverse-probability weighting bounds long-form expansion without changing the estimand |
| Paper decision clock | Solve once at each 15-minute boundary, commit the next minute-level segment, and compare with a matched realized-volume oracle | Forecast and optimization clocks must agree; minute replay remains responsible for realized capacity and cost attribution |
| Historical paper execution | Use manifest-derived folds, v2 token-valid XNYS sessions, exact 300-minute TCA windows, bounded sequence loaders, device-aware JEPA, long-form LightGBM shape rows, and 15-minute cached forecasts | The authorized historical study must run without per-session glue, fabricated bars, or target leakage |
| Paper inference | Intersect exact complete cases before date averaging and resample blocks within folds | Pairing and fold contribution must be preserved before uncertainty estimation |
| Historical forecast availability | Record unavailable exact EWMA requests and method-specific TCA cases; retain other methods and report paired coverage | Missing eligible history is not a fabricated zero forecast or a reason to remove an instrument globally; see ADR 0023 |
| Result timestamp storage | Merge timestamp fields at the finest input precision with identical timezones and safe casts; keep other types strict | Arrow storage-unit differences must neither abort compatible results nor truncate instants; see ADR 0024 |
| Evaluation derived state | Build one compact base per fold and publish learned predictions with checksum-bound atomic manifests; reuse completed predictions within the new evaluation identity | Reconstructing raw rows or rerunning identical frozen predictions adds work without information; incomplete artifacts must never be treated as completed stages |
| Paper protocol freeze | Require a source-hashed protocol freeze and a checksummed validation-only parameter freeze before locked-test stages | Missing or changed selections must fail closed rather than be reconstructed after test inspection |
| Paper resource planning | Derive step, long-shape-row, embedding-storage, and run-count bounds from sequence manifests before training | The full protocol must abort before exceeding configured safe limits and timing estimates must name their measured device and scope |
| Paper reporting hierarchy | Keep four main tables and four main figures for data, accessibility, forecasting, and execution; write support, regime, block, sparsity, and sensitivity output as appendix artifacts | Secondary characterization cannot replace a failed primary comparison |
| Paper dependencies | Keep PyTorch 2.13.0, safetensors 0.8.0, and LightGBM 4.7.0 in exact-version optional extras | Ordinary simulation remains lightweight while paper artifacts record the evaluated dependency identity |
| Paper execution authorization | Disable network acquisition, historical fitting, and full paper runs in both configuration and CLI by default | Synthetic acceptance must never silently trigger licensed downloads or empirical training |
| Paper artifact provenance | Use safetensors and native LightGBM models plus checksummed, compatibility-complete checkpoint, embedding, and run manifests; hash Git-tracked normative text after canonical LF normalization while preserving empirical and receipt artifacts byte-for-byte | Reuse must fail closed when any data, fold, cutoff, model, environment, or downstream identity changes without making protocol identity depend on checkout line-ending conversion |

Apply this order of precedence:

1. Preserve mathematical correctness and point-in-time integrity.
2. Follow repository-specific terms and contracts in this document.
3. Follow the language-native Python and Markdown conventions configured in `pyproject.toml`.
4. Apply the broadly useful Microsoft, Google, and GitHub technical-writing practices cited below.

Do not copy vendor-specific product, user-interface, branding, or internal publishing conventions when they do not apply to ExecSim.

For the rationale and consequences behind these directions, use the [architecture decision index](../ADRs/README.md). A material direction change is incomplete until its ADR, this table, specifications, and tests agree.

## Write for the reader

Write for a researcher, quantitative developer, reviewer, or learner who needs to understand and reproduce a result.

- Put the outcome, purpose, or constraint before implementation history.
- Use active voice and name the actor: “The simulator rejects duplicate bars,” not “Duplicate bars are rejected.”
- Address the reader as “you” only in task-oriented instructions. Use the component name in reference documentation.
- Prefer short, direct sentences and one main idea per paragraph.
- Use plain language without diluting mathematical precision.
- Define specialized terms and abbreviations on first use.
- Use one term for one concept. Do not alternate synonyms for variety.
- Avoid marketing language, filler, idioms, humor, anthropomorphism, and claims such as “powerful,” “smart,” “seamless,” or “production-ready” without measured evidence.
- State assumptions, units, provenance, limitations, and failure modes close to the claim they qualify.

## Use canonical ExecSim terminology

Use these terms consistently in public APIs, logs, reports, tests, and prose.

| Term | Meaning | Avoid |
|---|---|---|
| parent order | The single-asset order objective with side, quantity, date, and execution window | master order, trade request |
| bucket | One policy decision and execution interval | slice when the interval is meant |
| bar | The aggregated OHLCV market observation for a bucket | tick, quote |
| planned quantity | Shares requested by a static plan or adaptive decision | filled amount, target fill |
| executed quantity | Shares realized after inventory and hard-cap constraints | planned fill |
| remaining inventory | Parent-order shares not yet executed | position, residual order without definition |
| forecast volume | Point-in-time expected future market volume | predicted liquidity when only volume is forecast |
| actual market volume | Realized bar volume used by the fill constraint | available liquidity, because bar volume is only a proxy |
| planned participation | Planned quantity divided by forecast or actual volume, as explicitly named | participation without its denominator |
| realized participation | Executed quantity divided by actual market volume | fill ratio |
| reference price | Bar VWAP or the documented OHLC fallback before modeled costs | midprice, market price |
| execution price | Reference price plus side-aware modeled spread and impact | fill price when no actual venue fill exists |
| half-spread | Assumed, measured, estimated, or supplied currency cost per share for crossing one side of the spread | bid-ask spread when only half is used |
| temporary impact | The execution-price displacement attributed to current-bucket participation | slippage as a catch-all |
| implementation shortfall | Side-aware execution cost relative to arrival, in currency or basis points | profit and loss |
| point-in-time | Computed only from information available at the declared cutoff | real-time, leakage-free without proof |
| deployable policy | An ex-ante policy that obeys the information contract; not a claim of production readiness | live strategy |
| oracle policy | An evaluation-only hindsight baseline with future information | optimal policy |
| assumed parameter | A value selected for research sensitivity rather than estimated from data | calibrated parameter |

Use `buy` and `sell` for sides, positive cost for worse execution on either side, shares for quantity, currency per share for price inputs, currency for aggregate costs, basis points for normalized price differences, and fractions in `[0, 1]` for participation.

## Structure documents for scanning

- Use one level-1 heading that names the document.
- Use sentence case for all headings.
- Keep heading levels hierarchical. Do not skip from `##` to `####`.
- Do not end headings with a period or colon.
- Use descriptive headings that remain meaningful in GitHub’s generated outline.
- Put purpose and reader outcome first, then prerequisites, procedure or contract, interpretation, limitations, and references.
- Use numbered lists only for sequences or ranked priorities. Use bullets for unordered sets.
- Introduce every list and table with a complete sentence.
- Keep list items grammatically parallel.
- Use tables when readers compare three or more consistent fields. Do not use a table for prose that reads better as a short list.
- Use restrained emphasis. Do not use bold text as a substitute for headings.
- Use descriptive link text, not “click here,” “this,” or a bare URL.
- Prefer relative repository links in tracked Markdown so links work in clones.

## Write procedures that run as shown

State prerequisites before a procedure. Use numbered steps and begin each step with an imperative verb. Put one action or decision in each step and include the final verification step.

Introduce each command with its purpose:

```text
Validate the complete test suite:
```

Then provide a copyable command without a shell prompt:

```powershell
.\.venv\Scripts\python.exe -m pytest
```

- Tag every fenced block with its language or content type.
- Use obvious uppercase placeholders such as `DATASET_PATH` or `RUN_ID` and explain them after the block.
- Do not mix optional-argument notation into a command intended for direct copying.
- Show expected output when it proves success, but keep output separate from the command.
- Never include credentials, tokens, local secrets, or a real `.env` value.
- Test every tracked command from the documented working directory.
- Mark unexecuted commands `NOT RUN`; do not imply that syntactic plausibility is execution evidence.

## Specify every production module

Register every production module in `repo_manifest.yaml`. Specify its behavior in `docs/SPECIFICATIONS.md` or a focused document linked from that file.

Each component specification must cover the applicable fields below:

1. Purpose and non-goals.
2. Public interface, inputs, outputs, and types.
3. Units, sign conventions, parameter provenance, and defaults.
4. Preconditions and validation errors.
5. State transitions and side effects.
6. Point-in-time information set and prohibited future data.
7. Mathematical formula or algorithm, with a primary reference when the choice is not obvious.
8. Determinism, ordering, seed, hashing, and numerical-tolerance rules.
9. Capacity, performance, and complexity expectations where material.
10. Artifact or log schema and compatibility behavior.
11. Known limitations and deliberately unsupported behavior.
12. Tests that prove ordinary behavior, boundary cases, and invariants.

Do not describe planned behavior as implemented. Use these exact lifecycle labels where needed:

- `IMPLEMENTED`: code and risk-proportionate tests exist.
- `PLANNED`: accepted scope without implementation evidence.
- `EVALUATION_ONLY`: deliberately non-deployable research behavior.
- `NOT RUN`: the verification command did not execute.
- `BLOCKED`: a named condition prevents execution.

## Write Python that exposes the model

- Use Python 3.11+ syntax and the `src/` package layout.
- Use `snake_case` for modules, functions, methods, and variables; `CapWords` for classes; and uppercase names for module constants.
- Use descriptive public names. Mathematical symbols such as `q`, `x`, `L`, and `D` are acceptable only in tightly scoped derivations that map directly to documented notation.
- Add type annotations to public functions, protocols, dataclasses, and returned structures.
- Use a protocol only for a real substitution boundary.
- Prefer frozen, slotted dataclasses for validated immutable values. Use mutable state only when the lifecycle requires it, such as an MPC warm start.
- Validate at boundaries and raise specific errors with the invalid concept and expected condition.
- Keep core simulation independent of reporting and ML imports.
- Preserve exact integer quantities and deterministic ordering. Never rely on an unordered container for a research artifact.
- Make numerical tolerances configurable or named. Do not hide a tolerance in a comparison.
- Keep functions focused on one level of abstraction. Extract formulas when an isolated invariant can be tested.
- Use vectorized NumPy for measured hot paths, not as a reason to obscure a short, auditable algorithm.
- Use comments to explain why a choice exists, which information is available, or which paper defines a formula. Do not narrate visible syntax.

Public modules, classes, and non-obvious functions require docstrings. Begin with a one-line purpose. Add concise sections for parameters, returns, raises, units, point-in-time behavior, or references only when they add information not carried by names and types.

## Document mathematics and research claims

- Define notation before the first equation and map each symbol to code names.
- State units beside parameters and distinguish per-share, per-bucket, per-session, currency, and dimensionless values.
- State the optimization objective, constraints, feasibility handling, and numerical acceptance criteria.
- Distinguish a model identity from an empirical claim and a configured assumption.
- Cite primary papers, standards, official documentation, or first-party sources for material non-obvious choices.
- Use “matches within tolerance,” not “proves,” for numerical reference comparisons.
- Label synthetic, fixture, historical-sample, out-of-sample, and production evidence separately.
- Report the sample size and paired unit before statistical conclusions.
- Do not call an assumed parameter calibrated or estimated.
- Do not call an oracle schedule deployable or universally optimal.
- Do not claim ML quality when only a synthetic pipeline fit ran.

## Write tests as executable specifications

Name tests after observable behavior or an invariant, for example:

```python
def test_integer_projection_preserves_quantity_and_capacities() -> None: ...
```

- Keep tests deterministic and independent of network services by default.
- Use a focused mathematical test for each formula and monotonicity rule.
- Add explicit leakage tests that introduce forbidden future data and expect rejection.
- Test both buy and sell sign behavior.
- Test invalid, zero, boundary, partial-fill, shortened-session, and solver-failure paths.
- Use approximate comparisons with a named or justified tolerance.
- Test public behavior instead of private implementation shape unless the internal matrix is itself part of the mathematical contract.
- Do not delete, skip, weaken, or narrow an existing test to make a gate pass. If the specification changes, explain why the prior expectation was wrong and replace it with a stronger contract test.

## Make content accessible and portable

- Use semantic headings, lists, and tables rather than visual spacing.
- Give informative images and diagrams meaningful alt text and explain essential information in prose.
- Do not communicate status by color alone.
- Avoid directional references such as “above” when a heading or link can identify the content.
- Use literal, inclusive language and avoid culturally specific idioms or ableist, violent, gendered, or patronizing metaphors.
- Keep Markdown useful in a local clone even when GitHub-specific rendering is unavailable.

## Review with one checklist

Before a code or documentation checkpoint, verify all applicable statements:

- The terminology matches the canonical table.
- Public behavior has a current specification and tests.
- The point-in-time information boundary is explicit.
- Units, signs, provenance, defaults, and tolerances are visible.
- Commands are copyable and were executed or labeled accurately.
- Headings use sentence case and follow a logical hierarchy.
- Lists are introduced and parallel; tables have a comparison purpose.
- Links are descriptive and repository links are relative where practical.
- Claims match the evidence level and name limitations.
- Ruff, mypy, pytest, and repository-context checks report their actual status.

## Sources

This standard adopts broadly applicable guidance from these official sources:

- [Microsoft Writing Style Guide: top 10 tips](https://learn.microsoft.com/en-us/style-guide/top-10-tips-style-voice)
- [Microsoft guidance for code examples](https://learn.microsoft.com/en-us/style-guide/developer-content/code-examples)
- [Microsoft guidance for accessible writing](https://learn.microsoft.com/en-us/style-guide/accessibility/writing-all-abilities)
- [Google developer documentation style guide](https://developers.google.com/style)
- [Google guidance for headings and titles](https://developers.google.com/style/headings)
- [Google guidance for code samples](https://developers.google.com/style/code-samples)
- [Google Python style guide](https://google.github.io/styleguide/pyguide.html)
- [GitHub Docs content design principles](https://docs.github.com/en/contributing/writing-for-github-docs/content-design-principles)
- [GitHub Docs style guide](https://docs.github.com/en/contributing/style-guide-and-content-model/style-guide)
- [GitHub guidance for repository README files](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/customizing-your-repository/about-readmes)

### 2026-09-11 — pre-evaluation identity hardening (ADR 0025)

The locked evaluator now treats causal ADV20 as required derived evidence for
every exact-window eligible instrument/session.  Validation occurs before TCA
workers and is repeated at the direct replay boundary; missing, duplicate,
non-finite, or non-positive values fail closed rather than changing the sample.
Forecast and TCA stages resolve relocated runtime data through the canonical
`PaperRunConfig.data_path` abstraction.  Before consuming TEST inputs they bind
the runtime universe bytes to one shared `universe_manifest_hash` recorded by
every fold sequence manifest.  Reseal now validates the complete configured
primary JEPA final inventory and requires one non-empty training `code_commit`
across all coordinates, before publishing a new execution namespace.  These
checks are identity/provenance safeguards only and do not alter scientific
configuration, estimands, or model artifacts.  See ADR 0025.

### 2026-09-12 — cross-link resealed evaluation artifacts (ADR 0026)

Reseal must verify the existing identity links between each fold sequence
manifest, JEPA checkpoint, compatibility record, embedding export, and frozen
LightGBM coordinate.  Independent file checksums do not prove that a complete
artifact set was trained from the same sequence, universe, representation
source, and TRAIN/VALIDATION embedding bytes.  Cross-link checks run before
`execution.json` publication and preserve all scientific inputs and estimands.
See ADR 0026.

### 2026-09-12 — execution-share ADV20 and chained reseal provenance (ADRs 0027 and 0028)

TCA history builders now restate each prior daily volume into the target
session's raw execution-share basis using the existing point-in-time split
factor and the 10:30 quantity-decision information clock. Replay bars remain
raw, the 20-session lag is strict, and the corporate-action manifest identity
is part of the history artifact identity. New reseals preserve v2 evidence but
use a typed supersession receipt and v3 execution identity to distinguish the
root TEST authorization from the immediate prior evaluator. See ADRs 0027 and
0028.

### 2026-09-12 — bounded reuse of immutable evaluation inputs (ADR 0029)

Frozen probe tensors may be encoded once and replayed with their original batch
boundaries, dtypes, masks, and RNG transitions. Temporary caches are checksummed,
source-bound, atomically published, and retired after coordinate publication.
Use cgroup-aware native thread limits for representation evaluation. Batch raw
forecast feature construction by consecutive session and reuse verified index
and learned date slices without sharing mutable forecast-provider state.
Retain an optimization only after equivalence and real-input performance checks.
See ADR 0029 and the evaluation implementation contract. Scientific configuration,
trained artifacts, estimands, and cross-source result isolation remain unchanged.
