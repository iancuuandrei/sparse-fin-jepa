# ADR 0019: Freeze observable capacity and a block-aware null test before locked evaluation

- Status: Accepted
- Date: 2026-09-08
- Owners: ExecSim maintainers

## Context

The frozen paper design already fixes future-volume surprise as an observable target and fixes an affine/MLP-64/MLP-256 capacity ladder for latent accessibility. Before any locked-test effectiveness inspection, the final execution brief requires the same capacity ladder for the observable target. The downstream harness also had confidence intervals and Holm adjustment but no block-aware raw p-value generator for the five confirmatory contrasts.

## Decision

Fit separate observable probes at the three frozen capacities from the same frozen linked-context and context-mask input used by the latent probes. Select ridge alpha on VALIDATION by mean observable MAE across the four frozen horizons. Keep the MLP architectures and 20-epoch budget fixed. Report the observable ladder as secondary and exploratory.

For confirmatory raw p-values, use a two-sided null distribution from globally centered paired date differences resampled with the same fold-stratified moving blocks as the confidence interval. Preserve each fold's date contribution and use the finite-replicate plus-one correction. Apply Holm adjustment only to the five frozen confirmatory tests.

## Rationale

Using the same decoder capacities distinguishes low-capacity accessibility from retained observable information without changing the trained representations or primary endpoints. Reusing the fold-safe block structure for the null test respects temporal dependence and supplies the missing predeclared p-value path before TEST is opened.

## Consequences

Observable evaluation requires separate MLP fits and additional runtime. Result rows record capacity-specific error, parameter, operation, timing, and count fields. The extension cannot select RDM lambda, JEPA checkpoints, LightGBM parameters, subgroups, or confirmatory endpoints.

The original representation artifacts and design-freeze receipt remain immutable. A separate pre-lock amendment receipt binds this downstream implementation commit and records that locked-test effectiveness was not inspected when the amendment was frozen.

## Alternatives considered

- Keeping only the affine observable probe was rejected because it cannot distinguish information retention from decoder accessibility at higher capacities.
- Adding the observable ladder to the confirmatory family was rejected because it is a pre-lock secondary mechanism analysis, not one of the five frozen contrasts.
- Using IID row-level or date-level p-values was rejected because it ignores the predeclared temporal dependence structure.
- Selecting MLP architecture or epochs on TEST was rejected because it would breach the locked-test firewall.

## Verification

- `tests/test_paper_sequences.py::test_multisession_multifold_corpus_builder_includes_spy_and_records_corruption`
- `tests/test_paper_pipeline.py::test_block_bootstrap_null_pvalue_uses_fold_safe_centered_blocks`
- Ruff and mypy checks for the changed downstream modules
