# Secondary observable-capacity extension

## Status and scope

This pre-lock amendment adds an exploratory mechanism analysis to `sparse-jepa-v2`. It does not change the five confirmatory contrasts, representation training, the selected RDM coefficient, LightGBM selection, or TCA.

The extension must be frozen before any locked-test effectiveness result is inspected.

## Target

For horizon (h \in \{1,2,4,8\}), the observable target is the future-volume surprise

\[
y_h = \log(1 + V_h) - \log(1 + \widehat{V}^{\mathrm{causal}}_h),
\]

where (V_h) is the observed future bucket volume and \(\widehat{V}^{\mathrm{causal}}_h\) is the historical seasonal baseline available at the sample's point-in-time cutoff.

## Inputs and capacities

Every probe receives only the frozen eight-token linked latent context and its eight-value context mask. Actual future latents and future market observations are targets only and never probe inputs.

The capacity ladder is:

- affine ridge with `alpha` selected from `{0.1, 1.0, 10.0}` by mean validation MAE across the four horizons;
- a fixed one-hidden-layer 64-unit GELU MLP;
- a fixed one-hidden-layer 256-unit GELU MLP.

The MLPs use the same fixed 20-epoch budget and learning rate as the existing latent-capacity ladder. TRAIN fits parameters, VALIDATION performs only the declared ridge selection, and TEST is scored once after the locked-test firewall opens.

## Outputs

For every fold, geometry, seed, horizon, and capacity, persist:

- MAE and RMSE;
- parameter count and approximate multiply-accumulate count;
- measured inference time;
- evaluated row count.

Label these outputs `SECONDARY / EXPLORATORY`. They do not enter Holm correction for the five confirmatory tests.

## Block-aware confirmatory p-values

The primary interval and null test use the same fold-stratified moving-block resampling plan. Blocks contain contiguous dates and never cross fold boundaries. Each replicate preserves the original number of dates contributed by each fold.

For a two-sided null p-value, subtract the observed overall paired-date mean from every date-level difference, resample these centered values with the same fold-safe blocks, and compare the absolute null replicate mean with the absolute observed mean. Apply the finite-replicate correction

\[
p = \frac{1 + \#\{|\bar d_b^0| \ge |\bar d|\}}{B + 1}.
\]

This null procedure is frozen before TEST opens. Holm adjustment applies only to the five predeclared confirmatory p-values.
