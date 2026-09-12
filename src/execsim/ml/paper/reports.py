"""Strictly separated synthetic and historical paper-bundle generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

TABLE_NAMES = (
    "dataset_folds_exclusions",
    "representation_accessibility",
    "forecasting",
    "execution",
)
FIGURE_NAMES = (
    "capacity_vs_normalized_latent_error",
    "forecast_performance_by_model",
    "forecast_error_vs_asof",
    "allocation_regret_with_paired_intervals",
)
HISTORICAL_TABLE_NAMES = (
    "dataset_folds_exclusions",
    "jepa_representation_diagnostics",
    "representation_accessibility",
    "observable_financial_accessibility",
    "forecast_performance",
    "forecast_by_asof",
    "lightgbm_selected_parameters",
    "tca_execution",
    "confirmatory_statistics",
    "support_regime_diagnostics",
    "appendix_sensitivities",
)
HISTORICAL_FIGURE_NAMES = (
    "capacity_vs_normalized_latent_error",
    "observable_capacity_vs_error",
    "forecast_performance_by_model",
    "forecast_error_vs_asof",
    "allocation_regret_with_paired_intervals",
)
HISTORICAL_EMPTY_TABLE_NAMES = frozenset({"forecast_performance", "forecast_by_asof"})


def write_paper_bundle(
    output_root: Path,
    *,
    paper_run_id: str,
    tables: dict[str, pd.DataFrame],
    provenance: dict[str, object],
) -> Path:
    """Write reviewable JSON, Parquet, Markdown, and LaTeX outputs from supplied results."""
    missing = set(TABLE_NAMES).difference(tables)
    if missing:
        raise ValueError(f"Paper bundle missing tables: {sorted(missing)}")
    destination = output_root / paper_run_id
    destination.mkdir(parents=True, exist_ok=False)
    table_dir = destination / "tables"
    figure_dir = destination / "figures"
    table_dir.mkdir()
    figure_dir.mkdir()
    for name in TABLE_NAMES:
        frame = tables[name]
        frame.to_parquet(table_dir / f"{name}.parquet", index=False)
        (table_dir / f"{name}.json").write_text(
            frame.to_json(orient="records", indent=2) + "\n", encoding="utf-8"
        )
        (table_dir / f"{name}.md").write_text(
            frame.to_markdown(index=False) + "\n", encoding="utf-8"
        )
        (table_dir / f"{name}.tex").write_text(frame.to_latex(index=False), encoding="utf-8")
    _render_bundle_figures(figure_dir, tables)
    from execsim.ml.paper.provenance import build_run_provenance

    complete_provenance = build_run_provenance(paper_run_id=paper_run_id, supplied=provenance)
    (destination / "provenance.json").write_text(
        json.dumps(complete_provenance, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (destination / "PAPER_OUTLINE.md").write_text(
        "# Sparse-JEPA paper outline\n\n"
        "Results in this bundle are synthetic plumbing evidence unless provenance "
        "states otherwise.\n\n"
        "## Abstract\n\nState the controlled question, data, matched methods, primary endpoints, "
        "and evidence boundary.\n\n"
        "## 1. Introduction\n\nMotivate predictable intraday activity and the sparse-geometry "
        "hypothesis without a trading-performance claim.\n\n"
        "## 2. Closest work\n\nCompare JEPA, sparse predictive representations, financial "
        "representation learning, volume forecasting, and optimal execution.\n\n"
        "## 3. Data and point-in-time protocol\n\nDescribe universe formation, SIP acquisition, "
        "exclusions, features, folds, and leakage controls.\n\n"
        "## 4. Method\n\nDefine the shared encoder, frozen predictor ladder, dense Gaussian "
        "and rectified-Gaussian sparse RDMReg objectives, embeddings, placebo, and LightGBM.\n\n"
        "## 5. Evaluation\n\nPredeclare forecast endpoints, diagnostics, regimes, block "
        "inference, and fixed MPC/TCA assumptions.\n\n"
        "## 6. Results\n\nReport covariance-normalized accessibility, fixed-observable "
        "retention, matched forecast comparisons, and oracle-relative modeled allocation "
        "outcomes with seed evidence.\n\n"
        "## 7. Limitations and conclusion\n\nSeparate modeled execution from actual impact and "
        "state positive, negative, or inconclusive findings exactly as supported.\n",
        encoding="utf-8",
    )
    (destination / "APPENDIX.md").write_text(
        "# Appendix\n\n"
        "Synthetic fixture output; historical robustness analyses are `NOT RUN`.\n\n"
        "## Full hyperparameters\n\nRead the checksummed configuration and provenance records in "
        "this bundle.\n\n"
        "## Data-quality exclusions\n\nReport corpus receipt failures, missing buckets, early "
        "closes, symbol mapping, and corporate-action exclusions.\n\n"
        "## Seed variability and geometry controls\n\nReport seeds 13, 29, and 47, the "
        "rectified-Laplace control, and the Fold 1 "
        "zero-fraction targets 0.50, 0.75, and 0.875.\n\n"
        "## Cost sensitivity\n\nReport the locked 1% and 5% ADV sensitivity on ten stocks and "
        "both sides without describing modeled displacement as actual impact.\n\n"
        "## Selected LightGBM parameters\n\nReport each fold's validation-selected grid point, "
        "iteration, category metadata, thread count, and native-model hash.\n\n"
        "## Support, regimes, and statistical diagnostics\n\nReport target matching, "
        "effective rank, active and dead dimensions, identity ratios, exact block-bootstrap "
        "configuration, and all "
        "predeclared failure criteria.\n\n"
        "## Limitations\n\nRecord entitlement, survivorship, minute-bar, counterfactual-impact, "
        "compute, multiplicity, and novelty-search limitations.\n",
        encoding="utf-8",
    )
    (destination / "REPORT.md").write_text(
        "# Sparse-JEPA research report\n\n"
        "This bundle records software-pipeline evidence. Read `provenance.json` before "
        "interpreting results.\n\n"
        "## Evidence status\n\nDetermine whether the bundle is synthetic, historical, or a full "
        "locked evaluation from provenance. Synthetic values test plumbing only.\n\n"
        "## Primary results\n\nUse the accessibility, forecast, and execution tables in "
        "hierarchical order. Apply the predeclared paired date-block intervals and Holm "
        "adjustment before any formal superiority claim.\n\n"
        "## Representation diagnostics\n\nUse the representation/regime table and "
        "collapse gates.\n\n"
        "## Limitations\n\nDo not claim live profitability, actual market impact, or sparse-JEPA "
        "effectiveness from fixtures.\n",
        encoding="utf-8",
    )
    return destination


def write_synthetic_paper_bundle(
    output_root: Path,
    *,
    paper_run_id: str,
    tables: dict[str, pd.DataFrame],
    provenance: dict[str, object],
) -> Path:
    """Write the visibly synthetic plumbing bundle retained for fast CI."""
    return write_paper_bundle(
        output_root, paper_run_id=paper_run_id, tables=tables, provenance=provenance
    )


HISTORICAL_TABLE_SCHEMAS = {
    "dataset_folds_exclusions": {"fold_id", "partition", "included", "excluded"},
    "representation_accessibility": {
        "geometry",
        "seed",
        "horizon",
        "probe_capacity",
        "parameter_count",
        "approximate_macs",
        "inference_seconds",
        "normalized_latent_error",
        "zero_baseline",
        "train_mean_baseline",
        "persistence_baseline",
    },
    "jepa_representation_diagnostics": {
        "fold_id",
        "geometry",
        "seed",
        "zero_fraction",
        "mean_active_dimensions",
    },
    "observable_financial_accessibility": {
        "geometry",
        "seed",
        "horizon",
        "probe_capacity",
        "observable_volume_probe_mae",
        "observable_volume_probe_rmse",
        "observable_parameter_count",
        "observable_approximate_macs",
        "observable_inference_seconds",
        "observable_test_rows",
    },
    "forecast_performance": {
        "method",
        "seed",
        "log_remaining_volume_mae",
        "conditional_curve_error",
        "matched_cases",
    },
    "forecast_by_asof": {
        "method",
        "seed",
        "as_of_token",
        "log_remaining_volume_mae",
        "conditional_curve_error",
        "matched_cases",
    },
    "lightgbm_selected_parameters": {
        "fold_id",
        "method",
        "seed",
        "scale_num_leaves",
        "scale_min_child_samples",
        "scale_reg_lambda",
        "scale_best_iteration",
        "shape_num_leaves",
        "shape_min_child_samples",
        "shape_reg_lambda",
        "shape_best_iteration",
    },
    "tca_execution": {
        "method",
        "comparison_baseline",
        "seed",
        "normalized_allocation_regret",
        "absolute_modeled_impact_cost",
        "completion_rate",
        "implementation_shortfall_bps",
        "mean_difference",
        "ci_lower",
        "ci_upper",
    },
    "confirmatory_statistics": {
        "contrast_id",
        "stage",
        "candidate",
        "baseline",
        "endpoint",
        "mean_difference",
        "median_difference",
        "ci_lower",
        "ci_upper",
        "paired_dates",
        "date_win_rate",
        "standardized_effect",
        "raw_p_value",
        "holm_adjusted_p_value",
    },
    "support_regime_diagnostics": {
        "fold_id",
        "geometry",
        "seed",
        "zero_fraction",
        "mean_active_dimensions",
    },
    "appendix_sensitivities": {"analysis"},
}


def write_historical_paper_bundle(
    output_root: Path,
    *,
    paper_run_id: str,
    tables: dict[str, pd.DataFrame],
    provenance: dict[str, object],
    historical_schema_fixture: bool = False,
) -> Path:
    """Write historical named schemas and figures without synthetic substitutions."""
    expected_classification = "synthetic_fixture" if historical_schema_fixture else "historical"
    if provenance.get("data_classification") != expected_classification:
        raise ValueError("Historical bundle requires historical data classification.")
    for name, required in HISTORICAL_TABLE_SCHEMAS.items():
        frame = tables.get(name)
        missing = required if frame is None else required.difference(frame.columns)
        if frame is None or missing or (frame.empty and name not in HISTORICAL_EMPTY_TABLE_NAMES):
            raise ValueError(f"Historical table {name} is empty or missing {sorted(missing)}")
    destination = output_root / paper_run_id
    destination.mkdir(parents=True, exist_ok=False)
    table_dir = destination / "tables"
    figure_dir = destination / "figures"
    table_dir.mkdir()
    figure_dir.mkdir()
    for name in HISTORICAL_TABLE_NAMES:
        frame = tables[name]
        frame.to_parquet(table_dir / f"{name}.parquet", index=False)
        (table_dir / f"{name}.json").write_text(
            frame.to_json(orient="records", indent=2) + "\n", encoding="utf-8"
        )
        (table_dir / f"{name}.md").write_text(
            frame.to_markdown(index=False) + "\n", encoding="utf-8"
        )
        (table_dir / f"{name}.tex").write_text(frame.to_latex(index=False), encoding="utf-8")
    _render_historical_figures(
        figure_dir, tables, fixture_label="synthetic fixture" if historical_schema_fixture else None
    )
    from execsim.ml.paper.provenance import build_run_provenance

    complete = build_run_provenance(paper_run_id=paper_run_id, supplied=provenance)
    (destination / "provenance.json").write_text(
        json.dumps(complete, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )
    heading = (
        "# Sparse-JEPA historical-schema synthetic fixture"
        if historical_schema_fixture
        else "# Sparse-JEPA historical experiment report"
    )
    boundary = (
        "This bundle exercises historical schemas with synthetic inputs and contains no "
        "empirical result."
        if historical_schema_fixture
        else "This report contains matched historical experiment outputs."
    )
    (destination / "REPORT.md").write_text(
        f"{heading}\n\n{boundary} It describes modeled execution costs, not live profitability "
        "or realized market impact.\n",
        encoding="utf-8",
    )
    return destination


def _render_historical_figures(
    figure_dir: Path, tables: dict[str, pd.DataFrame], *, fixture_label: str | None = None
) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the reporting or paper extra for figures.") from exc

    capacity = tables["representation_accessibility"]
    capacity = capacity.groupby(
        ["geometry", "horizon", "probe_capacity", "parameter_count"],
        sort=True,
        as_index=False,
    ).agg(normalized_latent_error=("normalized_latent_error", "mean"))
    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    for (geometry, horizon), group in capacity.groupby(["geometry", "horizon"], sort=True):
        group = group.sort_values("parameter_count", kind="stable")
        axis.plot(
            group["parameter_count"],
            group["normalized_latent_error"],
            marker="o",
            label=f"{geometry} / h{horizon}",
        )
    axis.set_xscale("log")
    axis.legend()
    axis.set(xlabel="probe parameter count", ylabel="TRAIN-covariance-normalized latent error")
    _save_historical(figure, figure_dir / "capacity_vs_normalized_latent_error.png", fixture_label)

    observable = tables["observable_financial_accessibility"]
    observable = observable.groupby(
        ["geometry", "horizon", "probe_capacity", "observable_parameter_count"],
        sort=True,
        as_index=False,
    ).agg(observable_volume_probe_mae=("observable_volume_probe_mae", "mean"))
    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    for (geometry, horizon), group in observable.groupby(["geometry", "horizon"], sort=True):
        group = group.sort_values("observable_parameter_count", kind="stable")
        axis.plot(
            group["observable_parameter_count"],
            group["observable_volume_probe_mae"],
            marker="o",
            label=f"{geometry} / h{horizon}",
        )
    axis.set_xscale("log")
    axis.legend()
    axis.set(xlabel="probe parameter count", ylabel="future-volume-surprise MAE")
    _save_historical(figure, figure_dir / "observable_capacity_vs_error.png", fixture_label)

    combined = tables["forecast_by_asof"]
    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    if combined.empty:
        _annotate_no_common_forecast_cases(axis)
    else:
        summary = combined.groupby("method", sort=True, as_index=False).agg(
            log_remaining_volume_mae=("log_remaining_volume_mae", "mean"),
            conditional_curve_error=("conditional_curve_error", "mean"),
        )
        positions = np.arange(len(summary))
        width = 0.38
        axis.bar(
            positions - width / 2,
            summary["log_remaining_volume_mae"],
            width,
            label="remaining-volume log MAE",
        )
        axis.bar(
            positions + width / 2,
            summary["conditional_curve_error"],
            width,
            label="conditional-curve error",
        )
        axis.set_xticks(positions, summary["method"], rotation=30)
        axis.legend()
    _save_historical(figure, figure_dir / "forecast_performance_by_model.png", fixture_label)

    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    if combined.empty:
        _annotate_no_common_forecast_cases(axis)
    else:
        for name, group in combined.groupby("method", sort=True):
            axis.plot(
                group["as_of_token"], group["log_remaining_volume_mae"], marker="o", label=name
            )
        axis.legend()
    axis.set(xlabel="as-of token", ylabel="forecast error")
    _save_historical(figure, figure_dir / "forecast_error_vs_asof.png", fixture_label)

    paired = tables["tca_execution"].drop_duplicates(["method", "seed"])
    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    axis.vlines(paired["method"], paired["ci_lower"], paired["ci_upper"], color="#214761")
    axis.scatter(paired["method"], paired["mean_difference"], color="#214761")
    axis.axhline(0, color="#777777", linewidth=0.8)
    axis.tick_params(axis="x", rotation=30)
    axis.set_ylabel("paired normalized allocation regret difference")
    _save_historical(
        figure, figure_dir / "allocation_regret_with_paired_intervals.png", fixture_label
    )


def _annotate_no_common_forecast_cases(axis: Any) -> None:
    axis.text(
        0.5,
        0.5,
        "No common forecast cases",
        ha="center",
        va="center",
        transform=axis.transAxes,
    )


def _save_historical(figure: Any, path: Path, fixture_label: str | None) -> None:
    import matplotlib.pyplot as plt

    if fixture_label:
        figure.suptitle(fixture_label)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _render_bundle_figures(figure_dir: Path, tables: dict[str, pd.DataFrame]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the 'reporting' or 'paper' extra for report figures.") from exc
    fixture_label = "synthetic fixture"

    capacity = tables["representation_accessibility"]
    capacity_values = _numeric_values(capacity)
    x_values = _column_or_default(capacity, "parameters", len(capacity_values))
    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    axis.plot(x_values, capacity_values, marker="o", color="#214761")
    axis.set_xlabel("predictor parameters or fixture index")
    axis.set_ylabel("latent error or fixture value")
    _save_figure(
        figure, axis, figure_dir / "capacity_vs_normalized_latent_error.png", fixture_label
    )

    combined = tables["forecasting"]
    combined_values = _numeric_values(combined)
    asof = _column_or_default(combined, "as_of_bucket", len(combined_values))
    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    axis.plot(asof, combined_values, marker="o", color="#214761")
    axis.set_xlabel("as-of bucket or fixture index")
    axis.set_ylabel("forecast error or fixture value")
    _save_figure(figure, axis, figure_dir / "forecast_error_vs_asof.png", fixture_label)

    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    axis.bar(range(len(combined_values)), combined_values, color="#214761")
    axis.set_xlabel("forecast method or fixture index")
    axis.set_ylabel("forecast error or fixture value")
    _save_figure(figure, axis, figure_dir / "forecast_performance_by_model.png", fixture_label)

    figure, axis = plt.subplots(figsize=(6.4, 3.6))
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.errorbar(
        range(len(combined_values)),
        combined_values,
        yerr=np.zeros(len(combined_values)),
        fmt="o",
        color="#214761",
    )
    axis.set_xlabel("paired method comparison")
    axis.set_ylabel("MPC cost difference")
    _save_figure(
        figure,
        axis,
        figure_dir / "allocation_regret_with_paired_intervals.png",
        fixture_label,
    )


def _numeric_values(frame: pd.DataFrame) -> np.ndarray:
    numeric = frame.select_dtypes(include="number")
    if numeric.empty:
        return np.asarray([0.0])
    values = numeric.iloc[:, -1].to_numpy(dtype=float)
    return values if len(values) else np.asarray([0.0])


def _column_or_default(frame: pd.DataFrame, column: str, length: int) -> np.ndarray:
    if column in frame and pd.api.types.is_numeric_dtype(frame[column]):
        return frame[column].to_numpy(dtype=float)
    return np.arange(length, dtype=float)


def _save_figure(figure: Any, axis: Any, path: Path, fixture_label: str) -> None:
    import matplotlib.pyplot as plt

    axis.set_title(f"{path.stem.replace('_', ' ')} — {fixture_label}")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    if fixture_label:
        figure.suptitle(fixture_label)
