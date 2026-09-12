"""Focused report-contract checks using synthetic historical-schema fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from execsim.ml.paper.reports import (
    HISTORICAL_FIGURE_NAMES,
    HISTORICAL_TABLE_NAMES,
    _render_historical_figures,
    _save_historical,
    write_historical_paper_bundle,
)


def _synthetic_historical_tables() -> dict[str, pd.DataFrame]:
    """Return one-row synthetic inputs that exercise the named report schemas."""
    return {
        "dataset_folds_exclusions": pd.DataFrame(
            {"fold_id": ["fixture-fold"], "partition": ["test"], "included": [4], "excluded": [1]}
        ),
        "jepa_representation_diagnostics": pd.DataFrame(
            {
                "fold_id": ["fixture-fold"],
                "geometry": ["sparse"],
                "seed": [13],
                "zero_fraction": [0.75],
                "mean_active_dimensions": [8.0],
            }
        ),
        "representation_accessibility": pd.DataFrame(
            {
                "geometry": ["sparse"],
                "seed": [13],
                "horizon": [1],
                "probe_capacity": ["affine_ridge"],
                "parameter_count": [32],
                "approximate_macs": [64],
                "inference_seconds": [0.01],
                "normalized_latent_error": [0.25],
                "zero_baseline": [1.0],
                "train_mean_baseline": [0.9],
                "persistence_baseline": [0.8],
            }
        ),
        "observable_financial_accessibility": pd.DataFrame(
            {
                "geometry": ["sparse"],
                "seed": [13],
                "horizon": [1],
                "probe_capacity": ["affine_ridge"],
                "observable_volume_probe_mae": [0.15],
                "observable_volume_probe_rmse": [0.2],
                "observable_parameter_count": [36],
                "observable_approximate_macs": [72],
                "observable_inference_seconds": [0.01],
                "observable_test_rows": [4],
            }
        ),
        "forecast_performance": pd.DataFrame(
            {
                "method": ["raw"],
                "seed": [None],
                "log_remaining_volume_mae": [0.2],
                "conditional_curve_error": [0.1],
                "matched_cases": [4],
            }
        ),
        "forecast_by_asof": pd.DataFrame(
            {
                "method": ["raw"],
                "seed": [None],
                "as_of_token": [4],
                "log_remaining_volume_mae": [0.2],
                "conditional_curve_error": [0.1],
                "matched_cases": [4],
                "causal_baseline_remaining_volume": [100.0],
            }
        ),
        "lightgbm_selected_parameters": pd.DataFrame(
            {
                "fold_id": ["fixture-fold"],
                "method": ["lightgbm_raw"],
                "seed": [None],
                "scale_num_leaves": [15],
                "scale_min_child_samples": [50],
                "scale_reg_lambda": [1.0],
                "scale_best_iteration": [10],
                "shape_num_leaves": [31],
                "shape_min_child_samples": [50],
                "shape_reg_lambda": [10.0],
                "shape_best_iteration": [12],
            }
        ),
        "tca_execution": pd.DataFrame(
            {
                "method": ["raw"],
                "comparison_baseline": ["lightgbm_raw"],
                "seed": [13],
                "normalized_allocation_regret": [0.1],
                "absolute_modeled_impact_cost": [12.0],
                "completion_rate": [1.0],
                "implementation_shortfall_bps": [0.5],
                "mean_difference": [0.3],
                "ci_lower": [-0.1],
                "ci_upper": [0.1],
            }
        ),
        "confirmatory_statistics": pd.DataFrame(
            {
                "contrast_id": [1],
                "stage": ["representation"],
                "candidate": ["sparse"],
                "baseline": ["dense"],
                "endpoint": ["affine_normalized_latent_error"],
                "mean_difference": [-0.1],
                "median_difference": [-0.1],
                "ci_lower": [-0.2],
                "ci_upper": [-0.05],
                "paired_dates": [10],
                "date_win_rate": [0.8],
                "standardized_effect": [-0.5],
                "raw_p_value": [0.01],
                "holm_adjusted_p_value": [0.05],
            }
        ),
        "support_regime_diagnostics": pd.DataFrame(
            {
                "fold_id": ["fixture-fold"],
                "geometry": ["sparse"],
                "seed": [13],
                "zero_fraction": [0.75],
                "mean_active_dimensions": [8.0],
            }
        ),
        "appendix_sensitivities": pd.DataFrame(
            {"analysis": ["fixture_sensitivity"], "block_length_dates": [2]}
        ),
    }


def test_historical_writer_accepts_typed_empty_all_method_forecast_summaries(
    tmp_path: Path,
) -> None:
    tables = _synthetic_historical_tables()
    for name in ("forecast_performance", "forecast_by_asof"):
        tables[name] = tables[name].iloc[:0].copy()

    assert tables["forecast_by_asof"]["as_of_token"].dtype == np.dtype("int64")
    output = write_historical_paper_bundle(
        tmp_path,
        paper_run_id="synthetic-empty-all-method-forecast",
        tables=tables,
        provenance={"data_classification": "synthetic_fixture"},
        historical_schema_fixture=True,
    )

    assert len(list((output / "tables").glob("*.parquet"))) == len(HISTORICAL_TABLE_NAMES)
    assert len(list((output / "figures").glob("*.png"))) == len(HISTORICAL_FIGURE_NAMES)
    for name in ("forecast_performance", "forecast_by_asof"):
        result = pd.read_parquet(output / "tables" / f"{name}.parquet")
        assert result.empty
        assert set(tables[name].columns).issubset(result.columns)
    assert pd.read_parquet(output / "tables" / "forecast_by_asof.parquet")[
        "as_of_token"
    ].dtype == np.dtype("int64")


def test_historical_writer_rejects_malformed_and_empty_structural_tables(
    tmp_path: Path,
) -> None:
    tables = _synthetic_historical_tables()
    tables["forecast_by_asof"] = tables["forecast_by_asof"].iloc[:0].drop(columns="as_of_token")
    with pytest.raises(ValueError, match="Historical table forecast_by_asof"):
        write_historical_paper_bundle(
            tmp_path,
            paper_run_id="synthetic-malformed-empty-forecast",
            tables=tables,
            provenance={"data_classification": "synthetic_fixture"},
            historical_schema_fixture=True,
        )

    tables = _synthetic_historical_tables()
    tables["representation_accessibility"] = tables["representation_accessibility"].iloc[:0]
    with pytest.raises(ValueError, match="Historical table representation_accessibility"):
        write_historical_paper_bundle(
            tmp_path,
            paper_run_id="synthetic-empty-representation",
            tables=tables,
            provenance={"data_classification": "synthetic_fixture"},
            historical_schema_fixture=True,
        )


def test_historical_figures_render_absolute_tca_interval_and_mean_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tables = _synthetic_historical_tables()
    tables["forecast_by_asof"] = tables["forecast_by_asof"].iloc[:0].copy()
    axes_by_figure: dict[str, object] = {}
    save_historical = _save_historical

    def capture_axis(figure: object, path: Path, fixture_label: str | None) -> None:
        axes_by_figure[path.name] = figure.axes[0]  # type: ignore[attr-defined]
        save_historical(figure, path, fixture_label)  # type: ignore[arg-type]

    monkeypatch.setattr("execsim.ml.paper.reports._save_historical", capture_axis)
    _render_historical_figures(tmp_path, tables, fixture_label="synthetic fixture")

    assert (tmp_path / "allocation_regret_with_paired_intervals.png").is_file()
    interval_axis = axes_by_figure["allocation_regret_with_paired_intervals.png"]
    interval = interval_axis.collections[0].get_segments()  # type: ignore[attr-defined]
    np.testing.assert_allclose(interval[0][:, 1], [-0.1, 0.1])
    mean_marker = interval_axis.collections[1].get_offsets()  # type: ignore[attr-defined]
    np.testing.assert_allclose(mean_marker[:, 1], [0.3])

    for filename in ("forecast_performance_by_model.png", "forecast_error_vs_asof.png"):
        label_axis = axes_by_figure[filename]
        assert [text.get_text() for text in label_axis.texts] == [  # type: ignore[attr-defined]
            "No common forecast cases"
        ]
