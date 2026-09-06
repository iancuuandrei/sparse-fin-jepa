from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from datetime import date, time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from execsim.cli import main
from execsim.data.paper.manifests import file_sha256, write_json_atomic
from execsim.ml.models.lightgbm_adapter import LightGBMConfig, LightGBMVolumeModel
from execsim.ml.models.random_projection import projection_hash, random_projection_matrix
from execsim.ml.paper.benchmark import estimate_manifest_resources, predictor_capacity_smoke
from execsim.ml.paper.configs import load_paper_config, load_runtime_approval
from execsim.ml.paper.features import (
    append_embedding,
    build_raw_feature_frame,
    build_untrained_neural_control,
)
from execsim.ml.paper.forecast_provider import PaperLightGBMForecastProvider
from execsim.ml.paper.orchestration import (
    _formation_artifacts_ready,
    _has_frozen_v2_formation_evidence,
    _require_parameter_freeze,
    run_authorized_stages,
)
from execsim.ml.paper.provenance import build_run_provenance
from execsim.ml.paper.regimes import (
    fit_regime_thresholds,
    fit_unusual_session_thresholds,
    label_regimes,
    label_unusual_sessions,
)
from execsim.ml.paper.reports import (
    TABLE_NAMES,
    write_historical_paper_bundle,
    write_paper_bundle,
)
from execsim.ml.paper.statistics import (
    construct_complete_case_differences,
    holm_adjust_pvalues,
    moving_block_bootstrap,
    paper_forecast_metrics,
)
from execsim.ml.paper.tca import (
    SegmentCommittedMPCPolicy,
    balanced_sides,
    expand_volume_forecast,
    mean_seed_forecast,
    realized_volume_oracle_cost,
    select_liquidity_spaced_instruments,
)
from execsim.orders import ParentOrder
from execsim.policies import AdaptiveMPCPolicy, ExecutionConstraints
from execsim.simulator import simulate_policy


def test_lightgbm_raw_and_hybrid_fixture_and_random_placebo(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(40, 12))
    projection = random_projection_matrix(12, seed=13)
    hybrid = np.column_stack((raw, raw @ projection))
    remaining = np.exp(raw[:, 0] + 10)
    shape = np.exp(raw[:, :4])
    shape /= shape.sum(axis=1, keepdims=True)
    config = LightGBMConfig(n_estimators=8, min_child_samples=2, num_threads=1)

    for features in (raw, hybrid):
        model = LightGBMVolumeModel(config).fit(features[:30], remaining[:30], shape[:30])
        total, shares = model.predict(features[30:])
        assert (total >= 0).all()
        assert np.allclose(shares.sum(axis=1), 1.0)
    artifact = model.save_native(
        tmp_path / "lightgbm",
        {
            "fold_id": "fold-1",
            "feature_schema_version": "paper-lgbm-v1",
            "training_cutoff": "2023-12-29",
            "validation_range": ["2024-01-02", "2024-03-28"],
            "categorical_features": [],
        },
    )
    restored, metadata = LightGBMVolumeModel.load_native(artifact)
    restored_total, restored_shape = restored.predict(hybrid[30:])
    assert metadata["fold_id"] == "fold-1"
    assert np.allclose(restored_total, total)
    assert np.allclose(restored_shape, shares)
    assert projection_hash(projection) == projection_hash(random_projection_matrix(12, seed=13))


def test_lightgbm_uses_pandas_categories_and_long_shape_valid_horizons() -> None:
    scale = pd.DataFrame(
        {
            "symbol": ["AAPL", "MSFT"] * 8,
            "x": np.arange(16),
            "baseline_remaining_volume": np.full(16, 90.0),
        }
    )
    total = np.arange(16, dtype=float) + 100
    shape = scale.loc[scale.index.repeat(3)].reset_index(drop=True)
    shape["case_id"] = np.repeat(np.arange(16), 3)
    shape["target_bucket"] = np.tile(np.arange(3), 16)
    shares = np.tile([0.2, 0.3, 0.5], 16)
    model = LightGBMVolumeModel(
        LightGBMConfig(n_estimators=8, min_child_samples=2, num_threads=1)
    ).fit_frames(scale, total, shape, shares, categorical_features=("symbol",))
    prediction_shape = shape.iloc[:6].copy()
    prediction_shape["target_valid"] = [True, True, True, True, False, False]
    predicted_total, predicted = model.predict_frames(
        scale.iloc[:2], prediction_shape, group_columns=("case_id",)
    )

    assert len(predicted_total) == 2
    assert predicted.groupby("case_id")["conditional_share"].sum().eq(1.0).all()
    assert predicted.loc[~prediction_shape["target_valid"], "conditional_share"].eq(0).all()
    with pytest.raises(ValueError, match="categorical state"):
        model.predict_frames(
            pd.DataFrame({"symbol": ["NVDA"], "x": [1], "baseline_remaining_volume": [90.0]}),
            prediction_shape.iloc[:3].assign(symbol="NVDA", case_id=0),
            group_columns=("case_id",),
        )


def test_learned_provider_reuses_boundary_forecast_and_preserves_partial_token_position() -> None:
    calls = 0

    class FakeModel:
        def predict_frames(self, scale, shape, *, group_columns):
            del scale, group_columns
            result = shape[["case_id", "target_bucket"]].copy()
            result["conditional_share"] = 1 / len(result)
            return np.asarray([22_000.0]), result

    def resolver(symbol, session_date, generated_at, observations):
        nonlocal calls
        del symbol, session_date, observations
        calls += 1
        remaining_tokens = (16 * 60 - (generated_at.hour * 60 + generated_at.minute)) // 15
        return (
            pd.DataFrame({"case_id": [0]}),
            pd.DataFrame({"case_id": 0, "target_bucket": np.arange(remaining_tokens)}),
        )

    provider = PaperLightGBMForecastProvider(
        FakeModel(),  # type: ignore[arg-type]
        feature_resolver=resolver,
        within_token_profile=np.arange(1, 16) / np.arange(1, 16).sum(),
        training_cutoff=date(2023, 12, 29),
        manifest_hash="a" * 64,
        method_id="fixture",
    )
    full = pd.date_range("2024-04-01 10:30", "2024-04-01 15:30", freq="min", tz="America/New_York")
    boundary = provider.forecast(
        symbol="AAPL",
        session_date=date(2024, 4, 1),
        generated_at=full[0],
        bucket_timestamps=tuple(full),
    )
    for offset in (1, 7, 14):
        truncated = provider.forecast(
            symbol="AAPL",
            session_date=date(2024, 4, 1),
            generated_at=full[offset],
            bucket_timestamps=tuple(full[offset:]),
        )
        assert truncated.expected_volumes[0] == boundary.expected_volumes[offset]
    provider.forecast(
        symbol="AAPL",
        session_date=date(2024, 4, 1),
        generated_at=full[15],
        bucket_timestamps=tuple(full[15:]),
    )
    assert calls == 2


def test_predictor_capacity_protocol_records_cost_and_horizon_errors() -> None:
    rows = predictor_capacity_smoke(repetitions=1)

    assert [row["predictor_family"] for row in rows] == [
        "affine_ridge",
        "mlp_64",
        "mlp_256",
    ]
    assert all(int(row["parameters"]) > 0 for row in rows)
    assert all(len(row["mse_by_horizon"]) == 4 for row in rows)


class _PaperFixtureForecaster:
    @property
    def provider_id(self) -> str:
        return "paper-fixture"

    def forecast(
        self,
        *,
        symbol: str,
        session_date: date,
        generated_at: pd.Timestamp,
        bucket_timestamps,
        observations=None,
    ):
        del observations
        timestamps = tuple(bucket_timestamps)
        return expand_volume_forecast(
            symbol=symbol,
            session_date=session_date,
            generated_at=generated_at,
            minute_timestamps=timestamps,
            expected_remaining_volume=100.0 * len(timestamps),
            conditional_token_shape=np.full(
                int(np.ceil(len(timestamps) / 15)), 1 / int(np.ceil(len(timestamps) / 15))
            ),
            within_token_profile=np.full(15, 1 / 15),
            training_cutoff=date(2024, 1, 2),
            manifest_hash="a" * 64,
            forecaster_id=self.provider_id,
        )


def test_paper_forecast_object_feeds_deterministic_mpc() -> None:
    timestamps = pd.date_range("2024-01-03 10:30", periods=4, freq="min", tz="America/New_York")
    bars = pd.DataFrame(
        {
            "symbol": "AAPL",
            "timestamp": timestamps,
            "open": 100.0,
            "high": 100.1,
            "low": 99.9,
            "close": 100.0,
            "volume": 100,
            "trade_count": 10,
            "vwap": 100.0,
        }
    )
    result = simulate_policy(
        parent_order=ParentOrder("AAPL", "buy", 20, date(2024, 1, 3), time(10, 30), time(10, 34)),
        bars=bars,
        policy=AdaptiveMPCPolicy(temporary_impact=0.001, risk_aversion=0.0),
        constraints=ExecutionConstraints(0.1, 0.1),
        forecast_provider=_PaperFixtureForecaster(),
    )

    assert result.summary.filled_qty == 20
    assert result.summary.n_optimization_decisions == 4
    forecast = _PaperFixtureForecaster().forecast(
        symbol="AAPL",
        session_date=date(2024, 1, 3),
        generated_at=timestamps[0],
        bucket_timestamps=tuple(timestamps),
    )
    ensemble = mean_seed_forecast((forecast, forecast, forecast))
    assert ensemble.expected_volumes == forecast.expected_volumes


def test_paper_mpc_commits_each_fifteen_minute_segment() -> None:
    timestamps = pd.date_range("2024-01-03 10:30", periods=31, freq="min", tz="America/New_York")
    bars = pd.DataFrame(
        {
            "symbol": "AAPL",
            "timestamp": timestamps,
            "open": 100.0,
            "high": 100.1,
            "low": 99.9,
            "close": 100.0,
            "volume": 1_000,
            "trade_count": 10,
            "vwap": 100.0,
        }
    )
    policy = SegmentCommittedMPCPolicy(half_spread=0.005, temporary_impact=0.1)
    result = simulate_policy(
        parent_order=ParentOrder("AAPL", "buy", 300, date(2024, 1, 3), time(10, 30), time(11, 1)),
        bars=bars,
        policy=policy,
        constraints=ExecutionConstraints(0.1, 0.1),
        forecast_provider=_PaperFixtureForecaster(),
    )
    assert 0 < result.summary.filled_qty <= 300
    assert policy.solve_count == 3
    assert policy.solve_count < len(result.execution_log)


def test_realized_volume_oracle_uses_the_continuous_full_horizon_solution() -> None:
    bars = pd.DataFrame(
        {
            "timestamp": pd.date_range(
                "2024-01-03 10:30", periods=2, freq="min", tz="America/New_York"
            ),
            "volume": [100.0, 300.0],
        }
    )
    cost = realized_volume_oracle_cost(
        bars,
        quantity=40,
        start=time(10, 30),
        end=time(10, 32),
        arrival_price=100.0,
    )
    # The continuous cap is [10, 30], which is also the proportional optimum.
    assert cost == pytest.approx(0.4)


def test_block_bootstrap_and_synthetic_report_bundle(tmp_path: Path) -> None:
    rows = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=12, freq="D"),
            "fold_id": ["fold-1"] * 6 + ["fold-2"] * 6,
            "difference": [-1.0, -0.5, 0.2, -0.1, -0.3, 0.1] * 2,
        }
    )
    result = moving_block_bootstrap(rows, repetitions=200, seed=13)
    forecast_metrics = paper_forecast_metrics(
        np.asarray([100.0, 200.0]),
        np.asarray([110.0, 180.0]),
        np.asarray([[0.4, 0.6], [0.5, 0.5]]),
        np.asarray([[0.5, 0.5], [0.4, 0.6]]),
    )
    tables = {name: pd.DataFrame({"metric": [name], "value": [1.0]}) for name in TABLE_NAMES}
    output = write_paper_bundle(
        tmp_path,
        paper_run_id="synthetic-fixture",
        tables=tables,
        provenance={"data_classification": "synthetic_fixture", "historical_training": "NOT RUN"},
    )

    assert result.paired_dates == 12
    assert result.mean_difference < 0
    assert forecast_metrics["log_remaining_volume_mae"] > 0
    assert forecast_metrics["conditional_curve_wasserstein"] > 0
    assert (output / "PAPER_OUTLINE.md").is_file()
    assert len(list((output / "tables").glob("*.parquet"))) == 4
    assert len(list((output / "figures").glob("*.png"))) == 4
    provenance = build_run_provenance(
        paper_run_id="synthetic-fixture",
        supplied={"sequence_manifest_hash": "a" * 64},
        repository_root=Path(__file__).resolve().parents[1],
    )
    assert len(str(provenance["git_commit"])) == 40
    assert provenance["sequence_manifest_hash"] == "a" * 64
    assert "torch" in provenance["dependency_versions"]


def test_statistics_intersect_exact_complete_cases_before_date_averaging() -> None:
    rows = pd.DataFrame(
        {
            "method": ["base", "base", "candidate", "candidate"],
            "fold_id": ["fold-1"] * 4,
            "date": [date(2024, 4, 1), date(2024, 4, 2), date(2024, 4, 1), date(2024, 4, 3)],
            "instrument_id": ["a", "b", "a", "c"],
            "cost": [10.0, 20.0, 8.0, 9.0],
        }
    )
    result = construct_complete_case_differences(
        rows,
        baseline="base",
        candidate="candidate",
        value_column="cost",
        identity_columns=("fold_id", "date", "instrument_id"),
    )

    assert result.matched_rows == 1
    assert result.dropped_baseline_rows == result.dropped_candidate_rows == 1
    assert result.paired_rows["difference"].iloc[0] == -2


def test_historical_report_requires_named_schemas_and_measured_intervals(tmp_path: Path) -> None:
    tables = {
        "dataset_folds_exclusions": pd.DataFrame(
            {"fold_id": ["fold-1"], "partition": ["test"], "included": [4], "excluded": [1]}
        ),
        "representation_accessibility": pd.DataFrame(
            {
                "geometry": ["sparse"],
                "seed": [13],
                "horizon": [1],
                "probe_capacity": ["affine_ridge"],
                "parameter_count": [1_024],
                "approximate_macs": [1_000],
                "inference_seconds": [0.01],
                "normalized_latent_error": [0.2],
                "zero_baseline": [1.0],
                "train_mean_baseline": [0.9],
                "persistence_baseline": [0.8],
                "observable_volume_probe_mae": [0.15],
                "observable_volume_probe_rmse": [0.2],
                "zero_fraction": [0.75],
                "mean_active_dimensions": [32.0],
            }
        ),
        "forecasting": pd.DataFrame(
            {
                "method": ["raw"],
                "as_of_token": [4],
                "log_remaining_volume_mae": [0.2],
                "conditional_curve_error": [0.1],
            }
        ),
        "execution": pd.DataFrame(
            {
                "method": ["raw"],
                "comparison_baseline": ["raw"],
                "seed": [13],
                "normalized_allocation_regret": [0.1],
                "absolute_modeled_impact_cost": [12.0],
                "completion_rate": [1.0],
                "implementation_shortfall_bps": [0.5],
                "mean_difference": [-0.1],
                "ci_lower": [-0.2],
                "ci_upper": [-0.05],
            }
        ),
    }
    output = write_historical_paper_bundle(
        tmp_path,
        paper_run_id="historical-schema-fixture",
        tables=tables,
        provenance={"data_classification": "synthetic_fixture"},
        historical_schema_fixture=True,
    )

    assert (output / "REPORT.md").is_file()
    assert len(list((output / "figures").glob("*.png"))) == 4
    with pytest.raises(ValueError, match="Historical table"):
        write_historical_paper_bundle(
            tmp_path,
            paper_run_id="broken",
            tables={**tables, "forecasting": pd.DataFrame({"value": [1]})},
            provenance={"data_classification": "synthetic_fixture"},
            historical_schema_fixture=True,
        )


def test_holm_adjustment_is_monotone_in_sorted_pvalues() -> None:
    values = np.asarray([0.01, 0.04, 0.03, 0.20, 0.001])
    adjusted = holm_adjust_pvalues(values)
    ordered = np.argsort(values, kind="stable")
    assert np.all(np.diff(adjusted[ordered]) >= 0)
    assert np.all((0 <= adjusted) & (adjusted <= 1))


def test_protocol_freeze_is_complete_and_checksum_bound() -> None:
    config = load_paper_config(Path("configs/paper/sparse_jepa"))
    freeze = config.design_freeze
    freeze_path = config.root / "design-freeze-v1.json"
    sidecar = freeze_path.with_suffix(".sha256").read_text(encoding="utf-8").split()
    safe_path = config.root / "safe-default-receipt-v1.json"
    safe_sidecar = safe_path.with_suffix(".sha256").read_text(encoding="utf-8").split()

    assert freeze["schema_version"] == "paper-design-freeze-v1"
    assert freeze["parameter_selection_receipt"] == "NOT RUN"
    assert sidecar == [file_sha256(freeze_path), freeze_path.name]
    assert safe_sidecar == [file_sha256(safe_path), safe_path.name]


def test_locked_test_parameter_freeze_requires_the_exact_model_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    loaded = load_paper_config(Path("configs/paper/sparse_jepa"))
    sections = {**loaded.sections, "data": {**loaded.data, "artifact_root": str(tmp_path)}}
    config = replace(loaded, sections=sections)
    monkeypatch.setattr("execsim.ml.paper.orchestration._git_head", lambda: "f" * 40)
    selection = tmp_path / "selection" / "rdm-lambda.json"
    write_json_atomic(
        selection,
        {
            "schema_version": "paper-rdm-lambda-selection-v1",
            "selection_partition": "fold-1/validation",
            "seed": 13,
            "paper_config_hash": config.config_hash,
            "selected_rdm_lambda": 1.0,
            "candidates": [
                {
                    "rdm_lambda": value,
                    "geometry": geometry,
                    "fold_id": "fold-1",
                    "seed": 13,
                    "observable_probe_error": {0.1: 0.5, 1.0: 0.2, 10.0: 0.8}[value],
                    "collapse_gate_status": "PASS",
                    "checkpoint_hash": f"{geometry}-{value}",
                }
                for value in (0.1, 1.0, 10.0)
                for geometry in ("dense", "sparse")
            ],
            "test_or_tca_used": False,
        },
    )
    records = []
    for fold in config.evaluation["folds"]:
        for method, seed in (
            ("raw", None),
            ("untrained_neural", None),
            *(
                (geometry, int(seed))
                for geometry in ("dense", "sparse")
                for seed in config.representation["seeds"]
            ),
        ):
            relative = (
                Path("lightgbm")
                / str(fold["id"])
                / method
                / str(seed if seed is not None else "shared")
                / "manifest.json"
            )
            artifact = tmp_path / relative
            write_json_atomic(artifact, {"method": method, "seed": seed})
            records.append({"path": relative.as_posix(), "sha256": file_sha256(artifact)})
    freeze = tmp_path / "selection" / "parameter-freeze-v1.json"
    payload = {
        "status": "PARAMETERS_FROZEN",
        "git_commit": "f" * 40,
        "paper_config_hash": config.config_hash,
        "rdm_lambda_receipt_sha256": file_sha256(selection),
        "selected_rdm_lambda": 1.0,
        "lightgbm_manifests": records,
        "test_or_tca_used": False,
    }
    write_json_atomic(freeze, payload)
    assert _require_parameter_freeze(config)["status"] == "PARAMETERS_FROZEN"

    write_json_atomic(freeze, {**payload, "lightgbm_manifests": records[:-1]})
    with pytest.raises(ValueError, match="incomplete or duplicated"):
        _require_parameter_freeze(config)


def test_manifest_resource_estimate_is_derived_and_fail_closed(tmp_path: Path) -> None:
    manifests = []
    for fold_id in ("fold-1", "fold-2", "fold-3"):
        path = tmp_path / f"{fold_id}.json"
        write_json_atomic(
            path,
            {"partition_counts": {"train": 100, "validation": 20, "test": 30}},
        )
        manifests.append(path)
    bounds = {
        "maximum_representation_runs": 29,
        "maximum_jepa_steps": 1_000_000,
        "maximum_shape_rows": 1_000_000,
        "maximum_embedding_bytes": 1_000_000_000,
    }
    estimate = estimate_manifest_resources(
        tuple(manifests), batch_size=32, max_epochs=2, bounds=bounds
    )
    assert estimate["sequence_sessions_across_expanding_folds"] == 450
    assert estimate["maximum_representation_runs"] == 29
    assert estimate["resource_gate"] == "PASS"

    with pytest.raises(RuntimeError, match="maximum_jepa_steps"):
        estimate_manifest_resources(
            tuple(manifests),
            batch_size=32,
            max_epochs=2,
            bounds={**bounds, "maximum_jepa_steps": 1},
        )


def test_regimes_liquidity_spacing_and_side_assignment_are_train_defined() -> None:
    training = pd.DataFrame(
        {
            "volume_surprise": np.arange(20),
            "realized_volatility": np.arange(20) / 10,
            "shape_error": np.arange(20) / 20,
        }
    )
    thresholds = fit_regime_thresholds(training)
    labeled = label_regimes(training, thresholds)
    universe = pd.DataFrame(
        {"rank": range(1, 101), "instrument_id": [f"id-{index:03d}" for index in range(100)]}
    )
    selected = select_liquidity_spaced_instruments(universe)
    sides = balanced_sides(selected, date(2024, 4, 1))

    assert len(selected) == 30
    assert set(labeled.columns).issuperset(
        {"high_volume_surprise", "high_volatility", "abnormal_shape", "ordinary"}
    )
    assert list(sides.values()).count("buy") == 15
    assert list(sides.values()).count("sell") == 15


def test_unusual_session_composite_uses_the_same_historical_baseline_statistic() -> None:
    training = pd.DataFrame(
        {
            "volume_surprise": np.arange(10, dtype=float),
            "realized_volatility": np.arange(10, dtype=float) / 10,
            "historical_baseline_curve_error": np.arange(10, dtype=float) / 100,
        }
    )
    thresholds = fit_unusual_session_thresholds(training)
    heldout = pd.DataFrame(
        {
            "volume_surprise": [0.0, 10.0],
            "realized_volatility": [0.0, 0.0],
            "historical_baseline_curve_error": [0.0, 0.0],
        }
    )
    labeled = label_unusual_sessions(heldout, thresholds)
    assert labeled["regime"].tolist() == ["ordinary", "unusual"]


def test_lightgbm_raw_hybrid_and_untrained_placebo_share_the_causal_context() -> None:
    context = np.zeros((3, 8, 18))
    mask = np.ones((3, 8), dtype=bool)
    metadata = pd.DataFrame(
        {
            "as_of_bucket": [4, 5, 6],
            "target_bucket": [5, 6, 7],
            "horizon_offset": [1, 1, 1],
            "minutes_remaining": [330, 315, 300],
            "weekday": [0, 0, 0],
            "month": [4, 4, 4],
            "is_month_end": [False, False, False],
            "is_quarter_end": [False, False, False],
            "symbol": ["AAPL", "MSFT", "NVDA"],
            "liquidity_group": [1, 2, 3],
        }
    )
    raw = build_raw_feature_frame(context, mask, metadata)
    neural_values, network_hash = build_untrained_neural_control(
        context, mask, np.ones((3, 4), dtype=bool), fold_seed=13
    )
    repeated, repeated_hash = build_untrained_neural_control(
        context, mask, np.ones((3, 4), dtype=bool), fold_seed=13
    )
    hybrid = append_embedding(raw, neural_values)

    assert raw.shape[1] == 162
    assert neural_values.shape == (3, 644)
    assert np.array_equal(neural_values, repeated)
    assert network_hash == repeated_hash and len(network_hash) == 64
    assert hybrid.shape[1] == raw.shape[1] + 644


def test_paper_cli_dry_run_does_not_enable_expensive_operations(capsys) -> None:
    from execsim.cli import main

    status = main(["ml", "paper", "plan", "--dry-run"])

    assert status == 0
    output = capsys.readouterr().out
    assert '"network_enabled": false' in output
    assert "No network acquisition" in output
    assert main(["ml", "paper", "run", "--dry-run"]) == 0
    assert "No network acquisition" in capsys.readouterr().out

    with pytest.raises(SystemExit):
        main(["ml", "paper", "download-data"])

    assert main(["ml", "paper", "train-representation", "--synthetic-fixture"]) == 0
    representation_output = capsys.readouterr().out
    assert '"data_classification": "synthetic_fixture"' in representation_output
    assert main(["ml", "paper", "train-volume-model", "--synthetic-fixture"]) == 0
    volume_output = capsys.readouterr().out
    assert '"shape_row_sums"' in volume_output


def test_all_paper_cli_commands_have_executable_synthetic_fixtures(tmp_path: Path, capsys) -> None:
    from execsim.cli import main

    commands = (
        ("export-embeddings", ["--output", str(tmp_path / "embeddings")]),
        ("evaluate-forecast", []),
        ("evaluate-representation", []),
        ("run-tca", []),
        ("report", ["--output", str(tmp_path / "reports")]),
    )
    for command, extra in commands:
        assert main(["ml", "paper", command, "--synthetic-fixture", *extra]) == 0
        assert '"data_classification": "synthetic_fixture"' in capsys.readouterr().out

    assert (tmp_path / "embeddings" / "synthetic-embedding-manifest.json").is_file()
    assert (tmp_path / "reports").is_dir()


def test_ordinary_execsim_imports_when_deep_and_lightgbm_are_unavailable() -> None:
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'lightgbm', 'safetensors'}:
        raise ImportError('paper extra intentionally unavailable')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
import execsim
import execsim.ml
from execsim.cli import main
assert main(['smoke']) == 0
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    assert "smoke: ok" in result.stdout


def test_tracked_paper_configs_keep_privileged_operations_disabled() -> None:
    root = Path(__file__).resolve().parents[1] / "configs" / "paper" / "sparse_jepa"
    data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    sequences = yaml.safe_load((root / "sequences.yaml").read_text(encoding="utf-8"))
    representation = yaml.safe_load((root / "representation.yaml").read_text(encoding="utf-8"))

    assert data["allow_network"] is False
    assert data["allow_historical_training"] is False
    assert data["allow_full_paper_run"] is False
    assert sequences["horizons"] == [1, 2, 4, 8]
    assert representation["seeds"] == [13, 29, 47]


def _runtime_approval_payload(config, **approved: bool) -> dict[str, object]:
    scopes = {
        "target_acquisition": False,
        "historical_training": False,
        "locked_result_evaluation": False,
    }
    scopes.update(approved)
    return {
        "schema_version": "paper-runtime-approval-v1",
        "approval_id": "test-explicit-approval",
        "approved_at_utc": "2026-09-06T00:00:00Z",
        "protocol_id": config.paper_run_id,
        "paper_config_sha256": config.config_hash,
        "approvals": scopes,
    }


def _runtime_approval_path(tmp_path: Path, name: str) -> Path:
    root = Path(".runtime/paper-approvals") / tmp_path.name
    root.mkdir(parents=True, exist_ok=True)
    return root / name


def test_v2_runtime_authorization_is_external_identity_bound_and_scoped(
    tmp_path: Path,
) -> None:
    config = load_paper_config(Path("configs/paper/sparse_jepa_v2"))
    approval_path = _runtime_approval_path(tmp_path, "network-approval.json")
    write_json_atomic(
        approval_path,
        _runtime_approval_payload(config, target_acquisition=True),
    )
    approval = load_runtime_approval(approval_path, config)

    with pytest.raises(PermissionError, match="matching runtime approval"):
        config.authorize("target_acquisition", approval=None, cli_enabled=False)
    with pytest.raises(PermissionError, match="matching runtime approval"):
        config.authorize("target_acquisition", approval=None, cli_enabled=True)
    with pytest.raises(PermissionError, match="matching runtime approval"):
        config.authorize("target_acquisition", approval=approval, cli_enabled=False)
    config.authorize("target_acquisition", approval=approval, cli_enabled=True)
    with pytest.raises(PermissionError, match="matching runtime approval"):
        config.authorize("historical_training", approval=approval, cli_enabled=True)

    training_path = _runtime_approval_path(tmp_path, "training-approval.json")
    write_json_atomic(
        training_path,
        _runtime_approval_payload(config, historical_training=True),
    )
    training = load_runtime_approval(training_path, config)
    with pytest.raises(PermissionError, match="matching runtime approval"):
        config.authorize("locked_result_evaluation", approval=training, cli_enabled=True)


def test_cli_requires_both_runtime_approval_and_matching_flag(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import execsim.ml.paper.orchestration as orchestration

    config_path = Path("configs/paper/sparse_jepa_v2/data.yaml")
    config = load_paper_config(config_path)
    approval_path = _runtime_approval_path(tmp_path, "cli-network-approval.json")
    write_json_atomic(
        approval_path,
        _runtime_approval_payload(config, target_acquisition=True),
    )
    base = ["ml", "paper", "download-data", "--config", str(config_path)]

    with pytest.raises(SystemExit) as flag_only:
        main([*base, "--enable-network"])
    assert flag_only.value.code == 2
    with pytest.raises(SystemExit) as approval_only:
        main([*base, "--runtime-approval", str(approval_path)])
    assert approval_only.value.code == 2

    calls: list[tuple[bool, str]] = []

    def authorized_stub(config, *, cli_enabled, runtime_approval):
        calls.append((cli_enabled, runtime_approval.approval_id))
        return {"status": "authorization boundary reached; network not called"}

    monkeypatch.setattr(orchestration, "download_data_stage", authorized_stub)
    assert (
        main(
            [
                *base,
                "--runtime-approval",
                str(approval_path),
                "--enable-network",
            ]
        )
        == 0
    )
    assert calls == [(True, "test-explicit-approval")]
    assert "network not called" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("field", "value"),
    (("protocol_id", "sparse-jepa-v1"), ("paper_config_sha256", "0" * 64)),
)
def test_runtime_approval_for_another_identity_is_denied(
    tmp_path: Path, field: str, value: str
) -> None:
    config = load_paper_config(Path("configs/paper/sparse_jepa_v2"))
    approval_path = _runtime_approval_path(tmp_path, f"wrong-{field}.json")
    payload = _runtime_approval_payload(config, target_acquisition=True)
    payload[field] = value
    write_json_atomic(approval_path, payload)

    with pytest.raises(PermissionError, match="does not match"):
        load_runtime_approval(approval_path, config)


def test_v2_run_uses_frozen_daily_formation_state_without_v1_key() -> None:
    config = load_paper_config(Path("configs/paper/sparse_jepa_v2"))

    result = run_authorized_stages(
        config,
        network_cli_enabled=False,
        training_cli_enabled=False,
        full_run_cli_enabled=False,
    )

    assert file_sha256(config.root / "design-freeze-v2.json") == (
        "eea790c79c16e69ee3997c8c964e7716049c379e1cbb248b2602acd2e19b8d27"
    )
    assert "formation_corpus_root" not in config.data
    assert result == {"build_universe": "reused", "download_data": "DATA NOT ACQUIRED"}


def test_cli_v2_run_reaches_target_gate_without_authorization(capsys) -> None:
    assert (
        main(
            [
                "ml",
                "paper",
                "run",
                "--config",
                "configs/paper/sparse_jepa_v2/data.yaml",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    assert payload == {"build_universe": "reused", "download_data": "DATA NOT ACQUIRED"}


def test_formation_readiness_dispatches_by_protocol(tmp_path: Path, monkeypatch) -> None:
    import execsim.ml.paper.orchestration as orchestration

    v1 = load_paper_config(Path("configs/paper/sparse_jepa"))
    loaded_v2 = load_paper_config(Path("configs/paper/sparse_jepa_v2"))
    daily = tmp_path / "daily-bars.parquet"
    daily.write_bytes(b"direct daily formation artifact")
    v2 = replace(
        loaded_v2,
        sections={
            **loaded_v2.sections,
            "data": {**loaded_v2.data, "formation_daily_corpus": str(daily)},
        },
    )
    observed: list[Path] = []

    def record_minute_corpus(path: Path) -> bool:
        observed.append(path)
        return True

    monkeypatch.setattr(orchestration, "_has_parquet_corpus", record_minute_corpus)

    assert _formation_artifacts_ready(v1)
    assert observed == [Path(v1.data["formation_corpus_root"])]
    observed.clear()
    assert _formation_artifacts_ready(v2)
    assert observed == []


def test_frozen_v2_formation_evidence_reuses_preapproval_receipts(tmp_path: Path) -> None:
    loaded = load_paper_config(Path("configs/paper/sparse_jepa_v2"))
    daily = tmp_path / "daily.parquet"
    receipt = tmp_path / "daily-receipt.json"
    universe = tmp_path / "universe.json"
    daily.write_bytes(b"frozen daily corpus")
    receipt.write_bytes(b"frozen preapproval receipt")
    write_json_atomic(
        universe,
        {
            "status": "complete",
            "paper_config_hash": loaded.config_hash,
            "members": [{"instrument_id": f"asset-{index}"} for index in range(100)],
        },
    )
    evidence = {
        "status": "COMPLETE",
        "daily_corpus_sha256": file_sha256(daily),
        "daily_receipt_sha256": file_sha256(receipt),
        "universe_manifest_sha256": file_sha256(universe),
    }
    config = replace(
        loaded,
        sections={
            **loaded.sections,
            "data": {
                **loaded.data,
                "formation_daily_corpus": str(daily),
                "formation_daily_receipt": str(receipt),
                "universe_manifest": str(universe),
            },
        },
        design_freeze={**loaded.design_freeze, "formation_evidence": evidence},
    )

    assert _has_frozen_v2_formation_evidence(config)
    receipt.write_bytes(b"changed receipt")
    assert not _has_frozen_v2_formation_evidence(config)
