"""Exercise report construction and freezing with synthetic production-shaped inputs."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
from test_paper_sequences import _paper_session
from test_representation_contracts import export_fixture_checkpoint
from test_representation_contracts import regime_corpus as _fixture_regime_corpus  # noqa: F401

from execsim.data.paper.manifests import file_sha256, read_json, write_json_atomic
from execsim.ml.paper import orchestration
from execsim.ml.paper.configs import load_paper_config
from execsim.ml.paper.evaluation_artifacts import (
    forecast_metric_frame,
    merge_result_shards,
    publish_frames,
)
from execsim.ml.paper.evaluation_execution import evaluation_report_root, evaluation_root
from execsim.ml.paper.lightgbm_data import LightGBMFrames
from execsim.ml.paper.reports import (
    HISTORICAL_FIGURE_NAMES,
    HISTORICAL_TABLE_NAMES,
    HISTORICAL_TABLE_SCHEMAS,
)
from execsim.ml.sequences.corpus import build_fold_sequence_corpus

_SOURCE_COMMIT = "synthetic-report-source"
_SOURCE_TREE = "synthetic-report-tree"
_REPRESENTATION_SOURCE_COMMIT = "fixture-only"


def _make_config(tmp_path: Path) -> tuple[SimpleNamespace, list[dict[str, Any]]]:
    loaded = load_paper_config(Path("configs/paper/sparse_jepa_v2"))
    folds = deepcopy(loaded.evaluation["folds"])
    evaluation = {
        **deepcopy(loaded.evaluation),
        "folds": folds[:1],
        "bootstrap_repetitions": 99,
        "bootstrap_block_sensitivity_dates": [1, 2],
    }
    representation = {
        **deepcopy(loaded.representation),
        "seeds": [13],
        "rdm_projections_evaluation": 16,
        "probe_ridge_alphas": [1.0],
        "probe_mlp_epochs": 1,
    }
    config = SimpleNamespace(
        artifact_root=tmp_path,
        runtime_evaluation_root=tmp_path / "evaluation",
        runtime_representation_root=None,
        report_root=tmp_path / "default-reports",
        paper_run_id="report-stage-contract-fixture",
        config_hash="synthetic-report-contract-config",
        evaluation=evaluation,
        representation=representation,
        sequences={
            **deepcopy(loaded.sequences),
            "session_cache_size": 2,
            "num_workers": 0,
            "prefetch_factor": 1,
        },
        authorize=lambda *args, **kwargs: None,
    )
    return config, folds


def _execution_identity(config: SimpleNamespace) -> dict[str, str]:
    return {
        "source_commit": _SOURCE_COMMIT,
        "source_tree": _SOURCE_TREE,
        "paper_config_hash": config.config_hash,
        "parameter_freeze_sha256": file_sha256(
            config.artifact_root / "selection/parameter-freeze-v1.json"
        ),
    }


def _write_three_fold_sequence_manifests(
    config: SimpleNamespace,
    folds: list[dict[str, Any]],
    first_sequence: Path,
) -> None:
    """Retain the real first-fold fixture and build two more real sequence corpora."""
    assert read_json(first_sequence)["fold_id"] == "fold-1"
    instruments = (("asset-2", "BBB"), ("asset-1", "AAA"), ("benchmark-spy", "SPY"))
    dates_by_fold = {
        "fold-2": tuple(
            date.fromisoformat(value)
            for value in (
                "2024-06-03",
                "2024-06-04",
                "2024-07-02",
                "2024-10-02",
            )
        ),
        "fold-3": tuple(
            date.fromisoformat(value)
            for value in (
                "2024-12-02",
                "2024-12-03",
                "2025-01-03",
                "2025-07-02",
            )
        ),
    }
    for fold in folds[1:]:
        fold_id = str(fold["id"])
        bars = pd.concat(
            [
                _paper_session(instrument, symbol, day, index)
                for index, (instrument, symbol) in enumerate(instruments)
                for day in dates_by_fold[fold_id]
            ],
            ignore_index=True,
        )
        build_fold_sequence_corpus(
            bars,
            universe_members=tuple(
                {"instrument_id": instrument, "formation_symbol": symbol}
                for instrument, symbol in instruments[:-1]
            ),
            corporate_actions=pd.DataFrame(),
            fold_id=fold_id,
            output_root=config.artifact_root / "sequences",
            universe_manifest_hash="a" * 64,
            corporate_action_manifest_hash="b" * 64,
            config_hash=config.config_hash,
            spy_instrument_id="benchmark-spy",
            data_classification="synthetic_fixture",
            quality_protocol="resolution-aware-v2",
        )
    for fold in folds:
        manifest = config.artifact_root / "sequences" / str(fold["id"]) / "sequence-manifest.json"
        payload = read_json(manifest)
        assert set(payload["partition_counts"]) == {"train", "validation", "test"}


def _write_frozen_manifest_matrix(config: SimpleNamespace, folds: list[dict[str, Any]]) -> None:
    """Create the exact synthetic 18-checkpoint and 24-model freeze inventories."""
    seeds = (13, 29, 47)
    for fold in folds:
        fold_id = str(fold["id"])
        for geometry in ("dense", "sparse"):
            for seed in seeds:
                path = (
                    config.artifact_root
                    / "representations"
                    / fold_id
                    / geometry
                    / str(seed)
                    / "final"
                    / "manifest.json"
                )
                if not path.exists():
                    write_json_atomic(
                        path,
                        {
                            "fold_id": fold_id,
                            "geometry": geometry,
                            "seed": seed,
                            "code_commit": _REPRESENTATION_SOURCE_COMMIT,
                            "data_classification": "synthetic_fixture",
                        },
                    )

        methods = (
            ("raw", None),
            ("untrained_neural", None),
            *((geometry, seed) for geometry in ("dense", "sparse") for seed in seeds),
        )
        for method, seed in methods:
            path = (
                config.artifact_root
                / "lightgbm"
                / fold_id
                / method
                / str(seed if seed is not None else "shared")
                / "manifest.json"
            )
            write_json_atomic(
                path,
                {
                    "fold_id": fold_id,
                    "method": method,
                    "seed": seed,
                    "selected_scale_config": {
                        "num_leaves": 15,
                        "min_child_samples": 10,
                        "reg_lambda": 1.0,
                    },
                    "selected_shape_config": {
                        "num_leaves": 31,
                        "min_child_samples": 20,
                        "reg_lambda": 2.0,
                    },
                    "selected_iterations": {"scale": 7, "shape": 9},
                    "data_classification": "synthetic_fixture",
                },
            )


def _forecast_metric_rows() -> pd.DataFrame:
    """Use the production metric producer to make matched rows plus disjoint EWMA cases."""
    days = tuple(pd.date_range("2024-04-01", periods=7, freq="B").date)

    def metric_rows(
        *, method: str, seed: int | None, sample_prefix: str, scale_bias: float, shape_bias: float
    ) -> pd.DataFrame:
        sample_ids = [f"{sample_prefix}-{day.isoformat()}" for day in days]
        actual_remaining = np.asarray([100.0 + 11.0 * index for index in range(len(days))])
        scale = pd.DataFrame(
            {
                "sample_id": sample_ids,
                "instrument_id": ["asset-a"] * len(days),
                "session_date": list(days),
                "as_of": [24] * len(days),
                "symbol": pd.Categorical(["AAA"] * len(days)),
                "baseline_remaining_volume": actual_remaining * 0.9,
            }
        )
        sample_targets = np.tile(np.asarray([0.65, 0.35]), len(days))
        shape = pd.DataFrame(
            {
                "sample_id": np.repeat(sample_ids, 2),
                "case_id": np.repeat(sample_ids, 2),
                "target_bucket": np.tile(np.asarray([24, 25], dtype=np.int64), len(days)),
                "sample_weight": np.full(2 * len(days), 0.5),
            }
        )
        base = LightGBMFrames(scale, actual_remaining, shape, sample_targets)
        date_jitter = np.asarray([index % 3 for index in range(len(days))]) * 0.003
        totals = actual_remaining * (1.0 + scale_bias + date_jitter)
        predicted = shape.loc[:, ["case_id", "target_bucket"]].copy()
        predicted["conditional_share"] = np.tile(
            np.asarray([0.65 + shape_bias, 0.35 - shape_bias]), len(days)
        )
        result = forecast_metric_frame(
            base,
            totals,
            predicted,
            fold_id="fold-1",
            method=method,
            seed=seed,
        )
        result["seed"] = pd.array([seed] * len(result), dtype="Int64")
        return result

    parts = [
        metric_rows(
            method="raw", seed=None, sample_prefix="shared", scale_bias=0.14, shape_bias=-0.13
        ),
        metric_rows(
            method="untrained_neural",
            seed=None,
            sample_prefix="shared",
            scale_bias=0.12,
            shape_bias=-0.09,
        ),
    ]
    for seed in (13, 29, 47):
        parts.extend(
            (
                metric_rows(
                    method="dense",
                    seed=seed,
                    sample_prefix="shared",
                    scale_bias=0.08 + seed / 10_000,
                    shape_bias=-0.04,
                ),
                metric_rows(
                    method="sparse",
                    seed=seed,
                    sample_prefix="shared",
                    scale_bias=0.04 + seed / 10_000,
                    shape_bias=0.02,
                ),
            )
        )
    parts.append(
        metric_rows(
            method="ewma", seed=None, sample_prefix="ewma-only", scale_bias=0.10, shape_bias=0.04
        )
    )
    return pd.concat(parts, ignore_index=True)


def _publish_forecast_rows(config: SimpleNamespace, rows: pd.DataFrame) -> Path:
    root = evaluation_root(config) / "evaluation"
    sources: dict[str, tuple[Path, str]] = {}
    for (method, seed), group in rows.groupby(["method", "seed"], dropna=False, sort=True):
        normalized_seed = None if pd.isna(seed) else int(seed)
        key = f"{method}/{normalized_seed if normalized_seed is not None else 'shared'}"
        directory = root / "forecast-fixture-shards" / key
        receipt = publish_frames(
            directory,
            identity={
                **_execution_identity(config),
                "method": str(method),
                "seed": normalized_seed,
            },
            frames={"metrics.parquet": group.reset_index(drop=True)},
        )
        path = directory / "metrics.parquet"
        sources[key] = (path, receipt["files"]["metrics.parquet"]["sha256"])
    destination = root / "forecast-results.parquet"
    merge_result_shards(
        destination,
        sources=sources,
        keys=(
            "fold_id",
            "method",
            "seed",
            "instrument_id",
            "session_date",
            "as_of_token",
            "sample_id",
        ),
        identity=_execution_identity(config),
        schema_version="paper-forecast-evaluation-v1",
    )
    return destination


def _publish_tca_results(config: SimpleNamespace, name: str, rows: pd.DataFrame) -> Path:
    root = evaluation_root(config) / "tca"
    sources: dict[str, tuple[Path, str]] = {}
    for method, group in rows.groupby("method", sort=True):
        key = str(method)
        directory = root / "worker-result-shards" / name / key
        receipt = publish_frames(
            directory,
            identity={**_execution_identity(config), "method": key, "result": name},
            frames={"results.parquet": group.reset_index(drop=True)},
        )
        path = directory / "results.parquet"
        sources[key] = (path, receipt["files"]["results.parquet"]["sha256"])
    destination = root / f"{name}.parquet"
    merge_result_shards(
        destination,
        sources=sources,
        keys=("fold_id", "date", "instrument_id", "method", "order_fraction_adv20"),
        identity=_execution_identity(config),
        schema_version="paper-tca-merged-v3",
    )
    return destination


def _make_test_identity_receipts(config: SimpleNamespace) -> None:
    root = evaluation_root(config)
    write_json_atomic(root / "execution.json", {"identity": _execution_identity(config)})
    write_json_atomic(
        root / "tca/manifest.json",
        {
            "schema_version": "paper-tca-v1",
            "paper_config_hash": config.config_hash,
            "evaluation_identity": _execution_identity(config),
            "files": {
                name: {
                    "path": str(root / f"tca/{name}.parquet"),
                    "sha256": file_sha256(root / f"tca/{name}.parquet"),
                }
                for name in ("main", "sensitivity")
            },
        },
    )


def test_real_report_stage_publication_and_freeze_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    pytest.importorskip("torch")
    sequence_manifest = request.getfixturevalue("_fixture_regime_corpus")
    config, folds = _make_config(tmp_path)
    write_json_atomic(
        config.artifact_root / "selection/parameter-freeze-v1.json",
        {"status": "SYNTHETIC_FIXTURE_ONLY"},
    )
    write_json_atomic(
        config.artifact_root / "selection/locked-test-opened-v1.json",
        {"status": "SYNTHETIC_FIXTURE_ONLY"},
    )
    _write_three_fold_sequence_manifests(config, folds, sequence_manifest)

    monkeypatch.setattr(orchestration, "_git_head", lambda: _SOURCE_COMMIT)
    monkeypatch.setattr(orchestration, "_git_tree", lambda: _SOURCE_TREE)
    monkeypatch.setattr(orchestration, "_require_parameter_freeze", lambda *_args: {})
    monkeypatch.setattr(orchestration, "_require_locked_test_opened", lambda *_args: {})

    for geometry in ("dense", "sparse"):
        export_fixture_checkpoint(tmp_path, sequence_manifest, geometry)
    representation_outputs = orchestration.evaluate_representations_stage(
        config, full_run_cli_enabled=True, runtime_approval=None
    )
    accessibility_path = Path(representation_outputs["accessibility"])
    date_metrics_path = Path(representation_outputs["date_metrics"])
    support_path = Path(representation_outputs["support_regimes"])
    assert not pd.read_parquet(accessibility_path).empty
    assert not pd.read_parquet(date_metrics_path).empty
    assert not pd.read_parquet(support_path).empty

    # Report construction consumes all three sequence manifests and the full frozen matrix.
    config.evaluation["folds"] = folds
    config.representation["seeds"] = [13, 29, 47]
    _write_frozen_manifest_matrix(config, folds)
    forecast_input = _publish_forecast_rows(config, _forecast_metric_rows())

    from test_tca_production_contracts import build_published_tca_rows

    (tmp_path / "real-tca-worker").mkdir()
    tca_outputs = build_published_tca_rows(
        tmp_path / "real-tca-worker", include_representation_methods=True
    )
    tca_main = tca_outputs["main"]
    tca_sensitivity = tca_outputs["sensitivity"]
    expected_methods = {
        "ewma",
        "lightgbm_raw",
        "raw_untrained_neural",
        *{f"raw_dense_jepa_seed_{seed}" for seed in (13, 29, 47)},
        *{f"raw_sparse_jepa_seed_{seed}" for seed in (13, 29, 47)},
    }
    assert expected_methods.issubset(set(tca_main["method"].astype(str)))
    tca_main_path = _publish_tca_results(config, "main", tca_main)
    tca_sensitivity_path = _publish_tca_results(config, "sensitivity", tca_sensitivity)
    _make_test_identity_receipts(config)

    original_build = orchestration._build_report_stage
    monkeypatch.setattr(
        orchestration,
        "_build_report_stage",
        partial(original_build, historical_schema_fixture=True),
    )
    published = orchestration.report_stage(config, full_run_cli_enabled=True, runtime_approval=None)
    assert published["reuse"] == "created"
    report_root = evaluation_report_root(config) / config.paper_run_id
    completion_path = report_root / "completion.json"
    completion_bytes = completion_path.read_bytes()
    completion = read_json(completion_path)

    assert (report_root / "REPORT.md").is_file()
    assert (report_root / "provenance.json").is_file()
    provenance = read_json(report_root / "provenance.json")
    assert provenance["data_classification"] == "synthetic_fixture"
    assert provenance["network_acquisition"] == "NOT RUN"
    assert provenance["historical_training"] == "NOT RUN"

    tables_root = report_root / "tables"
    assert {path.stem for path in tables_root.glob("*.parquet")} == set(HISTORICAL_TABLE_NAMES)
    for table_name, required_columns in HISTORICAL_TABLE_SCHEMAS.items():
        frame = pd.read_parquet(tables_root / f"{table_name}.parquet")
        assert required_columns.issubset(frame.columns), table_name
    assert len(HISTORICAL_TABLE_SCHEMAS) == 11

    figures_root = report_root / "figures"
    assert {path.name for path in figures_root.glob("*.png")} == {
        f"{name}.png" for name in HISTORICAL_FIGURE_NAMES
    }
    assert len(HISTORICAL_FIGURE_NAMES) == 5

    appendix_root = report_root / "appendix"
    assert {
        "bootstrap-block-sensitivity.parquet",
        "support-regimes.parquet",
        "unavailable-comparisons.parquet",
    }.issubset({path.name for path in appendix_root.iterdir()})
    assert not pd.read_parquet(appendix_root / "bootstrap-block-sensitivity.parquet").empty

    report_forecast = pd.read_parquet(forecast_input)
    assert pd.isna(report_forecast.loc[report_forecast["method"] == "raw", "seed"]).all()
    assert set(
        report_forecast.loc[report_forecast["method"].isin(["dense", "sparse"]), "seed"]
    ) == {13, 29, 47}
    assert (
        report_forecast.loc[report_forecast["method"] == "ewma", "sample_id"]
        .str.startswith("ewma-only-")
        .all()
    )
    forecast_summary = pd.read_parquet(tables_root / "forecast_performance.parquet")
    forecast_by_asof = pd.read_parquet(tables_root / "forecast_by_asof.parquet")
    assert forecast_summary.empty and forecast_by_asof.empty
    confirmatory = pd.read_parquet(tables_root / "confirmatory_statistics.parquet")
    assert len(confirmatory) == 5
    assert (confirmatory["matched_cases"] > 0).all()

    report_inputs = orchestration._report_input_names(config)
    identity = completion["identity"]
    assert set(identity["input_sha256"]) == set(report_inputs)
    assert identity["source_commit"] == _SOURCE_COMMIT
    assert identity["source_tree"] == _SOURCE_TREE
    for relative, digest in identity["input_sha256"].items():
        assert file_sha256(evaluation_root(config) / relative) == digest
    for relative, digest in completion["files"].items():
        assert file_sha256(report_root / relative) == digest
    assert file_sha256(accessibility_path) == file_sha256(
        evaluation_root(config) / "evaluation/representation-accessibility.parquet"
    )
    assert file_sha256(date_metrics_path) == file_sha256(
        evaluation_root(config) / "evaluation/representation-date-metrics.parquet"
    )
    assert file_sha256(support_path) == file_sha256(
        evaluation_root(config) / "evaluation/support-regimes.parquet"
    )
    assert file_sha256(tca_main_path) == file_sha256(evaluation_root(config) / "tca/main.parquet")
    assert file_sha256(tca_sensitivity_path) == file_sha256(
        evaluation_root(config) / "tca/sensitivity.parquet"
    )

    # Report inputs reject a merge receipt bound to a different source tree.
    input_receipt_path = (evaluation_root(config) / "tca/main.parquet").with_suffix(
        ".manifest.json"
    )
    input_receipt_bytes = input_receipt_path.read_bytes()
    input_receipt = read_json(input_receipt_path)
    input_receipt["merge_identity"]["source_tree"] = "different-source-tree"
    write_json_atomic(input_receipt_path, input_receipt)
    with pytest.raises(ValueError, match="source identity mismatch"):
        orchestration.report_stage(config, full_run_cli_enabled=True, runtime_approval=None)
    with pytest.raises(ValueError, match="source identity mismatch"):
        orchestration.write_final_result_freeze(config)
    input_receipt_path.write_bytes(input_receipt_bytes)

    for key in ("paper_config_hash", "parameter_freeze_sha256"):
        input_receipt = read_json(input_receipt_path)
        input_receipt["merge_identity"][key] = "different-frozen-identity"
        write_json_atomic(input_receipt_path, input_receipt)
        with pytest.raises(ValueError, match="source identity mismatch"):
            orchestration.report_stage(config, full_run_cli_enabled=True, runtime_approval=None)
        with pytest.raises(ValueError, match="source identity mismatch"):
            orchestration.write_final_result_freeze(config)
        input_receipt_path.write_bytes(input_receipt_bytes)

    # A changed required numerical input is caught against completion's input hashes.
    required_input = evaluation_root(config) / "tca/sensitivity.parquet"
    input_bytes = required_input.read_bytes()
    required_input.write_bytes(input_bytes + b"corrupt")
    with pytest.raises(ValueError, match="numerical input checksum"):
        orchestration.write_final_result_freeze(config)
    required_input.write_bytes(input_bytes)

    # Both table and figure bytes are bound in completion.files.
    for output_path in (
        tables_root / "dataset_folds_exclusions.parquet",
        figures_root / f"{HISTORICAL_FIGURE_NAMES[0]}.png",
    ):
        output_bytes = output_path.read_bytes()
        output_path.write_bytes(output_bytes + b"corrupt")
        with pytest.raises(ValueError, match="inventory/checksum"):
            orchestration.write_final_result_freeze(config)
        output_path.write_bytes(output_bytes)

    # The completion inventory must contain every hashed file, and input hashes are exact.
    incomplete_inventory = deepcopy(completion)
    incomplete_inventory["files"].pop("REPORT.md")
    write_json_atomic(completion_path, incomplete_inventory)
    with pytest.raises(ValueError, match="inventory/checksum"):
        orchestration.write_final_result_freeze(config)
    completion_path.write_bytes(completion_bytes)

    incomplete_inputs = deepcopy(completion)
    incomplete_inputs["identity"]["input_sha256"].pop("tca/sensitivity.parquet")
    write_json_atomic(completion_path, incomplete_inputs)
    with pytest.raises(ValueError, match="numerical input inventory"):
        orchestration.write_final_result_freeze(config)
    completion_path.write_bytes(completion_bytes)

    aggregate_paths = (
        evaluation_root(config) / "evaluation/representation-evaluation-manifest.json",
        evaluation_root(config) / "tca/manifest.json",
    )
    for path in aggregate_paths:
        original = path.read_bytes()
        manifest = read_json(path)
        if "accessibility_sha256" in manifest:
            manifest["accessibility_sha256"] = "stale"
        else:
            manifest["files"]["main"]["sha256"] = "stale"
        write_json_atomic(path, manifest)
        with pytest.raises(ValueError, match="manifest output"):
            orchestration.write_final_result_freeze(config)
        path.write_bytes(original)
    tca_path = aggregate_paths[1]
    original = tca_path.read_bytes()
    for field in ("path", "sha256"):
        manifest = read_json(tca_path)
        manifest["files"]["main"][field] = "wrong"
        write_json_atomic(tca_path, manifest)
        with pytest.raises(ValueError, match="manifest output"):
            orchestration.write_final_result_freeze(config)
        tca_path.write_bytes(original)
    freeze = orchestration.write_final_result_freeze(config)
    assert freeze["status"] == "FINAL-RESULTS-FROZEN"
    assert orchestration.write_final_result_freeze(config)["sha256"] == freeze["sha256"]
    assert len(freeze["result_files"]) == len(completion["files"]) + 1
    for item in freeze["result_files"]:
        assert file_sha256(report_root / item["path"]) == item["sha256"]


def test_default_report_freeze_requires_completion_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = SimpleNamespace(
        artifact_root=tmp_path,
        runtime_evaluation_root=None,
        report_root=tmp_path / "reports",
        paper_run_id="default-path-fixture",
        config_hash="synthetic-default-path",
    )
    monkeypatch.setattr(orchestration, "_require_parameter_freeze", lambda *_args: {})
    monkeypatch.setattr(orchestration, "_require_locked_test_opened", lambda *_args: {})
    root = evaluation_root(config)
    report = evaluation_report_root(config) / config.paper_run_id
    write_json_atomic(tmp_path / "selection/parameter-freeze-v1.json", {})
    write_json_atomic(tmp_path / "selection/locked-test-opened-v1.json", {})
    write_json_atomic(root / "evaluation/representation-evaluation-manifest.json", {})
    write_json_atomic(root / "evaluation/forecast-results.manifest.json", {})
    write_json_atomic(root / "tca/manifest.json", {})
    write_json_atomic(report / "provenance.json", {"fixture": True})

    with pytest.raises(RuntimeError, match="report_completion"):
        orchestration.write_final_result_freeze(config)
