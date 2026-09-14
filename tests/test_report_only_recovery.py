"""Native A -> B -> C report-only recovery integration coverage.

The fixture uses the production report builder and TCA publication helpers.  A
retains forecast/representation, B reruns only TCA, and C inherits A's first
two stages plus B's completed TCA before publishing report and final-freeze.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import test_inherited_report_contracts as inherited_fixture
import test_report_stage_contracts as report_fixture
from test_representation_contracts import regime_corpus as _fixture_regime_corpus

from execsim.data.paper.manifests import file_sha256, read_json, write_json_atomic
from execsim.ml.paper import orchestration
from execsim.ml.paper.evaluation_artifacts import merge_result_shards, publish_frames
from execsim.ml.paper.evaluation_execution import (
    REPORT_INHERITED_SCHEMA,
    seal_evaluation_execution,
    write_evaluation_supersession_receipt,
)
from execsim.ml.paper.reports import (
    HISTORICAL_FIGURE_NAMES,
    HISTORICAL_TABLE_NAMES,
    HISTORICAL_TABLE_SCHEMAS,
)
from execsim.ml.paper.stage_inheritance import (
    stage_input_root,
    stage_provenance,
    stage_source,
    verify_stage_inheritance,
    write_stage_inheritance_receipt,
)

_SOURCE_A = inherited_fixture._SOURCE_A
_SOURCE_B = inherited_fixture._SOURCE_B
_SOURCE_C = {"commit": "synthetic-source-c", "tree": "synthetic-tree-c"}


def _clear_destination(config: SimpleNamespace) -> Path:
    destination = Path(config.runtime_evaluation_root)
    destination.mkdir(parents=True, exist_ok=True)
    for child in destination.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    return destination


def _build_source_a(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    sequence_manifest: Path,
) -> tuple[SimpleNamespace, list[dict[str, Any]], Path]:
    """Build A's real forecast/representation producer namespace."""
    config, folds = report_fixture._make_config(tmp_path)
    config.historical_schema_fixture = True
    config.runtime_evaluation_root = tmp_path / "evaluation-executions/source-a"
    inherited_fixture._write_selection_receipts(config)
    report_fixture._write_three_fold_sequence_manifests(config, folds, sequence_manifest)

    monkeypatch.setattr(orchestration, "_git_head", lambda: _SOURCE_A["commit"])
    monkeypatch.setattr(orchestration, "_git_tree", lambda: _SOURCE_A["tree"])
    monkeypatch.setattr(report_fixture, "_SOURCE_COMMIT", _SOURCE_A["commit"])
    monkeypatch.setattr(report_fixture, "_SOURCE_TREE", _SOURCE_A["tree"])
    monkeypatch.setattr(orchestration, "_require_parameter_freeze", lambda *_args: {})
    monkeypatch.setattr(orchestration, "_require_locked_test_opened", lambda *_args: {})
    for geometry in ("dense", "sparse"):
        report_fixture.export_fixture_checkpoint(tmp_path, sequence_manifest, geometry)
    orchestration.evaluate_representations_stage(
        config, full_run_cli_enabled=True, runtime_approval=None
    )

    # The report/freeze contract requires its complete configured matrices;
    # retain the real producer outputs and expand only the fixture inventory.
    config.evaluation["folds"] = folds
    config.representation["seeds"] = [13, 29, 47]
    inherited_fixture._materialize_frozen_representation_matrix(config, folds)
    lightgbm_records = inherited_fixture._materialize_frozen_lightgbm_matrix(config, folds)
    inherited_fixture._write_complete_selection_receipts(config, lightgbm_records)
    freeze_sha = file_sha256(config.artifact_root / "selection/parameter-freeze-v1.json")
    for geometry in ("dense", "sparse"):
        retained_manifest = (
            config.runtime_evaluation_root
            / "evaluation-v2"
            / "representations"
            / "fold-1"
            / geometry
            / "13"
            / "manifest.json"
        )
        retained_receipt = read_json(retained_manifest)
        retained_receipt["identity"]["parameter_freeze_sha256"] = freeze_sha
        write_json_atomic(retained_manifest, retained_receipt)

    from test_tca_production_contracts import build_published_tca_rows

    tca_fixture_root = tmp_path / "tca-production-fixture-a"
    tca_fixture_root.mkdir()
    build_published_tca_rows(tca_fixture_root, include_representation_methods=True)
    inherited_fixture._copy_forecast_inventory(
        config, config.runtime_evaluation_root, tca_fixture_root
    )
    inherited_fixture._copy_representation_inventory(
        config, config.runtime_evaluation_root, config.runtime_evaluation_root, folds
    )
    inherited_fixture._merge_forecast_inventory(config, config.runtime_evaluation_root)
    predecessor = inherited_fixture._write_predecessor_execution(
        config, config.runtime_evaluation_root
    )
    return config, folds, predecessor


def _build_source_b(
    config: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    predecessor: Path,
) -> tuple[Path, Path, dict[str, list[str]]]:
    """Seal B as a v4 TCA recovery and publish B's complete TCA outputs."""
    from test_tca_production_contracts import build_published_tca_rows

    config.runtime_evaluation_root = tmp_path / "evaluation-executions/source-b"
    monkeypatch.setattr(orchestration, "_git_head", lambda: _SOURCE_B["commit"])
    monkeypatch.setattr(orchestration, "_git_tree", lambda: _SOURCE_B["tree"])
    monkeypatch.setattr(report_fixture, "_SOURCE_COMMIT", _SOURCE_B["commit"])
    monkeypatch.setattr(report_fixture, "_SOURCE_TREE", _SOURCE_B["tree"])
    tca_fixture_root = tmp_path / "tca-production-fixture-b"
    tca_fixture_root.mkdir()
    tca_outputs = build_published_tca_rows(tca_fixture_root, include_representation_methods=True)

    inheritance = tmp_path / "stage-inheritance-b.json"
    write_stage_inheritance_receipt(
        config,
        superseded_execution=predecessor,
        output=inheritance,
        replacement_source_commit=_SOURCE_B["commit"],
        replacement_source_tree=_SOURCE_B["tree"],
        reason="synthetic TCA recovery",
    )
    supersession = tmp_path / "evaluation-supersession-b.json"
    write_evaluation_supersession_receipt(
        config,
        superseded_execution=predecessor,
        output=supersession,
        replacement_source_commit=_SOURCE_B["commit"],
        replacement_source_tree=_SOURCE_B["tree"],
        reason="synthetic TCA recovery",
    )
    _clear_destination(config)
    sealed = seal_evaluation_execution(
        config,
        source_commit=_SOURCE_B["commit"],
        source_tree=_SOURCE_B["tree"],
        supersession=supersession,
        inheritance=inheritance,
    )
    assert sealed["schema_version"] == "paper-evaluation-execution-v4"

    # The report-only receipt binds the native date-shard layout, not a
    # method-level test shortcut.  Re-publish the real worker output frames
    # under the production fold/date paths and merge those exact shards.  The
    # production TCA helper intentionally emits one fold; replicate its
    # native rows over every configured fixture fold so v3 can bind a complete
    # independent fold/date inventory without hand-constructing TCA values.
    source_date = str(pd.Timestamp(tca_outputs["main"]["date"].iloc[0]).date())
    fold_ids = tuple(str(fold["id"]) for fold in config.evaluation["folds"])
    expected_tca_dates = {fold_id: [source_date] for fold_id in fold_ids}
    tca_root = config.runtime_evaluation_root / "tca"
    main_sources: dict[str, tuple[Path, str]] = {}
    sensitivity_sources: dict[str, tuple[Path, str]] = {}
    for fold_id in fold_ids:
        session_date = expected_tca_dates[fold_id][0]
        shard_directory = (
            config.runtime_evaluation_root / "evaluation-v2" / "tca-shards" / fold_id / session_date
        )
        input_directory = (
            config.runtime_evaluation_root / "evaluation-v2" / "tca-inputs" / fold_id / session_date
        )
        publish_frames(
            input_directory,
            identity={
                "source_commit": _SOURCE_B["commit"],
                "source_tree": _SOURCE_B["tree"],
                "paper_config_hash": config.config_hash,
                "fold_id": fold_id,
                "session_date": session_date,
            },
            frames={"input.parquet": pd.DataFrame({"value": [1]})},
        )
        shard_identity = {
            "source_commit": _SOURCE_B["commit"],
            "source_tree": _SOURCE_B["tree"],
            "paper_config_hash": config.config_hash,
            "parameter_freeze_sha256": file_sha256(
                config.artifact_root / "selection/parameter-freeze-v1.json"
            ),
            "schema_version": "paper-tca-date-shard-v3",
            "fold_id": fold_id,
            "session_date": session_date,
        }
        shard_receipt = publish_frames(
            shard_directory,
            identity=shard_identity,
            frames={
                "main.parquet": tca_outputs["shard_main"].assign(
                    fold_id=fold_id, date=session_date
                ),
                "sensitivity.parquet": tca_outputs["shard_sensitivity"].assign(
                    fold_id=fold_id, date=session_date
                ),
            },
        )
        shard_relative_root = (
            config.runtime_evaluation_root.relative_to(config.artifact_root).as_posix()
            + f"/evaluation-v2/tca-shards/{fold_id}/{session_date}"
        )
        main_sources[f"{shard_relative_root}/main.parquet"] = (
            shard_directory / "main.parquet",
            shard_receipt["files"]["main.parquet"]["sha256"],
        )
        sensitivity_sources[f"{shard_relative_root}/sensitivity.parquet"] = (
            shard_directory / "sensitivity.parquet",
            shard_receipt["files"]["sensitivity.parquet"]["sha256"],
        )
    merge_result_shards(
        tca_root / "main.parquet",
        sources=main_sources,
        keys=("fold_id", "date", "instrument_id", "method", "order_fraction_adv20"),
        identity={
            "source_commit": _SOURCE_B["commit"],
            "source_tree": _SOURCE_B["tree"],
            "paper_config_hash": config.config_hash,
            "parameter_freeze_sha256": shard_identity["parameter_freeze_sha256"],
        },
        schema_version="paper-tca-merged-v3",
    )
    merge_result_shards(
        tca_root / "sensitivity.parquet",
        sources=sensitivity_sources,
        keys=("fold_id", "date", "instrument_id", "method", "order_fraction_adv20"),
        identity={
            "source_commit": _SOURCE_B["commit"],
            "source_tree": _SOURCE_B["tree"],
            "paper_config_hash": config.config_hash,
            "parameter_freeze_sha256": shard_identity["parameter_freeze_sha256"],
        },
        schema_version="paper-tca-merged-v3",
    )
    tca_manifest = inherited_fixture._write_tca_manifest(config)
    assert tca_manifest.is_file()
    return config.runtime_evaluation_root / "execution.json", tca_manifest, expected_tca_dates


def _build_source_c(
    config: SimpleNamespace,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    predecessor: Path,
    expected_tca_dates: dict[str, list[str]],
) -> tuple[Path, Path]:
    """Seal C as report-only v5, inheriting A forecast/representation and B TCA."""
    config.runtime_evaluation_root = tmp_path / "evaluation-executions/source-c"
    monkeypatch.setattr(orchestration, "_git_head", lambda: _SOURCE_C["commit"])
    monkeypatch.setattr(orchestration, "_git_tree", lambda: _SOURCE_C["tree"])
    inheritance = tmp_path / "stage-inheritance-c.json"
    write_stage_inheritance_receipt(
        config,
        superseded_execution=predecessor,
        output=inheritance,
        replacement_source_commit=_SOURCE_C["commit"],
        replacement_source_tree=_SOURCE_C["tree"],
        reason="synthetic report-only recovery",
        invalidation_frontier="report",
        expected_tca_dates=expected_tca_dates,
    )
    payload = verify_stage_inheritance(config, inheritance)
    assert payload["schema_version"] == "paper-evaluation-stage-inheritance-v3"
    assert payload["inherited_stages"] == [
        "evaluate-forecast",
        "evaluate-representation",
        "run-tca",
    ]
    assert payload["invalidation_frontier"] == "report"
    supersession = tmp_path / "evaluation-supersession-c.json"
    write_evaluation_supersession_receipt(
        config,
        superseded_execution=predecessor,
        output=supersession,
        replacement_source_commit=_SOURCE_C["commit"],
        replacement_source_tree=_SOURCE_C["tree"],
        reason="synthetic report-only recovery",
    )
    _clear_destination(config)
    sealed = seal_evaluation_execution(
        config,
        source_commit=_SOURCE_C["commit"],
        source_tree=_SOURCE_C["tree"],
        supersession=supersession,
        inheritance=inheritance,
    )
    assert sealed["schema_version"] == REPORT_INHERITED_SCHEMA
    assert sealed["inherited_stages"] == [
        "evaluate-forecast",
        "evaluate-representation",
        "run-tca",
    ]
    return config.runtime_evaluation_root / "execution.json", inheritance


@pytest.fixture(scope="module")
def report_only_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> Iterator[dict[str, Any]]:
    pytest.importorskip("torch")
    tmp_path = tmp_path_factory.mktemp("report-only-recovery")
    sequence_manifest = _fixture_regime_corpus.__wrapped__(tmp_path)
    monkeypatch = pytest.MonkeyPatch()
    try:
        config, folds, predecessor_a = _build_source_a(tmp_path, monkeypatch, sequence_manifest)
        execution_b, tca_manifest, expected_tca_dates = _build_source_b(
            config, tmp_path, monkeypatch, predecessor_a
        )
        execution_c, inheritance_c = _build_source_c(
            config, tmp_path, monkeypatch, execution_b, expected_tca_dates
        )
        yield {
            "config": config,
            "folds": folds,
            "source_a": predecessor_a.parent,
            "source_b": execution_b.parent,
            "source_c": execution_c.parent,
            "inheritance": inheritance_c,
            "tca_manifest": tca_manifest,
            "expected_tca_dates": expected_tca_dates,
        }
    finally:
        monkeypatch.undo()


def _run_report_and_freeze(run: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    config = run["config"]
    published = orchestration.report_stage(
        config,
        full_run_cli_enabled=True,
        runtime_approval=None,
        historical_schema_fixture=True,
    )
    freeze = orchestration.write_final_result_freeze(config)
    return published, freeze


def test_report_only_chain_uses_native_mixed_sources_and_stable_freeze(report_only_run) -> None:
    run = report_only_run
    config = run["config"]
    source_a, source_b, source_c = run["source_a"], run["source_b"], run["source_c"]
    published, freeze = _run_report_and_freeze(run)
    report_root = source_c / "reports" / config.paper_run_id
    assert not (source_c / "evaluation-v2").exists()
    assert not (source_c / "tca").exists()
    completion_path = report_root / "completion.json"
    completion_bytes = completion_path.read_bytes()
    completion = read_json(completion_path)
    assert published["reuse"] == "created"
    assert report_root.is_dir()
    assert (report_root / "REPORT.md").is_file()
    assert {path.stem for path in (report_root / "tables").glob("*.parquet")} == set(
        HISTORICAL_TABLE_NAMES
    )
    for table_name, columns in HISTORICAL_TABLE_SCHEMAS.items():
        assert columns.issubset(
            pd.read_parquet(report_root / "tables" / f"{table_name}.parquet").columns
        )
        assert (report_root / "tables" / f"{table_name}.md").is_file()
        assert (report_root / "tables" / f"{table_name}.tex").is_file()
    assert {path.name for path in (report_root / "figures").glob("*.png")} == {
        f"{name}.png" for name in HISTORICAL_FIGURE_NAMES
    }

    expected_roots = {
        "evaluation/forecast-results.parquet": source_a,
        "evaluation/representation-accessibility.parquet": source_a,
        "evaluation/representation-date-metrics.parquet": source_a,
        "evaluation/support-regimes.parquet": source_a,
        "tca/main.parquet": source_b,
        "tca/sensitivity.parquet": source_b,
    }
    assert set(completion["identity"]["input_sha256"]) == set(
        orchestration._report_input_names(config)
    )
    for name, digest in completion["identity"]["input_sha256"].items():
        path = orchestration._report_input_path(config, name)
        assert path.is_relative_to(expected_roots[name])
        assert file_sha256(path) == digest

    provenance = read_json(report_root / "provenance.json")
    stage_sources = provenance["stage_sources"]
    for stage in ("evaluate-forecast", "evaluate-representation"):
        assert stage_sources[stage]["inherited"] is True
        assert stage_sources[stage]["commit"] == _SOURCE_A["commit"]
    assert stage_sources["run-tca"]["inherited"] is True
    assert stage_sources["run-tca"]["commit"] == _SOURCE_B["commit"]
    for stage in ("report", "final-result-freeze"):
        assert stage_sources[stage]["inherited"] is False
        assert stage_sources[stage]["commit"] == _SOURCE_C["commit"]
    assert provenance["stage_inheritance_receipt_sha256"] == file_sha256(run["inheritance"])
    assert completion["identity"]["stage_sources"] == stage_sources
    assert freeze["stage_sources"] == stage_sources
    assert read_json(run["inheritance"])["expected_tca_dates"] == run["expected_tca_dates"]
    assert read_json(source_c / "execution.json")["invalidation_frontier"] == "report"
    assert freeze["stage_inheritance_receipt_sha256"] == file_sha256(run["inheritance"])
    assert stage_input_root(config, "forecast") == source_a.resolve()
    assert stage_input_root(config, "representation") == source_a.resolve()
    assert stage_input_root(config, "tca") == source_b.resolve()
    native_provenance = stage_provenance(
        config,
        source_commit=_SOURCE_C["commit"],
        source_tree=_SOURCE_C["tree"],
    )
    assert (
        stage_source(
            config,
            "tca",
            source_commit=_SOURCE_C["commit"],
            source_tree=_SOURCE_C["tree"],
        )
        == _SOURCE_B
    )
    assert native_provenance["stage_sources"] == stage_sources
    assert native_provenance["stage_inheritance_receipt_sha256"] == file_sha256(run["inheritance"])

    # Re-entry verifies the entire native bundle and freeze bytes without
    # rebuilding or adding files to the immutable report tree.
    second, repeated_freeze = _run_report_and_freeze(run)
    assert second["reuse"] == "validated"
    assert completion_path.read_bytes() == completion_bytes
    assert freeze["sha256"] == repeated_freeze["sha256"]


@pytest.mark.parametrize(
    "fault",
    [
        "inherited-forecast",
        "inherited-representation",
        "tca-shard",
        "merged-receipt",
        "merged-output",
        "tca-aggregate",
        "inheritance",
        "predecessor",
        "source",
    ],
)
def test_report_only_chain_rejects_corrupt_inherited_inputs(report_only_run, fault: str) -> None:
    run = report_only_run
    config = run["config"]
    source_a, source_b = run["source_a"], run["source_b"]
    path: Path
    original: bytes
    if fault == "inherited-forecast":
        path = source_a / "evaluation/forecast-results.parquet"
        original = path.read_bytes()
        path.write_bytes(original + b"corrupt inherited stage")
    elif fault == "inherited-representation":
        path = source_a / "evaluation/representation-accessibility.parquet"
        original = path.read_bytes()
        path.write_bytes(original + b"corrupt inherited representation")
    elif fault == "tca-shard":
        path = next((source_b / "evaluation-v2/tca-shards").rglob("main.parquet"))
        original = path.read_bytes()
        path.write_bytes(original + b"corrupt tca shard")
    elif fault == "merged-receipt":
        path = source_b / "tca/main.manifest.json"
        original = path.read_bytes()
        payload = read_json(path)
        payload["parquet_sha256"] = "corrupt-merged-receipt"
        write_json_atomic(path, payload)
    elif fault == "merged-output":
        path = source_b / "tca/main.parquet"
        original = path.read_bytes()
        path.write_bytes(original + b"corrupt merged output")
    elif fault == "tca-aggregate":
        path = run["tca_manifest"]
        original = path.read_bytes()
        payload = read_json(path)
        payload["files"]["main"]["sha256"] = "corrupt-tca-aggregate"
        write_json_atomic(path, payload)
    elif fault == "inheritance":
        path = run["inheritance"]
        original = path.read_bytes()
        payload = read_json(path)
        payload["stage_inventory"]["run-tca"]["files"] = {}
        write_json_atomic(path, payload)
    elif fault == "predecessor":
        path = source_b / "execution.json"
        original = path.read_bytes()
        payload = read_json(path)
        payload["source_tree"] = "corrupt-predecessor-tree"
        write_json_atomic(path, payload)
    else:
        path = run["inheritance"]
        original = path.read_bytes()
        payload = read_json(path)
        payload["replacement_evaluation_source"]["commit"] = "foreign-source"
        write_json_atomic(path, payload)

    try:
        with pytest.raises(
            (ValueError, RuntimeError),
            match=r"checksum|identity|inventory|source|mismatch|manifest|inherited|TCA",
        ):
            orchestration.report_stage(
                config,
                full_run_cli_enabled=True,
                runtime_approval=None,
                historical_schema_fixture=True,
            )
    finally:
        path.write_bytes(original)
