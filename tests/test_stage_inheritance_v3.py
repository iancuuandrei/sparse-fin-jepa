"""Focused proof for report-only (v3) stage inheritance."""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from execsim.data.paper.manifests import file_sha256, read_json, write_json_atomic
from execsim.ml.paper.evaluation_artifacts import merge_result_shards, publish_frames
from execsim.ml.paper.evaluation_execution import (
    seal_evaluation_execution,
    write_evaluation_supersession_receipt,
)
from execsim.ml.paper.stage_inheritance import (
    SCHEMA_V3,
    stage_input_root,
    stage_provenance,
    stage_source,
    verify_stage_inheritance,
    write_stage_inheritance_receipt,
)


def _load_fixture_module() -> ModuleType:
    path = Path(__file__).with_name("test_stage_inheritance.py")
    spec = importlib.util.spec_from_file_location("stage_inheritance_v3_fixture", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load inheritance fixture at {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_D0_SOURCE = {"commit": "d0-commit", "tree": "d0-tree"}
_D1_SOURCE = {"commit": "d1-commit", "tree": "d1-tree"}


def _clear(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _publish_tca(config, d0: Path) -> None:
    """Publish two tiny production-shaped TCA shards and both merge receipts."""
    identity = {
        "source_commit": _D0_SOURCE["commit"],
        "source_tree": _D0_SOURCE["tree"],
        "paper_config_hash": config.config_hash,
        "parameter_freeze_sha256": file_sha256(
            config.artifact_root / "selection/parameter-freeze-v1.json"
        ),
    }
    shard_sources: dict[str, dict[str, tuple[Path, str]]] = {"main": {}, "sensitivity": {}}
    for fold, day in (("fold-1", "2024-01-02"), ("fold-1", "2024-01-03")):
        input_dir = d0 / "evaluation-v2" / "tca-inputs" / fold / day
        publish_frames(
            input_dir,
            identity={**identity, "fold_id": fold, "session_date": day},
            frames={"input.parquet": pd.DataFrame({"value": [1]})},
        )
        shard_dir = d0 / "evaluation-v2" / "tca-shards" / fold / day
        receipt = publish_frames(
            shard_dir,
            identity={
                **identity,
                "schema_version": "paper-tca-date-shard-v3",
                "fold_id": fold,
                "session_date": day,
                "profile_corpus_manifest_sha256": "p" * 64,
                "training_cutoff": "2023-12-29",
                "sequence_hash": "s" * 64,
                "tca_config": {"fixture": True},
                "ledgers": [],
                "ewma_ledgers": {},
            },
            frames={
                "main.parquet": pd.DataFrame({"fold_id": [fold], "date": [day], "value": [1]}),
                "sensitivity.parquet": pd.DataFrame(
                    {"fold_id": [fold], "date": [day], "value": [2]}
                ),
            },
        )
        for name in ("main.parquet", "sensitivity.parquet"):
            path = shard_dir / name
            shard_sources[name.removesuffix(".parquet")][
                path.relative_to(config.artifact_root).as_posix()
            ] = (path, receipt["files"][name]["sha256"])

    for name in ("main", "sensitivity"):
        merge_result_shards(
            d0 / "tca" / f"{name}.parquet",
            sources=shard_sources[name],
            keys=("fold_id", "date", "value"),
            identity=identity,
            schema_version="paper-tca-merged-v3",
        )
    write_json_atomic(
        d0 / "tca" / "manifest.json",
        {
            "schema_version": "paper-tca-v1",
            "paper_config_hash": config.config_hash,
            "evaluation_identity": identity,
            "files": {
                name: {
                    "path": str(d0 / "tca" / f"{name}.parquet"),
                    "sha256": file_sha256(d0 / "tca" / f"{name}.parquet"),
                }
                for name in ("main", "sensitivity")
            },
        },
    )


@pytest.fixture
def report_only_run(tmp_path: Path):
    """Create an original -> v4 d0 -> report-only v3 chain."""
    fixture = _load_fixture_module()
    config, old, _old_receipt, _opened_sha = fixture.inherited_run.__wrapped__(tmp_path)
    d0 = config.runtime_evaluation_root
    supersession = config.artifact_root / "d0-supersession.json"
    v1_receipt = config.artifact_root / "d0-stage-inheritance.json"
    write_stage_inheritance_receipt(
        config,
        superseded_execution=old,
        output=v1_receipt,
        replacement_source_commit=_D0_SOURCE["commit"],
        replacement_source_tree=_D0_SOURCE["tree"],
        reason="complete TCA before report-only recovery",
    )
    write_evaluation_supersession_receipt(
        config,
        superseded_execution=old,
        output=supersession,
        replacement_source_commit=_D0_SOURCE["commit"],
        replacement_source_tree=_D0_SOURCE["tree"],
        reason="complete TCA before report-only recovery",
    )
    _clear(d0)
    seal_evaluation_execution(
        config,
        source_commit=_D0_SOURCE["commit"],
        source_tree=_D0_SOURCE["tree"],
        supersession=supersession,
        inheritance=v1_receipt,
    )
    _publish_tca(config, d0)
    replacement = config.artifact_root / "evaluation-executions" / "d1"
    config.runtime_evaluation_root = replacement
    inheritance = config.artifact_root / "report-only-inheritance.json"
    receipt = write_stage_inheritance_receipt(
        config,
        superseded_execution=d0,
        output=inheritance,
        replacement_source_commit=_D1_SOURCE["commit"],
        replacement_source_tree=_D1_SOURCE["tree"],
        reason="report serialization recovery",
        invalidation_frontier="report",
        expected_tca_dates={"fold-1": ["2024-01-02", "2024-01-03"]},
    )
    d0_sha = file_sha256(d0 / "execution.json")
    replacement.mkdir(parents=True)
    write_json_atomic(
        replacement / "execution.json",
        {
            "schema_version": "paper-evaluation-execution-v5",
            "status": "EVALUATION_RESEALED",
            "protocol_id": "sparse-jepa-v2",
            "paper_config_hash": config.config_hash,
            "parameter_freeze_sha256": file_sha256(
                config.artifact_root / "selection/parameter-freeze-v1.json"
            ),
            "root_evaluation_source": {"commit": "root", "tree": "root-tree"},
            "root_evaluation_open_receipt_sha256": file_sha256(
                config.artifact_root / "selection/locked-test-opened-v1.json"
            ),
            "superseded_execution_namespace": "evaluation-executions/new",
            "superseded_execution_receipt_sha256": d0_sha,
            "stage_inheritance_path": inheritance.relative_to(config.artifact_root).as_posix(),
            "stage_inheritance_sha256": receipt["sha256"],
            "stage_inheritance_receipt_sha256": receipt["sha256"],
            "inherited_stages": [
                "evaluate-forecast",
                "evaluate-representation",
                "run-tca",
            ],
            "invalidation_frontier": "report",
            "evaluation_source": _D1_SOURCE,
        },
    )
    return config, old, d0, inheritance


def test_v3_resolves_each_original_stage_and_records_immediate_predecessor(report_only_run):
    config, old, d0, inheritance = report_only_run
    payload = verify_stage_inheritance(config, inheritance)
    assert payload["schema_version"] == SCHEMA_V3
    assert payload["inherited_stages"] == [
        "evaluate-forecast",
        "evaluate-representation",
        "run-tca",
    ]
    assert payload["invalidation_frontier"] == "report"
    assert payload["superseded_execution_namespace"] == "evaluation-executions/new"
    assert payload["stage_inventory"]["evaluate-forecast"]["producer_execution_namespace"] == (
        "evaluation-executions/old"
    )
    assert (
        payload["stage_inventory"]["evaluate-representation"]["producer_execution_namespace"]
        == "evaluation-executions/old"
    )
    assert payload["stage_inventory"]["run-tca"]["producer_execution_namespace"] == (
        "evaluation-executions/new"
    )
    assert stage_input_root(config, "forecast") == old.resolve()
    assert stage_input_root(config, "representation") == old.resolve()
    assert stage_input_root(config, "tca") == d0.resolve()
    assert (
        stage_source(
            config, "tca", source_commit=_D1_SOURCE["commit"], source_tree=_D1_SOURCE["tree"]
        )
        == _D0_SOURCE
    )
    provenance = stage_provenance(
        config, source_commit=_D1_SOURCE["commit"], source_tree=_D1_SOURCE["tree"]
    )
    assert provenance["stage_sources"]["run-tca"]["inherited"] is True
    assert provenance["stage_sources"]["run-tca"]["commit"] == _D0_SOURCE["commit"]
    assert provenance["stage_sources"]["report"]["inherited"] is False


@pytest.mark.parametrize(
    "relative",
    [
        "tca/main.parquet",
        "tca/main.manifest.json",
        "evaluation-v2/tca-shards/fold-1/2024-01-02/main.parquet",
        "evaluation-v2/tca-shards/fold-1/2024-01-03/manifest.json",
        "tca/manifest.json",
    ],
)
def test_v3_rejects_tca_mutation_or_removal(report_only_run, relative: str):
    config, _old, d0, inheritance = report_only_run
    path = d0 / relative
    original = path.read_bytes()
    path.write_bytes(original + b"changed")
    with pytest.raises(ValueError, match=r"TCA|checksum|inventory|mismatch|changed|Extra data"):
        verify_stage_inheritance(config, inheritance)


def test_v3_rejects_same_count_but_wrong_expected_tca_date(report_only_run):
    config, _old, _d0, inheritance = report_only_run
    payload = read_json(inheritance)
    payload["expected_tca_dates"]["fold-1"] = ["2024-01-02", "2024-01-04"]
    write_json_atomic(inheritance, payload)
    with pytest.raises(ValueError, match=r"TCA|date|inventory|mismatch"):
        verify_stage_inheritance(config, inheritance)


def test_v4_resolver_rejects_report_only_inheritance(report_only_run):
    config, _old, _d0, _inheritance = report_only_run
    execution_path = config.runtime_evaluation_root / "execution.json"
    execution = read_json(execution_path)
    execution["schema_version"] = "paper-evaluation-execution-v4"
    write_json_atomic(execution_path, execution)
    with pytest.raises(ValueError, match="v4 cannot resolve"):
        stage_input_root(config, "tca")


def test_v3_rejects_consistent_merge_and_shard_date_removal(report_only_run):
    config, _old, d0, inheritance = report_only_run
    shard = d0 / "evaluation-v2/tca-shards/fold-1/2024-01-03"
    for name in ("main", "sensitivity"):
        receipt_path = d0 / f"tca/{name}.manifest.json"
        receipt = read_json(receipt_path)
        sources = receipt["merge_identity"]["sources"]
        sources = {path: digest for path, digest in sources.items() if "/2024-01-03/" not in path}
        receipt["merge_identity"]["sources"] = sources
        write_json_atomic(receipt_path, receipt)
    shutil.rmtree(shard)
    with pytest.raises(ValueError, match=r"TCA|date|inventory|mismatch"):
        verify_stage_inheritance(config, inheritance)
