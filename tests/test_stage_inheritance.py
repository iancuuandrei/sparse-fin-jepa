"""Focused proof for the typed forecast/representation inheritance boundary."""

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest

from execsim.data.paper.manifests import file_sha256, read_json, write_json_atomic
from execsim.ml.paper.evaluation_artifacts import merge_result_shards, publish_frames
from execsim.ml.paper.evaluation_workers import compact_profile_corpus, instrument_key
from execsim.ml.paper.stage_inheritance import (
    stage_input_root,
    stage_provenance,
    stage_source,
    verify_stage_inheritance,
    write_stage_inheritance_receipt,
)


def _json(path, payload):
    write_json_atomic(path, payload)
    return file_sha256(path)


@pytest.fixture
def inherited_run(tmp_path):
    root = tmp_path / "artifacts"
    old = root / "evaluation-executions" / "old"
    old.mkdir(parents=True)
    source = {
        "source_commit": "old-commit",
        "source_tree": "old-tree",
        "paper_config_hash": "fixture",
        "parameter_freeze_sha256": "pending",
    }
    sequence = root / "sequences/fold-1/sequence-manifest.json"
    sequence_hash = _json(sequence, {"universe_manifest_hash": "u" * 64})
    normalization_hash = "n" * 64
    lightgbm_manifests = []
    for method, seed in (("raw", None), ("untrained_neural", None), ("dense", 13), ("sparse", 13)):
        folder = root / "lightgbm/fold-1" / method / str(seed or "shared")
        model_records = []
        for name in ("scale.txt", "shape.txt"):
            path = folder / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(name)
            model_records.append({"path": name, "sha256": file_sha256(path)})
        grid_hash = _json(folder / "grid-results.json", {})
        embedding_hashes = {"train": None, "validation": None}
        if seed is not None:
            representation = root / "representations/fold-1" / method / str(seed)
            weights = representation / "final/model.safetensors"
            weights.parent.mkdir(parents=True, exist_ok=True)
            weights.write_text("fixture weights")
            weights_hash = file_sha256(weights)
            checkpoint_hash = _json(
                representation / "final/manifest.json",
                {
                    "fold_id": "fold-1",
                    "geometry": method,
                    "seed": seed,
                    "paper_config_hash": "fixture",
                    "calibrated_rdm_lambda": 10.0,
                    "code_commit": "old-commit",
                    "sequence_manifest_hash": sequence_hash,
                    "universe_manifest_hash": "u" * 64,
                    "normalization_hash": normalization_hash,
                    "weights_sha256": weights_hash,
                },
            )
            _json(representation / "compatibility.json", {"sequence_manifest_hash": sequence_hash})
            embedding = root / "embeddings/fold-1" / method / str(seed)
            files = []
            for partition in ("train", "validation", "test"):
                path = embedding / f"partition={partition}" / "embeddings.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("fixture embedding")
                digest = file_sha256(path)
                files.append(
                    {
                        "partition": partition,
                        "path": f"partition={partition}/embeddings.parquet",
                        "sha256": digest,
                    }
                )
                if partition in embedding_hashes:
                    embedding_hashes[partition] = digest
            _json(
                embedding / "manifest.json",
                {
                    "fold_id": "fold-1",
                    "seed": seed,
                    "geometry": method,
                    "adaptation": "none",
                    "checkpoint_hash": weights_hash,
                    "checkpoint_manifest_hash": checkpoint_hash,
                    "sequence_manifest_hash": sequence_hash,
                    "normalization_hash": normalization_hash,
                    "paper_config_hash": "fixture",
                    "files": files,
                },
            )
        manifest = {
            "paper_config_hash": "fixture",
            "method": method,
            "seed": seed,
            "sequence_manifest_hash": sequence_hash,
            "embedding_sha256": embedding_hashes,
            "models": model_records,
            "grid_results_sha256": grid_hash,
        }
        manifest_path = folder / "manifest.json"
        lightgbm_manifests.append(
            {
                "path": manifest_path.relative_to(root).as_posix(),
                "sha256": _json(manifest_path, manifest),
            }
        )
    _json(root / "selection/rdm-lambda.json", {})
    _json(root / "lightgbm/execution-receipt.json", {})
    freeze = root / "selection/parameter-freeze-v1.json"
    freeze_sha = _json(
        freeze,
        {
            "paper_config_hash": "fixture",
            "status": "PARAMETERS_FROZEN",
            "selected_rdm_lambda": 10.0,
            "test_or_tca_used": False,
            "git_commit": "old",
            "git_tree": "old-tree",
            "representation_source_commit": "old-commit",
            "lightgbm_manifests": lightgbm_manifests,
            "rdm_lambda_receipt_sha256": file_sha256(root / "selection/rdm-lambda.json"),
            "lightgbm_execution_receipt_sha256": file_sha256(
                root / "lightgbm/execution-receipt.json"
            ),
        },
    )
    source["parameter_freeze_sha256"] = freeze_sha
    ready_sha = _json(
        root / "selection/locked-test-ready-v1.json",
        {"paper_config_hash": "fixture", "parameter_freeze_sha256": freeze_sha},
    )
    opened_sha = _json(
        root / "selection" / "locked-test-opened-v1.json",
        {
            "paper_config_hash": "fixture",
            "status": "LOCKED-TEST-OPENED",
            "locked_test_ready_receipt_sha256": ready_sha,
            "evaluation_git_commit": "root",
            "evaluation_git_tree": "root-tree",
        },
    )
    _json(
        old / "execution.json",
        {
            "schema_version": "paper-evaluation-execution-v2",
            "status": "EVALUATION_RESEALED",
            "protocol_id": "sparse-jepa-v2",
            **source,
            "evaluation_source": {"commit": "old-commit", "tree": "old-tree"},
            "previous_evaluation_source": {"commit": "root", "tree": "root-tree"},
            "upstream_files": {"selection/parameter-freeze-v1.json": freeze_sha},
        },
    )
    raw = tmp_path / "raw.parquet"
    pd.DataFrame(
        {
            "instrument_id": ["A"],
            "symbol": ["A"],
            "timestamp": pd.to_datetime(["2022-01-01"], utc=True),
            "volume": [1.0],
        }
    ).to_parquet(raw, index=False)
    execution_identity = {
        "source_commit": "old-commit",
        "source_tree": "old-tree",
        "paper_config_hash": "fixture",
        "parameter_freeze_sha256": freeze_sha,
    }
    compact_profile_corpus(
        raw, old / "evaluation-v2" / "profile-corpus", identity=execution_identity
    )
    identity = {
        "source_commit": "old-commit",
        "source_tree": "old-tree",
        "paper_config_hash": "fixture",
        "parameter_freeze_sha256": freeze_sha,
    }
    artifacts = [
        (
            ("scale-base.parquet", "shape-base.parquet"),
            old / "evaluation-v2" / "bases" / "fold-1",
            {},
        ),
        (
            ("scale.parquet", "shape.parquet", "metrics.parquet"),
            old / "evaluation-v2" / "forecasts" / "fold-1" / "raw" / "shared",
            {"fold_id": "fold-1", "method": "raw", "seed": None},
        ),
        (
            ("scale.parquet", "shape.parquet", "metrics.parquet"),
            old / "evaluation-v2" / "forecasts" / "fold-1" / "untrained_neural" / "shared",
            {"fold_id": "fold-1", "method": "untrained_neural", "seed": None},
        ),
        (
            ("scale.parquet", "shape.parquet", "metrics.parquet"),
            old / "evaluation-v2" / "forecasts" / "fold-1" / "dense" / "13",
            {"fold_id": "fold-1", "method": "dense", "seed": 13},
        ),
        (
            ("scale.parquet", "shape.parquet", "metrics.parquet"),
            old / "evaluation-v2" / "forecasts" / "fold-1" / "sparse" / "13",
            {"fold_id": "fold-1", "method": "sparse", "seed": 13},
        ),
        (
            (
                "scale.parquet",
                "shape.parquet",
                "metrics.parquet",
                "minute-forecasts.parquet",
                "unavailable.parquet",
            ),
            old / "evaluation-v2" / "forecasts" / "fold-1" / "ewma" / instrument_key("A"),
            {"fold_id": "fold-1", "instrument_id": "A", "schema_version": "paper-ewma-ledger-v4"},
        ),
    ]
    for index, (names, directory, extra) in enumerate(artifacts):
        artifact_identity = {
            **identity,
            "schema_version": "paper-evaluation-base-v2"
            if "base" in names[0]
            else "paper-forecast-ledger-v2",
            **extra,
        }
        if names[0] == "scale-base.parquet":
            artifact_identity.update(
                {"sequence_manifest_sha256": sequence_hash, "partition": "test"}
            )
        if "base" not in names[0] and "minute-forecasts.parquet" not in names:
            artifact_identity.update(
                {
                    "model_manifest_sha256": file_sha256(
                        root
                        / "lightgbm/fold-1"
                        / extra["method"]
                        / str(extra["seed"] or "shared")
                        / "manifest.json"
                    ),
                    "base_manifest_sha256": file_sha256(
                        old / "evaluation-v2/bases/fold-1/manifest.json"
                    ),
                    "embedding_sha256": (
                        file_sha256(
                            root
                            / "embeddings/fold-1"
                            / extra["method"]
                            / str(extra["seed"])
                            / "partition=test/embeddings.parquet"
                        )
                        if extra["seed"] is not None
                        else None
                    ),
                }
            )
        if "minute-forecasts.parquet" in names:
            artifact_identity["schema_version"] = "paper-ewma-ledger-v4"
            artifact_identity.update(
                {
                    "base_manifest_sha256": file_sha256(
                        old / "evaluation-v2/bases/fold-1/manifest.json"
                    ),
                    "market_sha256": file_sha256(
                        old / "evaluation-v2/profile-corpus" / f"{instrument_key('A')}.parquet"
                    ),
                }
            )
        frames = {name: pd.DataFrame({"value": [index]}) for name in names}
        if names[0] == "scale-base.parquet":
            frames[names[0]] = pd.DataFrame({"instrument_id": ["A"], "value": [index]})
        publish_frames(directory, identity=artifact_identity, frames=frames)
    fold = old / "evaluation-v2" / "forecasts" / "fold-1"
    forecast_paths = [
        fold / method / seed / "metrics.parquet"
        for method, seed in (
            ("raw", "shared"),
            ("untrained_neural", "shared"),
            ("dense", "13"),
            ("sparse", "13"),
            ("ewma", instrument_key("A")),
        )
    ]
    forecast_sources = {
        path.relative_to(root).as_posix(): (path, file_sha256(path)) for path in forecast_paths
    }
    merge_result_shards(
        old / "evaluation" / "forecast-results.parquet",
        sources=forecast_sources,
        keys=["value"],
        identity=identity,
        schema_version="paper-forecast-evaluation-v1",
    )
    for index, geometry in enumerate(("dense", "sparse"), start=1):
        coordinate = old / "evaluation-v2" / "representations" / f"fold-1/{geometry}/13"
        checkpoint = root / "representations/fold-1" / geometry / "13" / "final/manifest.json"
        compatibility = root / "representations/fold-1" / geometry / "13/compatibility.json"
        embedding_manifest = root / "embeddings/fold-1" / geometry / "13/manifest.json"
        coord_identity = {
            **identity,
            "schema_version": "paper-representation-coordinate-v1",
            "coordinate": f"fold-1/{geometry}/13",
            "sequence_sha256": sequence_hash,
            "checkpoint_sha256": file_sha256(checkpoint),
            "compatibility_sha256": file_sha256(compatibility),
            "embedding_manifest_sha256": file_sha256(embedding_manifest),
        }
        publish_frames(
            coordinate,
            identity=coord_identity,
            frames={
                "accessibility.parquet": pd.DataFrame({"value": [index]}),
                "date-metrics.parquet": pd.DataFrame({"value": [index]}),
                **(
                    {"support.parquet": pd.DataFrame({"value": [index]})}
                    if geometry == "sparse"
                    else {}
                ),
            },
        )
    for name in (
        "representation-accessibility.parquet",
        "representation-date-metrics.parquet",
        "support-regimes.parquet",
    ):
        source_name = (
            "support.parquet"
            if name == "support-regimes.parquet"
            else "accessibility.parquet"
            if name == "representation-accessibility.parquet"
            else "date-metrics.parquet"
        )
        geometries = ("sparse",) if name == "support-regimes.parquet" else ("dense", "sparse")
        sources = {
            f"fold-1/{geometry}/13": (
                old / "evaluation-v2" / "representations" / f"fold-1/{geometry}/13/{source_name}",
                file_sha256(
                    old
                    / "evaluation-v2"
                    / "representations"
                    / f"fold-1/{geometry}/13/{source_name}"
                ),
            )
            for geometry in geometries
        }
        merge_result_shards(
            old / "evaluation" / name,
            sources=sources,
            keys=["value"],
            identity=identity,
            schema_version="paper-representation-result-v1",
        )
    _json(
        old / "evaluation" / "representation-evaluation-manifest.json",
        {
            "schema_version": "paper-representation-evaluation-v2",
            "paper_config_hash": "fixture",
            "accessibility_sha256": file_sha256(
                old / "evaluation/representation-accessibility.parquet"
            ),
            "date_metrics_sha256": file_sha256(
                old / "evaluation/representation-date-metrics.parquet"
            ),
            "support_regimes_sha256": file_sha256(old / "evaluation/support-regimes.parquet"),
        },
    )
    config = SimpleNamespace(
        artifact_root=root,
        config_hash="fixture",
        evaluation={"folds": [{"id": "fold-1"}]},
        representation={"seeds": [13]},
        runtime_evaluation_root=root / "evaluation-executions" / "new",
    )
    inheritance = root / "stage-inheritance.json"
    write_stage_inheritance_receipt(
        config,
        superseded_execution=old,
        output=inheritance,
        replacement_source_commit="new-commit",
        replacement_source_tree="new-tree",
        reason="TCA implementation correction",
    )
    old_receipt_sha = file_sha256(old / "execution.json")
    execution = {
        "schema_version": "paper-evaluation-execution-v4",
        "status": "EVALUATION_RESEALED",
        "protocol_id": "sparse-jepa-v2",
        "paper_config_hash": "fixture",
        "parameter_freeze_sha256": freeze_sha,
        "root_evaluation_source": {"commit": "root", "tree": "root-tree"},
        "root_evaluation_open_receipt_sha256": opened_sha,
        "superseded_execution_namespace": "evaluation-executions/old",
        "superseded_execution_receipt_sha256": old_receipt_sha,
        "stage_inheritance_path": "stage-inheritance.json",
        "stage_inheritance_sha256": file_sha256(inheritance),
        "evaluation_source": {"commit": "new-commit", "tree": "new-tree"},
    }
    config.runtime_evaluation_root.mkdir(parents=True)
    write_json_atomic(config.runtime_evaluation_root / "execution.json", execution)
    return config, old, inheritance, opened_sha


def test_stage_roots_and_provenance_are_typed(inherited_run):
    config, old, inheritance, _opened_sha = inherited_run
    assert stage_input_root(config, "forecast") == old.resolve()
    assert stage_input_root(config, "representation") == old.resolve()
    assert stage_input_root(config, "tca") == config.runtime_evaluation_root
    assert stage_source(config, "forecast", source_commit="new-commit", source_tree="new-tree") == {
        "commit": "old-commit",
        "tree": "old-tree",
    }
    provenance = stage_provenance(config, source_commit="new-commit", source_tree="new-tree")
    assert provenance["stage_sources"]["evaluate-forecast"]["inherited"] is True
    assert provenance["stage_sources"]["run-tca"]["inherited"] is False
    assert verify_stage_inheritance(config, inheritance)["invalidation_frontier"] == "run-tca"


def test_stage_mutation_is_detected(inherited_run):
    config, old, inheritance, _opened_sha = inherited_run
    verify_stage_inheritance(config, inheritance)
    path = old / "evaluation-v2/representations/fold-1/dense/13/accessibility.parquet"
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        verify_stage_inheritance(config, inheritance)


@pytest.mark.parametrize(
    "fault",
    [
        "corrupt-bytes",
        "missing-coordinate",
        "wrong-config",
        "wrong-freeze",
        "wrong-source",
        "wrong-test-root",
        "old-execution",
        "nonallowed-tca",
    ],
)
def test_stage_receipt_rejects_typed_contract_breaks(inherited_run, fault):
    config, old, inheritance, _opened_sha = inherited_run
    if fault == "corrupt-bytes":
        path = old / "evaluation-v2/profile-corpus" / f"{instrument_key('A')}.parquet"
        path.write_bytes(path.read_bytes() + b"corrupt")
    elif fault == "missing-coordinate":
        (old / "evaluation-v2/representations/fold-1/dense/13/accessibility.parquet").unlink()
    elif fault == "wrong-config":
        config.config_hash = "different-config"
    elif fault == "wrong-freeze":
        freeze = config.artifact_root / "selection/parameter-freeze-v1.json"
        freeze.write_bytes(freeze.read_bytes() + b"changed")
    elif fault == "wrong-source":
        payload = read_json(inheritance)
        payload["superseded_evaluation_source"]["commit"] = "wrong-source"
        write_json_atomic(inheritance, payload)
    elif fault == "wrong-test-root":
        opened = config.artifact_root / "selection/locked-test-opened-v1.json"
        payload = read_json(opened)
        payload["evaluation_git_commit"] = "wrong-root"
        write_json_atomic(opened, payload)
    elif fault == "old-execution":
        execution = old / "execution.json"
        payload = read_json(execution)
        payload["schema_version"] = "paper-evaluation-execution-v1"
        write_json_atomic(execution, payload)
    else:
        payload = read_json(inheritance)
        payload["stage_inventory"]["run-tca"] = {"files": {}}
        write_json_atomic(inheritance, payload)
    with pytest.raises(ValueError):
        verify_stage_inheritance(config, inheritance)
