import json
from types import SimpleNamespace

import pytest

from execsim.data.paper.manifests import file_sha256
from execsim.ml.paper.evaluation_execution import (
    CHAINED_SCHEMA,
    _validate_primary_jepa_source_uniformity,
    seal_evaluation_execution,
    verify_evaluation_execution,
    write_evaluation_supersession_receipt,
)


@pytest.fixture
def frozen_run(tmp_path):
    root = tmp_path / "artifacts"

    def write(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return file_sha256(path)

    sequence_universe_hash = "u" * 64
    normalization_hash = "n" * 64
    sequence_path = root / "sequences/fold-1/sequence-manifest.json"
    sequence_path.parent.mkdir(parents=True, exist_ok=True)
    sequence_path.write_text(json.dumps({"universe_manifest_hash": sequence_universe_hash}))
    sequence_hash = file_sha256(sequence_path)

    manifests = []
    for method, seed in [("raw", None), ("untrained_neural", None), ("dense", 13), ("sparse", 13)]:
        folder = f"lightgbm/fold-1/{method}/{seed or 'shared'}"
        embedding_hashes = {"train": None, "validation": None}
        if seed is not None:
            rep = f"representations/fold-1/{method}/{seed}"
            weights = write(f"{rep}/final/model.safetensors", "immutable fixture model")
            checkpoint_hash = write(
                f"{rep}/final/manifest.json",
                {
                    "fold_id": "fold-1",
                    "geometry": method,
                    "seed": seed,
                    "paper_config_hash": "fixture",
                    "calibrated_rdm_lambda": 10.0,
                    "code_commit": "jepa-commit",
                    "sequence_manifest_hash": sequence_hash,
                    "universe_manifest_hash": sequence_universe_hash,
                    "normalization_hash": normalization_hash,
                    "weights_sha256": weights,
                },
            )
            write(f"{rep}/compatibility.json", {"sequence_manifest_hash": sequence_hash})
            export = f"embeddings/fold-1/{method}/{seed}"
            files = [
                {
                    "partition": part,
                    "path": f"{part}.parquet",
                    "sha256": write(f"{export}/{part}.parquet", "immutable fixture embedding"),
                }
                for part in ("train", "validation", "test")
            ]
            embedding_hashes = {
                item["partition"]: item["sha256"]
                for item in files
                if item["partition"] in {"train", "validation"}
            }
            write(
                f"{export}/manifest.json",
                {
                    "fold_id": "fold-1",
                    "seed": seed,
                    "geometry": method,
                    "adaptation": "none",
                    "checkpoint_hash": weights,
                    "checkpoint_manifest_hash": checkpoint_hash,
                    "sequence_manifest_hash": sequence_hash,
                    "normalization_hash": normalization_hash,
                    "paper_config_hash": "fixture",
                    "files": files,
                },
            )
        models = [
            {"path": name, "sha256": write(f"{folder}/{name}", name)}
            for name in ("scale.txt", "shape.txt")
        ]
        manifest = {
            "paper_config_hash": "fixture",
            "method": method,
            "seed": seed,
            "sequence_manifest_hash": sequence_hash,
            "embedding_sha256": embedding_hashes,
            "models": models,
            "grid_results_sha256": write(f"{folder}/grid-results.json", {}),
        }
        name = f"{folder}/manifest.json"
        manifests.append({"path": name, "sha256": write(name, manifest)})
    freeze_hash = write(
        "selection/parameter-freeze-v1.json",
        {
            "status": "PARAMETERS_FROZEN",
            "paper_config_hash": "fixture",
            "selected_rdm_lambda": 10.0,
            "test_or_tca_used": False,
            "git_commit": "old",
            "git_tree": "old-tree",
            "representation_source_commit": "jepa-commit",
            "lightgbm_manifests": manifests,
            "rdm_lambda_receipt_sha256": write("selection/rdm-lambda.json", {}),
            "lightgbm_execution_receipt_sha256": write("lightgbm/execution-receipt.json", {}),
        },
    )
    ready_hash = write(
        "selection/locked-test-ready-v1.json",
        {"paper_config_hash": "fixture", "parameter_freeze_sha256": freeze_hash},
    )
    write(
        "selection/locked-test-opened-v1.json",
        {
            "status": "LOCKED-TEST-OPENED",
            "paper_config_hash": "fixture",
            "locked_test_ready_receipt_sha256": ready_hash,
            "evaluation_git_commit": "old",
            "evaluation_git_tree": "old-tree",
        },
    )
    write(
        "supersession.json",
        {"status": "SUPERSEDED_PRE_OPTIMIZATION_TEST", "parameter_freeze_sha256": freeze_hash},
    )
    config = SimpleNamespace(
        artifact_root=root,
        config_hash="fixture",
        evaluation={"folds": [{"id": "fold-1"}]},
        representation={"seeds": [13]},
        runtime_evaluation_root=root / "evaluation-executions/new",
    )
    return config


def test_reseal_preserves_original_bytes_and_rejects_changed_models(frozen_run, monkeypatch):
    config = frozen_run
    before = {
        str(path): file_sha256(path) for path in config.artifact_root.rglob("*") if path.is_file()
    }
    kwargs = dict(source_commit="new", source_tree="new-tree")
    receipt = seal_evaluation_execution(
        config, supersession=config.artifact_root / "supersession.json", **kwargs
    )
    assert receipt["initial_completed_stages"] == 0
    assert receipt["upstream_parameter_source"]["commit"] == "old"
    assert verify_evaluation_execution(config, **kwargs) == receipt
    from execsim.ml.paper import orchestration

    monkeypatch.setattr(orchestration, "_git_head", lambda: "new")
    monkeypatch.setattr(orchestration, "_git_tree", lambda: "new-tree")
    monkeypatch.setattr(
        orchestration,
        "_load_common_lambda_receipt",
        lambda config: {
            "paper_config_hash": "fixture",
            "selected_rdm_lambda": 10.0,
            "test_or_tca_used": False,
        },
    )
    assert orchestration._require_parameter_freeze(config)["git_commit"] == "old"
    assert orchestration._require_locked_test_opened(config)["evaluation_git_commit"] == "old"
    namespace = config.runtime_evaluation_root
    config.runtime_evaluation_root = None
    with pytest.raises(ValueError, match="incompatible"):
        orchestration._require_parameter_freeze(config)
    config.runtime_evaluation_root = namespace
    assert (
        seal_evaluation_execution(
            config, supersession=config.artifact_root / "supersession.json", **kwargs
        )
        == receipt
    )
    from pathlib import Path

    assert all(file_sha256(Path(path)) == digest for path, digest in before.items())
    with pytest.raises(ValueError, match="identity"):
        verify_evaluation_execution(config, source_commit="other", source_tree="new-tree")
    model = config.artifact_root / "lightgbm/fold-1/raw/shared/scale.txt"
    model.write_text("modified")
    with pytest.raises(ValueError, match="checksum"):
        verify_evaluation_execution(config, **kwargs)


@pytest.mark.parametrize("fault", ["populated", "missing-model", "wrong-freeze", "old-root"])
def test_reseal_fails_before_publishing_incomplete_or_mixed_execution(frozen_run, fault):
    config = frozen_run
    if fault == "populated":
        config.runtime_evaluation_root.mkdir(parents=True)
        (config.runtime_evaluation_root / "old-results.parquet").write_text("old")
    elif fault == "missing-model":
        (config.artifact_root / "representations/fold-1/dense/13/final/model.safetensors").unlink()
    elif fault == "wrong-freeze":
        (config.artifact_root / "supersession.json").write_text("{}")
    else:
        config.runtime_evaluation_root = config.artifact_root
    with pytest.raises((ValueError, RuntimeError)):
        seal_evaluation_execution(
            config,
            source_commit="new",
            source_tree="new-tree",
            supersession=config.artifact_root / "supersession.json",
        )
    assert not (config.runtime_evaluation_root / "execution.json").exists()


def test_reseal_binds_explicit_cuda_import_without_replacing_retained_copies(frozen_run):
    import shutil

    config = frozen_run
    original = config.artifact_root / "representations"
    imported = config.artifact_root / "upstream-cuda/representations"
    shutil.copytree(original, imported)
    retained = original / "fold-1/dense/13/compatibility.json"
    retained.write_text('{"retained_cpu_copy": true}')
    config.runtime_representation_root = imported
    kwargs = dict(source_commit="new", source_tree="new-tree")
    receipt = seal_evaluation_execution(
        config, supersession=config.artifact_root / "supersession.json", **kwargs
    )
    assert any(
        name.startswith("upstream-cuda/representations/") for name in receipt["upstream_files"]
    )
    assert not any(name.startswith("representations/") for name in receipt["upstream_files"])
    assert verify_evaluation_execution(config, **kwargs) == receipt
    assert json.loads(retained.read_text()) == {"retained_cpu_copy": True}
    config.runtime_representation_root = original
    with pytest.raises(ValueError, match="identity"):
        verify_evaluation_execution(config, **kwargs)


def test_verified_execution_hashes_once_then_detects_changed_files(frozen_run, monkeypatch):
    config = frozen_run
    kwargs = dict(source_commit="new", source_tree="new-tree")
    seal_evaluation_execution(
        config, supersession=config.artifact_root / "supersession.json", **kwargs
    )
    verified = verify_evaluation_execution(config, **kwargs)

    def unexpected_rehash(*args):
        raise AssertionError("Immutable inventory should not be rehashed within one process")

    monkeypatch.setattr("execsim.ml.paper.evaluation_execution.frozen_inventory", unexpected_rehash)
    assert verify_evaluation_execution(config, **kwargs) == verified
    verified["upstream_files"].clear()
    assert verify_evaluation_execution(config, **kwargs)["upstream_files"]
    model = config.artifact_root / "lightgbm/fold-1/raw/shared/scale.txt"
    model.write_text("changed")
    with pytest.raises(ValueError, match="changed"):
        verify_evaluation_execution(config, **kwargs)


@pytest.mark.parametrize(
    ("commits", "matches"),
    [
        (["jepa"] * 18, None),
        (["jepa"] * 17, "exactly 18"),
        (["jepa"] * 17 + [None], "non-empty"),
        (["jepa"] * 17 + ["other"], "one shared"),
    ],
)
def test_primary_jepa_source_uniformity_is_fail_closed(commits, matches):
    if matches is None:
        _validate_primary_jepa_source_uniformity(commits, expected_count=18)
    else:
        with pytest.raises(ValueError, match=matches):
            _validate_primary_jepa_source_uniformity(commits, expected_count=18)


def _reseal_fixture(config):
    return seal_evaluation_execution(
        config,
        source_commit="new",
        source_tree="new-tree",
        supersession=config.artifact_root / "supersession.json",
    )


def _rewrite_fixture_json(path, payload):
    path.write_text(json.dumps(payload))


def test_reseal_rejects_changed_sequence_manifest_before_publishing(frozen_run):
    sequence = frozen_run.artifact_root / "sequences/fold-1/sequence-manifest.json"
    _rewrite_fixture_json(sequence, {"universe_manifest_hash": "u" * 64, "changed": True})

    with pytest.raises(ValueError, match="sequence identity"):
        _reseal_fixture(frozen_run)
    assert not (frozen_run.runtime_evaluation_root / "execution.json").exists()


def test_reseal_rejects_consistently_swapped_checkpoint_and_embedding(frozen_run):
    root = frozen_run.artifact_root
    checkpoint_path = root / "representations/fold-1/dense/13/final/manifest.json"
    checkpoint = json.loads(checkpoint_path.read_text())
    weights_path = root / "representations/fold-1/dense/13/final/model.safetensors"
    weights_path.write_text("foreign fixture model")
    checkpoint["weights_sha256"] = file_sha256(weights_path)
    _rewrite_fixture_json(checkpoint_path, checkpoint)

    export_path = root / "embeddings/fold-1/dense/13/manifest.json"
    export = json.loads(export_path.read_text())
    export["checkpoint_hash"] = checkpoint["weights_sha256"]
    export["checkpoint_manifest_hash"] = file_sha256(checkpoint_path)
    train = root / "embeddings/fold-1/dense/13/train.parquet"
    train.write_text("foreign fixture embedding")
    export["files"][0]["sha256"] = file_sha256(train)
    _rewrite_fixture_json(export_path, export)

    with pytest.raises(ValueError, match="embedding checksum"):
        _reseal_fixture(frozen_run)
    assert not (frozen_run.runtime_evaluation_root / "execution.json").exists()


def test_reseal_rejects_changed_train_embedding_even_with_updated_export_manifest(frozen_run):
    root = frozen_run.artifact_root
    export_path = root / "embeddings/fold-1/dense/13/manifest.json"
    export = json.loads(export_path.read_text())
    train = root / "embeddings/fold-1/dense/13/train.parquet"
    train.write_text("changed fixture embedding")
    export["files"][0]["sha256"] = file_sha256(train)
    _rewrite_fixture_json(export_path, export)

    with pytest.raises(ValueError, match="embedding checksum"):
        _reseal_fixture(frozen_run)
    assert not (frozen_run.runtime_evaluation_root / "execution.json").exists()


def test_reseal_rejects_foreign_jepa_source_commit_before_publishing(frozen_run):
    root = frozen_run.artifact_root
    for method in ("dense", "sparse"):
        checkpoint_path = root / f"representations/fold-1/{method}/13/final/manifest.json"
        checkpoint = json.loads(checkpoint_path.read_text())
        checkpoint["code_commit"] = "foreign-jepa-commit"
        _rewrite_fixture_json(checkpoint_path, checkpoint)
        export_path = root / f"embeddings/fold-1/{method}/13/manifest.json"
        export = json.loads(export_path.read_text())
        export["checkpoint_manifest_hash"] = file_sha256(checkpoint_path)
        _rewrite_fixture_json(export_path, export)

    with pytest.raises(ValueError, match="source commit"):
        _reseal_fixture(frozen_run)
    assert not (frozen_run.runtime_evaluation_root / "execution.json").exists()


@pytest.mark.parametrize("mutated_ancestor", ["execution", "supersession"])
def test_chained_reseal_binds_original_authorization_and_immediate_prior(
    frozen_run, mutated_ancestor
):
    config = frozen_run
    first = seal_evaluation_execution(
        config,
        source_commit="evaluator-b",
        source_tree="tree-b",
        supersession=config.artifact_root / "supersession.json",
    )
    first_root = config.runtime_evaluation_root
    supersession_path = config.artifact_root / "superseded-b7.json"
    write_evaluation_supersession_receipt(
        config,
        superseded_execution=first_root,
        output=supersession_path,
        replacement_source_commit="evaluator-c",
        replacement_source_tree="tree-c",
        reason="corrected evaluation provenance contract",
    )
    config.runtime_evaluation_root = config.artifact_root / "evaluation-executions/second"
    second = seal_evaluation_execution(
        config,
        source_commit="evaluator-c",
        source_tree="tree-c",
        supersession=supersession_path,
    )
    assert second["schema_version"] == CHAINED_SCHEMA
    assert second["root_evaluation_source"] == {"commit": "old", "tree": "old-tree"}
    assert second["previous_evaluation_source"] == first["evaluation_source"]
    assert (
        verify_evaluation_execution(config, source_commit="evaluator-c", source_tree="tree-c")
        == second
    )
    second_root = config.runtime_evaluation_root
    (second_root / "partial-forecast.parquet").write_bytes(b"preserved old output")
    next_supersession = config.artifact_root / "superseded-c.json"
    write_evaluation_supersession_receipt(
        config,
        superseded_execution=second_root,
        output=next_supersession,
        replacement_source_commit="evaluator-d",
        replacement_source_tree="tree-d",
        reason="operator performance abort; immutable predecessors retained",
    )
    config.runtime_evaluation_root = config.artifact_root / "evaluation-executions/third"
    third = seal_evaluation_execution(
        config, source_commit="evaluator-d", source_tree="tree-d", supersession=next_supersession
    )
    assert third["root_evaluation_source"] == second["root_evaluation_source"]
    assert third["previous_evaluation_source"] == second["evaluation_source"]
    assert not (config.runtime_evaluation_root / "partial-forecast.parquet").exists()
    assert (second_root / "partial-forecast.parquet").read_bytes() == b"preserved old output"
    assert (
        verify_evaluation_execution(config, source_commit="evaluator-d", source_tree="tree-d")
        == third
    )
    # The third generation must continue checking the first resealed ancestor.
    ancestor_path = (
        first_root / "execution.json" if mutated_ancestor == "execution" else supersession_path
    )
    with ancestor_path.open("a") as handle:
        handle.write(" ")
    with pytest.raises(ValueError, match="checksum"):
        verify_evaluation_execution(config, source_commit="evaluator-d", source_tree="tree-d")


def test_chained_reseal_rejects_mutated_prior_execution(frozen_run):
    config = frozen_run
    first = seal_evaluation_execution(
        config,
        source_commit="evaluator-b",
        source_tree="tree-b",
        supersession=config.artifact_root / "supersession.json",
    )
    first_root = config.runtime_evaluation_root
    supersession_path = config.artifact_root / "superseded-b7.json"
    write_evaluation_supersession_receipt(
        config,
        superseded_execution=first_root,
        output=supersession_path,
        replacement_source_commit="evaluator-c",
        replacement_source_tree="tree-c",
        reason="corrected evaluation provenance contract",
    )
    # Preserve the namespace but mutate only the prior receipt bytes.  The
    # typed receipt's checksum must make the second seal fail closed.
    (first_root / "execution.json").write_text(json.dumps({**first, "mutated": True}))
    config.runtime_evaluation_root = config.artifact_root / "evaluation-executions/second"
    with pytest.raises(ValueError, match="checksum"):
        seal_evaluation_execution(
            config,
            source_commit="evaluator-c",
            source_tree="tree-c",
            supersession=supersession_path,
        )
    assert not (config.runtime_evaluation_root / "execution.json").exists()


@pytest.mark.parametrize("field", ["paper_config_hash", "parameter_freeze_sha256"])
def test_chained_reseal_rejects_wrong_typed_identity(frozen_run, field):
    config = frozen_run
    _first = seal_evaluation_execution(
        config,
        source_commit="evaluator-b",
        source_tree="tree-b",
        supersession=config.artifact_root / "supersession.json",
    )
    supersession_path = config.artifact_root / "superseded-b7.json"
    write_evaluation_supersession_receipt(
        config,
        superseded_execution=config.runtime_evaluation_root,
        output=supersession_path,
        replacement_source_commit="evaluator-c",
        replacement_source_tree="tree-c",
        reason="corrected evaluation provenance contract",
    )
    typed = json.loads(supersession_path.read_text())
    typed[field] = "wrong"
    _rewrite_fixture_json(supersession_path, typed)
    config.runtime_evaluation_root = config.artifact_root / "evaluation-executions/second"
    with pytest.raises(ValueError, match="incompatible"):
        seal_evaluation_execution(
            config,
            source_commit="evaluator-c",
            source_tree="tree-c",
            supersession=supersession_path,
        )


def test_chained_reseal_rejects_populated_destination(frozen_run):
    config = frozen_run
    _first = seal_evaluation_execution(
        config,
        source_commit="evaluator-b",
        source_tree="tree-b",
        supersession=config.artifact_root / "supersession.json",
    )
    supersession_path = config.artifact_root / "superseded-b7.json"
    write_evaluation_supersession_receipt(
        config,
        superseded_execution=config.runtime_evaluation_root,
        output=supersession_path,
        replacement_source_commit="evaluator-c",
        replacement_source_tree="tree-c",
        reason="corrected evaluation provenance contract",
    )
    destination = config.artifact_root / "evaluation-executions/second"
    destination.mkdir(parents=True)
    (destination / "partial.parquet").write_text("not a completed artifact")
    config.runtime_evaluation_root = destination
    with pytest.raises(ValueError, match="empty"):
        seal_evaluation_execution(
            config,
            source_commit="evaluator-c",
            source_tree="tree-c",
            supersession=supersession_path,
        )
