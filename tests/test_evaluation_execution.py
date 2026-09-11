import json
from types import SimpleNamespace

import pytest

from execsim.data.paper.manifests import file_sha256
from execsim.ml.paper.evaluation_execution import (
    _validate_primary_jepa_source_uniformity,
    seal_evaluation_execution,
    verify_evaluation_execution,
)


@pytest.fixture
def frozen_run(tmp_path):
    root = tmp_path / "artifacts"

    def write(name, value):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return file_sha256(path)

    manifests = []
    for method, seed in [("raw", None), ("untrained_neural", None), ("dense", 13), ("sparse", 13)]:
        folder = f"lightgbm/fold-1/{method}/{seed or 'shared'}"
        models = [
            {"path": name, "sha256": write(f"{folder}/{name}", name)}
            for name in ("scale.txt", "shape.txt")
        ]
        manifest = {
            "paper_config_hash": "fixture",
            "method": method,
            "seed": seed,
            "models": models,
            "grid_results_sha256": write(f"{folder}/grid-results.json", {}),
        }
        name = f"{folder}/manifest.json"
        manifests.append({"path": name, "sha256": write(name, manifest)})
        if seed is None:
            continue
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
                "weights_sha256": weights,
            },
        )
        write(f"{rep}/compatibility.json", {})
        export = f"embeddings/fold-1/{method}/{seed}"
        files = [
            {
                "partition": part,
                "path": f"{part}.parquet",
                "sha256": write(f"{export}/{part}.parquet", "immutable fixture embedding"),
            }
            for part in ("train", "validation", "test")
        ]
        write(
            f"{export}/manifest.json",
            {
                "checkpoint_hash": weights,
                "checkpoint_manifest_hash": checkpoint_hash,
                "files": files,
            },
        )
    write("sequences/fold-1/sequence-manifest.json", {})
    freeze_hash = write(
        "selection/parameter-freeze-v1.json",
        {
            "status": "PARAMETERS_FROZEN",
            "paper_config_hash": "fixture",
            "selected_rdm_lambda": 10.0,
            "test_or_tca_used": False,
            "git_commit": "old",
            "git_tree": "old-tree",
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
