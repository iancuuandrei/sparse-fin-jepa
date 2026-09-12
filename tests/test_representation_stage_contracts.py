"""Exercise the representation stage with production-shaped local artifacts."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
from test_representation_contracts import export_fixture_checkpoint
from test_representation_contracts import regime_corpus as _fixture_regime_corpus  # noqa: F401

from execsim.data.paper.manifests import read_json, write_json_atomic
from execsim.ml.paper import evaluation_artifacts, orchestration
from execsim.ml.representations import checkpoints, probe_cache


def _assert_complete_probe_cache(root: Path) -> None:
    from execsim.ml.representations.probe_cache import EncodedProbeBatches

    assert {path.name for path in root.iterdir()} == {"train", "validation", "test"}
    for partition in ("train", "validation", "test"):
        cache = root / partition
        receipt = read_json(cache / "manifest.json")
        # Construction validates the real cache's complete inventory, checksums, and tensor sizes.
        EncodedProbeBatches(cache, receipt["identity"], None)
        assert receipt["arrays"].keys() == {"features", "targets", "observable", "complete"}


def test_representation_stage_publication_resume_and_identity_contracts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    pytest.importorskip("torch")
    sequence_manifest = request.getfixturevalue("_fixture_regime_corpus")

    config = SimpleNamespace(
        artifact_root=tmp_path,
        runtime_evaluation_root=None,
        runtime_representation_root=None,
        config_hash="synthetic-stage-contract",
        evaluation={"folds": [{"id": "fold-1"}]},
        representation={
            "seeds": [13],
            "rdm_projections_evaluation": 16,
            "batch_size": 64,
            "probe_ridge_alphas": [1.0],
            "probe_mlp_epochs": 1,
        },
        sequences={"session_cache_size": 2, "num_workers": 0, "prefetch_factor": 1},
        authorize=lambda *args, **kwargs: None,
    )
    write_json_atomic(tmp_path / "selection/parameter-freeze-v1.json", {"fixture": True})
    for geometry in ("dense", "sparse"):
        export_fixture_checkpoint(tmp_path, sequence_manifest, geometry)

    source = {"commit": "fixture-source"}
    monkeypatch.setattr(orchestration, "_git_head", lambda: source["commit"])
    monkeypatch.setattr(orchestration, "_git_tree", lambda: "fixture-tree")
    monkeypatch.setattr(orchestration, "_require_parameter_freeze", lambda _config: {})
    monkeypatch.setattr(orchestration, "_require_locked_test_opened", lambda _config: {})

    real_load_checkpoint = checkpoints.load_checkpoint
    loaded_checkpoints: list[Path] = []

    def observe_checkpoint_load(model: Any, directory: Path, **kwargs: Any) -> Any:
        loaded_checkpoints.append(directory)
        return real_load_checkpoint(model, directory, **kwargs)

    monkeypatch.setattr(checkpoints, "load_checkpoint", observe_checkpoint_load)

    real_materialize = probe_cache.materialize_probe_batches
    materialized: list[tuple[Path, bool]] = []

    def observe_materialize(root: Path, **kwargs: Any) -> Any:
        materialized.append((root, root.exists()))
        return real_materialize(root, **kwargs)

    monkeypatch.setattr(probe_cache, "materialize_probe_batches", observe_materialize)

    coordinate_root = tmp_path / "evaluation-v2/representations/fold-1"
    dense_destination = coordinate_root / "dense/13"
    sparse_destination = coordinate_root / "sparse/13"
    cache_root = tmp_path / "probe-cache/fold-1"

    real_publish = evaluation_artifacts.publish_frames
    injected = {"before_dense": True, "after_sparse": True}

    def interrupt_at_publication(directory: Path, **kwargs: Any) -> dict[str, Any]:
        relative = directory.relative_to(coordinate_root).as_posix()
        if relative == "dense/13" and injected["before_dense"]:
            injected["before_dense"] = False
            raise RuntimeError("synthetic interruption before coordinate publication")
        receipt = real_publish(directory, **kwargs)
        if relative == "sparse/13" and injected["after_sparse"]:
            injected["after_sparse"] = False
            raise RuntimeError("synthetic interruption after coordinate publication")
        return receipt

    monkeypatch.setattr(evaluation_artifacts, "publish_frames", interrupt_at_publication)

    with pytest.raises(RuntimeError, match="before coordinate publication"):
        orchestration.evaluate_representations_stage(
            config, full_run_cli_enabled=True, runtime_approval=None
        )

    # Probe completion and diagnostics precede publication. A failure at that seam
    # leaves only verified cache partitions, never a partial coordinate directory.
    assert not dense_destination.exists()
    assert not list(dense_destination.parent.glob(".evaluation-*"))
    _assert_complete_probe_cache(cache_root / "dense/13")
    assert len(loaded_checkpoints) == 1
    assert len(materialized) == 3
    assert all(not existed for _, existed in materialized)

    materialized.clear()
    with pytest.raises(RuntimeError, match="after coordinate publication"):
        orchestration.evaluate_representations_stage(
            config, full_run_cli_enabled=True, runtime_approval=None
        )

    # The retry reuses the completed dense cache, then publishes sparse atomically.
    # The injected post-publish interruption leaves a complete, verifiable coordinate.
    assert dense_destination.is_dir()
    assert not (cache_root / "dense/13").exists()
    _assert_complete_probe_cache(cache_root / "sparse/13")
    sparse_receipt = read_json(sparse_destination / "manifest.json")
    evaluation_artifacts.verify_artifact(
        sparse_destination,
        identity=sparse_receipt["identity"],
        names=tuple(sparse_receipt["files"]),
    )
    assert [existed for _, existed in materialized] == [True] * 3 + [False] * 3
    assert len(loaded_checkpoints) == 3

    coordinate_bytes = {
        path.relative_to(coordinate_root).as_posix(): path.read_bytes()
        for destination in (dense_destination, sparse_destination)
        for path in destination.iterdir()
        if path.is_file()
    }
    materialized.clear()
    result = orchestration.evaluate_representations_stage(
        config, full_run_cli_enabled=True, runtime_approval=None
    )

    # Published coordinates are accepted as immutable same-source results without
    # loading a checkpoint, fitting probes, rebuilding labels, or reading caches.
    assert len(loaded_checkpoints) == 3
    assert materialized == []
    assert (cache_root / "sparse/13").is_dir()
    assert coordinate_bytes == {
        path.relative_to(coordinate_root).as_posix(): path.read_bytes()
        for destination in (dense_destination, sparse_destination)
        for path in destination.iterdir()
        if path.is_file()
    }

    accessibility = pd.read_parquet(result["accessibility"])
    date_metrics = pd.read_parquet(result["date_metrics"])
    support = pd.read_parquet(result["support_regimes"])
    assert set(accessibility["geometry"]) == {"dense", "sparse"}
    assert len(accessibility) == 24
    assert not date_metrics.empty
    assert date_metrics["sample_identity_sha256"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert (date_metrics["row_count"] > 0).all()
    assert len(support) == 1
    assert support.loc[0, "ordinary_rows"] + support.loc[0, "unusual_rows"] > 0
    merged_manifest = (
        Path(result["accessibility"]).parent / "representation-evaluation-manifest.json"
    )
    assert read_json(merged_manifest)

    # The coordinate receipt binds its results to the evaluator source, and the
    # checksums reject changed data even when the source identity is unchanged.
    source["commit"] = "different-source"
    with pytest.raises(ValueError, match="identity mismatch"):
        orchestration.evaluate_representations_stage(
            config, full_run_cli_enabled=True, runtime_approval=None
        )
    assert len(loaded_checkpoints) == 3

    source["commit"] = "fixture-source"
    (sparse_destination / "support.parquet").write_bytes(b"corrupt coordinate data")
    with pytest.raises(ValueError, match="checksum mismatch"):
        orchestration.evaluate_representations_stage(
            config, full_run_cli_enabled=True, runtime_approval=None
        )
    assert len(loaded_checkpoints) == 3
