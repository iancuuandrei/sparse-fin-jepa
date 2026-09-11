from pathlib import Path
from types import SimpleNamespace

import pytest

from execsim.data.paper.manifests import file_sha256, write_json_atomic
from scripts import runpod


def test_transfer_verification_rejects_corruption_duplicates_and_escape(tmp_path: Path):
    member = tmp_path / "data.bin"
    member.write_bytes(b"immutable fixture")
    payload = {
        "schema": "runpod-transfer-v1",
        "fold_id": "fold-1",
        "files": [
            {"path": "data.bin", "bytes": member.stat().st_size, "sha256": file_sha256(member)}
        ],
    }
    manifest = tmp_path / "transfer.json"
    write_json_atomic(manifest, payload)
    assert runpod.verify_bundle(tmp_path, file_sha256(manifest))["fold_id"] == "fold-1"
    with pytest.raises(ValueError, match="manifest SHA"):
        runpod.verify_bundle(tmp_path, "0" * 64)
    member.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        runpod.verify_bundle(tmp_path)
    payload["files"] *= 2
    write_json_atomic(manifest, payload)
    with pytest.raises(ValueError, match="duplicated"):
        runpod.verify_bundle(tmp_path)
    for name in ("../secret", "/etc/passwd", "C:/secret", "x\\..\\secret"):
        with pytest.raises(ValueError):
            runpod.safe_path(tmp_path, name)


def test_launch_barrier_requires_three_distinct_pods_same_source(tmp_path: Path, monkeypatch):
    identity = {"source_commit": "a" * 40, "source_tree": "b" * 40}
    monkeypatch.setattr(runpod, "source_identity", lambda _: identity)
    monkeypatch.setattr(
        runpod, "load_paper_config", lambda _: SimpleNamespace(config_hash="c" * 64)
    )
    paths = []
    for fold in runpod.FOLDS:
        path = tmp_path / f"{fold}.json"
        write_json_atomic(
            path,
            {
                **identity,
                "fold_id": fold,
                "status": "PASS",
                "hardware": {"pod_id": fold},
                "paper_config_hash": "c" * 64,
                "execution": {"device_type": "gpu", "gpu_use_dp": False},
                "lightgbm_version": "4.7.0",
                "opencl": {"device": "NVIDIA GeForce RTX 3090"},
            },
        )
        paths.append(path)
    args = SimpleNamespace(repo=tmp_path, receipts=paths, output=tmp_path / "ready.json")
    assert runpod.ready(args)["status"] == "THREE_PODS_QUALIFIED"
    args.receipts = [paths[0], paths[0], paths[2]]
    with pytest.raises(ValueError, match="distinct qualified folds"):
        runpod.ready(args)


def test_fold_cli_routes_only_requested_fold():
    from execsim.cli import build_parser

    args = build_parser().parse_args(
        [
            "ml",
            "paper",
            "train-volume-model",
            "--fold",
            "fold-2",
            "--sequence-root",
            "/inputs/sequences",
            "--model-output-root",
            "/outputs",
        ]
    )
    assert args.fold == "fold-2"
    assert args.sequence_root == Path("/inputs/sequences")


def test_reassembly_requires_24_native_artifacts_and_rejects_corruption(tmp_path, monkeypatch):
    from dataclasses import asdict
    from itertools import product

    import numpy as np
    import pandas as pd

    from execsim.ml.models.lightgbm_adapter import (
        LightGBMConfig,
        LightGBMExecutionOptions,
        LightGBMVolumeModel,
    )

    source = {"source_commit": "a" * 40, "source_tree": "b" * 40}
    config_hash = "c" * 64
    monkeypatch.setattr(runpod, "source_identity", lambda _: source)
    monkeypatch.setattr(
        runpod, "load_paper_config", lambda _: SimpleNamespace(config_hash=config_hash)
    )
    scale = pd.DataFrame({"x": np.arange(8), "baseline_remaining_volume": np.ones(8)})
    shape = pd.DataFrame(
        {
            "x": np.repeat(np.arange(8), 2),
            "case_id": np.repeat(np.arange(8), 2),
            "target_bucket": np.tile([0, 1], 8),
        }
    )
    model = LightGBMVolumeModel().fit_frames(
        scale, np.ones(8) * 2, shape, np.ones(16) * 0.5, categorical_features=()
    )
    # Only artifact/backend metadata is synthetic here; native fixture models train on CPU.
    model.execution = LightGBMExecutionOptions(
        device_type="gpu", gpu_platform_id=0, gpu_device_id=0
    )
    models = tmp_path / "models"
    packages = []
    for fold in runpod.FOLDS:
        receipt = models / fold / "execution-receipt.json"
        write_json_atomic(
            receipt,
            {
                "git_commit": source["source_commit"],
                "paper_config_hash": config_hash,
                "fold_id": fold,
                "lightgbm_execution": model.execution.identity(),
            },
        )
        for method, seed in runpod.COORDINATES:
            model.config = LightGBMConfig(seed=seed or 13)
            model.scale_config = model.shape_config = model.config
            destination = models / fold / method / str(seed or "shared")
            model.save_native(
                destination,
                {
                    "fold_id": fold,
                    "method": method,
                    "seed": seed,
                    "git_commit": source["source_commit"],
                    "paper_config_hash": config_hash,
                    "feature_schema_version": "fixture",
                    "training_cutoff": "fixture",
                    "validation_range": ["fixture"],
                    "categorical_features": [],
                    "execution_receipt_sha256": file_sha256(receipt),
                },
            )
            grid = {
                "selection_data": "validation_only",
                "selected_scale_config": asdict(model.config),
                "selected_shape_config": asdict(model.config),
                "candidates": [
                    {
                        "config": asdict(LightGBMConfig(a, b, c, seed=seed or 13)),
                        "scale_mae": 1.0,
                        "shape_error": 1.0,
                    }
                    for a, b, c in product((15, 31), (50, 200), (1.0, 10.0))
                ],
            }
            write_json_atomic(destination / "grid-results.json", grid)
            payload = runpod.read_json(destination / "manifest.json")
            payload["grid_results_sha256"] = file_sha256(destination / "grid-results.json")
            write_json_atomic(destination / "manifest.json", payload)
        package = tmp_path / fold
        runpod.export_fold(SimpleNamespace(repo=tmp_path, models=models, fold=fold, output=package))
        packages.append(package)
    args = SimpleNamespace(
        repo=tmp_path,
        inputs=packages,
        output=tmp_path / "merged",
        input_sha256=[file_sha256(p / "transfer.json") for p in packages],
    )
    assert runpod.reassemble(args)["coordinates"] == 24
    assert (args.output / "reassembly.json").is_file()
    args.output = tmp_path / "bad"
    (packages[1] / "fold-2/raw/shared/scale.txt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        runpod.reassemble(args)
