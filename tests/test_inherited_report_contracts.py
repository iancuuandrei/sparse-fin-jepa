"""Exercise the real mixed-source report/final-freeze recovery contract.

The fixture deliberately keeps forecast and representation bytes in the old
execution namespace while publishing a new TCA and report namespace.  It is
small and synthetic, but it uses the production report builder and the real
TCA producer fixture; only the surrounding stage inventories are expanded to
the configured 18/24 matrix needed by the final freeze contract.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest
import test_report_stage_contracts as report_fixture
from test_representation_contracts import regime_corpus as _fixture_regime_corpus  # noqa: F401

from execsim.data.paper.manifests import file_sha256, read_json, write_json_atomic
from execsim.ml.paper import orchestration
from execsim.ml.paper.evaluation_artifacts import merge_result_shards, publish_frames
from execsim.ml.paper.evaluation_execution import evaluation_root
from execsim.ml.paper.evaluation_workers import compact_profile_corpus, instrument_key

_SOURCE_A = {"commit": "synthetic-source-a", "tree": "synthetic-tree-a"}
_SOURCE_B = {"commit": "synthetic-source-b", "tree": "synthetic-tree-b"}
_ROOT_SOURCE = {"commit": "synthetic-root", "tree": "synthetic-root-tree"}


def _stage_identity(config: SimpleNamespace, source: dict[str, str]) -> dict[str, Any]:
    return {
        "source_commit": source["commit"],
        "source_tree": source["tree"],
        "paper_config_hash": config.config_hash,
        "parameter_freeze_sha256": file_sha256(
            config.artifact_root / "selection/parameter-freeze-v1.json"
        ),
    }


def _write_selection_receipts(config: SimpleNamespace) -> None:
    write_json_atomic(
        config.artifact_root / "selection/parameter-freeze-v1.json",
        {"paper_config_hash": config.config_hash, "status": "SYNTHETIC_FIXTURE_ONLY"},
    )
    write_json_atomic(
        config.artifact_root / "selection/locked-test-opened-v1.json",
        {
            "paper_config_hash": config.config_hash,
            "status": "SYNTHETIC_FIXTURE_ONLY",
            "evaluation_git_commit": _ROOT_SOURCE["commit"],
            "evaluation_git_tree": _ROOT_SOURCE["tree"],
        },
    )


def _write_predecessor_execution(config: SimpleNamespace, source_root: Path) -> Path:
    """Write the minimal resealed predecessor receipt accepted by v1 inheritance."""
    path = source_root / "execution.json"
    upstream_files = {
        file.relative_to(config.artifact_root).as_posix(): file_sha256(file)
        for file in sorted(config.artifact_root.rglob("*"))
        if file.is_file() and file != path
    }
    write_json_atomic(
        path,
        {
            "schema_version": "paper-evaluation-execution-v2",
            "status": "EVALUATION_RESEALED",
            "protocol_id": "sparse-jepa-v2",
            "paper_config_hash": config.config_hash,
            "parameter_freeze_sha256": file_sha256(
                config.artifact_root / "selection/parameter-freeze-v1.json"
            ),
            "previous_evaluation_source": dict(_ROOT_SOURCE),
            "evaluation_source": dict(_SOURCE_A),
            "upstream_files": upstream_files,
        },
    )
    return path


def _copy_forecast_inventory(
    config: SimpleNamespace, source_root: Path, tca_fixture_root: Path
) -> None:
    """Republish the real TCA fixture's forecast inputs under source A identity."""
    source = _stage_identity(config, _SOURCE_A)
    from test_tca_production_contracts import _INSTRUMENT

    compact_profile_corpus(
        tca_fixture_root / "market.parquet",
        source_root / "evaluation-v2/profile-corpus",
        identity={
            "paper_config_hash": config.config_hash,
            "source_commit": _SOURCE_A["commit"],
            "source_tree": _SOURCE_A["tree"],
            "parameter_freeze_sha256": file_sha256(
                config.artifact_root / "selection/parameter-freeze-v1.json"
            ),
        },
    )
    profile_manifest = read_json(source_root / "evaluation-v2/profile-corpus/manifest.json")

    variants = (
        ("raw", None),
        ("untrained_neural", None),
        *((geometry, seed) for geometry in ("dense", "sparse") for seed in (13, 29, 47)),
    )
    base_source = tca_fixture_root / "evaluation-base"
    for fold in config.evaluation["folds"]:
        fold_id = str(fold["id"])
        base_frames = {}
        for name in ("scale-base.parquet", "shape-base.parquet"):
            frame = pd.read_parquet(base_source / name)
            if "fold_id" in frame:
                frame["fold_id"] = fold_id
            base_frames[name] = frame
        publish_frames(
            source_root / "evaluation-v2/bases" / fold_id,
            identity={
                **source,
                "schema_version": "paper-evaluation-base-v2",
                "sequence_manifest_sha256": file_sha256(
                    config.artifact_root / "sequences" / fold_id / "sequence-manifest.json"
                ),
                "partition": "test",
            },
            frames=base_frames,
        )
        for method, seed in variants:
            fixture_directory = tca_fixture_root / "forecasts" / method / str(seed or "shared")
            directory = (
                source_root / "evaluation-v2/forecasts" / fold_id / method / str(seed or "shared")
            )
            frames = {}
            for name in ("scale.parquet", "shape.parquet", "metrics.parquet"):
                frame = pd.read_parquet(fixture_directory / name)
                if "fold_id" in frame:
                    frame["fold_id"] = fold_id
                frames[name] = frame
            publish_frames(
                directory,
                identity={
                    **source,
                    "schema_version": "paper-forecast-ledger-v2",
                    "fold_id": fold_id,
                    "method": method,
                    "seed": seed,
                    "model_manifest_sha256": file_sha256(
                        config.artifact_root
                        / "lightgbm"
                        / fold_id
                        / method
                        / str(seed or "shared")
                        / "manifest.json"
                    ),
                    "base_manifest_sha256": file_sha256(
                        source_root / "evaluation-v2/bases" / fold_id / "manifest.json"
                    ),
                    "embedding_sha256": (
                        file_sha256(
                            config.artifact_root
                            / "embeddings"
                            / fold_id
                            / method
                            / str(seed)
                            / "partition=test/embeddings.parquet"
                        )
                        if seed is not None
                        else None
                    ),
                },
                frames=frames,
            )

        ewma_source = tca_fixture_root / "ewma-ledger"
        ewma_frames = {}
        for name in (
            "scale.parquet",
            "shape.parquet",
            "metrics.parquet",
            "minute-forecasts.parquet",
            "unavailable.parquet",
        ):
            frame = pd.read_parquet(ewma_source / name)
            if "fold_id" in frame:
                frame["fold_id"] = fold_id
            ewma_frames[name] = frame
        publish_frames(
            source_root
            / "evaluation-v2/forecasts"
            / fold_id
            / "ewma"
            / instrument_key(_INSTRUMENT),
            identity={
                **source,
                "schema_version": "paper-ewma-ledger-v4",
                "fold_id": fold_id,
                "instrument_id": _INSTRUMENT,
                "base_manifest_sha256": file_sha256(
                    source_root / "evaluation-v2/bases" / fold_id / "manifest.json"
                ),
                "market_sha256": file_sha256(
                    source_root
                    / "evaluation-v2/profile-corpus"
                    / profile_manifest["instruments"][_INSTRUMENT]
                ),
            },
            frames=ewma_frames,
        )


def _merge_forecast_inventory(config: SimpleNamespace, source_root: Path) -> Path:
    """Merge the retained coordinate ledgers and bind every source path."""
    source = _stage_identity(config, _SOURCE_A)
    sources: dict[str, tuple[Path, str]] = {}
    variants = (
        ("raw", None),
        ("untrained_neural", None),
        *((geometry, seed) for geometry in ("dense", "sparse") for seed in (13, 29, 47)),
    )
    for fold in config.evaluation["folds"]:
        fold_id = str(fold["id"])
        for method, seed in variants:
            path = (
                source_root
                / "evaluation-v2"
                / "forecasts"
                / fold_id
                / method
                / str(seed or "shared")
                / "metrics.parquet"
            )
            sources[path.relative_to(config.artifact_root).as_posix()] = (path, file_sha256(path))
        instrument = "instrument-contract"
        path = (
            source_root
            / "evaluation-v2"
            / "forecasts"
            / fold_id
            / "ewma"
            / instrument_key(instrument)
            / "metrics.parquet"
        )
        sources[path.relative_to(config.artifact_root).as_posix()] = (path, file_sha256(path))
    destination = source_root / "evaluation/forecast-results.parquet"
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
        identity=source,
        schema_version="paper-forecast-evaluation-v1",
    )
    return destination


def _copy_representation_inventory(
    config: SimpleNamespace, source_root: Path, real_root: Path, folds: list[dict[str, Any]]
) -> None:
    """Expand two real coordinates into the complete configured 18-coordinate inventory."""
    source = _stage_identity(config, _SOURCE_A)
    real_coordinates = {
        "dense": real_root / "evaluation-v2/representations/fold-1/dense/13",
        "sparse": real_root / "evaluation-v2/representations/fold-1/sparse/13",
    }
    coordinates: dict[str, dict[str, tuple[Path, str]]] = {
        "accessibility": {},
        "date-metrics": {},
        "support": {},
    }
    for fold in folds:
        fold_id = str(fold["id"])
        for geometry in ("dense", "sparse"):
            for seed in (13, 29, 47):
                coordinate = f"{fold_id}/{geometry}/{seed}"
                destination = source_root / "evaluation-v2/representations" / coordinate
                fixture_directory = real_coordinates[geometry]
                names = ["accessibility.parquet", "date-metrics.parquet"]
                if geometry == "sparse":
                    names.append("support.parquet")
                frames = {}
                for name in names:
                    frame = pd.read_parquet(fixture_directory / name)
                    for column, value in (
                        ("fold_id", fold_id),
                        ("geometry", geometry),
                        ("seed", seed),
                    ):
                        if column in frame:
                            frame[column] = value
                    frames[name] = frame
                checkpoint_manifest = (
                    config.artifact_root
                    / "representations"
                    / fold_id
                    / geometry
                    / str(seed)
                    / "final/manifest.json"
                )
                compatibility = (
                    config.artifact_root
                    / "representations"
                    / fold_id
                    / geometry
                    / str(seed)
                    / "compatibility.json"
                )
                embedding_manifest = (
                    config.artifact_root
                    / "embeddings"
                    / fold_id
                    / geometry
                    / str(seed)
                    / "manifest.json"
                )
                sequence = config.artifact_root / "sequences" / fold_id / "sequence-manifest.json"
                coordinate_identity = {
                    **source,
                    "schema_version": "paper-representation-coordinate-v1",
                    "coordinate": coordinate,
                    "sequence_sha256": file_sha256(sequence),
                    "checkpoint_sha256": file_sha256(checkpoint_manifest),
                    "compatibility_sha256": file_sha256(compatibility),
                    "embedding_manifest_sha256": file_sha256(embedding_manifest),
                }
                if destination.exists():
                    receipt = read_json(destination / "manifest.json")
                    receipt["identity"] = coordinate_identity
                    write_json_atomic(destination / "manifest.json", receipt)
                else:
                    receipt = publish_frames(
                        destination,
                        identity=coordinate_identity,
                        frames=frames,
                    )
                for name in names:
                    coordinates[Path(name).stem][coordinate] = (
                        destination / name,
                        receipt["files"][name]["sha256"],
                    )

    aggregate_root = source_root / "evaluation"
    for name in (
        "representation-accessibility.parquet",
        "representation-date-metrics.parquet",
        "support-regimes.parquet",
        "representation-accessibility.manifest.json",
        "representation-date-metrics.manifest.json",
        "support-regimes.manifest.json",
        "representation-evaluation-manifest.json",
    ):
        path = aggregate_root / name
        if path.exists():
            path.unlink()
    for kind, name, keys in (
        ("accessibility", "representation-accessibility.parquet", ["probe_capacity", "horizon"]),
        (
            "date-metrics",
            "representation-date-metrics.parquet",
            ["probe_capacity", "horizon", "date"],
        ),
        ("support", "support-regimes.parquet", []),
    ):
        destination = aggregate_root / name
        merge_result_shards(
            destination,
            sources=coordinates[kind],
            keys=["fold_id", "geometry", "seed", *keys],
            identity=source,
            schema_version="paper-representation-result-v1",
        )
    write_json_atomic(
        aggregate_root / "representation-evaluation-manifest.json",
        {
            "schema_version": "paper-representation-evaluation-v2",
            "paper_config_hash": config.config_hash,
            "accessibility_sha256": file_sha256(
                aggregate_root / "representation-accessibility.parquet"
            ),
            "date_metrics_sha256": file_sha256(
                aggregate_root / "representation-date-metrics.parquet"
            ),
            "support_regimes_sha256": file_sha256(aggregate_root / "support-regimes.parquet"),
        },
    )


def _materialize_frozen_representation_matrix(
    config: SimpleNamespace, folds: list[dict[str, Any]]
) -> None:
    """Expand the two real fixture coordinates into a freeze-valid 18-run matrix."""
    representation_root = config.artifact_root / "representations"
    source_roots = {
        geometry: representation_root / "fold-1" / geometry / "13"
        for geometry in ("dense", "sparse")
    }
    for fold in folds:
        fold_id = str(fold["id"])
        sequence = config.artifact_root / "sequences" / fold_id / "sequence-manifest.json"
        sequence_hash = file_sha256(sequence)
        universe_hash = read_json(sequence)["universe_manifest_hash"]
        for geometry in ("dense", "sparse"):
            for seed in (13, 29, 47):
                source = source_roots[geometry]
                destination = representation_root / fold_id / geometry / str(seed)
                if destination.resolve() != source.resolve():
                    shutil.copytree(source, destination)
                source_embedding = config.artifact_root / "embeddings" / "fold-1" / geometry / "13"
                embedding_path = (
                    config.artifact_root / "embeddings" / fold_id / geometry / str(seed)
                )
                if embedding_path.resolve() != source_embedding.resolve():
                    shutil.copytree(source_embedding, embedding_path)

                checkpoint_path = destination / "final" / "manifest.json"
                checkpoint = read_json(checkpoint_path)
                checkpoint.update(
                    {
                        "fold_id": fold_id,
                        "geometry": geometry,
                        "seed": seed,
                        "paper_config_hash": config.config_hash,
                        "calibrated_rdm_lambda": 10.0,
                        "sequence_manifest_hash": sequence_hash,
                        "universe_manifest_hash": universe_hash,
                        "code_commit": "fixture-only",
                    }
                )
                write_json_atomic(checkpoint_path, checkpoint)
                checkpoint_hash = file_sha256(checkpoint_path)

                compatibility_path = destination / "compatibility.json"
                compatibility = read_json(compatibility_path)
                compatibility.update(
                    {
                        "fold_id": fold_id,
                        "sequence_manifest_hash": sequence_hash,
                        "universe_manifest_hash": universe_hash,
                        "paper_config_hash": config.config_hash,
                        "calibrated_rdm_lambda": 10.0,
                    }
                )
                write_json_atomic(compatibility_path, compatibility)

                embedding_manifest_path = embedding_path / "manifest.json"
                embedding = read_json(embedding_manifest_path)
                embedding.update(
                    {
                        "fold_id": fold_id,
                        "geometry": geometry,
                        "seed": seed,
                        "paper_config_hash": config.config_hash,
                        "sequence_manifest_hash": sequence_hash,
                        "checkpoint_hash": checkpoint["weights_sha256"],
                        "checkpoint_manifest_hash": checkpoint_hash,
                    }
                )
                write_json_atomic(embedding_manifest_path, embedding)


def _materialize_frozen_lightgbm_matrix(
    config: SimpleNamespace, folds: list[dict[str, Any]]
) -> list[dict[str, str]]:
    """Create small immutable model files and complete 24-coordinate manifests."""
    records: list[dict[str, str]] = []
    variants = (
        ("raw", None),
        ("untrained_neural", None),
        *((geometry, seed) for geometry in ("dense", "sparse") for seed in (13, 29, 47)),
    )
    for fold in folds:
        fold_id = str(fold["id"])
        sequence = config.artifact_root / "sequences" / fold_id / "sequence-manifest.json"
        sequence_hash = file_sha256(sequence)
        for method, seed in variants:
            directory = (
                config.artifact_root
                / "lightgbm"
                / fold_id
                / method
                / str(seed if seed is not None else "shared")
            )
            directory.mkdir(parents=True, exist_ok=True)
            for name, content in (("scale.txt", "fixture-scale"), ("shape.txt", "fixture-shape")):
                (directory / name).write_text(content, encoding="utf-8")
            grid_results = directory / "grid-results.json"
            write_json_atomic(grid_results, {"status": "SYNTHETIC_FIXTURE_ONLY"})
            embedding_hashes: dict[str, str | None] = {"train": None, "validation": None}
            if seed is not None:
                embedding = read_json(
                    config.artifact_root
                    / "embeddings"
                    / fold_id
                    / method
                    / str(seed)
                    / "manifest.json"
                )
                embedding_hashes = {
                    str(item["partition"]): str(item["sha256"])
                    for item in embedding["files"]
                    if item["partition"] in {"train", "validation"}
                }
            manifest = {
                "paper_config_hash": config.config_hash,
                "fold_id": fold_id,
                "method": method,
                "seed": seed,
                "sequence_manifest_hash": sequence_hash,
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
                "embedding_sha256": embedding_hashes,
                "models": [
                    {"path": name, "sha256": file_sha256(directory / name)}
                    for name in ("scale.txt", "shape.txt")
                ],
                "grid_results_sha256": file_sha256(grid_results),
            }
            manifest_path = directory / "manifest.json"
            write_json_atomic(manifest_path, manifest)
            records.append(
                {
                    "path": manifest_path.relative_to(config.artifact_root).as_posix(),
                    "sha256": file_sha256(manifest_path),
                }
            )
    return records


def _write_complete_selection_receipts(
    config: SimpleNamespace, lightgbm_records: list[dict[str, str]]
) -> None:
    """Bind the synthetic model matrix to a canonical parameter/test freeze."""
    selection = config.artifact_root / "selection" / "rdm-lambda.json"
    candidates = []
    for rdm_lambda, error in ((0.1, 1.0), (1.0, 0.5), (10.0, 0.1)):
        for geometry in ("dense", "sparse"):
            checkpoint = (
                config.artifact_root
                / "representations"
                / "fold-1"
                / geometry
                / "13"
                / "final/model.safetensors"
            )
            candidates.append(
                {
                    "rdm_lambda": rdm_lambda,
                    "geometry": geometry,
                    "fold_id": "fold-1",
                    "seed": 13,
                    "observable_probe_error": error,
                    "collapse_gate_status": "PASS",
                    "checkpoint_hash": file_sha256(checkpoint),
                }
            )
    write_json_atomic(
        selection,
        {
            "schema_version": "paper-rdm-lambda-selection-v1",
            "selection_partition": "fold-1/validation",
            "seed": 13,
            "paper_config_hash": config.config_hash,
            "selected_rdm_lambda": 10.0,
            "test_or_tca_used": False,
            "candidates": candidates,
        },
    )
    lightgbm_receipt = config.artifact_root / "lightgbm" / "execution-receipt.json"
    write_json_atomic(lightgbm_receipt, {"status": "SYNTHETIC_FIXTURE_ONLY"})
    freeze = config.artifact_root / "selection" / "parameter-freeze-v1.json"
    write_json_atomic(
        freeze,
        {
            "status": "PARAMETERS_FROZEN",
            "paper_config_hash": config.config_hash,
            "selected_rdm_lambda": 10.0,
            "test_or_tca_used": False,
            "git_commit": _SOURCE_A["commit"],
            "git_tree": _SOURCE_A["tree"],
            "representation_source_commit": "fixture-only",
            "lightgbm_manifests": lightgbm_records,
            "rdm_lambda_receipt_sha256": file_sha256(selection),
            "lightgbm_execution_receipt_sha256": file_sha256(lightgbm_receipt),
        },
    )
    ready = config.artifact_root / "selection" / "locked-test-ready-v1.json"
    write_json_atomic(
        ready,
        {
            "status": "LOCKED-TEST-READY",
            "paper_config_hash": config.config_hash,
            "parameter_freeze_sha256": file_sha256(freeze),
        },
    )
    write_json_atomic(
        config.artifact_root / "selection" / "locked-test-opened-v1.json",
        {
            "status": "LOCKED-TEST-OPENED",
            "paper_config_hash": config.config_hash,
            "locked_test_ready_receipt_sha256": file_sha256(ready),
            "evaluation_git_commit": _ROOT_SOURCE["commit"],
            "evaluation_git_tree": _ROOT_SOURCE["tree"],
        },
    )


def _write_tca_manifest(config: SimpleNamespace) -> Path:
    """Write the TCA aggregate manifest without replacing the sealed execution."""
    root = evaluation_root(config)
    identity = _stage_identity(config, _SOURCE_B)
    manifest = root / "tca" / "manifest.json"
    write_json_atomic(
        manifest,
        {
            "schema_version": "paper-tca-v1",
            "paper_config_hash": config.config_hash,
            "evaluation_identity": identity,
            "files": {
                name: {
                    "path": str(root / f"tca/{name}.parquet"),
                    "sha256": file_sha256(root / f"tca/{name}.parquet"),
                }
                for name in ("main", "sensitivity")
            },
        },
    )
    return manifest


def test_real_inherited_report_and_freeze_preserve_mixed_stage_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> None:
    """A typed A->B recovery publishes a mixed-source report and stable freeze."""
    pytest.importorskip("torch")
    sequence_manifest = request.getfixturevalue("_fixture_regime_corpus")
    config, folds = report_fixture._make_config(tmp_path)
    config.historical_schema_fixture = True
    config.runtime_evaluation_root = tmp_path / "evaluation-executions/source-a"
    _write_selection_receipts(config)
    report_fixture._write_three_fold_sequence_manifests(config, folds, sequence_manifest)

    monkeypatch.setattr(orchestration, "_git_head", lambda: _SOURCE_A["commit"])
    monkeypatch.setattr(orchestration, "_git_tree", lambda: _SOURCE_A["tree"])
    monkeypatch.setattr(report_fixture, "_SOURCE_COMMIT", _SOURCE_A["commit"])
    monkeypatch.setattr(report_fixture, "_SOURCE_TREE", _SOURCE_A["tree"])
    monkeypatch.setattr(orchestration, "_require_parameter_freeze", lambda *_args: {})
    monkeypatch.setattr(orchestration, "_require_locked_test_opened", lambda *_args: {})
    for geometry in ("dense", "sparse"):
        report_fixture.export_fixture_checkpoint(tmp_path, sequence_manifest, geometry)

    # The representation producer itself runs for both geometries.  Remaining
    # configured coordinates are expanded from those real fixture outputs and
    # retain source-A identity; the report tables are never hand-constructed.
    orchestration.evaluate_representations_stage(
        config, full_run_cli_enabled=True, runtime_approval=None
    )
    source_a = config.runtime_evaluation_root
    tca_fixture_root = tmp_path / "tca-production-fixture"
    from test_tca_production_contracts import build_published_tca_rows

    tca_fixture_root.mkdir()
    build_published_tca_rows(tca_fixture_root, include_representation_methods=True)
    config.evaluation["folds"] = folds
    config.representation["seeds"] = [13, 29, 47]
    # Build the complete immutable model inventory before resealing.  The
    # representation evaluation itself above remains the real producer for the
    # retained source-A coordinate; these extra coordinates only make the
    # canonical 18/24 freeze inventory complete.
    _materialize_frozen_representation_matrix(config, folds)
    lightgbm_records = _materialize_frozen_lightgbm_matrix(config, folds)
    _write_complete_selection_receipts(config, lightgbm_records)
    freeze_sha = file_sha256(config.artifact_root / "selection/parameter-freeze-v1.json")
    for geometry in ("dense", "sparse"):
        retained_manifest = (
            source_a
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
    _copy_forecast_inventory(config, source_a, tca_fixture_root)
    _copy_representation_inventory(config, source_a, source_a, folds)
    forecast_path = _merge_forecast_inventory(config, source_a)
    assert forecast_path.is_file()
    retained_base_identity = read_json(source_a / "evaluation-v2/bases/fold-1/manifest.json")[
        "identity"
    ]
    assert all(
        retained_base_identity.get(key) == value
        for key, value in _stage_identity(config, _SOURCE_A).items()
    )

    predecessor = _write_predecessor_execution(config, source_a)
    config.runtime_evaluation_root = tmp_path / "evaluation-executions/source-b"
    source_b = config.runtime_evaluation_root
    monkeypatch.setattr(orchestration, "_git_head", lambda: _SOURCE_B["commit"])
    monkeypatch.setattr(orchestration, "_git_tree", lambda: _SOURCE_B["tree"])
    # TCA is the only fresh numerical stage.  It is published by the real
    # production fixture and receives source-B merge identity.
    monkeypatch.setattr(report_fixture, "_SOURCE_COMMIT", _SOURCE_B["commit"])
    monkeypatch.setattr(report_fixture, "_SOURCE_TREE", _SOURCE_B["tree"])
    tca_fixture_root_b = tmp_path / "tca-production-fixture-b"
    tca_fixture_root_b.mkdir()
    tca_outputs = build_published_tca_rows(tca_fixture_root_b, include_representation_methods=True)

    from execsim.ml.paper.evaluation_execution import (
        STAGE_INHERITED_SCHEMA,
        seal_evaluation_execution,
        write_evaluation_supersession_receipt,
    )
    from execsim.ml.paper.stage_inheritance import write_stage_inheritance_receipt

    # Preserve an intervening failed recovery that inherited these same A bytes.
    # The final B evaluator must supersede M, not pretend A was its predecessor.
    source_middle = tmp_path / "evaluation-executions/source-middle"
    config.runtime_evaluation_root = source_middle
    middle_source = {"commit": "middle-commit", "tree": "middle-tree"}
    middle_inheritance = tmp_path / "middle-inheritance.json"
    middle_supersession = tmp_path / "middle-supersession.json"
    write_stage_inheritance_receipt(
        config,
        superseded_execution=predecessor,
        output=middle_inheritance,
        replacement_source_commit=middle_source["commit"],
        replacement_source_tree=middle_source["tree"],
        reason="fixture first recovery",
    )
    write_evaluation_supersession_receipt(
        config,
        superseded_execution=predecessor,
        output=middle_supersession,
        replacement_source_commit=middle_source["commit"],
        replacement_source_tree=middle_source["tree"],
        reason="fixture first recovery",
    )
    seal_evaluation_execution(
        config,
        source_commit=middle_source["commit"],
        source_tree=middle_source["tree"],
        supersession=middle_supersession,
        inheritance=middle_inheritance,
    )
    predecessor = source_middle / "execution.json"
    middle_bytes = predecessor.read_bytes()
    config.runtime_evaluation_root = source_b

    inheritance = tmp_path / "stage-inheritance.json"
    write_stage_inheritance_receipt(
        config,
        superseded_execution=predecessor,
        output=inheritance,
        replacement_source_commit=_SOURCE_B["commit"],
        replacement_source_tree=_SOURCE_B["tree"],
        reason="synthetic TCA optimizer recovery",
    )
    inheritance_payload = read_json(inheritance)
    assert inheritance_payload["inherited_stages"] == [
        "evaluate-forecast",
        "evaluate-representation",
    ]
    assert set(inheritance_payload["stage_inventory"]) == {
        "evaluate-forecast",
        "evaluate-representation",
    }
    assert inheritance_payload["invalidation_frontier"] == "run-tca"
    assert inheritance_payload["replacement_evaluation_source"] == _SOURCE_B
    supersession = tmp_path / "evaluation-supersession.json"
    write_evaluation_supersession_receipt(
        config,
        superseded_execution=predecessor,
        output=supersession,
        replacement_source_commit=_SOURCE_B["commit"],
        replacement_source_tree=_SOURCE_B["tree"],
        reason="synthetic TCA optimizer recovery",
    )
    sealed = seal_evaluation_execution(
        config,
        source_commit=_SOURCE_B["commit"],
        source_tree=_SOURCE_B["tree"],
        supersession=supersession,
        inheritance=inheritance,
    )
    assert sealed["initial_completed_stages"] == 0
    assert sealed["previous_evaluation_source"] == middle_source
    assert read_json(source_b / "execution.json")["schema_version"] == STAGE_INHERITED_SCHEMA

    tca_main = report_fixture._publish_tca_results(config, "main", tca_outputs["main"])
    tca_sensitivity = report_fixture._publish_tca_results(
        config, "sensitivity", tca_outputs["sensitivity"]
    )
    tca_manifest = _write_tca_manifest(config)
    tca_payload = read_json(tca_manifest)
    assert tca_payload["evaluation_identity"]["source_commit"] == _SOURCE_B["commit"]
    assert tca_main.is_relative_to(source_b)
    assert not (source_a / "tca/main.parquet").exists()

    # Real report builder -> historical bundle writer -> report publication.
    published = orchestration.report_stage(
        config,
        full_run_cli_enabled=True,
        runtime_approval=None,
        historical_schema_fixture=True,
    )
    report_root = source_b / "reports" / config.paper_run_id
    completion_path = report_root / "completion.json"
    completion_bytes = completion_path.read_bytes()
    completion = read_json(completion_path)
    assert published["reuse"] == "created"
    assert (report_root / "REPORT.md").is_file()
    assert read_json(report_root / "provenance.json")["data_classification"] == "synthetic_fixture"

    report_inputs = orchestration._report_input_names(config)
    expected_roots = {
        "evaluation/forecast-results.parquet": source_a,
        "evaluation/representation-accessibility.parquet": source_a,
        "evaluation/representation-date-metrics.parquet": source_a,
        "evaluation/support-regimes.parquet": source_a,
        "tca/main.parquet": source_b,
        "tca/sensitivity.parquet": source_b,
    }
    assert set(completion["identity"]["input_sha256"]) == set(report_inputs)
    for name, digest in completion["identity"]["input_sha256"].items():
        stage = orchestration._report_input_stage(name)
        path = orchestration._report_input_path(config, name)
        assert path.is_relative_to(expected_roots[name])
        assert file_sha256(path) == digest
        assert path == orchestration._report_input_path(config, name)
        assert stage in {"forecast", "representation", "tca"}

    provenance = read_json(report_root / "provenance.json")
    stage_sources = provenance["stage_sources"]
    assert stage_sources["evaluate-forecast"]["inherited"] is True
    assert stage_sources["evaluate-representation"]["inherited"] is True
    assert stage_sources["evaluate-forecast"]["commit"] == _SOURCE_A["commit"]
    assert stage_sources["evaluate-representation"]["commit"] == _SOURCE_A["commit"]
    assert stage_sources["run-tca"]["commit"] == _SOURCE_B["commit"]
    assert stage_sources["report"]["commit"] == _SOURCE_B["commit"]
    assert stage_sources["final-result-freeze"]["commit"] == _SOURCE_B["commit"]
    assert provenance["stage_inheritance_receipt_sha256"] == file_sha256(inheritance)
    assert completion["identity"]["stage_sources"] == stage_sources

    # A changed inherited byte is rejected even though the replacement source
    # identity is unchanged; restore it so the later valid freeze remains a
    # clean resume check.
    inherited = source_a / "evaluation/forecast-results.parquet"
    original_inherited = inherited.read_bytes()
    inherited.write_bytes(original_inherited + b"mutated-inherited-byte")
    with pytest.raises(ValueError, match=r"checksum|changed|mismatch|inventory"):
        orchestration.report_stage(
            config,
            full_run_cli_enabled=True,
            runtime_approval=None,
            historical_schema_fixture=True,
        )
    inherited.write_bytes(original_inherited)

    # The new TCA aggregate manifest is independently bound by final freeze.
    tca_manifest_path = source_b / "tca/manifest.json"
    original_tca_manifest = tca_manifest_path.read_bytes()
    mutated_tca_manifest = read_json(tca_manifest_path)
    mutated_tca_manifest["files"]["main"]["sha256"] = "mutated-tca-manifest"
    write_json_atomic(tca_manifest_path, mutated_tca_manifest)
    with pytest.raises(ValueError, match=r"manifest|identity|checksum"):
        orchestration.write_final_result_freeze(config)
    tca_manifest_path.write_bytes(original_tca_manifest)

    # Valid report/freeze resumes are byte-stable and do not rebuild the report.
    second = orchestration.report_stage(
        config,
        full_run_cli_enabled=True,
        runtime_approval=None,
        historical_schema_fixture=True,
    )
    freeze = orchestration.write_final_result_freeze(config)
    repeated_freeze = orchestration.write_final_result_freeze(config)
    assert second["reuse"] == "validated"
    assert completion_path.is_file()
    assert completion_path.read_bytes() == completion_bytes
    assert freeze["status"] == "FINAL-RESULTS-FROZEN"
    assert freeze["sha256"] == repeated_freeze["sha256"]
    assert freeze["stage_inheritance_receipt_sha256"] == file_sha256(inheritance)
    assert freeze["stage_sources"]["run-tca"]["inherited"] is False
    assert freeze["stage_sources"]["evaluate-forecast"]["inherited"] is True
    assert tca_sensitivity.is_relative_to(source_b)
    assert predecessor.read_bytes() == middle_bytes
