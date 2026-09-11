"""Reseal an evaluator against immutable model evidence, without rebuilding its freeze."""

from __future__ import annotations

import os
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Any

from execsim.data.paper.manifests import file_sha256, read_json, stable_hash, write_json_atomic
from execsim.ml.paper.configs import PaperRunConfig

SCHEMA = "paper-evaluation-execution-v2"
SCIENTIFIC_PRESERVATION = [
    "MODEL PARAMETERS UNCHANGED",
    "ESTIMANDS UNCHANGED",
    "TEST PARTITIONS UNCHANGED",
    "EVALUATION IMPLEMENTATION OPTIMIZED",
    "PREVIOUS PARTIAL TEST SUPERSEDED",
]
ARTIFACT_SCHEMAS = {
    "base": "paper-evaluation-base-v2",
    "learned": "paper-forecast-ledger-v2",
    "ewma": "paper-ewma-ledger-v4",
    "tca": "paper-tca-date-shard-v3",
}


_VERIFIED_EXECUTIONS: dict[str, tuple[dict[str, Any], dict[Path, tuple[int, int, int, int]]]] = {}


def _file_state(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino


def evaluation_root(config: PaperRunConfig) -> Path:
    """Keep new execution outputs separate from every immutable upstream artifact."""
    return getattr(config, "runtime_evaluation_root", None) or config.artifact_root


def evaluation_report_root(config: PaperRunConfig) -> Path:
    """Isolate reports as well as numerical results when an execution is resealed."""
    if getattr(config, "runtime_evaluation_root", None) is not None:
        return evaluation_root(config) / "reports"
    return config.report_root


def representation_root(config: PaperRunConfig) -> Path:
    """Resolve an immutable evaluation-only checkpoint import without altering old copies."""
    return (
        getattr(config, "runtime_representation_root", None)
        or config.artifact_root / "representations"
    )


def _safe_child(root: Path, name: str) -> Path:
    if Path(name).is_absolute() or PureWindowsPath(name).is_absolute():
        raise ValueError("Upstream artifact paths must be relative.")
    result = (root / name).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError("Upstream artifact path escapes its manifest directory.")
    # Preserve the logical artifact path: upstream exports may be directory junctions.
    return Path(os.path.abspath(root / name))


def _validate_primary_jepa_source_uniformity(
    code_commits: list[object], *, expected_count: int
) -> None:
    """Require the configured primary JEPA matrix to come from one source commit."""
    if len(code_commits) != expected_count:
        raise ValueError(
            "Frozen JEPA primary inventory must contain exactly "
            f"{expected_count} final manifests; found {len(code_commits)}."
        )
    normalized = []
    for value in code_commits:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("Frozen JEPA final manifest is missing a non-empty code_commit.")
        normalized.append(value.strip())
    if len(set(normalized)) != 1:
        raise ValueError("Frozen JEPA primary final manifests must have one shared code_commit.")


def frozen_inventory(config: PaperRunConfig) -> dict[str, str]:
    """Hash the exact configured model matrix and verify its original checksum links."""
    root = config.artifact_root.resolve()
    inventory: dict[str, str] = {}

    def bind(path: Path, expected: str | None = None) -> dict[str, Any]:
        if not path.is_file():
            raise RuntimeError(f"BLOCKED: immutable upstream artifact missing: {path}")
        digest = file_sha256(path)
        if expected is not None and digest != expected:
            raise ValueError(f"Immutable upstream checksum mismatch: {path}")
        inventory[Path(os.path.abspath(path)).relative_to(root).as_posix()] = digest
        return read_json(path) if path.suffix == ".json" else {}

    freeze = bind(root / "selection/parameter-freeze-v1.json")
    if (
        freeze.get("status") != "PARAMETERS_FROZEN"
        or freeze.get("paper_config_hash") != config.config_hash
        or freeze.get("selected_rdm_lambda") != 10.0
        or freeze.get("test_or_tca_used") is not False
    ):
        raise ValueError("Reseal requires the unchanged frozen scientific parameters.")
    bind(root / "selection/rdm-lambda.json", freeze["rdm_lambda_receipt_sha256"])
    bind(root / "lightgbm/execution-receipt.json", freeze["lightgbm_execution_receipt_sha256"])
    opened = bind(root / "selection/locked-test-opened-v1.json")
    ready = bind(
        root / "selection/locked-test-ready-v1.json", opened["locked_test_ready_receipt_sha256"]
    )
    if (
        opened.get("status") != "LOCKED-TEST-OPENED"
        or opened.get("paper_config_hash") != config.config_hash
        or ready.get("paper_config_hash") != config.config_hash
        or ready.get("parameter_freeze_sha256") != inventory["selection/parameter-freeze-v1.json"]
    ):
        raise ValueError("Original TEST authorization identity mismatch.")
    records = freeze["lightgbm_manifests"]
    recorded = {record["path"]: record["sha256"] for record in records}
    expected_paths = set()
    primary_jepa_code_commits: list[object] = []
    configured_primary_count = sum(
        1
        for _fold in config.evaluation["folds"]
        for _geometry in ("dense", "sparse")
        for _seed in config.representation["seeds"]
    )
    for fold in config.evaluation["folds"]:
        fold_id = str(fold["id"])
        bind(root / "sequences" / fold_id / "sequence-manifest.json")
        variants: list[tuple[str, int | None]] = [("raw", None), ("untrained_neural", None)]
        variants.extend(
            (geometry, int(seed))
            for geometry in ("dense", "sparse")
            for seed in config.representation["seeds"]
        )
        for method, seed in variants:
            relative = f"lightgbm/{fold_id}/{method}/{seed or 'shared'}/manifest.json"
            expected_paths.add(relative)
            if relative not in recorded:
                raise ValueError("Frozen LightGBM matrix is missing a configured coordinate.")
            model_path = root / relative
            manifest = bind(model_path, recorded[relative])
            if (
                manifest.get("paper_config_hash") != config.config_hash
                or manifest.get("method") != method
                or manifest.get("seed") != seed
            ):
                raise ValueError("LightGBM coordinate identity mismatch.")
            if len(manifest["models"]) != 2 or {
                record["path"] for record in manifest["models"]
            } != {"scale.txt", "shape.txt"}:
                raise ValueError("Frozen LightGBM booster inventory mismatch.")
            for record in manifest["models"]:
                bind(_safe_child(model_path.parent, record["path"]), record["sha256"])
            bind(model_path.parent / "grid-results.json", manifest["grid_results_sha256"])
            if seed is None:
                continue
            rep = representation_root(config).resolve() / fold_id / method / str(seed)
            embedding = root / "embeddings" / fold_id / method / str(seed)
            export = bind(embedding / "manifest.json")
            checkpoint = bind(rep / "final/manifest.json", export["checkpoint_manifest_hash"])
            primary_jepa_code_commits.append(checkpoint.get("code_commit"))
            if (
                checkpoint.get("geometry") != method
                or checkpoint.get("seed") != seed
                or checkpoint.get("fold_id") != fold_id
                or checkpoint.get("paper_config_hash") != config.config_hash
                or checkpoint.get("calibrated_rdm_lambda") != 10.0
                or export.get("checkpoint_hash") != checkpoint.get("weights_sha256")
            ):
                raise ValueError("Frozen JEPA/export coordinate identity mismatch.")
            bind(rep / "final/model.safetensors", checkpoint["weights_sha256"])
            bind(rep / "compatibility.json")
            if len(export["files"]) != 3 or {item["partition"] for item in export["files"]} != {
                "train",
                "validation",
                "test",
            }:
                raise ValueError("Embedding export partition inventory mismatch.")
            for item in export["files"]:
                bind(_safe_child(embedding, item["path"]), item["sha256"])
    if len(recorded) != len(records) or set(recorded) != expected_paths:
        raise ValueError("Frozen LightGBM matrix contains extra or duplicated coordinates.")
    _validate_primary_jepa_source_uniformity(
        primary_jepa_code_commits, expected_count=configured_primary_count
    )
    return dict(sorted(inventory.items()))


def seal_evaluation_execution(
    config: PaperRunConfig,
    *,
    source_commit: str,
    source_tree: str,
    supersession: Path,
) -> dict[str, Any]:
    """Create a new, empty, source-bound execution; never overwrite upstream receipts."""
    directory = evaluation_root(config).resolve()
    if directory == config.artifact_root.resolve() or not directory.is_relative_to(
        config.artifact_root.resolve() / "evaluation-executions"
    ):
        raise ValueError("New evaluation root must be inside artifact-root/evaluation-executions/.")
    prior = read_json(supersession)
    freeze_path = config.artifact_root / "selection/parameter-freeze-v1.json"
    if prior.get("status") != "SUPERSEDED_PRE_OPTIMIZATION_TEST" or prior.get(
        "parameter_freeze_sha256"
    ) != file_sha256(freeze_path):
        raise ValueError("Supersession receipt does not bind the original parameter freeze.")
    inventory = frozen_inventory(config)
    freeze = read_json(freeze_path)
    opened = read_json(config.artifact_root / "selection/locked-test-opened-v1.json")
    identity = {
        "schema_version": SCHEMA,
        "status": "EVALUATION_RESEALED",
        "protocol_id": "sparse-jepa-v2",
        "paper_config_hash": config.config_hash,
        "parameter_freeze_sha256": file_sha256(freeze_path),
        "upstream_parameter_source": {"commit": freeze["git_commit"], "tree": freeze["git_tree"]},
        "previous_evaluation_source": {
            "commit": opened["evaluation_git_commit"],
            "tree": opened["evaluation_git_tree"],
        },
        "evaluation_source": {"commit": source_commit, "tree": source_tree},
        "supersession_sha256": file_sha256(supersession),
        "artifact_schemas": ARTIFACT_SCHEMAS,
        "preservation": SCIENTIFIC_PRESERVATION,
        "upstream_files": inventory,
        "initial_completed_stages": 0,
    }
    receipt = directory / "execution.json"
    if receipt.exists():
        existing = read_json(receipt)
        if {key: value for key, value in existing.items() if key != "sealed_at_utc"} != identity:
            raise ValueError("Existing evaluation execution is incompatible.")
        return existing
    if directory.exists() and any(directory.iterdir()):
        raise ValueError("New authoritative evaluation namespace must start empty.")
    payload = {**identity, "sealed_at_utc": datetime.now(UTC).isoformat()}
    write_json_atomic(receipt, payload)
    return payload


def verify_evaluation_execution(
    config: PaperRunConfig, *, source_commit: str, source_tree: str
) -> dict[str, Any]:
    """Require the original freeze and every model byte to match the resealed source."""
    root = config.artifact_root.resolve()
    receipt_path = evaluation_root(config) / "execution.json"
    cache_key = stable_hash(
        {
            "root": str(root),
            "receipt": str(receipt_path.absolute()),
            "representation_root": str(representation_root(config).absolute()),
            "commit": source_commit,
            "tree": source_tree,
            "config_hash": config.config_hash,
            "sections": getattr(
                config,
                "sections",
                {"evaluation": config.evaluation, "representation": config.representation},
            ),
        }
    )
    if cache_key in _VERIFIED_EXECUTIONS:
        cached, states = _VERIFIED_EXECUTIONS[cache_key]
        if any(_file_state(path) != state for path, state in states.items()):
            raise ValueError("Resealed checksum-bound upstream files changed during execution.")
        return deepcopy(cached)
    payload = read_json(evaluation_root(config) / "execution.json")
    freeze = read_json(config.artifact_root / "selection/parameter-freeze-v1.json")
    opened = read_json(config.artifact_root / "selection/locked-test-opened-v1.json")
    if (
        payload.get("schema_version") != SCHEMA
        or payload.get("status") != "EVALUATION_RESEALED"
        or payload.get("evaluation_source") != {"commit": source_commit, "tree": source_tree}
        or payload.get("paper_config_hash") != config.config_hash
        or payload.get("artifact_schemas") != ARTIFACT_SCHEMAS
        or payload.get("preservation") != SCIENTIFIC_PRESERVATION
        or payload.get("initial_completed_stages") != 0
        or payload.get("parameter_freeze_sha256")
        != file_sha256(config.artifact_root / "selection/parameter-freeze-v1.json")
        or payload.get("upstream_parameter_source")
        != {"commit": freeze["git_commit"], "tree": freeze["git_tree"]}
        or payload.get("previous_evaluation_source")
        != {"commit": opened["evaluation_git_commit"], "tree": opened["evaluation_git_tree"]}
        or payload.get("upstream_files") != frozen_inventory(config)
    ):
        raise ValueError(
            "Resealed evaluation source, configuration, or upstream identity mismatch."
        )
    paths = [receipt_path, *(root / name for name in payload["upstream_files"])]
    _VERIFIED_EXECUTIONS[cache_key] = (
        deepcopy(payload),
        {path: _file_state(path) for path in paths},
    )
    return payload
