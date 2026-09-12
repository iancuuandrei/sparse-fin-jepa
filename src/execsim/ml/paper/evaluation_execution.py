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
CHAINED_SCHEMA = "paper-evaluation-execution-v3"
SUPERSESSION_SCHEMA = "paper-evaluation-supersession-v1"
SUPERSESSION_STATUS = "SUPERSEDED_RESEALED_EVALUATION"
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


def _nonempty_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


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


def _relative_artifact_path(root: Path, path: Path) -> str:
    """Return a portable artifact-relative path, rejecting escapes."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("Evaluation provenance path must remain inside artifact root.") from exc


def _validate_prior_execution(
    config: PaperRunConfig,
    execution_path: Path,
    *,
    expected_sha256: str | None = None,
    verified_paths: set[Path] | None = None,
) -> dict[str, Any]:
    """Validate an immediately prior execution receipt without copying its outputs."""
    if not execution_path.is_file() or execution_path.name != "execution.json":
        raise ValueError("Superseded evaluation execution receipt is unavailable.")
    actual_sha = file_sha256(execution_path)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise ValueError("Superseded evaluation receipt checksum mismatch.")
    if verified_paths is not None:
        verified_paths.add(execution_path)
    prior = read_json(execution_path)
    if prior.get("status") != "EVALUATION_RESEALED" or prior.get("schema_version") not in {
        SCHEMA,
        CHAINED_SCHEMA,
    }:
        raise ValueError("Superseded receipt is not a resealed evaluation execution.")
    freeze_path = config.artifact_root / "selection" / "parameter-freeze-v1.json"
    if (
        prior.get("protocol_id") != "sparse-jepa-v2"
        or prior.get("paper_config_hash") != config.config_hash
        or prior.get("parameter_freeze_sha256") != file_sha256(freeze_path)
    ):
        raise ValueError("Superseded execution is incompatible with this paper run.")
    opened = read_json(config.artifact_root / "selection" / "locked-test-opened-v1.json")
    root_source = {
        "commit": opened.get("evaluation_git_commit"),
        "tree": opened.get("evaluation_git_tree"),
    }
    prior_root_source = prior.get("root_evaluation_source")
    if prior.get("schema_version") == SCHEMA:
        prior_root_source = prior.get("previous_evaluation_source")
    if prior_root_source != root_source:
        raise ValueError(
            "Superseded execution does not descend from the original TEST authorization."
        )
    if not isinstance(prior.get("evaluation_source"), dict) or not all(
        isinstance(prior["evaluation_source"].get(key), str) and prior["evaluation_source"].get(key)
        for key in ("commit", "tree")
    ):
        raise ValueError("Superseded execution is missing evaluator source identity.")
    upstream_files = prior.get("upstream_files")
    if not isinstance(upstream_files, dict):
        raise ValueError("Superseded execution is missing its immutable inventory.")
    for relative, digest in upstream_files.items():
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise ValueError("Superseded execution inventory is malformed.")
        upstream_path = _safe_child(config.artifact_root, relative)
        if not upstream_path.is_file() or file_sha256(upstream_path) != digest:
            raise ValueError("Superseded execution immutable inventory checksum mismatch.")
        if verified_paths is not None:
            verified_paths.add(upstream_path)
    if prior.get("schema_version") == CHAINED_SCHEMA:
        prior_supersession = prior.get("supersession_path")
        if not isinstance(prior_supersession, str):
            raise ValueError("Superseded chained execution is missing its supersession receipt.")
        prior_supersession_path = _safe_child(config.artifact_root, prior_supersession)
        if not prior_supersession_path.is_file() or prior.get("supersession_sha256") != file_sha256(
            prior_supersession_path
        ):
            raise ValueError("Superseded chained execution supersession checksum mismatch.")
        _validate_chained_supersession(
            config,
            prior_supersession_path,
            replacement_source=prior.get("evaluation_source"),
            verified_paths=verified_paths,
        )
    return prior


def _validate_chained_supersession(
    config: PaperRunConfig,
    supersession_path: Path,
    *,
    replacement_source: dict[str, str] | None = None,
    verified_paths: set[Path] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Validate a typed receipt linking the root authorization to one prior execution."""
    if not supersession_path.resolve().is_relative_to(config.artifact_root.resolve()):
        raise ValueError("Supersession receipt must remain inside the artifact root.")
    receipt = read_json(supersession_path)
    if verified_paths is not None:
        verified_paths.add(supersession_path)
    freeze_path = config.artifact_root / "selection" / "parameter-freeze-v1.json"
    if (
        receipt.get("schema_version") != SUPERSESSION_SCHEMA
        or receipt.get("status") != SUPERSESSION_STATUS
        or receipt.get("protocol_id") != "sparse-jepa-v2"
        or receipt.get("paper_config_hash") != config.config_hash
        or receipt.get("parameter_freeze_sha256") != file_sha256(freeze_path)
    ):
        raise ValueError("Typed evaluation supersession receipt is incompatible.")
    opened_path = config.artifact_root / "selection" / "locked-test-opened-v1.json"
    opened_sha = file_sha256(opened_path)
    if receipt.get("locked_test_open_receipt_sha256") != opened_sha:
        raise ValueError("Typed supersession does not bind the original TEST-open receipt.")
    opened = read_json(opened_path)
    root_source = {
        "commit": opened.get("evaluation_git_commit"),
        "tree": opened.get("evaluation_git_tree"),
    }
    if receipt.get("root_evaluation_source") != root_source:
        raise ValueError("Typed supersession root evaluator identity mismatch.")
    namespace = receipt.get("superseded_execution_namespace")
    if not isinstance(namespace, str) or not namespace:
        raise ValueError("Typed supersession is missing the prior evaluation namespace.")
    namespace_path = (config.artifact_root / namespace).resolve()
    execution_root = (config.artifact_root / "evaluation-executions").resolve()
    if not namespace_path.is_relative_to(execution_root):
        raise ValueError("Superseded evaluation namespace escapes evaluation-executions.")
    prior_path = namespace_path / "execution.json"
    prior = _validate_prior_execution(
        config,
        prior_path,
        expected_sha256=receipt.get("superseded_execution_receipt_sha256"),
        verified_paths=verified_paths,
    )
    if receipt.get("superseded_evaluation_source") != prior.get("evaluation_source"):
        raise ValueError("Typed supersession prior evaluator identity mismatch.")
    if (
        replacement_source is not None
        and receipt.get("replacement_evaluation_source") != replacement_source
    ):
        raise ValueError("Typed supersession replacement evaluator identity mismatch.")
    if not isinstance(receipt.get("reason"), str) or not receipt["reason"].strip():
        raise ValueError("Typed supersession requires a non-empty classification reason.")
    replacement = receipt.get("replacement_evaluation_source")
    if not isinstance(replacement, dict) or not all(
        _nonempty_text(replacement.get(key)) for key in ("commit", "tree")
    ):
        raise ValueError("Typed supersession is missing replacement evaluator identity.")
    return receipt, prior, prior_path


def write_evaluation_supersession_receipt(
    config: PaperRunConfig,
    *,
    superseded_execution: Path,
    output: Path,
    replacement_source_commit: str,
    replacement_source_tree: str,
    reason: str,
) -> dict[str, Any]:
    """Create a durable, typed receipt for superseding a resealed execution."""
    root = config.artifact_root.resolve()
    if not all(
        _nonempty_text(value) for value in (replacement_source_commit, replacement_source_tree)
    ):
        raise ValueError("Replacement evaluator identity must contain non-empty commit and tree.")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Supersession requires a non-empty classification reason.")
    if not output.resolve().is_relative_to(root):
        raise ValueError("Supersession receipt must remain inside the artifact root.")
    namespace_path = superseded_execution.resolve()
    if namespace_path.name == "execution.json":
        namespace_path = namespace_path.parent
    namespace = _relative_artifact_path(root, namespace_path)
    if not namespace.startswith("evaluation-executions/"):
        raise ValueError("Superseded execution must live below evaluation-executions.")
    prior_path = namespace_path / "execution.json"
    prior = _validate_prior_execution(config, prior_path)
    opened_path = root / "selection" / "locked-test-opened-v1.json"
    freeze_path = root / "selection" / "parameter-freeze-v1.json"
    opened = read_json(opened_path)
    payload: dict[str, Any] = {
        "schema_version": SUPERSESSION_SCHEMA,
        "status": SUPERSESSION_STATUS,
        "protocol_id": "sparse-jepa-v2",
        "paper_config_hash": config.config_hash,
        "parameter_freeze_sha256": file_sha256(freeze_path),
        "locked_test_open_receipt_sha256": file_sha256(opened_path),
        "root_evaluation_source": {
            "commit": opened["evaluation_git_commit"],
            "tree": opened["evaluation_git_tree"],
        },
        "superseded_execution_receipt_sha256": file_sha256(prior_path),
        "superseded_execution_namespace": namespace,
        "superseded_evaluation_source": prior["evaluation_source"],
        "replacement_evaluation_source": {
            "commit": replacement_source_commit,
            "tree": replacement_source_tree,
        },
        "reason": reason,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    if output.exists():
        existing = read_json(output)
        if {key: value for key, value in existing.items() if key != "created_at_utc"} != {
            key: value for key, value in payload.items() if key != "created_at_utc"
        }:
            raise ValueError("Existing supersession receipt is incompatible.")
        return existing
    write_json_atomic(output, payload)
    return {**payload, "path": str(output), "sha256": file_sha256(output)}


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
    representation_source_commit = freeze.get("representation_source_commit")
    if (
        not isinstance(representation_source_commit, str)
        or not representation_source_commit.strip()
    ):
        raise ValueError("Parameter-selection freeze is missing representation source identity.")
    representation_source_commit = representation_source_commit.strip()
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
        sequence_path = root / "sequences" / fold_id / "sequence-manifest.json"
        sequence = bind(sequence_path)
        sequence_key = Path(os.path.abspath(sequence_path)).relative_to(root).as_posix()
        sequence_hash = inventory[sequence_key]
        sequence_universe_hash = sequence.get("universe_manifest_hash")
        if not isinstance(sequence_universe_hash, str) or not sequence_universe_hash:
            raise ValueError("Frozen sequence manifest is missing universe identity.")
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
                or manifest.get("sequence_manifest_hash") != sequence_hash
            ):
                raise ValueError("LightGBM coordinate or sequence identity mismatch.")
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
            if checkpoint.get("sequence_manifest_hash") != sequence_hash:
                raise ValueError("Frozen JEPA checkpoint sequence identity mismatch.")
            if checkpoint.get("universe_manifest_hash") != sequence_universe_hash:
                raise ValueError("Frozen JEPA checkpoint universe identity mismatch.")
            if checkpoint.get("code_commit") != representation_source_commit:
                raise ValueError("Frozen JEPA checkpoint source commit mismatch.")
            if export.get("sequence_manifest_hash") != sequence_hash:
                raise ValueError("Frozen embedding export sequence identity mismatch.")
            if export.get("normalization_hash") != checkpoint.get("normalization_hash"):
                raise ValueError("Frozen embedding export normalization identity mismatch.")
            if export.get("paper_config_hash") != config.config_hash:
                raise ValueError("Frozen embedding export paper configuration mismatch.")
            bind(rep / "final/model.safetensors", checkpoint["weights_sha256"])
            compatibility = bind(rep / "compatibility.json")
            if compatibility.get("sequence_manifest_hash") != sequence_hash:
                raise ValueError("Frozen JEPA compatibility sequence identity mismatch.")
            if len(export["files"]) != 3 or {item["partition"] for item in export["files"]} != {
                "train",
                "validation",
                "test",
            }:
                raise ValueError("Embedding export partition inventory mismatch.")
            embedding_hashes: dict[str, str] = {}
            for item in export["files"]:
                partition = item["partition"]
                path = _safe_child(embedding, item["path"])
                bind(path, item["sha256"])
                relative_path = Path(os.path.abspath(path)).relative_to(root).as_posix()
                embedding_hashes[partition] = inventory[relative_path]
            recorded_embeddings = manifest.get("embedding_sha256")
            if not isinstance(recorded_embeddings, dict) or any(
                recorded_embeddings.get(partition) != embedding_hashes.get(partition)
                for partition in ("train", "validation")
            ):
                raise ValueError("Frozen LightGBM embedding checksum mismatch.")
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
    freeze_path = config.artifact_root / "selection/parameter-freeze-v1.json"
    prior = read_json(supersession)
    chained = prior.get("schema_version") == SUPERSESSION_SCHEMA
    if chained:
        supersession_receipt, superseded, superseded_path = _validate_chained_supersession(
            config,
            supersession,
            replacement_source={"commit": source_commit, "tree": source_tree},
        )
        root_source = supersession_receipt["root_evaluation_source"]
        previous_source = superseded["evaluation_source"]
    else:
        if prior.get("status") != "SUPERSEDED_PRE_OPTIMIZATION_TEST" or prior.get(
            "parameter_freeze_sha256"
        ) != file_sha256(freeze_path):
            raise ValueError("Supersession receipt does not bind the original parameter freeze.")
        supersession_receipt = None
        superseded_path = None
        root_source = None
        previous_source = None
    inventory = frozen_inventory(config)
    freeze = read_json(freeze_path)
    opened = read_json(config.artifact_root / "selection/locked-test-opened-v1.json")
    identity = {
        "schema_version": CHAINED_SCHEMA if chained else SCHEMA,
        "status": "EVALUATION_RESEALED",
        "protocol_id": "sparse-jepa-v2",
        "paper_config_hash": config.config_hash,
        "parameter_freeze_sha256": file_sha256(freeze_path),
        "upstream_parameter_source": {"commit": freeze["git_commit"], "tree": freeze["git_tree"]},
        "previous_evaluation_source": previous_source
        or {
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
    if chained and supersession_receipt is not None and superseded_path is not None:
        identity.update(
            {
                "root_evaluation_source": root_source,
                "root_evaluation_open_receipt_sha256": supersession_receipt[
                    "locked_test_open_receipt_sha256"
                ],
                "superseded_execution_receipt_sha256": supersession_receipt[
                    "superseded_execution_receipt_sha256"
                ],
                "superseded_execution_namespace": supersession_receipt[
                    "superseded_execution_namespace"
                ],
                "supersession_path": _relative_artifact_path(
                    config.artifact_root.resolve(), supersession.resolve()
                ),
            }
        )
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
    base_valid = (
        payload.get("status") == "EVALUATION_RESEALED"
        and payload.get("evaluation_source") == {"commit": source_commit, "tree": source_tree}
        and payload.get("paper_config_hash") == config.config_hash
        and payload.get("artifact_schemas") == ARTIFACT_SCHEMAS
        and payload.get("preservation") == SCIENTIFIC_PRESERVATION
        and payload.get("initial_completed_stages") == 0
        and payload.get("parameter_freeze_sha256")
        == file_sha256(config.artifact_root / "selection/parameter-freeze-v1.json")
        and payload.get("upstream_parameter_source")
        == {"commit": freeze["git_commit"], "tree": freeze["git_tree"]}
        and payload.get("upstream_files") == frozen_inventory(config)
    )
    if payload.get("schema_version") == SCHEMA:
        valid = base_valid and payload.get("previous_evaluation_source") == {
            "commit": opened["evaluation_git_commit"],
            "tree": opened["evaluation_git_tree"],
        }
        extra_paths: list[Path] = []
    elif payload.get("schema_version") == CHAINED_SCHEMA:
        supersession_name = payload.get("supersession_path")
        if not isinstance(supersession_name, str):
            # The path is derived from the immutable receipt hash and remains
            # explicit in the execution identity for portable verification.
            supersession_name = None
        supersession_path = config.artifact_root / supersession_name if supersession_name else None
        valid = base_valid and isinstance(supersession_path, Path) and supersession_path.is_file()
        extra_paths = []
        if valid and supersession_path is not None:
            try:
                verified_provenance: set[Path] = set()
                typed, prior, _prior_path = _validate_chained_supersession(
                    config,
                    supersession_path,
                    replacement_source={"commit": source_commit, "tree": source_tree},
                    verified_paths=verified_provenance,
                )
                valid = valid and (
                    payload.get("previous_evaluation_source") == prior["evaluation_source"]
                    and payload.get("root_evaluation_source") == typed["root_evaluation_source"]
                    and payload.get("root_evaluation_open_receipt_sha256")
                    == typed["locked_test_open_receipt_sha256"]
                    and payload.get("superseded_execution_receipt_sha256")
                    == typed["superseded_execution_receipt_sha256"]
                    and payload.get("superseded_execution_namespace")
                    == typed["superseded_execution_namespace"]
                    and payload.get("supersession_sha256") == file_sha256(supersession_path)
                )
                extra_paths = sorted(verified_provenance)
            except (OSError, TypeError, ValueError):
                valid = False
        else:
            valid = False
    else:
        valid = False
        extra_paths = []
    if not valid:
        raise ValueError(
            "Resealed evaluation source, configuration, or upstream identity mismatch."
        )
    paths = [receipt_path, *(root / name for name in payload["upstream_files"]), *extra_paths]
    _VERIFIED_EXECUTIONS[cache_key] = (
        deepcopy(payload),
        {path: _file_state(path) for path in paths},
    )
    return payload
