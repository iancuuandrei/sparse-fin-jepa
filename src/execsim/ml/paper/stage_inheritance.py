"""Typed, read-only inheritance of completed paper-evaluation stages.

The stage receipt is deliberately small and explicit.  It is not a workflow
engine: v1 and v2 can inherit the forecast and representation stages only, and
the invalidation frontier is ``run-tca``.  v1 binds those stages to its
immediate predecessor.  v2 is used when that predecessor already inherited
the stages and therefore records the immediate predecessor separately from
the immutable stage producer.  v3 is a separately typed report-only recovery:
it can additionally inherit a complete TCA stage, with the report as the
invalidation frontier.  Every receipt and artifact consumed from the chain
remains checksum-bound; no output is copied into the replacement namespace.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path, PureWindowsPath
from typing import Any

from execsim.data.paper.manifests import file_sha256, read_json, stable_hash, write_json_atomic

SCHEMA = "paper-evaluation-stage-inheritance-v1"
SCHEMA_V2 = "paper-evaluation-stage-inheritance-v2"
SCHEMA_V3 = "paper-evaluation-stage-inheritance-v3"
SUPPORTED_SCHEMAS = (SCHEMA, SCHEMA_V2, SCHEMA_V3)
STATUS = "STAGE_INHERITANCE_VALIDATED"
EXECUTION_SCHEMA = {
    "paper-evaluation-execution-v2",
    "paper-evaluation-execution-v3",
    "paper-evaluation-execution-v4",
}
FRONTIER = "run-tca"
REPORT_FRONTIER = "report"
INHERITED_STAGES = ("evaluate-forecast", "evaluate-representation")
REPORT_INHERITED_STAGES = (*INHERITED_STAGES, "run-tca")
_STAGE_ALIASES = {
    "forecast": "evaluate-forecast",
    "evaluate-forecast": "evaluate-forecast",
    "evaluate_forecast": "evaluate-forecast",
    "representation": "evaluate-representation",
    "evaluate-representation": "evaluate-representation",
    "evaluate_representation": "evaluate-representation",
    "tca": "run-tca",
    "run-tca": "run-tca",
    "run_tca": "run-tca",
    "report": "report",
    "final-result-freeze": "final-result-freeze",
    "final_result_freeze": "final-result-freeze",
}


def _nonempty(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _canonical_stage(stage: str) -> str:
    try:
        return _STAGE_ALIASES[stage]
    except KeyError as exc:
        raise ValueError(f"Unknown paper evaluation stage: {stage}") from exc


def _evaluation_root(config: Any) -> Path:
    return Path(getattr(config, "runtime_evaluation_root", None) or config.artifact_root)


def _artifact_root(config: Any) -> Path:
    return Path(config.artifact_root).resolve()


def _safe_relative(root: Path, name: str) -> Path:
    """Resolve a manifest member without allowing an absolute or escaping path."""
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("Stage inheritance paths must be non-empty relative paths.")
    if PureWindowsPath(name).is_absolute() or PureWindowsPath(name).drive:
        raise ValueError("Stage inheritance paths must be portable relative paths.")
    candidate = Path(name)
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise ValueError("Stage inheritance paths contain an unsafe component.")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Stage inheritance path escapes its source root.")
    return root / candidate


def _relative(root: Path, path: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError("Stage inheritance artifact is outside its source root.") from exc


def _state(path: Path) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino


def _source(value: object, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping) or not all(
        _nonempty(value.get(key)) for key in ("commit", "tree")
    ):
        raise ValueError(f"{label} evaluator identity is incomplete.")
    return {"commit": str(value["commit"]), "tree": str(value["tree"])}


def _freeze_path(config: Any) -> Path:
    return _artifact_root(config) / "selection" / "parameter-freeze-v1.json"


def _opened_path(config: Any) -> Path:
    return _artifact_root(config) / "selection" / "locked-test-opened-v1.json"


def _current_metadata(config: Any) -> tuple[str, str, dict[str, str], str]:
    freeze_path = _freeze_path(config)
    opened_path = _opened_path(config)
    if not freeze_path.is_file() or not opened_path.is_file():
        raise ValueError("Stage inheritance requires the parameter freeze and TEST-open receipt.")
    freeze = read_json(freeze_path)
    opened = read_json(opened_path)
    if freeze.get("paper_config_hash") != config.config_hash:
        raise ValueError("Stage inheritance parameter freeze configuration mismatch.")
    if opened.get("paper_config_hash") != config.config_hash:
        raise ValueError("Stage inheritance TEST-open configuration mismatch.")
    root_source = _source(
        {"commit": opened.get("evaluation_git_commit"), "tree": opened.get("evaluation_git_tree")},
        label="Root",
    )
    return file_sha256(freeze_path), file_sha256(opened_path), root_source, str(config.config_hash)


def _check_parquet_metadata(path: Path, metadata: Mapping[str, Any]) -> None:
    """Check receipt metadata without inspecting result rows."""
    if "rows" not in metadata or "schema" not in metadata:
        raise ValueError(f"Completed artifact metadata is incomplete: {path}")
    try:
        import pyarrow.parquet as parquet

        actual = parquet.ParquetFile(path)
        if (
            actual.metadata.num_rows != metadata["rows"]
            or str(actual.schema_arrow) != metadata["schema"]
        ):
            raise ValueError(f"Completed artifact schema or row-count mismatch: {path}")
    except ImportError as exc:  # pragma: no cover - pyarrow is a core paper dependency
        raise RuntimeError(
            "BLOCKED: pyarrow is required to verify inherited Parquet artifacts."
        ) from exc


def _artifact_files(
    root: Path,
    directory: Path,
    *,
    schema: str,
    expected_names: set[str] | None = None,
    expected_identity: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    """Verify one publish_frames artifact and return its complete byte inventory."""
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Inherited stage manifest is missing: {manifest_path}")
    manifest = read_json(manifest_path)
    identity = manifest.get("identity")
    if not isinstance(identity, Mapping) or identity.get("schema_version") != schema:
        raise ValueError(f"Inherited stage manifest schema is incompatible: {manifest_path}")
    if expected_identity is not None:
        for key, value in expected_identity.items():
            if identity.get(key) != value:
                raise ValueError(f"Inherited stage manifest identity mismatch: {manifest_path}")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError(f"Inherited stage manifest is incomplete: {manifest_path}")
    names = set(files)
    if expected_names is not None and names != expected_names:
        raise ValueError(f"Inherited stage file inventory mismatch: {manifest_path}")
    children = list(directory.iterdir())
    if any(child.is_dir() for child in children) or {
        child.name for child in children if child.is_file()
    } != {"manifest.json", *names}:
        raise ValueError(f"Inherited stage directory contains undeclared members: {directory}")
    inventory = {_relative(root, manifest_path): file_sha256(manifest_path)}
    for name, metadata in files.items():
        if not isinstance(name, str) or not isinstance(metadata, Mapping):
            raise ValueError(f"Inherited stage manifest member is malformed: {manifest_path}")
        path = _safe_relative(directory, name)
        if path.suffix != ".parquet" or not path.is_file() or not _nonempty(metadata.get("sha256")):
            raise ValueError(f"Inherited stage manifest member is missing: {path}")
        if file_sha256(path) != metadata["sha256"]:
            raise ValueError(f"Inherited stage artifact checksum mismatch: {path}")
        _check_parquet_metadata(path, metadata)
        inventory[_relative(root, path)] = str(metadata["sha256"])
    return inventory


def _merged_file(
    root: Path,
    directory: Path,
    name: str,
    expected_schema: str,
    *,
    expected_identity: Mapping[str, Any] | None = None,
    expected_sources: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Verify one merge output and its completion receipt."""
    output = directory / name
    receipt_path = output.with_suffix(".manifest.json")
    if not output.is_file() or not receipt_path.is_file():
        raise ValueError(f"Inherited merged stage output is incomplete: {output}")
    receipt = read_json(receipt_path)
    if receipt.get("schema_version") != expected_schema or receipt.get(
        "parquet_sha256"
    ) != file_sha256(output):
        raise ValueError(f"Inherited merged stage receipt identity/checksum mismatch: {output}")
    merge_identity = receipt.get("merge_identity")
    if not isinstance(merge_identity, Mapping):
        raise ValueError(f"Inherited merged stage receipt is incomplete: {receipt_path}")
    if expected_identity is not None and any(
        merge_identity.get(key) != value for key, value in expected_identity.items()
    ):
        raise ValueError(f"Inherited merged stage config/source identity mismatch: {output}")
    if expected_sources is not None:
        sources = merge_identity.get("sources")
        if not isinstance(sources, Mapping) or dict(sources) != dict(expected_sources):
            raise ValueError(f"Inherited merged stage source inventory mismatch: {output}")
    _check_parquet_metadata(output, receipt)
    return {
        _relative(root, output): file_sha256(output),
        _relative(root, receipt_path): file_sha256(receipt_path),
    }


def _merge_identity_source(path: Path) -> dict[str, str]:
    payload = read_json(path)
    return _source(
        {
            "commit": payload.get("merge_identity", {}).get("source_commit"),
            "tree": payload.get("merge_identity", {}).get("source_tree"),
        },
        label="Merged stage",
    )


def _execution_stage_receipt(
    config: Any,
    execution_path: Path,
    execution: Mapping[str, Any],
    *,
    source: dict[str, str],
) -> tuple[dict[str, Any], Path, str] | None:
    """Verify and return an execution's stage receipt, when it has one.

    A v4 execution can itself be the immediate predecessor of another
    recovery.  Keeping this link in one helper avoids accidentally treating
    its evaluator source as the producer of inherited stages.
    """
    if execution.get("schema_version") != "paper-evaluation-execution-v4":
        return None
    relative = execution.get("stage_inheritance_path")
    digest = execution.get("stage_inheritance_sha256")
    receipt_digest = execution.get("stage_inheritance_receipt_sha256")
    if receipt_digest is not None and receipt_digest != digest:
        raise ValueError("v4 stage-inheritance receipt aliases disagree.")
    if not isinstance(relative, str) or not _nonempty(digest):
        raise ValueError("v4 execution is missing its typed stage-inheritance receipt.")
    root = _artifact_root(config)
    receipt_path = _safe_relative(root, relative)
    if not receipt_path.is_file() or file_sha256(receipt_path) != digest:
        raise ValueError("v4 stage-inheritance receipt checksum mismatch.")
    predecessor_namespace = execution.get("superseded_execution_namespace")
    predecessor_sha = execution.get("superseded_execution_receipt_sha256")
    if not isinstance(predecessor_namespace, str) or not isinstance(predecessor_sha, str):
        raise ValueError("v4 execution is missing its predecessor identity.")
    typed = verify_stage_inheritance(
        config,
        receipt_path,
        replacement_source=source,
        expected_predecessor=(predecessor_namespace, predecessor_sha),
    )
    return typed, receipt_path, str(digest)


def _stage_producer_descriptor(
    root: Path, payload: Mapping[str, Any], stage: str
) -> tuple[str, str, dict[str, str]]:
    """Return the immutable producer identity for one inherited stage.

    v1 records intentionally retain their original fields.  v2 uses distinct
    ``producer_*`` fields so the v1 direct-predecessor meaning is not silently
    repurposed for a chained receipt.
    """
    inventories = payload.get("stage_inventory")
    if not isinstance(inventories, Mapping):
        raise ValueError("Stage inheritance stage inventory is incomplete.")
    record = inventories.get(stage)
    if not isinstance(record, Mapping):
        raise ValueError(f"Inherited {stage} inventory is incomplete.")
    if payload.get("schema_version") == SCHEMA:
        namespace = record.get("source_execution_namespace")
        source = _source(
            {"commit": record.get("source_commit"), "tree": record.get("source_tree")},
            label=f"Inherited {stage}",
        )
        receipt_sha = payload.get("superseded_execution_receipt_sha256")
    elif payload.get("schema_version") in {SCHEMA_V2, SCHEMA_V3}:
        namespace = record.get("producer_execution_namespace")
        source = _source(record.get("producer_evaluation_source"), label=f"Inherited {stage}")
        receipt_sha = record.get("producer_execution_receipt_sha256")
    else:
        raise ValueError("Stage inheritance receipt schema is incompatible.")
    if not isinstance(namespace, str) or not _nonempty(namespace):
        raise ValueError(f"Inherited {stage} producer namespace is incomplete.")
    if not isinstance(receipt_sha, str) or not _nonempty(receipt_sha):
        raise ValueError(f"Inherited {stage} producer receipt identity is incomplete.")
    producer_path = _safe_relative(root, f"{namespace}/execution.json")
    if not producer_path.is_file() or file_sha256(producer_path) != receipt_sha:
        raise ValueError(f"Inherited {stage} producer receipt checksum mismatch.")
    return namespace, receipt_sha, source


def _validate_ancestor_stage_link(
    config: Any,
    payload: Mapping[str, Any],
    predecessor: Mapping[str, Any],
) -> tuple[dict[str, Any], Path, str]:
    """Require v2's explicit link to the immediate predecessor's receipt."""
    if predecessor.get("schema_version") != "paper-evaluation-execution-v4":
        raise ValueError("Chained stage inheritance requires a v4 predecessor.")
    relative = predecessor.get("stage_inheritance_path")
    digest = predecessor.get("stage_inheritance_sha256")
    receipt_digest = predecessor.get("stage_inheritance_receipt_sha256")
    if receipt_digest is not None and receipt_digest != digest:
        raise ValueError("v4 stage-inheritance receipt aliases disagree.")
    if not isinstance(relative, str) or not _nonempty(digest):
        raise ValueError("Chained predecessor is missing its stage-inheritance receipt.")
    root = _artifact_root(config)
    expected_path = _safe_relative(root, relative)
    if not expected_path.is_file() or file_sha256(expected_path) != digest:
        raise ValueError("Chained predecessor stage-inheritance receipt checksum mismatch.")
    link = payload.get("ancestor_stage_inheritance")
    if not isinstance(link, Mapping):
        raise ValueError("Chained stage inheritance is missing its ancestor receipt link.")
    if (
        link.get("path") != relative
        or link.get("sha256") != digest
        or not isinstance(link.get("predecessor_namespace"), str)
        or not isinstance(link.get("predecessor_receipt_sha256"), str)
        or link.get("predecessor_namespace") != predecessor.get("superseded_execution_namespace")
        or link.get("predecessor_receipt_sha256")
        != predecessor.get("superseded_execution_receipt_sha256")
    ):
        raise ValueError("Chained stage inheritance ancestor receipt link is inconsistent.")
    predecessor_namespace = link["predecessor_namespace"]
    predecessor_sha = link["predecessor_receipt_sha256"]
    typed = verify_stage_inheritance(
        config,
        expected_path,
        replacement_source=_source(predecessor.get("evaluation_source"), label="Predecessor"),
        expected_predecessor=(predecessor_namespace, predecessor_sha),
    )
    return typed, expected_path, str(digest)


def _base_instruments(path: Path) -> tuple[str, ...]:
    """Read only the base scale identity column used to schedule EWMA ledgers."""
    try:
        import pyarrow.parquet as parquet

        schema = parquet.ParquetFile(path).schema_arrow
        if "instrument_id" not in schema.names:
            raise ValueError(f"Inherited scale base lacks instrument identity: {path}")
        values = (
            parquet.read_table(path, columns=["instrument_id"], use_threads=False)
            .column("instrument_id")
            .to_pylist()
        )
    except ImportError as exc:  # pragma: no cover - pyarrow is a core paper dependency
        raise RuntimeError(
            "BLOCKED: pyarrow is required to verify inherited base identities."
        ) from exc
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(
            f"Inherited scale base has missing or non-string instrument identities: {path}"
        )
    instruments = set(values)
    if not instruments:
        raise ValueError(f"Inherited scale base has no instrument identities: {path}")
    return tuple(sorted(instruments))


def _validate_profile(
    root: Path, source_root: Path, config: Any, source: dict[str, str]
) -> dict[str, str]:
    directory = source_root / "evaluation-v2" / "profile-corpus"
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("Inherited forecast profile corpus manifest is missing.")
    manifest = read_json(manifest_path)
    identity = manifest.get("identity")
    if (
        not isinstance(identity, Mapping)
        or identity.get("schema_version") != "paper-profile-corpus-v2"
    ):
        raise ValueError("Inherited forecast profile corpus schema is incompatible.")
    for key, value in {
        "paper_config_hash": config.config_hash,
        "parameter_freeze_sha256": file_sha256(_freeze_path(config)),
        "source_commit": source["commit"],
        "source_tree": source["tree"],
    }.items():
        if identity.get(key) != value:
            raise ValueError("Inherited forecast profile corpus identity mismatch.")
    instruments = manifest.get("instruments")
    if not isinstance(instruments, Mapping) or not instruments:
        raise ValueError("Inherited forecast profile corpus has no complete instrument inventory.")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or set(files) != set(instruments.values()):
        raise ValueError("Inherited forecast profile corpus instrument inventory mismatch.")
    children = list(directory.iterdir())
    if any(child.is_dir() for child in children) or {
        child.name for child in children if child.is_file()
    } != {"manifest.json", *files}:
        raise ValueError("Inherited forecast profile corpus contains undeclared members.")
    source_inventory = manifest.get("source_inventory")
    if (
        not isinstance(source_inventory, Mapping)
        or not source_inventory
        or any(
            not _nonempty(name) or not _nonempty(digest)
            for name, digest in source_inventory.items()
        )
        or identity.get("source_inventory_sha256") != stable_hash(dict(source_inventory))
    ):
        raise ValueError("Inherited forecast profile source inventory binding is incomplete.")
    inventory = {_relative(root, manifest_path): file_sha256(manifest_path)}
    for name, metadata in files.items():
        if not isinstance(name, str) or not isinstance(metadata, Mapping):
            raise ValueError("Inherited forecast profile corpus manifest member is malformed.")
        path = _safe_relative(directory, name)
        if not path.is_file() or not _nonempty(metadata.get("sha256")):
            raise ValueError(f"Inherited forecast profile member is missing: {path}")
        if file_sha256(path) != metadata["sha256"]:
            raise ValueError(f"Inherited forecast profile checksum mismatch: {path}")
        _check_parquet_metadata(path, metadata)
        inventory[_relative(root, path)] = str(metadata["sha256"])
    return inventory


def _folds(config: Any) -> list[str]:
    evaluation = getattr(config, "evaluation", None)
    if not isinstance(evaluation, Mapping):
        evaluation = getattr(config, "sections", {}).get("evaluation", {})
    folds = evaluation.get("folds", []) if isinstance(evaluation, Mapping) else []
    values = [
        str(item["id"]) for item in folds if isinstance(item, Mapping) and _nonempty(item.get("id"))
    ]
    if not values:
        raise ValueError("Stage inheritance requires configured evaluation folds.")
    return values


def _seeds(config: Any) -> list[int]:
    representation = getattr(config, "representation", None)
    if not isinstance(representation, Mapping):
        representation = getattr(config, "sections", {}).get("representation", {})
    values = representation.get("seeds", []) if isinstance(representation, Mapping) else []
    result = [int(value) for value in values]
    if not result:
        raise ValueError("Stage inheritance requires configured representation seeds.")
    return result


def _validate_forecast(
    root: Path, source_root: Path, config: Any, source: dict[str, str]
) -> dict[str, str]:
    inventory = _validate_profile(root, source_root, config, source)
    freeze_sha = file_sha256(_freeze_path(config))
    identity = {
        "paper_config_hash": config.config_hash,
        "parameter_freeze_sha256": freeze_sha,
        "source_commit": source["commit"],
        "source_tree": source["tree"],
    }
    folds = _folds(config)
    seeds = _seeds(config)
    variants: list[tuple[str, int | None]] = [("raw", None), ("untrained_neural", None)]
    variants.extend((geometry, seed) for geometry in ("dense", "sparse") for seed in seeds)
    profile_manifest = read_json(source_root / "evaluation-v2" / "profile-corpus" / "manifest.json")
    profile_instruments = {str(key) for key in profile_manifest["instruments"] if _nonempty(key)}
    forecast_sources: dict[str, str] = {}
    unavailable_sources: dict[str, str] = {}
    for fold in folds:
        base = source_root / "evaluation-v2" / "bases" / fold
        base_identity = {
            **identity,
            "sequence_manifest_sha256": file_sha256(
                root / "sequences" / fold / "sequence-manifest.json"
            ),
            "partition": "test",
        }
        inventory.update(
            _artifact_files(
                root,
                base,
                schema="paper-evaluation-base-v2",
                expected_names={"scale-base.parquet", "shape-base.parquet"},
                expected_identity=base_identity,
            )
        )
        base_instruments = _base_instruments(base / "scale-base.parquet")
        if not set(base_instruments).issubset(profile_instruments):
            raise ValueError(
                "Inherited base instrument identities are absent from the profile corpus."
            )
        for method, seed in variants:
            directory = (
                source_root / "evaluation-v2" / "forecasts" / fold / method / str(seed or "shared")
            )
            learned_identity = {
                **identity,
                "model_manifest_sha256": file_sha256(
                    root / "lightgbm" / fold / method / str(seed or "shared") / "manifest.json"
                ),
                "base_manifest_sha256": file_sha256(base / "manifest.json"),
                "embedding_sha256": (
                    file_sha256(
                        root
                        / "embeddings"
                        / fold
                        / method
                        / str(seed)
                        / "partition=test"
                        / "embeddings.parquet"
                    )
                    if seed is not None
                    else None
                ),
            }
            inventory.update(
                _artifact_files(
                    root,
                    directory,
                    schema="paper-forecast-ledger-v2",
                    expected_names={"scale.parquet", "shape.parquet", "metrics.parquet"},
                    expected_identity={
                        **learned_identity,
                        "fold_id": fold,
                        "method": method,
                        "seed": seed,
                    },
                )
            )
            metrics = directory / "metrics.parquet"
            forecast_sources[_relative(root, metrics)] = file_sha256(metrics)
        for instrument in base_instruments:
            directory = (
                source_root
                / "evaluation-v2"
                / "forecasts"
                / fold
                / "ewma"
                / hashlib.sha256(instrument.encode()).hexdigest()
            )
            profile_name = profile_manifest["instruments"].get(instrument)
            if not isinstance(profile_name, str):
                raise ValueError("Inherited EWMA instrument is absent from the profile corpus.")
            ewma_identity = {
                **identity,
                "fold_id": fold,
                "instrument_id": instrument,
                "base_manifest_sha256": file_sha256(base / "manifest.json"),
                "market_sha256": file_sha256(
                    source_root / "evaluation-v2" / "profile-corpus" / profile_name
                ),
            }
            inventory.update(
                _artifact_files(
                    root,
                    directory,
                    schema="paper-ewma-ledger-v4",
                    expected_names={
                        "scale.parquet",
                        "shape.parquet",
                        "metrics.parquet",
                        "minute-forecasts.parquet",
                        "unavailable.parquet",
                    },
                    expected_identity=ewma_identity,
                )
            )
            metrics = directory / "metrics.parquet"
            forecast_sources[_relative(root, metrics)] = file_sha256(metrics)
            unavailable = directory / "unavailable.parquet"
            unavailable_sources[_relative(root, unavailable)] = file_sha256(unavailable)
    merged = source_root / "evaluation"
    forecast = _merged_file(
        root,
        merged,
        "forecast-results.parquet",
        "paper-forecast-evaluation-v1",
        expected_identity=identity,
        expected_sources=forecast_sources,
    )
    if _merge_identity_source(merged / "forecast-results.manifest.json") != source:
        raise ValueError("Inherited merged forecast source identity mismatch.")
    inventory.update(forecast)
    unavailable = merged / "forecast-unavailable.parquet"
    if unavailable.exists() or unavailable.with_suffix(".manifest.json").exists():
        inventory.update(
            _merged_file(
                root,
                merged,
                "forecast-unavailable.parquet",
                "paper-forecast-unavailable-v1",
                expected_identity=identity,
                expected_sources=unavailable_sources,
            )
        )
        if _merge_identity_source(merged / "forecast-unavailable.manifest.json") != source:
            raise ValueError("Inherited unavailable forecast source identity mismatch.")
    return inventory


def _validate_representation(
    root: Path,
    source_root: Path,
    config: Any,
    source: dict[str, str],
    *,
    verified_paths: set[Path] | None = None,
) -> dict[str, str]:
    inventory: dict[str, str] = {}
    freeze_sha = file_sha256(_freeze_path(config))
    identity = {
        "paper_config_hash": config.config_hash,
        "parameter_freeze_sha256": freeze_sha,
        "source_commit": source["commit"],
        "source_tree": source["tree"],
    }
    # Representation outputs are produced from the immutable checkpoint/export
    # matrix.  Recompute those identities from the canonical inventory instead
    # of trusting coordinate manifests to name arbitrary model bytes.
    from execsim.ml.paper.evaluation_execution import frozen_inventory, representation_root

    frozen = frozen_inventory(config)
    if verified_paths is not None:
        verified_paths.update(_safe_relative(root, relative) for relative in frozen)
    representation_source_root = representation_root(config).resolve()
    merged_sources: dict[str, dict[str, str]] = {
        "representation-accessibility.parquet": {},
        "representation-date-metrics.parquet": {},
        "support-regimes.parquet": {},
    }
    for fold in _folds(config):
        for geometry in ("dense", "sparse"):
            for seed in _seeds(config):
                coordinate = f"{fold}/{geometry}/{seed}"
                inventory.update(
                    _artifact_files(
                        root,
                        source_root / "evaluation-v2" / "representations" / coordinate,
                        schema="paper-representation-coordinate-v1",
                        expected_names=(
                            {"accessibility.parquet", "date-metrics.parquet"}
                            | ({"support.parquet"} if geometry == "sparse" else set())
                        ),
                        expected_identity={**identity, "coordinate": coordinate},
                    )
                )
                coordinate_root = source_root / "evaluation-v2" / "representations" / coordinate
                fold_id, geometry, seed_text = coordinate.split("/")
                checkpoint_manifest = (
                    representation_source_root
                    / fold_id
                    / geometry
                    / seed_text
                    / "final"
                    / "manifest.json"
                )
                compatibility = (
                    representation_source_root
                    / fold_id
                    / geometry
                    / seed_text
                    / "compatibility.json"
                )
                embedding_manifest = (
                    root / "embeddings" / fold_id / geometry / seed_text / "manifest.json"
                )
                sequence = root / "sequences" / fold_id / "sequence-manifest.json"
                expected_upstream = {
                    checkpoint_manifest: "checkpoint_sha256",
                    compatibility: "compatibility_sha256",
                    embedding_manifest: "embedding_manifest_sha256",
                    sequence: "sequence_sha256",
                }
                expected_identity = {
                    **identity,
                    **{
                        field: frozen.get(Path(os.path.abspath(path)).relative_to(root).as_posix())
                        for path, field in expected_upstream.items()
                    },
                }
                # All model/embedding/sequence members are already checksum-
                # validated by frozen_inventory; require every coordinate to
                # carry the same exact links.
                if any(value is None for value in expected_identity.values()):
                    raise ValueError("Frozen representation coordinate identity is incomplete.")
                coordinate_manifest = read_json(coordinate_root / "manifest.json")
                coordinate_identity = coordinate_manifest.get("identity")
                if not isinstance(coordinate_identity, Mapping) or any(
                    coordinate_identity.get(key) != value
                    for key, value in expected_identity.items()
                ):
                    raise ValueError(
                        "Inherited representation coordinate source identity mismatch."
                    )
                for output_name, source_name in (
                    ("representation-accessibility.parquet", "accessibility.parquet"),
                    ("representation-date-metrics.parquet", "date-metrics.parquet"),
                    ("support-regimes.parquet", "support.parquet"),
                ):
                    if (coordinate_root / source_name).is_file():
                        merged_sources[output_name][coordinate] = file_sha256(
                            coordinate_root / source_name
                        )
    merged = source_root / "evaluation"
    for name in (
        "representation-accessibility.parquet",
        "representation-date-metrics.parquet",
        "support-regimes.parquet",
    ):
        manifest = (merged / name).with_suffix(".manifest.json")
        inventory.update(
            _merged_file(
                root,
                merged,
                name,
                "paper-representation-result-v1",
                expected_identity=identity,
                expected_sources=merged_sources[name],
            )
        )
        if _merge_identity_source(manifest) != source:
            raise ValueError("Inherited merged representation source identity mismatch.")
    aggregate = merged / "representation-evaluation-manifest.json"
    if not aggregate.is_file():
        raise ValueError("Inherited representation aggregate manifest is missing.")
    payload = read_json(aggregate)
    if (
        payload.get("schema_version") != "paper-representation-evaluation-v2"
        or payload.get("paper_config_hash") != config.config_hash
    ):
        raise ValueError("Inherited representation aggregate manifest schema/config mismatch.")
    for key, name in (
        ("accessibility_sha256", "representation-accessibility.parquet"),
        ("date_metrics_sha256", "representation-date-metrics.parquet"),
        ("support_regimes_sha256", "support-regimes.parquet"),
    ):
        if payload.get(key) != file_sha256(merged / name):
            raise ValueError("Inherited representation aggregate output checksum mismatch.")
    inventory[_relative(root, aggregate)] = file_sha256(aggregate)
    return inventory


def _reject_symlink(path: Path, *, label: str) -> None:
    """Reject reparse/symlink substitutions in the report-only inventory."""
    if path.is_symlink():
        raise ValueError(f"Inherited {label} must be a regular path, not a symlink: {path}")


def _tca_identity(config: Any, source: Mapping[str, str]) -> dict[str, str]:
    """Return the identity shared by TCA shards, merges, and their aggregate."""
    return {
        "source_commit": source["commit"],
        "source_tree": source["tree"],
        "paper_config_hash": str(config.config_hash),
        "parameter_freeze_sha256": file_sha256(_freeze_path(config)),
    }


def _canonical_tca_shard_path(
    root: Path, source_root: Path, relative: object, name: str
) -> tuple[Path, str, str]:
    """Resolve a merge source and require its canonical fold/day shard layout."""
    if not isinstance(relative, str) or not _nonempty(relative):
        raise ValueError("Inherited TCA merge source path is incomplete.")
    path = _safe_relative(root, relative)
    _reject_symlink(path, label="TCA shard member")
    shards_root = source_root / "evaluation-v2" / "tca-shards"
    if not path.is_relative_to(shards_root) or path.name != name:
        raise ValueError("Inherited TCA merge source is outside its canonical shard inventory.")
    relative_to_shards = path.relative_to(shards_root)
    if len(relative_to_shards.parts) != 3:
        raise ValueError("Inherited TCA merge source must be fold/day/<result>.parquet.")
    fold_id, session_date, filename = relative_to_shards.parts
    if not _nonempty(fold_id) or not _nonempty(session_date) or filename != name:
        raise ValueError("Inherited TCA merge source has an invalid fold/day identity.")
    canonical = _relative(root, path)
    if relative != canonical:
        raise ValueError("Inherited TCA merge source path is not canonical.")
    return path, fold_id, session_date


def _validate_tca(
    root: Path,
    source_root: Path,
    config: Any,
    source: Mapping[str, str],
    expected_tca_dates: object,
) -> dict[str, str]:
    """Verify complete TCA merges, aggregate identity, and every date shard.

    Independent qualification supplies the expected dates.  Merge receipts
    and physical directories must agree with that inventory, so a date removed
    consistently from both cannot be hidden by a valid merge.
    """
    expected_dates = _normalize_expected_tca_dates(config, expected_tca_dates)
    expected_date_dirs = {
        (source_root / "evaluation-v2" / "tca-shards" / fold_id / day).resolve()
        for fold_id, dates in expected_dates.items()
        for day in dates
    }
    tca_root = source_root / "tca"
    aggregate = tca_root / "manifest.json"
    shards_root = source_root / "evaluation-v2" / "tca-shards"
    for path, label in (
        (tca_root, "TCA root"),
        (aggregate, "TCA aggregate manifest"),
        (shards_root, "TCA shard root"),
    ):
        _reject_symlink(path, label=label)
    if not tca_root.is_dir() or not aggregate.is_file() or not shards_root.is_dir():
        raise ValueError("Inherited TCA stage is incomplete.")

    identity = _tca_identity(config, source)
    aggregate_payload = read_json(aggregate)
    if (
        aggregate_payload.get("schema_version") != "paper-tca-v1"
        or aggregate_payload.get("paper_config_hash") != config.config_hash
        or aggregate_payload.get("evaluation_identity") != identity
    ):
        raise ValueError("Inherited TCA aggregate manifest identity mismatch.")
    aggregate_files = aggregate_payload.get("files")
    if not isinstance(aggregate_files, Mapping) or set(aggregate_files) != {
        "main",
        "sensitivity",
    }:
        raise ValueError("Inherited TCA aggregate manifest file inventory mismatch.")

    # The TCA directory is itself a closed artifact.  Unexpected reports or
    # temporary files must not become implicit inputs to the new report.
    expected_tca_members = {
        "manifest.json",
        "main.parquet",
        "main.manifest.json",
        "sensitivity.parquet",
        "sensitivity.manifest.json",
    }
    if {child.name for child in tca_root.iterdir()} != expected_tca_members or any(
        child.is_dir() for child in tca_root.iterdir()
    ):
        raise ValueError("Inherited TCA root contains undeclared members.")

    merge_sources: dict[str, dict[str, str]] = {}
    inventory: dict[str, str] = {}
    for name in ("main.parquet", "sensitivity.parquet"):
        output = tca_root / name
        receipt_path = output.with_suffix(".manifest.json")
        _reject_symlink(output, label="TCA merged output")
        _reject_symlink(receipt_path, label="TCA merge receipt")
        if not output.is_file() or not receipt_path.is_file():
            raise ValueError(f"Inherited TCA merged output is incomplete: {output}")
        receipt = read_json(receipt_path)
        merge_identity = receipt.get("merge_identity")
        if not isinstance(merge_identity, Mapping):
            raise ValueError(f"Inherited TCA merge receipt is incomplete: {receipt_path}")
        sources = merge_identity.get("sources")
        if not isinstance(sources, Mapping) or not sources:
            raise ValueError(
                f"Inherited TCA merge receipt shard inventory is incomplete: {receipt_path}"
            )
        normalized_sources: dict[str, str] = {}
        for relative, digest in sources.items():
            if not isinstance(digest, str) or not _nonempty(digest):
                raise ValueError(
                    f"Inherited TCA merge receipt contains an invalid shard hash: {receipt_path}"
                )
            shard_path, _fold, _day = _canonical_tca_shard_path(root, source_root, relative, name)
            normalized_sources[relative] = digest
            if not shard_path.is_file() or file_sha256(shard_path) != digest:
                raise ValueError(f"Inherited TCA shard checksum mismatch: {shard_path}")
        merge_sources[name] = normalized_sources
        # _merged_file validates the merge schema, output checksum, Parquet
        # metadata, source identity, and exact source map.
        inventory.update(
            _merged_file(
                root,
                tca_root,
                name,
                "paper-tca-merged-v3",
                expected_identity=identity,
                expected_sources=normalized_sources,
            )
        )
        record = aggregate_files[name.removesuffix(".parquet")]
        if (
            not isinstance(record, Mapping)
            or record.get("path") != str(output)
            or record.get("sha256") != file_sha256(output)
        ):
            raise ValueError("Inherited TCA aggregate output identity/checksum mismatch.")

    main_sources = merge_sources["main.parquet"]
    sensitivity_sources = merge_sources["sensitivity.parquet"]
    main_dirs = {_safe_relative(root, relative).parent.resolve() for relative in main_sources}
    sensitivity_dirs = {
        _safe_relative(root, relative).parent.resolve() for relative in sensitivity_sources
    }
    if main_dirs != sensitivity_dirs:
        raise ValueError("Inherited TCA main/sensitivity shard inventories differ.")

    merge_dirs: set[Path] = set()
    for relative in main_sources:
        path = _safe_relative(root, relative)
        merge_dirs.add(path.parent.resolve())
    if merge_dirs != expected_date_dirs:
        raise ValueError("Inherited TCA merge inventory does not match expected dates.")

    # TCA workers are launched once per complete fold/day input.  Requiring
    # that input-date inventory to match the merged sources closes the gap in
    # which both merge receipts could be edited consistently while silently
    # dropping a date.  Input bytes remain outside the inherited output
    # inventory; the shard and merge checks below bind the completed results.
    inputs_root = source_root / "evaluation-v2" / "tca-inputs"
    _reject_symlink(inputs_root, label="TCA input root")
    if not inputs_root.is_dir():
        raise ValueError("Inherited TCA input inventory is missing.")
    actual_input_dirs: set[Path] = set()
    for fold in inputs_root.iterdir():
        _reject_symlink(fold, label="TCA input fold directory")
        if not fold.is_dir():
            raise ValueError("Inherited TCA input root contains a non-directory member.")
        for day in fold.iterdir():
            _reject_symlink(day, label="TCA input day directory")
            if not day.is_dir() or not (day / "manifest.json").is_file():
                raise ValueError("Inherited TCA input date is incomplete.")
            actual_input_dirs.add(
                (source_root / "evaluation-v2" / "tca-shards" / fold.name / day.name).resolve()
            )
    if actual_input_dirs != expected_date_dirs:
        raise ValueError("Inherited TCA input-date inventory does not match completed shards.")

    # Every physical shard directory must be represented by both merge
    # receipts.  Enumerate only the declared two-level fold/day layout and
    # reject files, nested directories, and undeclared temporary members.
    actual_dirs: set[Path] = set()
    for fold in shards_root.iterdir():
        _reject_symlink(fold, label="TCA fold directory")
        if not fold.is_dir():
            raise ValueError("Inherited TCA shard root contains a non-directory member.")
        for day in fold.iterdir():
            _reject_symlink(day, label="TCA day directory")
            if not day.is_dir():
                raise ValueError("Inherited TCA fold contains a non-directory member.")
            actual_dirs.add(day.resolve())
            shard_inventory = _artifact_files(
                root,
                day,
                schema="paper-tca-date-shard-v3",
                expected_names={"main.parquet", "sensitivity.parquet"},
                expected_identity={
                    "source_commit": source["commit"],
                    "source_tree": source["tree"],
                    "paper_config_hash": config.config_hash,
                    "parameter_freeze_sha256": identity["parameter_freeze_sha256"],
                    "fold_id": fold.name,
                    "session_date": day.name,
                },
            )
            for member, digest in shard_inventory.items():
                if member.endswith("/main.parquet"):
                    expected = main_sources.get(member)
                elif member.endswith("/sensitivity.parquet"):
                    expected = sensitivity_sources.get(member)
                else:
                    # The shard manifest is not a merge input, but it is part
                    # of the complete inherited byte inventory below.
                    expected = None
                if expected is not None and expected != digest:
                    raise ValueError(f"Inherited TCA shard inventory checksum mismatch: {member}")
                inventory[member] = digest
    if actual_dirs != expected_date_dirs:
        raise ValueError(
            "Inherited TCA shard directory inventory is incomplete or contains extras."
        )
    inventory[_relative(root, aggregate)] = file_sha256(aggregate)
    return inventory


def _normalize_expected_tca_dates(config: Any, expected: object) -> dict[str, list[str]]:
    """Validate the independently qualified fold/date TCA inventory."""
    folds = tuple(_folds(config))
    if not isinstance(expected, Mapping) or set(expected) != set(folds):
        raise ValueError(
            "Report-only inheritance requires an expected TCA date list for every fold."
        )
    normalized: dict[str, list[str]] = {}
    for fold_id in folds:
        values = expected.get(fold_id)
        if not isinstance(values, (list, tuple)) or not values:
            raise ValueError("Report-only expected TCA dates are empty or malformed.")
        dates: list[str] = []
        for value in values:
            if not isinstance(value, str) or not _nonempty(value):
                raise ValueError("Report-only expected TCA date is incomplete.")
            try:
                parsed = datetime.fromisoformat(value).date()
            except ValueError as exc:
                raise ValueError("Report-only expected TCA date is not ISO-8601.") from exc
            if parsed.isoformat() != value:
                raise ValueError("Report-only expected TCA date is not canonical.")
            dates.append(value)
        if dates != sorted(set(dates)):
            raise ValueError("Report-only expected TCA dates must be sorted and unique.")
        normalized[fold_id] = dates
    return normalized


def _validate_receipt_creation(payload: Mapping[str, Any]) -> None:
    """Require a reason and timezone-aware creation clock for every generation."""
    created_at = payload.get("created_at_utc")
    if not _nonempty(payload.get("reason")) or not _nonempty(created_at):
        raise ValueError("Stage inheritance receipt reason or creation time is incomplete.")
    try:
        timestamp = datetime.fromisoformat(str(created_at))
    except ValueError as exc:
        raise ValueError("Stage inheritance receipt creation time is malformed.") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("Stage inheritance receipt creation time must be timezone-aware.")


def _validate_receipt_sources(
    payload: Mapping[str, Any],
    predecessor: Mapping[str, Any],
    replacement_source: dict[str, str] | None,
    expected_predecessor: tuple[str, str] | None,
) -> dict[str, str]:
    """Bind original and replacement sources without conflating their roles."""
    source = _source(payload.get("superseded_evaluation_source"), label="Superseded")
    if source != _source(predecessor.get("evaluation_source"), label="Predecessor"):
        raise ValueError("Stage inheritance predecessor source identity mismatch.")
    replacement = _source(payload.get("replacement_evaluation_source"), label="Replacement")
    if replacement_source is not None and replacement != replacement_source:
        raise ValueError("Stage inheritance replacement evaluator identity mismatch.")
    actual_predecessor = (
        payload.get("superseded_execution_namespace"),
        str(payload.get("superseded_execution_receipt_sha256")),
    )
    if expected_predecessor is not None and actual_predecessor != expected_predecessor:
        raise ValueError("Stage inheritance predecessor does not match supersession receipt.")
    return source


def _validate_report_receipt(
    config: Any,
    receipt_path: Path,
    payload: Mapping[str, Any],
    *,
    replacement_source: dict[str, str] | None = None,
    expected_predecessor: tuple[str, str] | None = None,
) -> tuple[dict[str, Any], dict[Path, tuple[int, int, int, int]]]:
    """Validate the report-only v3 chain and return its complete byte states."""
    root = _artifact_root(config)
    _validate_receipt_creation(payload)
    freeze_sha, opened_sha, root_source, config_hash = _current_metadata(config)
    if (
        payload.get("protocol_id") != "sparse-jepa-v2"
        or payload.get("paper_config_hash") != config_hash
        or payload.get("parameter_freeze_sha256") != freeze_sha
        or payload.get("locked_test_open_receipt_sha256") != opened_sha
        or payload.get("root_evaluation_source") != root_source
        or payload.get("root_evaluation_open_receipt_sha256", opened_sha) != opened_sha
        or payload.get("invalidation_frontier") != REPORT_FRONTIER
        or tuple(payload.get("inherited_stages", ())) != REPORT_INHERITED_STAGES
    ):
        raise ValueError("Report-only stage inheritance protocol/frontier identity mismatch.")

    predecessor_ns = payload.get("superseded_execution_namespace")
    predecessor_sha = payload.get("superseded_execution_receipt_sha256")
    if not isinstance(predecessor_ns, str) or not _nonempty(predecessor_sha):
        raise ValueError("Stage inheritance receipt is missing its predecessor identity.")
    predecessor_receipt = _safe_relative(root, f"{predecessor_ns}/execution.json")
    predecessor, namespace_path, _base_inventory, states = _validate_predecessor(
        config, predecessor_receipt, str(predecessor_sha)
    )
    if _relative(root, namespace_path) != predecessor_ns:
        raise ValueError("Stage inheritance predecessor namespace is not canonical.")
    if predecessor.get("schema_version") != "paper-evaluation-execution-v4":
        raise ValueError("Report-only inheritance requires a v4 TCA-complete predecessor.")
    if (
        predecessor.get("inherited_stages") != list(INHERITED_STAGES)
        or predecessor.get("invalidation_frontier") != FRONTIER
    ):
        raise ValueError("Report-only predecessor does not have the TCA-restart contract.")

    source = _validate_receipt_sources(
        payload, predecessor, replacement_source, expected_predecessor
    )

    inventories = payload.get("stage_inventory")
    if not isinstance(inventories, Mapping) or set(inventories) != set(REPORT_INHERITED_STAGES):
        raise ValueError("Report-only stage inventory is incomplete or contains extra stages.")

    # A v4 predecessor already verified its v1/v2 receipt.  Re-validate and
    # bind that exact receipt here so the v3 chain cannot silently replace the
    # original forecast/representation producers.
    predecessor_stage = _execution_stage_receipt(
        config,
        predecessor_receipt,
        predecessor,
        source=source,
    )
    if predecessor_stage is None:
        raise ValueError("Report-only predecessor is missing its typed stage receipt.")
    ancestor_payload, ancestor_path, ancestor_sha = predecessor_stage
    if ancestor_payload.get("schema_version") not in {SCHEMA, SCHEMA_V2}:
        raise ValueError("Report-only predecessor stage receipt is not v1/v2.")
    if (
        tuple(ancestor_payload.get("inherited_stages", ())) != INHERITED_STAGES
        or ancestor_payload.get("invalidation_frontier") != FRONTIER
    ):
        raise ValueError("Report-only predecessor stage receipt has incompatible frontier.")
    states[ancestor_path] = _state(ancestor_path)
    ancestor_cached = _CACHE.get(_cache_key(config, ancestor_path))
    if ancestor_cached is not None:
        states.update(ancestor_cached[1])

    # Require an explicit link to the predecessor's inheritance receipt.  This
    # mirrors v2's chain contract and keeps d0 in the succession even when the
    # forecast/representation bytes originated in the earlier execution.
    link = payload.get("ancestor_stage_inheritance")
    if not isinstance(link, Mapping):
        raise ValueError("Report-only stage inheritance is missing its ancestor receipt link.")
    if (
        link.get("path") != _relative(root, ancestor_path)
        or link.get("sha256") != ancestor_sha
        or link.get("predecessor_namespace") != predecessor.get("superseded_execution_namespace")
        or link.get("predecessor_receipt_sha256")
        != predecessor.get("superseded_execution_receipt_sha256")
    ):
        raise ValueError("Report-only ancestor receipt link is inconsistent.")

    for stage in INHERITED_STAGES:
        producer_ns, producer_sha, producer_source = _stage_producer_descriptor(
            root, ancestor_payload, stage
        )
        ancestor_record = ancestor_payload.get("stage_inventory", {}).get(stage)
        if not isinstance(ancestor_record, Mapping) or not isinstance(
            ancestor_record.get("files"), Mapping
        ):
            raise ValueError(f"Report-only ancestor inventory is incomplete for {stage}.")
        expected_record = {
            "producer_execution_namespace": producer_ns,
            "producer_execution_receipt_sha256": producer_sha,
            "producer_evaluation_source": producer_source,
            "files": dict(ancestor_record["files"]),
        }
        record = inventories.get(stage)
        if not isinstance(record, Mapping) or any(
            record.get(key) != value for key, value in expected_record.items()
        ):
            raise ValueError(f"Report-only inherited {stage} producer or inventory changed.")
    expected_tca_dates = payload.get("expected_tca_dates")
    tca_files = _validate_tca(root, namespace_path, config, source, expected_tca_dates)
    run_tca = inventories.get("run-tca")
    expected_tca_record = {
        "producer_execution_namespace": predecessor_ns,
        "producer_execution_receipt_sha256": str(predecessor_sha),
        "producer_evaluation_source": source,
        "files": tca_files,
    }
    if not isinstance(run_tca, Mapping) or any(
        run_tca.get(key) != value for key, value in expected_tca_record.items()
    ):
        raise ValueError("Report-only inherited TCA producer or inventory changed.")

    for relative in tca_files:
        path = _safe_relative(root, relative)
        states[path] = _state(path)
    states[receipt_path] = _state(receipt_path)
    states[predecessor_receipt] = _state(predecessor_receipt)
    return dict(payload), states


def _validate_predecessor(
    config: Any, receipt_path: Path, expected_sha: str
) -> tuple[dict[str, Any], Path, dict[str, str], dict[Path, tuple[int, int, int, int]]]:
    root = _artifact_root(config)
    if not receipt_path.is_file() or receipt_path.name != "execution.json":
        raise ValueError("Stage inheritance predecessor execution receipt is unavailable.")
    if file_sha256(receipt_path) != expected_sha:
        raise ValueError("Stage inheritance predecessor execution receipt checksum mismatch.")
    # Reuse the canonical execution provenance verifier so stage inheritance
    # and execution supersession cannot drift into two ancestry contracts.
    from execsim.ml.paper.evaluation_execution import _validate_prior_execution

    verified_paths: set[Path] = set()
    predecessor = _validate_prior_execution(
        config,
        receipt_path,
        expected_sha256=expected_sha,
        verified_paths=verified_paths,
    )
    namespace_path = receipt_path.parent.resolve()
    execution_root = (root / "evaluation-executions").resolve()
    if not namespace_path.is_relative_to(execution_root):
        raise ValueError("Stage inheritance predecessor must be below evaluation-executions.")
    inventory: dict[str, str] = {_relative(root, receipt_path): file_sha256(receipt_path)}
    states: dict[Path, tuple[int, int, int, int]] = {
        path: _state(path) for path in verified_paths if path.is_file()
    }
    states[receipt_path] = _state(receipt_path)
    return predecessor, namespace_path, inventory, states


def _validate_receipt(
    config: Any,
    receipt_path: Path,
    *,
    replacement_source: dict[str, str] | None = None,
    expected_predecessor: tuple[str, str] | None = None,
) -> tuple[dict[str, Any], dict[Path, tuple[int, int, int, int]]]:
    root = _artifact_root(config)
    if not receipt_path.is_file() or not receipt_path.resolve().is_relative_to(root):
        raise ValueError("Stage inheritance receipt is unavailable or outside the artifact root.")
    payload = read_json(receipt_path)
    schema = payload.get("schema_version")
    if schema not in SUPPORTED_SCHEMAS or payload.get("status") != STATUS:
        raise ValueError("Stage inheritance receipt schema/status is incompatible.")
    if schema == SCHEMA_V3:
        return _validate_report_receipt(
            config,
            receipt_path,
            payload,
            replacement_source=replacement_source,
            expected_predecessor=expected_predecessor,
        )
    _validate_receipt_creation(payload)
    freeze_sha, opened_sha, root_source, config_hash = _current_metadata(config)
    if (
        payload.get("protocol_id") != "sparse-jepa-v2"
        or payload.get("paper_config_hash") != config_hash
        or payload.get("parameter_freeze_sha256") != freeze_sha
        or payload.get("locked_test_open_receipt_sha256") != opened_sha
        or payload.get("root_evaluation_source") != root_source
        or payload.get("root_evaluation_open_receipt_sha256", opened_sha) != opened_sha
    ):
        raise ValueError("Stage inheritance receipt protocol/config/freeze/TEST identity mismatch.")
    if (
        payload.get("invalidation_frontier") != FRONTIER
        or tuple(payload.get("inherited_stages", ())) != INHERITED_STAGES
    ):
        raise ValueError("Stage inheritance receipt frontier or stage set is incompatible.")
    predecessor_ns = payload.get("superseded_execution_namespace")
    predecessor_sha = payload.get("superseded_execution_receipt_sha256")
    if not isinstance(predecessor_ns, str) or not _nonempty(predecessor_sha):
        raise ValueError("Stage inheritance receipt is missing its predecessor identity.")
    predecessor_receipt = _safe_relative(root, f"{predecessor_ns}/execution.json")
    predecessor, namespace_path, _base_inventory, states = _validate_predecessor(
        config, predecessor_receipt, str(predecessor_sha)
    )
    if _relative(root, namespace_path) != predecessor_ns:
        raise ValueError("Stage inheritance predecessor namespace is not canonical.")
    source = _validate_receipt_sources(
        payload, predecessor, replacement_source, expected_predecessor
    )
    inventories = payload.get("stage_inventory")
    if not isinstance(inventories, Mapping) or set(inventories) != set(INHERITED_STAGES):
        raise ValueError(
            "Stage inheritance stage inventory is incomplete or contains extra stages."
        )
    verified_crosslinks: set[Path] = set()
    if schema == SCHEMA:
        # Preserve the v1 contract byte-for-byte: each stage is produced by
        # the receipt's immediate predecessor and uses the original fields.
        expected = {
            "evaluate-forecast": _validate_forecast(root, namespace_path, config, source),
            "evaluate-representation": _validate_representation(
                root,
                namespace_path,
                config,
                source,
                verified_paths=verified_crosslinks,
            ),
        }
        for stage in INHERITED_STAGES:
            record = inventories[stage]
            if (
                not isinstance(record, Mapping)
                or record.get("source_execution_namespace") != predecessor_ns
                or record.get("source_commit") != source["commit"]
                or record.get("source_tree") != source["tree"]
                or record.get("files") != expected[stage]
            ):
                raise ValueError(f"Inherited {stage} inventory is incomplete or changed.")
            for relative in expected[stage]:
                path = _safe_relative(root, relative)
                states[path] = _state(path)
    else:
        # v2 is only valid when the immediate predecessor is itself a typed
        # stage-inherited v4 execution.  Its receipt link is checked in
        # addition to v4's native execution verification, so an ancestor
        # cannot be silently replaced by an equivalent-looking directory.
        ancestor_payload, ancestor_path, _ancestor_sha = _validate_ancestor_stage_link(
            config, payload, predecessor
        )
        states[ancestor_path] = _state(ancestor_path)
        ancestor_cached = _CACHE.get(_cache_key(config, ancestor_path))
        if ancestor_cached is not None:
            states.update(ancestor_cached[1])
        expected = {}
        for stage in INHERITED_STAGES:
            producer_ns, producer_sha, producer_source = _stage_producer_descriptor(
                root, payload, stage
            )
            ancestor_ns, ancestor_sha, ancestor_source = _stage_producer_descriptor(
                root, ancestor_payload, stage
            )
            ancestor_record = ancestor_payload["stage_inventory"][stage]
            if (
                producer_ns != ancestor_ns
                or producer_sha != ancestor_sha
                or producer_source != ancestor_source
                or not isinstance(ancestor_record, Mapping)
                or not isinstance(ancestor_record.get("files"), Mapping)
            ):
                raise ValueError(f"Inherited {stage} producer does not match its ancestor.")
            # The ancestor verifier has already checked every manifest member
            # and native execution receipt.  Equality to its complete record
            # prevents a v2 receipt from selecting another valid producer.
            expected[stage] = dict(ancestor_record["files"])
            record = inventories[stage]
            if (
                not isinstance(record, Mapping)
                or record.get("producer_execution_namespace") != producer_ns
                or record.get("producer_execution_receipt_sha256") != producer_sha
                or record.get("producer_evaluation_source") != producer_source
                or record.get("files") != expected[stage]
            ):
                raise ValueError(f"Inherited {stage} inventory is incomplete or changed.")
            for relative in expected[stage]:
                path = _safe_relative(root, relative)
                states[path] = _state(path)
    states[receipt_path] = _state(receipt_path)
    states[predecessor_receipt] = _state(predecessor_receipt)
    states.update({path: _state(path) for path in verified_crosslinks if path.is_file()})
    return payload, states


_CACHE: dict[str, tuple[dict[str, Any], dict[Path, tuple[int, int, int, int]]]] = {}
_VALIDATION_STACK: set[Path] = set()


def _cache_key(config: Any, receipt: Path) -> str:
    representation = getattr(config, "runtime_representation_root", None)
    representation_root = Path(representation or (_artifact_root(config) / "representations"))
    return stable_hash(
        {
            "root": str(_artifact_root(config)),
            "receipt": str(receipt.resolve()),
            "paper_config_hash": str(config.config_hash),
            "folds": _folds(config),
            "seeds": _seeds(config),
            "representation_root": str(representation_root.resolve()),
        }
    )


def verify_stage_inheritance(
    config: Any,
    receipt: Path,
    *,
    replacement_source: dict[str, str] | None = None,
    expected_predecessor: tuple[str, str] | None = None,
) -> dict[str, Any]:
    """Verify a typed receipt and every inherited manifest member/checksum."""
    receipt = Path(receipt).resolve()
    if receipt in _VALIDATION_STACK:
        raise ValueError("Stage inheritance receipt ancestry contains a cycle.")
    key = _cache_key(config, receipt)
    if key in _CACHE:
        cached, states = _CACHE[key]
        try:
            if all(path.is_file() and _state(path) == state for path, state in states.items()):
                if (
                    replacement_source is None
                    or cached.get("replacement_evaluation_source") == replacement_source
                ):
                    if (
                        expected_predecessor is None
                        or (
                            cached.get("superseded_execution_namespace"),
                            cached.get("superseded_execution_receipt_sha256"),
                        )
                        == expected_predecessor
                    ):
                        return deepcopy(cached)
        except OSError:
            pass
    _VALIDATION_STACK.add(receipt)
    try:
        payload, states = _validate_receipt(
            config,
            receipt,
            replacement_source=replacement_source,
            expected_predecessor=expected_predecessor,
        )
    finally:
        _VALIDATION_STACK.remove(receipt)
    _CACHE[key] = (deepcopy(payload), states)
    return payload


def _current_inheritance(config: Any) -> tuple[dict[str, Any], Path] | None:
    current_root = _evaluation_root(config)
    if current_root.resolve() == _artifact_root(config):
        return None
    execution_path = current_root / "execution.json"
    if not execution_path.is_file():
        return None
    execution = read_json(execution_path)
    if execution.get("schema_version") not in {
        "paper-evaluation-execution-v4",
        "paper-evaluation-execution-v5",
    }:
        return None
    freeze_sha, opened_sha, root_source, config_hash = _current_metadata(config)
    if (
        execution.get("status") != "EVALUATION_RESEALED"
        or execution.get("protocol_id") != "sparse-jepa-v2"
        or execution.get("paper_config_hash") != config_hash
        or execution.get("parameter_freeze_sha256") != freeze_sha
        or execution.get("root_evaluation_source") != root_source
        or execution.get("root_evaluation_open_receipt_sha256", opened_sha) != opened_sha
    ):
        raise ValueError("Current v4 execution is incompatible with the locked paper run.")
    relative = execution.get("stage_inheritance_path")
    digest = execution.get("stage_inheritance_sha256")
    receipt_digest = execution.get("stage_inheritance_receipt_sha256")
    if receipt_digest is not None and receipt_digest != digest:
        raise ValueError("v4 stage-inheritance receipt aliases disagree.")
    if not isinstance(relative, str) or not _nonempty(digest):
        raise ValueError("v4 execution is missing its typed stage-inheritance receipt.")
    receipt_path = _safe_relative(_artifact_root(config), relative)
    if not receipt_path.is_file() or file_sha256(receipt_path) != digest:
        raise ValueError("v4 stage-inheritance receipt checksum mismatch.")
    source = _source(execution.get("evaluation_source"), label="Current")
    predecessor_namespace = execution.get("superseded_execution_namespace")
    predecessor_sha = execution.get("superseded_execution_receipt_sha256")
    if not isinstance(predecessor_namespace, str) or not isinstance(predecessor_sha, str):
        raise ValueError("Current v4 execution is missing its predecessor identity.")
    typed = verify_stage_inheritance(
        config,
        receipt_path,
        replacement_source=source,
        expected_predecessor=(predecessor_namespace, predecessor_sha),
    )
    if (
        execution.get("schema_version") == "paper-evaluation-execution-v4"
        and typed.get("schema_version") == SCHEMA_V3
    ):
        raise ValueError("Execution v4 cannot resolve report-only inheritance v3.")
    # Native v5 requires its explicit execution-side stage fields; legacy v4
    # routing still accepts its original v1/v2 receipt contract only.
    if execution.get("schema_version") == "paper-evaluation-execution-v5":
        from execsim.ml.paper.evaluation_execution import _validate_inheritance_generation

        _validate_inheritance_generation(execution, typed)
    return typed, receipt_path


def stage_input_root(config: Any, stage: str) -> Path:
    """Resolve a stage input root after verifying v4 inheritance, if present."""
    canonical = _canonical_stage(stage)
    current_root = _evaluation_root(config)
    inherited = _current_inheritance(config)
    if inherited is None or canonical not in tuple(inherited[0].get("inherited_stages", ())):
        return current_root
    receipt, _path = inherited
    namespace, _receipt_sha, _source_identity = _stage_producer_descriptor(
        _artifact_root(config), receipt, canonical
    )
    return _safe_relative(_artifact_root(config), namespace).resolve()


def stage_source(
    config: Any, stage: str, *, source_commit: str, source_tree: str
) -> dict[str, str]:
    """Return the producing source identity for one logical stage."""
    if not _nonempty(source_commit) or not _nonempty(source_tree):
        raise ValueError("Stage source identity must contain non-empty commit and tree.")
    canonical = _canonical_stage(stage)
    inherited = _current_inheritance(config)
    supplied = {"commit": source_commit, "tree": source_tree}
    if inherited is not None and supplied != inherited[0]["replacement_evaluation_source"]:
        raise ValueError("Current evaluator identity does not match the v4 execution receipt.")
    if inherited is not None and canonical in tuple(inherited[0].get("inherited_stages", ())):
        _namespace, _receipt_sha, producer_source = _stage_producer_descriptor(
            _artifact_root(config), inherited[0], canonical
        )
        return producer_source
    return supplied


def stage_provenance(config: Any, *, source_commit: str, source_tree: str) -> dict[str, Any]:
    """Describe mixed stage lineage for report and final-freeze provenance."""
    inherited = _current_inheritance(config)
    if inherited is None:
        return {}
    payload, receipt_path = inherited
    current_source = {"commit": source_commit, "tree": source_tree}
    if current_source != payload["replacement_evaluation_source"]:
        raise ValueError("Current evaluator identity does not match the v4 execution receipt.")
    current_root = _evaluation_root(config)
    current_namespace = _relative(_artifact_root(config), current_root)
    stage_sources = {}
    inherited_stages = tuple(payload.get("inherited_stages", ()))
    for stage in inherited_stages:
        namespace, _receipt_sha, source = _stage_producer_descriptor(
            _artifact_root(config), payload, stage
        )
        stage_sources[stage] = {
            "execution": namespace,
            "execution_namespace": namespace,
            **source,
            "inherited": True,
        }
        if payload.get("schema_version") in {SCHEMA_V2, SCHEMA_V3}:
            immediate_source = payload["superseded_evaluation_source"]
            immediate_namespace = str(payload["superseded_execution_namespace"])
            stage_sources[stage].update(
                {
                    "immediate_predecessor": immediate_namespace,
                    "immediate_predecessor_source": immediate_source,
                }
            )
    for stage in ("run-tca", "report", "final-result-freeze"):
        if stage in inherited_stages:
            continue
        stage_sources[stage] = {
            "execution": current_namespace,
            "execution_namespace": current_namespace,
            **current_source,
            "inherited": False,
        }
    digest = file_sha256(receipt_path)
    return {
        "stage_sources": stage_sources,
        "stage_inheritance_receipt_sha256": digest,
        "stage_inheritance_sha256": digest,
    }


def _stage_inventory(
    config: Any, superseded_execution: Path
) -> tuple[dict[str, Any], dict[str, str]]:
    root = _artifact_root(config)
    execution_path = (
        superseded_execution / "execution.json"
        if superseded_execution.name != "execution.json"
        else superseded_execution
    )
    if not execution_path.is_file():
        raise ValueError("Stage inheritance predecessor execution receipt is unavailable.")
    predecessor = read_json(execution_path)
    source = _source(predecessor.get("evaluation_source"), label="Predecessor")
    namespace = _relative(root, execution_path.parent.resolve())
    if not namespace.startswith("evaluation-executions/"):
        raise ValueError("Stage inheritance predecessor must be below evaluation-executions.")
    inherited = _execution_stage_receipt(config, execution_path, predecessor, source=source)
    if inherited is not None:
        inherited_payload, _ancestor_path, _ancestor_sha = inherited
        if inherited_payload.get("schema_version") == SCHEMA_V3:
            raise ValueError(
                "Report-only stage inheritance cannot be used with the run-tca frontier."
            )
        inventories: dict[str, Any] = {}
        for stage in INHERITED_STAGES:
            producer_ns, producer_sha, producer_source = _stage_producer_descriptor(
                root, inherited_payload, stage
            )
            record = inherited_payload["stage_inventory"][stage]
            if not isinstance(record, Mapping) or record.get("files") is None:
                raise ValueError(f"Inherited {stage} inventory is incomplete.")
            inventories[stage] = {
                "producer_execution_namespace": producer_ns,
                "producer_execution_receipt_sha256": producer_sha,
                "producer_evaluation_source": producer_source,
                "files": record["files"],
            }
        return inventories, source
    forecast = _validate_forecast(root, execution_path.parent.resolve(), config, source)
    representation = _validate_representation(root, execution_path.parent.resolve(), config, source)
    return {
        "evaluate-forecast": {
            "source_execution_namespace": namespace,
            "source_commit": source["commit"],
            "source_tree": source["tree"],
            "files": forecast,
        },
        "evaluate-representation": {
            "source_execution_namespace": namespace,
            "source_commit": source["commit"],
            "source_tree": source["tree"],
            "files": representation,
        },
    }, source


def write_stage_inheritance_receipt(
    config: Any,
    *,
    superseded_execution: Path,
    output: Path,
    replacement_source_commit: str,
    replacement_source_tree: str,
    reason: str,
    invalidation_frontier: str = FRONTIER,
    expected_tca_dates: Mapping[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Write an immutable typed stage receipt.

    The default ``run-tca`` path is the original v1/v2 writer.  Passing the
    explicit ``report`` frontier selects the separately typed v3 report-only
    contract; it never changes the meaning or bytes of an existing v1/v2
    receipt.
    """
    frontier = _canonical_stage(invalidation_frontier)
    if frontier == REPORT_FRONTIER:
        return _write_report_stage_inheritance_receipt(
            config,
            superseded_execution=superseded_execution,
            output=output,
            replacement_source_commit=replacement_source_commit,
            replacement_source_tree=replacement_source_tree,
            reason=reason,
            expected_tca_dates=expected_tca_dates,
        )
    if frontier != FRONTIER:
        raise ValueError("Stage inheritance supports only run-tca or explicit report frontier.")
    root = _artifact_root(config)
    if not output.resolve().is_relative_to(root):
        raise ValueError("Stage inheritance receipt must remain inside the artifact root.")
    if not _nonempty(replacement_source_commit) or not _nonempty(replacement_source_tree):
        raise ValueError("Replacement evaluator identity is incomplete.")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Stage inheritance requires a non-empty reason.")
    execution_path = (
        superseded_execution / "execution.json"
        if superseded_execution.name != "execution.json"
        else superseded_execution
    )
    if not execution_path.resolve().is_relative_to((root / "evaluation-executions").resolve()):
        raise ValueError("Stage inheritance predecessor must be below evaluation-executions.")
    if output.resolve().is_relative_to(execution_path.parent.resolve()):
        raise ValueError("Stage inheritance receipt cannot overwrite its predecessor namespace.")
    predecessor = read_json(execution_path)
    if (
        predecessor.get("status") != "EVALUATION_RESEALED"
        or predecessor.get("schema_version") not in EXECUTION_SCHEMA
        or not isinstance(predecessor.get("upstream_files"), Mapping)
        or not predecessor["upstream_files"]
    ):
        raise ValueError("Stage inheritance predecessor is not a complete resealed execution.")
    freeze_sha, opened_sha, root_source, config_hash = _current_metadata(config)
    if (
        predecessor.get("paper_config_hash") != config_hash
        or predecessor.get("parameter_freeze_sha256") != freeze_sha
    ):
        raise ValueError("Stage inheritance predecessor configuration/freeze mismatch.")
    _validate_predecessor(config, execution_path, file_sha256(execution_path))
    inventories, source = _stage_inventory(config, execution_path)
    predecessor_stage = _execution_stage_receipt(config, execution_path, predecessor, source=source)
    schema = SCHEMA_V2 if predecessor_stage is not None else SCHEMA
    payload: dict[str, Any] = {
        "schema_version": schema,
        "status": STATUS,
        "protocol_id": "sparse-jepa-v2",
        "paper_config_hash": config_hash,
        "parameter_freeze_sha256": freeze_sha,
        "locked_test_open_receipt_sha256": opened_sha,
        "root_evaluation_open_receipt_sha256": opened_sha,
        "root_evaluation_source": root_source,
        "superseded_execution_namespace": _relative(root, execution_path.parent.resolve()),
        "superseded_execution_receipt_sha256": file_sha256(execution_path),
        "superseded_evaluation_source": source,
        "replacement_evaluation_source": {
            "commit": replacement_source_commit,
            "tree": replacement_source_tree,
        },
        "inherited_stages": list(INHERITED_STAGES),
        "invalidation_frontier": FRONTIER,
        "stage_inventory": inventories,
        "reason": reason,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    if predecessor_stage is not None:
        _ancestor_payload, ancestor_path, ancestor_sha = predecessor_stage
        ancestor_namespace = predecessor.get("superseded_execution_namespace")
        ancestor_receipt_sha = predecessor.get("superseded_execution_receipt_sha256")
        if not isinstance(ancestor_namespace, str) or not isinstance(ancestor_receipt_sha, str):
            raise ValueError("Chained predecessor is missing its predecessor identity.")
        payload["ancestor_stage_inheritance"] = {
            "path": _relative(root, ancestor_path),
            "sha256": ancestor_sha,
            "predecessor_namespace": ancestor_namespace,
            "predecessor_receipt_sha256": ancestor_receipt_sha,
        }
    if output.exists():
        existing = read_json(output)
        if {key: value for key, value in existing.items() if key != "created_at_utc"} != {
            key: value for key, value in payload.items() if key != "created_at_utc"
        }:
            raise ValueError("Existing stage inheritance receipt is incompatible.")
        return {**existing, "path": str(output), "sha256": file_sha256(output)}
    write_json_atomic(output, payload)
    return {**payload, "path": str(output), "sha256": file_sha256(output)}


def _write_report_stage_inheritance_receipt(
    config: Any,
    *,
    superseded_execution: Path,
    output: Path,
    replacement_source_commit: str,
    replacement_source_tree: str,
    reason: str,
    expected_tca_dates: Mapping[str, list[str]] | None,
) -> dict[str, Any]:
    """Write the explicit v3 receipt for a report-only recovery."""
    root = _artifact_root(config)
    output = Path(output)
    if not output.resolve().is_relative_to(root):
        raise ValueError("Stage inheritance receipt must remain inside the artifact root.")
    if not _nonempty(replacement_source_commit) or not _nonempty(replacement_source_tree):
        raise ValueError("Replacement evaluator identity is incomplete.")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Stage inheritance requires a non-empty reason.")
    execution_path = (
        Path(superseded_execution) / "execution.json"
        if Path(superseded_execution).name != "execution.json"
        else Path(superseded_execution)
    )
    if not execution_path.resolve().is_relative_to((root / "evaluation-executions").resolve()):
        raise ValueError("Stage inheritance predecessor must be below evaluation-executions.")
    if output.resolve().is_relative_to(execution_path.parent.resolve()):
        raise ValueError("Stage inheritance receipt cannot overwrite its predecessor namespace.")
    if output.is_symlink():
        raise ValueError("Stage inheritance receipt must be a regular file.")
    if execution_path.is_symlink() or not execution_path.is_file():
        raise ValueError("Stage inheritance predecessor execution receipt is unavailable.")
    predecessor = read_json(execution_path)
    if (
        predecessor.get("status") != "EVALUATION_RESEALED"
        or predecessor.get("schema_version") != "paper-evaluation-execution-v4"
        or not isinstance(predecessor.get("upstream_files"), Mapping)
        or not predecessor["upstream_files"]
        or predecessor.get("inherited_stages") != list(INHERITED_STAGES)
        or predecessor.get("invalidation_frontier") != FRONTIER
    ):
        raise ValueError("Report-only inheritance requires a complete v4 TCA predecessor.")
    freeze_sha, opened_sha, root_source, config_hash = _current_metadata(config)
    if (
        predecessor.get("paper_config_hash") != config_hash
        or predecessor.get("parameter_freeze_sha256") != freeze_sha
    ):
        raise ValueError("Stage inheritance predecessor configuration/freeze mismatch.")
    predecessor_sha = file_sha256(execution_path)
    _validate_predecessor(config, execution_path, predecessor_sha)
    source = _source(predecessor.get("evaluation_source"), label="Predecessor")
    predecessor_stage = _execution_stage_receipt(config, execution_path, predecessor, source=source)
    if predecessor_stage is None:
        raise ValueError("Report-only predecessor is missing its typed stage receipt.")
    ancestor_payload, ancestor_path, ancestor_sha = predecessor_stage
    if ancestor_payload.get("schema_version") not in {SCHEMA, SCHEMA_V2}:
        raise ValueError("Report-only predecessor stage receipt is not v1/v2.")
    inventories: dict[str, Any] = {}
    for stage in INHERITED_STAGES:
        producer_ns, producer_receipt_sha, producer_source = _stage_producer_descriptor(
            root, ancestor_payload, stage
        )
        record = ancestor_payload.get("stage_inventory", {}).get(stage)
        if not isinstance(record, Mapping) or not isinstance(record.get("files"), Mapping):
            raise ValueError(f"Report-only predecessor inventory is incomplete for {stage}.")
        inventories[stage] = {
            "producer_execution_namespace": producer_ns,
            "producer_execution_receipt_sha256": producer_receipt_sha,
            "producer_evaluation_source": producer_source,
            "files": dict(record["files"]),
        }
    normalized_tca_dates = _normalize_expected_tca_dates(config, expected_tca_dates)
    tca_files = _validate_tca(
        root,
        execution_path.parent.resolve(),
        config,
        source,
        normalized_tca_dates,
    )
    inventories["run-tca"] = {
        "producer_execution_namespace": _relative(root, execution_path.parent.resolve()),
        "producer_execution_receipt_sha256": predecessor_sha,
        "producer_evaluation_source": source,
        "files": tca_files,
    }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_V3,
        "status": STATUS,
        "protocol_id": "sparse-jepa-v2",
        "paper_config_hash": config_hash,
        "parameter_freeze_sha256": freeze_sha,
        "locked_test_open_receipt_sha256": opened_sha,
        "root_evaluation_open_receipt_sha256": opened_sha,
        "root_evaluation_source": root_source,
        "superseded_execution_namespace": _relative(root, execution_path.parent.resolve()),
        "superseded_execution_receipt_sha256": predecessor_sha,
        "superseded_evaluation_source": source,
        "replacement_evaluation_source": {
            "commit": replacement_source_commit,
            "tree": replacement_source_tree,
        },
        "inherited_stages": list(REPORT_INHERITED_STAGES),
        "invalidation_frontier": REPORT_FRONTIER,
        "expected_tca_dates": normalized_tca_dates,
        "stage_inventory": inventories,
        "ancestor_stage_inheritance": {
            "path": _relative(root, ancestor_path),
            "sha256": ancestor_sha,
            "predecessor_namespace": predecessor.get("superseded_execution_namespace"),
            "predecessor_receipt_sha256": predecessor.get("superseded_execution_receipt_sha256"),
        },
        "reason": reason,
        "created_at_utc": datetime.now(UTC).isoformat(),
    }
    if output.exists():
        existing = read_json(output)
        if {key: value for key, value in existing.items() if key != "created_at_utc"} != {
            key: value for key, value in payload.items() if key != "created_at_utc"
        }:
            raise ValueError("Existing stage inheritance receipt is incompatible.")
        return {**existing, "path": str(output), "sha256": file_sha256(output)}
    write_json_atomic(output, payload)
    return {**payload, "path": str(output), "sha256": file_sha256(output)}


def write_report_stage_inheritance_receipt(
    config: Any,
    *,
    superseded_execution: Path,
    output: Path,
    replacement_source_commit: str,
    replacement_source_tree: str,
    reason: str,
    expected_tca_dates: Mapping[str, list[str]] | None,
) -> dict[str, Any]:
    """Write a v3 receipt with the report-only frontier explicitly selected."""
    return _write_report_stage_inheritance_receipt(
        config,
        superseded_execution=superseded_execution,
        output=output,
        replacement_source_commit=replacement_source_commit,
        replacement_source_tree=replacement_source_tree,
        reason=reason,
        expected_tca_dates=expected_tca_dates,
    )


verify_stage_inheritance_receipt = verify_stage_inheritance


def stage_inheritance_paths(config: Any, receipt: Path) -> tuple[Path, ...]:
    """Return the verified receipt, ancestry, and inherited artifact paths.

    This is used by the execution verifier to bind its own process-local cache
    to the same complete byte set as the stage resolver.
    """
    verify_stage_inheritance(config, receipt)
    key = _cache_key(config, receipt)
    cached = _CACHE.get(key)
    if cached is None:  # pragma: no cover - verify_stage_inheritance always populates it
        raise ValueError("Stage inheritance verification did not produce a state set.")
    return tuple(cached[1])
