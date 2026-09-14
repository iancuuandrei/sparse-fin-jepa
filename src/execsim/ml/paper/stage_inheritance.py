"""Typed, read-only inheritance of completed paper-evaluation stages.

The stage receipt is deliberately small and explicit.  It is not a workflow
engine: v1 can inherit the forecast and representation stages only, and the
invalidation frontier is always ``run-tca``.  The receipt names every byte
that is consumed from the predecessor execution so a replacement evaluator
cannot accidentally reuse a partial TCA result or an unverified directory.
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
STATUS = "STAGE_INHERITANCE_VALIDATED"
EXECUTION_SCHEMA = {
    "paper-evaluation-execution-v2",
    "paper-evaluation-execution-v3",
    "paper-evaluation-execution-v4",
}
FRONTIER = "run-tca"
INHERITED_STAGES = ("evaluate-forecast", "evaluate-representation")
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
    if payload.get("schema_version") != SCHEMA or payload.get("status") != STATUS:
        raise ValueError("Stage inheritance receipt schema/status is incompatible.")
    created_at = payload.get("created_at_utc")
    if not _nonempty(payload.get("reason")) or not _nonempty(created_at):
        raise ValueError("Stage inheritance receipt reason or creation time is incomplete.")
    try:
        timestamp = datetime.fromisoformat(str(created_at))
    except ValueError as exc:
        raise ValueError("Stage inheritance receipt creation time is malformed.") from exc
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("Stage inheritance receipt creation time must be timezone-aware.")
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
    source = _source(payload.get("superseded_evaluation_source"), label="Superseded")
    if source != _source(predecessor.get("evaluation_source"), label="Predecessor"):
        raise ValueError("Stage inheritance predecessor source identity mismatch.")
    replacement = _source(payload.get("replacement_evaluation_source"), label="Replacement")
    if replacement_source is not None and replacement != replacement_source:
        raise ValueError("Stage inheritance replacement evaluator identity mismatch.")
    if (
        expected_predecessor is not None
        and (predecessor_ns, str(predecessor_sha)) != expected_predecessor
    ):
        raise ValueError("Stage inheritance predecessor does not match supersession receipt.")
    inventories = payload.get("stage_inventory")
    if not isinstance(inventories, Mapping) or set(inventories) != set(INHERITED_STAGES):
        raise ValueError(
            "Stage inheritance stage inventory is incomplete or contains extra stages."
        )
    verified_crosslinks: set[Path] = set()
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
    states[receipt_path] = _state(receipt_path)
    states[predecessor_receipt] = _state(predecessor_receipt)
    states.update({path: _state(path) for path in verified_crosslinks if path.is_file()})
    return payload, states


_CACHE: dict[str, tuple[dict[str, Any], dict[Path, tuple[int, int, int, int]]]] = {}


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
    payload, states = _validate_receipt(
        config,
        receipt,
        replacement_source=replacement_source,
        expected_predecessor=expected_predecessor,
    )
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
    if execution.get("schema_version") != "paper-evaluation-execution-v4":
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
    return (
        verify_stage_inheritance(
            config,
            receipt_path,
            replacement_source=source,
            expected_predecessor=(predecessor_namespace, predecessor_sha),
        ),
        receipt_path,
    )


def stage_input_root(config: Any, stage: str) -> Path:
    """Resolve a stage input root after verifying v4 inheritance, if present."""
    canonical = _canonical_stage(stage)
    current_root = _evaluation_root(config)
    inherited = _current_inheritance(config)
    if inherited is None or canonical not in INHERITED_STAGES:
        return current_root
    receipt, _path = inherited
    namespace = str(receipt["superseded_execution_namespace"])
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
    if inherited is not None and canonical in INHERITED_STAGES:
        return {
            "commit": str(inherited[0]["superseded_evaluation_source"]["commit"]),
            "tree": str(inherited[0]["superseded_evaluation_source"]["tree"]),
        }
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
    old_namespace = str(payload["superseded_execution_namespace"])
    old_source = payload["superseded_evaluation_source"]
    stage_sources = {}
    for stage in (*INHERITED_STAGES, "run-tca", "report", "final-result-freeze"):
        inherited_stage = stage in INHERITED_STAGES
        namespace = old_namespace if inherited_stage else current_namespace
        source = old_source if inherited_stage else current_source
        stage_sources[stage] = {
            "execution": namespace,
            "execution_namespace": namespace,
            **source,
            "inherited": inherited_stage,
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
) -> dict[str, Any]:
    """Write an immutable v1 receipt for forecast/representation inheritance."""
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
    payload: dict[str, Any] = {
        "schema_version": SCHEMA,
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
    if output.exists():
        existing = read_json(output)
        if {key: value for key, value in existing.items() if key != "created_at_utc"} != {
            key: value for key, value in payload.items() if key != "created_at_utc"
        }:
            raise ValueError("Existing stage inheritance receipt is incompatible.")
        return {**existing, "path": str(output), "sha256": file_sha256(output)}
    write_json_atomic(output, payload)
    return {**payload, "path": str(output), "sha256": file_sha256(output)}


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
