"""Real v4 seal and verification coverage for the narrow recovery contract."""

from __future__ import annotations

import importlib.util
import json
import shutil
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from execsim.data.paper.manifests import write_json_atomic
from execsim.ml.paper.evaluation_execution import (
    STAGE_INHERITED_SCHEMA,
    seal_evaluation_execution,
    verify_evaluation_execution,
    write_evaluation_supersession_receipt,
)


def _load_inheritance_fixture_module() -> ModuleType:
    """Reuse the canonical complete-artifact fixture maintained with inheritance tests."""
    fixture_path = Path(__file__).with_name("test_stage_inheritance.py")
    spec = importlib.util.spec_from_file_location("stage_inheritance_fixture", fixture_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load fixture module at {fixture_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def inherited_run(tmp_path: Path) -> Any:
    """Build the complete immutable predecessor used by the production receipt verifier."""
    module = _load_inheritance_fixture_module()
    return module.inherited_run.__wrapped__(tmp_path)


def _clear_destination(config: Any) -> Path:
    destination = config.runtime_evaluation_root
    destination.mkdir(parents=True, exist_ok=True)
    for child in destination.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()
    return destination


def _prepare_chain(config: Any, old: Path, inheritance: Path) -> Path:
    """Create the typed root -> prior -> replacement chain input."""
    supersession = config.artifact_root / "superseded-prior.json"
    write_evaluation_supersession_receipt(
        config,
        superseded_execution=old,
        output=supersession,
        replacement_source_commit="new-commit",
        replacement_source_tree="new-tree",
        reason="recover only the invalidated TCA frontier",
    )
    _clear_destination(config)
    assert inheritance.is_file()
    return supersession


def test_real_seal_and_verify_v4_preserve_root_prior_and_inherited_sources(inherited_run):
    config, old, inheritance, _opened_sha = inherited_run
    supersession = _prepare_chain(config, old, inheritance)

    receipt = seal_evaluation_execution(
        config,
        source_commit="new-commit",
        source_tree="new-tree",
        supersession=supersession,
        inheritance=inheritance,
    )
    assert receipt["schema_version"] == STAGE_INHERITED_SCHEMA
    assert receipt["evaluation_source"] == {"commit": "new-commit", "tree": "new-tree"}
    assert receipt["previous_evaluation_source"] == {
        "commit": "old-commit",
        "tree": "old-tree",
    }
    assert receipt["root_evaluation_source"] == {"commit": "root", "tree": "root-tree"}
    assert receipt["superseded_execution_namespace"] == "evaluation-executions/old"
    assert receipt["inherited_stages"] == ["evaluate-forecast", "evaluate-representation"]
    assert receipt["invalidation_frontier"] == "run-tca"
    assert (
        verify_evaluation_execution(config, source_commit="new-commit", source_tree="new-tree")
        == receipt
    )

    inherited_member = old / "evaluation-v2/representations/fold-1/dense/13/accessibility.parquet"
    inherited_member.write_bytes(inherited_member.read_bytes() + b"changed")
    with pytest.raises(ValueError, match=r"checksum|changed"):
        verify_evaluation_execution(config, source_commit="new-commit", source_tree="new-tree")


@pytest.mark.parametrize("fault", ["populated", "wrong-predecessor", "wrong-source", "wrong-root"])
def test_v4_reseal_fails_closed_for_destination_or_chain_mismatch(inherited_run, fault):
    config, old, inheritance, _opened_sha = inherited_run
    supersession = _prepare_chain(config, old, inheritance)

    if fault == "populated":
        (config.runtime_evaluation_root / "partial-tca.parquet").write_bytes(b"partial")
    elif fault == "wrong-predecessor":
        payload = json.loads(inheritance.read_text(encoding="utf-8"))
        payload["superseded_execution_namespace"] = "evaluation-executions/not-prior"
        write_json_atomic(inheritance, payload)
    elif fault == "wrong-source":
        payload = json.loads(inheritance.read_text(encoding="utf-8"))
        payload["replacement_evaluation_source"] = {
            "commit": "foreign-commit",
            "tree": "foreign-tree",
        }
        write_json_atomic(inheritance, payload)
    else:
        payload = json.loads(supersession.read_text(encoding="utf-8"))
        payload["root_evaluation_source"] = {
            "commit": "foreign-root",
            "tree": "foreign-root-tree",
        }
        write_json_atomic(supersession, payload)

    with pytest.raises((ValueError, RuntimeError)):
        seal_evaluation_execution(
            config,
            source_commit="new-commit",
            source_tree="new-tree",
            supersession=supersession,
            inheritance=inheritance,
        )
    assert not (config.runtime_evaluation_root / "execution.json").exists()
