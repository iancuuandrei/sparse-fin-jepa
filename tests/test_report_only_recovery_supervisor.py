"""Focused proof for the explicit v5/v3 report-only recovery frontier."""

from __future__ import annotations

import argparse
import importlib.util
import json
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest.mock import patch

import pytest
from test_recovery_supervisor import TestRecoverySupervisor as _LegacyRecoverySupervisor

SUPERVISOR_PATH = Path(__file__).parents[1] / "scripts" / "recovery_evaluation_supervisor.py"
SPEC = importlib.util.spec_from_file_location("report_only_recovery_supervisor", SUPERVISOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load supervisor module at {SUPERVISOR_PATH}")
SUPERVISOR: ModuleType = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUPERVISOR)


class FakeProcess:
    def __init__(self, pid: int, returncode: int = 0) -> None:
        self.pid = pid
        self.returncode = returncode

    def poll(self) -> int:
        return self.returncode

    def wait(self) -> int:
        return self.returncode


def _receipt_fixture(tmp_path: Path) -> tuple[SimpleNamespace, dict[str, Any], Path]:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    inheritance_path = artifact_root / "stage-inheritance-v3.json"
    inheritance_path.write_text("{}\n", encoding="utf-8")
    source = {"commit": "replacement-commit", "tree": "replacement-tree"}
    config = SimpleNamespace(artifact_root=artifact_root)
    execution = {
        "schema_version": SUPERVISOR.REPORT_ONLY_EXECUTION_SCHEMA,
        "initial_completed_stages": 0,
        "evaluation_source": source,
        "inherited_stages": list(SUPERVISOR.REPORT_ONLY_INHERITED_STAGES),
        "invalidation_frontier": SUPERVISOR.REPORT_ONLY_INVALIDATION_FRONTIER,
        "stage_inheritance_path": inheritance_path.name,
        "stage_inheritance_sha256": "b" * 64,
        "stage_inheritance_receipt_sha256": "b" * 64,
        "superseded_execution_namespace": "evaluation-executions/prior",
        "superseded_execution_receipt_sha256": "c" * 64,
    }
    return config, execution, inheritance_path


def _report_only_preflight(
    _args: argparse.Namespace,
    *,
    lineage_validated: bool = True,
    missing: str | None = None,
) -> dict[str, Any]:
    """Build the complete production preflight identity used by dispatch tests."""
    identity: dict[str, Any] = {
        "execution_schema": SUPERVISOR.REPORT_ONLY_EXECUTION_SCHEMA,
        "execution_sha256": "e" * 64,
        "approval_id": "fixture-approval",
        "approval_sha256": "a" * 64,
        "inherited_validated": list(SUPERVISOR.REPORT_ONLY_INHERITED_STAGES),
        "invalidation_frontier": SUPERVISOR.REPORT_ONLY_INVALIDATION_FRONTIER,
        "recovery_stages": list(SUPERVISOR.REPORT_ONLY_STAGES),
        "lineage_validated": lineage_validated,
    }
    if missing is not None:
        identity.pop(missing)
    return identity


def test_v5_report_only_preflight_uses_native_verification_and_lineage(tmp_path: Path) -> None:
    config, execution, inheritance_path = _receipt_fixture(tmp_path)
    execution_path = (
        config.artifact_root / "evaluation-executions" / "replacement" / "execution.json"
    )
    execution_path.parent.mkdir(parents=True)
    execution_path.write_text("execution\n", encoding="utf-8")
    source = execution["evaluation_source"]
    provenance = {
        "stage_sources": {
            stage: {
                "execution_namespace": "evaluation-executions/prior",
                "commit": "producer-commit",
                "tree": "producer-tree",
                "inherited": True,
            }
            for stage in SUPERVISOR.REPORT_ONLY_INHERITED_STAGES
        }
        | {
            stage: {
                "execution_namespace": "evaluation-executions/replacement",
                "commit": source["commit"],
                "tree": source["tree"],
                "inherited": False,
            }
            for stage in ("report", "final-result-freeze")
        },
        "stage_inheritance_receipt_sha256": "b" * 64,
        "stage_inheritance_sha256": "b" * 64,
    }
    with (
        patch(
            "execsim.ml.paper.stage_inheritance.verify_stage_inheritance",
            return_value={
                "schema_version": "paper-evaluation-stage-inheritance-v3",
                "status": SUPERVISOR.INHERITANCE_STATUS,
                "inherited_stages": list(SUPERVISOR.REPORT_ONLY_INHERITED_STAGES),
                "invalidation_frontier": SUPERVISOR.REPORT_ONLY_INVALIDATION_FRONTIER,
                "replacement_evaluation_source": source,
                "stage_inventory": {
                    stage: {"files": {"member": "a" * 64}}
                    for stage in SUPERVISOR.REPORT_ONLY_INHERITED_STAGES
                },
            },
        ) as verify,
        patch(
            "execsim.ml.paper.stage_inheritance.stage_provenance",
            return_value=provenance,
        ) as native_provenance,
        patch(
            "execsim.data.paper.manifests.file_sha256",
            return_value="b" * 64,
        ),
    ):
        typed, validated_provenance = SUPERVISOR._validate_v5_report_only_inheritance(
            config,
            execution,
            execution_path,
            source_commit=source["commit"],
            source_tree=source["tree"],
        )

    assert typed["schema_version"] == "paper-evaluation-stage-inheritance-v3"
    assert validated_provenance["stage_sources"]["run-tca"]["inherited"] is True
    verify.assert_called_once_with(
        config,
        inheritance_path,
        replacement_source=source,
        expected_predecessor=("evaluation-executions/prior", "c" * 64),
    )
    native_provenance.assert_called_once_with(
        config,
        source_commit=source["commit"],
        source_tree=source["tree"],
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("evaluation_source", {"commit": "wrong", "tree": "wrong-tree"}),
        ("inherited_stages", list(SUPERVISOR.INHERITED_STAGES)),
        ("invalidation_frontier", SUPERVISOR.INVALIDATION_FRONTIER),
    ],
)
def test_v5_report_only_rejects_incompatible_execution_identity(
    tmp_path: Path, field: str, value: Any
) -> None:
    config, execution, _inheritance_path = _receipt_fixture(tmp_path)
    execution[field] = value
    execution_path = config.artifact_root / "execution.json"
    execution_path.write_text("execution\n", encoding="utf-8")
    source = {"commit": "replacement-commit", "tree": "replacement-tree"}
    with (
        patch("execsim.ml.paper.stage_inheritance.verify_stage_inheritance") as verify,
        patch("execsim.ml.paper.stage_inheritance.stage_provenance") as provenance,
    ):
        with pytest.raises(ValueError):
            SUPERVISOR._validate_v5_report_only_inheritance(
                config,
                execution,
                execution_path,
                source_commit=source["commit"],
                source_tree=source["tree"],
            )
    verify.assert_not_called()
    provenance.assert_not_called()


class TestReportOnlyDispatch:
    def setup_method(self) -> None:
        _LegacyRecoverySupervisor.setup_method(self)
        self.args.report_only = True

    def teardown_method(self) -> None:
        _LegacyRecoverySupervisor.teardown_method(self)

    @contextmanager
    def unlocked(self, _runtime_root: Path):
        yield

    def test_report_only_plan_never_dispatches_run_tca(self) -> None:
        launched: list[list[str]] = []
        next_pid = 5000

        def spawn(command: list[str], **_kwargs: Any) -> FakeProcess:
            nonlocal next_pid
            launched.append(command)
            process = FakeProcess(next_pid)
            next_pid += 1
            return process

        preflight = partial(_report_only_preflight, lineage_validated=True)

        supervisor = SUPERVISOR.StageSupervisor(
            self.args,
            source_verifier=lambda _args: None,
            preflight=preflight,
            lock_factory=self.unlocked,
            popen_factory=spawn,
        )
        assert supervisor.run() == 0
        stages = [command[4] for command in launched]
        assert stages == list(SUPERVISOR.REPORT_ONLY_STAGES)
        assert "run-tca" not in stages
        status = json.loads(supervisor.status_path.read_text(encoding="utf-8"))
        assert status["status"] == "COMPLETED"
        assert status["inherited_validated"] == list(SUPERVISOR.REPORT_ONLY_INHERITED_STAGES)
        assert status["recovery_stages"] == list(SUPERVISOR.REPORT_ONLY_STAGES)
        assert status["lineage_validated"] is True

    def test_report_only_plan_rejects_unverified_lineage_before_children(self) -> None:
        launched: list[list[str]] = []

        def spawn(command: list[str], **_kwargs: Any) -> FakeProcess:
            launched.append(command)
            return FakeProcess(5100)

        preflight = partial(_report_only_preflight, lineage_validated=False)

        supervisor = SUPERVISOR.StageSupervisor(
            self.args,
            source_verifier=lambda _args: None,
            preflight=preflight,
            lock_factory=self.unlocked,
            popen_factory=spawn,
        )
        assert supervisor.run() == 1
        assert launched == []
        status = json.loads(supervisor.status_path.read_text(encoding="utf-8"))
        assert status["status"] == "FAILED"
        assert status["completed_here"] == []
        assert "lineage" in status["last_error"]

    def test_v5_receipt_requires_explicit_report_only_opt_in(self) -> None:
        launched: list[list[str]] = []

        def spawn(command: list[str], **_kwargs: Any) -> FakeProcess:
            launched.append(command)
            return FakeProcess(5200)

        self.args.report_only = False

        preflight = partial(_report_only_preflight, lineage_validated=True)

        supervisor = SUPERVISOR.StageSupervisor(
            self.args,
            source_verifier=lambda _args: None,
            preflight=preflight,
            lock_factory=self.unlocked,
            popen_factory=spawn,
        )
        assert supervisor.run() == 1
        assert launched == []
        status = json.loads(supervisor.status_path.read_text(encoding="utf-8"))
        assert status["status"] == "FAILED"
        assert "explicitly requested legacy mode" in status["last_error"]

    @pytest.mark.parametrize(
        "missing",
        ["inherited_validated", "invalidation_frontier", "recovery_stages", "lineage_validated"],
    )
    def test_report_only_preflight_does_not_default_missing_identity_fields(
        self, missing: str
    ) -> None:
        launched: list[list[str]] = []

        def spawn(command: list[str], **_kwargs: Any) -> FakeProcess:
            launched.append(command)
            return FakeProcess(5300)

        preflight = partial(_report_only_preflight, missing=missing)

        supervisor = SUPERVISOR.StageSupervisor(
            self.args,
            source_verifier=lambda _args: None,
            preflight=preflight,
            lock_factory=self.unlocked,
            popen_factory=spawn,
        )
        assert supervisor.run() == 1
        assert launched == []
        status = json.loads(supervisor.status_path.read_text(encoding="utf-8"))
        assert status["status"] == "FAILED"
        assert missing in status["last_error"]
