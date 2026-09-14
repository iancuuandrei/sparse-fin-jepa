"""Focused control-loop tests for the v4 evaluation-recovery supervisor."""

from __future__ import annotations

import argparse
import ast
import importlib.util
import json
import signal
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch

import pytest

SUPERVISOR_PATH = Path(__file__).parents[1] / "scripts" / "recovery_evaluation_supervisor.py"
SPEC = importlib.util.spec_from_file_location("recovery_evaluation_supervisor", SUPERVISOR_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"Cannot load supervisor module at {SUPERVISOR_PATH}")
SUPERVISOR: ModuleType = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUPERVISOR)


@pytest.mark.parametrize(
    "schema",
    [
        "paper-evaluation-stage-inheritance-v1",
        "paper-evaluation-stage-inheritance-v2",
        "unknown-stage-schema",
    ],
)
def test_supervisor_requires_verified_supported_inheritance_schema(tmp_path, schema):
    from types import SimpleNamespace

    from execsim.data.paper.manifests import file_sha256

    path = tmp_path / "inheritance.json"
    path.write_text("{}", encoding="utf-8")
    source = {"commit": "replacement", "tree": "replacement-tree"}
    stages = list(SUPERVISOR.INHERITED_STAGES)
    execution = {
        "schema_version": SUPERVISOR.EXECUTION_SCHEMA,
        "initial_completed_stages": 0,
        "evaluation_source": source,
        "inherited_stages": stages,
        "invalidation_frontier": "run-tca",
        "stage_inheritance_path": path.name,
        "stage_inheritance_sha256": file_sha256(path),
        "superseded_execution_namespace": "evaluation-executions/prior",
        "superseded_execution_receipt_sha256": "prior-sha",
    }
    typed = {
        "schema_version": schema,
        "status": SUPERVISOR.INHERITANCE_STATUS,
        "inherited_stages": stages,
        "invalidation_frontier": "run-tca",
        "replacement_evaluation_source": source,
        "stage_inventory": {stage: {"files": {"producer/member": "sha"}} for stage in stages},
    }
    with patch(
        "execsim.ml.paper.stage_inheritance.verify_stage_inheritance", return_value=typed
    ) as verifier:
        args = (SimpleNamespace(artifact_root=tmp_path), execution, tmp_path / "execution.json")
        options = {"source_commit": source["commit"], "source_tree": source["tree"]}
        if schema == "unknown-stage-schema":
            with pytest.raises(ValueError, match="permitted v4 contract"):
                SUPERVISOR._require_v4_inheritance(*args, **options)
        else:
            assert SUPERVISOR._require_v4_inheritance(*args, **options) == typed
        verifier.assert_called_once_with(
            args[0],
            path,
            replacement_source=source,
            expected_predecessor=("evaluation-executions/prior", "prior-sha"),
        )


class FakeProcess:
    def __init__(self, pid: int, returncode: int, wait_hook: Any = None) -> None:
        self.pid = pid
        self.returncode = returncode
        self.wait_hook = wait_hook

    def poll(self) -> int | None:
        return None if self.returncode == -999 else self.returncode

    def wait(self) -> int:
        if self.wait_hook is not None:
            hook, self.wait_hook = self.wait_hook, None
            hook()
        if self.returncode == -999:
            self.returncode = 0
        return self.returncode


class TestRecoverySupervisor:
    def setup_method(self) -> None:
        self.environment_patch = patch.dict("os.environ")
        self.environment_patch.start()
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.source = root / "source"
        self.artifact_root = root / "artifacts"
        self.evaluation_root = self.artifact_root / "evaluation-executions" / "recovery"
        self.runtime_root = root / "runtime"
        self.data_root = root / "data"
        self.representation_root = self.artifact_root / "representations"
        self.config = self.source / "configs" / "paper" / "sparse_jepa_v2" / "data.yaml"
        self.approval = self.runtime_root / "paper-approvals" / "evaluation.json"
        for path in (
            self.source,
            self.config.parent,
            self.artifact_root,
            self.evaluation_root,
            self.runtime_root,
            self.runtime_root / "paper-approvals",
            self.data_root,
            self.representation_root,
        ):
            path.mkdir(parents=True, exist_ok=True)
        self.config.write_text("fixture only\n", encoding="utf-8")
        self.approval.write_text("fixture only\n", encoding="utf-8")
        self.args = argparse.Namespace(
            command="run",
            source=self.source,
            commit="c" * 40,
            tree="t" * 40,
            artifact_root=self.artifact_root,
            data_root=self.data_root,
            representation_root=self.representation_root,
            evaluation_root=self.evaluation_root,
            runtime_root=self.runtime_root,
            approval=self.approval,
            config=self.config,
            operational_workers=3,
            status_path=None,
        )

    def teardown_method(self) -> None:
        self.temp.cleanup()
        self.environment_patch.stop()

    @contextmanager
    def unlocked(self, _runtime_root: Path) -> Iterator[None]:
        yield

    def supervisor(self, popen_factory: Any, *, preflight: Any = None) -> Any:
        return SUPERVISOR.StageSupervisor(
            self.args,
            source_verifier=lambda _args: None,
            preflight=preflight
            or (
                lambda _args: {
                    "execution_schema": SUPERVISOR.EXECUTION_SCHEMA,
                    "execution_sha256": "e" * 64,
                    "approval_id": "fixture-approval",
                    "approval_sha256": "a" * 64,
                    "inherited_validated": list(SUPERVISOR.INHERITED_STAGES),
                    "invalidation_frontier": SUPERVISOR.INVALIDATION_FRONTIER,
                }
            ),
            lock_factory=self.unlocked,
            popen_factory=popen_factory,
        )

    def test_runs_only_three_explicit_downstream_stages_in_order(self) -> None:
        launched: list[tuple[list[str], dict[str, Any]]] = []
        next_pid = 4100

        def spawn(command: list[str], **kwargs: Any) -> FakeProcess:
            nonlocal next_pid
            launched.append((command, kwargs))
            process = FakeProcess(next_pid, 0)
            next_pid += 1
            return process

        supervisor = self.supervisor(spawn)
        with patch.dict("os.environ", {}, clear=True):
            result = supervisor.run()
        assert result == 0
        assert [command[4] for command, _ in launched] == list(SUPERVISOR.STAGES)
        assert all(kwargs["start_new_session"] for _, kwargs in launched)
        assert all("evaluate_forecasts_stage" not in command[3] for command, _ in launched)
        status = json.loads(supervisor.status_path.read_text(encoding="utf-8"))
        assert status["status"] == "COMPLETED"
        assert status["inherited_validated"] == list(SUPERVISOR.INHERITED_STAGES)
        assert status["completed_here"] == list(SUPERVISOR.STAGES)
        assert status["current_stage"] is None
        assert status["execution_sha256"] == "e" * 64
        assert status["approval_sha256"] == "a" * 64

    def test_preflight_failure_blocks_all_children(self) -> None:
        launches: list[list[str]] = []

        def spawn(command: list[str], **_kwargs: Any) -> FakeProcess:
            launches.append(command)
            return FakeProcess(4200, 0)

        supervisor = self.supervisor(
            spawn,
            preflight=lambda _args: (_ for _ in ()).throw(ValueError("typed inheritance mismatch")),
        )
        result = supervisor.run()
        assert result == 1
        assert launches == []
        status = json.loads(supervisor.status_path.read_text(encoding="utf-8"))
        assert status["status"] == "FAILED"
        assert status["completed_here"] == []
        assert status["current_stage"] is None
        assert "typed inheritance mismatch" in status["last_error"]

    def test_nonzero_stage_stops_later_stages_and_preserves_status(self) -> None:
        launched: list[str] = []

        def spawn(command: list[str], **_kwargs: Any) -> FakeProcess:
            launched.append(command[4])
            return FakeProcess(4300, 23 if len(launched) == 1 else 0)

        supervisor = self.supervisor(spawn)
        result = supervisor.run()
        assert result == 23
        assert launched == ["run-tca"]
        status = json.loads(supervisor.status_path.read_text(encoding="utf-8"))
        assert status["status"] == "FAILED"
        assert status["current_stage"] == "run-tca"
        assert status["completed_here"] == []
        assert status["exit_code"] == 23

    def test_sigterm_targets_only_active_child_group(self) -> None:
        process_holder: list[FakeProcess] = []
        supervisor_holder: list[Any] = []

        def spawn(_command: list[str], **_kwargs: Any) -> FakeProcess:
            def deliver_term() -> None:
                supervisor_holder[0]._handle_signal(signal.SIGTERM, None)

            process = FakeProcess(4400, -999, wait_hook=deliver_term)
            process_holder.append(process)
            return process

        supervisor = self.supervisor(spawn)
        supervisor_holder.append(supervisor)

        def kill_group(group_id: int, signum: int) -> None:
            assert group_id == process_holder[0].pid
            assert signum == signal.SIGTERM
            process_holder[0].returncode = -signal.SIGTERM

        with patch.object(SUPERVISOR.os, "killpg", side_effect=kill_group, create=True) as killpg:
            result = supervisor.run()
        assert killpg.call_count == 1
        assert result == 128 + signal.SIGTERM
        status = json.loads(supervisor.status_path.read_text(encoding="utf-8"))
        assert status["status"] == "INTERRUPTED"

    def test_stage_program_is_valid_and_has_no_generic_dispatch(self) -> None:
        ast.parse(SUPERVISOR._STAGE_PROGRAM)
        assert "run_authorized_stages" not in SUPERVISOR._STAGE_PROGRAM
        assert "evaluate_forecasts_stage" not in SUPERVISOR._STAGE_PROGRAM
        assert "evaluate_representations_stage" not in SUPERVISOR._STAGE_PROGRAM
