#!/usr/bin/env python3
"""Run the narrow v4 evaluation-recovery frontier.

The supervisor validates a typed forecast/representation inheritance receipt
before starting any worker.  It then runs exactly ``run-tca``, ``report``, and
``final-result-freeze`` as isolated child processes.  It is intentionally an
operational wrapper: it does not acquire data, train models, create approvals,
reseal an execution, or infer a workflow from partial artifacts.

Run this program in the foreground.  Operators can keep it detached with a
process supervisor such as tmux or systemd; this module does not provision
remote sessions.  ``status`` reads the last atomic status file and ``stop``
sends SIGTERM to the verified supervisor PID.  The supervisor forwards that
signal only to its active child process group.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import importlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any

STAGES = ("run-tca", "report", "final-result-freeze")
INHERITED_STAGES = ("evaluate-forecast", "evaluate-representation")
INVALIDATION_FRONTIER = "run-tca"
EXECUTION_SCHEMA = "paper-evaluation-execution-v4"
INHERITANCE_SCHEMAS = {
    "paper-evaluation-stage-inheritance-v1",
    "paper-evaluation-stage-inheritance-v2",
}
INHERITANCE_STATUS = "STAGE_INHERITANCE_VALIDATED"
DEFAULT_STATUS_NAME = "recovery-evaluation-supervisor-status.json"
DEFAULT_WORKERS = 16
NATIVE_THREAD_VARIABLES = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
SENSITIVE_ENV_SUFFIXES = (
    "_API_KEY",
    "_SECRET",
    "_TOKEN",
    "_PASSWORD",
    "_CREDENTIAL",
    "_CREDENTIALS",
    "_PRIVATE_KEY",
    "_ACCESS_KEY",
    "_ACCESS_KEY_ID",
)


class SupervisorStop(Exception):
    """Internal control flow after SIGTERM or SIGINT."""

    def __init__(self, signum: int) -> None:
        super().__init__(f"termination requested by signal {signum}")
        self.signum = signum
        self.exit_code = 128 + signum


class StageFailure(Exception):
    """A child stage returned a nonzero exit code."""

    def __init__(self, stage: str, exit_code: int) -> None:
        super().__init__(f"stage {stage} exited with status {exit_code}")
        self.stage = stage
        self.exit_code = exit_code


def _utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate v4 forecast/representation inheritance, then run only "
            "run-tca, report, and final-result-freeze."
        )
    )
    parser.add_argument(
        "command",
        nargs="?",
        choices=("run", "status", "stop"),
        default="run",
        help="run in the foreground, or inspect/stop an existing run (default: run).",
    )
    parser.add_argument("--source", type=Path, default=None, help="Exact clean source checkout.")
    parser.add_argument("--commit", default=None, help="Expected source Git commit.")
    parser.add_argument("--tree", default=None, help="Expected source Git tree.")
    parser.add_argument(
        "--artifact-root",
        "--artifacts",
        dest="artifact_root",
        type=Path,
        default=None,
        help="Paper artifact root containing evaluation-executions.",
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--representation-root", type=Path, default=None)
    parser.add_argument("--evaluation-root", type=Path, default=None)
    parser.add_argument("--runtime-root", type=Path, default=None)
    parser.add_argument(
        "--approval",
        "--runtime-approval",
        dest="approval",
        type=Path,
        default=None,
        help="Evaluation-only runtime approval below runtime-root/paper-approvals.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=(
            "Paper config directory/file (default: source/configs/paper/sparse_jepa_v2/data.yaml)."
        ),
    )
    parser.add_argument("--operational-workers", type=int, default=None)
    parser.add_argument(
        "--status-path",
        type=Path,
        default=None,
        help=f"Status JSON path (default: runtime-root/{DEFAULT_STATUS_NAME}).",
    )
    return parser


def _required_directory(path: Path | None, name: str) -> Path:
    if path is None:
        raise ValueError(f"{name} is required for run.")
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{name} must be an existing directory: {resolved}")
    return resolved


def _required_file(path: Path | None, name: str) -> Path:
    if path is None:
        raise ValueError(f"{name} is required for run.")
    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{name} must be an existing regular file: {resolved}")
    return resolved


def _resolve_arguments(args: argparse.Namespace) -> argparse.Namespace:
    """Resolve and constrain all run paths without creating scientific state."""
    args.source = _required_directory(getattr(args, "source", None), "source")
    args.artifact_root = _required_directory(getattr(args, "artifact_root", None), "artifact-root")
    args.data_root = _required_directory(getattr(args, "data_root", None), "data-root")
    args.representation_root = _required_directory(
        getattr(args, "representation_root", None), "representation-root"
    )
    args.evaluation_root = _required_directory(
        getattr(args, "evaluation_root", None), "evaluation-root"
    )
    args.runtime_root = _required_directory(getattr(args, "runtime_root", None), "runtime-root")

    executions_root = (args.artifact_root / "evaluation-executions").resolve()
    if not args.evaluation_root.is_relative_to(executions_root):
        raise ValueError(
            "evaluation-root must be an existing namespace below "
            "artifact-root/evaluation-executions/."
        )
    if not args.representation_root.is_relative_to(args.artifact_root):
        raise ValueError("representation-root must remain inside artifact-root.")

    if getattr(args, "config", None) is None:
        args.config = args.source / "configs/paper/sparse_jepa_v2/data.yaml"
    args.config = args.config.expanduser().resolve(strict=True)
    config_root = args.config if args.config.is_dir() else args.config.parent
    if not config_root.is_relative_to(args.source):
        raise ValueError("config must remain inside the exact source checkout.")

    args.approval = _required_file(getattr(args, "approval", None), "approval")
    approval_root = (args.runtime_root / "paper-approvals").resolve()
    if not args.approval.is_relative_to(approval_root):
        raise ValueError("approval must remain below runtime-root/paper-approvals/.")
    workers = getattr(args, "operational_workers", None)
    if workers is not None and workers < 1:
        raise ValueError("operational-workers must be positive.")
    args.status_path = _status_path(args)
    return args


def _status_path(args: argparse.Namespace) -> Path:
    runtime_root = getattr(args, "runtime_root", None)
    requested = getattr(args, "status_path", None)
    if requested is not None:
        path = requested.expanduser().resolve()
        if runtime_root is not None:
            root = Path(runtime_root).expanduser().resolve()
            if not path.is_relative_to(root):
                raise ValueError("status-path must remain below runtime-root.")
        return path
    if runtime_root is None:
        raise ValueError("runtime-root is required for status or stop.")
    return Path(runtime_root).expanduser().resolve() / DEFAULT_STATUS_NAME


def _git(source: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(source), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _verify_source_checkout(args: argparse.Namespace) -> None:
    """Require the exact committed source checkout and no tracked/untracked changes."""
    root = Path(_git(args.source, "rev-parse", "--show-toplevel")).resolve(strict=True)
    if root != args.source:
        raise ValueError(f"source is not the Git checkout root: {args.source}")
    actual_commit = _git(args.source, "rev-parse", "HEAD")
    actual_tree = _git(args.source, "rev-parse", "HEAD^{tree}")
    if actual_commit != args.commit or actual_tree != args.tree:
        raise ValueError(
            "source commit/tree do not match the requested identity "
            f"(actual {actual_commit}/{actual_tree})."
        )
    changes = _git(args.source, "status", "--porcelain=v1", "--untracked-files=all")
    if changes:
        raise ValueError(f"source checkout is not clean: {changes.splitlines()[:20]}")


def _safe_receipt_path(root: Path, relative: object, name: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise ValueError(f"{name} path is missing from the v4 execution receipt.")
    candidate = (root / relative).resolve(strict=True)
    if not candidate.is_file() or not candidate.is_relative_to(root):
        raise ValueError(f"{name} must be a regular file below artifact-root.")
    return candidate


def _require_v4_inheritance(
    config: Any,
    execution: dict[str, Any],
    execution_path: Path,
    *,
    source_commit: str,
    source_tree: str,
) -> dict[str, Any]:
    """Validate the typed v4 receipt and force verification of every inherited byte."""
    if execution.get("schema_version") != EXECUTION_SCHEMA:
        raise ValueError("Recovery supervisor requires paper-evaluation-execution-v4.")
    if execution.get("initial_completed_stages") != 0:
        raise ValueError("v4 recovery execution must start with zero newly completed stages.")
    if execution.get("evaluation_source") != {"commit": source_commit, "tree": source_tree}:
        raise ValueError("v4 execution source identity does not match the requested source.")
    if execution.get("inherited_stages") != list(INHERITED_STAGES):
        raise ValueError("v4 execution must inherit forecast and representation only.")
    if execution.get("invalidation_frontier") != INVALIDATION_FRONTIER:
        raise ValueError("v4 execution frontier must be run-tca.")

    root = config.artifact_root.resolve(strict=True)
    inheritance_path = _safe_receipt_path(
        root, execution.get("stage_inheritance_path"), "stage-inheritance receipt"
    )
    from execsim.data.paper.manifests import file_sha256

    inheritance_sha = file_sha256(inheritance_path)
    if (
        execution.get("stage_inheritance_sha256") != inheritance_sha
        or execution.get("stage_inheritance_receipt_sha256", inheritance_sha) != inheritance_sha
    ):
        raise ValueError("v4 stage-inheritance receipt checksum mismatch.")

    from execsim.ml.paper.stage_inheritance import verify_stage_inheritance

    expected_predecessor = (
        execution.get("superseded_execution_namespace"),
        execution.get("superseded_execution_receipt_sha256"),
    )
    typed = verify_stage_inheritance(
        config,
        inheritance_path,
        replacement_source={"commit": source_commit, "tree": source_tree},
        expected_predecessor=expected_predecessor,
    )
    if (
        typed.get("schema_version") not in INHERITANCE_SCHEMAS
        or typed.get("status") != INHERITANCE_STATUS
        or typed.get("inherited_stages") != list(INHERITED_STAGES)
        or typed.get("invalidation_frontier") != INVALIDATION_FRONTIER
        or typed.get("replacement_evaluation_source")
        != {"commit": source_commit, "tree": source_tree}
    ):
        raise ValueError("Typed stage inheritance receipt is not the permitted v4 contract.")
    inventory = typed.get("stage_inventory")
    if not isinstance(inventory, dict) or set(inventory) != set(INHERITED_STAGES):
        raise ValueError("Typed inheritance inventory must contain exactly two stages.")
    for stage in INHERITED_STAGES:
        record = inventory.get(stage)
        if not isinstance(record, dict) or not isinstance(record.get("files"), dict):
            raise ValueError(f"Typed inheritance inventory is incomplete for {stage}.")
        if not record["files"]:
            raise ValueError(f"Typed inheritance inventory is empty for {stage}.")
    if execution_path.resolve() == inheritance_path.resolve():
        raise ValueError("Execution and stage-inheritance receipts must be distinct files.")
    return typed


def _preflight_execution(args: argparse.Namespace) -> dict[str, Any]:
    """Import only from the requested checkout and verify v4 before launch."""
    source_src = args.source / "src"
    package_init = source_src / "execsim" / "__init__.py"
    if not package_init.is_file():
        raise FileNotFoundError(f"ExecSim package is missing below {source_src}")
    sys.dont_write_bytecode = True
    importlib.invalidate_caches()
    sys.path.insert(0, str(source_src))
    existing = sys.modules.get("execsim")
    if existing is not None:
        existing_file = getattr(existing, "__file__", None)
        if not isinstance(existing_file, str) or not Path(existing_file).resolve().is_relative_to(
            source_src.resolve(strict=True)
        ):
            raise ImportError(
                "ExecSim is already imported from outside the requested source checkout."
            )
    import execsim

    imported_from = Path(execsim.__file__).resolve(strict=True)
    if not imported_from.is_relative_to(source_src.resolve(strict=True)):
        raise ImportError(
            f"ExecSim imported outside the requested source checkout: {imported_from}"
        )

    from execsim.data.paper.manifests import file_sha256
    from execsim.ml.paper.configs import load_paper_config, load_runtime_approval
    from execsim.ml.paper.evaluation_execution import verify_evaluation_execution

    config = load_paper_config(args.config).with_runtime_roots(
        artifact_root=args.artifact_root,
        data_root=args.data_root,
        cache_root=args.runtime_root,
        evaluation_root=args.evaluation_root,
        representation_root=args.representation_root,
    )
    approval = load_runtime_approval(args.approval, config)
    _require_evaluation_only_approval(approval)
    config.authorize("locked_result_evaluation", approval=approval, cli_enabled=True)

    execution_path = args.evaluation_root / "execution.json"
    if execution_path.is_symlink() or not execution_path.is_file():
        raise ValueError("v4 execution receipt must be a regular file in evaluation-root.")
    # This verifier binds the execution v4 receipt, supersession, parameter
    # freeze, typed inheritance receipt, and every inherited manifest member.
    execution = verify_evaluation_execution(
        config, source_commit=args.commit, source_tree=args.tree
    )
    typed = _require_v4_inheritance(
        config,
        execution,
        execution_path,
        source_commit=args.commit,
        source_tree=args.tree,
    )
    return {
        "execution_schema": execution["schema_version"],
        "execution_receipt": str(execution_path),
        "execution_sha256": file_sha256(execution_path),
        "approval_id": approval.approval_id,
        "approval_path": str(args.approval),
        "approval_sha256": file_sha256(args.approval),
        "stage_inheritance_receipt": str(
            _safe_receipt_path(
                args.artifact_root.resolve(strict=True),
                execution["stage_inheritance_path"],
                "stage-inheritance receipt",
            )
        ),
        "stage_inheritance_sha256": file_sha256(
            _safe_receipt_path(
                args.artifact_root.resolve(strict=True),
                execution["stage_inheritance_path"],
                "stage-inheritance receipt",
            )
        ),
        "inherited_validated": list(typed["inherited_stages"]),
        "invalidation_frontier": typed["invalidation_frontier"],
        "source_import": str(imported_from),
    }


def _require_evaluation_only_approval(approval: Any) -> None:
    if approval.approved_operations != frozenset({"locked_result_evaluation"}):
        raise PermissionError(
            "Recovery supervisor requires evaluation-only approval; acquisition and "
            "training must be false."
        )


def _worker_count(args: argparse.Namespace) -> int:
    explicit = getattr(args, "operational_workers", None)
    inherited = os.environ.get("EXECSIM_EVALUATION_WORKERS")
    count = explicit if explicit is not None else (int(inherited) if inherited else DEFAULT_WORKERS)
    if count < 1:
        raise ValueError("operational-workers must be positive.")
    return count


def _child_environment(source: Path, workers: int) -> dict[str, str]:
    """Keep the qualified runtime while excluding recognizable credentials."""
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().endswith(SENSITIVE_ENV_SUFFIXES)
    }
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(source / "src") + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["EXECSIM_EVALUATION_WORKERS"] = str(workers)
    for variable in NATIVE_THREAD_VARIABLES:
        environment[variable] = "1"
    return environment


# Keep stage dispatch explicit.  In particular, no generic ``paper run``
# command and no forecast/representation evaluation API can be selected here.
_STAGE_PROGRAM = r"""
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
(
    source_src,
    stage,
    config_path,
    artifact_root,
    data_root,
    representation_root,
    evaluation_root,
    runtime_root,
    approval_path,
) = sys.argv[1:10]
sys.path.insert(0, source_src)

from execsim.ml.paper.configs import load_paper_config, load_runtime_approval

config = load_paper_config(Path(config_path)).with_runtime_roots(
    artifact_root=Path(artifact_root),
    data_root=Path(data_root),
    cache_root=Path(runtime_root),
    evaluation_root=Path(evaluation_root),
    representation_root=Path(representation_root),
)
approval = load_runtime_approval(Path(approval_path), config)
if approval.approved_operations != frozenset({"locked_result_evaluation"}):
    raise PermissionError("Recovery supervisor requires evaluation-only approval.")
config.authorize("locked_result_evaluation", approval=approval, cli_enabled=True)

if stage == "run-tca":
    from execsim.ml.paper.orchestration import run_tca_stage
    result = run_tca_stage(
        config, full_run_cli_enabled=True, runtime_approval=approval
    )
elif stage == "report":
    from execsim.ml.paper.orchestration import report_stage
    result = report_stage(
        config, full_run_cli_enabled=True, runtime_approval=approval
    )
elif stage == "final-result-freeze":
    from execsim.ml.paper.orchestration import write_final_result_freeze
    result = write_final_result_freeze(config)
else:
    raise SystemExit("unsupported recovery stage")

print(json.dumps(result, indent=2, default=str))
"""


def _exclusive_lock(runtime_root: Path) -> AbstractContextManager[None]:
    """Return a nonblocking POSIX lock held for the complete run."""
    return _exclusive_lock_context(runtime_root)


@contextlib.contextmanager
def _exclusive_lock_context(runtime_root: Path) -> Iterator[None]:
    try:
        fcntl = importlib.import_module("fcntl")
    except ImportError as exc:  # pragma: no cover - guarded by main on Windows
        raise RuntimeError("Recovery supervisor requires POSIX/Linux fcntl locks.") from exc
    lock_path = runtime_root / ".recovery-evaluation-supervisor.lock"
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    stream = os.fdopen(descriptor, "a+b", buffering=0)
    try:
        try:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another recovery supervisor holds {lock_path}.") from exc
        yield
    finally:
        with contextlib.suppress(OSError):
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
        stream.close()


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class StageSupervisor:
    """Own one recovery run, one lock, and one child process at a time."""

    def __init__(
        self,
        args: argparse.Namespace,
        *,
        source_verifier: Callable[[argparse.Namespace], None] = _verify_source_checkout,
        preflight: Callable[[argparse.Namespace], dict[str, Any]] = _preflight_execution,
        lock_factory: Callable[[Path], AbstractContextManager[None]] = _exclusive_lock,
        popen_factory: Callable[..., Any] = subprocess.Popen,
    ) -> None:
        self.args = args
        self.source_verifier = source_verifier
        self.preflight = preflight
        self.lock_factory = lock_factory
        self.popen_factory = popen_factory
        self.run_id = f"{dt.datetime.now(dt.UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4().hex[:12]}"
        self.status_path: Path | None = None
        self.logs_root: Path | None = None
        self.status: dict[str, Any] = {}
        self.active_child: Any | None = None
        self.termination_signal: int | None = None
        self._signal_forward_error: str | None = None
        self._old_handlers: dict[int, Any] = {}
        self._workers = DEFAULT_WORKERS

    def _write_status(self) -> None:
        assert self.status_path is not None
        self.status["updated_at_utc"] = _utc_now()
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.status_path.name}.", suffix=".tmp", dir=self.status_path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(self.status, stream, sort_keys=True, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.status_path)
            _fsync_directory(self.status_path.parent)
        finally:
            with contextlib.suppress(FileNotFoundError):
                temporary.unlink()

    def _set_status(self, **updates: Any) -> None:
        self.status.update(updates)
        self._write_status()

    def _handle_signal(self, signum: int, _frame: Any) -> None:
        self.termination_signal = signum
        child = self.active_child
        if child is None or child.poll() is not None:
            return
        try:
            # start_new_session=True makes the child PID its own process-group ID.
            os.killpg(child.pid, signum)  # type: ignore[attr-defined]
        except ProcessLookupError:
            return
        except OSError as exc:
            self._signal_forward_error = f"{type(exc).__name__}: {exc}"

    def _install_signal_handlers(self) -> None:
        for signum in (signal.SIGTERM, signal.SIGINT):
            self._old_handlers[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle_signal)

    def _restore_signal_handlers(self) -> None:
        for signum, previous in self._old_handlers.items():
            signal.signal(signum, previous)
        self._old_handlers.clear()

    def _initialize_status(self) -> None:
        args = self.args
        self.status_path = args.status_path
        if self.status_path.exists() and self.status_path.is_symlink():
            raise ValueError(f"Status path must not be a symbolic link: {self.status_path}")
        logs_parent = args.runtime_root / "recovery-evaluation-supervisor-logs"
        if logs_parent.exists() and (logs_parent.is_symlink() or not logs_parent.is_dir()):
            raise ValueError(f"Stage log parent is not a real directory: {logs_parent}")
        logs_parent.mkdir(mode=0o700, exist_ok=True)
        self.logs_root = logs_parent / self.run_id
        self.logs_root.mkdir(mode=0o700, exist_ok=False)
        namespace = args.evaluation_root.relative_to(args.artifact_root).as_posix()
        self._workers = _worker_count(args)
        wrapper_digest = hashlib.sha256(
            Path(__file__).resolve(strict=True).read_bytes()
        ).hexdigest()
        self.status = {
            "schema_version": "recovery-evaluation-supervisor-v1",
            "run_id": self.run_id,
            "status": "PREFLIGHT",
            "pid": os.getpid(),
            "child_pid": None,
            "child_process_group": None,
            "current_stage": None,
            "completed_here": [],
            "inherited_validated": [],
            "last_error": None,
            "source": {
                "path": str(args.source),
                "commit": args.commit,
                "tree": args.tree,
            },
            "namespace": namespace,
            "runtime_root": str(args.runtime_root),
            "execution_sha256": None,
            "approval_sha256": None,
            "approval_id": None,
            "approval": {
                "path": str(args.approval),
                "id": None,
                "sha256": None,
            },
            "execution": None,
            "logs": {},
            "supervisor_sha256": wrapper_digest,
            "started_at_utc": _utc_now(),
            "exit_code": None,
        }
        self._write_status()

    def _apply_preflight_identity(self, identity: dict[str, Any]) -> None:
        inherited = identity.get("inherited_validated", list(INHERITED_STAGES))
        if inherited != list(INHERITED_STAGES):
            raise ValueError("Preflight did not validate the required inherited stages.")
        if identity.get("invalidation_frontier", INVALIDATION_FRONTIER) != INVALIDATION_FRONTIER:
            raise ValueError("Preflight did not validate the run-tca frontier.")
        self.status["execution"] = identity
        self.status["execution_sha256"] = identity.get("execution_sha256") or identity.get(
            "execution_receipt_sha256"
        )
        self.status["approval_sha256"] = identity.get("approval_sha256")
        self.status["approval_id"] = identity.get("approval_id")
        self.status["approval"] = {
            "path": str(self.args.approval),
            "id": identity.get("approval_id"),
            "sha256": identity.get("approval_sha256"),
        }
        self.status["inherited_validated"] = list(inherited)

    def _stage_command(self, stage: str) -> list[str]:
        args = self.args
        return [
            sys.executable,
            "-c",
            _STAGE_PROGRAM,
            str(args.source / "src"),
            stage,
            str(args.config),
            str(args.artifact_root),
            str(args.data_root),
            str(args.representation_root),
            str(args.evaluation_root),
            str(args.runtime_root),
            str(args.approval),
        ]

    def _run_stage(self, stage: str) -> int:
        assert self.logs_root is not None
        log_path = self.logs_root / f"{stage}.log"
        descriptor = os.open(
            log_path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "ab", buffering=0) as log_stream:
            log_stream.write(f"\n=== stage {stage} started {_utc_now()} ===\n".encode())
            os.fsync(log_stream.fileno())
            _fsync_directory(self.logs_root)
            child = self.popen_factory(
                self._stage_command(stage),
                cwd=str(self.args.source),
                env=_child_environment(self.args.source, self._workers),
                stdin=subprocess.DEVNULL,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            self.active_child = child
            self._set_status(
                status="RUNNING",
                current_stage=stage,
                child_pid=child.pid,
                child_process_group=child.pid,
                logs={**self.status["logs"], stage: str(log_path)},
            )
            if self.termination_signal is not None:
                self._handle_signal(self.termination_signal, None)
            exit_code = child.wait()
            self.active_child = None
            os.fsync(log_stream.fileno())
        if exit_code < 0:
            exit_code = 128 + (-exit_code)
        return int(exit_code)

    def run(self) -> int:
        for variable in NATIVE_THREAD_VARIABLES:
            os.environ[variable] = "1"
        try:
            self.args = _resolve_arguments(self.args)
            with self.lock_factory(self.args.runtime_root):
                self._initialize_status()
                self._install_signal_handlers()
                try:
                    self.source_verifier(self.args)
                    if self.termination_signal is not None:
                        raise SupervisorStop(self.termination_signal)
                    identity = self.preflight(self.args)
                    self._apply_preflight_identity(identity)
                    self._set_status(status="READY")
                    if self.termination_signal is not None:
                        raise SupervisorStop(self.termination_signal)

                    for stage in STAGES:
                        if self.termination_signal is not None:
                            raise SupervisorStop(self.termination_signal)
                        self.source_verifier(self.args)
                        self._set_status(status="RUNNING", current_stage=stage)
                        stage_exit = self._run_stage(stage)
                        if self.termination_signal is not None:
                            raise SupervisorStop(self.termination_signal)
                        if stage_exit != 0:
                            raise StageFailure(stage, stage_exit)
                        self.status["completed_here"].append(stage)
                        self._set_status(
                            status="RUNNING",
                            current_stage=None,
                            child_pid=None,
                            child_process_group=None,
                        )

                    self._set_status(
                        status="COMPLETED",
                        current_stage=None,
                        child_pid=None,
                        child_process_group=None,
                        exit_code=0,
                        finished_at_utc=_utc_now(),
                    )
                    return 0
                except SupervisorStop as exc:
                    self._set_status(
                        status="INTERRUPTED",
                        exit_code=exc.exit_code,
                        child_pid=None,
                        child_process_group=None,
                        last_error=str(exc),
                        signal_forward_error=self._signal_forward_error,
                        finished_at_utc=_utc_now(),
                    )
                    return exc.exit_code
                except StageFailure as exc:
                    self._set_status(
                        status="FAILED",
                        current_stage=exc.stage,
                        exit_code=exc.exit_code,
                        child_pid=None,
                        child_process_group=None,
                        last_error=str(exc),
                        finished_at_utc=_utc_now(),
                    )
                    return exc.exit_code
                except Exception as exc:
                    child = self.active_child
                    if child is not None and child.poll() is None:
                        self._handle_signal(signal.SIGTERM, None)
                        with contextlib.suppress(Exception):
                            child.wait()
                        self.active_child = None
                    self._set_status(
                        status="FAILED",
                        exit_code=1,
                        child_pid=None,
                        child_process_group=None,
                        last_error=f"{type(exc).__name__}: {exc}"[:2000],
                        signal_forward_error=self._signal_forward_error,
                        finished_at_utc=_utc_now(),
                    )
                    print(
                        f"Recovery supervisor failed: {type(exc).__name__}: {exc}",
                        file=sys.stderr,
                    )
                    return 1
                finally:
                    self._restore_signal_handlers()
        except Exception as exc:
            print(
                f"Recovery supervisor could not start: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 1


def _read_status(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Status file is unavailable or unsafe: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Status file must contain a JSON object.")
    return payload


def _pid_commandline(pid: int) -> str | None:
    if pid <= 1:
        return None
    proc_cmdline = Path("/proc") / str(pid) / "cmdline"
    if not proc_cmdline.is_file():
        return None
    try:
        return proc_cmdline.read_bytes().replace(b"\x00", b" ").decode(errors="replace")
    except OSError:
        return None


def _pid_is_this_supervisor(pid: int, status: dict[str, Any] | None = None) -> bool:
    command = _pid_commandline(pid)
    if command is None or Path(__file__).name not in command:
        return False
    if status is None:
        return True
    source = status.get("source")
    if not isinstance(source, dict):
        return False
    expected = (
        source.get("path"),
        source.get("commit"),
        source.get("tree"),
        status.get("runtime_root"),
    )
    return all(isinstance(value, str) and value in command for value in expected)


def _stop_from_status(path: Path) -> int:
    payload = _read_status(path)
    pid = payload.get("pid")
    if not isinstance(pid, int) or not _pid_is_this_supervisor(pid, payload):
        raise ValueError("Status PID is not an identifiable recovery supervisor process.")
    if payload.get("status") in {"COMPLETED", "FAILED", "INTERRUPTED"}:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    os.kill(pid, signal.SIGTERM)
    print(json.dumps({"status": "SIGTERM_SENT", "pid": pid, "status_path": str(path)}))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = getattr(args, "command", "run")
    if command in {"status", "stop"}:
        try:
            path = _status_path(args)
            if command == "status":
                print(json.dumps(_read_status(path), indent=2, sort_keys=True))
                return 0
            return _stop_from_status(path)
        except Exception as exc:
            print(
                f"Recovery supervisor {command} failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            return 1
    if os.name != "posix" or not hasattr(os, "killpg"):
        _parser().error("run requires a POSIX/Linux process environment")
    supervisor = StageSupervisor(args)
    result = supervisor.run()
    if supervisor.status_path is not None:
        print(f"Recovery supervisor status: {supervisor.status_path}")
    return result


if __name__ == "__main__":
    raise SystemExit(main())
