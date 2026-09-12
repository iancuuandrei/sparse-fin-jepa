"""Compare legacy and indexed learned-ledger TCA preflight on synthetic data only."""

from __future__ import annotations

import argparse
import ast
import re
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from collections.abc import Callable
from datetime import date, time
from pathlib import Path
from time import perf_counter
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

_REPOSITORY = Path(__file__).resolve().parents[1]
_SOURCE_PATH = "src/execsim/ml/paper/tca_workers.py"
_INSTRUMENT_ID = "SYNTH"
_FOLD_ID = "fold-synthetic"
_CUTOFF = date(2039, 12, 31)


def _load_modules() -> tuple[ModuleType, Callable[..., None]]:
    """Import the active checkout directly, independent of installed package state."""
    sys.path.insert(0, str(_REPOSITORY / "src"))
    from execsim.ml.paper import tca_workers
    from execsim.ml.paper.evaluation_artifacts import publish_frames

    return tca_workers, publish_frames


def _load_legacy_preflight(
    tca_workers: ModuleType, revision: str
) -> tuple[Callable[..., None], str]:
    """Compile only baseline preflight functions from a read-only `git show`."""
    if re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ValueError("Benchmark baseline must be a full lowercase Git commit SHA.")
    resolved_revision = subprocess.run(
        ["git", "rev-parse", "--verify", "--end-of-options", f"{revision}^{{commit}}"],
        cwd=_REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source = subprocess.run(
        ["git", "show", f"{resolved_revision}:{_SOURCE_PATH}"],
        cwd=_REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    parsed = ast.parse(source, filename=f"{resolved_revision}:{_SOURCE_PATH}")
    function_nodes = {node.name: node for node in parsed.body if isinstance(node, ast.FunctionDef)}
    required = (
        "_tca_as_of_origins",
        "_validate_learned_case",
        "_validate_ewma_case",
        "preflight_tca_ledgers",
    )
    missing = set(required).difference(function_nodes)
    if missing:
        raise RuntimeError(f"Baseline source lacks required functions: {sorted(missing)}")
    legacy_preflight_source = ast.get_source_segment(
        source, function_nodes["preflight_tca_ledgers"]
    )
    if legacy_preflight_source is None or "date_positions" in legacy_preflight_source:
        raise RuntimeError(
            "Baseline already contains the indexed preflight; select a pre-change revision."
        )

    future_annotations = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    isolated = ast.Module(
        body=[future_annotations, *(function_nodes[name] for name in required)],
        type_ignores=[],
    )
    ast.fix_missing_locations(isolated)
    namespace = vars(tca_workers).copy()
    namespace["__name__"] = "execsim_tca_preflight_baseline"
    exec(
        compile(isolated, f"<git show {resolved_revision}:{_SOURCE_PATH}>", "exec"),
        namespace,
    )
    return namespace["preflight_tca_ledgers"], resolved_revision


def synthetic_ledger_frames(
    *,
    session_dates: tuple[date, ...],
    origins: tuple[int, ...],
    instrument_id: str,
    fold_id: str,
    training_cutoff: date,
) -> dict[str, pd.DataFrame]:
    """Build complete learned and EWMA fixtures for tests and timing runs."""
    learned_scale_rows: list[dict[str, Any]] = []
    learned_shape_rows: list[dict[str, Any]] = []
    ewma_scale_rows: list[dict[str, Any]] = []
    minute_rows: list[dict[str, Any]] = []

    for session_date in session_dates:
        opened = pd.Timestamp.combine(session_date, time(9, 30)).tz_localize("America/New_York")
        for origin in origins:
            sample_id = f"{session_date.isoformat()}-{instrument_id}-{origin:02d}"
            learned_scale_rows.append(
                {
                    "sample_id": sample_id,
                    "fold_id": fold_id,
                    "instrument_id": instrument_id,
                    "symbol": instrument_id,
                    "session_date": session_date.isoformat(),
                    "as_of": origin,
                    "training_cutoff": training_cutoff.isoformat(),
                }
            )
            ewma_scale_rows.append(
                {
                    "sample_id": sample_id,
                    "instrument_id": instrument_id,
                    "symbol": instrument_id,
                    "session_date": session_date.isoformat(),
                    "as_of": origin,
                }
            )
            buckets = np.arange(origin, 26)
            learned_shape_rows.extend(
                {
                    "case_id": sample_id,
                    "target_bucket": int(bucket),
                    "conditional_share": 1.0 / len(buckets),
                }
                for bucket in buckets
            )
            minute_rows.extend(
                {
                    "sample_id": sample_id,
                    "generated_at": opened + pd.Timedelta(minutes=15 * origin + offset),
                    "end_token": 24,
                }
                for offset in range(15)
            )

    return {
        "learned_scale": pd.DataFrame(learned_scale_rows),
        "learned_shape": pd.DataFrame(learned_shape_rows)
        .sample(frac=1, random_state=41)
        .reset_index(drop=True),
        "learned_metrics": pd.DataFrame({"metric": pd.Series(dtype="string")}),
        "ewma_scale": pd.DataFrame(ewma_scale_rows),
        "ewma_shape": pd.DataFrame(
            {"case_id": pd.Series(dtype="string"), "target_bucket": pd.Series(dtype="int64")}
        ),
        "ewma_metrics": pd.DataFrame({"metric": pd.Series(dtype="string")}),
        "ewma_minutes": pd.DataFrame(minute_rows),
        "ewma_unavailable": pd.DataFrame(
            {
                "sample_id": pd.Series(dtype="string"),
                "generated_at": pd.Series(dtype="datetime64[ns, America/New_York]"),
                "end_token": pd.Series(dtype="int64"),
                "status": pd.Series(dtype="string"),
                "session_date": pd.Series(dtype="string"),
                "reason": pd.Series(dtype="string"),
            }
        ),
    }


def _synthetic_ledgers(
    root: Path,
    *,
    dates_count: int,
    origins_count: int,
    learned_ledger_count: int,
    publish_frames: Callable[..., dict[str, Any]],
) -> tuple[
    tuple[tuple[str, int | None, Path, dict[str, Any]], ...],
    dict[str, tuple[Path, dict[str, Any]]],
    dict[date, tuple[str, ...]],
    dict[str, list[str]],
]:
    """Publish deterministic checksum-bound artifacts built only in memory."""
    dates = tuple(pd.bdate_range("2040-01-02", periods=dates_count).date)
    origins = tuple(range(4, 4 + origins_count))
    end_offset = 10 * 60 + 30 + 15 * origins_count
    end_time = time(end_offset // 60, end_offset % 60).strftime("%H:%M")
    tca_config = {"window": ["10:30", end_time]}
    frames = synthetic_ledger_frames(
        session_dates=dates,
        origins=origins,
        instrument_id=_INSTRUMENT_ID,
        fold_id=_FOLD_ID,
        training_cutoff=_CUTOFF,
    )
    learned_records: list[tuple[str, int | None, Path, dict[str, Any]]] = []
    for ledger_number in range(learned_ledger_count):
        identity = {
            "fold_id": _FOLD_ID,
            "method": f"synthetic-{ledger_number + 1}",
            "seed": ledger_number + 1,
            "parameter_freeze_sha256": "synthetic-freeze",
            "model_manifest_sha256": f"synthetic-model-{ledger_number + 1}",
            "source_commit": "synthetic-source",
            "source_tree": "synthetic-tree",
            "paper_config_hash": "synthetic-config",
            "base_manifest_sha256": "synthetic-base",
            "embedding_sha256": None,
        }
        learned_directory = root / f"learned-{ledger_number + 1}"
        publish_frames(
            learned_directory,
            identity=identity,
            frames={
                "scale.parquet": frames["learned_scale"],
                "shape.parquet": frames["learned_shape"],
                "metrics.parquet": frames["learned_metrics"],
            },
        )
        learned_records.append(
            (f"synthetic-{ledger_number + 1}", ledger_number + 1, learned_directory, identity)
        )

    ewma_identity = {
        "fold_id": _FOLD_ID,
        "instrument_id": _INSTRUMENT_ID,
        "fixture": "synthetic-ewma",
    }
    ewma_directory = root / "ewma"
    publish_frames(
        ewma_directory,
        identity=ewma_identity,
        frames={
            "scale.parquet": frames["ewma_scale"],
            "shape.parquet": frames["ewma_shape"],
            "metrics.parquet": frames["ewma_metrics"],
            "minute-forecasts.parquet": frames["ewma_minutes"],
            "unavailable.parquet": frames["ewma_unavailable"],
        },
    )

    ledger_records = tuple(learned_records)
    ewma_records = {_INSTRUMENT_ID: (ewma_directory, ewma_identity)}
    eligible_cases = {session_date: (_INSTRUMENT_ID,) for session_date in dates}
    return ledger_records, ewma_records, eligible_cases, tca_config


def _timed_preflight(
    preflight: Callable[..., None],
    *,
    root: Path,
    ledger_records: tuple[tuple[str, int | None, Path, dict[str, Any]], ...],
    ewma_records: dict[str, tuple[Path, dict[str, Any]]],
    eligible_cases: dict[date, tuple[str, ...]],
    tca_config: dict[str, list[str]],
) -> tuple[float, Counter[str], Counter[str], Counter[str]]:
    """Run valid preflight while timing validators and counting parquet reads."""
    original_read_parquet = pd.read_parquet
    read_calls: Counter[str] = Counter()
    read_rows: Counter[str] = Counter()
    component_seconds: Counter[str] = Counter()
    namespace = preflight.__globals__
    original_learned_validator = namespace["_validate_learned_case"]
    original_ewma_validator = namespace["_validate_ewma_case"]

    def timed_validator(function: Callable[..., Any], label: str) -> Callable[..., Any]:
        def invoke(*args: Any, **kwargs: Any) -> Any:
            started = perf_counter()
            try:
                return function(*args, **kwargs)
            finally:
                component_seconds[label] += perf_counter() - started

        return invoke

    def counted_read(path: str | Path, *args: Any, **kwargs: Any) -> pd.DataFrame:
        result = original_read_parquet(path, *args, **kwargs)
        resolved = Path(path).resolve()
        relative = resolved.relative_to(root.resolve()).as_posix()
        read_calls[relative] += 1
        read_rows[relative] += len(result)
        return result

    namespace["_validate_learned_case"] = timed_validator(original_learned_validator, "learned")
    namespace["_validate_ewma_case"] = timed_validator(original_ewma_validator, "ewma")
    pd.read_parquet = counted_read
    started = perf_counter()
    try:
        preflight(
            ledger_records=ledger_records,
            ewma_records=ewma_records,
            eligible_cases=eligible_cases,
            training_cutoff=_CUTOFF,
            tca_config=tca_config,
        )
    finally:
        elapsed = perf_counter() - started
        pd.read_parquet = original_read_parquet
        namespace["_validate_learned_case"] = original_learned_validator
        namespace["_validate_ewma_case"] = original_ewma_validator
    return elapsed, read_calls, read_rows, component_seconds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-revision", default="3eb221bf787421fd4d5397b725397861b403fdaa")
    parser.add_argument("--dates", type=int, default=128, help="Synthetic dates, maximum 128.")
    parser.add_argument("--origins", type=int, default=22, help="Synthetic origins, maximum 22.")
    parser.add_argument(
        "--learned-ledgers", type=int, default=8, help="Learned ledgers, maximum 8."
    )
    parser.add_argument("--repeats", type=int, default=1, help="Paired timings, maximum 5.")
    args = parser.parse_args()
    if not 1 <= args.dates <= 128 or not 1 <= args.origins <= 22:
        parser.error("--dates must be 1..128 and --origins must be 1..22.")
    if not 1 <= args.learned_ledgers <= 8:
        parser.error("--learned-ledgers must be 1..8.")
    if not 1 <= args.repeats <= 5:
        parser.error("--repeats must be 1..5.")

    tca_workers, publish_frames = _load_modules()
    legacy_preflight, baseline_sha = _load_legacy_preflight(tca_workers, args.baseline_revision)
    with tempfile.TemporaryDirectory(prefix="execsim-tca-preflight-") as temporary:
        root = Path(temporary)
        ledger_records, ewma_records, eligible_cases, tca_config = _synthetic_ledgers(
            root,
            dates_count=args.dates,
            origins_count=args.origins,
            learned_ledger_count=args.learned_ledgers,
            publish_frames=publish_frames,
        )
        timings: dict[str, list[float]] = {"legacy": [], "indexed": []}
        component_timings: dict[str, dict[str, list[float]]] = {
            name: {"learned": [], "ewma": []} for name in ("legacy", "indexed")
        }
        last_stats: dict[str, tuple[Counter[str], Counter[str]]] = {}
        for _ in range(args.repeats):
            for name, function in (
                ("legacy", legacy_preflight),
                ("indexed", tca_workers.preflight_tca_ledgers),
            ):
                elapsed, reads, rows, components = _timed_preflight(
                    function,
                    root=root,
                    ledger_records=ledger_records,
                    ewma_records=ewma_records,
                    eligible_cases=eligible_cases,
                    tca_config=tca_config,
                )
                timings[name].append(elapsed)
                last_stats[name] = reads, rows
                for component in ("learned", "ewma"):
                    component_timings[name][component].append(components[component])

    legacy_seconds = statistics.median(timings["legacy"])
    indexed_seconds = statistics.median(timings["indexed"])
    legacy_reads, legacy_rows = last_stats["legacy"]
    indexed_reads, indexed_rows = last_stats["indexed"]
    print(
        "TCA preflight benchmark: synthetic artifacts only; no historical TEST data or providers."
    )
    print(
        f"Workload: dates={args.dates}, origins={args.origins}, "
        f"learned_ledgers={args.learned_ledgers}, ewma_ledgers=1, instruments=1, "
        f"complete future-bucket shapes, repeats={args.repeats}"
    )
    print(f"Legacy source: git show {baseline_sha}:{_SOURCE_PATH}")
    print(f"Legacy elapsed median: {legacy_seconds:.6f} s")
    print(f"Indexed elapsed median: {indexed_seconds:.6f} s")
    for name in ("legacy", "indexed"):
        learned_seconds = statistics.median(component_timings[name]["learned"])
        ewma_seconds = statistics.median(component_timings[name]["ewma"])
        print(
            f"{name.title()} validator-call medians: "
            f"learned={learned_seconds:.6f} s, EWMA={ewma_seconds:.6f} s"
        )
    if indexed_seconds > 0:
        print(f"Elapsed ratio (legacy/indexed): {legacy_seconds / indexed_seconds:.3f}x")
    print(
        f"Legacy parquet reads: {sum(legacy_reads.values())}; "
        f"rows returned: {sum(legacy_rows.values())}"
    )
    print(
        f"Indexed parquet reads: {sum(indexed_reads.values())}; "
        f"rows returned: {sum(indexed_rows.values())}"
    )
    print(f"Per-file legacy read counts: {dict(sorted(legacy_reads.items()))}")
    print(f"Per-file indexed read counts: {dict(sorted(indexed_reads.items()))}")
    if legacy_reads != indexed_reads or legacy_rows != indexed_rows:
        raise RuntimeError("Legacy and indexed preflight read different parquet rows.")
    print("Read parity: PASS")
    print("Wall-clock thresholds: none (informational synthetic timing only).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
