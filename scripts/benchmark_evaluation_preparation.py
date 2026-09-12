"""Bounded synthetic benchmarks for paper-evaluation preparation paths.

This script creates only synthetic data in a temporary directory. It does not
open historical results, TEST data, checkpoints, or fitted models. Matching
timings stop before bootstrap. Fixed-grid DataLoader reuse is intentionally not
benchmarked: sharing a loader can advance a shared worker-generator state and
change RNG behavior even when the dataset itself is deterministic.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import statistics
import sys
import tempfile
from collections.abc import Callable
from datetime import date, time, timedelta
from pathlib import Path
from time import perf_counter, process_time
from typing import Any

import numpy as np
import pandas as pd

_REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPOSITORY / "src"))

_EWMA_MAX_SAMPLES = 128
_SEQUENCE_MAX_SAMPLES = 100_000
_REPORT_MAX_CASES = 25_000
_REPEATS_MAX = 8
_SYNTHETIC_SYMBOL = "SYNTH"
_TARGET_ORIGIN = 20
_IDENTITY_COLUMNS = (
    "fold_id",
    "date",
    "instrument_id",
    "order_fraction_adv20",
    "parent_quantity",
    "side",
    "start",
    "end",
    "planned_participation",
    "hard_participation",
    "risk_aversion",
    "tracking_penalty",
)


def _process_peak_rss_bytes() -> int | None:
    """Return this process's peak RSS when the platform API is available."""
    if sys.platform == "win32":
        try:
            from ctypes import wintypes

            class ProcessMemoryCountersEx(ctypes.Structure):
                _fields_ = [
                    ("cb", wintypes.DWORD),
                    ("PageFaultCount", wintypes.DWORD),
                    ("PeakWorkingSetSize", ctypes.c_size_t),
                    ("WorkingSetSize", ctypes.c_size_t),
                    ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                    ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                    ("PagefileUsage", ctypes.c_size_t),
                    ("PeakPagefileUsage", ctypes.c_size_t),
                    ("PrivateUsage", ctypes.c_size_t),
                ]

            process = ctypes.windll.kernel32.GetCurrentProcess()
            counters = ProcessMemoryCountersEx()
            counters.cb = ctypes.sizeof(counters)
            ok = ctypes.windll.psapi.GetProcessMemoryInfo(
                process, ctypes.byref(counters), counters.cb
            )
            return int(counters.PeakWorkingSetSize) if ok else None
        except (AttributeError, OSError, TypeError):
            return None

    try:
        import resource

        peak = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except (ImportError, OSError, ValueError):
        return None
    return peak if sys.platform == "darwin" else peak * 1024


def _timed_repeats(action: Callable[[], Any], repeats: int) -> dict[str, float]:
    """Measure bounded repeat wall and process CPU time; return median milliseconds."""
    wall_ms: list[float] = []
    cpu_ms: list[float] = []
    for _ in range(repeats):
        cpu_start = process_time()
        wall_start = perf_counter()
        action()
        wall_ms.append((perf_counter() - wall_start) * 1000.0)
        cpu_ms.append((process_time() - cpu_start) * 1000.0)
    return {
        "median_wall_ms": statistics.median(wall_ms),
        "median_process_cpu_ms": statistics.median(cpu_ms),
        "repeats": repeats,
    }


def _synthetic_market_bars(sample_count: int) -> tuple[pd.DataFrame, tuple[date, ...]]:
    """Create causal synthetic sessions, with a few window-specific history gaps."""
    session_count = 20 + sample_count
    days = pd.bdate_range("2040-01-02", periods=session_count)
    frames: list[pd.DataFrame] = []
    for session_index, day_value in enumerate(days):
        opened = pd.Timestamp.combine(day_value.date(), time(9, 30)).tz_localize("America/New_York")
        timestamps = pd.date_range(opened, periods=390, freq="min")
        minute_number = np.arange(390, dtype=np.int64)
        volumes = (
            1_000
            + 17 * session_index
            + minute_number * (1 + session_index % 7)
            + minute_number**2 // 97
        ).astype(float)
        frame = pd.DataFrame(
            {
                "symbol": _SYNTHETIC_SYMBOL,
                "timestamp": timestamps,
                "volume": volumes,
            }
        )
        # Offset 14:35 falls inside offsets 0-5 of the requested 14:30-15:30
        # windows and outside later offsets. The eligible historical sessions
        # therefore vary by exact request, as they do for real incomplete inputs.
        if session_index % 5 == 0:
            frame = frame.loc[frame["timestamp"] != opened + pd.Timedelta(hours=5, minutes=5)]
        frames.append(frame)
    target_dates = tuple(days[20:].date)
    return pd.concat(frames, ignore_index=True), target_dates


def _ewma_requests(target_dates: tuple[date, ...]) -> tuple[dict[str, Any], ...]:
    """Build the one full-session and exact 15-offset TCA requests per sample."""
    requests: list[dict[str, Any]] = []
    for session_date in target_dates:
        opened = pd.Timestamp.combine(session_date, time(9, 30)).tz_localize("America/New_York")
        generated = opened + pd.Timedelta(minutes=15 * _TARGET_ORIGIN)
        requests.append(
            {
                "symbol": _SYNTHETIC_SYMBOL,
                "session_date": session_date,
                "generated_at": generated,
                "bucket_timestamps": tuple(
                    pd.date_range(
                        generated,
                        opened + pd.Timedelta(minutes=15 * 26 - 1),
                        freq="min",
                    )
                ),
                "end_token": 26,
                "offset": 0,
            }
        )
        for offset in range(15):
            request_time = generated + pd.Timedelta(minutes=offset)
            requests.append(
                {
                    "symbol": _SYNTHETIC_SYMBOL,
                    "session_date": session_date,
                    "generated_at": request_time,
                    "bucket_timestamps": tuple(
                        pd.date_range(
                            request_time,
                            opened + pd.Timedelta(minutes=15 * 24 - 1),
                            freq="min",
                        )
                    ),
                    "end_token": 24,
                    "offset": offset,
                }
            )
    return tuple(requests)


def _run_ewma_requests(bars: pd.DataFrame, requests: tuple[dict[str, Any], ...]) -> tuple[Any, ...]:
    """Run the production estimator once for each exact synthetic request."""
    from execsim.forecasting import HistoricalProfileForecaster

    provider = HistoricalProfileForecaster(
        bars,
        estimator="ewma",
        lookback_sessions=20,
        data_manifest_hash="synthetic-market-sha256",
    )
    return tuple(
        provider.forecast(
            symbol=str(request["symbol"]),
            session_date=request["session_date"],
            generated_at=request["generated_at"],
            bucket_timestamps=request["bucket_timestamps"],
        )
        for request in requests
    )


def _run_ewma_benchmark(sample_count: int, repeats: int) -> dict[str, Any]:
    bars, target_dates = _synthetic_market_bars(sample_count)
    requests = _ewma_requests(target_dates)
    expected = _run_ewma_requests(bars, requests)
    observed: tuple[Any, ...] = ()

    def timed_pass() -> None:
        nonlocal observed
        observed = _run_ewma_requests(bars, requests)

    timing = _timed_repeats(timed_pass, repeats)
    if observed != expected:
        raise AssertionError("Repeated exact-window EWMA requests changed their outputs.")
    offset_forecasts = [
        forecast
        for request, forecast in zip(requests, observed, strict=True)
        if int(request["end_token"]) == 24
    ]
    if len(offset_forecasts) != 15 * sample_count:
        raise AssertionError("The synthetic EWMA path did not evaluate all 15 offsets per sample.")
    if any(
        forecast.bucket_timestamps != request["bucket_timestamps"]
        or forecast.generated_at != request["generated_at"]
        for request, forecast in zip(requests, observed, strict=True)
    ):
        raise AssertionError("An EWMA forecast changed its exact requested window identity.")
    return {
        "status": "PASS",
        "samples": sample_count,
        "exact_tca_offset_requests": len(offset_forecasts),
        "full_session_requests": sample_count,
        "total_forecast_calls_per_pass": len(requests),
        "timing": timing,
        "deterministic_outputs": True,
        "exact_request_grids": True,
        "synthetic_prior_sessions": 20,
        "history_gap_pattern": "synthetic 14:35 gaps vary eligible rows across offsets 0-5",
    }


def _synthetic_sequence_index(sample_count: int) -> tuple[dict[str, Any], pd.DataFrame]:
    """Build the manifest and index-only TRAIN fixture consumed by the dataset."""
    session_count = max(1, math.ceil(sample_count / 22))
    session_dates = pd.bdate_range("2040-02-01", periods=session_count)
    session_ids = [f"SYNTH-{index:06d}" for index in range(session_count)]
    rows: list[dict[str, Any]] = []
    for sample_index in range(sample_count):
        session_index = sample_index // 22
        date_value = session_dates[session_index]
        session_id = session_ids[session_index]
        token = 4 + sample_index % 22
        offsets = (0, 1, 3, 7)
        rows.append(
            {
                "sample_id": f"{session_id}-T{token:02d}",
                "session_id": session_id,
                "fold_id": "fold-1",
                "partition": "train",
                "as_of_token": token,
                "context_start": max(0, token - 8),
                "context_end": token,
                "target_indices": [min(25, token + offset) for offset in offsets],
                "target_mask": [token + offset < 26 for offset in offsets],
                "as_of_ns": int(date_value.value + token * 15 * 60 * 1_000_000_000),
                "source_sequence_hash": hashlib.sha256(session_id.encode()).hexdigest(),
                "cutoff": (date_value.date() - timedelta(days=1)).isoformat(),
                "training_cutoff": (date_value.date() - timedelta(days=1)).isoformat(),
                "market_information_as_of": (
                    date_value.tz_localize("America/New_York")
                    + pd.Timedelta(hours=9, minutes=30 + token * 15)
                ).isoformat(),
                "feature_history_end": (date_value.date() - timedelta(days=1)).isoformat(),
            }
        )
    # Force the fixed-grid path to restore the deterministic session/token order
    # instead of receiving already-sorted input rows.
    frame = pd.DataFrame(rows).sample(frac=1.0, random_state=29).reset_index(drop=True)
    manifest = {
        "fold_id": "fold-1",
        "sequence_files": [f"sessions/train/{session_id}.parquet" for session_id in session_ids],
        "index_files": ["indexes/train/index.parquet"],
    }
    return manifest, frame


def _fixed_grid_group_sort(samples: tuple[Any, ...]) -> tuple[Any, ...]:
    """Reproduce the current fixed-grid `set_epoch` selection and ordering."""
    grouped: dict[str, list[Any]] = {}
    for sample in samples:
        grouped.setdefault(sample.session_id, []).append(sample)
    return tuple(
        sample
        for session_id in sorted(grouped)
        for sample in sorted(grouped[session_id], key=lambda item: item.as_of_token)
    )


def _run_sequence_benchmark(sample_count: int, repeats: int, workspace: Path) -> dict[str, Any]:
    try:
        import pyarrow  # noqa: F401

        from execsim.ml.sequences.streaming import PaperSequenceDataset
    except ImportError as exc:
        return {"status": "BLOCKED", "reason": f"optional sequence dependency unavailable: {exc}"}

    manifest, frame = _synthetic_sequence_index(sample_count)
    root = workspace / "sequence-fixture"
    (root / "indexes" / "train").mkdir(parents=True)
    manifest_path = root / "sequence-manifest.json"
    index_path = root / "indexes" / "train" / "index.parquet"
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    frame.to_parquet(index_path, index=False)

    def make_dataset() -> Any:
        return PaperSequenceDataset(
            manifest_path,
            partition="train",
            seed=13,
            cache_size=8,
            sample_train_positions=False,
        )

    dataset = make_dataset()  # warms only the temporary manifest-bound index cache
    baseline_order = _fixed_grid_group_sort(dataset._all_samples)
    if baseline_order != dataset._samples:
        raise AssertionError("Synthetic dataset did not retain the production fixed-grid order.")

    preordered_start = perf_counter()
    preordered = tuple(
        sorted(dataset._all_samples, key=lambda item: (item.session_id, item.as_of_token))
    )
    precompute_ms = (perf_counter() - preordered_start) * 1000.0
    if preordered != baseline_order:
        raise AssertionError("Precomputed fixed-grid order differs from current group/sort order.")

    current_set_epoch = _timed_repeats(
        lambda: [dataset.set_epoch(epoch) for epoch in range(repeats)], 1
    )

    def cached_order_assignment() -> None:
        selected: tuple[Any, ...] = ()
        for _ in range(repeats):
            selected = preordered
        if selected != baseline_order:
            raise AssertionError("Cached fixed-grid assignment changed sample ordering.")

    cached_assignment = _timed_repeats(cached_order_assignment, 1)
    construction = _timed_repeats(make_dataset, repeats)
    return {
        "status": "PASS",
        "train_samples": sample_count,
        "synthetic_sessions": len(dataset._sessions),
        "constructor_timing_with_temporary_index_cache": construction,
        "current_group_sort_timing": current_set_epoch,
        "preordered_tuple_build_ms": precompute_ms,
        "preordered_assignment_timing": cached_assignment,
        "fixed_grid_order_exact": True,
        "data_loader_reuse_benchmarked": False,
    }


def _synthetic_report_rows(case_count: int) -> pd.DataFrame:
    """Create a wide deterministic report ledger with missing and unavailable cases."""
    case = np.arange(case_count, dtype=np.int64)
    identity = pd.DataFrame(
        {
            "fold_id": np.asarray(["fold-1", "fold-2", "fold-3"])[case % 3],
            "date": pd.Timestamp("2040-01-02") + pd.to_timedelta(case % 252, unit="D"),
            "instrument_id": np.asarray([f"SYNTH-{value:03d}" for value in case % 100]),
            "order_fraction_adv20": np.asarray([0.01, 0.025, 0.05, 0.1])[case % 4],
            "parent_quantity": case + 1,
            "side": np.asarray(["BUY", "SELL"])[case % 2],
            "start": "10:30",
            "end": "15:30",
            "planned_participation": np.asarray([0.02, 0.04, 0.08])[case % 3],
            "hard_participation": np.asarray([0.1, 0.15])[case % 2],
            "risk_aversion": np.asarray([0.0, 0.001, 0.01])[case % 3],
            "tracking_penalty": np.asarray([0.0, 0.1])[case % 2],
        }
    )
    method_specs = (
        ("lightgbm_raw", np.ones(case_count, dtype=bool), False, 0.0),
        ("candidate_dense", case % 31 != 0, False, -0.1),
        ("candidate_sparse", case % 19 != 0, False, 0.1),
        ("ewma", np.ones(case_count, dtype=bool), True, 0.03),
    )
    pieces: list[pd.DataFrame] = []
    payload = {
        f"telemetry_{column:02d}": (case.astype(float) + column) / (column + 1)
        for column in range(16)
    }
    base_metric = np.sin(case.astype(float) / 37.0)
    for method, keep, unavailable_ewma, offset in method_specs:
        positions = case[keep]
        data: dict[str, Any] = identity.loc[keep].reset_index(drop=True).to_dict("list")
        data["method"] = np.full(len(positions), method, dtype=object)
        metric = base_metric[keep] + offset
        status = np.full(len(positions), "AVAILABLE", dtype=object)
        if unavailable_ewma:
            unavailable = positions % 23 == 0
            metric[unavailable] = np.nan
            status[unavailable] = "EWMA_UNAVAILABLE"
        data["normalized_allocation_regret"] = metric
        data["status"] = status
        data.update({name: values[keep] for name, values in payload.items()})
        pieces.append(pd.DataFrame(data))
    return pd.concat(pieces, ignore_index=True)


def _projected_complete_case_differences(
    rows: pd.DataFrame,
    *,
    baseline: str,
    candidate: str,
    value_column: str,
    identity_columns: tuple[str, ...],
    method_column: str = "method",
) -> Any:
    """Benchmark candidate: project columns in the selected-row copy only."""
    from execsim.ml.paper.statistics import CompleteCaseResult

    required = {method_column, value_column, *identity_columns}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Complete-case ledger missing columns: {sorted(missing)}")
    selected_columns = [method_column, *identity_columns, value_column]
    if "status" in rows and "status" not in selected_columns:
        selected_columns.append("status")
    selected = rows.loc[rows[method_column].isin((baseline, candidate)), selected_columns].copy()
    if selected.duplicated([method_column, *identity_columns]).any():
        raise ValueError("Complete-case ledger contains duplicated method/case rows.")
    requested = selected
    if "status" in selected:
        if not selected["status"].isin(("AVAILABLE", "EWMA_UNAVAILABLE")).all():
            raise ValueError("Complete-case ledger has unknown availability status.")
        unavailable = selected["status"].eq("EWMA_UNAVAILABLE")
        if (
            not selected.loc[unavailable, method_column].eq("ewma").all()
            or selected.loc[unavailable, value_column].notna().any()
        ):
            raise ValueError("Unavailable cases must be EWMA rows without a metric.")
        selected = selected.loc[~unavailable]
    if not np.isfinite(selected[value_column].to_numpy(dtype=float)).all():
        raise ValueError("Available complete-case metrics must be finite.")
    base = selected.loc[selected[method_column] == baseline, [*identity_columns, value_column]]
    other = selected.loc[selected[method_column] == candidate, [*identity_columns, value_column]]
    paired = base.merge(
        other,
        on=list(identity_columns),
        how="inner",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    paired["difference"] = paired[f"{value_column}_candidate"] - paired[f"{value_column}_baseline"]
    return CompleteCaseResult(
        paired_rows=paired,
        baseline_rows=int(requested[method_column].eq(baseline).sum()),
        candidate_rows=int(requested[method_column].eq(candidate).sum()),
        matched_rows=len(paired),
        dropped_baseline_rows=int(requested[method_column].eq(baseline).sum()) - len(paired),
        dropped_candidate_rows=int(requested[method_column].eq(candidate).sum()) - len(paired),
    )


def _assert_matching_equal(expected: Any, observed: Any) -> None:
    """Require exact paired rows and all availability/drop counts to agree."""
    pd.testing.assert_frame_equal(expected.paired_rows, observed.paired_rows, check_exact=True)
    for name in (
        "baseline_rows",
        "candidate_rows",
        "matched_rows",
        "dropped_baseline_rows",
        "dropped_candidate_rows",
    ):
        if getattr(expected, name) != getattr(observed, name):
            raise AssertionError(f"Projected matcher changed {name}.")


def _run_matching_comparisons(
    frame: pd.DataFrame, candidates: tuple[str, ...], projected: bool
) -> tuple[tuple[str, int, int, int], ...]:
    from execsim.ml.paper.statistics import construct_complete_case_differences

    matcher = (
        _projected_complete_case_differences if projected else construct_complete_case_differences
    )
    results: list[tuple[str, int, int, int]] = []
    for candidate in candidates:
        matched = matcher(
            frame,
            baseline="lightgbm_raw",
            candidate=candidate,
            value_column="normalized_allocation_regret",
            identity_columns=_IDENTITY_COLUMNS,
        )
        results.append(
            (
                candidate,
                matched.matched_rows,
                matched.dropped_baseline_rows,
                matched.dropped_candidate_rows,
            )
        )
    return tuple(results)


def _run_report_benchmark(case_count: int, repeats: int) -> dict[str, Any]:
    frame = _synthetic_report_rows(case_count)
    candidates = ("candidate_dense", "candidate_sparse", "ewma")
    from execsim.ml.paper.statistics import construct_complete_case_differences

    for candidate in candidates:
        current = construct_complete_case_differences(
            frame,
            baseline="lightgbm_raw",
            candidate=candidate,
            value_column="normalized_allocation_regret",
            identity_columns=_IDENTITY_COLUMNS,
        )
        projected = _projected_complete_case_differences(
            frame,
            baseline="lightgbm_raw",
            candidate=candidate,
            value_column="normalized_allocation_regret",
            identity_columns=_IDENTITY_COLUMNS,
        )
        _assert_matching_equal(current, projected)

    current_summary = _run_matching_comparisons(frame, candidates, projected=False)
    projected_summary = _run_matching_comparisons(frame, candidates, projected=True)
    if current_summary != projected_summary:
        raise AssertionError("Projected report matching changed exact intersection counts.")

    current_timing = _timed_repeats(
        lambda: _run_matching_comparisons(frame, candidates, projected=False), repeats
    )
    projected_timing = _timed_repeats(
        lambda: _run_matching_comparisons(frame, candidates, projected=True), repeats
    )
    return {
        "status": "PASS",
        "synthetic_case_count": case_count,
        "synthetic_method_rows": len(frame),
        "ledger_columns": len(frame.columns),
        "comparisons_per_pass": len(candidates),
        "current_full_row_copy_timing": current_timing,
        "projected_selected_row_copy_timing": projected_timing,
        "exact_paired_rows_and_drop_counts": True,
        "matched_counts": [
            {
                "candidate": candidate,
                "matched": matched,
                "dropped_baseline": dropped_baseline,
                "dropped_candidate": dropped_candidate,
            }
            for candidate, matched, dropped_baseline, dropped_candidate in current_summary
        ],
        "bootstrap_timed": False,
    }


def _bounded(parser: argparse.ArgumentParser, name: str, value: int, upper: int) -> None:
    if value < 1 or value > upper:
        parser.error(f"{name} must be between 1 and {upper}.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ewma-samples", type=int, default=16)
    parser.add_argument("--sequence-samples", type=int, default=4096)
    parser.add_argument("--report-cases", type=int, default=5000)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    _bounded(parser, "--ewma-samples", args.ewma_samples, _EWMA_MAX_SAMPLES)
    _bounded(parser, "--sequence-samples", args.sequence_samples, _SEQUENCE_MAX_SAMPLES)
    _bounded(parser, "--report-cases", args.report_cases, _REPORT_MAX_CASES)
    _bounded(parser, "--repeats", args.repeats, _REPEATS_MAX)

    with tempfile.TemporaryDirectory(prefix="execsim-evaluation-preparation-") as temporary:
        workspace = Path(temporary)
        output = {
            "scope": (
                "synthetic-only; no historical TEST, result tables, checkpoints, or fitted models"
            ),
            "ewma_exact_windows": _run_ewma_benchmark(args.ewma_samples, args.repeats),
            "fixed_grid_sequence": _run_sequence_benchmark(
                args.sequence_samples, args.repeats, workspace
            ),
            "report_complete_case_matching": _run_report_benchmark(args.report_cases, args.repeats),
            "process_peak_rss_bytes": _process_peak_rss_bytes(),
            "peak_rss_scope": "whole benchmark process high-water mark; None means unavailable",
        }
    print(json.dumps(output, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
