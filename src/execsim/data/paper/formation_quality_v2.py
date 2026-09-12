"""Bounded formation-year resolution-quality scan for sparse-JEPA v2."""

from __future__ import annotations

import ctypes
import os
import time
from collections import defaultdict
from datetime import date
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from execsim.data.paper.manifests import file_sha256, read_json, write_json_atomic
from execsim.data.paper.resolution_quality import (
    TOKEN_COUNT,
    assess_session_resolution_quality,
)
from execsim.data.paper.schemas import PAPER_BAR_COLUMNS
from execsim.data.paper.validation import expected_xnys_minutes


def scan_formation_resolution_quality(
    snapshot: pd.DataFrame,
    daily_bars: pd.DataFrame,
    *,
    corpus_root: Path,
    expected_sessions: tuple[object, ...],
    spy_instrument_id: str,
    output_path: Path,
    receipt_path: Path,
    protocol_hash: str,
) -> dict[str, Any]:
    """Scan monthly minute chunks while retaining only one instrument in memory."""
    identities = {
        str(row.instrument_id): str(row.symbol).upper() for row in snapshot.itertuples(index=False)
    }
    identities[spy_instrument_id] = "SPY"
    expected_dates = tuple(pd.Timestamp(value).date() for value in expected_sessions)
    if not expected_dates or len(set(expected_dates)) != len(expected_dates):
        raise ValueError("V2 quality scan requires unique expected XNYS sessions.")
    daily_valid = _daily_valid_identities(daily_bars, set(expected_dates))
    paths, source_bytes = _verified_response_paths(corpus_root, set(identities))
    temporary = output_path.with_suffix(output_path.suffix + ".part")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if temporary.exists():
        temporary.unlink()
    writer: pq.ParquetWriter | None = None
    started = time.perf_counter()
    observed_rows = 0
    observed_sessions = 0
    valid_tokens = 0
    try:
        for instrument_id, symbol in sorted(identities.items()):
            records: dict[object, dict[str, object]] = {}
            for response_path, response_hash in paths.get(instrument_id, ()):
                chunk = pd.read_parquet(response_path)
                observed_rows += len(chunk)
                if chunk.empty:
                    continue
                if set(chunk["instrument_id"].astype(str)) != {instrument_id}:
                    raise ValueError(f"Minute response stable identity mismatch: {response_path}")
                timestamps = pd.to_datetime(chunk["timestamp"], errors="raise")
                local_dates = timestamps.dt.tz_convert("America/New_York").dt.date
                for session_date, session in chunk.groupby(local_dates, sort=True):
                    if session_date in records:
                        raise ValueError(
                            f"Duplicate v2 quality session across chunks: "
                            f"{instrument_id} {session_date}"
                        )
                    quality = assess_session_resolution_quality(
                        session.reset_index(drop=True),
                        daily_valid=(instrument_id, session_date) in daily_valid,
                    )
                    record = quality.to_dict()
                    record["source_response_sha256"] = response_hash
                    record["quality_protocol"] = "resolution-aware-v2"
                    records[session_date] = record
                    observed_sessions += 1
                    valid_tokens += quality.valid_token_count
            rows = [
                records.get(
                    session_date,
                    _missing_session_record(
                        instrument_id,
                        symbol,
                        session_date,
                        daily_valid=(instrument_id, session_date) in daily_valid,
                    ),
                )
                for session_date in expected_dates
            ]
            table = pa.Table.from_pylist(rows)
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise ValueError("V2 quality scan produced no instrument rows.")
    os.replace(temporary, output_path)
    elapsed = time.perf_counter() - started
    result = pd.read_parquet(
        output_path,
        columns=[
            "instrument_id",
            "daily_valid",
            "minute_exact_full_session",
            "token_valid_full_session",
            "tca_window_exact",
            "early_close",
            "valid_token_count",
        ],
    )
    expected_rows = len(identities) * len(expected_dates)
    if len(result) != expected_rows:
        raise ValueError("V2 quality output does not cover every instrument/session identity.")
    receipt: dict[str, Any] = {
        "schema_version": "paper-v2-formation-resolution-quality-v1",
        "protocol_id": "sparse-jepa-v2",
        "protocol_hash": protocol_hash,
        "status": "complete",
        "instrument_count": len(identities),
        "constituent_count": len(identities) - 1,
        "spy_instrument_id": spy_instrument_id,
        "expected_daily_sessions": len(expected_dates),
        "expected_standard_sessions": sum(
            len(expected_xnys_minutes(value)) == 390 for value in expected_dates
        ),
        "quality_rows": len(result),
        "observed_minute_rows": observed_rows,
        "observed_sessions": observed_sessions,
        "valid_tokens": valid_tokens,
        "daily_valid_rows": int(result["daily_valid"].sum()),
        "minute_exact_rows": int(result["minute_exact_full_session"].sum()),
        "token_valid_rows": int(result["token_valid_full_session"].sum()),
        "tca_window_exact_rows": int(result["tca_window_exact"].sum()),
        "early_close_rows": int(result["early_close"].sum()),
        "elapsed_seconds": elapsed,
        "minute_rows_per_second": observed_rows / max(elapsed, 1e-12),
        "sessions_per_second": observed_sessions / max(elapsed, 1e-12),
        "token_aggregation_attempts_per_second": (
            observed_sessions * TOKEN_COUNT / max(elapsed, 1e-12)
        ),
        "peak_rss_bytes": _peak_rss_bytes(),
        "source_response_bytes": source_bytes,
        "quality_parquet_bytes": output_path.stat().st_size,
        "quality_sha256": file_sha256(output_path),
    }
    write_json_atomic(receipt_path, receipt)
    return receipt


def _verified_response_paths(
    corpus_root: Path, identities: set[str]
) -> tuple[dict[str, tuple[tuple[Path, str], ...]], int]:
    grouped: dict[str, list[tuple[Path, str]]] = defaultdict(list)
    total_bytes = 0
    for receipt_path in sorted(corpus_root.glob("*.json")):
        receipt = read_json(receipt_path)
        if receipt.get("status") != "complete":
            continue
        instrument_id = str(receipt.get("instrument_id", ""))
        if instrument_id not in identities:
            continue
        response_path = receipt_path.with_suffix(".response")
        response_hash = file_sha256(response_path) if response_path.is_file() else ""
        if (
            receipt.get("feed") != "sip"
            or receipt.get("adjustment") != "raw"
            or not response_path.is_file()
            or receipt.get("response_sha256") != response_hash
        ):
            raise ValueError(f"Formation minute receipt is incompatible: {receipt_path}")
        grouped[instrument_id].append((response_path, response_hash))
        total_bytes += response_path.stat().st_size
    return {key: tuple(value) for key, value in grouped.items()}, total_bytes


def _daily_valid_identities(
    daily_bars: pd.DataFrame, expected_dates: set[date]
) -> set[tuple[str, date]]:
    if missing := set(PAPER_BAR_COLUMNS).difference(daily_bars.columns):
        raise ValueError(f"V2 daily corpus is missing columns: {sorted(missing)}")
    timestamps = pd.to_datetime(daily_bars["timestamp"], errors="coerce")
    if timestamps.isna().any() or timestamps.dt.tz is None:
        raise ValueError("V2 daily corpus timestamps must be timezone-aware.")
    local_dates = timestamps.dt.tz_convert("America/New_York").dt.date
    numeric = daily_bars.loc[
        :, ["open", "high", "low", "close", "volume", "trade_count", "vwap"]
    ].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy()).all(axis=1)
    positive_prices = (numeric[["open", "high", "low", "close", "vwap"]] > 0).all(axis=1)
    positive_activity = (numeric[["volume", "trade_count"]] > 0).all(axis=1)
    valid_ohlc = (numeric["high"] >= numeric[["open", "close", "low"]].max(axis=1)) & (
        numeric["low"] <= numeric[["open", "close", "high"]].min(axis=1)
    )
    valid_identity = daily_bars["instrument_id"].astype(str).str.strip().ne("") & daily_bars[
        "symbol"
    ].astype(str).str.strip().ne("")
    valid_mask = (
        local_dates.isin(expected_dates)
        & finite
        & positive_prices
        & positive_activity
        & valid_ohlc
        & valid_identity
    )
    identities = pd.DataFrame(
        {
            "instrument_id": daily_bars.loc[valid_mask, "instrument_id"].astype(str),
            "session_date": local_dates.loc[valid_mask],
        }
    )
    if identities.duplicated().any():
        duplicate = identities.loc[identities.duplicated(keep=False)].iloc[0]
        raise ValueError(
            "Duplicate valid v2 daily identity: "
            f"{(duplicate['instrument_id'], duplicate['session_date'])}"
        )
    return set(identities.itertuples(index=False, name=None))


def _missing_session_record(
    instrument_id: str, symbol: str, session_date: object, *, daily_valid: bool
) -> dict[str, object]:
    early_close = len(expected_xnys_minutes(session_date)) != 390
    return {
        "instrument_id": instrument_id,
        "symbol": symbol,
        "session_date": pd.Timestamp(session_date).date().isoformat(),
        "daily_valid": daily_valid,
        "minute_exact_full_session": False,
        "token_valid_full_session": False,
        "tca_window_exact": False,
        "early_close": early_close,
        "provider_gap_count": len(expected_xnys_minutes(session_date)),
        "observed_minute_count": 0,
        "valid_token_count": 0,
        "invalid_token_reason": "missing_provider_session",
        "source_response_sha256": "",
        "quality_protocol": "resolution-aware-v2",
    }


def _peak_rss_bytes() -> int | None:
    """Return the operating system's peak working set for this process."""
    if os.name != "nt":
        return None

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_: ClassVar[list[tuple[str, Any]]] = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    # WinDLL is intentionally resolved at runtime because it is absent from
    # ctypes on non-Windows hosts and therefore from their type stubs.
    win_dll = getattr(ctypes, "Win" + "DLL")
    kernel32 = win_dll("kernel32", use_last_error=True)
    psapi = win_dll("psapi", use_last_error=True)
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    psapi.GetProcessMemoryInfo.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ProcessMemoryCounters),
        ctypes.c_ulong,
    ]
    psapi.GetProcessMemoryInfo.restype = ctypes.c_int
    handle = kernel32.GetCurrentProcess()
    ok = psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
    return int(counters.PeakWorkingSetSize) if ok else None
