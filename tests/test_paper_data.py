from __future__ import annotations

import hashlib
from datetime import date
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from execsim.data.paper.acquisition import (
    _normalize_alpaca_frame,
    acquire_chunk,
    authorize_acquisition,
    monthly_chunks,
)
from execsim.data.paper.corporate_action_acquisition import (
    acquire_split_actions,
    normalize_split_actions,
)
from execsim.data.paper.corporate_actions import (
    apply_point_in_time_split_adjustment,
    point_in_time_split_factor,
)
from execsim.data.paper.formation import (
    build_formation_candidates,
    build_formation_candidates_from_corpus,
)
from execsim.data.paper.identity import resolve_provider_symbol, validate_symbol_history
from execsim.data.paper.manifests import file_sha256, read_json, stable_hash, write_json_atomic
from execsim.data.paper.partitions import (
    PAPER_FOLDS,
    resolve_fold_partition,
    validate_fold_membership,
)
from execsim.data.paper.planning import build_acquisition_plan
from execsim.data.paper.schemas import (
    InstrumentSymbolInterval,
    PaperDataConfig,
    ProviderResponse,
)
from execsim.data.paper.sources import acquire_constituent_identity_sources
from execsim.data.paper.universe import select_frozen_universe
from execsim.data.paper.validation import (
    classify_session,
    validate_exact_xnys_session,
    validate_paper_bars,
)
from execsim.ml.paper.orchestration import (
    _acquire_period,
    _audit_acquisition_period,
    _expected_primary_session_count,
)


def test_paper_acquisition_is_monthly_sip_and_disabled_by_default() -> None:
    chunks = monthly_chunks("asset-1", "aapl", date(2024, 1, 15), date(2024, 3, 2))

    assert [chunk.identity for chunk in chunks] == [
        "asset-1-AAPL-20240115-20240131-sip-raw",
        "asset-1-AAPL-20240201-20240229-sip-raw",
        "asset-1-AAPL-20240301-20240302-sip-raw",
    ]
    with pytest.raises(PermissionError, match="disabled"):
        authorize_acquisition(PaperDataConfig(), cli_enabled=True)


def test_corporate_action_requires_both_effective_and_known_times() -> None:
    actions = pd.DataFrame(
        {
            "instrument_id": ["asset-1"] * 4,
            "effective_date": ["2024-04-10", "2024-04-01", "2024-04-01", "2024-04-01"],
            "available_at": pd.to_datetime(
                [
                    "2024-03-01T12:00:00Z",
                    "2024-04-20T12:00:00Z",
                    "2024-03-20T12:00:00Z",
                    "2024-04-02T12:00:00Z",
                ],
                utc=True,
            ),
            "factor": [2.0, 3.0, 5.0, 7.0],
        }
    )
    before_effective = pd.Timestamp("2024-04-05T16:00:00Z")
    assert point_in_time_split_factor(
        actions,
        instrument_id="asset-1",
        observation_at=before_effective,
        market_information_as_of=before_effective,
    ) == pytest.approx(5.0 * 7.0)
    after_all_known = pd.Timestamp("2024-04-25T16:00:00Z")
    assert point_in_time_split_factor(
        actions,
        instrument_id="asset-1",
        observation_at=after_all_known,
        market_information_as_of=after_all_known,
    ) == pytest.approx(2.0 * 3.0 * 5.0 * 7.0)


def test_alpaca_split_normalization_uses_causal_pre_split_share_basis() -> None:
    intervals = (
        InstrumentSymbolInterval(
            "asset-a", "AAPL", date(2021, 1, 4), date(2025, 12, 31), "fixture"
        ),
        InstrumentSymbolInterval("asset-b", "XYZ", date(2021, 1, 4), date(2025, 12, 31), "fixture"),
    )
    payloads = (
        {
            "corporate_actions": {
                "forward_splits": [
                    {
                        "id": "forward-1",
                        "symbol": "AAPL",
                        "old_rate": 1,
                        "new_rate": 4,
                        "process_date": "2022-06-01",
                        "ex_date": "2022-06-03",
                    }
                ]
            },
            "next_page_token": "page-2",
        },
        {
            "corporate_actions": {
                "reverse_splits": [
                    {
                        "id": "reverse-1",
                        "symbol": "XYZ",
                        "old_rate": 10,
                        "new_rate": 1,
                        "process_date": "2023-08-10",
                        "ex_date": "2023-08-14",
                    }
                ]
            },
            "next_page_token": None,
        },
    )

    actions = normalize_split_actions(
        payloads,
        symbol_history=intervals,
        instrument_ids=("asset-a", "asset-b"),
        effective_end=date(2025, 12, 31),
    )

    assert actions["factor"].tolist() == pytest.approx([0.25, 10.0])
    assert actions["available_at"].tolist() == [
        pd.Timestamp("2022-06-02T00:00:00Z"),
        pd.Timestamp("2023-08-11T00:00:00Z"),
    ]
    bars = pd.DataFrame(
        {
            "open": [25.0],
            "high": [25.0],
            "low": [25.0],
            "close": [25.0],
            "vwap": [25.0],
            "volume": [400.0],
        }
    )
    restated = apply_point_in_time_split_adjustment(bars, pd.Series([0.25]))
    assert restated.loc[0, "close"] == pytest.approx(100.0)
    assert restated.loc[0, "volume"] == pytest.approx(100.0)


def test_corporate_action_acquisition_is_paginated_idempotent_and_fail_closed(
    tmp_path: Path,
) -> None:
    intervals = (
        InstrumentSymbolInterval(
            "asset-a", "AAPL", date(2021, 1, 4), date(2025, 12, 31), "fixture"
        ),
    )
    pages = {
        None: {
            "corporate_actions": {
                "forward_splits": [
                    {
                        "id": "forward-1",
                        "symbol": "AAPL",
                        "old_rate": 1,
                        "new_rate": 4,
                        "process_date": "2022-06-01",
                        "ex_date": "2022-06-03",
                    }
                ]
            },
            "next_page_token": "page-2",
        },
        "page-2": {
            "corporate_actions": {"reverse_splits": []},
            "next_page_token": None,
        },
    }
    calls: list[str | None] = []

    def fetch(params):
        token = params.get("page_token")
        calls.append(token)
        return pages[token], {"X-RateLimit-Remaining": "199"}

    output = tmp_path / "actions.parquet"
    raw = tmp_path / "actions.raw.json"
    receipt = tmp_path / "acquisition-receipt.json"
    kwargs = {
        "start": date(2021, 1, 4),
        "end": date(2025, 12, 31),
        "output_path": output,
        "raw_output_path": raw,
        "receipt_path": receipt,
        "paper_config_hash": "a" * 64,
        "config": PaperDataConfig(allow_network=True, paper_config_hash="a" * 64),
        "cli_enabled": True,
    }
    result = acquire_split_actions(intervals, ("asset-a",), fetch_page=fetch, **kwargs)
    reused = acquire_split_actions(
        intervals,
        ("asset-a",),
        fetch_page=lambda _: (_ for _ in ()).throw(AssertionError("network reused")),
        **kwargs,
    )

    assert calls == [None, "page-2"]
    assert result == reused
    assert result["row_count"] == 1
    assert result["available_at_policy"] == "provider_process_date_plus_one_utc_day"
    assert file_sha256(output) == result["source_sha256"]

    with pytest.raises(RuntimeError, match="BLOCKED: corporate-action symbol"):
        normalize_split_actions(
            (
                {
                    "forward_splits": [
                        {
                            "id": "unknown-1",
                            "symbol": "UNKNOWN",
                            "old_rate": 1,
                            "new_rate": 2,
                            "process_date": "2022-01-01",
                            "ex_date": "2022-01-03",
                        }
                    ]
                },
            ),
            symbol_history=intervals,
            instrument_ids=("asset-a",),
            effective_end=date(2025, 12, 31),
        )


def test_authorized_chunk_is_atomic_idempotent_and_checksummed(tmp_path) -> None:
    chunk = monthly_chunks("asset-1", "AAPL", date(2024, 1, 1), date(2024, 1, 31))[0]
    calls = 0

    def fetch(_chunk):
        nonlocal calls
        calls += 1
        timestamps = pd.date_range(
            "2024-01-03 09:30", periods=390, freq="min", tz="America/New_York"
        )
        frame = pd.DataFrame(
            {
                "instrument_id": "asset-1",
                "symbol": "AAPL",
                "timestamp": timestamps,
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.0,
                "volume": 1_000,
                "trade_count": 10,
                "vwap": 100.0,
            }
        )
        buffer = BytesIO()
        frame.to_parquet(buffer, index=False)
        return ProviderResponse(buffer.getvalue(), 390)

    config = PaperDataConfig(allow_network=True)
    first = acquire_chunk(
        chunk,
        output_directory=tmp_path,
        fetch=fetch,
        config=config,
        cli_enabled=True,
    )
    second = acquire_chunk(
        chunk,
        output_directory=tmp_path,
        fetch=fetch,
        config=config,
        cli_enabled=True,
    )

    assert first == second
    assert first.row_count == 390
    assert calls == 1
    assert not list(tmp_path.glob("*.part"))


def test_nonempty_provider_chunk_with_invalid_bar_values_fails_closed(tmp_path: Path) -> None:
    chunk = monthly_chunks("asset-1", "AAPL", date(2024, 1, 1), date(2024, 1, 31))[0]
    timestamps = pd.date_range("2024-01-03 09:30", periods=390, freq="min", tz="America/New_York")
    frame = pd.DataFrame(
        {
            "instrument_id": "asset-1",
            "symbol": "AAPL",
            "timestamp": timestamps,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1_000,
            "trade_count": 10,
            "vwap": 100.0,
        }
    )
    frame.loc[0, "high"] = 98.0
    buffer = BytesIO()
    frame.to_parquet(buffer, index=False)

    with pytest.raises(RuntimeError, match="Acquisition failed") as raised:
        acquire_chunk(
            chunk,
            output_directory=tmp_path,
            fetch=lambda _: ProviderResponse(buffer.getvalue(), len(frame)),
            config=PaperDataConfig(allow_network=True),
            cli_enabled=True,
            max_attempts=1,
        )

    assert "structurally invalid bars" in str(raised.value.__cause__)
    receipt = read_json(tmp_path / f"{chunk.identity}.json")
    assert receipt["status"] == "failed"
    assert not (tmp_path / f"{chunk.identity}.response").exists()


def test_target_acquisition_audit_requires_every_compatible_terminal_receipt(
    tmp_path: Path,
) -> None:
    interval = InstrumentSymbolInterval(
        "asset-1", "AAPL", date(2024, 1, 1), date(2024, 1, 31), "fixture"
    )
    with pytest.raises(RuntimeError, match="receipt set is incomplete"):
        _audit_acquisition_period(
            ("asset-1",),
            (interval,),
            start=date(2024, 1, 1),
            end=date(2024, 1, 31),
            output=tmp_path,
            monthly_chunks=monthly_chunks,
            paper_config_hash="a" * 64,
        )

    chunk = monthly_chunks("asset-1", "AAPL", date(2024, 1, 1), date(2024, 1, 31))[0]
    timestamps = pd.date_range("2024-01-03 09:30", periods=390, freq="min", tz="America/New_York")
    frame = pd.DataFrame(
        {
            "instrument_id": "asset-1",
            "symbol": "AAPL",
            "timestamp": timestamps,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1_000,
            "trade_count": 10,
            "vwap": 100.0,
        }
    )
    buffer = BytesIO()
    frame.to_parquet(buffer, index=False)
    acquire_chunk(
        chunk,
        output_directory=tmp_path,
        fetch=lambda _: ProviderResponse(buffer.getvalue(), len(frame)),
        config=PaperDataConfig(allow_network=True, paper_config_hash="a" * 64),
        cli_enabled=True,
    )
    audit = _audit_acquisition_period(
        ("asset-1",),
        (interval,),
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
        output=tmp_path,
        monthly_chunks=monthly_chunks,
        paper_config_hash="a" * 64,
    )
    assert audit["status"] == "complete"
    assert audit["complete_chunks"] == 1

    receipt_path = tmp_path / f"{chunk.identity}.json"
    receipt = read_json(receipt_path)
    write_json_atomic(receipt_path, {**receipt, "symbol": "MSFT"})
    with pytest.raises(ValueError, match="symbol"):
        _audit_acquisition_period(
            ("asset-1",),
            (interval,),
            start=date(2024, 1, 1),
            end=date(2024, 1, 31),
            output=tmp_path,
            monthly_chunks=monthly_chunks,
            paper_config_hash="a" * 64,
        )


def test_zero_row_acquisition_fails_and_sourced_ticker_history_resolves_identity(tmp_path) -> None:
    chunk = monthly_chunks("asset-1", "OLD", date(2024, 1, 1), date(2024, 1, 31))[0]
    with pytest.raises(RuntimeError, match="Acquisition failed"):
        acquire_chunk(
            chunk,
            output_directory=tmp_path,
            fetch=lambda _: ProviderResponse(b"not-empty", 0),
            config=PaperDataConfig(allow_network=True),
            cli_enabled=True,
            max_attempts=1,
        )
    intervals = (
        InstrumentSymbolInterval("asset-1", "OLD", date(2022, 1, 1), date(2024, 1, 31), "provider"),
        InstrumentSymbolInterval(
            "asset-1", "NEW", date(2024, 2, 1), date(2025, 12, 31), "provider"
        ),
    )
    validate_symbol_history(intervals)
    assert resolve_provider_symbol(intervals, "asset-1", date(2024, 1, 15)) == "OLD"
    assert resolve_provider_symbol(intervals, "asset-1", date(2024, 2, 1)) == "NEW"
    with pytest.raises(RuntimeError, match="BLOCKED"):
        resolve_provider_symbol(intervals, "asset-2", date(2024, 2, 1))


def test_nonempty_incomplete_chunk_is_retained_for_exclusion_accounting(tmp_path: Path) -> None:
    chunk = monthly_chunks("asset-1", "APD", date(2024, 1, 1), date(2024, 1, 31))[0]
    timestamps = pd.date_range("2024-01-03 09:30", periods=389, freq="min", tz="America/New_York")
    frame = pd.DataFrame(
        {
            "instrument_id": "asset-1",
            "symbol": "APD",
            "timestamp": timestamps,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1_000,
            "trade_count": 10,
            "vwap": 100.0,
        }
    )
    buffer = BytesIO()
    frame.to_parquet(buffer, index=False)
    receipt = acquire_chunk(
        chunk,
        output_directory=tmp_path,
        fetch=lambda _: ProviderResponse(buffer.getvalue(), len(frame)),
        config=PaperDataConfig(allow_network=True),
        cli_enabled=True,
        max_attempts=1,
    )

    assert receipt.status == "complete"
    assert receipt.observed_sessions == 0
    assert receipt.expected_sessions > 0


def test_empty_delisted_symbol_response_normalizes_to_explicit_zero_row_schema() -> None:
    chunk = monthly_chunks("asset-1", "TIF", date(2021, 2, 1), date(2021, 2, 28))[0]
    normalized = _normalize_alpaca_frame(pd.DataFrame(), chunk)

    assert normalized.empty
    assert {"instrument_id", "symbol", "timestamp", "trade_count", "vwap"}.issubset(
        normalized.columns
    )


def test_acquisition_period_uses_sourced_partial_aliases_and_blocks_trading_day_gaps(
    tmp_path,
) -> None:
    intervals = (
        InstrumentSymbolInterval("asset-1", "OLD", date(2024, 1, 1), date(2024, 1, 15), "src"),
        InstrumentSymbolInterval("asset-1", "NEW", date(2024, 1, 16), date(2024, 1, 31), "src"),
    )
    acquired = []

    def record(chunk, **_):
        acquired.append(chunk)

    completed = _acquire_period(
        ("asset-1",),
        intervals,
        start=date(2024, 1, 1),
        end=date(2024, 1, 31),
        output=tmp_path,
        fetcher=object(),
        data=PaperDataConfig(allow_network=True),
        acquire_chunk=record,
        monthly_chunks=monthly_chunks,
    )

    assert completed == 2
    assert [(item.symbol, item.start, item.end) for item in acquired] == [
        ("OLD", date(2024, 1, 1), date(2024, 1, 15)),
        ("NEW", date(2024, 1, 16), date(2024, 1, 31)),
    ]
    with pytest.raises(RuntimeError, match="BLOCKED"):
        _acquire_period(
            ("asset-1",),
            intervals[:1],
            start=date(2024, 1, 1),
            end=date(2024, 1, 31),
            output=tmp_path,
            fetcher=object(),
            data=PaperDataConfig(allow_network=True),
            acquire_chunk=record,
            monthly_chunks=monthly_chunks,
        )


def test_acquisition_period_records_zero_row_month_without_claiming_completion(
    tmp_path: Path,
) -> None:
    intervals = (
        InstrumentSymbolInterval("asset-1", "OLD", date(2024, 1, 1), date(2024, 2, 29), "src"),
    )
    attempted = []

    def record(chunk, **_):
        attempted.append(chunk)
        if chunk.start.month == 1:
            cause = ValueError("Provider response row count is zero or does not match metadata.")
            raise RuntimeError("Acquisition failed") from cause

    completed = _acquire_period(
        ("asset-1",),
        intervals,
        start=date(2024, 1, 1),
        end=date(2024, 2, 29),
        output=tmp_path,
        fetcher=object(),
        data=PaperDataConfig(allow_network=True),
        acquire_chunk=record,
        monthly_chunks=monthly_chunks,
    )

    assert completed == 1
    assert [item.start.month for item in attempted] == [1, 2]


def test_streaming_formation_statistics_match_in_memory_builder(tmp_path: Path) -> None:
    snapshot = pd.DataFrame(
        {
            "instrument_id": ["asset-1"],
            "symbol": ["AAPL"],
            "security_type": ["ordinary_common_stock"],
            "effective_date": ["2021-01-04"],
            "source": ["fixture"],
        }
    )
    sessions = []
    for session_date, close, volume in (
        ("2021-01-04", 100.0, 1_000),
        ("2021-02-01", 120.0, 2_000),
    ):
        timestamps = pd.date_range(
            f"{session_date} 09:30", periods=390, freq="min", tz="America/New_York"
        )
        sessions.append(
            pd.DataFrame(
                {
                    "instrument_id": "asset-1",
                    "symbol": "AAPL",
                    "timestamp": timestamps,
                    "open": close,
                    "high": close + 1,
                    "low": close - 1,
                    "close": close,
                    "volume": volume,
                    "trade_count": 10,
                    "vwap": close,
                }
            )
        )
    for index, session in enumerate(sessions):
        stem = tmp_path / f"chunk-{index}"
        session.to_parquet(stem.with_suffix(".response"), index=False)
        write_json_atomic(
            stem.with_suffix(".json"),
            {"status": "complete", "instrument_id": "asset-1"},
        )

    in_memory, expected_exclusions = build_formation_candidates(
        snapshot, pd.concat(sessions, ignore_index=True), expected_session_count=2
    )
    streamed, actual_exclusions = build_formation_candidates_from_corpus(
        snapshot, tmp_path, expected_session_count=2
    )

    pd.testing.assert_frame_equal(streamed.reset_index(drop=True), in_memory.reset_index(drop=True))
    assert actual_exclusions == expected_exclusions


def test_formation_completeness_denominator_excludes_early_closes() -> None:
    import exchange_calendars as xcals

    calendar = xcals.get_calendar("XNYS")

    assert _expected_primary_session_count(calendar, "2021-01-04", "2021-12-31") == 251


def test_universe_is_frozen_by_formation_liquidity_with_stable_ties() -> None:
    candidates = pd.DataFrame(
        {
            "instrument_id": [f"id-{index:03d}" for index in range(101)],
            "symbol": [f"S{index:03d}" for index in range(101)],
            "security_type": ["ordinary_common_stock"] * 101,
            "in_sp500_on_formation_date": [True] * 101,
            "median_price": [10.0] * 101,
            "session_completeness": [0.99] * 101,
            "median_daily_dollar_volume": list(range(101, 0, -1)),
        }
    )

    members = select_frozen_universe(candidates)

    assert len(members) == 100
    assert members[0].instrument_id == "id-000"
    assert members[-1].instrument_id == "id-099"
    assert members[0].liquidity_group == 1
    assert members[-1].liquidity_group == 5


def test_split_adjustment_preserves_dollar_notional_and_fold_dates_are_locked() -> None:
    bars = pd.DataFrame(
        {
            "open": [100.0, 50.0],
            "high": [101.0, 51.0],
            "low": [99.0, 49.0],
            "close": [100.0, 50.0],
            "vwap": [100.0, 50.0],
            "volume": [10.0, 20.0],
        }
    )
    adjusted = apply_point_in_time_split_adjustment(bars, pd.Series([2.0, 1.0]))

    assert np.allclose(bars["vwap"] * bars["volume"], adjusted["vwap"] * adjusted["volume"])
    with pytest.raises(ValueError, match="known after"):
        apply_point_in_time_split_adjustment(
            bars,
            pd.Series([2.0, 1.0]),
            factor_available_at=pd.Series(pd.to_datetime(["2024-01-03", "2024-01-01"], utc=True)),
            as_of=pd.Timestamp("2024-01-02", tz="UTC"),
        )
    assert PAPER_FOLDS[0].partition(date(2024, 4, 1)) == "test"
    assert PAPER_FOLDS[1].partition(date(2024, 4, 1)) == "train"
    assert stable_hash({"b": 2, "a": 1}) == stable_hash({"a": 1, "b": 2})


def test_regular_session_validation_never_treats_missing_minutes_as_zero() -> None:
    timestamps = pd.date_range("2024-01-03 09:30", periods=389, freq="min", tz="America/New_York")
    bars = pd.DataFrame(
        {
            "instrument_id": "asset-1",
            "symbol": "AAPL",
            "timestamp": timestamps,
            "open": 100.0,
            "high": 100.1,
            "low": 99.9,
            "close": 100.0,
            "volume": 100,
            "trade_count": 10,
            "vwap": 100.0,
        }
    )

    errors = validate_paper_bars(bars)

    assert any("requires 390 observed minutes" in error for error in errors)
    assert classify_session(observed_minutes=390, calendar_early_close=False) == "primary_regular"
    assert (
        classify_session(observed_minutes=210, calendar_early_close=True)
        == "robustness_early_close"
    )


def test_fold_membership_is_derived_and_expanding_dates_are_fold_scoped() -> None:
    assert resolve_fold_partition("fold-1", date(2024, 4, 1)) == "test"
    assert resolve_fold_partition("fold-2", date(2024, 4, 1)) == "train"
    validate_fold_membership(
        (
            ("fold-1", "asset-1", date(2024, 4, 1), "test"),
            ("fold-2", "asset-1", date(2024, 4, 1), "train"),
        )
    )
    with pytest.raises(ValueError, match="mismatch"):
        validate_fold_membership((("fold-1", "asset-1", date(2024, 4, 1), "train"),))


def test_exact_xnys_grid_rejects_compensating_extra_timezone_and_order_corruption() -> None:
    timestamps = pd.date_range("2024-01-03 09:30", periods=390, freq="min", tz="America/New_York")
    frame = pd.DataFrame(
        {
            "instrument_id": "asset-1",
            "symbol": "AAPL",
            "timestamp": timestamps,
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1_000,
            "trade_count": 10,
            "vwap": 100.0,
        }
    )
    assert not validate_exact_xnys_session(frame)
    compensated = frame.loc[
        frame["timestamp"] != pd.Timestamp("2024-01-03 10:00", tz="America/New_York")
    ].copy()
    extra = frame.iloc[[0]].copy()
    extra["timestamp"] = pd.Timestamp("2024-01-03 16:00", tz="America/New_York")
    compensated = pd.concat((compensated, extra), ignore_index=True).sort_values("timestamp")
    assert any("exact XNYS grid" in error for error in validate_exact_xnys_session(compensated))
    wrong_timezone = frame.copy()
    wrong_timezone["timestamp"] = wrong_timezone["timestamp"].dt.tz_convert("UTC")
    assert any("timezone" in error for error in validate_exact_xnys_session(wrong_timezone))
    reordered = frame.copy()
    reordered.iloc[[0, 1]] = reordered.iloc[[1, 0]].to_numpy()
    assert any("strictly ordered" in error for error in validate_exact_xnys_session(reordered))
    duplicated = frame.copy()
    duplicated.loc[1, "timestamp"] = duplicated.loc[0, "timestamp"]
    assert any("duplicate" in error for error in validate_exact_xnys_session(duplicated))


def test_pinned_constituent_source_builds_snapshot_identity_and_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rows = [
        {
            "symbol": f"S{index:03d}",
            "cik": f"{index + 1:010d}",
            "name": f"Issuer {index}",
            "sector": "industrials",
            "date_added": "2020-01-01",
            "date_removed": "",
            "created_at": "2020-01-01",
        }
        for index in range(500)
    ]
    rows.append(
        {
            **rows[0],
            "symbol": "RENAMED",
            "created_at": "2022-06-01",
        }
    )
    content = pd.DataFrame(rows).to_csv(index=False).encode()
    monkeypatch.setattr(
        "execsim.data.paper.sources.COMPONENTS_SHA256", hashlib.sha256(content).hexdigest()
    )
    snapshot_path = tmp_path / "formation" / "constituents.parquet"
    ticker_path = tmp_path / "formation" / "ticker_history.parquet"
    receipt = acquire_constituent_identity_sources(
        formation_date=date(2021, 1, 4),
        target_end=date(2025, 12, 31),
        snapshot_output=snapshot_path,
        ticker_history_output=ticker_path,
        receipt_output=tmp_path / "formation-source.json",
        spy_instrument_id="benchmark-spy",
        content=content,
    )
    snapshot = pd.read_parquet(snapshot_path)
    ticker = pd.read_parquet(ticker_path)
    intervals = tuple(
        InstrumentSymbolInterval(
            row.instrument_id,
            row.symbol,
            date.fromisoformat(row.start),
            date.fromisoformat(row.end),
            row.source,
        )
        for row in ticker.itertuples(index=False)
    )
    plan = build_acquisition_plan(
        snapshot=snapshot,
        intervals=intervals,
        formation_start=date(2021, 1, 4),
        formation_end=date(2021, 12, 31),
        target_start=date(2022, 1, 3),
        target_end=date(2025, 12, 31),
        target_universe_size=100,
        spy_instrument_id="benchmark-spy",
        output_directory=tmp_path / "plan",
        paper_config_hash="f" * 64,
    )

    assert receipt["snapshot_rows"] == 500
    assert ticker.loc[ticker["instrument_id"].str.endswith("S000"), "symbol"].tolist() == [
        "S000",
        "RENAMED",
    ]
    assert plan["formation"]["candidate_instruments_including_spy"] == 501
    assert (tmp_path / "plan" / "ACQUISITION_PLAN.md").is_file()
