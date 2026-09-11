from __future__ import annotations

from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from execsim.data.paper.formation_quality_v2 import scan_formation_resolution_quality
from execsim.data.paper.formation_report_v2 import write_v2_formation_bundle
from execsim.data.paper.formation_v2 import (
    build_daily_formation_candidates,
    select_v2_universe,
)
from execsim.data.paper.manifests import file_sha256, write_json_atomic
from execsim.data.paper.resolution_quality import (
    aggregate_observed_tokens,
    assess_session_resolution_quality,
    validate_daily_observation,
)
from execsim.ml.paper.configs import _portable_document_sha256, load_paper_config
from execsim.ml.paper.orchestration import validate_data_stage
from execsim.ml.sequences.builder import build_session_sequence


def _minute_session(symbol: str = "AAPL") -> pd.DataFrame:
    timestamps = pd.date_range("2021-05-05 09:30", periods=390, freq="min", tz="America/New_York")
    close = 100.0 + np.arange(390) / 100.0
    return pd.DataFrame(
        {
            "instrument_id": f"id-{symbol.lower()}",
            "symbol": symbol,
            "timestamp": timestamps,
            "open": close - 0.02,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": np.arange(390) + 1,
            "trade_count": np.arange(390) + 2,
            "vwap": close - 0.01,
        }
    )


def _daily_frame(count: int, *, instrument_id: str = "id-aapl") -> pd.DataFrame:
    timestamps = pd.date_range("2021-01-04 00:00", periods=count, freq="B", tz="America/New_York")
    return pd.DataFrame(
        {
            "instrument_id": instrument_id,
            "symbol": "AAPL",
            "timestamp": timestamps,
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": 101.0,
            "volume": 1_000,
            "trade_count": 100,
            "vwap": 100.5,
        }
    )


def test_exact_and_sparse_minute_sessions_have_separate_quality() -> None:
    exact = assess_session_resolution_quality(_minute_session())
    sparse = _minute_session().drop(index=[1, 18, 80, 200, 388]).reset_index(drop=True)
    sparse_quality = assess_session_resolution_quality(sparse)

    assert exact.minute_exact_full_session
    assert exact.token_valid_full_session
    assert exact.tca_window_exact
    assert not sparse_quality.minute_exact_full_session
    assert sparse_quality.token_valid_full_session
    assert not sparse_quality.tca_window_exact
    assert sparse_quality.provider_gap_count == 5
    assert sparse_quality.valid_token_count == 26
    assert sparse_quality.token_observed_bar_counts == (
        14,
        14,
        15,
        15,
        15,
        14,
        *([15] * 7),
        14,
        *([15] * 11),
        14,
    )
    assert sparse_quality.token_provider_gap_counts == tuple(
        15 - value for value in sparse_quality.token_observed_bar_counts
    )


@pytest.mark.parametrize(
    "bars",
    [
        _minute_session(),
        _minute_session().drop(index=[1, 18, 80, 200, 388]).reset_index(drop=True),
        _minute_session().drop(index=range(30, 45)).reset_index(drop=True),
        _minute_session().drop(index=[61]).reset_index(drop=True),
    ],
)
def test_quality_only_scan_matches_materialized_token_validation(bars: pd.DataFrame) -> None:
    quality_only = assess_session_resolution_quality(bars)
    materialized, _ = assess_session_resolution_quality(bars, return_tokens=True)

    assert quality_only.to_dict() == materialized.to_dict()


def test_token_aggregation_uses_observations_without_fake_minute_rows() -> None:
    bars = _minute_session().drop(index=[1, 2, 3]).reset_index(drop=True)
    tokens = aggregate_observed_tokens(bars)
    first = bars.iloc[:12]
    first_token = tokens.iloc[0]
    expected_vwap = float(np.average(first["vwap"], weights=first["volume"]))
    expected_volatility = float(np.sqrt(np.square(np.diff(np.log(first["close"]))).sum()))

    assert len(tokens) == 26
    assert first_token["observed_bar_count"] == 12
    assert first_token["volume"] == pytest.approx(first["volume"].sum())
    assert first_token["trade_count"] == pytest.approx(first["trade_count"].sum())
    assert first_token["vwap"] == pytest.approx(expected_vwap)
    assert first_token["open"] == pytest.approx(first["open"].iloc[0])
    assert first_token["high"] == pytest.approx(first["high"].max())
    assert first_token["low"] == pytest.approx(first["low"].min())
    assert first_token["close"] == pytest.approx(first["close"].iloc[-1])
    assert first_token["realized_volatility"] == pytest.approx(expected_volatility)
    assert first_token["timestamp"] == pd.Timestamp("2021-05-05 09:45", tz="America/New_York")


def test_missing_or_insufficient_token_is_rejected_without_interpolation() -> None:
    missing_interval = _minute_session().drop(index=range(30, 45)).reset_index(drop=True)
    one_observation = _minute_session().drop(index=range(46, 60)).reset_index(drop=True)

    missing_quality = assess_session_resolution_quality(missing_interval)
    one_quality = assess_session_resolution_quality(one_observation)

    assert not missing_quality.token_valid_full_session
    assert "token_02_observed_bars_lt_2" in missing_quality.invalid_token_reason
    assert not one_quality.token_valid_full_session
    assert "token_03_observed_bars_lt_2" in one_quality.invalid_token_reason
    with pytest.raises(ValueError, match="Token-invalid"):
        aggregate_observed_tokens(missing_interval)


def test_v2_sequence_cross_token_return_uses_previous_observed_close() -> None:
    bars = _minute_session().drop(index=[1, 18, 80, 200, 388]).reset_index(drop=True)
    tokens = aggregate_observed_tokens(bars)
    record = build_session_sequence(
        bars,
        instrument_id="id-aapl",
        symbol="AAPL",
        source_sha256="a" * 64,
        cutoff="2021-05-04",
        previous_close=99.0,
        data_classification="synthetic_fixture",
        quality_protocol="resolution-aware-v2",
    )

    assert record.features[0, 0] == pytest.approx(np.log(tokens.loc[0, "close"] / 99.0))
    assert record.features[1, 0] == pytest.approx(
        np.log(tokens.loc[1, "close"] / tokens.loc[0, "close"])
    )


def test_v2_sequence_rejects_token_invalid_and_early_close_sessions() -> None:
    token_invalid = _minute_session().drop(index=range(30, 45)).reset_index(drop=True)
    early_close = _minute_session().iloc[:210].copy()
    early_close["timestamp"] = pd.date_range(
        "2021-11-26 09:30", periods=210, freq="min", tz="America/New_York"
    )
    for bars in (token_invalid, early_close):
        with pytest.raises(ValueError, match="v2 token quality"):
            build_session_sequence(
                bars,
                instrument_id="id-aapl",
                symbol="AAPL",
                source_sha256="a" * 64,
                cutoff="2021-05-04",
                data_classification="synthetic_fixture",
                quality_protocol="resolution-aware-v2",
            )


def test_tca_requires_exact_1030_through_1529_but_not_unrelated_minutes() -> None:
    outside_window_gap = _minute_session().drop(index=[1]).reset_index(drop=True)
    inside_window_gap = _minute_session().drop(index=[61]).reset_index(drop=True)

    assert assess_session_resolution_quality(outside_window_gap).tca_window_exact
    assert not assess_session_resolution_quality(inside_window_gap).tca_window_exact


def test_spy_uses_identical_token_quality_contract() -> None:
    stock = _minute_session("AAPL").drop(index=[1, 18]).reset_index(drop=True)
    spy = _minute_session("SPY").drop(index=[1, 18]).reset_index(drop=True)

    assert assess_session_resolution_quality(stock).token_valid_full_session
    assert assess_session_resolution_quality(spy).token_valid_full_session


def test_v2_target_validation_accepts_token_valid_gaps_and_checks_session_symbol(
    tmp_path: Path,
) -> None:
    loaded = load_paper_config(Path("configs/paper/sparse_jepa_v2"))
    universe_path = tmp_path / "universe.json"
    write_json_atomic(
        universe_path,
        {
            "protocol_id": "sparse-jepa-v2",
            "members": [{"instrument_id": "id-aapl", "formation_symbol": "AAPL"}],
            "symbol_history": [
                {
                    "instrument_id": "id-aapl",
                    "symbol": "AAPL",
                    "start": "2021-01-01",
                    "end": "2025-12-31",
                    "source": "fixture",
                },
                {
                    "instrument_id": "benchmark-spy",
                    "symbol": "SPY",
                    "start": "2021-01-01",
                    "end": "2025-12-31",
                    "source": "fixture",
                },
            ],
        },
    )
    sections = {
        **loaded.sections,
        "data": {**loaded.data, "universe_manifest": str(universe_path)},
    }
    config = replace(loaded, sections=sections)
    spy = _minute_session("SPY").drop(index=[1, 18, 80, 200, 388])
    spy["instrument_id"] = "benchmark-spy"
    corpus = pd.concat(
        [
            _minute_session().drop(index=[1, 18, 80, 200, 388]),
            spy,
        ],
        ignore_index=True,
    )
    corpus_path = tmp_path / "target.parquet"
    corpus.to_parquet(corpus_path, index=False)

    accepted = validate_data_stage(config, corpus_path)
    assert accepted["valid"] is True
    assert accepted["quality_protocol"] == "resolution-aware-v2"
    assert accepted["valid_sessions"] == 2

    corpus_root = tmp_path / "target-root"
    corpus_root.mkdir()
    for instrument_id, frame in corpus.groupby("instrument_id", sort=True):
        frame.to_parquet(corpus_root / f"{instrument_id}-fixture.response", index=False)
    streamed = validate_data_stage(config, corpus_root)
    assert streamed["valid"] is True
    assert streamed["valid_sessions"] == accepted["valid_sessions"]
    assert streamed["session_quality"] == accepted["session_quality"]

    incompatible = corpus.copy()
    incompatible.loc[incompatible["instrument_id"] == "id-aapl", "symbol"] = "META"
    incompatible.to_parquet(corpus_path, index=False)
    rejected = validate_data_stage(config, corpus_path)
    assert rejected["valid"] is False
    assert "does not match sourced symbol" in rejected["invalid_sessions"][0]["errors"][0]


def test_daily_quality_accepts_early_close_as_an_observed_trading_day() -> None:
    row = pd.Series(
        {
            "instrument_id": "id-aapl",
            "symbol": "AAPL",
            "timestamp": pd.Timestamp("2021-11-26 00:00", tz="America/New_York"),
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": 101.0,
            "volume": 1_000,
            "trade_count": 100,
            "vwap": 100.5,
        }
    )

    assert not validate_daily_observation(row, expected_dates={date(2021, 11, 26)})


def test_daily_formation_uses_daily_95_percent_not_minute_completeness() -> None:
    expected = tuple(pd.date_range("2021-01-04", periods=20, freq="B"))
    daily = _daily_frame(19)
    snapshot = pd.DataFrame(
        {
            "instrument_id": ["id-aapl"],
            "symbol": ["AAPL"],
            "security_type": ["ordinary_common_stock"],
        }
    )

    candidates = build_daily_formation_candidates(
        snapshot,
        daily,
        expected_session_dates=expected,
        identity_source_hash="a" * 64,
    )

    assert candidates.loc[0, "observed_valid_daily_sessions"] == 19
    assert candidates.loc[0, "daily_completeness"] == pytest.approx(0.95)
    assert "daily_completeness_below_95_percent" not in candidates.loc[0, "exclusion_reasons"]


def test_v2_ranking_depends_on_daily_values_not_minute_gap_counts() -> None:
    candidates = pd.DataFrame(
        {
            "instrument_id": [f"id-{index:03d}" for index in range(100)],
            "formation_symbol": [f"S{index:03d}" for index in range(100)],
            "security_type": "ordinary_common_stock",
            "in_sp500_on_formation_date": True,
            "median_daily_price": 10.0,
            "daily_completeness": 1.0,
            "median_daily_share_volume": 1_000.0,
            "median_daily_dollar_volume": np.arange(100, 0, -1),
        }
    )

    members = select_v2_universe(candidates)

    assert members[0].instrument_id == "id-000"
    assert members[-1].instrument_id == "id-099"


def test_v1_terminal_evidence_is_immutable_and_v2_paths_are_distinct() -> None:
    v1 = load_paper_config(Path("configs/paper/sparse_jepa"))
    v2 = load_paper_config(Path("configs/paper/sparse_jepa_v2"))
    evidence = Path("configs/paper/sparse_jepa/v1-evidence-final.json")

    assert file_sha256(evidence) == (
        "24bf983669df9b699abf2b7193d5b42dd464bec08b4993fd88a4916ae6b37585"
    )
    assert v1.paper_run_id == "sparse-jepa-v1"
    assert v2.paper_run_id == "sparse-jepa-v2"
    assert v1.artifact_root != v2.artifact_root
    assert v1.data["universe_manifest"] != v2.data["universe_manifest"]
    assert v1.data["target_corpus_root"] != v2.data["target_corpus_root"]


def test_normative_document_hash_is_portable_across_line_endings(tmp_path: Path) -> None:
    lf_path = tmp_path / "lf.md"
    crlf_path = tmp_path / "crlf.md"
    lf_path.write_bytes(b"# Protocol\n\nSame content.\n")
    crlf_path.write_bytes(b"# Protocol\r\n\r\nSame content.\r\n")

    assert _portable_document_sha256(lf_path) == _portable_document_sha256(crlf_path)


def test_resolution_quality_manifest_keeps_daily_token_and_minute_fields(
    tmp_path: Path,
) -> None:
    session_dates = (pd.Timestamp("2021-05-05"), pd.Timestamp("2021-05-06"))
    corpus = tmp_path / "minute"
    corpus.mkdir()
    stock = pd.concat(
        [
            _minute_session().assign(
                timestamp=pd.date_range(day, periods=390, freq="min", tz="America/New_York")
            )
            for day in ("2021-05-05 09:30", "2021-05-06 09:30")
        ],
        ignore_index=True,
    ).drop(index=[1])
    spy = stock.assign(instrument_id="benchmark-spy", symbol="SPY")
    for instrument_id, symbol, frame in (
        ("id-aapl", "AAPL", stock),
        ("benchmark-spy", "SPY", spy),
    ):
        receipt_path = corpus / f"{instrument_id}.json"
        response_path = receipt_path.with_suffix(".response")
        frame.to_parquet(response_path, index=False)
        write_json_atomic(
            receipt_path,
            {
                "status": "complete",
                "instrument_id": instrument_id,
                "symbol": symbol,
                "feed": "sip",
                "adjustment": "raw",
                "response_sha256": file_sha256(response_path),
            },
        )
    daily = pd.concat(
        [
            _daily_frame(2),
            _daily_frame(2, instrument_id="benchmark-spy").assign(symbol="SPY"),
        ],
        ignore_index=True,
    )
    daily["timestamp"] = pd.to_datetime(
        [
            "2021-05-05 00:00-04:00",
            "2021-05-06 00:00-04:00",
            "2021-05-05 00:00-04:00",
            "2021-05-06 00:00-04:00",
        ]
    )
    quality_path = tmp_path / "quality.parquet"
    receipt = scan_formation_resolution_quality(
        pd.DataFrame({"instrument_id": ["id-aapl"], "symbol": ["AAPL"]}),
        daily,
        corpus_root=corpus,
        expected_sessions=session_dates,
        spy_instrument_id="benchmark-spy",
        output_path=quality_path,
        receipt_path=tmp_path / "quality.json",
        protocol_hash="a" * 64,
    )
    quality = pd.read_parquet(quality_path)

    assert receipt["quality_rows"] == 4
    assert {
        "daily_valid",
        "minute_exact_full_session",
        "token_valid_full_session",
        "tca_window_exact",
    }.issubset(quality.columns)
    stock_first = quality.loc[
        (quality["instrument_id"] == "id-aapl") & (quality["session_date"] == "2021-05-05")
    ].iloc[0]
    assert bool(stock_first["daily_valid"])
    assert not bool(stock_first["minute_exact_full_session"])
    assert bool(stock_first["token_valid_full_session"])
    assert bool(stock_first["tca_window_exact"])


def test_v2_formation_report_uses_named_quality_and_stops_before_target(
    tmp_path: Path,
) -> None:
    ids = [f"id-{index:03d}" for index in range(505)]
    v1 = pd.DataFrame(
        {
            "instrument_id": ids,
            "symbol": [f"S{index:03d}" for index in range(505)],
            "session_completeness": np.linspace(0.5, 1.0, 505),
        }
    )
    v2 = pd.DataFrame(
        {
            "instrument_id": ids,
            "formation_symbol": [f"S{index:03d}" for index in range(505)],
            "in_sp500_on_formation_date": True,
            "security_type": "ordinary_common_stock",
            "expected_daily_sessions": 2,
            "observed_valid_daily_sessions": 2,
            "daily_completeness": 1.0,
            "median_daily_price": 10.0,
            "median_daily_share_volume": 1_000.0,
            "median_daily_dollar_volume": np.arange(505, 0, -1, dtype=float),
            "first_valid_formation_date": "2021-05-05",
            "last_valid_formation_date": "2021-05-06",
            "identity_source_hash": "a" * 64,
            "formation_data_hash": "b" * 64,
            "exclusion_reasons": "[]",
        }
    )
    quality_rows = []
    for instrument_index, instrument_id in enumerate([*ids, "benchmark-spy"]):
        for session_index, session_date in enumerate(("2021-05-05", "2021-05-06")):
            token_valid = instrument_index < 100 or session_index == 0
            observed_minutes = max(2, 390 - instrument_index % 100)
            quality_rows.append(
                {
                    "instrument_id": instrument_id,
                    "symbol": "SPY" if instrument_id == "benchmark-spy" else instrument_id,
                    "session_date": session_date,
                    "daily_valid": True,
                    "minute_exact_full_session": True,
                    "token_valid_full_session": token_valid,
                    "tca_window_exact": True,
                    "early_close": False,
                    "provider_gap_count": 0,
                    "observed_minute_count": observed_minutes,
                    "valid_token_count": 26 if token_valid else 25,
                    "invalid_token_reason": "" if token_valid else "token_00_observed_bars_lt_2",
                }
            )
    paths = {
        "v1": tmp_path / "v1.parquet",
        "v2": tmp_path / "v2.parquet",
        "quality": tmp_path / "quality.parquet",
        "universe": tmp_path / "universe.json",
        "ticker": tmp_path / "ticker.parquet",
        "daily_receipt": tmp_path / "daily.json",
        "quality_receipt": tmp_path / "quality.json",
    }
    v1.to_parquet(paths["v1"], index=False)
    v2.to_parquet(paths["v2"], index=False)
    pd.DataFrame(quality_rows).to_parquet(paths["quality"], index=False)
    members = [
        {"rank": index + 1, "instrument_id": ids[index], "liquidity_group": 1}
        for index in range(100)
    ]
    write_json_atomic(
        paths["universe"],
        {"protocol_id": "sparse-jepa-v2", "members": members},
    )
    pd.DataFrame(
        {
            "instrument_id": [*[item["instrument_id"] for item in members], "benchmark-spy"],
            "symbol": [*[f"S{index:03d}" for index in range(100)], "SPY"],
            "start": "2021-01-04",
            "end": "2025-12-31",
        }
    ).to_parquet(paths["ticker"], index=False)
    write_json_atomic(
        paths["daily_receipt"],
        {"rows": 1_012, "symbols_observed": 506, "content_sha256": "c" * 64},
    )
    write_json_atomic(
        paths["quality_receipt"],
        {
            "observed_minute_rows": 1_012 * 390,
            "elapsed_seconds": 1.0,
            "minute_rows_per_second": 1_000.0,
            "token_aggregation_attempts_per_second": 100.0,
            "peak_rss_bytes": 1_000,
            "source_response_bytes": 2_000,
            "quality_parquet_bytes": paths["quality"].stat().st_size,
        },
    )
    report = tmp_path / "V2_FORMATION_QUALITY_REPORT.md"
    result = write_v2_formation_bundle(
        v1_candidates_path=paths["v1"],
        v2_candidates_path=paths["v2"],
        quality_path=paths["quality"],
        universe_path=paths["universe"],
        ticker_history_path=paths["ticker"],
        daily_receipt_path=paths["daily_receipt"],
        quality_receipt_path=paths["quality_receipt"],
        output_directory=tmp_path / "output",
        report_path=report,
        protocol_hash="d" * 64,
    )

    assert result["candidate_count"] == 505
    assert result["selected_token_band_counts"] == {"high": 100, "medium": 0, "low": 0}
    assert result["target_data_status"] == "NOT RUN"
    assert result["terminal_status"] == "AWAITING V2 FORMATION APPROVAL"
    assert "AWAITING V2 FORMATION APPROVAL" in report.read_text(encoding="utf-8")
