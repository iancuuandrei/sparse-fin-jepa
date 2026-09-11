from datetime import date
from types import SimpleNamespace

import pandas as pd
import pytest

from execsim.data.paper.resolution_quality import assess_session_resolution_quality
from execsim.ml.paper.tca import run_historical_tca
from execsim.ml.paper.tca_inputs import filter_tca_window_exact, tca_eligible_instrument_ids
from execsim.ml.paper.tca_workers import preflight_tca_ledgers


def _session(
    instrument: str = "A", *, periods: int = 390, session_start: str = "2024-11-27 09:30"
) -> pd.DataFrame:
    timestamps = pd.date_range(session_start, periods=periods, freq="min", tz="America/New_York")
    return pd.DataFrame(
        {
            "instrument_id": instrument,
            "symbol": instrument,
            "timestamp": timestamps,
            "open": 100.0,
            "high": 100.1,
            "low": 99.9,
            "close": 100.0,
            "volume": 1000.0,
            "trade_count": 10,
            "vwap": 100.0,
        }
    )


def _run_with_fake_replay(
    monkeypatch,
    bars: pd.DataFrame,
    calls: list[str],
    provider_calls: list[tuple[str, date]] | None = None,
) -> pd.DataFrame:
    universe = pd.DataFrame({"rank": [1, 2], "instrument_id": ["A", "B"]})
    adv = pd.DataFrame(
        {
            "instrument_id": ["A", "B"],
            "session_date": [date(2024, 11, 27)] * 2,
            "adv20": [100_000.0, 100_000.0],
        }
    )

    def fake_simulate_policy(**kwargs):
        calls.append(kwargs["parent_order"].symbol)
        return SimpleNamespace(
            summary=SimpleNamespace(
                total_modeled_execution_cost=1.0,
                modeled_temporary_impact_cost=0.5,
                implementation_shortfall_bps=2.0,
                completion_rate=1.0,
            )
        )

    monkeypatch.setattr("execsim.simulator.simulate_policy", fake_simulate_policy)

    def provider(instrument: str, day: date) -> object:
        if provider_calls is not None:
            provider_calls.append((instrument, day))
        return object()

    return run_historical_tca(
        bars,
        universe,
        adv,
        {"ewma": provider},
        liquidity_size=2,
        required_methods=("ewma",),
    )


def test_early_close_is_not_a_tca_case_and_does_not_lookup_provider(monkeypatch):
    bars = _session(periods=210, session_start="2024-11-29 09:30")
    quality = assess_session_resolution_quality(bars)
    assert quality.early_close
    assert not quality.tca_window_exact
    assert not bool(tca_eligible_instrument_ids(bars, {"A"}))
    calls: list[str] = []
    provider_calls: list[tuple[str, date]] = []
    with pytest.raises(ValueError, match="no matched cases"):
        _run_with_fake_replay(monkeypatch, bars, calls, provider_calls)
    assert calls == []
    assert provider_calls == []


def test_full_session_with_exact_tca_grid_is_scheduled(monkeypatch):
    bars = _session()
    filtered = filter_tca_window_exact(bars, {"A"})
    assert len(filtered) == 390
    calls: list[str] = []
    result = _run_with_fake_replay(monkeypatch, bars, calls)
    assert calls == ["A"]
    assert set(result["instrument_id"]) == {"A"}


@pytest.mark.parametrize(
    "corrupt",
    [
        "missing_consumed_plus_off_hours",
        "wrong_timezone",
        "two_session_dates",
        "duplicated_minute",
        "reordered_minutes",
    ],
)
def test_corrupted_minute_grid_is_not_tca_eligible(corrupt: str) -> None:
    bars = _session()
    if corrupt == "missing_consumed_plus_off_hours":
        bars.loc[30, "timestamp"] = pd.Timestamp("2024-11-27 16:00", tz="America/New_York")
    elif corrupt == "wrong_timezone":
        bars["timestamp"] = bars["timestamp"].dt.tz_convert("UTC")
    elif corrupt == "two_session_dates":
        bars.loc[100, "timestamp"] = pd.Timestamp("2024-11-26 11:10", tz="America/New_York")
    elif corrupt == "duplicated_minute":
        bars.loc[31, "timestamp"] = bars.loc[30, "timestamp"]
    elif corrupt == "reordered_minutes":
        bars = bars.iloc[::-1].reset_index(drop=True)
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(corrupt)

    assert filter_tca_window_exact(bars, {"A"}).empty
    assert not tca_eligible_instrument_ids(bars, {"A"})


def test_provider_gap_excludes_only_that_instrument_and_balances_survivors(monkeypatch):
    valid = _session("A")
    invalid = _session("B").drop(index=[61]).reset_index(drop=True)
    bars = pd.concat([valid, invalid], ignore_index=True)
    assert tca_eligible_instrument_ids(bars, {"A", "B"}) == ("A",)
    calls: list[str] = []
    result = _run_with_fake_replay(monkeypatch, bars, calls)
    assert calls == ["A"]
    assert result["instrument_id"].tolist() == ["A"]
    assert result["side"].tolist() == ["buy"]


def test_invalid_timezone_does_not_remove_valid_same_date_instrument() -> None:
    valid = _session("A")
    invalid = _session("B")
    invalid["timestamp"] = invalid["timestamp"].dt.tz_convert("UTC")
    bars = pd.concat([valid, invalid], ignore_index=True)

    assert tca_eligible_instrument_ids(bars, {"A", "B"}) == ("A",)


def test_all_invalid_tca_population_has_no_surviving_cases(monkeypatch):
    bars = pd.concat(
        [_session("A").drop(index=[61]), _session("B").drop(index=[62])], ignore_index=True
    )
    assert filter_tca_window_exact(bars, {"A", "B"}).empty
    calls: list[str] = []
    with pytest.raises(ValueError, match="no matched cases"):
        _run_with_fake_replay(monkeypatch, bars, calls)
    assert calls == []


def test_tca_ledger_preflight_fails_before_workers_for_missing_eligible_ledger(tmp_path):
    with pytest.raises(FileNotFoundError):
        preflight_tca_ledgers(
            ledger_records=(("raw", None, tmp_path / "missing", {}),),
            ewma_records={},
            eligible_cases={date(2024, 11, 27): ("A",)},
            training_cutoff=date(2024, 6, 28),
            tca_config={"window": ["10:30", "15:30"]},
        )
