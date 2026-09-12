from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from execsim.data.paper.resolution_quality import assess_session_resolution_quality
from execsim.ml.paper.evaluation_artifacts import publish_frames
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
    adv: pd.DataFrame | None = None,
) -> pd.DataFrame:
    universe = pd.DataFrame({"rank": [1, 2], "instrument_id": ["A", "B"]})
    if adv is None:
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


@pytest.mark.parametrize(
    ("parameter", "frozen_value"),
    [
        ("planned_participation", 0.10),
        ("hard_participation", 0.10),
        ("risk_aversion", 0.0),
        ("tracking_penalty", 0.0),
        ("half_spread_arrival_fraction", 5e-5),
        ("temporary_impact_arrival_fraction", 1e-3),
    ],
)
def test_tca_rejects_even_one_ulp_change_to_frozen_contract(parameter, frozen_value):
    def unexpected_provider(*args):
        pytest.fail("Changed protocol must fail before provider access")

    with pytest.raises(ValueError, match="contradict the locked experiment"):
        run_historical_tca(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            {"ewma": unexpected_provider},
            required_methods=("ewma",),
            **{parameter: float(np.nextafter(frozen_value, np.inf))},
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


def test_missing_adv_for_one_eligible_case_fails_before_any_replay(monkeypatch):
    bars = pd.concat([_session("A"), _session("B")], ignore_index=True)
    calls: list[str] = []
    provider_calls: list[tuple[str, date]] = []
    adv = pd.DataFrame(
        {
            "instrument_id": ["A"],
            "session_date": [date(2024, 11, 27)],
            "adv20": [100_000.0],
        }
    )
    with pytest.raises(ValueError, match="ADV20 required"):
        _run_with_fake_replay(monkeypatch, bars, calls, provider_calls, adv)
    assert calls == []
    assert provider_calls == []


@pytest.mark.parametrize("value", [np.nan, np.inf, 0.0, -1.0])
def test_nonpositive_or_nonfinite_adv_for_eligible_case_fails_closed(monkeypatch, value):
    bars = pd.concat([_session("A"), _session("B")], ignore_index=True)
    calls: list[str] = []
    adv = pd.DataFrame(
        {
            "instrument_id": ["A", "B"],
            "session_date": [date(2024, 11, 27)] * 2,
            "adv20": [100_000.0, value],
        }
    )
    with pytest.raises(ValueError, match="finite and positive"):
        _run_with_fake_replay(monkeypatch, bars, calls, adv=adv)
    assert calls == []


def test_duplicate_adv_for_eligible_case_fails_closed(monkeypatch):
    bars = pd.concat([_session("A"), _session("B")], ignore_index=True)
    calls: list[str] = []
    adv = pd.DataFrame(
        {
            "instrument_id": ["A", "B", "B"],
            "session_date": [date(2024, 11, 27)] * 3,
            "adv20": [100_000.0, 100_000.0, 100_000.0],
        }
    )
    with pytest.raises(ValueError, match="exactly one row"):
        _run_with_fake_replay(monkeypatch, bars, calls, adv=adv)
    assert calls == []


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


def _publish_multi_session_preflight_ledgers(tmp_path):
    """Create two-date ledgers, including rows outside the date under test."""
    dates = (date(2024, 11, 27), date(2024, 11, 28))
    origins = tuple(range(4, 24))
    cutoff = date(2024, 6, 28)

    learned_rows = []
    ewma_rows = []
    for day in dates:
        for origin in origins:
            sample_id = f"{day.isoformat()}-{origin}"
            learned_rows.append(
                {
                    "sample_id": sample_id,
                    "fold_id": "fold-1",
                    "instrument_id": "A",
                    "symbol": "AAA",
                    "session_date": day.isoformat(),
                    "as_of": origin,
                    "training_cutoff": cutoff.isoformat(),
                }
            )
            ewma_rows.append(
                {
                    "sample_id": sample_id,
                    "instrument_id": "A",
                    "symbol": "AAA",
                    "session_date": day.isoformat(),
                    "as_of": origin,
                }
            )
    learned_scale = pd.DataFrame(learned_rows)
    learned_scale["baseline_remaining_volume"] = 1_000.0
    learned_scale["predicted_remaining_volume"] = 1_000.0
    learned_shape = learned_scale.loc[
        learned_scale.index.repeat(26 - learned_scale["as_of"])
    ].reset_index(drop=True)
    learned_shape["case_id"] = learned_shape["sample_id"]
    learned_shape["target_bucket"] = np.concatenate(
        [np.arange(origin, 26) for origin in learned_scale["as_of"]]
    )
    learned_shape["conditional_share"] = 1.0 / (26 - learned_shape["as_of"])
    learned = tmp_path / "learned"
    learned_identity = {
        "fold_id": "fold-1",
        "method": "raw",
        "seed": None,
        "parameter_freeze_sha256": "freeze",
        "model_manifest_sha256": "model",
        "source_commit": "source",
        "source_tree": "tree",
        "paper_config_hash": "config",
        "base_manifest_sha256": "base",
        "embedding_sha256": None,
    }
    publish_frames(
        learned,
        identity=learned_identity,
        frames={
            "scale.parquet": learned_scale,
            "shape.parquet": learned_shape,
            "metrics.parquet": pd.DataFrame({"metric": pd.Series(dtype="string")}),
        },
    )

    opened = pd.Timestamp("2024-11-27 09:30", tz="America/New_York")
    minute_rows = []
    for origin in origins:
        sample_id = f"2024-11-27-{origin}"
        for offset in range(15):
            minute_rows.append(
                {
                    "sample_id": sample_id,
                    "generated_at": opened + pd.Timedelta(minutes=15 * origin + offset),
                    "end_token": 24,
                }
            )
    # These two rows belong to the other session and must be ignored after the
    # session/end-token filter; without it they contaminate sample validation.
    minute_rows.append(
        {
            "sample_id": "2024-11-28-4",
            "generated_at": opened + pd.Timedelta(days=1, minutes=60),
            "end_token": 24,
        }
    )
    unavailable = pd.DataFrame(
        {
            "sample_id": ["2024-11-28-4"],
            "generated_at": [opened + pd.Timedelta(days=1, minutes=61)],
            "end_token": [24],
            "status": ["EWMA_UNAVAILABLE"],
            "session_date": ["2024-11-28"],
            "reason": ["other session"],
        }
    )
    ewma = tmp_path / "ewma"
    ewma_identity = {"fold_id": "fold-1", "instrument_id": "A", "fixture": "ewma"}
    publish_frames(
        ewma,
        identity=ewma_identity,
        frames={
            "scale.parquet": pd.DataFrame(ewma_rows),
            "shape.parquet": pd.DataFrame(
                {"case_id": pd.Series(dtype="string"), "target_bucket": pd.Series(dtype="int64")}
            ),
            "metrics.parquet": pd.DataFrame({"metric": pd.Series(dtype="string")}),
            "minute-forecasts.parquet": pd.DataFrame(minute_rows),
            "unavailable.parquet": unavailable,
        },
    )
    return learned, learned_identity, ewma, ewma_identity, minute_rows


def test_tca_preflight_indexed_slices_equal_original_filters(tmp_path, monkeypatch):
    from execsim.ml.paper import tca_workers

    learned, identity, ewma, ewma_identity, _ = _publish_multi_session_preflight_ledgers(tmp_path)
    full_scale = pd.read_parquet(learned / "scale.parquet")
    full_shape = pd.read_parquet(learned / "shape.parquet")
    validate = tca_workers._validate_learned_case
    compared = []

    def compare(directory, **kwargs):
        expected_scale = full_scale.loc[
            pd.to_datetime(full_scale["session_date"]).dt.date == kwargs["session_date"]
        ]
        expected_shape = full_shape.loc[
            full_shape["case_id"].astype(str).isin(expected_scale["sample_id"].astype(str))
        ]
        pd.testing.assert_frame_equal(kwargs["scale_frame"], expected_scale)
        pd.testing.assert_frame_equal(kwargs["shape_frame"], expected_shape)
        validate(directory, **kwargs)
        validate(directory, **{**kwargs, "scale_frame": full_scale, "shape_frame": full_shape})
        compared.append(kwargs["session_date"])

    monkeypatch.setattr(tca_workers, "_validate_learned_case", compare)
    preflight_tca_ledgers(
        ledger_records=(("raw", None, learned, identity),),
        ewma_records={"A": (ewma, ewma_identity)},
        eligible_cases={date(2024, 11, 27): ("A",)},
        training_cutoff=date(2024, 6, 28),
        tca_config={"window": ["10:30", "15:30"]},
    )
    assert compared == [date(2024, 11, 27)]


def test_tca_ledger_preflight_scopes_ewma_to_current_multi_session(tmp_path):
    (
        learned,
        learned_identity,
        ewma,
        ewma_identity,
        minute_rows,
    ) = _publish_multi_session_preflight_ledgers(tmp_path)
    records = (("raw", None, learned, learned_identity),)
    kwargs = {
        "ledger_records": records,
        "eligible_cases": {date(2024, 11, 27): ("A",)},
        "training_cutoff": date(2024, 6, 28),
        "tca_config": {"window": ["10:30", "15:30"]},
    }

    # EWMA scale deliberately has no fold_id column; the artifact identity binds
    # the fold. Rows for 2024-11-28 are also present in both minute ledgers.
    assert "fold_id" not in pd.read_parquet(ewma / "scale.parquet").columns
    preflight_tca_ledgers(ewma_records={"A": (ewma, ewma_identity)}, **kwargs)

    ewma_frames = {
        name: pd.read_parquet(ewma / name)
        for name in ("scale.parquet", "shape.parquet", "metrics.parquet", "unavailable.parquet")
    }
    missing_current = tmp_path / "ewma-missing-current"
    publish_frames(
        missing_current,
        identity=ewma_identity,
        frames={
            **ewma_frames,
            "minute-forecasts.parquet": pd.DataFrame(minute_rows[1:]),
        },
    )
    with pytest.raises(ValueError, match="missing or conflicting"):
        preflight_tca_ledgers(
            ewma_records={"A": (missing_current, ewma_identity)},
            **kwargs,
        )

    wrong_current = tmp_path / "ewma-wrong-current"
    wrong_rows = list(minute_rows)
    wrong_rows[0] = {**wrong_rows[0], "sample_id": "2024-11-27-5"}
    publish_frames(
        wrong_current,
        identity=ewma_identity,
        frames={
            **ewma_frames,
            "minute-forecasts.parquet": pd.DataFrame(wrong_rows),
        },
    )
    with pytest.raises(ValueError, match=r"as-of/sample identity|sample identity"):
        preflight_tca_ledgers(
            ewma_records={"A": (wrong_current, ewma_identity)},
            **kwargs,
        )
