from datetime import date

import numpy as np
import pandas as pd
import pytest

from execsim.forecasting import HistoricalProfileForecaster
from execsim.forecasting.historical import HistoricalForecastUnavailable
from execsim.ml.paper.evaluation_artifacts import publish_frames
from execsim.ml.paper.evaluation_workers import EWMAWork, ewma_ledger_identity, run_ewma_work
from execsim.ml.paper.forecast_ledger import EWMAForecastLedgerProvider
from execsim.ml.paper.statistics import construct_complete_case_differences
from execsim.ml.paper.tca import run_historical_tca


def _work(tmp_path, missing):
    scale = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "instrument_id": ["A"] * 2,
            "symbol": ["AAA"] * 2,
            "session_date": ["2024-04-12"] * 2,
            "as_of": [23, 25],
            "baseline_remaining_volume": [450.0, 150.0],
            "__evaluation_target": [450.0, 150.0],
        }
    )
    shape = scale.iloc[[0, 0, 0, 1]].reset_index(drop=True)
    shape["case_id"] = shape["sample_id"]
    shape["target_bucket"] = [23, 24, 25, 25]
    shape["__evaluation_target"] = [1 / 3, 1 / 3, 1 / 3, 1.0]
    publish_frames(
        tmp_path / "base",
        identity={"fixture": True},
        frames={
            "scale-base.parquet": scale,
            "shape-base.parquet": shape,
        },
    )
    stamps = pd.date_range("2024-04-11 09:30", periods=390, freq="min", tz="America/New_York")
    bars = pd.DataFrame({"symbol": "AAA", "timestamp": stamps, "volume": 10.0})
    bars = bars.loc[~bars.timestamp.dt.strftime("%H:%M").isin(missing)]
    bars.to_parquet(tmp_path / "market.parquet", index=False)
    return EWMAWork(
        tmp_path / "base", tmp_path / "market.parquet", tmp_path / "out", "A", {"fold_id": "fold-1"}
    )


@pytest.mark.parametrize("missing", [["15:20"], ["15:59"], ["15:20", "15:59"], ["15:29", "15:59"]])
def test_unavailability_is_per_exact_window_and_resume_is_verified(tmp_path, missing):
    work = _work(tmp_path, missing)
    run_ewma_work(work)
    absent = pd.read_parquet(work.output_directory / "unavailable.parquet")
    assert not absent.empty
    assert set(absent.status) == {"EWMA_UNAVAILABLE"}
    metrics = pd.read_parquet(work.output_directory / "metrics.parquet")
    assert len(metrics) == (1 if "15:59" not in missing else 0)
    assert np.isfinite(metrics.predicted_remaining_volume).all()
    replay = EWMAForecastLedgerProvider(
        work.output_directory,
        expected_identity=ewma_ledger_identity(work),
        symbol="AAA",
        session_date=date(2024, 4, 12),
    )
    for minute in (15, 21):
        start = pd.Timestamp(f"2024-04-12 15:{minute}", tz="America/New_York")
        request = dict(
            symbol="AAA",
            session_date=date(2024, 4, 12),
            generated_at=start,
            bucket_timestamps=pd.date_range(start, periods=30 - minute, freq="min"),
        )
        if "15:29" in missing or ("15:20" in missing and minute == 15):
            with pytest.raises(HistoricalForecastUnavailable):
                replay.forecast(**request)
        else:
            assert replay.forecast(**request).expected_remaining_volume > 0
    before = (work.output_directory / "manifest.json").stat().st_mtime_ns
    run_ewma_work(work)
    assert before == (work.output_directory / "manifest.json").stat().st_mtime_ns
    with (work.output_directory / "unavailable.parquet").open("ab") as handle:
        handle.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        run_ewma_work(work)


def test_unrelated_errors_are_not_converted_to_unavailability(tmp_path, monkeypatch):
    work = _work(tmp_path, [])

    def broken(*args, **kwargs):
        raise ValueError("unrelated corruption")

    monkeypatch.setattr(HistoricalProfileForecaster, "forecast", broken)
    with pytest.raises(ValueError, match="unrelated corruption"):
        run_ewma_work(work)
    assert not work.output_directory.exists()


def test_tca_keeps_other_methods_sides_and_records_unavailable_case():
    stamps = pd.date_range("2024-04-12 09:30", periods=390, freq="min", tz="America/New_York")
    bars = pd.DataFrame(
        {
            "instrument_id": "A",
            "symbol": "AAA",
            "timestamp": stamps,
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "vwap": 100.0,
            "volume": 1000.0,
            "trade_count": 10,
        }
    )
    history = bars.copy()
    history["timestamp"] -= pd.Timedelta(days=1)
    valid = HistoricalProfileForecaster(history, estimator="ewma")

    def absent(*args):
        raise HistoricalForecastUnavailable("no eligible window")

    inputs = (
        bars,
        pd.DataFrame({"rank": [1], "instrument_id": ["A"]}),
        pd.DataFrame(
            {"instrument_id": ["A"], "session_date": [date(2024, 4, 12)], "adv20": [390000.0]}
        ),
    )
    expected = run_historical_tca(
        *inputs,
        {"ewma": lambda *args: valid, "raw": lambda *args: valid},
        liquidity_size=1,
        required_methods=("ewma", "raw"),
    )
    actual = run_historical_tca(
        *inputs,
        {"ewma": absent, "raw": lambda *args: valid},
        liquidity_size=1,
        required_methods=("ewma", "raw"),
    )
    pd.testing.assert_frame_equal(
        expected.loc[expected.method == "raw"], actual.loc[actual.method == "raw"]
    )
    ewma = actual.loc[actual.method == "ewma"].iloc[0]
    assert ewma.status == "EWMA_UNAVAILABLE"
    assert pd.isna(ewma.normalized_allocation_regret)
    pair = construct_complete_case_differences(
        actual,
        baseline="ewma",
        candidate="raw",
        value_column="normalized_allocation_regret",
        identity_columns=("date", "instrument_id", "side"),
    )
    assert pair.matched_rows == 0
    assert pair.dropped_baseline_rows == pair.dropped_candidate_rows == 1


def test_complete_cases_reject_unmarked_nonfinite_and_duplicates():
    rows = pd.DataFrame(
        {
            "method": ["ewma", "raw", "dense", "sparse"],
            "id": [1] * 4,
            "value": [np.nan, 1.0, 2.0, 3.0],
            "status": ["EWMA_UNAVAILABLE", "AVAILABLE", "AVAILABLE", "AVAILABLE"],
        }
    )
    opts = dict(
        baseline="dense", candidate="sparse", value_column="value", identity_columns=("id",)
    )
    assert construct_complete_case_differences(rows, **opts).matched_rows == 1
    rows.loc[2, "value"] = np.nan
    with pytest.raises(ValueError, match="finite"):
        construct_complete_case_differences(rows, **opts)
    with pytest.raises(ValueError, match="duplicated"):
        construct_complete_case_differences(pd.concat([rows, rows]), **opts)
