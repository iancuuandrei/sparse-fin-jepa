from dataclasses import FrozenInstanceError
from datetime import date

import numpy as np
import pandas as pd
import pytest

from execsim.forecasting import HistoricalProfileForecaster
from execsim.forecasting.historical import HistoricalForecastUnavailable
from execsim.ml.paper.tca import TCAExecutionFailure, run_historical_tca


def _case_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, HistoricalProfileForecaster]:
    timestamps = pd.date_range("2024-04-12 09:30", periods=390, freq="min", tz="America/New_York")
    bars = pd.DataFrame(
        {
            "instrument_id": "A",
            "symbol": "AAA",
            "timestamp": timestamps,
            "open": 100.0,
            "high": 100.0,
            "low": 100.0,
            "close": 100.0,
            "vwap": 100.0,
            "volume": 1_000.0,
            "trade_count": 10,
        }
    )
    history = bars.copy()
    history["timestamp"] -= pd.Timedelta(days=1)
    return (
        bars,
        pd.DataFrame({"rank": [1], "instrument_id": ["A"]}),
        pd.DataFrame(
            {"instrument_id": ["A"], "session_date": [date(2024, 4, 12)], "adv20": [100_000.0]}
        ),
        HistoricalProfileForecaster(history, estimator="ewma"),
    )


@pytest.mark.parametrize("exception_type", [RuntimeError, ValueError])
def test_solver_runtime_failure_preserves_cause_and_tca_context(monkeypatch, exception_type):
    bars, universe, adv20, provider = _case_inputs()
    original = exception_type("forced integer projection failure")

    def fail(*args, **kwargs):
        raise original

    monkeypatch.setattr("execsim.optimization.OptimalExecutionWorkspace.solve", fail)

    with pytest.raises(TCAExecutionFailure) as caught:
        run_historical_tca(
            bars,
            universe,
            adv20,
            {"raw_dense_jepa_seed_13": lambda *_: provider},
            liquidity_size=1,
            required_methods=("raw_dense_jepa_seed_13",),
            fold_id="fold-1",
        )

    failure = caught.value
    assert failure.__cause__ is original
    context = failure.case_context
    assert context is failure.context
    assert context.instrument_id == "A"
    assert context.symbol == "AAA"
    assert context.session_date == date(2024, 4, 12)
    assert context.method == "raw_dense_jepa"
    assert context.seed == 13
    assert context.encoded_method == "raw_dense_jepa_seed_13"
    assert context.order_fraction_adv20 == 0.03
    assert context.parent_quantity == 3_000
    assert context.fold_id == "fold-1"
    assert context.decision_timestamp == pd.Timestamp("2024-04-12 10:30", tz="America/New_York")
    assert context.remaining_inventory == 3_000
    assert context.horizon == 300
    import pickle

    restored = pickle.loads(pickle.dumps(failure))
    assert restored.context == context
    assert str(restored) == str(failure)
    with pytest.raises(FrozenInstanceError):
        context.seed = 29


def test_ewma_unavailability_remains_an_explicit_case_status():
    bars, universe, adv20, _ = _case_inputs()

    def absent(*args, **kwargs):
        raise HistoricalForecastUnavailable("no eligible window")

    result = run_historical_tca(
        bars,
        universe,
        adv20,
        {"ewma": absent},
        liquidity_size=1,
        required_methods=("ewma",),
    )

    assert len(result) == 1
    row = result.iloc[0]
    assert row.status == "EWMA_UNAVAILABLE"
    assert row.unavailable_reason == "no eligible window"
    assert np.isnan(row.total_modeled_execution_cost)
