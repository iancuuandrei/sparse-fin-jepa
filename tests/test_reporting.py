from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from execsim.reporting import aggregate_results, paired_strategy_differences
from execsim.reporting.statistics import bootstrap_mean_interval


@pytest.mark.parametrize("values", [[], [np.nan, np.inf, -np.inf]])
def test_empty_finite_bootstrap_population_returns_nan_bounds(values):
    assert all(np.isnan(value) for value in bootstrap_mean_interval(values))


def test_nonempty_bootstrap_population_reaches_resampling():
    assert bootstrap_mean_interval([7.0, 7.0], samples=20) == (7.0, 7.0)


def _results() -> pd.DataFrame:
    rows = []
    for sample, twap, vwap in (("a", 2.0, 1.0), ("b", 4.0, 5.0), ("c", 6.0, 3.0)):
        for strategy, cost in (("twap", twap), ("vwap", vwap)):
            rows.append(
                {
                    "sample": sample,
                    "strategy": strategy,
                    "implementation_shortfall_bps": cost,
                    "completion_rate": 1.0,
                    "total_modeled_execution_cost": cost * 10,
                }
            )
    return pd.DataFrame(rows)


def test_aggregate_statistics_and_bootstrap_are_reproducible() -> None:
    first = aggregate_results(_results(), seed=5)
    second = aggregate_results(_results(), seed=5)

    pd.testing.assert_frame_equal(first, second)
    assert set(first["strategy"]) == {"twap", "vwap"}
    assert (first["run_count"] == 3).all()


def test_paired_comparison_uses_identical_samples_and_reports_win_rate() -> None:
    paired = paired_strategy_differences(
        _results(), baseline="twap", pair_columns=("sample",), seed=5
    )

    assert paired.loc[0, "paired_count"] == 3
    assert paired.loc[0, "mean_difference_bps"] == pytest.approx(-1.0)
    assert paired.loc[0, "win_rate"] == pytest.approx(2 / 3)
