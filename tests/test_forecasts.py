from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from execsim.forecasting import HistoricalProfileForecaster, VolumeForecast
from execsim.forecasting.historical import _HistoricalMatrix


def _history() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for session, volumes in (
        ("2026-03-12", [100, 200, 300]),
        ("2026-03-13", [200, 200, 400]),
        ("2026-03-16", [9_999, 1, 1]),
    ):
        for timestamp, volume in zip(
            pd.date_range(f"{session} 09:30", periods=3, freq="min", tz="America/New_York"),
            volumes,
            strict=True,
        ):
            rows.append({"symbol": "AAPL", "timestamp": timestamp, "volume": volume})
    return pd.DataFrame(rows)


def test_historical_profile_uses_only_sessions_preceding_target() -> None:
    target_buckets = tuple(
        pd.date_range("2026-03-16 09:30", periods=3, freq="min", tz="America/New_York")
    )
    forecaster = HistoricalProfileForecaster(_history(), estimator="mean")
    forecast = forecaster.forecast(
        symbol="AAPL",
        session_date=date(2026, 3, 16),
        generated_at=target_buckets[0],
        bucket_timestamps=target_buckets,
    )

    assert forecast.training_data_cutoff == date(2026, 3, 13)
    assert forecast.normalized_shares == pytest.approx(
        ((1 / 6 + 1 / 4) / 2, (2 / 6 + 1 / 4) / 2, (3 / 6 + 2 / 4) / 2)
    )
    assert sum(forecast.normalized_shares) == pytest.approx(1.0)


def test_forecast_rejects_future_generated_buckets_and_target_day_cutoff() -> None:
    buckets = tuple(pd.date_range("2026-03-16 09:30", periods=2, freq="min", tz="America/New_York"))
    with pytest.raises(ValueError, match="cannot precede"):
        VolumeForecast(
            symbol="AAPL",
            session_date=date(2026, 3, 16),
            generated_at=buckets[1],
            first_forecast_bucket=buckets[0],
            bucket_timestamps=buckets,
            expected_volumes=(1.0, 1.0),
            normalized_shares=(0.5, 0.5),
            expected_remaining_volume=2.0,
            forecaster_id="bad",
            feature_schema_version="v1",
            training_data_cutoff=date(2026, 3, 16),
            data_manifest_hash="x",
        )


def test_historical_profile_reuses_only_same_point_in_time_history() -> None:
    forecaster = HistoricalProfileForecaster(_history(), estimator="mean")
    march_16 = tuple(
        pd.date_range("2026-03-16 09:30", periods=3, freq="min", tz="America/New_York")
    )
    first = forecaster.forecast(
        symbol="AAPL",
        session_date=date(2026, 3, 16),
        generated_at=march_16[0],
        bucket_timestamps=march_16,
    )
    repeated = forecaster.forecast(
        symbol="AAPL",
        session_date=date(2026, 3, 16),
        generated_at=march_16[1],
        bucket_timestamps=march_16[1:],
    )
    march_17 = tuple(
        pd.date_range("2026-03-17 09:30", periods=3, freq="min", tz="America/New_York")
    )
    next_day = forecaster.forecast(
        symbol="AAPL",
        session_date=date(2026, 3, 17),
        generated_at=march_17[0],
        bucket_timestamps=march_17,
    )

    assert len(forecaster._history_cache) == 1
    assert not forecaster._history_cache["AAPL"].volumes.flags.writeable
    assert first.training_data_cutoff == date(2026, 3, 13)
    assert repeated.training_data_cutoff == date(2026, 3, 13)
    assert next_day.training_data_cutoff == date(2026, 3, 16)
    assert next_day.expected_volumes != first.expected_volumes


class _PrefixReference(HistoricalProfileForecaster):
    """Original per-date pandas construction retained only as a semantic oracle."""

    def _history_matrix(self, symbol: str, session_date: date) -> _HistoricalMatrix:
        bars = self.historical_bars
        mask = bars["_session_date"] < session_date
        if not self.pooled:
            mask &= bars["symbol"].astype(str).str.upper() == symbol.upper()
        prior = bars.loc[mask]
        if prior.empty:
            raise ValueError(f"No prior sessions are available for {symbol} before {session_date}.")
        order = (
            prior.groupby("_session_key", sort=False)["timestamp"]
            .min()
            .sort_values(kind="stable")
            .index
        )
        matrix = prior.pivot_table(
            index="_session_key", columns="_bucket_time", values="volume", aggfunc="sum"
        ).reindex(index=order)
        dates = tuple(
            prior.groupby("_session_key", sort=False)["_session_date"]
            .first()
            .reindex(order)
            .tolist()
        )
        volumes = matrix.to_numpy(dtype=float)
        return _HistoricalMatrix(
            volumes,
            {str(value): i for i, value in enumerate(matrix.columns)},
            dates,
            np.asarray([value.toordinal() for value in dates], dtype=np.int64),
            np.isfinite(volumes),
        )


@pytest.mark.parametrize("estimator", ["mean", "median", "previous", "ewma"])
@pytest.mark.parametrize("pooled", [False, True])
@pytest.mark.parametrize("lookback", [1, 20, None])
def test_indexed_history_matches_prefix_semantics(estimator, pooled, lookback) -> None:
    rows = []
    dates = pd.bdate_range("2024-01-02", periods=35)
    for index, day in enumerate(dates):
        for symbol in ("BBB", "AAA"):
            for bucket, stamp in enumerate(
                pd.date_range(f"{day.date()} 09:30", periods=4, freq="min", tz="America/New_York")
            ):
                if index % 7 == 0 and bucket == 1:
                    continue
                volume = 0 if index % 9 == 0 else (index + 1) * (bucket + 1)
                rows.append({"symbol": symbol, "timestamp": stamp, "volume": volume})
    bars = pd.DataFrame(rows).sample(frac=1, random_state=13)
    options = dict(estimator=estimator, pooled=pooled, lookback_sessions=lookback)
    reference = _PrefixReference(bars, **options)
    indexed = HistoricalProfileForecaster(bars, **options)
    for day in pd.date_range("2024-01-01", "2024-02-22", freq="3D"):
        for symbol in ("AAA", "BBB", "UNKNOWN"):
            for start, length in ((0, 4), (2, 2), (4, 1)):
                timestamps = pd.date_range(
                    f"{day.date()} 09:30", periods=5, freq="min", tz="America/New_York"
                )[start : start + length]
                request = dict(
                    symbol=symbol,
                    session_date=day.date(),
                    generated_at=timestamps[0],
                    bucket_timestamps=timestamps,
                )
                try:
                    expected = reference.forecast(**request)
                except ValueError as error:
                    with pytest.raises(ValueError) as actual:
                        indexed.forecast(**request)
                    assert str(actual.value) == str(error)
                else:
                    assert indexed.forecast(**request) == expected
    assert len(indexed._history_cache) == (1 if pooled else 2)
