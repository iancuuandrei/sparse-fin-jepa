from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from execsim.forecasting.models import VolumeForecast

ProfileEstimator = Literal["mean", "median", "ewma", "previous"]


class HistoricalForecastUnavailable(ValueError):
    """The unchanged historical estimator has no eligible causal observations."""


@dataclass(frozen=True, slots=True)
class _HistoricalMatrix:
    volumes: NDArray[np.float64]
    bucket_columns: dict[str, int]
    session_dates: tuple[date, ...]
    date_ordinals: NDArray[np.int64]
    finite: NDArray[np.bool_]


@dataclass(slots=True)
class HistoricalProfileForecaster:
    """Point-in-time volume curve estimated only from preceding complete windows."""

    historical_bars: pd.DataFrame
    estimator: ProfileEstimator = "mean"
    lookback_sessions: int | None = 20
    ewma_alpha: float = 0.25
    pooled: bool = False
    feature_schema_version: str = "volume-profile-v1"
    data_manifest_hash: str | None = None
    _history_cache: dict[str, _HistoricalMatrix] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        required = {"symbol", "timestamp", "volume"}
        missing = required.difference(self.historical_bars.columns)
        if missing:
            raise ValueError(f"Historical bars missing required columns: {sorted(missing)}")
        if self.estimator not in {"mean", "median", "ewma", "previous"}:
            raise ValueError(f"Unknown profile estimator: {self.estimator}")
        if self.lookback_sessions is not None and self.lookback_sessions <= 0:
            raise ValueError("lookback_sessions must be positive or None.")
        if not 0 < self.ewma_alpha <= 1:
            raise ValueError("ewma_alpha must be in (0, 1].")
        bars = self.historical_bars.copy()
        bars["timestamp"] = pd.to_datetime(bars["timestamp"])
        if bars["timestamp"].dt.tz is None:
            raise ValueError("Historical profile timestamps must be timezone-aware.")
        volumes = pd.to_numeric(bars["volume"], errors="coerce")
        if volumes.isna().any() or (volumes < 0).any():
            raise ValueError("Historical profile volumes must be finite and non-negative.")
        bars["volume"] = volumes.astype(float)
        bars["_session_key"] = (
            bars["symbol"].astype(str).str.upper() + "|" + bars["timestamp"].dt.date.astype(str)
        )
        bars["_session_date"] = bars["timestamp"].dt.date
        bars["_bucket_time"] = bars["timestamp"].dt.strftime("%H:%M")
        self.historical_bars = bars
        if self.data_manifest_hash is None:
            digest_input = bars.loc[:, ["symbol", "timestamp", "volume"]].to_csv(index=False)
            self.data_manifest_hash = hashlib.sha256(digest_input.encode()).hexdigest()

    @property
    def provider_id(self) -> str:
        scope = "pooled" if self.pooled else "symbol"
        return f"historical-{self.estimator}-{scope}-v1"

    def forecast(
        self,
        *,
        symbol: str,
        session_date: date,
        generated_at: pd.Timestamp,
        bucket_timestamps: Sequence[pd.Timestamp],
        observations: pd.DataFrame | None = None,
    ) -> VolumeForecast:
        del observations  # deterministic V1 profile does not adapt its scale intraday
        timestamps = tuple(pd.Timestamp(value) for value in bucket_timestamps)
        if not timestamps:
            raise ValueError("At least one future bucket is required.")
        expected_times = [timestamp.strftime("%H:%M") for timestamp in timestamps]
        history = self._history_matrix(symbol, session_date)
        end = int(np.searchsorted(history.date_ordinals, session_date.toordinal(), side="left"))
        if not end:
            raise HistoricalForecastUnavailable(
                f"No prior sessions are available for {symbol} before {session_date}."
            )
        try:
            column_indices = [history.bucket_columns[value] for value in expected_times]
        except KeyError:
            column_indices = []
        window = history.volumes[:end, column_indices] if column_indices else np.empty((end, 0))
        complete = (
            np.all(history.finite[:end, column_indices], axis=1)
            if window.shape[1]
            else np.zeros(len(window), dtype=bool)
        )
        selected_rows = np.flatnonzero(complete)
        if not len(selected_rows):
            raise HistoricalForecastUnavailable(
                "No preceding sessions contain the complete requested forecast window."
            )
        if self.lookback_sessions is not None:
            selected_rows = selected_rows[-self.lookback_sessions :]
        window = window[selected_rows]
        totals = window.sum(axis=1)
        positive = totals > 0
        window = window[positive]
        totals = totals[positive]
        selected_rows = selected_rows[positive]
        if not len(window):
            raise HistoricalForecastUnavailable(
                "Historical profile requires at least one positive-volume session."
            )
        shapes = window / totals[:, None]

        if self.estimator == "mean":
            raw_shape = shapes.mean(axis=0)
            expected_total = float(totals.mean())
        elif self.estimator == "median":
            raw_shape = np.median(shapes, axis=0)
            expected_total = float(np.median(totals))
        elif self.estimator == "previous":
            raw_shape = shapes[-1]
            expected_total = float(totals[-1])
        else:
            count = len(shapes)
            weights = (1.0 - self.ewma_alpha) ** np.arange(count - 1, -1, -1)
            weights /= weights.sum()
            raw_shape = np.average(shapes, axis=0, weights=weights)
            expected_total = float(np.average(totals, weights=weights))

        raw_shape = np.maximum(raw_shape, 0.0)
        shape = raw_shape / raw_shape.sum()
        expected = expected_total * shape
        cutoff = max(history.session_dates[index] for index in selected_rows)
        warnings: list[str] = []
        if len(shapes) < 5:
            warnings.append("fewer_than_five_complete_prior_sessions")
        return VolumeForecast(
            symbol=symbol.upper(),
            session_date=session_date,
            generated_at=pd.Timestamp(generated_at),
            first_forecast_bucket=timestamps[0],
            bucket_timestamps=timestamps,
            expected_volumes=tuple(float(value) for value in expected),
            normalized_shares=tuple(float(value) for value in shape),
            expected_remaining_volume=float(expected.sum()),
            forecaster_id=self.provider_id,
            feature_schema_version=self.feature_schema_version,
            training_data_cutoff=cutoff,
            data_manifest_hash=str(self.data_manifest_hash),
            warnings=tuple(warnings),
        )

    def _history_matrix(self, symbol: str, session_date: date) -> _HistoricalMatrix:
        """Build one immutable scope index; callers slice strictly before their cutoff."""
        cache_symbol = "*" if self.pooled else symbol.upper()
        key = cache_symbol
        cached = self._history_cache.get(key)
        if cached is not None:
            return cached

        bars = self.historical_bars
        prior = (
            bars
            if self.pooled
            else bars.loc[bars["symbol"].astype(str).str.upper() == cache_symbol]
        )
        if prior.empty:
            raise HistoricalForecastUnavailable(
                f"No prior sessions are available for {symbol} before {session_date}."
            )
        session_order = (
            prior.groupby("_session_key", sort=False)["timestamp"]
            .min()
            .sort_values(kind="stable")
            .index
        )
        matrix = prior.pivot_table(
            index="_session_key",
            columns="_bucket_time",
            values="volume",
            aggfunc="sum",
        ).reindex(index=session_order)
        session_dates = tuple(
            prior.groupby("_session_key", sort=False)["_session_date"]
            .first()
            .reindex(session_order)
            .tolist()
        )
        volumes = matrix.to_numpy(dtype=float)
        ordinals = np.asarray([value.toordinal() for value in session_dates], dtype=np.int64)
        finite = np.isfinite(volumes)
        for array in (volumes, ordinals, finite):
            array.setflags(write=False)
        cached = _HistoricalMatrix(
            volumes=volumes,
            bucket_columns={str(value): index for index, value in enumerate(matrix.columns)},
            session_dates=session_dates,
            date_ordinals=ordinals,
            finite=finite,
        )
        self._history_cache[key] = cached
        return cached
