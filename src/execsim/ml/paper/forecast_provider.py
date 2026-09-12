"""Causal 15-minute LightGBM provider for the existing VolumeForecast contract."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from execsim.forecasting.models import VolumeForecast
from execsim.ml.models.lightgbm_adapter import LightGBMVolumeModel
from execsim.ml.paper.tca import expand_volume_forecast, mean_seed_forecast

FeatureResolver = Callable[
    [str, date, pd.Timestamp, pd.DataFrame | None], tuple[pd.DataFrame, pd.DataFrame]
]


@dataclass(frozen=True, slots=True)
class _MinuteGrid:
    """Immutable minute grid whose timestamps came from one explicit date range."""

    start: pd.Timestamp
    frequency: pd.Timedelta
    periods: int
    timestamps: tuple[pd.Timestamp, ...]

    @classmethod
    def from_range(cls, start: pd.Timestamp, end: pd.Timestamp) -> _MinuteGrid:
        """Create a trusted one-minute grid, inclusive of both endpoints."""
        frequency = pd.Timedelta(minutes=1)
        timestamps = tuple(pd.date_range(start, end, freq=frequency))
        if not timestamps:
            raise ValueError("A forecast minute grid must not be empty.")
        return cls(
            start=timestamps[0],
            frequency=frequency,
            periods=len(timestamps),
            timestamps=timestamps,
        )

    def contiguous_offset(self, requested: tuple[pd.Timestamp, ...]) -> int | None:
        """Return the exact start offset only when request matches one grid slice."""
        if not requested:
            return None
        try:
            delta = requested[0] - self.start
            offset, remainder = divmod(delta.value, self.frequency.value)
        except (OverflowError, TypeError, ValueError):
            return None
        if remainder or offset < 0:
            return None
        stop = offset + len(requested)
        if stop > self.periods or self.timestamps[offset:stop] != requested:
            return None
        return int(offset)


class PaperLightGBMForecastProvider:
    """Run the frozen model only on token boundaries and truncate it between updates."""

    def __init__(
        self,
        model: LightGBMVolumeModel,
        *,
        feature_resolver: FeatureResolver,
        within_token_profile: np.ndarray,
        training_cutoff: date,
        manifest_hash: str,
        method_id: str,
    ) -> None:
        profile = np.asarray(within_token_profile, dtype=float)
        if profile.shape != (15,) or (profile < 0).any() or not np.isclose(profile.sum(), 1):
            raise ValueError("TRAIN-only within-token profile must be a normalized 15-vector.")
        self.model = model
        self.feature_resolver = feature_resolver
        self.within_token_profile = profile
        self.training_cutoff = training_cutoff
        self.manifest_hash = manifest_hash
        self._provider_id = method_id
        self._latest: dict[tuple[str, date], VolumeForecast] = {}
        self._latest_grids: dict[tuple[str, date], _MinuteGrid] = {}

    @property
    def provider_id(self) -> str:
        return self._provider_id

    def forecast(
        self,
        *,
        symbol: str,
        session_date: date,
        generated_at: pd.Timestamp,
        bucket_timestamps: Sequence[pd.Timestamp],
        observations: pd.DataFrame | None = None,
    ) -> VolumeForecast:
        """Return a fresh boundary forecast or the causally truncated cached forecast."""
        requested = tuple(pd.Timestamp(value) for value in bucket_timestamps)
        if not requested or requested[0] < generated_at:
            raise ValueError("Forecast request must contain non-past minute buckets.")
        key = (symbol, session_date)
        if _is_update_boundary(generated_at):
            scale_frame, shape_frame = self.feature_resolver(
                symbol, session_date, generated_at, observations
            )
            total, long_shape = self.model.predict_frames(
                scale_frame, shape_frame, group_columns=("case_id",)
            )
            valid = long_shape.loc[long_shape["conditional_share"] > 0].sort_values(
                "target_bucket", kind="stable"
            )
            token_shape = valid["conditional_share"].to_numpy(dtype=float)
            local = generated_at.tz_convert("America/New_York")
            minute_grid = _MinuteGrid.from_range(
                local,
                pd.Timestamp.combine(session_date, pd.Timestamp("15:59").time()).tz_localize(
                    "America/New_York"
                ),
            )
            fresh = expand_volume_forecast(
                symbol=symbol,
                session_date=session_date,
                generated_at=generated_at,
                minute_timestamps=minute_grid.timestamps,
                expected_remaining_volume=float(total[0]),
                conditional_token_shape=token_shape,
                within_token_profile=self.within_token_profile,
                training_cutoff=self.training_cutoff,
                manifest_hash=self.manifest_hash,
                forecaster_id=self.provider_id,
            )
            self._latest[key] = fresh
            self._latest_grids[key] = minute_grid
            return _truncate_forecast(fresh, requested, generated_at, minute_grid=minute_grid)
        cached = self._latest.get(key)
        if cached is None:
            raise ValueError("A between-boundary request has no prior causal model forecast.")
        return _truncate_forecast(
            cached, requested, generated_at, minute_grid=self._latest_grids[key]
        )


def _truncate_forecast(
    cached: VolumeForecast,
    requested: tuple[pd.Timestamp, ...],
    generated_at: pd.Timestamp,
    *,
    minute_grid: _MinuteGrid | None = None,
) -> VolumeForecast:
    """Return the requested forecast horizon, preserving order and normalization math."""
    volumes: np.ndarray | None = None
    if minute_grid is not None and cached.bucket_timestamps is minute_grid.timestamps:
        offset = minute_grid.contiguous_offset(requested)
        if offset is not None:
            stop = offset + len(requested)
            volumes = np.asarray(list(cached.expected_volumes[offset:stop]), dtype=float)
    if volumes is None:
        cached_by_time = dict(zip(cached.bucket_timestamps, cached.expected_volumes, strict=True))
        if any(timestamp not in cached_by_time for timestamp in requested):
            raise ValueError("Requested horizon is incompatible with the latest boundary forecast.")
        volumes = np.asarray([cached_by_time[timestamp] for timestamp in requested], dtype=float)
    remaining_sum = float(volumes.sum())
    shares = volumes / remaining_sum if remaining_sum > 0 else np.zeros_like(volumes)
    return VolumeForecast(
        symbol=cached.symbol,
        session_date=cached.session_date,
        generated_at=generated_at,
        first_forecast_bucket=requested[0],
        bucket_timestamps=requested,
        expected_volumes=tuple(map(float, volumes)),
        normalized_shares=tuple(map(float, shares)),
        expected_remaining_volume=remaining_sum,
        forecaster_id=cached.forecaster_id,
        feature_schema_version=cached.feature_schema_version,
        training_data_cutoff=cached.training_data_cutoff,
        data_manifest_hash=cached.data_manifest_hash,
        warnings=cached.warnings,
    )


def _is_update_boundary(timestamp: pd.Timestamp) -> bool:
    local = timestamp.tz_convert("America/New_York")
    minutes = (local.hour * 60 + local.minute) - (9 * 60 + 30)
    return minutes >= 0 and minutes % 15 == 0


class MeanSeedForecastProvider:
    """Average three JEPA seeds for the appendix-only ensemble."""

    def __init__(self, providers: tuple[PaperLightGBMForecastProvider, ...]) -> None:
        if len(providers) != 3:
            raise ValueError("Seed-mean provider requires exactly three providers.")
        self.providers = providers

    @property
    def provider_id(self) -> str:
        return "jepa-seed-mean-13-29-47"

    def forecast(
        self,
        *,
        symbol: str,
        session_date: date,
        generated_at: pd.Timestamp,
        bucket_timestamps: Sequence[pd.Timestamp],
        observations: pd.DataFrame | None = None,
    ) -> VolumeForecast:
        forecasts = tuple(
            provider.forecast(
                symbol=symbol,
                session_date=session_date,
                generated_at=generated_at,
                bucket_timestamps=bucket_timestamps,
                observations=observations,
            )
            for provider in self.providers
        )
        return mean_seed_forecast(forecasts)
