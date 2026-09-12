from dataclasses import replace
from datetime import date, time
from statistics import median
from time import perf_counter

import numpy as np
import pandas as pd
import pytest

from execsim.forecasting.models import VolumeForecast
from execsim.ml.paper.forecast_provider import (
    _MinuteGrid,
    _truncate_forecast,
)
from execsim.orders import ParentOrder
from execsim.policies import ExecutionConstraints, PolicyDecision
from execsim.simulator import simulate_policy


def _forecast(
    timestamps: tuple[pd.Timestamp, ...],
    *,
    generated_at: pd.Timestamp,
    volumes: tuple[float, ...],
    label: str,
) -> VolumeForecast:
    total = float(sum(volumes))
    shares = tuple(value / total for value in volumes) if total else (0.0,) * len(volumes)
    return VolumeForecast(
        symbol="AAA",
        session_date=date(2024, 4, 1),
        generated_at=generated_at,
        first_forecast_bucket=timestamps[0],
        bucket_timestamps=timestamps,
        expected_volumes=volumes,
        normalized_shares=shares,
        expected_remaining_volume=total,
        forecaster_id=label,
        feature_schema_version="fixture-v1",
        training_data_cutoff=date(2024, 3, 29),
        data_manifest_hash="a" * 64,
        warnings=(label,),
    )


def _reference_truncate(
    cached: VolumeForecast,
    requested: tuple[pd.Timestamp, ...],
    generated_at: pd.Timestamp,
) -> VolumeForecast:
    """Preserve the original dictionary lookup as an exact-output reference."""
    cached_by_time = dict(zip(cached.bucket_timestamps, cached.expected_volumes, strict=True))
    if any(timestamp not in cached_by_time for timestamp in requested):
        raise ValueError("Requested horizon is incompatible with the latest boundary forecast.")
    volumes = np.asarray([cached_by_time[timestamp] for timestamp in requested], dtype=float)
    remaining_sum = float(volumes.sum())
    shares = volumes / remaining_sum if remaining_sum > 0 else np.zeros_like(volumes)
    # Preserve metadata independently of production's field-by-field constructor.
    return replace(
        cached,
        generated_at=generated_at,
        first_forecast_bucket=requested[0],
        bucket_timestamps=requested,
        expected_volumes=tuple(map(float, volumes)),
        normalized_shares=tuple(map(float, shares)),
        expected_remaining_volume=remaining_sum,
    )


def test_minute_grid_truncation_matches_dictionary_reference_for_300_minutes() -> None:
    tca_minutes = tuple(
        pd.date_range(
            "2024-04-01 10:30",
            "2024-04-01 15:29",
            freq="min",
            tz="America/New_York",
        )
    )
    forecast_end = pd.Timestamp("2024-04-01 15:59", tz="America/New_York")
    minute_grid: _MinuteGrid | None = None
    cached: VolumeForecast | None = None
    boundary_offset = 0

    for offset, generated_at in enumerate(tca_minutes):
        if offset % 15 == 0:
            boundary_offset = offset
            minute_grid = _MinuteGrid.from_range(generated_at, forecast_end)
            volumes = tuple(float((index + offset) % 19) for index in range(minute_grid.periods))
            cached = _forecast(
                minute_grid.timestamps,
                generated_at=generated_at,
                volumes=volumes,
                label=f"boundary-{offset:03d}",
            )

        assert minute_grid is not None
        assert cached is not None
        requested = tca_minutes[offset:]
        assert minute_grid.contiguous_offset(requested) == offset - boundary_offset
        optimized = _truncate_forecast(
            cached,
            requested,
            generated_at,
            minute_grid=minute_grid,
        )
        reference = _reference_truncate(cached, requested, generated_at)
        assert optimized == reference


def test_minute_grid_truncation_falls_back_for_noncontiguous_requests() -> None:
    start = pd.Timestamp("2024-04-01 10:30", tz="America/New_York")
    minute_grid = _MinuteGrid.from_range(start, start + pd.Timedelta(minutes=4))
    cached = _forecast(
        minute_grid.timestamps,
        generated_at=start - pd.Timedelta(minutes=1),
        volumes=(0.0, 2.0, 3.0, 5.0, 7.0),
        label="noncontiguous",
    )
    requests = (
        (minute_grid.timestamps[0], minute_grid.timestamps[2]),
        (minute_grid.timestamps[3], minute_grid.timestamps[1]),
    )

    for requested in requests:
        assert minute_grid.contiguous_offset(requested) is None
        assert _truncate_forecast(
            cached,
            requested,
            start,
            minute_grid=minute_grid,
        ) == _reference_truncate(cached, requested, start)


def test_generic_forecast_keeps_last_duplicate_key_and_unsorted_request_semantics() -> None:
    start = pd.Timestamp("2024-04-01 10:30", tz="America/New_York")
    timestamps = (
        start,
        start + pd.Timedelta(minutes=2),
        start + pd.Timedelta(minutes=1),
        start + pd.Timedelta(minutes=1),
    )
    cached = _forecast(
        timestamps,
        generated_at=start - pd.Timedelta(minutes=1),
        volumes=(1.0, 2.0, 3.0, 4.0),
        label="duplicate-cache-time",
    )
    requested = (timestamps[2], timestamps[0], timestamps[2], timestamps[1])
    grid = _MinuteGrid.from_range(start, start + pd.Timedelta(minutes=2))

    optimized = _truncate_forecast(cached, requested, start, minute_grid=grid)

    assert optimized == _reference_truncate(cached, requested, start)
    assert optimized.expected_volumes == (4.0, 1.0, 4.0, 2.0)


def test_minute_grid_truncation_preserves_missing_key_error() -> None:
    start = pd.Timestamp("2024-04-01 10:30", tz="America/New_York")
    grid = _MinuteGrid.from_range(start, start + pd.Timedelta(minutes=2))
    cached = _forecast(
        grid.timestamps,
        generated_at=start,
        volumes=(1.0, 2.0, 3.0),
        label="missing-key",
    )
    requested = (start, start + pd.Timedelta(minutes=3))

    with pytest.raises(ValueError, match="incompatible"):
        _truncate_forecast(cached, requested, start, minute_grid=grid)
    with pytest.raises(ValueError, match="incompatible"):
        _reference_truncate(cached, requested, start)


def test_minute_grid_truncation_preserves_all_zero_volume_forecast() -> None:
    start = pd.Timestamp("2024-04-01 10:30", tz="America/New_York")
    grid = _MinuteGrid.from_range(start, start + pd.Timedelta(minutes=3))
    cached = _forecast(
        grid.timestamps,
        generated_at=start,
        volumes=(0.0, 0.0, 0.0, 0.0),
        label="zero-volume",
    )
    requested = grid.timestamps[1:]

    optimized = _truncate_forecast(cached, requested, requested[0], minute_grid=grid)

    assert optimized == _reference_truncate(cached, requested, requested[0])
    assert optimized.expected_volumes == (0.0, 0.0, 0.0)
    assert optimized.normalized_shares == (0.0, 0.0, 0.0)
    assert optimized.expected_remaining_volume == 0.0


def test_300_minute_simulation_output_is_exact_with_minute_grid_fast_path(monkeypatch) -> None:
    from execsim.ml.paper import forecast_provider as forecast_provider_module
    from execsim.ml.paper.forecast_provider import PaperLightGBMForecastProvider

    start = pd.Timestamp("2024-04-01 10:30", tz="America/New_York")
    tca_minutes = tuple(pd.date_range(start, periods=300, freq="min"))

    class Model:
        def predict_frames(self, scale, shape, *, group_columns):
            del group_columns
            predicted = shape.copy()
            predicted["conditional_share"] = 1.0 / len(predicted)
            return scale["remaining_volume"].to_numpy(dtype=float), predicted

    close = pd.Timestamp("2024-04-01 15:59", tz=start.tz)

    def resolve(symbol, session_date, generated_at, observations):
        del symbol, session_date, observations
        remaining_minutes = int((close - generated_at) / pd.Timedelta(minutes=1)) + 1
        token_count = (remaining_minutes + 14) // 15
        return (
            pd.DataFrame({"remaining_volume": [500_000.0 + generated_at.minute]}),
            pd.DataFrame({"target_bucket": np.arange(token_count)}),
        )

    def provider():
        return PaperLightGBMForecastProvider(
            Model(),  # type: ignore[arg-type]
            feature_resolver=resolve,
            within_token_profile=np.full(15, 1 / 15),
            training_cutoff=date(2024, 3, 29),
            manifest_hash="b" * 64,
            method_id="simulation-parity",
        )

    class OneSharePerMinutePolicy:
        policy_name = "one-share-per-minute"

        def reset(self) -> None:
            pass

        def decide(self, context):
            return PolicyDecision(
                policy_name=self.policy_name,
                planned_quantity=1,
                forecast_id=context.forecast.forecaster_id,
                decision_id=f"decision-{context.elapsed_buckets:03d}",
            )

    bars = pd.DataFrame(
        {
            "symbol": "AAA",
            "timestamp": tca_minutes,
            "open": 100.0,
            "high": 100.1,
            "low": 99.9,
            "close": 100.0,
            "volume": 1_000.0,
        }
    )
    order = ParentOrder("AAA", "buy", 300, date(2024, 4, 1), time(10, 30), time(15, 30))
    constraints = ExecutionConstraints(0.1, 0.1)

    fast = simulate_policy(
        parent_order=order,
        bars=bars,
        policy=OneSharePerMinutePolicy(),
        constraints=constraints,
        forecast_provider=provider(),
    )

    original_truncate = forecast_provider_module._truncate_forecast

    def dictionary_truncate(cached, requested, generated_at, *, minute_grid=None):
        del minute_grid
        return original_truncate(cached, requested, generated_at)

    monkeypatch.setattr(forecast_provider_module, "_truncate_forecast", dictionary_truncate)
    reference = simulate_policy(
        parent_order=order,
        bars=bars,
        policy=OneSharePerMinutePolicy(),
        constraints=constraints,
        forecast_provider=provider(),
    )

    assert fast.summary == reference.summary
    pd.testing.assert_frame_equal(fast.execution_log, reference.execution_log, check_exact=True)
    pd.testing.assert_frame_equal(fast.decision_trace, reference.decision_trace, check_exact=True)


def _benchmark_300_minute_truncation(repeats: int = 5) -> dict[str, float]:
    """Return bounded synthetic 300-minute fallback/fast-path elapsed seconds."""
    tca_minutes = tuple(
        pd.date_range(
            "2024-04-01 10:30",
            "2024-04-01 15:29",
            freq="min",
            tz="America/New_York",
        )
    )
    forecast_end = pd.Timestamp("2024-04-01 15:59", tz="America/New_York")
    cases: list[tuple[VolumeForecast, tuple[pd.Timestamp, ...], pd.Timestamp, _MinuteGrid]] = []
    minute_grid: _MinuteGrid | None = None
    cached: VolumeForecast | None = None
    for offset, generated_at in enumerate(tca_minutes):
        if offset % 15 == 0:
            minute_grid = _MinuteGrid.from_range(generated_at, forecast_end)
            volumes = tuple(float((index + offset) % 19) for index in range(minute_grid.periods))
            cached = _forecast(
                minute_grid.timestamps,
                generated_at=generated_at,
                volumes=volumes,
                label=f"benchmark-{offset:03d}",
            )
        assert minute_grid is not None
        assert cached is not None
        cases.append((cached, tca_minutes[offset:], generated_at, minute_grid))

    samples: dict[str, list[float]] = {"dictionary": [], "minute_grid": []}
    for repeat in range(repeats):
        paths = (("dictionary", False), ("minute_grid", True))
        if repeat % 2:
            paths = tuple(reversed(paths))
        for name, use_grid in paths:
            started = perf_counter()
            for cached_forecast, requested, generated_at, grid in cases:
                _truncate_forecast(
                    cached_forecast,
                    requested,
                    generated_at,
                    minute_grid=grid if use_grid else None,
                )
            samples[name].append(perf_counter() - started)
    return {name: median(values) for name, values in samples.items()}
