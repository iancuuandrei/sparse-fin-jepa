"""Build causal 15-minute token tensors from one regular session."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from execsim.data.paper.resolution_quality import aggregate_observed_tokens
from execsim.data.paper.validation import validate_exact_xnys_session
from execsim.ml.sequences.schemas import FEATURE_COUNT, TOKEN_COUNT, SequenceRecord


@dataclass(frozen=True, slots=True)
class _SeasonalBaseline:
    volume_profile: np.ndarray
    dollar_profile: np.ndarray
    trade_profile: np.ndarray
    adv20: float


def build_session_sequence(
    bars: pd.DataFrame,
    *,
    instrument_id: str,
    symbol: str,
    source_sha256: str,
    cutoff: str,
    seasonal: pd.DataFrame | None = None,
    spy_bars: pd.DataFrame | None = None,
    spy_seasonal: pd.DataFrame | None = None,
    previous_close: float | None = None,
    training_cutoff: str | None = None,
    data_classification: str = "historical",
    quality_protocol: str = "exact-minute-v1",
) -> SequenceRecord:
    """Aggregate one quality-valid session into 26 causal feature tokens."""
    required = {"timestamp", "open", "high", "low", "close", "volume", "trade_count", "vwap"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"Sequence input is missing columns: {sorted(missing)}")
    if data_classification != "synthetic_fixture" and (
        seasonal is None or spy_bars is None or spy_seasonal is None or previous_close is None
    ):
        raise ValueError(
            "Historical sequence builds require stock and SPY seasonal history, "
            "aligned SPY bars, and previous_close."
        )
    ordered = bars.sort_values("timestamp", kind="stable").reset_index(drop=True)
    timestamps = pd.to_datetime(ordered["timestamp"])
    if timestamps.dt.tz is None or timestamps.duplicated().any():
        raise ValueError("Sequence timestamps must be unique and timezone-aware.")
    tokens = _tokenize_session(ordered, quality_protocol=quality_protocol, label="paper")
    spy = None
    if spy_bars is not None:
        ordered_spy = spy_bars.sort_values("timestamp", kind="stable").reset_index(drop=True)
        spy = _tokenize_session(ordered_spy, quality_protocol=quality_protocol, label="SPY paper")
        if not pd.to_datetime(spy["timestamp"]).equals(pd.to_datetime(tokens["timestamp"])):
            raise ValueError("SPY and instrument token intervals must align exactly.")
    baseline = _baseline(seasonal, cutoff)
    spy_baseline = _baseline(spy_seasonal, cutoff)
    values = np.zeros((TOKEN_COUNT, FEATURE_COUNT), dtype=np.float32)
    prior_close = previous_close if previous_close is not None else float(tokens["open"].iloc[0])
    if not np.isfinite(prior_close) or prior_close <= 0:
        raise ValueError("previous_close must be finite and positive.")
    spy_prior_close = float(spy["open"].iloc[0]) if spy is not None else 1.0
    prior_surprise = 0.0
    cumulative = 0.0
    adv20 = baseline.adv20
    for index, row in enumerate(tokens.itertuples(index=False)):
        expected_volume = float(baseline.volume_profile[index])
        expected_dollar = float(baseline.dollar_profile[index])
        expected_trades = float(baseline.trade_profile[index])
        dollar = float(row.vwap * row.volume)
        surprise = math.log1p(float(row.volume)) - math.log1p(expected_volume)
        cumulative += float(row.volume)
        expected_cumulative = float(np.sum(baseline.volume_profile[: index + 1]))
        spy_values = (0.0, 0.0, 0.0)
        if spy is not None:
            spy_row = spy.iloc[index]
            spy_values = (
                math.log(float(spy_row["close"]) / spy_prior_close),
                math.log1p(float(spy_row["volume"]))
                - math.log1p(float(spy_baseline.volume_profile[index])),
                float(spy_row["realized_volatility"]),
            )
            spy_prior_close = float(spy_row["close"])
        angle = 2.0 * math.pi * (index + 0.5) / TOKEN_COUNT
        values[index] = (
            math.log(float(row.close) / prior_close),
            math.log(float(row.close) / float(row.open)),
            math.log(float(row.high) / float(row.low)),
            float(row.realized_volatility),
            surprise,
            math.log1p(dollar) - math.log1p(expected_dollar),
            math.log1p(float(row.trade_count)) - math.log1p(expected_trades),
            math.log((cumulative + 1.0) / (expected_cumulative + 1.0)),
            math.log1p(float(row.volume) / max(adv20 / TOKEN_COUNT, 1.0)),
            surprise - prior_surprise,
            math.log(float(row.vwap) / prior_close),
            math.log1p(adv20),
            math.sin(angle),
            math.cos(angle),
            (index + 1) / TOKEN_COUNT,
            *spy_values,
        )
        prior_close = float(row.close)
        prior_surprise = surprise
    available = tokens["timestamp"].astype("int64").to_numpy()
    session_date = timestamps.iloc[0].date().isoformat()
    return SequenceRecord(
        session_id=f"{instrument_id}-{session_date}",
        instrument_id=instrument_id,
        symbol=symbol.upper(),
        session_date=session_date,
        features=values,
        token_mask=np.ones(TOKEN_COUNT, dtype=bool),
        available_at_ns=available,
        raw_volume=tokens["volume"].to_numpy(dtype=float),
        raw_vwap=tokens["vwap"].to_numpy(dtype=float),
        causal_baseline_volume=baseline.volume_profile.astype(float),
        observed_bar_count=(
            tokens["observed_bar_count"].to_numpy(dtype=np.int16)
            if "observed_bar_count" in tokens
            else np.full(TOKEN_COUNT, 15, dtype=np.int16)
        ),
        provider_gap_count=390 - len(ordered),
        quality_protocol=quality_protocol,
        source_sha256=source_sha256,
        cutoff=cutoff,
        training_cutoff=training_cutoff or cutoff,
        market_information_as_of=str(timestamps.iloc[-1]),
        feature_history_end=cutoff,
    )


def _aggregate_tokens(bars: pd.DataFrame) -> pd.DataFrame:
    if len(bars) != 390:
        raise ValueError("Token aggregation requires a complete 390-minute session.")
    rows = []
    for start in range(0, 390, 15):
        bucket = bars.iloc[start : start + 15]
        closes = pd.to_numeric(bucket["close"]).to_numpy(dtype=float)
        returns = np.diff(np.log(closes))
        volume = float(pd.to_numeric(bucket["volume"]).sum())
        dollar = float((pd.to_numeric(bucket["vwap"]) * pd.to_numeric(bucket["volume"])).sum())
        rows.append(
            {
                "timestamp": pd.to_datetime(bucket["timestamp"].iloc[-1]) + pd.Timedelta(minutes=1),
                "open": float(bucket["open"].iloc[0]),
                "high": float(pd.to_numeric(bucket["high"]).max()),
                "low": float(pd.to_numeric(bucket["low"]).min()),
                "close": float(bucket["close"].iloc[-1]),
                "volume": volume,
                "trade_count": float(pd.to_numeric(bucket["trade_count"]).sum()),
                "vwap": dollar / max(volume, 1.0),
                "realized_volatility": float(np.sqrt(np.square(returns).sum())),
            }
        )
    return pd.DataFrame(rows)


def _tokenize_session(bars: pd.DataFrame, *, quality_protocol: str, label: str) -> pd.DataFrame:
    if quality_protocol == "resolution-aware-v2":
        try:
            return aggregate_observed_tokens(bars)
        except ValueError as exc:
            raise ValueError(f"Invalid {label} session under v2 token quality: {exc}") from exc
    if quality_protocol != "exact-minute-v1":
        raise ValueError(f"Unknown paper quality protocol: {quality_protocol}")
    if len(bars) != 390:
        raise ValueError(f"{label.capitalize()} sequence input requires 390 minutes.")
    grid_errors = validate_exact_xnys_session(bars)
    if grid_errors:
        raise ValueError(f"Invalid {label} session: " + "; ".join(grid_errors))
    return _aggregate_tokens(bars)


def _baseline(seasonal: pd.DataFrame | None, cutoff: str | None = None) -> _SeasonalBaseline:
    if seasonal is None:
        profile = np.ones(TOKEN_COUNT, dtype=float)
        return _SeasonalBaseline(profile, profile, profile, 1.0)
    required = {"session_date", "bucket_index", "volume", "dollar_volume", "trade_count"}
    missing = required.difference(seasonal.columns)
    if missing:
        raise ValueError(f"Seasonal baseline missing columns: {sorted(missing)}")
    session_dates = pd.to_datetime(seasonal["session_date"]).dt.date
    if cutoff is not None:
        cutoff_date = pd.Timestamp(cutoff).date()
        seasonal = seasonal.loc[session_dates <= cutoff_date].copy()
        session_dates = pd.to_datetime(seasonal["session_date"]).dt.date
        if not len(seasonal) or max(session_dates) > cutoff_date:
            raise ValueError("Seasonal history must be available by the declared cutoff.")
    ordered_dates = sorted(session_dates.unique())[-20:]
    history = seasonal.loc[session_dates.isin(ordered_dates)]
    history = history.sort_values(["session_date", "bucket_index"], kind="stable")
    expected_buckets = np.tile(np.arange(TOKEN_COUNT), len(ordered_dates))
    observed_buckets = pd.to_numeric(history["bucket_index"], errors="coerce").to_numpy()
    if len(history) != len(ordered_dates) * TOKEN_COUNT or not np.array_equal(
        observed_buckets, expected_buckets
    ):
        raise ValueError("Seasonal baselines require exactly 26 ordered buckets per session.")
    alpha = 2.0 / 21.0
    weights = np.power(1.0 - alpha, np.arange(len(ordered_dates) - 1, -1, -1))
    weights /= weights.sum()

    def ewma_profile(column: str) -> np.ndarray:
        values = pd.to_numeric(history[column], errors="coerce").to_numpy(dtype=float)
        return weights @ values.reshape(len(ordered_dates), TOKEN_COUNT)

    volume = ewma_profile("volume")
    dollar = ewma_profile("dollar_volume")
    trades = ewma_profile("trade_count")
    if not all(
        np.isfinite(value).all() and (value >= 0).all() for value in (volume, dollar, trades)
    ):
        raise ValueError("Seasonal baselines require all 26 finite non-negative buckets.")
    daily_volume = pd.to_numeric(history["volume"], errors="coerce").to_numpy(dtype=float)
    adv20 = float(daily_volume.reshape(len(ordered_dates), TOKEN_COUNT).sum(axis=1).mean())
    return _SeasonalBaseline(volume, dollar, trades, adv20)
