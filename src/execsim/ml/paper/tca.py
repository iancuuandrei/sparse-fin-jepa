"""Causal bridge from 15-minute model updates to minute-level VolumeForecast objects."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import date, time
from typing import Literal

import numpy as np
import pandas as pd

from execsim.forecasting.models import VolumeForecast, VolumeForecastProvider
from execsim.policies.models import DecisionContext, PolicyDecision

PAPER_METHODS = (
    "ewma",
    "lightgbm_raw",
    "raw_untrained_neural",
    "raw_dense_jepa_seed_13",
    "raw_dense_jepa_seed_29",
    "raw_dense_jepa_seed_47",
    "raw_sparse_jepa_seed_13",
    "raw_sparse_jepa_seed_29",
    "raw_sparse_jepa_seed_47",
)


@dataclass(slots=True)
class SegmentCommittedMPCPolicy:
    """Solve every 15 minutes and commit the next minute-level allocation segment."""

    half_spread: float
    temporary_impact: float
    risk_aversion: float = 0.0
    tracking_penalty: float = 0.0
    segment_minutes: int = 15
    volatility: float = 0.01
    policy_name: str = "paper-segment-committed-mpc"
    _segment: list[int] = field(default_factory=list, init=False, repr=False)
    _decision_number: int = field(default=0, init=False, repr=False)
    solve_count: int = field(default=0, init=False)

    def reset(self) -> None:
        self._segment.clear()
        self._decision_number = 0
        self.solve_count = 0

    def decide(self, context: DecisionContext) -> PolicyDecision:
        from execsim.optimization import OptimalExecutionProblem, OptimalExecutionWorkspace

        if context.forecast is None:
            raise ValueError("Segment-committed MPC requires a point-in-time forecast.")
        solved = not self._segment
        if solved:
            n = context.remaining_buckets
            problem = OptimalExecutionProblem(
                quantity=max(context.remaining_inventory, 1),
                forecast_volumes=np.asarray(context.forecast.expected_volumes, dtype=float),
                forecast_volatilities=np.full(n, self.volatility),
                half_spreads=np.full(n, self.half_spread),
                temporary_impacts=np.full(n, self.temporary_impact),
                max_participation_rate=context.constraints.planned_participation_rate,
                risk_aversion=self.risk_aversion,
                tracking_penalty=self.tracking_penalty,
                forecast_weights=np.asarray(context.forecast.normalized_shares, dtype=float),
            )
            result = OptimalExecutionWorkspace(n, validation_level="structural").solve(problem)
            self._segment = [
                int(value) for value in result.integer_quantities[: self.segment_minutes]
            ]
            self.solve_count += 1
            self._decision_number += 1
        action = min(self._segment.pop(0), context.remaining_inventory)
        decision_id = f"segment-mpc-{self._decision_number:04d}"
        return PolicyDecision(
            policy_name=self.policy_name,
            planned_quantity=action,
            forecast_id=context.forecast.forecaster_id,
            decision_id=decision_id,
            trace={
                "decision_id": decision_id,
                "segment_offset": self.segment_minutes - len(self._segment) - 1,
                "solver_invocation": solved,
                "decision_clock_minutes": self.segment_minutes,
                "forecast_id": context.forecast.forecaster_id,
            },
        )


def mean_seed_forecast(forecasts: tuple[VolumeForecast, ...]) -> VolumeForecast:
    """Build the explicitly appendix-only three-seed forecast ensemble."""
    if len(forecasts) != 3:
        raise ValueError("The appendix JEPA ensemble requires exactly three seeds.")
    first = forecasts[0]
    identity = (
        first.symbol,
        first.session_date,
        first.generated_at,
        first.bucket_timestamps,
        first.training_data_cutoff,
        first.data_manifest_hash,
    )
    for forecast in forecasts[1:]:
        candidate = (
            forecast.symbol,
            forecast.session_date,
            forecast.generated_at,
            forecast.bucket_timestamps,
            forecast.training_data_cutoff,
            forecast.data_manifest_hash,
        )
        if candidate != identity:
            raise ValueError("Seed forecasts must describe the same causal case.")
    volume = np.mean([forecast.expected_volumes for forecast in forecasts], axis=0)
    total = float(volume.sum())
    shares = volume / total if total > 0 else np.zeros_like(volume)
    return VolumeForecast(
        symbol=first.symbol,
        session_date=first.session_date,
        generated_at=first.generated_at,
        first_forecast_bucket=first.first_forecast_bucket,
        bucket_timestamps=first.bucket_timestamps,
        expected_volumes=tuple(map(float, volume)),
        normalized_shares=tuple(map(float, shares)),
        expected_remaining_volume=total,
        forecaster_id="jepa-seed-mean-13-29-47",
        feature_schema_version=first.feature_schema_version,
        training_data_cutoff=first.training_data_cutoff,
        data_manifest_hash=first.data_manifest_hash,
        warnings=tuple(sorted({warning for item in forecasts for warning in item.warnings})),
    )


def select_liquidity_spaced_instruments(
    universe: pd.DataFrame, *, size: int = 30
) -> tuple[str, ...]:
    """Select stable approximately equal-rank points across the frozen liquidity ordering."""
    required = {"rank", "instrument_id"}
    missing = required.difference(universe.columns)
    if missing or len(universe) < size:
        raise ValueError(f"TCA universe is too small or missing columns: {sorted(missing)}")
    ordered = universe.sort_values(["rank", "instrument_id"], kind="stable")
    positions = np.linspace(0, len(ordered) - 1, size).round().astype(int)
    return tuple(ordered.iloc[positions]["instrument_id"].astype(str))


def balanced_sides(
    instrument_ids: tuple[str, ...], session_date: date
) -> dict[str, Literal["buy", "sell"]]:
    """Assign stable-hash sides with buy/sell counts differing by at most one."""
    ranked = sorted(
        instrument_ids,
        key=lambda value: hashlib.sha256(f"{session_date}|{value}".encode()).hexdigest(),
    )
    buy_count = (len(ranked) + 1) // 2
    return {
        instrument_id: "buy" if index < buy_count else "sell"
        for index, instrument_id in enumerate(ranked)
    }


def realized_volume_oracle_cost(
    bars: pd.DataFrame,
    *,
    quantity: int,
    start: time,
    end: time,
    arrival_price: float,
    max_participation_rate: float = 0.10,
    half_spread_arrival_fraction: float = 5e-5,
    temporary_impact_arrival_fraction: float = 1e-3,
) -> float:
    """Solve the evaluation-only continuous allocation against realized volume."""
    timestamps = pd.to_datetime(bars["timestamp"]).dt.tz_convert("America/New_York")
    window = bars.loc[(timestamps.dt.time >= start) & (timestamps.dt.time < end)]
    volumes = window["volume"].to_numpy(dtype=float)
    if (
        isinstance(quantity, bool)
        or quantity <= 0
        or not len(volumes)
        or not np.isfinite(volumes).all()
        or (volumes < 0).any()
        or not np.isfinite(arrival_price)
        or arrival_price <= 0
    ):
        raise ValueError("Realized-volume oracle inputs must be positive, finite, and non-empty.")
    if not 0 <= max_participation_rate <= 1:
        raise ValueError("Realized-volume oracle participation must be in [0, 1].")
    if half_spread_arrival_fraction < 0 or temporary_impact_arrival_fraction <= 0:
        raise ValueError("Realized-volume oracle cost fractions must be positive.")
    total_volume = float(volumes.sum())
    if total_volume == 0:
        return 0.0
    # With zero risk/tracking penalties and a common temporary-impact coefficient,
    # KKT conditions give q_i proportional to v_i. The feasible total is capped by
    # the continuous constraint q_i <= participation * v_i; no integer projection
    # enters this evaluation-only lower bound.
    feasible_quantity = min(float(quantity), max_participation_rate * total_volume)
    allocations = feasible_quantity * volumes / total_volume
    temporary_impact = arrival_price * temporary_impact_arrival_fraction
    return float(np.sum(temporary_impact * np.square(allocations) / np.maximum(volumes, 1.0)))


def expand_volume_forecast(
    *,
    symbol: str,
    session_date: date,
    generated_at: pd.Timestamp,
    minute_timestamps: tuple[pd.Timestamp, ...],
    expected_remaining_volume: float,
    conditional_token_shape: np.ndarray,
    within_token_profile: np.ndarray,
    training_cutoff: date,
    manifest_hash: str,
    forecaster_id: str,
) -> VolumeForecast:
    """Disaggregate with a train-only profile and causally truncate at the update time."""
    token_shape = np.asarray(conditional_token_shape, dtype=float)
    within = np.asarray(within_token_profile, dtype=float)
    if (
        token_shape.ndim != 1
        or within.shape != (15,)
        or (token_shape < 0).any()
        or (within < 0).any()
    ):
        raise ValueError("Token and within-bin profiles must be non-negative vectors.")
    if not np.isclose(token_shape.sum(), 1.0) or not np.isclose(within.sum(), 1.0):
        raise ValueError("Token and within-bin profiles must each sum to one.")
    shares = np.outer(token_shape, within).reshape(-1)[: len(minute_timestamps)]
    shares /= shares.sum()
    volumes = expected_remaining_volume * shares
    return VolumeForecast(
        symbol=symbol,
        session_date=session_date,
        generated_at=generated_at,
        first_forecast_bucket=minute_timestamps[0],
        bucket_timestamps=minute_timestamps,
        expected_volumes=tuple(float(value) for value in volumes),
        normalized_shares=tuple(float(value) for value in shares),
        expected_remaining_volume=float(expected_remaining_volume),
        forecaster_id=forecaster_id,
        feature_schema_version="paper-token-v1",
        training_data_cutoff=training_cutoff,
        data_manifest_hash=manifest_hash,
    )


ForecastFactory = Callable[[str, date], VolumeForecastProvider]


def run_historical_tca(
    bars: pd.DataFrame,
    universe: pd.DataFrame,
    adv20: pd.DataFrame,
    providers: Mapping[str, ForecastFactory],
    *,
    liquidity_size: int = 30,
    order_fraction: float = 0.03,
    required_methods: tuple[str, ...] = PAPER_METHODS,
    start: str = "10:30",
    end: str = "15:30",
    planned_participation: float = 0.10,
    hard_participation: float = 0.10,
    risk_aversion: float = 0.0,
    tracking_penalty: float = 0.0,
    half_spread_arrival_fraction: float = 5e-5,
    temporary_impact_arrival_fraction: float = 1e-3,
) -> pd.DataFrame:
    """Run matched 10:30-15:30 deterministic MPC cases with only provider variation."""
    from execsim.costs import CostParameter, LinearTemporaryImpactModel
    from execsim.forecasting.historical import HistoricalForecastUnavailable
    from execsim.ml.paper.tca_inputs import filter_tca_window_exact, validate_tca_adv20
    from execsim.orders import ParentOrder
    from execsim.policies import ExecutionConstraints
    from execsim.simulator import simulate_policy

    if not required_methods or set(providers) != set(required_methods):
        raise ValueError(f"Historical TCA provider set does not match: {required_methods}")
    if order_fraction not in {0.01, 0.03, 0.05}:
        raise ValueError("Paper TCA supports only primary 3% and appendix 1%/5% ADV sizes.")
    start_time = time.fromisoformat(start)
    end_time = time.fromisoformat(end)
    if (
        start_time != time(10, 30)
        or end_time != time(15, 30)
        or planned_participation != 0.10
        or hard_participation != 0.10
        or risk_aversion != 0.0
        or tracking_penalty != 0.0
        or half_spread_arrival_fraction != 5e-5
        or temporary_impact_arrival_fraction != 1e-3
    ):
        raise ValueError("Historical TCA parameters contradict the locked experiment.")
    instruments = select_liquidity_spaced_instruments(universe, size=liquidity_size)
    required_adv = {"instrument_id", "session_date", "adv20"}
    if missing := required_adv.difference(adv20.columns):
        raise ValueError(f"ADV20 input missing columns: {sorted(missing)}")
    adv_session_dates = pd.to_datetime(adv20["session_date"], errors="coerce").dt.date
    selected = filter_tca_window_exact(bars, instruments)
    selected["session_date"] = (
        pd.to_datetime(selected["timestamp"]).dt.tz_convert("America/New_York").dt.date
    )
    # Validate the complete exact-window population before invoking any provider
    # or replay.  ADV availability is derived evidence, not a population filter.
    eligible_cases = {
        session_date: tuple(sorted(date_bars["instrument_id"].astype(str).unique()))
        for session_date, date_bars in selected.groupby("session_date", sort=True)
    }
    validate_tca_adv20(adv20, eligible_cases)

    rows = []
    for session_date, date_bars in selected.groupby("session_date", sort=True):
        available = tuple(sorted(date_bars["instrument_id"].astype(str).unique()))
        sides = balanced_sides(available, session_date)
        for instrument_id in available:
            instrument_bars = date_bars.loc[
                date_bars["instrument_id"].astype(str) == instrument_id
            ].copy()
            symbol = str(instrument_bars["symbol"].iloc[0])
            adv_match = adv20.loc[
                (adv20["instrument_id"].astype(str) == instrument_id)
                & adv_session_dates.eq(session_date)
            ]
            # validate_tca_adv20 above makes this a defensive assertion rather
            # than a silent population change if this path is edited later.
            if len(adv_match) != 1:  # pragma: no cover - guarded by preflight
                raise ValueError(
                    "ADV20 required for eligible TCA case "
                    f"{instrument_id}/{session_date}: expected exactly one row."
                )
            quantity = max(1, round(order_fraction * float(adv_match["adv20"].iloc[0])))
            arrival = float(
                instrument_bars.loc[
                    pd.to_datetime(instrument_bars["timestamp"])
                    .dt.tz_convert("America/New_York")
                    .dt.time
                    == start_time,
                    "open",
                ].iloc[0]
            )
            cost_model = LinearTemporaryImpactModel(
                half_spread=CostParameter(arrival * half_spread_arrival_fraction),
                temporary_impact=CostParameter(arrival * temporary_impact_arrival_fraction),
            )
            parent_order = ParentOrder(
                symbol,
                sides[instrument_id],
                quantity,
                session_date,
                start_time,
                end_time,
            )

            def paper_policy(
                arrival_price: float = arrival,
            ) -> SegmentCommittedMPCPolicy:
                return SegmentCommittedMPCPolicy(
                    half_spread=arrival_price * half_spread_arrival_fraction,
                    temporary_impact=arrival_price * temporary_impact_arrival_fraction,
                    risk_aversion=risk_aversion,
                    tracking_penalty=tracking_penalty,
                )

            constraints = ExecutionConstraints(planned_participation, hard_participation)
            oracle_cost = realized_volume_oracle_cost(
                instrument_bars,
                quantity=quantity,
                start=start_time,
                end=end_time,
                arrival_price=arrival,
                max_participation_rate=hard_participation,
                half_spread_arrival_fraction=half_spread_arrival_fraction,
                temporary_impact_arrival_fraction=temporary_impact_arrival_fraction,
            )
            for method in required_methods:
                factory = providers[method]
                unavailable_reason = ""
                try:
                    result = simulate_policy(
                        parent_order=parent_order,
                        bars=instrument_bars,
                        policy=paper_policy(),
                        constraints=constraints,
                        cost_model=cost_model,
                        forecast_provider=factory(instrument_id, session_date),
                    )
                except HistoricalForecastUnavailable as exc:
                    if method != "ewma":
                        raise
                    unavailable_reason = str(exc)
                    result = None
                rows.append(
                    {
                        "method": method,
                        "date": session_date,
                        "instrument_id": instrument_id,
                        "symbol": symbol,
                        "order_fraction_adv20": order_fraction,
                        "parent_quantity": quantity,
                        "side": sides[instrument_id],
                        "start": start,
                        "end": end,
                        "planned_participation": planned_participation,
                        "hard_participation": hard_participation,
                        "risk_aversion": risk_aversion,
                        "tracking_penalty": tracking_penalty,
                        "status": "EWMA_UNAVAILABLE" if result is None else "AVAILABLE",
                        "unavailable_reason": unavailable_reason,
                        "total_modeled_execution_cost": result.summary.total_modeled_execution_cost
                        if result is not None
                        else np.nan,
                        "absolute_modeled_impact_cost": (
                            result.summary.modeled_temporary_impact_cost
                            if result is not None
                            else np.nan
                        ),
                        "oracle_modeled_impact_cost": oracle_cost,
                        "normalized_allocation_regret": (
                            (
                                result.summary.modeled_temporary_impact_cost
                                if result is not None
                                else np.nan
                            )
                            - oracle_cost
                        )
                        / max(oracle_cost, 1e-12),
                        "implementation_shortfall_bps": result.summary.implementation_shortfall_bps
                        if result is not None
                        else np.nan,
                        "completion_rate": result.summary.completion_rate
                        if result is not None
                        else np.nan,
                    }
                )
    output = pd.DataFrame(rows)
    if output.empty:
        raise ValueError("Historical TCA produced no matched cases.")
    counts = output.groupby(["date", "instrument_id"], sort=False)["method"].nunique()
    if not (counts == len(required_methods)).all():
        raise ValueError("Historical TCA did not produce all methods for every retained case.")
    return output
