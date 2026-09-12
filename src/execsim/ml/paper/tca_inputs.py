"""Bounded causal history and date slices for the unchanged TCA replay."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date, time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from execsim.data.paper.corporate_actions import point_in_time_split_factor
from execsim.data.paper.manifests import file_sha256
from execsim.data.paper.resolution_quality import assess_session_resolution_quality
from execsim.ml.paper.evaluation_artifacts import publish_frames, verify_artifact

HISTORY_FILES = ("bars.parquet", "adv.parquet", "profiles.parquet", "sessions.parquet")


def filter_tca_window_exact(
    bars: pd.DataFrame, selected_instruments: tuple[str, ...] | set[str]
) -> pd.DataFrame:
    """Retain only selected instrument-sessions with an exact 10:30-15:29 grid.

    The resolution-quality assessor is the authoritative contract. This function
    never fills, interpolates, or infers a missing minute from a calendar.
    """
    required = {"instrument_id", "timestamp"}
    missing = required.difference(bars.columns)
    if missing:
        raise ValueError(f"TCA population input missing columns: {sorted(missing)}")
    if bars.empty:
        return bars.copy()
    selected = (
        bars.loc[
            bars["instrument_id"].astype(str).isin({str(value) for value in selected_instruments})
        ]
        .copy()
        .reset_index(drop=True)
    )
    if selected.empty:
        return selected
    selected["__tca_session_date"] = None
    eligible: set[tuple[object, str]] = set()
    # Parse and assess each instrument independently. A malformed timezone in
    # one instrument must not suppress a valid instrument from the same date.
    for instrument_id, instrument_group in selected.groupby("instrument_id", sort=True):
        local_dates = []
        valid_timestamps = True
        for value in instrument_group["timestamp"]:
            try:
                timestamp = pd.Timestamp(value)
            except (TypeError, ValueError):
                valid_timestamps = False
                break
            if (
                pd.isna(timestamp)
                or timestamp.tzinfo is None
                or str(timestamp.tz) != "America/New_York"
            ):
                valid_timestamps = False
                break
            local_dates.append(timestamp.tz_convert("America/New_York").date())
        if not valid_timestamps:
            continue
        selected.loc[instrument_group.index, "__tca_session_date"] = local_dates
        dated_group = instrument_group.assign(__tca_session_date=local_dates)
        for session_date, group in dated_group.groupby("__tca_session_date", sort=True):
            try:
                quality = assess_session_resolution_quality(
                    group.drop(columns="__tca_session_date")
                )
            except (AttributeError, TypeError, ValueError):
                continue
            if quality.tca_window_exact:
                eligible.add((session_date, str(instrument_id)))
    keys = pd.MultiIndex.from_arrays(
        [selected["__tca_session_date"], selected["instrument_id"].astype(str)]
    )
    keep = keys.isin(eligible)
    return selected.loc[keep].drop(columns="__tca_session_date").reset_index(drop=True)


def tca_eligible_instrument_ids(
    bars: pd.DataFrame, selected_instruments: tuple[str, ...] | set[str]
) -> tuple[str, ...]:
    """Return the selected instruments that have at least one exact TCA session."""
    filtered = filter_tca_window_exact(bars, selected_instruments)
    return tuple(sorted(filtered["instrument_id"].astype(str).unique()))


def validate_tca_adv20(
    adv20: pd.DataFrame,
    eligible_cases: Mapping[date, tuple[str, ...] | set[str]],
) -> None:
    """Require one finite, positive causal ADV20 value for every eligible case.

    Exact-window eligibility is a scientific population rule and is intentionally
    independent of ADV availability.  Once that population is fixed, however,
    ADV20 is required evidence for constructing the frozen order quantity.  This
    validator therefore fails closed instead of allowing a missing or malformed
    derived row to silently remove a case.
    """
    required = {"instrument_id", "session_date", "adv20"}
    if missing := required.difference(adv20.columns):
        raise ValueError(f"ADV20 input missing columns: {sorted(missing)}")
    if not eligible_cases:
        return
    try:
        session_dates = pd.to_datetime(adv20["session_date"], errors="coerce").dt.date
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError("ADV20 session-date identity is invalid.") from exc
    for session_date, instruments in sorted(eligible_cases.items()):
        for instrument_id in sorted(str(value) for value in instruments):
            matches = adv20.loc[
                adv20["instrument_id"].astype(str).eq(instrument_id)
                & session_dates.eq(session_date)
            ]
            if len(matches) != 1:
                raise ValueError(
                    "ADV20 required for eligible TCA case "
                    f"{instrument_id}/{session_date}: expected exactly one row, "
                    f"found {len(matches)}."
                )
            try:
                value = float(matches.iloc[0]["adv20"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"ADV20 for eligible TCA case {instrument_id}/{session_date} is not numeric."
                ) from exc
            if not np.isfinite(value) or value <= 0:
                raise ValueError(
                    f"ADV20 for eligible TCA case {instrument_id}/{session_date} "
                    "must be finite and positive."
                )


def prepare_tca_history(
    market_path: Path,
    directory: Path,
    *,
    instrument_id: str,
    cutoffs: dict[str, date],
    identity: dict[str, Any],
    corporate_actions: pd.DataFrame | None = None,
    corporate_action_manifest_sha256: str | None = None,
) -> Path:
    """Reuse one instrument's history, profiles, and causal execution-share ADV20.

    When corporate-action evidence is supplied, historical daily volumes are
    converted into the target session's raw execution-share basis using only
    actions known and effective at the frozen 10:30 America/New_York quantity
    decision.  Replay bars remain untouched in their raw execution basis.
    """
    if (corporate_actions is None) != (corporate_action_manifest_sha256 is None):
        raise ValueError("Corporate-action rows and manifest identity must be supplied together.")
    history_schema = (
        "paper-tca-history-v3"
        if corporate_action_manifest_sha256 is not None
        else "paper-tca-history-v2"
    )
    expected = {
        **identity,
        "schema_version": history_schema,
        "instrument_id": instrument_id,
        "market_sha256": file_sha256(market_path),
        "training_cutoffs": {key: value.isoformat() for key, value in sorted(cutoffs.items())},
    }
    if corporate_action_manifest_sha256 is not None:
        expected["corporate_action_manifest_sha256"] = corporate_action_manifest_sha256
    if directory.exists():
        verify_artifact(directory, identity=expected, names=HISTORY_FILES)
        return directory
    bars = pd.read_parquet(market_path)
    if bars.empty or not bars["instrument_id"].astype(str).eq(instrument_id).all():
        raise ValueError("TCA compact history instrument mismatch.")
    # Aggregate in original corpus order before sorting derived replay rows.
    adv = causal_adv20(bars, corporate_actions=corporate_actions)
    profiles = pd.DataFrame(
        {
            "fold_id": list(sorted(cutoffs)),
            "instrument_id": instrument_id,
            "profile": [
                within_token_profile(bars, instrument_id, cutoffs[key]) for key in sorted(cutoffs)
            ],
        }
    )
    bars["session_date"] = (
        pd.to_datetime(bars["timestamp"]).dt.tz_convert("America/New_York").dt.date
    )
    bars = bars.sort_values(["session_date", "timestamp"], kind="stable").reset_index(drop=True)
    sessions = bars.loc[:, ["session_date"]].drop_duplicates().reset_index(drop=True)
    publish_frames(
        directory,
        identity=expected,
        row_group_size=390,
        frames={
            "bars.parquet": bars,
            "adv.parquet": adv,
            "profiles.parquet": profiles,
            "sessions.parquet": sessions,
        },
    )
    return directory


def read_tca_date(histories: dict[str, Path], day: date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read only one date across the fixed instrument population."""
    frames, advances = [], []
    for path in histories.values():
        frames.append(pd.read_parquet(path / "bars.parquet", filters=[("session_date", "==", day)]))
        advances.append(
            pd.read_parquet(path / "adv.parquet", filters=[("session_date", "==", day)])
        )
    return pd.concat(frames, ignore_index=True), pd.concat(advances, ignore_index=True)


def causal_adv20(
    bars: pd.DataFrame,
    *,
    corporate_actions: pd.DataFrame | None = None,
    decision_time: time = time(10, 30),
) -> pd.DataFrame:
    """Calculate causal ADV20 in each target session's execution-share units.

    ``volume`` is the raw daily share volume.  For target session ``t`` and a
    prior session ``h``, the adjusted contribution is

    ``V_h * F(h, I_t) / F(t, I_t)``

    where ``F`` is the cumulative point-in-time split factor resolved with
    ``observation_at`` and information clock ``I_t``.  Thus ADV20 and the
    target session replay bars use one raw execution-share basis.  The target
    session itself is excluded by the 20-session lag.
    """
    frame = bars.copy()
    timestamps = pd.to_datetime(frame["timestamp"])
    frame["session_date"] = timestamps.dt.tz_convert("America/New_York").dt.date
    # Preserve the historical no-action implementation exactly.  This keeps
    # old v2 histories numerically compatible while v3 histories bind action
    # evidence explicitly.
    if corporate_actions is None or corporate_actions.empty:
        daily = (
            frame.groupby(["instrument_id", "session_date"], sort=True, as_index=False)["volume"]
            .sum()
            .sort_values(["instrument_id", "session_date"], kind="stable")
        )
        daily["adv20"] = daily.groupby("instrument_id", sort=False)["volume"].transform(
            lambda values: values.shift(1).rolling(20, min_periods=20).mean()
        )
        return daily.dropna(subset=["adv20"])

    # Validate and normalize the sourced action contract once at the boundary.
    actions = ingest_corporate_actions_from_frame(corporate_actions)
    grouped = (
        frame.assign(__timestamp=timestamps)
        .groupby(["instrument_id", "session_date"], sort=True, as_index=False)
        .agg(volume=("volume", "sum"), session_observation_at=("__timestamp", "max"))
        .sort_values(["instrument_id", "session_date"], kind="stable")
    )
    rows: list[dict[str, object]] = []
    for instrument_id, group in grouped.groupby("instrument_id", sort=False):
        ordered = group.reset_index(drop=True)
        for index in range(20, len(ordered)):
            target = ordered.iloc[index]
            target_date = target["session_date"]
            target_info = pd.Timestamp(
                f"{target_date.isoformat()} {decision_time.isoformat()}",
                tz="America/New_York",
            )
            target_factor = point_in_time_split_factor(
                actions,
                instrument_id=str(instrument_id),
                observation_at=target_info,
                market_information_as_of=target_info,
            )
            adjusted: list[float] = []
            for historical in ordered.iloc[index - 20 : index].itertuples(index=False):
                historical_observation = pd.Timestamp(historical.session_observation_at)
                historical_factor = point_in_time_split_factor(
                    actions,
                    instrument_id=str(instrument_id),
                    observation_at=historical_observation,
                    market_information_as_of=target_info,
                )
                adjusted.append(float(historical.volume) * historical_factor / target_factor)
            rows.append(
                {
                    "instrument_id": instrument_id,
                    "session_date": target_date,
                    "volume": target["volume"],
                    "adv20": float(np.mean(adjusted)),
                }
            )
    return pd.DataFrame(rows, columns=["instrument_id", "session_date", "volume", "adv20"])


def ingest_corporate_actions_from_frame(actions: pd.DataFrame) -> pd.DataFrame:
    """Validate an already-loaded corporate-action frame without filesystem I/O."""
    required = {"instrument_id", "effective_date", "factor", "available_at", "source"}
    if missing := required.difference(actions.columns):
        raise ValueError(f"Corporate-action source missing columns: {sorted(missing)}")
    frame = actions.copy()
    frame["effective_date"] = pd.to_datetime(frame["effective_date"], errors="coerce").dt.date
    frame["available_at"] = pd.to_datetime(frame["available_at"], errors="coerce", utc=True)
    frame["factor"] = pd.to_numeric(frame["factor"], errors="coerce")
    if (
        frame[["effective_date", "available_at", "factor"]].isna().any().any()
        or (frame["factor"] <= 0).any()
        or frame["source"].astype(str).str.len().eq(0).any()
        or frame.duplicated(["instrument_id", "effective_date"]).any()
    ):
        raise ValueError("Corporate-action rows contain invalid dates, factors, or provenance.")
    return frame.sort_values(["instrument_id", "effective_date"], kind="stable")


def within_token_profile(
    bars: pd.DataFrame, instrument_id: str, training_cutoff: date
) -> np.ndarray:
    selected = bars.loc[
        (bars["instrument_id"].astype(str) == instrument_id)
        & (
            pd.to_datetime(bars["timestamp"]).dt.tz_convert("America/New_York").dt.date
            <= training_cutoff
        )
    ].copy()
    if selected.empty:
        raise ValueError(
            f"No TRAIN-only within-token history for {instrument_id}/{training_cutoff}."
        )
    local = pd.to_datetime(selected["timestamp"]).dt.tz_convert("America/New_York")
    selected["minute_in_token"] = ((local.dt.hour * 60 + local.dt.minute) - 570) % 15
    profile = selected.groupby("minute_in_token", sort=True)["volume"].mean().reindex(range(15))
    values = profile.to_numpy(dtype=float)
    if not np.isfinite(values).all() or values.sum() <= 0:
        raise ValueError("Within-token profile is incomplete or non-positive.")
    return values / values.sum()
