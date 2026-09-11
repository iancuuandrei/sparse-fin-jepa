"""Bounded causal history and date slices for the unchanged TCA replay."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from execsim.data.paper.manifests import file_sha256
from execsim.ml.paper.evaluation_artifacts import publish_frames, verify_artifact

HISTORY_FILES = ("bars.parquet", "adv.parquet", "profiles.parquet", "sessions.parquet")


def prepare_tca_history(
    market_path: Path,
    directory: Path,
    *,
    instrument_id: str,
    cutoffs: dict[str, date],
    identity: dict[str, Any],
) -> Path:
    """Reuse one instrument's history, all fold profiles, and original ADV20."""
    expected = {
        **identity,
        "schema_version": "paper-tca-history-v2",
        "instrument_id": instrument_id,
        "market_sha256": file_sha256(market_path),
        "training_cutoffs": {key: value.isoformat() for key, value in sorted(cutoffs.items())},
    }
    if directory.exists():
        verify_artifact(directory, identity=expected, names=HISTORY_FILES)
        return directory
    bars = pd.read_parquet(market_path)
    if bars.empty or not bars["instrument_id"].astype(str).eq(instrument_id).all():
        raise ValueError("TCA compact history instrument mismatch.")
    # Aggregate in original corpus order before sorting derived replay rows.
    adv = causal_adv20(bars)
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


def causal_adv20(bars: pd.DataFrame) -> pd.DataFrame:
    frame = bars.copy()
    frame["session_date"] = (
        pd.to_datetime(frame["timestamp"]).dt.tz_convert("America/New_York").dt.date
    )
    daily = (
        frame.groupby(["instrument_id", "session_date"], sort=True, as_index=False)["volume"]
        .sum()
        .sort_values(["instrument_id", "session_date"], kind="stable")
    )
    daily["adv20"] = daily.groupby("instrument_id", sort=False)["volume"].transform(
        lambda values: values.shift(1).rolling(20, min_periods=20).mean()
    )
    return daily.dropna(subset=["adv20"])


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
