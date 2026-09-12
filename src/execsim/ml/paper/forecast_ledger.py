"""Replay verified frozen predictions through the ordinary minute forecast contract."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from execsim.forecasting.historical import HistoricalForecastUnavailable
from execsim.forecasting.models import VolumeForecast
from execsim.ml.paper.evaluation_artifacts import VerifiedArtifact, verify_artifact
from execsim.ml.paper.evaluation_workers import EWMA_FILES
from execsim.ml.paper.forecast_provider import (
    _is_update_boundary,
    _MinuteGrid,
    _truncate_forecast,
)
from execsim.ml.paper.tca import expand_volume_forecast


@dataclass(frozen=True)
class ForecastLedgerDate:
    """One verified date slice shared as input, never as mutable provider state."""

    artifact: VerifiedArtifact
    session_date: date
    scale: pd.DataFrame
    shape: pd.DataFrame

    @classmethod
    def read(
        cls, artifact: VerifiedArtifact, session_date: date, instruments: Sequence[str]
    ) -> ForecastLedgerDate:
        artifact.check(artifact.directory, artifact.identity)
        scale = pd.read_parquet(
            artifact.directory / "scale.parquet",
            filters=[("instrument_id", "in", list(instruments))],
        )
        scale = scale.loc[pd.to_datetime(scale["session_date"]).dt.date == session_date].copy()
        shape = pd.read_parquet(
            artifact.directory / "shape.parquet",
            filters=[("case_id", "in", scale["sample_id"].tolist())],
        )
        return cls(artifact, session_date, scale, shape)


class PaperForecastLedgerProvider:
    """Serve one instrument/session from checksum-bound predictions without model calls."""

    def __init__(
        self,
        directory: Path,
        *,
        expected_identity: Mapping[str, Any],
        instrument_id: str,
        session_date: date,
        within_token_profile: np.ndarray,
        training_cutoff: date,
        sequence_hash: str,
        verified_artifact: VerifiedArtifact | None = None,
        date_slice: ForecastLedgerDate | None = None,
    ) -> None:
        required = {
            "fold_id",
            "method",
            "seed",
            "parameter_freeze_sha256",
            "model_manifest_sha256",
            "source_commit",
            "source_tree",
        }
        if required.difference(expected_identity):
            raise ValueError("Forecast ledger compatibility identity is incomplete.")
        if verified_artifact is None:
            verify_artifact(
                directory,
                identity=expected_identity,
                names=("scale.parquet", "shape.parquet", "metrics.parquet"),
            )
        else:
            verified_artifact.check(directory, expected_identity)
        if date_slice is None:
            scale = pd.read_parquet(
                directory / "scale.parquet", filters=[("instrument_id", "==", instrument_id)]
            )
        else:
            date_slice.artifact.check(directory, expected_identity)
            if date_slice.session_date != session_date:
                raise ValueError("Forecast ledger date slice belongs to another session.")
            scale = date_slice.scale.loc[date_slice.scale["instrument_id"] == instrument_id]
        scale = scale.loc[pd.to_datetime(scale["session_date"]).dt.date == session_date].copy()
        if (
            scale.empty
            or scale["sample_id"].duplicated().any()
            or scale["as_of"].duplicated().any()
        ):
            raise ValueError("Forecast ledger session sample identity is missing or duplicated.")
        if not scale["fold_id"].eq(expected_identity["fold_id"]).all():
            raise ValueError("Forecast ledger row belongs to the wrong fold.")
        if not (pd.to_datetime(scale["training_cutoff"]).dt.date == training_cutoff).all():
            raise ValueError("Forecast ledger training cutoff mismatch.")
        if training_cutoff >= session_date:
            raise ValueError("Forecast ledger training cutoff must precede the session.")
        if scale["symbol"].nunique() != 1:
            raise ValueError("Forecast ledger session symbol is ambiguous.")
        if date_slice is None:
            shape = pd.read_parquet(
                directory / "shape.parquet",
                filters=[("case_id", "in", scale["sample_id"].tolist())],
            )
        else:
            shape = date_slice.shape.loc[
                date_slice.shape["case_id"].isin(scale["sample_id"])
            ].copy()
        if shape.duplicated(["case_id", "target_bucket"]).any():
            raise ValueError("Forecast ledger contains duplicate future buckets.")
        profile = np.asarray(within_token_profile, dtype=float)
        if (
            profile.shape != (15,)
            or not np.isfinite(profile).all()
            or (profile < 0).any()
            or not np.isclose(profile.sum(), 1)
        ):
            raise ValueError("TRAIN-only within-token profile must be a normalized 15-vector.")
        self.scale = scale.set_index("as_of")
        self.shapes = shape.set_index("case_id").sort_values("target_bucket", kind="stable")
        self.symbol = str(scale["symbol"].iloc[0])
        self.session_date = session_date
        self.profile = profile.copy()
        self.training_cutoff = training_cutoff
        self.sequence_hash = sequence_hash
        self._provider_id = f"{expected_identity['method']}-{expected_identity['seed'] or 'shared'}"
        self._latest: VolumeForecast | None = None
        self._latest_grid: _MinuteGrid | None = None

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
        """Read the exact as-of row; between boundaries truncate the last issued forecast."""
        del observations
        requested = tuple(pd.Timestamp(value) for value in bucket_timestamps)
        if symbol != self.symbol or session_date != self.session_date:
            raise ValueError("Forecast ledger instrument/session mismatch.")
        if not requested or requested[0] < generated_at:
            raise ValueError("Forecast request must contain non-past minute buckets.")
        if _is_update_boundary(generated_at):
            local = generated_at.tz_convert("America/New_York")
            if local.date() != session_date:
                raise ValueError("Forecast ledger as-of date mismatch.")
            origin = (local.hour * 60 + local.minute - 570) // 15
            if origin not in self.scale.index:
                raise ValueError("Forecast ledger has no matching causal as-of sample.")
            row = self.scale.loc[origin]
            sample_id = row["sample_id"]
            if sample_id not in self.shapes.index:
                raise ValueError("Forecast ledger has no matching sample shape.")
            shape = self.shapes.loc[[sample_id]]
            if not np.array_equal(shape["target_bucket"].to_numpy(), np.arange(origin, 26)):
                raise ValueError("Forecast ledger future bucket grid mismatch.")
            shares = shape["conditional_share"].to_numpy(dtype=float)
            if (
                not np.isfinite(shares).all()
                or (shares < 0).any()
                or not np.isclose(shares.sum(), 1)
            ):
                raise ValueError("Forecast ledger conditional shares are invalid.")
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
                expected_remaining_volume=float(row["predicted_remaining_volume"]),
                conditional_token_shape=shares,
                within_token_profile=self.profile,
                training_cutoff=self.training_cutoff,
                manifest_hash=self.sequence_hash,
                forecaster_id=self.provider_id,
            )
            self._latest = fresh
            self._latest_grid = minute_grid
        if self._latest is None:
            raise ValueError("A between-boundary request has no prior causal model forecast.")
        return _truncate_forecast(
            self._latest, requested, generated_at, minute_grid=self._latest_grid
        )


class EWMAForecastLedgerProvider:
    """Replay exact minute/window EWMA requests, including intra-token TCA traces."""

    def __init__(
        self,
        directory: Path,
        *,
        expected_identity: Mapping[str, Any],
        symbol: str,
        session_date: date,
        verified_artifact: VerifiedArtifact | None = None,
    ) -> None:
        names = EWMA_FILES
        if expected_identity.get("schema_version") != "paper-ewma-ledger-v4":
            raise ValueError("EWMA ledger requires exact minute-window availability schema v4.")
        if verified_artifact is None:
            verify_artifact(directory, identity=expected_identity, names=names)
        else:
            verified_artifact.check(directory, expected_identity)
        opened = pd.Timestamp.combine(session_date, pd.Timestamp("09:30").time()).tz_localize(
            "America/New_York"
        )
        rows = pd.read_parquet(
            directory / "minute-forecasts.parquet",
            filters=[
                ("generated_at", ">=", opened),
                ("generated_at", "<", opened + pd.Timedelta(days=1)),
                ("end_token", "==", 24),
            ],
        )
        scale = pd.read_parquet(directory / "scale.parquet")
        samples = scale.loc[
            (pd.to_datetime(scale["session_date"]).dt.date == session_date)
            & (scale["symbol"] == symbol)
        ]
        unavailable = pd.read_parquet(
            directory / "unavailable.parquet",
            filters=[("session_date", "==", str(session_date)), ("end_token", "==", 24)],
        )
        if (
            samples.empty
            or not rows["sample_id"].isin(samples["sample_id"]).all()
            or not unavailable["sample_id"].isin(samples["sample_id"]).all()
        ):
            raise ValueError("EWMA ledger symbol/session/sample identity mismatch.")
        if (
            unavailable["generated_at"].duplicated().any()
            or not unavailable["status"].eq("EWMA_UNAVAILABLE").all()
        ):
            raise ValueError("EWMA ledger has invalid availability identities.")
        if rows["generated_at"].isin(unavailable["generated_at"]).any():
            raise ValueError("EWMA request is both available and unavailable.")
        if rows["generated_at"].duplicated().any():
            raise ValueError("EWMA ledger duplicates a minute as-of identity.")
        self.rows = rows.set_index("generated_at")
        self.unavailable = unavailable.set_index("generated_at")
        self.symbol = symbol
        self.session_date = session_date
        self.market_hash = str(expected_identity["market_sha256"])

    @property
    def provider_id(self) -> str:
        return "historical-ewma-symbol-v1"

    def forecast(
        self,
        *,
        symbol: str,
        session_date: date,
        generated_at: pd.Timestamp,
        bucket_timestamps: Sequence[pd.Timestamp],
        observations: pd.DataFrame | None = None,
    ) -> VolumeForecast:
        """Return only the stored exact request; never substitute a different horizon."""
        del observations
        if (
            symbol != self.symbol
            or session_date != self.session_date
            or (generated_at not in self.rows.index and generated_at not in self.unavailable.index)
        ):
            raise ValueError("EWMA ledger has no matching instrument/session/as-of request.")
        requested = tuple(pd.Timestamp(value) for value in bucket_timestamps)
        expected = tuple(
            pd.date_range(
                generated_at,
                pd.Timestamp.combine(session_date, pd.Timestamp("15:29").time()).tz_localize(
                    "America/New_York"
                ),
                freq="min",
            )
        )
        if requested != expected:
            raise ValueError("EWMA ledger exact requested window mismatch.")
        if generated_at in self.unavailable.index:
            raise HistoricalForecastUnavailable(str(self.unavailable.loc[generated_at, "reason"]))
        row = self.rows.loc[generated_at]
        return VolumeForecast(
            symbol=symbol,
            session_date=session_date,
            generated_at=generated_at,
            first_forecast_bucket=requested[0],
            bucket_timestamps=requested,
            expected_volumes=tuple(float(value) for value in row["expected_volumes"]),
            normalized_shares=tuple(float(value) for value in row["normalized_shares"]),
            expected_remaining_volume=float(row["expected_remaining_volume"]),
            forecaster_id=self.provider_id,
            feature_schema_version="volume-profile-v1",
            training_data_cutoff=pd.Timestamp(row["training_data_cutoff"]).date(),
            data_manifest_hash=self.market_hash,
            warnings=tuple(row["warnings"]),
        )
