"""Behavioral parity checks for indexed learned-ledger TCA preflight."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from datetime import date, time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from execsim.ml.paper import tca_workers
from execsim.ml.paper.evaluation_artifacts import publish_frames, verify_artifact
from execsim.ml.paper.tca_workers import (
    LEARNED_LEDGER_FILES,
    preflight_tca_ledgers,
)

_SESSION_DATES = (date(2040, 1, 2), date(2040, 1, 3))


@pytest.mark.parametrize("origins_count", [1, 2, 20, 22])
def test_benchmark_window_matches_all_generated_origins(tmp_path, origins_count):
    from scripts import benchmark_tca_preflight

    _, _, _, config = benchmark_tca_preflight._synthetic_ledgers(
        tmp_path,
        dates_count=1,
        origins_count=origins_count,
        learned_ledger_count=1,
        publish_frames=publish_frames,
    )
    assert tca_workers._tca_as_of_origins(config) == tuple(range(4, 4 + origins_count))


@pytest.mark.parametrize("revision", ["HEAD", "--help", "-x", "a" * 39, "z" * 40])
def test_benchmark_rejects_non_commit_arguments_before_git(revision, monkeypatch):
    from scripts import benchmark_tca_preflight

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid baseline reached subprocess execution")

    monkeypatch.setattr(benchmark_tca_preflight.subprocess, "run", unexpected)
    with pytest.raises(ValueError, match="full lowercase Git commit SHA"):
        benchmark_tca_preflight._load_legacy_preflight(tca_workers, revision)


_INSTRUMENT_ID = "SYNTH"
_FOLD_ID = "fold-synthetic"
_TRAINING_CUTOFF = date(2039, 12, 31)
_ORIGINS = tuple(range(4, 26))
_TCA_CONFIG = {"window": ["10:30", "16:00"]}


def _synthetic_frames() -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    """Build complete, deterministic two-date forecast ledgers."""
    learned_scale_rows: list[dict[str, Any]] = []
    learned_shape_rows: list[dict[str, Any]] = []
    ewma_scale_rows: list[dict[str, Any]] = []
    minute_rows: list[dict[str, Any]] = []

    for session_date in _SESSION_DATES:
        opened = pd.Timestamp.combine(session_date, time(9, 30)).tz_localize("America/New_York")
        for origin in _ORIGINS:
            sample_id = f"{session_date.isoformat()}-{_INSTRUMENT_ID}-{origin:02d}"
            learned_scale_rows.append(
                {
                    "sample_id": sample_id,
                    "fold_id": _FOLD_ID,
                    "instrument_id": _INSTRUMENT_ID,
                    "symbol": "SYNTH",
                    "session_date": session_date.isoformat(),
                    "as_of": origin,
                    "training_cutoff": _TRAINING_CUTOFF.isoformat(),
                }
            )
            ewma_scale_rows.append(
                {
                    "sample_id": sample_id,
                    "instrument_id": _INSTRUMENT_ID,
                    "symbol": "SYNTH",
                    "session_date": session_date.isoformat(),
                    "as_of": origin,
                }
            )
            buckets = np.arange(origin, 26)
            learned_shape_rows.extend(
                {
                    "case_id": sample_id,
                    "target_bucket": int(bucket),
                    "conditional_share": 1.0 / len(buckets),
                }
                for bucket in buckets
            )
            for offset in range(15):
                minute_rows.append(
                    {
                        "sample_id": sample_id,
                        "generated_at": opened + pd.Timedelta(minutes=15 * origin + offset),
                        "end_token": 24,
                    }
                )

    learned_scale = pd.DataFrame(learned_scale_rows)
    learned_shape = (
        pd.DataFrame(learned_shape_rows).sample(frac=1, random_state=41).reset_index(drop=True)
    )
    learned_identity = {
        "fold_id": _FOLD_ID,
        "method": "raw",
        "seed": None,
        "parameter_freeze_sha256": "synthetic-freeze",
        "model_manifest_sha256": "synthetic-model",
        "source_commit": "synthetic-source",
        "source_tree": "synthetic-tree",
        "paper_config_hash": "synthetic-config",
        "base_manifest_sha256": "synthetic-base",
        "embedding_sha256": None,
    }
    ewma_identity = {
        "fold_id": _FOLD_ID,
        "instrument_id": _INSTRUMENT_ID,
        "fixture": "synthetic-ewma",
    }
    ewma_unavailable = pd.DataFrame(
        {
            "sample_id": pd.Series(dtype="string"),
            "generated_at": pd.Series(dtype="datetime64[ns, America/New_York]"),
            "end_token": pd.Series(dtype="int64"),
            "status": pd.Series(dtype="string"),
            "session_date": pd.Series(dtype="string"),
            "reason": pd.Series(dtype="string"),
        }
    )
    return (
        {"learned": learned_identity, "ewma": ewma_identity},
        {
            "learned_scale": learned_scale,
            "learned_shape": learned_shape,
            "learned_metrics": pd.DataFrame({"metric": pd.Series(dtype="string")}),
            "ewma_scale": pd.DataFrame(ewma_scale_rows),
            "ewma_shape": pd.DataFrame(
                {"case_id": pd.Series(dtype="string"), "target_bucket": pd.Series(dtype="int64")}
            ),
            "ewma_metrics": pd.DataFrame({"metric": pd.Series(dtype="string")}),
            "ewma_minutes": pd.DataFrame(minute_rows),
            "ewma_unavailable": ewma_unavailable,
        },
    )


def _publish_case(
    root: Path, corruption: str | None = None
) -> tuple[Path, dict[str, Any], Path, dict[str, Any]]:
    identities, frames = _synthetic_frames()
    learned_scale = frames["learned_scale"]
    learned_shape = frames["learned_shape"]
    target_date = _SESSION_DATES[-1].isoformat()
    target_rows = learned_scale["session_date"].eq(target_date)
    target_sample = learned_scale.loc[target_rows, "sample_id"].iloc[0]

    if corruption == "duplicate_scale":
        learned_scale = pd.concat(
            [learned_scale, learned_scale.loc[[learned_scale.index[target_rows][0]]]],
            ignore_index=True,
        )
    elif corruption == "missing_origin":
        remove_index = learned_scale.index[target_rows][-1]
        learned_scale = learned_scale.drop(index=remove_index).reset_index(drop=True)
    elif corruption == "invalid_date":
        learned_scale.loc[learned_scale.index[target_rows][0], "session_date"] = "not-a-date"
    elif corruption == "cutoff_mismatch":
        learned_scale.loc[learned_scale.index[target_rows][0], "training_cutoff"] = "2039-12-30"
    elif corruption == "instrument_mismatch":
        learned_scale.loc[target_rows, "instrument_id"] = "OTHER"
    elif corruption == "fold_mismatch":
        learned_scale.loc[learned_scale.index[target_rows][0], "fold_id"] = "fold-other"
    elif corruption == "missing_shape_bucket":
        row = learned_shape.index[
            learned_shape["case_id"].eq(target_sample) & learned_shape["target_bucket"].eq(25)
        ][0]
        learned_shape = learned_shape.drop(index=row).reset_index(drop=True)
    elif corruption == "duplicate_shape_bucket":
        row = learned_shape.loc[
            learned_shape["case_id"].eq(target_sample) & learned_shape["target_bucket"].eq(25)
        ]
        learned_shape = pd.concat([learned_shape, row], ignore_index=True)
    elif corruption == "invalid_shape_share":
        row = learned_shape.index[learned_shape["case_id"].eq(target_sample)][0]
        learned_shape.loc[row, "conditional_share"] = -0.1
    elif corruption is not None:
        raise AssertionError(f"Unknown synthetic corruption: {corruption}")

    learned = root / "learned"
    publish_frames(
        learned,
        identity=identities["learned"],
        frames={
            "scale.parquet": learned_scale,
            "shape.parquet": learned_shape,
            "metrics.parquet": frames["learned_metrics"],
        },
    )
    ewma = root / "ewma"
    publish_frames(
        ewma,
        identity=identities["ewma"],
        frames={
            "scale.parquet": frames["ewma_scale"],
            "shape.parquet": frames["ewma_shape"],
            "metrics.parquet": frames["ewma_metrics"],
            "minute-forecasts.parquet": frames["ewma_minutes"],
            "unavailable.parquet": frames["ewma_unavailable"],
        },
    )
    return learned, identities["learned"], ewma, identities["ewma"]


def _legacy_learned_preflight(
    *,
    ledger_records: tuple[tuple[str, int | None, Path, dict[str, Any]], ...],
    eligible_cases: dict[date, tuple[str, ...]],
    training_cutoff: date,
    tca_config: dict[str, list[str]],
) -> None:
    """Apply the former whole-instrument-frame filtering before each date check."""
    if not eligible_cases:
        return
    origins = tca_workers._tca_as_of_origins(tca_config)
    eligible_by_instrument: dict[str, list[date]] = {}
    for session_date, instruments in sorted(eligible_cases.items()):
        for instrument_id in sorted(set(instruments)):
            eligible_by_instrument.setdefault(str(instrument_id), []).append(session_date)

    for _, _, directory, identity in ledger_records:
        verify_artifact(directory, identity=identity, names=LEARNED_LEDGER_FILES)
        for instrument_id, session_dates in eligible_by_instrument.items():
            scale = pd.read_parquet(
                directory / "scale.parquet", filters=[("instrument_id", "==", instrument_id)]
            )
            if "sample_id" in scale.columns and not scale.empty:
                sample_ids = scale["sample_id"].astype(str).drop_duplicates().tolist()
                shape = pd.read_parquet(
                    directory / "shape.parquet", filters=[("case_id", "in", sample_ids)]
                )
            else:
                shape = pd.DataFrame(columns=["case_id", "target_bucket", "conditional_share"])
            for session_date in session_dates:
                tca_workers._validate_learned_case(
                    directory,
                    instrument_id=instrument_id,
                    session_date=session_date,
                    fold_id=str(identity["fold_id"]),
                    training_cutoff=training_cutoff,
                    origins=origins,
                    scale_frame=scale,
                    shape_frame=shape,
                )


def _outcome(call: Callable[[], None]) -> tuple[type[BaseException], str] | None:
    try:
        call()
    except Exception as exc:  # The parity assertion intentionally captures failures.
        return type(exc), str(exc)
    return None


@pytest.mark.parametrize(
    "corruption",
    [
        "duplicate_scale",
        "missing_origin",
        "invalid_date",
        "cutoff_mismatch",
        "instrument_mismatch",
        "fold_mismatch",
        "missing_shape_bucket",
        "duplicate_shape_bucket",
        "invalid_shape_share",
    ],
)
def test_indexed_tca_preflight_matches_legacy_fail_closed_behavior(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, corruption: str
) -> None:
    learned, identity, ewma, ewma_identity = _publish_case(tmp_path, corruption)
    records = (("raw", None, learned, identity),)
    eligible_cases = {day: (_INSTRUMENT_ID,) for day in _SESSION_DATES}

    if corruption == "instrument_mismatch":
        # Force both paths to validate the persisted foreign identity instead of
        # relying on Parquet predicate pushdown to remove it from the read.
        read_parquet = pd.read_parquet

        def read_without_instrument_filter(path: str | Path, *args: Any, **kwargs: Any):
            if Path(path).name == "scale.parquet" and kwargs.get("filters"):
                kwargs = {key: value for key, value in kwargs.items() if key != "filters"}
            return read_parquet(path, *args, **kwargs)

        monkeypatch.setattr(pd, "read_parquet", read_without_instrument_filter)

    reference = _outcome(
        lambda: _legacy_learned_preflight(
            ledger_records=records,
            eligible_cases=eligible_cases,
            training_cutoff=_TRAINING_CUTOFF,
            tca_config=_TCA_CONFIG,
        )
    )
    actual = _outcome(
        lambda: preflight_tca_ledgers(
            ledger_records=records,
            ewma_records={_INSTRUMENT_ID: (ewma, ewma_identity)},
            eligible_cases=eligible_cases,
            training_cutoff=_TRAINING_CUTOFF,
            tca_config=_TCA_CONFIG,
        )
    )
    assert actual == reference
    assert actual is not None
    assert actual[0] is ValueError


def test_indexed_tca_preflight_preserves_shape_order_and_stays_pre_worker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    learned, identity, ewma, ewma_identity = _publish_case(tmp_path)
    full_scale = pd.read_parquet(learned / "scale.parquet")
    full_shape = pd.read_parquet(learned / "shape.parquet")
    original_validate = tca_workers._validate_learned_case
    observed_dates: list[date] = []
    launch_attempts: list[str] = []
    parquet_reads: Counter[str] = Counter()
    read_parquet = pd.read_parquet

    def count_reads(path: str | Path, *args: Any, **kwargs: Any) -> pd.DataFrame:
        path = Path(path)
        parquet_reads[f"{path.parent.name}/{path.name}"] += 1
        return read_parquet(path, *args, **kwargs)

    def record_slices(directory: Path, **kwargs: Any) -> None:
        expected_scale = full_scale.loc[
            pd.to_datetime(full_scale["session_date"]).dt.date.eq(kwargs["session_date"])
        ]
        expected_shape = full_shape.loc[
            full_shape["case_id"].astype(str).isin(expected_scale["sample_id"].astype(str))
        ]
        pd.testing.assert_frame_equal(
            kwargs["scale_frame"].reset_index(drop=True),
            expected_scale.reset_index(drop=True),
        )
        pd.testing.assert_frame_equal(
            kwargs["shape_frame"].reset_index(drop=True),
            expected_shape.reset_index(drop=True),
        )
        observed_dates.append(kwargs["session_date"])
        original_validate(directory, **kwargs)

    def unexpected_launch(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        launch_attempts.append("worker-or-provider")
        raise AssertionError("Preflight must finish before any worker or provider starts.")

    monkeypatch.setattr(pd, "read_parquet", count_reads)
    monkeypatch.setattr(tca_workers, "_validate_learned_case", record_slices)
    monkeypatch.setattr(tca_workers, "run_tca_work", unexpected_launch)
    monkeypatch.setattr(tca_workers, "run_tca_workers", unexpected_launch)
    monkeypatch.setattr(tca_workers, "PaperForecastLedgerProvider", unexpected_launch)
    monkeypatch.setattr(tca_workers, "EWMAForecastLedgerProvider", unexpected_launch)

    preflight_tca_ledgers(
        ledger_records=(("raw", None, learned, identity),),
        ewma_records={_INSTRUMENT_ID: (ewma, ewma_identity)},
        eligible_cases={day: (_INSTRUMENT_ID,) for day in _SESSION_DATES},
        training_cutoff=_TRAINING_CUTOFF,
        tca_config=_TCA_CONFIG,
    )

    assert observed_dates == list(_SESSION_DATES)
    assert launch_attempts == []
    assert parquet_reads == Counter(
        {
            "learned/scale.parquet": 1,
            "learned/shape.parquet": 1,
            "ewma/scale.parquet": 1,
            "ewma/minute-forecasts.parquet": 1,
            "ewma/unavailable.parquet": 1,
        }
    )
