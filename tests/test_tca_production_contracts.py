"""Production-path checks for TCA forecast ledgers, workers, and result merges."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import yaml

from execsim.data.paper.manifests import file_sha256, read_json
from execsim.ml.paper.evaluation_artifacts import (
    forecast_metric_frame,
    merge_result_shards,
    publish_frames,
)
from execsim.ml.paper.evaluation_workers import EWMAWork, ewma_ledger_identity, run_ewma_work
from execsim.ml.paper.lightgbm_data import LightGBMFrames
from execsim.ml.paper.tca_inputs import prepare_tca_history, read_tca_date
from execsim.ml.paper.tca_workers import (
    TCAWork,
    _validate_ewma_case,
    preflight_tca_ledgers,
    run_tca_work,
)

_FOLD = "fold-contract"
_INSTRUMENT = "instrument-contract"
_SYMBOL = "TST"


def _market_bars(session_count: int = 23) -> pd.DataFrame:
    """Build full minute sessions with changing daily and intraday volume."""
    frames = []
    sessions = pd.bdate_range("2024-01-02", periods=session_count)
    minute = np.arange(390, dtype=float)
    intraday = (
        1.0 + 0.22 * np.cos(2 * np.pi * minute / 390) + 0.03 * np.sin(2 * np.pi * minute / 15)
    )
    for day_index, session in enumerate(sessions):
        timestamps = pd.date_range(
            session + pd.Timedelta(hours=9, minutes=30),
            periods=390,
            freq="min",
            tz="America/New_York",
        )
        close = 100.0 + day_index * 0.04 + minute * 0.001
        frames.append(
            pd.DataFrame(
                {
                    "instrument_id": _INSTRUMENT,
                    "symbol": _SYMBOL,
                    "timestamp": timestamps,
                    "open": close - 0.01,
                    "high": close + 0.02,
                    "low": close - 0.02,
                    "close": close,
                    "volume": (900.0 + day_index * 12.0) * intraday,
                    "trade_count": 20 + (minute.astype(int) % 7),
                    "vwap": close,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _forecast_inputs(
    bars: pd.DataFrame, *, training_cutoff: date, session_date: date
) -> tuple[LightGBMFrames, np.ndarray, np.ndarray, np.ndarray]:
    """Build an evaluation base and history-only forecasts without fitting a model."""
    timestamps = pd.to_datetime(bars["timestamp"]).dt.tz_convert("America/New_York")
    frame = bars.assign(
        session_date=timestamps.dt.date,
        target_bucket=((timestamps.dt.hour * 60 + timestamps.dt.minute - 570) // 15).astype(
            "int64"
        ),
    )
    training = frame.loc[frame["session_date"] <= training_cutoff]
    target = frame.loc[frame["session_date"] == session_date]
    history_tokens = (
        training.groupby(["session_date", "target_bucket"], sort=True)["volume"]
        .sum()
        .unstack("target_bucket")
    )
    mean_tokens = history_tokens.mean(axis=0).reindex(range(26)).to_numpy(dtype=float)
    target_tokens = target.groupby("target_bucket", sort=True)["volume"].sum().reindex(range(26))

    scale_rows: list[dict[str, Any]] = []
    shape_rows: list[dict[str, Any]] = []
    targets: list[float] = []
    baseline: list[float] = []
    origins = np.arange(4, 26, dtype=np.int64)
    for origin in origins:
        sample_id = f"{session_date.isoformat()}-{int(origin):02d}"
        actual_remaining = float(target_tokens.iloc[int(origin) :].sum())
        train_remaining = history_tokens.iloc[:, int(origin) :].sum(axis=1)
        baseline_remaining = float(train_remaining.mean())
        scale_rows.append(
            {
                "sample_id": sample_id,
                "fold_id": _FOLD,
                "instrument_id": _INSTRUMENT,
                "symbol": _SYMBOL,
                "session_date": session_date,
                "as_of": int(origin),
                "training_cutoff": training_cutoff,
                "market_information_as_of": training_cutoff,
                "feature_history_end": training_cutoff,
                "baseline_remaining_volume": baseline_remaining,
            }
        )
        targets.append(actual_remaining)
        baseline.append(baseline_remaining)
        shape_tokens = np.arange(int(origin), 26, dtype=np.int64)
        for bucket in shape_tokens:
            shape_rows.append(
                {
                    "sample_id": sample_id,
                    "case_id": sample_id,
                    "instrument_id": _INSTRUMENT,
                    "target_bucket": int(bucket),
                }
            )

    scale = pd.DataFrame(scale_rows)
    shape = pd.DataFrame(shape_rows)
    # The target buckets remain one contiguous sequence per sample; use held-out
    # bars only for evaluation labels, never for the synthetic forecast values.
    actual_shape = np.concatenate(
        [
            target_tokens.iloc[int(origin) :].to_numpy(dtype=float)
            / float(target_tokens.iloc[int(origin) :].sum())
            for origin in origins
        ]
    )
    base = LightGBMFrames(scale, np.asarray(targets), shape, actual_shape)
    training_shape = np.concatenate(
        [mean_tokens[int(origin) :] / float(mean_tokens[int(origin) :].sum()) for origin in origins]
    )
    return base, origins, np.asarray(baseline), training_shape


def _method_variants(
    include_representation_methods: bool,
) -> list[tuple[str, int | None, float, float]]:
    variants: list[tuple[str, int | None, float, float]] = [("raw", None, 1.0, 1.0)]
    if include_representation_methods:
        variants.append(("untrained_neural", None, 0.97, 0.85))
        for method, multiplier in (("dense", 1.015), ("sparse", 0.985)):
            for seed in (13, 29, 47):
                seed_adjustment = 1.0 + (seed - 29) * 0.0005
                variants.append((method, seed, multiplier * seed_adjustment, 0.92 + seed * 0.0004))
    return variants


def build_published_tca_rows(
    tmp_path: Path,
    *,
    unavailable_ewma: bool = False,
    include_representation_methods: bool = False,
) -> dict[str, pd.DataFrame]:
    """Run the production TCA path and return its published shard and merge rows.

    All source bars and prediction ledgers are deterministic synthetic fixtures.
    Forecast values use prior TRAIN history; the real preflight, EWMA producer,
    TCA work function, replay, and shard merger execute without mocks.
    """
    config = yaml.safe_load(Path("configs/paper/sparse_jepa_v2/tca.yaml").read_text())
    config.update(universe_size=1, sensitivity_universe_size=1)
    bars = _market_bars()
    all_dates = sorted(pd.to_datetime(bars["timestamp"]).dt.date.unique())
    session_date = all_dates[-1]
    training_cutoff = all_dates[-2]
    source = tmp_path / "market.parquet"
    bars.to_parquet(source, index=False)
    history = prepare_tca_history(
        source,
        tmp_path / "history",
        instrument_id=_INSTRUMENT,
        cutoffs={_FOLD: training_cutoff},
        identity={"paper_config_hash": "tca-production-fixture"},
    )
    target_bars, adv = read_tca_date({_INSTRUMENT: history}, session_date)
    profiles = pd.read_parquet(history / "profiles.parquet")
    profiles = profiles.loc[profiles["fold_id"].eq(_FOLD)].reset_index(drop=True)
    universe = pd.DataFrame({"rank": [1], "instrument_id": [_INSTRUMENT]})
    input_directory = tmp_path / "tca-input"
    input_identity = {"fold_id": _FOLD, "session_date": session_date.isoformat()}
    publish_frames(
        input_directory,
        identity=input_identity,
        frames={
            "bars.parquet": target_bars,
            "adv.parquet": adv,
            "profiles.parquet": profiles,
            "universe.parquet": universe,
        },
    )

    base, origins, baseline, history_shape = _forecast_inputs(
        bars, training_cutoff=training_cutoff, session_date=session_date
    )
    base_directory = tmp_path / "evaluation-base"
    base_scale = base.scale.assign(__evaluation_target=base.scale_target)
    base_shape = base.shape.assign(__evaluation_target=base.shape_target)
    publish_frames(
        base_directory,
        identity={"fixture": "synthetic-test-base"},
        frames={
            "scale-base.parquet": base_scale,
            "shape-base.parquet": base_shape,
        },
    )

    variants = _method_variants(include_representation_methods)
    ledger_records: list[tuple[str, int | None, Path, dict[str, Any]]] = []
    freeze_path = tmp_path / "parameter-freeze.json"
    freeze_path.write_text('{"fixture": "frozen"}', encoding="utf-8")
    for variant_index, (method, seed, total_factor, shape_power) in enumerate(variants):
        model_manifest = tmp_path / f"model-{method}-{seed or 'shared'}.json"
        model_manifest.write_text(
            f'{{"fixture": "synthetic", "variant": {variant_index}}}', encoding="utf-8"
        )
        identity = {
            "schema_version": "paper-forecast-ledger-v2",
            "fold_id": _FOLD,
            "method": method,
            "seed": seed,
            "paper_config_hash": "tca-production-fixture",
            "source_commit": "synthetic-source-commit",
            "source_tree": "synthetic-source-tree",
            "parameter_freeze_sha256": file_sha256(freeze_path),
            "model_manifest_sha256": file_sha256(model_manifest),
            "base_manifest_sha256": file_sha256(base_directory / "manifest.json"),
            "embedding_sha256": None,
        }
        totals = baseline * total_factor
        predicted_chunks = []
        offset = 0
        for origin in origins:
            size = 26 - int(origin)
            historical_shape = history_shape[offset : offset + size]
            powered_shape = np.power(historical_shape, shape_power)
            predicted_chunks.append(powered_shape / powered_shape.sum())
            offset += size
        predicted_shape_values = np.concatenate(predicted_chunks)
        predicted_shape = base.shape.loc[:, ["case_id", "target_bucket"]].copy()
        predicted_shape["conditional_share"] = predicted_shape_values
        metrics = forecast_metric_frame(
            base,
            totals,
            predicted_shape,
            fold_id=_FOLD,
            method=method,
            seed=seed,
        )
        ledger_scale = base.scale.loc[
            :,
            [
                "sample_id",
                "fold_id",
                "instrument_id",
                "symbol",
                "session_date",
                "as_of",
                "training_cutoff",
                "market_information_as_of",
                "feature_history_end",
            ],
        ].copy()
        ledger_scale["predicted_remaining_volume"] = totals
        ledger = tmp_path / "forecasts" / method / str(seed or "shared")
        publish_frames(
            ledger,
            identity=identity,
            frames={
                "scale.parquet": ledger_scale,
                "shape.parquet": predicted_shape.sort_values(
                    ["case_id", "target_bucket"], kind="stable"
                ).reset_index(drop=True),
                "metrics.parquet": metrics,
            },
        )
        ledger_records.append((method, seed, ledger, identity))

    ewma_market = bars
    if unavailable_ewma:
        ewma_market = bars.loc[pd.to_datetime(bars["timestamp"]).dt.date == session_date].copy()
    ewma_market_path = tmp_path / "ewma-market.parquet"
    ewma_market.to_parquet(ewma_market_path, index=False)
    ewma_work = EWMAWork(
        base_directory,
        ewma_market_path,
        tmp_path / "ewma-ledger",
        _INSTRUMENT,
        {"fold_id": _FOLD, "paper_config_hash": "tca-production-fixture"},
    )
    ewma_directory = run_ewma_work(ewma_work)
    ewma_record = ewma_ledger_identity(ewma_work)
    ewma_records = {_INSTRUMENT: (ewma_directory, ewma_record)}

    eligible_cases = {session_date: (_INSTRUMENT,)}
    preflight_tca_ledgers(
        ledger_records=tuple(ledger_records),
        ewma_records=ewma_records,
        eligible_cases=eligible_cases,
        training_cutoff=training_cutoff,
        tca_config=config,
    )
    sequence_manifest = tmp_path / "sequence-manifest.json"
    sequence_manifest.write_text('{"fixture": "sequence"}', encoding="utf-8")
    work = TCAWork(
        input_directory=input_directory,
        output_directory=tmp_path / "tca-shard",
        ledger_records=tuple(ledger_records),
        ewma_records=ewma_records,
        training_cutoff=training_cutoff,
        sequence_hash=file_sha256(sequence_manifest),
        tca_config=config,
        identity=input_identity,
    )
    shard_directory = run_tca_work(work)
    shard_main_path = shard_directory / "main.parquet"
    shard_sensitivity_path = shard_directory / "sensitivity.parquet"
    shard_main = pd.read_parquet(shard_main_path)
    shard_sensitivity = pd.read_parquet(shard_sensitivity_path)

    merge_identity = {
        "paper_config_hash": "tca-production-fixture",
        "fold_id": _FOLD,
        "session_date": session_date.isoformat(),
    }
    main_path = tmp_path / "merged-main.parquet"
    main_receipt = merge_result_shards(
        main_path,
        sources={
            f"{_FOLD}/{session_date.isoformat()}/main": (
                shard_main_path,
                file_sha256(shard_main_path),
            )
        },
        keys=["fold_id", "date", "instrument_id", "method", "order_fraction_adv20"],
        identity=merge_identity,
        schema_version="paper-tca-main-fixture-v1",
    )
    sensitivity_path = tmp_path / "merged-sensitivity.parquet"
    sensitivity_receipt = merge_result_shards(
        sensitivity_path,
        sources={
            f"{_FOLD}/{session_date.isoformat()}/sensitivity": (
                shard_sensitivity_path,
                file_sha256(shard_sensitivity_path),
            )
        },
        keys=["fold_id", "date", "instrument_id", "method", "order_fraction_adv20"],
        identity=merge_identity,
        schema_version="paper-tca-sensitivity-fixture-v1",
    )
    assert main_receipt["rows"] == len(shard_main)
    assert sensitivity_receipt["rows"] == len(shard_sensitivity)
    assert read_json(shard_directory / "manifest.json")["identity"]["ledgers"] == [
        record[3] for record in ledger_records
    ]
    assert read_json(shard_directory / "manifest.json")["identity"]["ewma_ledgers"] == {
        _INSTRUMENT: ewma_record
    }
    return {
        "shard_main": shard_main,
        "shard_sensitivity": shard_sensitivity,
        "main": pd.read_parquet(main_path),
        "sensitivity": pd.read_parquet(sensitivity_path),
    }


def _corrupt_learned_ledger(
    directory: Path, *, key: str, value: Any
) -> tuple[Path, dict[str, Any], date, date, dict[str, Any]]:
    session_date = date(2024, 2, 1)
    training_cutoff = date(2024, 1, 31)
    identity = {
        "schema_version": "paper-forecast-ledger-v2",
        "fold_id": "fold-invalid-key",
        "method": "raw",
        "seed": None,
        "paper_config_hash": "fixture",
        "source_commit": "fixture",
        "source_tree": "fixture",
    }
    origins = np.arange(4, 7, dtype=np.int64)
    scale = pd.DataFrame(
        {
            "sample_id": [f"sample-{origin}" for origin in origins],
            "fold_id": identity["fold_id"],
            "instrument_id": "A",
            "symbol": "A",
            "session_date": session_date,
            "as_of": origins,
            "training_cutoff": training_cutoff,
        }
    )
    shape_rows = [
        {
            "case_id": f"sample-{origin}",
            "target_bucket": bucket,
            "conditional_share": 1.0 / (26 - origin),
        }
        for origin in origins
        for bucket in range(int(origin), 26)
    ]
    shape = pd.DataFrame(shape_rows)
    if key == "as_of":
        if value == "4":
            scale["as_of"] = scale["as_of"].astype(str)
            scale.loc[0, "as_of"] = value
        else:
            scale["as_of"] = scale["as_of"].astype(float)
            scale.loc[0, "as_of"] = value
    else:
        if value == "4":
            shape["target_bucket"] = shape["target_bucket"].astype(str)
            shape.loc[0, "target_bucket"] = value
        else:
            shape["target_bucket"] = shape["target_bucket"].astype(float)
            shape.loc[0, "target_bucket"] = value
    publish_frames(
        directory,
        identity=identity,
        frames={
            "scale.parquet": scale,
            "shape.parquet": shape,
            "metrics.parquet": pd.DataFrame({"sample_id": scale["sample_id"]}),
        },
    )
    config = {"window": ["10:30", "10:45"]}
    return directory, identity, session_date, training_cutoff, config


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("as_of", 4.5),
        ("as_of", "4"),
        ("as_of", None),
        ("target_bucket", 4.5),
        ("target_bucket", "4"),
        ("target_bucket", None),
    ],
)
def test_learned_preflight_rejects_noninteger_keys_before_worker_stage(tmp_path, key, value):
    directory, identity, session_date, training_cutoff, config = _corrupt_learned_ledger(
        tmp_path / f"ledger-{key}-{value}", key=key, value=value
    )
    worker_started = False
    with pytest.raises(ValueError, match=r"non-null integer learned key"):
        preflight_tca_ledgers(
            ledger_records=(("raw", None, directory, identity),),
            ewma_records={},
            eligible_cases={session_date: ("A",)},
            training_cutoff=training_cutoff,
            tca_config=config,
        )
        worker_started = True
    assert not worker_started


@pytest.mark.parametrize("value", [4.5, "4", None])
def test_ewma_preflight_validator_rejects_noninteger_as_of_before_coercion(value):
    scale = pd.DataFrame(
        {
            "sample_id": ["sample"],
            "instrument_id": ["A"],
            "symbol": ["A"],
            "session_date": [date(2024, 2, 1)],
            "as_of": [value],
        }
    )
    if value == "4":
        scale["as_of"] = scale["as_of"].astype("string")
    available = pd.DataFrame(columns=["sample_id", "generated_at", "end_token"])
    unavailable = pd.DataFrame(
        columns=["sample_id", "generated_at", "end_token", "status", "session_date"]
    )
    with pytest.raises(ValueError, match=r"non-null integer EWMA key"):
        _validate_ewma_case(
            Path("unused"),
            instrument_id="A",
            session_date=date(2024, 2, 1),
            fold_id="fold",
            origins=(4,),
            scale_frame=scale,
            available_frame=available,
            unavailable_frame=unavailable,
        )


def test_real_tca_production_chain_publishes_and_merges_raw_rows(tmp_path):
    result = build_published_tca_rows(tmp_path)
    assert set(result) == {"shard_main", "shard_sensitivity", "main", "sensitivity"}
    assert set(result["main"]["method"]) == {"ewma", "lightgbm_raw"}
    assert result["main"]["status"].eq("AVAILABLE").all()
    assert len(result["main"]) == 2
    assert len(result["sensitivity"]) == 4
    pd.testing.assert_frame_equal(
        result["shard_main"].sort_values("method").reset_index(drop=True),
        result["main"].sort_values("method").reset_index(drop=True),
    )


def test_real_ewma_producer_preserves_unavailable_case_through_tca_and_merge(tmp_path):
    result = build_published_tca_rows(tmp_path, unavailable_ewma=True)
    main = result["main"].set_index("method")
    assert main.loc["ewma", "status"] == "EWMA_UNAVAILABLE"
    assert main.loc["lightgbm_raw", "status"] == "AVAILABLE"
    assert pd.isna(main.loc["ewma", "completion_rate"])
    assert result["shard_main"]["status"].eq(result["main"]["status"].to_numpy()).all()


def test_report_helper_publishes_all_candidate_method_rows_from_real_tca(tmp_path):
    result = build_published_tca_rows(tmp_path, include_representation_methods=True)
    expected = {
        "ewma",
        "lightgbm_raw",
        "raw_untrained_neural",
        *(
            f"raw_{geometry}_jepa_seed_{seed}"
            for geometry in ("dense", "sparse")
            for seed in (13, 29, 47)
        ),
    }
    assert set(result["main"]["method"]) == expected
    assert len(result["main"]) == len(expected)
    assert (
        not result["main"]
        .duplicated(["fold_id", "date", "instrument_id", "method", "order_fraction_adv20"])
        .any()
    )
