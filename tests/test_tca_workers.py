from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from execsim.forecasting import HistoricalProfileForecaster
from execsim.ml.paper.evaluation_artifacts import publish_frames
from execsim.ml.paper.evaluation_workers import EWMAWork, ewma_ledger_identity, run_ewma_workers
from execsim.ml.paper.tca import run_historical_tca
from execsim.ml.paper.tca_workers import TCAWork, run_tca_workers


def test_parallel_date_shards_match_serial_tca_sides_costs_and_resume(tmp_path):
    config = yaml.safe_load(Path("configs/paper/sparse_jepa_v2/tca.yaml").read_text())
    config.update(universe_size=2, sensitivity_universe_size=2)
    universe = pd.DataFrame({"rank": [1, 2], "instrument_id": ["A", "B"]})
    profiles = pd.DataFrame({"instrument_id": ["A", "B"], "profile": [np.full(15, 1 / 15)] * 2})
    all_bars = []
    for day in (2, 3, 4):
        for instrument in ("A", "B"):
            all_bars.append(
                pd.DataFrame(
                    {
                        "instrument_id": instrument,
                        "symbol": instrument,
                        "timestamp": pd.date_range(
                            f"2024-01-0{day} 09:30", periods=390, freq="min", tz="America/New_York"
                        ),
                        "open": 100.0,
                        "high": 100.1,
                        "low": 99.9,
                        "close": 100.0,
                        "volume": 1000.0,
                        "trade_count": 10,
                        "vwap": 100.0,
                    }
                )
            )
    bars = pd.concat(all_bars, ignore_index=True)
    market = {}
    for instrument in ("A", "B"):
        path = tmp_path / f"history-{instrument}.parquet"
        bars.loc[bars["instrument_id"] == instrument, ["symbol", "timestamp", "volume"]].to_parquet(
            path, index=False
        )
        market[instrument] = path
    scale = pd.DataFrame(
        [
            (f"{instrument}-{day}-{origin}", instrument, instrument, f"2024-01-0{day}", origin)
            for instrument in ("A", "B")
            for day in (3, 4)
            for origin in range(4, 26)
        ],
        columns=["sample_id", "instrument_id", "symbol", "session_date", "as_of"],
    )
    scale["baseline_remaining_volume"] = (26 - scale["as_of"]) * 15000.0
    scale["__evaluation_target"] = scale["baseline_remaining_volume"]
    shape = scale.loc[scale.index.repeat(26 - scale["as_of"])].reset_index(drop=True)
    shape["case_id"] = shape["sample_id"]
    shape["target_bucket"] = np.concatenate([np.arange(origin, 26) for origin in scale["as_of"]])
    shape["__evaluation_target"] = 1.0 / (26 - shape["as_of"])
    base = tmp_path / "base"
    publish_frames(
        base,
        identity={"fixture": True},
        frames={
            "scale-base.parquet": scale,
            "shape-base.parquet": shape,
        },
    )
    baseline_work = [
        EWMAWork(
            base,
            market[instrument],
            tmp_path / f"ewma-{instrument}",
            instrument,
            {"fold_id": "fixture"},
        )
        for instrument in ("A", "B")
    ]
    run_ewma_workers(baseline_work, workers=1)
    ewma_records = {
        item.instrument_id: (item.output_directory, ewma_ledger_identity(item))
        for item in baseline_work
    }
    tasks, adv_rows = [], []
    for day in (date(2024, 1, 3), date(2024, 1, 4)):
        selected = bars.loc[bars["timestamp"].dt.date == day]
        adv = pd.DataFrame(
            {"instrument_id": ["A", "B"], "session_date": [day] * 2, "adv20": [390000.0, 390000.0]}
        )
        adv_rows.append(adv)
        directory = tmp_path / f"inputs-{day}"
        identity = {"fold_id": "fixture", "session_date": str(day)}
        publish_frames(
            directory,
            identity=identity,
            frames={
                "bars.parquet": selected,
                "adv.parquet": adv,
                "profiles.parquet": profiles,
                "universe.parquet": universe,
            },
        )
        tasks.append(
            TCAWork(
                directory,
                tmp_path / f"out-{day}",
                (),
                ewma_records,
                date(2024, 1, 2),
                "sequence",
                config,
                identity,
            )
        )
    outputs = run_tca_workers(tasks, workers=2)
    times = [(path / "manifest.json").stat().st_mtime_ns for path in outputs]
    assert run_tca_workers(tasks, workers=2) == outputs
    assert [(path / "manifest.json").stat().st_mtime_ns for path in outputs] == times
    actual = pd.concat(
        [pd.read_parquet(path / "main.parquet") for path in outputs], ignore_index=True
    )
    provider = HistoricalProfileForecaster(bars, estimator="ewma", lookback_sessions=20)
    expected = run_historical_tca(
        bars.loc[bars["timestamp"].dt.date > date(2024, 1, 2)],
        universe,
        pd.concat(adv_rows, ignore_index=True),
        {"ewma": lambda instrument, day: provider},
        liquidity_size=2,
        required_methods=("ewma",),
    )
    pd.testing.assert_frame_equal(
        actual.drop(columns="fold_id"), expected, check_exact=False, atol=1e-12, rtol=1e-12
    )
