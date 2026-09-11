import numpy as np
import pandas as pd
import pytest

from execsim.ml.paper.evaluation_artifacts import publish_frames
from execsim.ml.paper.evaluation_workers import EWMAWork, ewma_ledger_identity, run_ewma_workers
from execsim.ml.paper.forecast_ledger import EWMAForecastLedgerProvider


def test_ewma_spawn_workers_publish_disjoint_resumable_instrument_shards(tmp_path):
    scale = pd.DataFrame(
        {
            "sample_id": ["a", "b"],
            "instrument_id": ["A", "B"],
            "symbol": ["AAA", "BBB"],
            "session_date": ["2023-01-03"] * 2,
            "as_of": [25, 24],
            "baseline_remaining_volume": [150.0, 300.0],
            "__evaluation_target": [150.0, 300.0],
        }
    )
    shape = scale.iloc[[0, 1, 1]].reset_index(drop=True)
    shape["case_id"] = shape["sample_id"]
    shape["target_bucket"] = [25, 24, 25]
    shape["__evaluation_target"] = [1.0, 0.5, 0.5]
    directory = tmp_path / "base"
    publish_frames(
        directory,
        identity={"fixture": True},
        frames={
            "scale-base.parquet": scale,
            "shape-base.parquet": shape,
        },
    )
    stamps = pd.date_range("2023-01-02 09:30", periods=390, freq="min", tz="America/New_York")
    bars = pd.concat(
        [
            pd.DataFrame({"symbol": symbol, "timestamp": stamps, "volume": 10.0})
            for symbol in ("AAA", "BBB")
        ],
        ignore_index=True,
    )
    market = tmp_path / "market.parquet"
    bars.to_parquet(market, index=False)
    work = [
        EWMAWork(
            directory,
            market,
            tmp_path / f"out-{instrument}",
            instrument,
            {"fold_id": "fold-1", "parameter_freeze_sha256": "f" * 64},
        )
        for instrument in ("A", "B")
    ]
    outputs = run_ewma_workers(work, workers=2)
    before = [(path / "manifest.json").stat().st_mtime_ns for path in outputs]
    assert run_ewma_workers(work, workers=2) == outputs
    assert [(path / "manifest.json").stat().st_mtime_ns for path in outputs] == before
    for path, expected_total in zip(outputs, [150.0, 300.0], strict=True):
        result = pd.read_parquet(path / "scale.parquet")
        np.testing.assert_allclose(result["predicted_remaining_volume"], expected_total)
    with pytest.raises(ValueError, match="distinct"):
        run_ewma_workers([work[0], work[0]], workers=2)


def test_ewma_persists_distinct_exact_tca_window_estimator(tmp_path):
    from datetime import date

    from execsim.forecasting import HistoricalProfileForecaster

    scale = pd.DataFrame(
        {
            "sample_id": ["a"],
            "instrument_id": ["A"],
            "symbol": ["AAA"],
            "session_date": ["2023-01-04"],
            "as_of": [23],
            "baseline_remaining_volume": [600.0],
            "__evaluation_target": [600.0],
        }
    )
    shape = scale.loc[scale.index.repeat(3)].reset_index(drop=True)
    shape["case_id"] = "a"
    shape["target_bucket"] = [23, 24, 25]
    shape["__evaluation_target"] = [1.0 / 3] * 3
    directory = tmp_path / "base"
    publish_frames(
        directory,
        identity={"fixture": True},
        frames={
            "scale-base.parquet": scale,
            "shape-base.parquet": shape,
        },
    )
    bars = pd.concat(
        [
            pd.DataFrame(
                {
                    "symbol": "AAA",
                    "timestamp": pd.date_range(
                        f"2023-01-0{day} 09:30", periods=390, freq="min", tz="America/New_York"
                    ),
                    "volume": np.r_[np.full(360, first), np.full(30, last)],
                }
            )
            for day, first, last in ((2, 10.0, 100.0), (3, 100.0, 10.0))
        ],
        ignore_index=True,
    )
    market = tmp_path / "market.parquet"
    bars.to_parquet(market, index=False)
    work = EWMAWork(directory, market, tmp_path / "out", "A", {"fold_id": "fold-1"})
    run_ewma_workers([work], workers=1)
    ledger = pd.read_parquet(tmp_path / "out/minute-forecasts.parquet")
    generated = pd.Timestamp("2023-01-04 15:15", tz="America/New_York")
    identity = ewma_ledger_identity(work)
    direct = HistoricalProfileForecaster(
        bars, estimator="ewma", data_manifest_hash=identity["market_sha256"]
    )
    expected = direct.forecast(
        symbol="AAA",
        session_date=date(2023, 1, 4),
        generated_at=generated,
        bucket_timestamps=pd.date_range(generated, periods=15, freq="min"),
    )
    boundary = ledger.loc[ledger["generated_at"] == generated].set_index("end_token")
    np.testing.assert_array_equal(boundary.loc[24, "expected_volumes"], expected.expected_volumes)
    assert not np.allclose(boundary.loc[26, "expected_volumes"][:15], expected.expected_volumes)
    replay = EWMAForecastLedgerProvider(
        work.output_directory,
        expected_identity=identity,
        symbol="AAA",
        session_date=date(2023, 1, 4),
    )
    for offset in range(15):
        instant = generated + pd.Timedelta(minutes=offset)
        request = dict(
            symbol="AAA",
            session_date=date(2023, 1, 4),
            generated_at=instant,
            bucket_timestamps=pd.date_range(instant, periods=15 - offset, freq="min"),
        )
        assert replay.forecast(**request) == direct.forecast(**request)
    with pytest.raises(ValueError, match="window"):
        replay.forecast(
            symbol="AAA",
            session_date=date(2023, 1, 4),
            generated_at=generated,
            bucket_timestamps=pd.date_range(generated, periods=45, freq="min"),
        )
    with pytest.raises(ValueError, match="as-of"):
        replay.forecast(
            symbol="AAA",
            session_date=date(2023, 1, 4),
            generated_at=generated - pd.Timedelta(minutes=1),
            bucket_timestamps=[],
        )
