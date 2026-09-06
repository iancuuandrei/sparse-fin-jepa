from __future__ import annotations

from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from execsim.data.paper.manifests import read_json
from execsim.data.paper.validation import expected_xnys_minutes
from execsim.data.scenarios import ScenarioConfig, generate_scenario
from execsim.ml.sequences.builder import _baseline, build_session_sequence
from execsim.ml.sequences.corpus import (
    _adjust_for_market_information,
    _seasonal_frame,
    _seasonal_frame_from_token_cache,
    _token_cache,
    build_fold_sequence_corpus_from_root,
)
from execsim.ml.sequences.dataset import extract_window
from execsim.ml.sequences.index import (
    build_sample_index,
    sample_training_positions,
    write_sample_index,
)
from execsim.ml.sequences.manifests import (
    read_sequence_record,
    write_sequence_manifest,
    write_sequence_record,
)
from execsim.ml.sequences.normalization import RobustFoldNormalizer
from execsim.ml.sequences.schemas import PAPER_FEATURES
from execsim.ml.sequences.streaming import PaperSequenceDataset
from execsim.ml.sequences.validation import validate_sequence_sample


def _bars(symbol: str = "AAPL") -> pd.DataFrame:
    bars = generate_scenario(
        ScenarioConfig(
            symbol=symbol,
            session_date=date(2024, 1, 3),
            n_buckets=390,
            base_volume=10_000,
            seed=7,
        )
    )
    if "trade_count" not in bars:
        bars["trade_count"] = 20
    bars["instrument_id"] = "benchmark-spy" if symbol == "SPY" else "asset-1"
    return bars


def test_sequence_is_one_fixed_session_with_causal_complete_grid(tmp_path) -> None:
    record = build_session_sequence(
        _bars(),
        instrument_id="asset-1",
        symbol="AAPL",
        source_sha256="a" * 64,
        cutoff="2024-01-02",
        spy_bars=_bars("SPY"),
        data_classification="synthetic_fixture",
    )
    samples = build_sample_index(
        record,
        fold_id="fold-1",
        partition="validation",
        source_sequence_hash="b" * 64,
    )

    assert record.features.shape == (26, len(PAPER_FEATURES))
    assert samples[0].as_of_token == 4
    assert not validate_sequence_sample(record, samples[0])
    window = extract_window(record, samples[0])
    assert window["context"].shape == (8, 18)
    assert window["context_mask"].sum() == 4
    assert sample_training_positions(samples, epoch=2, seed=13) == sample_training_positions(
        samples, epoch=2, seed=13
    )
    assert len(sample_training_positions(samples, epoch=2, seed=13)) == 2
    artifact = write_sequence_record(record, tmp_path)
    restored = read_sequence_record(artifact)
    assert restored.session_id == record.session_id
    assert np.array_equal(restored.features, record.features)
    index_path = write_sample_index(samples, tmp_path / "index.parquet")
    assert len(pd.read_parquet(index_path)) == len(samples)
    with pytest.raises(ValueError, match="Historical sequence builds require"):
        build_session_sequence(
            _bars(),
            instrument_id="asset-1",
            symbol="AAPL",
            source_sha256="a" * 64,
            cutoff="2024-01-02",
        )


def test_prior_session_restatement_uses_current_causal_action_knowledge() -> None:
    session = _bars()
    actions = pd.DataFrame(
        {
            "instrument_id": ["asset-1"],
            "effective_date": [date(2024, 1, 2)],
            "factor": [0.5],
            "available_at": [pd.Timestamp("2024-01-04T00:00:00Z")],
        }
    )
    before_known = _adjust_for_market_information(
        session,
        actions,
        instrument_id="asset-1",
        market_information_as_of=pd.Timestamp("2024-01-03T14:30:00Z"),
    )
    after_known = _adjust_for_market_information(
        session,
        actions,
        instrument_id="asset-1",
        market_information_as_of=pd.Timestamp("2024-01-05T14:30:00Z"),
    )

    assert np.allclose(before_known["close"], session["close"])
    assert np.allclose(after_known["close"], session["close"] / 0.5)
    assert np.allclose(after_known["volume"], session["volume"] * 0.5)


def test_cached_seasonal_tokens_preserve_point_in_time_split_restatement() -> None:
    session = _bars()
    history = [(date(2024, 1, 3), session)]
    actions = pd.DataFrame(
        {
            "instrument_id": ["asset-1"],
            "effective_date": [date(2024, 1, 2)],
            "factor": [0.5],
            "available_at": [pd.Timestamp("2024-01-04T00:00:00Z")],
        }
    )
    information_time = pd.Timestamp("2024-01-05T14:30:00Z")
    adjusted = _adjust_for_market_information(
        session,
        actions,
        instrument_id="asset-1",
        market_information_as_of=information_time,
    )
    expected = _seasonal_frame([(date(2024, 1, 3), adjusted)], quality_protocol="exact-minute-v1")

    actual = _seasonal_frame_from_token_cache(
        history,
        _token_cache(history, quality_protocol="exact-minute-v1"),
        corporate_actions=actions,
        instrument_id="asset-1",
        market_information_as_of=information_time,
    )

    pd.testing.assert_frame_equal(actual, expected)


def test_vectorized_seasonal_baseline_matches_pandas_adjusted_ewma() -> None:
    dates = pd.date_range("2024-01-02", periods=7, freq="B").date
    rows = [
        {
            "session_date": session_date,
            "bucket_index": bucket,
            "volume": 100.0 + day * 10 + bucket,
            "dollar_volume": 1_000.0 + day * 20 + 2 * bucket,
            "trade_count": 10.0 + day + bucket / 10,
        }
        for day, session_date in enumerate(dates)
        for bucket in range(26)
    ]
    seasonal = pd.DataFrame(rows)

    result = _baseline(seasonal, cutoff=str(dates[-1]))

    for column, actual in (
        ("volume", result.volume_profile),
        ("dollar_volume", result.dollar_profile),
        ("trade_count", result.trade_profile),
    ):
        expected = np.asarray(
            [
                seasonal.loc[seasonal["bucket_index"] == bucket, column]
                .ewm(span=20, adjust=True)
                .mean()
                .iloc[-1]
                for bucket in range(26)
            ]
        )
        assert np.allclose(actual, expected, rtol=1e-13, atol=1e-13)
    assert result.adv20 == pytest.approx(seasonal.groupby("session_date")["volume"].sum().mean())


def test_normalizer_uses_persisted_training_statistics_and_zero_padding(tmp_path) -> None:
    rng = np.random.default_rng(3)
    training = rng.normal(size=(4, 26, 18))
    mask = np.ones((4, 26), dtype=bool)
    normalizer = RobustFoldNormalizer.fit(training, mask)
    values = np.full((1, 26, 18), 1_000.0)
    valid = np.ones((1, 26), dtype=bool)
    valid[:, -2:] = False

    transformed = normalizer.transform(values, valid)

    assert np.all(transformed[:, -2:] == 0)
    assert np.isfinite(transformed).all()
    assert len(normalizer.stable_payload()["median"]) == 18
    manifest = write_sequence_manifest(
        tmp_path / "manifest.json",
        fold_id="fold-1",
        cutoff="2023-12-29",
        raw_hashes=("a" * 64,),
        sequence_files=("session.parquet",),
        normalizer=normalizer,
    )
    assert manifest.manifest_id.startswith("sequence-manifest-")


def test_multisession_multifold_corpus_builder_includes_spy_and_records_corruption(
    tmp_path,
) -> None:
    import exchange_calendars as xcals

    calendar = xcals.get_calendar("XNYS")
    ranges = (
        ("2023-12-01", "2023-12-29"),
        ("2024-01-02", "2024-03-28"),
        ("2024-04-01", "2024-06-28"),
        ("2024-07-01", "2024-09-30"),
        ("2024-10-01", "2024-12-31"),
    )
    dates = [
        pd.Timestamp(value).date()
        for start, end in ranges
        for value in calendar.sessions_in_range(start, end)[:8]
    ]
    instruments = [(f"asset-{index}", f"S{index}") for index in range(4)]
    instruments.append(("benchmark-spy", "SPY"))
    frames = [
        _paper_session(instrument_id, symbol, session_date, instrument_index)
        for instrument_index, (instrument_id, symbol) in enumerate(instruments)
        for session_date in dates
    ]
    malformed = frames[3 * len(dates) + 17]
    malformed.loc[30, "timestamp"] = pd.Timestamp(
        f"{malformed['timestamp'].iloc[0].date()} 16:00", tz="America/New_York"
    )
    bars = pd.concat(frames, ignore_index=True)
    sparse_date = dates[6]
    sparse_mask = (bars["instrument_id"] == "asset-0") & (
        pd.to_datetime(bars["timestamp"]).dt.date == sparse_date
    )
    sparse_indexes = bars.loc[sparse_mask].index[[1, 18, 80, 200, 388]]
    bars = bars.drop(index=sparse_indexes).reset_index(drop=True)
    rename_date = date(2024, 4, 1)
    renamed = (bars["instrument_id"] == "asset-1") & (
        pd.to_datetime(bars["timestamp"]).dt.date >= rename_date
    )
    bars.loc[renamed, "symbol"] = "N1"
    actions = pd.DataFrame(
        {
            "instrument_id": ["asset-1"],
            "effective_date": [date(2024, 4, 1)],
            "factor": [2.0],
            "available_at": [pd.Timestamp("2024-03-29", tz="UTC")],
            "source": ["synthetic-fixture"],
        }
    )
    members = tuple(
        {"instrument_id": instrument_id, "formation_symbol": symbol}
        for instrument_id, symbol in instruments[:-1]
    )
    symbol_history = (
        *(
            {
                "instrument_id": instrument_id,
                "symbol": symbol,
                "start": "2021-01-04",
                "end": "2025-12-31",
            }
            for instrument_id, symbol in instruments
            if instrument_id != "asset-1"
        ),
        {
            "instrument_id": "asset-1",
            "symbol": "S1",
            "start": "2021-01-04",
            "end": "2024-03-31",
        },
        {
            "instrument_id": "asset-1",
            "symbol": "N1",
            "start": "2024-04-01",
            "end": "2025-12-31",
        },
    )
    corpus_root = tmp_path / "raw"
    corpus_root.mkdir()
    for instrument_id, frame in bars.groupby("instrument_id", sort=True):
        frame.to_parquet(corpus_root / f"{instrument_id}-fixture.response", index=False)
    manifests = []
    for fold_id in ("fold-1", "fold-2"):
        manifests.append(
            build_fold_sequence_corpus_from_root(
                corpus_root,
                universe_members=members,
                corporate_actions=actions,
                fold_id=fold_id,
                output_root=tmp_path / "sequences",
                universe_manifest_hash="a" * 64,
                corporate_action_manifest_hash="b" * 64,
                config_hash="c" * 64,
                spy_instrument_id="benchmark-spy",
                data_classification="synthetic_fixture",
                quality_protocol="resolution-aware-v2",
                symbol_history=symbol_history,
            )
        )
    first_payload = read_json(tmp_path / "sequences" / "fold-1" / "sequence-manifest.json")
    train = PaperSequenceDataset(
        tmp_path / "sequences" / "fold-1" / "sequence-manifest.json",
        partition="train",
        seed=13,
    )

    assert len(dates) == 40
    assert len(manifests) == 2
    assert first_payload["quality_protocol"] == "resolution-aware-v2"
    assert len(train) == 2 * first_payload["partition_counts"]["train"]
    assert any(item["instrument_id"] == "asset-3" for item in first_payload["exclusions"])
    assert all("benchmark-spy" not in path for path in first_payload["sequence_files"])
    sparse_file = next(
        path
        for path in first_payload["sequence_files"]
        if "asset-0" in path and sparse_date.isoformat() in path
    )
    sparse_record = read_sequence_record(tmp_path / "sequences" / "fold-1" / sparse_file)
    assert sparse_record.provider_gap_count == 5
    assert sparse_record.observed_bar_count.tolist() == [
        14,
        14,
        15,
        15,
        15,
        14,
        *([15] * 7),
        14,
        *([15] * 11),
        14,
    ]
    renamed_file = next(
        path
        for path in first_payload["sequence_files"]
        if "asset-1" in path and rename_date.isoformat() in path
    )
    assert read_sequence_record(tmp_path / "sequences" / "fold-1" / renamed_file).symbol == "N1"

    _run_fixture_pipeline(
        tmp_path,
        bars,
        members,
        tmp_path / "sequences" / "fold-1" / "sequence-manifest.json",
    )


def _paper_session(
    instrument_id: str, symbol: str, session_date: date, instrument_index: int
) -> pd.DataFrame:
    timestamps = expected_xnys_minutes(session_date)
    minute = np.arange(len(timestamps), dtype=float)
    base = 50.0 + 10 * instrument_index + 0.002 * minute
    volume = 1_000 + 20 * instrument_index + (minute % 30) * 3
    return pd.DataFrame(
        {
            "instrument_id": instrument_id,
            "symbol": symbol,
            "timestamp": timestamps,
            "open": base,
            "high": base + 0.1,
            "low": base - 0.1,
            "close": base + 0.01,
            "volume": volume,
            "trade_count": 20 + minute % 5,
            "vwap": base + 0.005,
        }
    )


def _run_fixture_pipeline(
    root: Path,
    bars: pd.DataFrame,
    members: tuple[dict[str, str], ...],
    sequence_manifest: Path,
) -> None:
    from execsim.costs import CostParameter, LinearTemporaryImpactModel
    from execsim.data.paper.manifests import file_sha256, stable_hash, write_json_atomic
    from execsim.ml.models.lightgbm_adapter import LightGBMConfig, LightGBMVolumeModel
    from execsim.ml.paper.benchmark import profile_paper_kernels
    from execsim.ml.paper.features import append_untrained_neural_control_frames
    from execsim.ml.paper.forecast_provider import PaperLightGBMForecastProvider
    from execsim.ml.paper.lightgbm_data import build_lightgbm_frames
    from execsim.ml.paper.reports import write_historical_paper_bundle
    from execsim.ml.paper.statistics import (
        construct_complete_case_differences,
        moving_block_bootstrap,
    )
    from execsim.ml.paper.tca import SegmentCommittedMPCPolicy, realized_volume_oracle_cost
    from execsim.ml.representations.checkpoints import load_checkpoint
    from execsim.ml.representations.embedding_pipeline import export_embedding_corpus
    from execsim.ml.representations.frozen_evaluation import (
        FrozenProbeOptions,
        evaluate_frozen_capacity_streaming,
    )
    from execsim.ml.representations.historical_trainer import (
        HistoricalTrainerOptions,
        HistoricalTrainingIdentity,
        HistoricalTrainingInterrupted,
        train_historical_representation,
    )
    from execsim.ml.representations.jepa import PredictiveRepresentationModel
    from execsim.ml.representations.schemas import (
        CheckpointCompatibility,
        RepresentationConfig,
    )
    from execsim.ml.sequences.streaming import PaperSequenceDataset, build_sequence_dataloader
    from execsim.orders import ParentOrder
    from execsim.policies import ExecutionConstraints
    from execsim.simulator import simulate_policy

    manifest_payload = read_json(sequence_manifest)
    profile = profile_paper_kernels(sequence_manifest)
    write_json_atomic(root / "kernel-profile.json", asdict(profile))
    assert profile.sequence_rows_measured > 0
    assert profile.sequence_rows_per_second is not None
    identity = HistoricalTrainingIdentity(
        fold_id="fold-1",
        cutoff="2023-12-29",
        universe_manifest_hash="a" * 64,
        dataset_manifest_hash=stable_hash(manifest_payload["raw_hashes"]),
        normalization_hash=stable_hash(manifest_payload["normalization"]),
        architecture_hash="d" * 64,
        config_hash="c" * 64,
        code_commit="e" * 40,
    )
    trainer_options = HistoricalTrainerOptions(
        batch_size=64,
        max_epochs=1,
        patience=6,
        prefetch_factor=1,
        cache_size=8,
        use_bfloat16=False,
        checkpoint_interval_steps=1,
    )
    representations = {}
    accessibility_rows = []
    for geometry in ("dense", "sparse"):
        representation = RepresentationConfig(
            geometry,
            rdm_projections_train=4,
            rdm_projections_evaluation=16,
            seed=13,
        )
        checkpoint_root = root / f"checkpoint-{geometry}"
        result = train_historical_representation(
            sequence_manifest,
            representation=representation,
            identity=identity,
            output_root=checkpoint_root,
            allow_historical_training=True,
            rdm_lambda=1.0,
            options=trainer_options,
            collapse_gate=lambda diagnostics: (),
        )
        assert result.global_steps > 0
        assert dict(result.diagnostics)["rdmreg_diagnostic_projections"] == 16
        compatibility = CheckpointCompatibility(**read_json(checkpoint_root / "compatibility.json"))
        embedding_root = root / f"embeddings-{geometry}"
        embedding_manifest = export_embedding_corpus(
            PredictiveRepresentationModel(representation),
            checkpoint_directory=checkpoint_root / "final",
            expected_checkpoint=compatibility,
            sequence_manifest_path=sequence_manifest,
            output_root=embedding_root,
            seed=13,
            geometry=geometry,
            adaptation="none",
            device="cpu",
            batch_size=256,
        )
        assert read_json(embedding_manifest)["rows"] > 0
        frozen = PredictiveRepresentationModel(representation)
        load_checkpoint(frozen, checkpoint_root / "final", expected=compatibility)

        def fixture_loader(partition: str):
            return build_sequence_dataloader(
                PaperSequenceDataset(
                    sequence_manifest,
                    partition=partition,
                    seed=13,
                    cache_size=8,
                    sample_train_positions=False,
                ),
                batch_size=256,
                num_workers=0,
                device="cpu",
            )

        capacity, observable = evaluate_frozen_capacity_streaming(
            frozen,
            fixture_loader("train"),
            fixture_loader("validation"),
            fixture_loader("test"),
            device="cpu",
            seed=13,
            options=FrozenProbeOptions(ridge_alphas=(1.0,), mlp_epochs=1),
        )
        observable_by_horizon = {int(item["horizon"]): item for item in observable}
        for row in capacity:
            accessibility_rows.append(
                {
                    "geometry": geometry,
                    "seed": 13,
                    **row,
                    **observable_by_horizon[int(row["horizon"])],
                    "zero_fraction": 0.75 if geometry == "sparse" else 0.0,
                    "mean_active_dimensions": 32.0 if geometry == "sparse" else 128.0,
                }
            )
        representations[geometry] = embedding_root
    interrupted_root = root / "checkpoint-dense-interrupted"
    dense_representation = RepresentationConfig("dense", rdm_projections_train=4, seed=13)
    with pytest.raises(HistoricalTrainingInterrupted):
        train_historical_representation(
            sequence_manifest,
            representation=dense_representation,
            identity=identity,
            output_root=interrupted_root,
            allow_historical_training=True,
            rdm_lambda=1.0,
            options=trainer_options,
            collapse_gate=lambda diagnostics: (),
            interrupt_after_steps=1,
        )
    train_historical_representation(
        sequence_manifest,
        representation=dense_representation,
        identity=identity,
        output_root=interrupted_root,
        allow_historical_training=True,
        rdm_lambda=1.0,
        options=trainer_options,
        collapse_gate=lambda diagnostics: (),
        resume_from=interrupted_root / "periodic" / "step=000000001",
        trusted_resume=True,
    )
    from safetensors.torch import load_file

    continuous = load_file(root / "checkpoint-dense" / "final" / "model.safetensors")
    resumed = load_file(interrupted_root / "final" / "model.safetensors")
    assert continuous.keys() == resumed.keys()
    assert all(torch.equal(continuous[name], resumed[name]) for name in continuous)
    liquidity = {member["instrument_id"]: 1 for member in members}
    frame_variants = {}
    for method in ("raw", "untrained", "dense", "sparse"):
        embedding_root = representations.get(method)
        training = build_lightgbm_frames(
            sequence_manifest,
            partition="train",
            liquidity_groups=liquidity,
            embedding_path=None
            if embedding_root is None
            else embedding_root / "partition=train" / "embeddings.parquet",
        )
        validation = build_lightgbm_frames(
            sequence_manifest,
            partition="validation",
            liquidity_groups=liquidity,
            embedding_path=None
            if embedding_root is None
            else embedding_root / "partition=validation" / "embeddings.parquet",
        )
        if method == "untrained":
            training = append_untrained_neural_control_frames(training, fold_seed=13)
            validation = append_untrained_neural_control_frames(validation, fold_seed=13)
        fitted = LightGBMVolumeModel(
            LightGBMConfig(n_estimators=8, min_child_samples=2, num_threads=1)
        ).fit_frames(*training, categorical_features=("symbol",), validation=validation)
        frame_variants[method] = (validation, fitted)

    validation, model = frame_variants["sparse"]
    predicted_total, predicted_shape = model.predict_frames(
        validation[0].iloc[[0]],
        validation[2].loc[validation[2]["case_id"] == validation[2]["case_id"].iloc[0]],
        group_columns=("case_id",),
    )
    assert predicted_total[0] >= 0
    assert predicted_shape["conditional_share"].sum() == pytest.approx(1.0)

    scale_case = validation[0].iloc[[0]]
    forecast_date = pd.Timestamp(scale_case["session_date"].iloc[0]).date()
    as_of = int(scale_case["as_of"].iloc[0])
    generated = pd.Timestamp.combine(forecast_date, pd.Timestamp("09:30").time()).tz_localize(
        "America/New_York"
    ) + pd.Timedelta(minutes=15 * as_of)

    def resolve_fixture(_symbol, _date, at, _observations):
        token = ((at.hour * 60 + at.minute) - 570) // 15
        scale = validation[0].loc[
            (validation[0]["instrument_id"] == scale_case["instrument_id"].iloc[0])
            & (validation[0]["session_date"] == scale_case["session_date"].iloc[0])
            & (validation[0]["as_of"] == token)
        ]
        shape = validation[2].loc[validation[2]["case_id"] == scale["sample_id"].iloc[0]]
        return scale, shape

    provider = PaperLightGBMForecastProvider(
        model,
        feature_resolver=resolve_fixture,
        within_token_profile=np.full(15, 1 / 15),
        training_cutoff=date(2023, 12, 29),
        manifest_hash=file_sha256(sequence_manifest),
        method_id="sparse-fixture",
    )
    session = bars.loc[
        (bars["instrument_id"] == scale_case["instrument_id"].iloc[0])
        & (
            pd.to_datetime(bars["timestamp"]).dt.tz_convert("America/New_York").dt.date
            == forecast_date
        )
    ]
    execution_bars = session.loc[
        (pd.to_datetime(session["timestamp"]).dt.time >= generated.time())
        & (
            pd.to_datetime(session["timestamp"]).dt.time
            < (generated + pd.Timedelta(minutes=30)).time()
        )
    ]
    simulation = simulate_policy(
        parent_order=ParentOrder(
            str(scale_case["symbol"].iloc[0]),
            "buy",
            100,
            forecast_date,
            generated.time(),
            (generated + pd.Timedelta(minutes=30)).time(),
        ),
        bars=execution_bars,
        policy=SegmentCommittedMPCPolicy(half_spread=0.005, temporary_impact=0.1),
        constraints=ExecutionConstraints(0.1, 0.1),
        cost_model=LinearTemporaryImpactModel(
            half_spread=CostParameter(0.005), temporary_impact=CostParameter(0.1)
        ),
        forecast_provider=provider,
    )
    assert simulation.summary.filled_qty == 100
    arrival_price = float(execution_bars["open"].iloc[0])
    oracle_cost = realized_volume_oracle_cost(
        execution_bars,
        quantity=100,
        start=generated.time(),
        end=(generated + pd.Timedelta(minutes=30)).time(),
        arrival_price=arrival_price,
        half_spread_arrival_fraction=0.005 / arrival_price,
        temporary_impact_arrival_fraction=0.1 / arrival_price,
    )
    normalized_regret = (simulation.summary.modeled_temporary_impact_cost - oracle_cost) / max(
        oracle_cost, 1e-12
    )
    assert oracle_cost > 0
    assert normalized_regret >= -1e-8

    cases = pd.DataFrame(
        {
            "method": ["raw", "sparse"] * 6,
            "fold_id": ["fold-1"] * 12,
            "date": np.repeat(pd.date_range("2024-04-01", periods=6), 2),
            "instrument_id": ["asset-0"] * 12,
            "cost": [10.0, 9.0] * 6,
        }
    )
    paired = construct_complete_case_differences(
        cases,
        baseline="raw",
        candidate="sparse",
        value_column="cost",
        identity_columns=("fold_id", "date", "instrument_id"),
    )
    inference = moving_block_bootstrap(
        paired.paired_rows[["fold_id", "date", "difference"]], repetitions=100
    )
    assert inference.mean_difference == -1.0
    forecast_rows = []
    for method, (method_frames, fitted) in frame_variants.items():
        candidate_scale = method_frames[0].iloc[[0]]
        candidate_id = candidate_scale["sample_id"].iloc[0]
        candidate_shape = method_frames[2].loc[method_frames[2]["case_id"] == candidate_id]
        total_prediction, share_prediction = fitted.predict_frames(
            candidate_scale, candidate_shape, group_columns=("case_id",)
        )
        actual_share = method_frames[3][candidate_shape.index]
        forecast_rows.append(
            {
                "method": method,
                "as_of_token": as_of,
                "log_remaining_volume_mae": abs(
                    np.log1p(total_prediction[0]) - np.log1p(method_frames[1][0])
                ),
                "conditional_curve_error": float(
                    np.mean(
                        np.abs(
                            np.cumsum(share_prediction["conditional_share"].to_numpy())
                            - np.cumsum(actual_share)
                        )
                    )
                ),
            }
        )
    manifest_counts = manifest_payload["partition_counts"]
    tables = {
        "dataset_folds_exclusions": pd.DataFrame(
            [
                {
                    "fold_id": "fold-1",
                    "partition": partition,
                    "included": count,
                    "excluded": len(manifest_payload["exclusions"]),
                }
                for partition, count in manifest_counts.items()
            ]
        ),
        "representation_accessibility": pd.DataFrame(accessibility_rows),
        "forecasting": pd.DataFrame(forecast_rows),
        "execution": pd.DataFrame(
            {
                "method": ["sparse"],
                "comparison_baseline": ["raw"],
                "seed": [13],
                "normalized_allocation_regret": [normalized_regret],
                "absolute_modeled_impact_cost": [simulation.summary.modeled_temporary_impact_cost],
                "completion_rate": [simulation.summary.completion_rate],
                "implementation_shortfall_bps": [simulation.summary.implementation_shortfall_bps],
                "mean_difference": [inference.mean_difference],
                "ci_lower": [inference.confidence_interval[0]],
                "ci_upper": [inference.confidence_interval[1]],
            }
        ),
    }
    bundle = write_historical_paper_bundle(
        root,
        paper_run_id="multi-session-historical-style-fixture",
        tables=tables,
        provenance={"data_classification": "synthetic_fixture"},
        historical_schema_fixture=True,
    )
    assert (bundle / "provenance.json").is_file()
