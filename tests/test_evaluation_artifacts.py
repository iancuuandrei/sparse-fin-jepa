from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from execsim.data.paper.manifests import file_sha256
from execsim.ml.paper.evaluation_artifacts import (
    evaluation_base,
    forecast_metric_frame,
    prediction_batches,
    validate_base,
)
from execsim.ml.paper.lightgbm_data import LightGBMFrames, attach_lightgbm_embeddings


@pytest.mark.parametrize("representation", ["raw", "hybrid", "untrained"])
def test_prediction_batches_preserve_full_frames_and_native_predictions(tmp_path, representation):
    from execsim.ml.models.lightgbm_adapter import LightGBMConfig, LightGBMVolumeModel
    from execsim.ml.paper.features import append_untrained_neural_control_frames

    base = _base()
    if representation == "untrained":
        extra = {
            f"context_t{token:02d}_f{feature:02d}": float(token + feature) / 100
            for token in range(8)
            for feature in range(18)
        }
        extra.update({f"context_mask_t{token:02d}": True for token in range(8)})
        base = LightGBMFrames(
            pd.concat([base.scale, pd.DataFrame(extra, index=base.scale.index)], axis=1),
            base.scale_target,
            pd.concat([base.shape, pd.DataFrame(extra, index=base.shape.index)], axis=1),
            base.shape_target,
        )
    embedding = tmp_path / "embedding.parquet"
    pd.DataFrame({"sample_id": ["a", "b"], "embedding": [np.ones(644), np.zeros(644)]}).to_parquet(
        embedding
    )
    path = embedding if representation == "hybrid" else None
    expected = attach_lightgbm_embeddings(base, embedding_path=path) if path else base
    if representation == "untrained":
        expected = LightGBMFrames(
            *append_untrained_neural_control_frames(base.as_tuple(), fold_seed=13)
        )
    batches = list(
        prediction_batches(
            base,
            embedding_path=path,
            untrained_control=representation == "untrained",
            batch_samples=1,
        )
    )
    assert len(batches) == 2
    for attr in ("scale", "shape"):
        pd.testing.assert_frame_equal(
            pd.concat([getattr(batch, attr) for batch in batches], ignore_index=True),
            getattr(expected, attr),
            check_exact=True,
        )
    for attr in ("scale_target", "shape_target"):
        np.testing.assert_array_equal(
            np.concatenate([getattr(batch, attr) for batch in batches]), getattr(expected, attr)
        )
    model = LightGBMVolumeModel(LightGBMConfig(n_estimators=2, min_child_samples=1)).fit_frames(
        *expected.as_tuple(), categorical_features=("symbol",)
    )
    total, shape = model.predict_frames(expected.scale, expected.shape, group_columns=("case_id",))
    predictions = [
        model.predict_frames(batch.scale, batch.shape, group_columns=("case_id",))
        for batch in batches
    ]
    np.testing.assert_allclose(
        np.concatenate([row[0] for row in predictions]), total, rtol=0, atol=0
    )
    pd.testing.assert_frame_equal(
        pd.concat([row[1] for row in predictions], ignore_index=True), shape
    )


def _base() -> LightGBMFrames:
    scale = pd.DataFrame(
        {
            "sample_id": ["b", "a"],
            "instrument_id": ["B", "A"],
            "session_date": ["2023-01-03"] * 2,
            "as_of": [24, 25],
            "symbol": pd.Categorical(["BBB", "AAA"], categories=["AAA", "BBB"]),
            "raw_0": [3.0, 7.0],
            "baseline_remaining_volume": [90.0, 180.0],
        }
    )
    shape = scale.iloc[[0, 0, 1]].reset_index(drop=True)
    shape["case_id"] = shape["sample_id"]
    shape["target_bucket"] = [24, 25, 25]
    shape["sample_weight"] = [0.5, 0.5, 1.0]
    return LightGBMFrames(scale, np.array([100.0, 200.0]), shape, np.array([0.4, 0.6, 1.0]))


def test_evaluation_cache_preserves_sorted_features_targets_and_embedding_join(tmp_path):
    manifest = tmp_path / "sequence.json"
    manifest.write_text("{}")
    calls = []

    def build(*args, **kwargs):
        calls.append(kwargs["partition"])
        return _base()

    identity = dict(
        parameter_freeze_sha256="a" * 64,
        source_commit="b" * 40,
        source_tree="c" * 40,
        paper_config_hash="d" * 64,
    )
    options = dict(
        partition="validation",
        liquidity_groups={"A": 1, "B": 2},
        directory=tmp_path / "base",
        execution_identity=identity,
        builder=build,
    )
    actual = evaluation_base(manifest, **options)
    repeated = evaluation_base(manifest, **options)
    assert calls == ["validation"]
    direct = _base()
    pd.testing.assert_frame_equal(actual.scale, direct.scale.iloc[[1, 0]].reset_index(drop=True))
    pd.testing.assert_frame_equal(actual.shape, direct.shape.iloc[[2, 0, 1]].reset_index(drop=True))
    np.testing.assert_array_equal(actual.scale_target, [200.0, 100.0])
    np.testing.assert_array_equal(actual.shape_target, [1.0, 0.4, 0.6])
    pd.testing.assert_frame_equal(actual.scale, repeated.scale)
    embeddings = tmp_path / "embeddings.parquet"
    pd.DataFrame(
        {"sample_id": ["b", "a"], "embedding": [np.ones(644), np.full(644, 2.0)]}
    ).to_parquet(embeddings)
    hybrid = attach_lightgbm_embeddings(actual, embedding_path=embeddings)
    direct_hybrid = attach_lightgbm_embeddings(direct, embedding_path=embeddings)
    pd.testing.assert_frame_equal(
        hybrid.scale, direct_hybrid.scale.iloc[[1, 0]].reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(
        hybrid.shape, direct_hybrid.shape.iloc[[2, 0, 1]].reset_index(drop=True)
    )
    with pytest.raises(ValueError, match="identity"):
        evaluation_base(
            manifest, **{**options, "execution_identity": {**identity, "source_commit": "e" * 40}}
        )
    (tmp_path / "base" / "shape-base.parquet").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        evaluation_base(manifest, **options)


@pytest.mark.parametrize("corruption", ["duplicate", "missing", "unavailable", "nan"])
def test_evaluation_base_rejects_invalid_identity_or_targets(corruption):
    base = _base()
    if corruption == "duplicate":
        base.scale.loc[1, "sample_id"] = "b"
    elif corruption == "missing":
        base.shape.loc[0, "sample_id"] = "missing"
    elif corruption == "unavailable":
        base.shape.loc[0, "target_bucket"] = 23
    else:
        base.shape_target[0] = np.nan
    with pytest.raises(ValueError):
        validate_base(base)


def test_vector_metrics_match_reference_and_reject_unmatched_forecasts():
    base = _base()
    shape = base.shape.loc[:, ["case_id", "target_bucket"]].copy()
    shape["conditional_share"] = [0.7, 0.3, 1.0]
    shape = shape.iloc[[2, 1, 0]].reset_index(drop=True)
    totals = np.array([120.0, 180.0])
    actual = forecast_metric_frame(base, totals, shape, fold_id="fold-1", method="raw", seed=None)
    expected_error = np.mean(np.abs(np.cumsum([0.4, 0.6]) - np.cumsum([0.7, 0.3])))
    np.testing.assert_allclose(
        actual["conditional_curve_wasserstein"], [expected_error, 0.0], atol=1e-15
    )
    np.testing.assert_array_equal(actual["sample_id"], ["b", "a"])
    np.testing.assert_array_equal(
        actual["log_remaining_volume_absolute_error"],
        np.abs(np.log1p(totals) - np.log1p(base.scale_target)),
    )
    with pytest.raises(ValueError, match="population"):
        forecast_metric_frame(
            base, totals, shape.iloc[1:], fold_id="fold-1", method="raw", seed=None
        )


@pytest.mark.parametrize("isolated", [False, True])
def test_forecast_stage_builds_one_base_and_resumes_published_predictions(
    tmp_path, monkeypatch, isolated
):
    from functools import partial
    from types import SimpleNamespace

    from execsim.ml.models.lightgbm_adapter import LightGBMVolumeModel
    from execsim.ml.paper import evaluation_artifacts, orchestration

    source = _base()
    source.scale["fold_id"] = "fold-1"
    for name in ("training_cutoff", "market_information_as_of", "feature_history_end"):
        source.scale[name] = "2023-01-02"
    sequence = tmp_path / "sequences/fold-1/sequence-manifest.json"
    sequence.parent.mkdir(parents=True)
    sequence.write_text("{}")
    freeze = tmp_path / "selection/parameter-freeze-v1.json"
    freeze.parent.mkdir()
    freeze.write_text("{}")
    universe = tmp_path / "universe.json"
    universe.write_text(
        '{"members": [{"instrument_id": "A", "liquidity_group": 1},'
        '{"instrument_id": "B", "liquidity_group": 2}]}'
    )
    sequence.write_text('{"universe_manifest_hash": "' + file_sha256(universe) + '"}')
    for method in ("raw", "untrained_neural"):
        path = tmp_path / "lightgbm/fold-1" / method / "shared/manifest.json"
        path.parent.mkdir(parents=True)
        path.write_text("{}")
    config = SimpleNamespace(
        authorize=lambda *args, **kwargs: None,
        artifact_root=tmp_path,
        config_hash="config",
        data={"universe_manifest": str(universe), "target_corpus_root": str(tmp_path)},
        evaluation={"folds": [{"id": "fold-1"}]},
        representation={"seeds": []},
    )
    config.data_path = lambda name: Path(config.data[name])
    if isolated:
        config.runtime_evaluation_root = tmp_path / "evaluation-executions" / "fixture"
    calls = []

    def build(*args, **kwargs):
        calls.append("build")
        return source

    class Model:
        def predict_frames(self, scale, shape, **kwargs):
            calls.append("predict")
            prediction = shape[["case_id", "target_bucket"]].copy()
            prediction["conditional_share"] = 1.0 / (26 - shape["as_of"])
            return scale["baseline_remaining_volume"].to_numpy(), prediction

    monkeypatch.setattr(
        evaluation_artifacts, "evaluation_base", partial(evaluation_base, builder=build)
    )
    for name in ("_require_parameter_freeze", "_require_locked_test_opened"):
        monkeypatch.setattr(orchestration, name, lambda *args: {})
    monkeypatch.setattr(orchestration, "_git_head", lambda: "a" * 40)
    monkeypatch.setattr(orchestration, "_git_tree", lambda: "b" * 40)
    monkeypatch.setattr(
        "execsim.ml.paper.features.untrained_neural_control_embedding",
        lambda scale, **kwargs: np.zeros((len(scale), 644), dtype=np.float32),
    )
    monkeypatch.setattr(
        LightGBMVolumeModel, "load_native", lambda *args: (Model(), {"paper_config_hash": "config"})
    )
    minutes = pd.date_range("2023-01-02 09:30", periods=390, freq="min", tz="America/New_York")
    bars = pd.concat(
        [
            pd.DataFrame({"timestamp": minutes, "symbol": symbol, "volume": 10.0})
            for symbol in ("AAA", "BBB")
        ],
        ignore_index=True,
    )
    bars["instrument_id"] = bars["symbol"].map({"AAA": "A", "BBB": "B"})
    market_path = tmp_path / "fixture-market.parquet"
    bars.to_parquet(market_path, index=False)
    config.data["target_corpus_root"] = str(market_path)
    monkeypatch.setenv("EXECSIM_EVALUATION_WORKERS", "1")
    first = orchestration.evaluate_forecasts_stage(
        config, full_run_cli_enabled=True, runtime_approval=None
    )
    assert first["rows"] == 6
    second = orchestration.evaluate_forecasts_stage(
        config, full_run_cli_enabled=True, runtime_approval=None
    )
    assert second["rows"] == 6
    assert calls == ["build", "predict", "predict"]
    if isolated:
        assert not (tmp_path / "evaluation-v2").exists()


def test_locked_runtime_universe_is_relocated_and_hash_bound(tmp_path):
    from types import SimpleNamespace

    from execsim.ml.paper.orchestration import _verify_frozen_universe_manifest

    runtime_root = tmp_path / "runtime-data"
    runtime_universe = runtime_root / "universe.json"
    runtime_universe.parent.mkdir(parents=True)
    runtime_universe.write_text('{"members": [{"instrument_id": "A"}]}')
    artifact_root = tmp_path / "artifacts"
    sequence = artifact_root / "sequences/fold-1/sequence-manifest.json"
    sequence.parent.mkdir(parents=True)
    sequence.write_text('{"universe_manifest_hash": "' + file_sha256(runtime_universe) + '"}')
    config = SimpleNamespace(
        artifact_root=artifact_root,
        data={"universe_manifest": "data/universe.json"},
        evaluation={"folds": [{"id": "fold-1"}]},
    )
    config.data_path = lambda name: runtime_root / Path(*Path(config.data[name]).parts[1:])

    resolved = _verify_frozen_universe_manifest(config)
    assert resolved == runtime_universe
    assert not (tmp_path / "repo/data/universe.json").exists()

    runtime_universe.write_text('{"members": [{"instrument_id": "B"}]}')
    with pytest.raises(ValueError, match="checksum"):
        _verify_frozen_universe_manifest(config)


def test_locked_runtime_universe_requires_one_hash_across_all_folds(tmp_path):
    from types import SimpleNamespace

    from execsim.ml.paper.orchestration import _verify_frozen_universe_manifest

    runtime_root = tmp_path / "runtime-data"
    universe = runtime_root / "universe.json"
    universe.parent.mkdir(parents=True)
    universe.write_text('{"members": [{"instrument_id": "A"}]}')
    artifact_root = tmp_path / "artifacts"
    for fold_id, digest in (("fold-1", file_sha256(universe)), ("fold-2", "0" * 64)):
        sequence = artifact_root / "sequences" / fold_id / "sequence-manifest.json"
        sequence.parent.mkdir(parents=True, exist_ok=True)
        sequence.write_text('{"universe_manifest_hash": "' + digest + '"}')
    config = SimpleNamespace(
        artifact_root=artifact_root,
        data={"universe_manifest": "data/universe.json"},
        evaluation={"folds": [{"id": "fold-1"}, {"id": "fold-2"}]},
    )
    config.data_path = lambda name: runtime_root / Path(*Path(config.data[name]).parts[1:])
    with pytest.raises(ValueError, match="one universe manifest identity"):
        _verify_frozen_universe_manifest(config)


@pytest.mark.parametrize(
    ("method", "seed"), [("raw", None), ("untrained_neural", None), ("dense", 13), ("sparse", 47)]
)
def test_learned_ledger_identity_never_uses_legacy_base(tmp_path, monkeypatch, method, seed):
    from types import SimpleNamespace

    from execsim.data.paper.manifests import file_sha256
    from execsim.ml.paper import orchestration

    config = SimpleNamespace(
        artifact_root=tmp_path,
        runtime_evaluation_root=tmp_path / "evaluation-executions" / "isolated",
        config_hash="frozen-config",
    )
    relative = "evaluation-v2/bases/fold-1/manifest.json"
    legacy = tmp_path / relative
    selected = config.runtime_evaluation_root / relative
    for path, content in (
        (legacy, "legacy"),
        (selected, "isolated"),
        (tmp_path / "selection/parameter-freeze-v1.json", "freeze"),
        (tmp_path / "lightgbm/fold-1" / method / str(seed or "shared") / "manifest.json", "model"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    if seed is not None:
        embedding = (
            tmp_path
            / "embeddings/fold-1"
            / method
            / str(seed)
            / "partition=test/embeddings.parquet"
        )
        embedding.parent.mkdir(parents=True)
        embedding.write_bytes(b"fixture-embedding")
    monkeypatch.setattr(orchestration, "_git_head", lambda: "a" * 40)
    monkeypatch.setattr(orchestration, "_git_tree", lambda: "b" * 40)
    identity = orchestration._learned_ledger_identity(config, "fold-1", method, seed)
    assert identity["base_manifest_sha256"] == file_sha256(selected)
    assert identity["base_manifest_sha256"] != file_sha256(legacy)
    selected.unlink()
    with pytest.raises(FileNotFoundError):
        orchestration._learned_ledger_identity(config, "fold-1", method, seed)


@pytest.mark.parametrize("tamper", ["current", "original", "freeze", "science", "yaml"])
def test_document_amendment_cannot_hide_changed_scientific_or_unrecorded_text(tmp_path, tamper):
    import json
    import shutil
    from pathlib import Path

    from execsim.ml.paper.configs import load_paper_config

    original = Path("configs/paper/sparse_jepa_v2")
    copied = tmp_path / original
    shutil.copytree(original, copied)
    freeze = json.loads((original / "design-freeze-v2.json").read_text())
    for relative in freeze["normative_document_sha256"]:
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(relative, destination)
    expected_hash = freeze["paper_config_sha256"]
    assert load_paper_config(copied).config_hash == expected_hash
    amendment_path = copied / "implementation-document-amendment.json"
    if tamper in {"original", "freeze"}:
        amendment = json.loads(amendment_path.read_text())
        if tamper == "freeze":
            amendment["design_freeze_sha256"] = "0" * 64
        else:
            amendment["documents"]["docs/SPECIFICATIONS.md"]["edits"][0]["original"] += "tampered"
        amendment_path.write_text(json.dumps(amendment))
    else:
        paths = {
            "current": "docs/SPECIFICATIONS.md",
            "science": "docs/PAPER_DESIGN.md",
            "yaml": "configs/paper/sparse_jepa_v2/lightgbm.yaml",
        }
        target = tmp_path / paths[tamper]
        if tamper == "yaml":
            target.write_text(target.read_text().replace("0.03", "0.04"))
        else:
            target.write_text(target.read_text() + "\nUnrecorded modification.\n")
    with pytest.raises(ValueError):
        load_paper_config(copied)


def test_forecast_ledger_matches_direct_provider_at_boundaries_and_partial_tokens(tmp_path):
    from datetime import date

    from execsim.ml.paper.evaluation_artifacts import publish_frames
    from execsim.ml.paper.forecast_ledger import PaperForecastLedgerProvider
    from execsim.ml.paper.forecast_provider import PaperLightGBMForecastProvider

    day, cutoff = date(2024, 4, 2), date(2023, 12, 29)
    scale = pd.DataFrame(
        {
            "sample_id": ["s24", "s25"],
            "fold_id": ["fold-1"] * 2,
            "instrument_id": ["A"] * 2,
            "symbol": ["AAA"] * 2,
            "session_date": [day] * 2,
            "training_cutoff": [cutoff] * 2,
            "as_of": [24, 25],
            "predicted_remaining_volume": [100.0, 40.0],
        }
    )
    shape = pd.DataFrame(
        {
            "case_id": ["s24", "s24", "s25"],
            "target_bucket": [24, 25, 25],
            "conditional_share": [0.6, 0.4, 1.0],
        }
    )
    identity = dict(
        fold_id="fold-1",
        method="dense",
        seed=13,
        parameter_freeze_sha256="f" * 64,
        model_manifest_sha256="m" * 64,
        source_commit="c" * 40,
        source_tree="t" * 40,
    )
    directory = tmp_path / "ledger"
    publish_frames(
        directory,
        identity=identity,
        frames={
            "scale.parquet": scale,
            "shape.parquet": shape,
            "metrics.parquet": pd.DataFrame({"sample_id": ["s24", "s25"]}),
        },
    )
    profile = np.arange(1, 16, dtype=float)
    profile /= profile.sum()
    options = dict(
        expected_identity=identity,
        instrument_id="A",
        session_date=day,
        within_token_profile=profile,
        training_cutoff=cutoff,
        sequence_hash="sequence",
    )
    ledger = PaperForecastLedgerProvider(directory, **options)

    class Model:
        def predict_frames(self, active_scale, active_shape, **kwargs):
            return active_scale["predicted_remaining_volume"].to_numpy(), active_shape

    def resolve(symbol, session_date, stamp, observations):
        origin = (stamp.hour * 60 + stamp.minute - 570) // 15
        selected = scale.loc[scale["as_of"] == origin]
        return selected, shape.loc[shape["case_id"].isin(selected["sample_id"])]

    direct = PaperLightGBMForecastProvider(
        Model(),
        feature_resolver=resolve,
        within_token_profile=profile,
        training_cutoff=cutoff,
        manifest_hash="sequence",
        method_id="dense-13",
    )
    start = pd.Timestamp("2024-04-02 15:30", tz="America/New_York")
    for offset in (0, 1, 7, 14, 15):
        stamp = start + pd.Timedelta(minutes=offset)
        request = dict(
            symbol="AAA",
            session_date=day,
            generated_at=stamp,
            bucket_timestamps=pd.date_range(stamp, start + pd.Timedelta(minutes=29), freq="min"),
        )
        assert ledger.forecast(**request) == direct.forecast(**request)
    for field, wrong in (
        ("fold_id", "fold-2"),
        ("seed", 29),
        ("method", "sparse"),
        ("parameter_freeze_sha256", "x" * 64),
        ("model_manifest_sha256", "y" * 64),
    ):
        with pytest.raises(ValueError, match="identity"):
            PaperForecastLedgerProvider(
                directory, **{**options, "expected_identity": {**identity, field: wrong}}
            )
    with pytest.raises(ValueError, match="as-of"):
        stamp = start - pd.Timedelta(minutes=15)
        ledger.forecast(
            symbol="AAA", session_date=day, generated_at=stamp, bucket_timestamps=[stamp]
        )
