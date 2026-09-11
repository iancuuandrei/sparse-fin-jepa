"""Coordinate completion avoids repeating frozen representation evaluation."""

from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from execsim.data.paper.manifests import write_json_atomic
from execsim.ml.paper import orchestration


@pytest.mark.parametrize("interrupted", [False, True])
def test_representation_coordinates_resume_without_model_loading(
    tmp_path, monkeypatch, interrupted
):
    pytest.importorskip("torch")
    from execsim.ml.representations import checkpoints, frozen_evaluation, jepa, schemas
    from execsim.ml.sequences import streaming

    config = SimpleNamespace(
        artifact_root=tmp_path,
        runtime_evaluation_root=tmp_path / "new-evaluation",
        runtime_representation_root=None,
        config_hash="science",
        evaluation={"folds": [{"id": "fold-1"}]},
        representation={
            "seeds": [13],
            "rdm_projections_evaluation": 2048,
            "batch_size": 4,
            "probe_ridge_alphas": [1.0],
            "probe_mlp_epochs": 1,
        },
        sequences={"session_cache_size": 2, "num_workers": 0, "prefetch_factor": 1},
        authorize=lambda *args, **kwargs: None,
    )
    for name in ("_require_parameter_freeze", "_require_locked_test_opened"):
        monkeypatch.setattr(orchestration, name, lambda *args: None)
    monkeypatch.setattr(orchestration, "_git_head", lambda: "source")
    monkeypatch.setattr(orchestration, "_git_tree", lambda: "tree")
    write_json_atomic(tmp_path / "selection/parameter-freeze-v1.json", {"frozen": True})
    write_json_atomic(tmp_path / "sequences/fold-1/sequence-manifest.json", {"fold": "fold-1"})
    for method in ("dense", "sparse"):
        root = tmp_path / "representations/fold-1" / method / "13"
        write_json_atomic(root / "final/manifest.json", {"model": method})
        write_json_atomic(
            root / "compatibility.json",
            {
                "geometry": method,
                "predictor_family": "P0",
                "generalized_gaussian_p": 2.0,
                "generalized_gaussian_mu": 0.0,
                "generalized_gaussian_sigma": 1.0,
                "rdm_projections": 16,
            },
        )
        write_json_atomic(tmp_path / "embeddings/fold-1" / method / "13/manifest.json", {})
    monkeypatch.setattr(
        schemas, "CheckpointCompatibility", lambda **kwargs: SimpleNamespace(**kwargs)
    )
    monkeypatch.setattr(schemas, "RepresentationConfig", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        jepa,
        "PredictiveRepresentationModel",
        lambda *args: SimpleNamespace(to=lambda device: object()),
    )
    loaded = []
    monkeypatch.setattr(
        checkpoints, "load_checkpoint", lambda *args, **kwargs: loaded.append(args[1])
    )
    monkeypatch.setattr(streaming, "PaperSequenceDataset", lambda *args, **kwargs: None)
    monkeypatch.setattr(streaming, "build_sequence_dataloader", lambda *args, **kwargs: [])
    import execsim.ml.paper.lightgbm_data as data
    import execsim.ml.paper.regimes as regimes

    monkeypatch.setattr(
        data, "build_historical_baseline_regime_frame", lambda *args, **kwargs: pd.DataFrame()
    )
    monkeypatch.setattr(regimes, "fit_unusual_session_thresholds", lambda *args: {})
    monkeypatch.setattr(regimes, "label_unusual_sessions", lambda *args: pd.DataFrame())
    monkeypatch.setattr(
        orchestration,
        "_stream_embedding_diagnostics",
        lambda *args: ({"zero_fraction": 0.75, "mean_active_dimensions": 32.0}, {}, {}),
    )
    calls = []

    def evaluate(*args, **kwargs):
        calls.append(1)
        if interrupted and len(calls) == 2:
            raise RuntimeError("synthetic interruption")
        row = {"probe_capacity": "affine", "horizon": 1, "error": 0.2}
        return [row], [row], [{**row, "date": "2023-01-03", "row_count": 1}]

    monkeypatch.setattr(frozen_evaluation, "evaluate_frozen_capacity_streaming", evaluate)
    if interrupted:
        with pytest.raises(RuntimeError, match="synthetic interruption"):
            orchestration.evaluate_representations_stage(
                config, full_run_cli_enabled=True, runtime_approval=None
            )
    result = orchestration.evaluate_representations_stage(
        config, full_run_cli_enabled=True, runtime_approval=None
    )
    expected_calls = 3 if interrupted else 2
    assert len(calls) == len(loaded) == expected_calls
    paths = [Path(result[key]) for key in ("accessibility", "date_metrics", "support_regimes")]
    before = [path.read_bytes() for path in paths]
    orchestration.evaluate_representations_stage(
        config, full_run_cli_enabled=True, runtime_approval=None
    )
    assert len(calls) == len(loaded) == expected_calls
    assert [path.read_bytes() for path in paths] == before
    assert len(pd.read_parquet(paths[0])) == 2
    assert len(pd.read_parquet(paths[2])) == 1
    shard = (
        config.runtime_evaluation_root
        / "evaluation-v2/representations/fold-1/dense/13/accessibility.parquet"
    )
    shard.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum mismatch"):
        orchestration.evaluate_representations_stage(
            config, full_run_cli_enabled=True, runtime_approval=None
        )
    assert len(calls) == len(loaded) == expected_calls
