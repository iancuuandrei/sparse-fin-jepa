"""Exercise the final TCA stage against canonical synthetic artifacts."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import yaml
from test_tca_production_contracts import _forecast_inputs, _market_bars

from execsim.data.paper.manifests import file_sha256, read_json, write_json_atomic
from execsim.ml.paper import orchestration
from execsim.ml.paper.evaluation_artifacts import forecast_metric_frame, publish_frames
from execsim.ml.paper.evaluation_workers import (
    EWMAWork,
    compact_profile_corpus,
    instrument_key,
    run_ewma_work,
)


def _write_stage_identity_inputs(config: SimpleNamespace) -> None:
    """Create the small immutable inputs used to derive canonical ledger identities."""
    root = config.artifact_root
    write_json_atomic(
        root / "selection" / "parameter-freeze-v1.json",
        {"fixture": "synthetic-parameter-freeze"},
    )
    for method in ("raw", "untrained_neural"):
        manifest = root / "lightgbm" / "fold-1" / method / "shared" / "manifest.json"
        write_json_atomic(manifest, {"fixture": "synthetic-learned-ledger", "method": method})


def _stage_config(tmp_path: Path, source: Path) -> SimpleNamespace:
    root = tmp_path / "artifacts"
    universe = tmp_path / "universe.json"
    write_json_atomic(
        universe,
        {
            "members": [{"instrument_id": "instrument-contract", "rank": 1}],
        },
    )
    sequence = root / "sequences" / "fold-1" / "sequence-manifest.json"
    write_json_atomic(sequence, {"universe_manifest_hash": file_sha256(universe)})

    tca = yaml.safe_load(Path("configs/paper/sparse_jepa_v2/tca.yaml").read_text())
    # The production configuration stays unchanged; this synthetic universe has one member.
    tca.update(universe_size=1, sensitivity_universe_size=1)
    config = SimpleNamespace(
        artifact_root=root,
        config_hash="synthetic-tca-stage-contract",
        data={
            "target_corpus_root": str(source),
            "universe_manifest": str(universe),
        },
        evaluation={
            "folds": [
                {
                    "id": "fold-1",
                    "train": ["2024-01-02", "2024-01-31"],
                    "test": ["2024-02-01", "2024-02-01"],
                }
            ]
        },
        representation={"seeds": []},
        tca=tca,
        # Full-run authorization is deliberately test-local for synthetic-only inputs.
        authorize=lambda *args, **kwargs: None,
    )
    config.data_path = lambda name: Path(config.data[name])
    return config


def _publish_synthetic_forecasts(
    config: SimpleNamespace, bars: pd.DataFrame, training_cutoff: date, session_date: date
) -> None:
    base, origins, baseline, history_shape = _forecast_inputs(
        bars, training_cutoff=training_cutoff, session_date=session_date
    )
    base_directory = config.artifact_root / "evaluation-v2" / "bases" / "fold-1"
    base_scale = base.scale.copy()
    base_scale["fold_id"] = "fold-1"
    base_scale = base_scale.assign(__evaluation_target=base.scale_target)
    base_shape = base.shape.assign(__evaluation_target=base.shape_target)
    publish_frames(
        base_directory,
        identity={"fold_id": "fold-1", "fixture": "synthetic-tca-base"},
        frames={"scale-base.parquet": base_scale, "shape-base.parquet": base_shape},
    )

    for method, total_factor, shape_power in (
        ("raw", 1.0, 1.0),
        ("untrained_neural", 0.97, 0.85),
    ):
        identity = orchestration._learned_ledger_identity(config, "fold-1", method, None)
        totals = baseline * total_factor
        predicted_chunks = []
        offset = 0
        for origin in origins:
            size = 26 - int(origin)
            shaped = np.power(history_shape[offset : offset + size], shape_power)
            predicted_chunks.append(shaped / shaped.sum())
            offset += size
        predicted_shape = base.shape.loc[:, ["case_id", "target_bucket"]].copy()
        predicted_shape["conditional_share"] = np.concatenate(predicted_chunks)
        metrics = forecast_metric_frame(
            base,
            totals,
            predicted_shape,
            fold_id="fold-1",
            method=method,
            seed=None,
        )
        scale = base.scale.loc[
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
        scale["fold_id"] = "fold-1"
        scale["predicted_remaining_volume"] = totals
        directory = (
            config.artifact_root / "evaluation-v2" / "forecasts" / "fold-1" / method / "shared"
        )
        publish_frames(
            directory,
            identity=identity,
            frames={
                "scale.parquet": scale,
                "shape.parquet": predicted_shape.sort_values(
                    ["case_id", "target_bucket"], kind="stable"
                ).reset_index(drop=True),
                "metrics.parquet": metrics,
            },
        )


def test_tca_stage_runs_real_preflight_ewma_providers_replay_and_merge(tmp_path, monkeypatch):
    bars = _market_bars()
    source = tmp_path / "source" / "bars.parquet"
    source.parent.mkdir(parents=True)
    bars.to_parquet(source, index=False)
    session_dates = sorted(pd.to_datetime(bars["timestamp"]).dt.date.unique())
    training_cutoff, session_date = session_dates[-2:]
    config = _stage_config(tmp_path, source)
    _write_stage_identity_inputs(config)
    _publish_synthetic_forecasts(config, bars, training_cutoff, session_date)
    market_profiles = compact_profile_corpus(
        source,
        config.artifact_root / "evaluation-v2" / "profile-corpus",
        identity={
            "source_commit": orchestration._git_head(),
            "source_tree": orchestration._git_tree(),
            "paper_config_hash": config.config_hash,
            "parameter_freeze_sha256": file_sha256(
                config.artifact_root / "selection" / "parameter-freeze-v1.json"
            ),
        },
    )
    ewma_work = EWMAWork(
        config.artifact_root / "evaluation-v2" / "bases" / "fold-1",
        market_profiles["instrument-contract"],
        config.artifact_root
        / "evaluation-v2"
        / "forecasts"
        / "fold-1"
        / "ewma"
        / instrument_key("instrument-contract"),
        "instrument-contract",
        {
            "source_commit": orchestration._git_head(),
            "source_tree": orchestration._git_tree(),
            "paper_config_hash": config.config_hash,
            "parameter_freeze_sha256": file_sha256(
                config.artifact_root / "selection" / "parameter-freeze-v1.json"
            ),
            "fold_id": "fold-1",
        },
    )
    run_ewma_work(ewma_work)
    # These locked-test prerequisites are covered independently; this regression
    # isolates the production TCA path using synthetic-only artifacts.
    monkeypatch.setattr(orchestration, "_require_parameter_freeze", lambda _config: {})
    monkeypatch.setattr(orchestration, "_require_locked_test_opened", lambda _config: {})
    monkeypatch.setenv("EXECSIM_EVALUATION_WORKERS", "1")

    result = orchestration.run_tca_stage(config, full_run_cli_enabled=True, runtime_approval=None)

    assert result["status"] == "SOFTWARE READY"
    main = pd.read_parquet(result["main"])
    sensitivity = pd.read_parquet(result["sensitivity"])
    assert set(main["method"]) == {"ewma", "lightgbm_raw", "raw_untrained_neural"}
    assert len(main) == 3
    assert len(sensitivity) == 6
    assert main["status"].eq("AVAILABLE").all()
    assert not main.duplicated(
        ["fold_id", "date", "instrument_id", "method", "order_fraction_adv20"]
    ).any()

    root = config.artifact_root / "evaluation-v2"
    market_manifest = read_json(root / "tca-market" / "manifest.json")
    assert market_manifest["identity"]["selected_instruments"] == ["instrument-contract"]
    input_manifest = read_json(
        root / "tca-inputs" / "fold-1" / session_date.isoformat() / "manifest.json"
    )
    assert input_manifest["identity"]["session_date"] == session_date.isoformat()
    shard_manifest = read_json(
        root / "tca-shards" / "fold-1" / session_date.isoformat() / "manifest.json"
    )
    assert shard_manifest["identity"]["sequence_hash"] == file_sha256(
        config.artifact_root / "sequences" / "fold-1" / "sequence-manifest.json"
    )
    assert [entry["method"] for entry in shard_manifest["identity"]["ledgers"]] == [
        "raw",
        "untrained_neural",
    ]
    assert set(shard_manifest["identity"]["ewma_ledgers"]) == {"instrument-contract"}
