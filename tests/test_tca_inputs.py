from datetime import date
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from execsim.data.paper.manifests import read_json
from execsim.ml.paper.evaluation_workers import compact_profile_corpus
from execsim.ml.paper.tca_inputs import causal_adv20, prepare_tca_history, read_tca_date


def test_compact_tca_history_preserves_bars_adv_profiles_and_resumes(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    frames = []
    for index, day in enumerate(pd.bdate_range("2024-01-02", periods=35)):
        for instrument in ("A", "B", "unused"):
            periods = 210 if day.date() == date(2024, 2, 8) else 390
            stamps = pd.date_range(
                day + pd.Timedelta(hours=9, minutes=30),
                periods=periods,
                freq="min",
                tz="America/New_York",
            )
            frames.append(
                pd.DataFrame(
                    {
                        "instrument_id": instrument,
                        "symbol": instrument,
                        "timestamp": stamps,
                        "volume": 1000.0 + index * (np.arange(periods) % 15 + 1),
                        "open": 100.0,
                        "high": 101.0,
                        "low": 99.0,
                        "close": 100.2,
                        "trade_count": 10,
                        "vwap": 100.1,
                    }
                )
            )
    original = pd.concat(frames, ignore_index=True)
    for index, chunk in enumerate(np.array_split(np.arange(len(original)), 7)):
        original.iloc[chunk].to_parquet(source / f"{index}.parquet", index=False)
    kwargs = dict(
        identity={"paper_config_hash": "fixture"},
        include_market_bars=True,
        selected_instruments=("A", "B"),
    )
    compact = compact_profile_corpus(source, tmp_path / "compact", **kwargs)
    assert set(compact) == {"A", "B"}
    assert compact_profile_corpus(source, tmp_path / "compact", **kwargs) == compact
    cutoffs = {"fold-a": date(2024, 1, 30), "fold-b": date(2024, 2, 5)}
    histories = {}
    for instrument, path in compact.items():
        histories[instrument] = prepare_tca_history(
            path,
            tmp_path / instrument,
            instrument_id=instrument,
            cutoffs=cutoffs,
            identity={"paper_config_hash": "fixture"},
        )
        before = (tmp_path / instrument / "manifest.json").stat().st_mtime_ns
        prepare_tca_history(
            path,
            tmp_path / instrument,
            instrument_id=instrument,
            cutoffs=cutoffs,
            identity={"paper_config_hash": "fixture"},
        )
        assert (tmp_path / instrument / "manifest.json").stat().st_mtime_ns == before
        stored = pd.read_parquet(tmp_path / instrument / "profiles.parquet").set_index("fold_id")
        for fold, cutoff in cutoffs.items():
            selected = original.loc[
                (original["instrument_id"] == instrument)
                & (original["timestamp"].dt.date <= cutoff)
            ].copy()
            selected["minute"] = (
                selected["timestamp"].dt.hour * 60 + selected["timestamp"].dt.minute - 570
            ) % 15
            means = selected.groupby("minute")["volume"].mean().reindex(range(15)).to_numpy()
            np.testing.assert_array_equal(stored.loc[fold, "profile"], means / means.sum())
    day = date(2024, 2, 9)
    actual_bars, actual_adv = read_tca_date(histories, day)
    selected = original.loc[original["instrument_id"].isin(["A", "B"])].copy()
    selected["session_date"] = selected["timestamp"].dt.date
    expected_bars = selected.loc[selected["session_date"] == day].sort_values(
        ["instrument_id", "timestamp"]
    )
    pd.testing.assert_frame_equal(
        actual_bars, expected_bars.reset_index(drop=True)[actual_bars.columns]
    )
    daily = selected.groupby(["instrument_id", "session_date"], sort=True, as_index=False)[
        "volume"
    ].sum()
    daily["adv20"] = daily.groupby("instrument_id")["volume"].transform(
        lambda s: s.shift(1).rolling(20, min_periods=20).mean()
    )
    pd.testing.assert_frame_equal(
        actual_adv, daily.loc[daily["session_date"] == day].reset_index(drop=True)
    )
    with pytest.raises(ValueError, match="identity"):
        prepare_tca_history(
            compact["A"],
            histories["A"],
            instrument_id="A",
            cutoffs={"fold-a": date(2024, 2, 1)},
            identity={"paper_config_hash": "fixture"},
        )

    # Exercise the production orchestrator without allowing its former full-corpus loader.
    import json
    from pathlib import Path
    from types import SimpleNamespace

    import yaml

    from execsim.data.paper.manifests import file_sha256
    from execsim.ml.paper import orchestration
    from execsim.ml.paper.evaluation_artifacts import publish_frames

    root = tmp_path / "artifacts"
    (root / "selection").mkdir(parents=True)
    freeze = root / "selection/parameter-freeze-v1.json"
    freeze.write_text("{}")
    universe = tmp_path / "universe.json"
    universe.write_text(
        json.dumps(
            {
                "members": [
                    {"instrument_id": "A", "rank": 1},
                    {"instrument_id": "B", "rank": 2},
                ]
            }
        )
    )
    sequence = root / "sequences/fold-1/sequence-manifest.json"
    sequence.parent.mkdir(parents=True)
    sequence.write_text(json.dumps({"universe_manifest_hash": file_sha256(universe)}))
    execution = {
        "source_commit": "fixture",
        "source_tree": "fixture",
        "paper_config_hash": "fixture",
        "parameter_freeze_sha256": file_sha256(freeze),
    }
    compact_profile_corpus(source, root / "evaluation-v2/profile-corpus", identity=execution)
    tca = yaml.safe_load(Path("configs/paper/sparse_jepa_v2/tca.yaml").read_text())
    tca.update(universe_size=2, sensitivity_universe_size=2)
    config = SimpleNamespace(
        authorize=lambda *args, **kwargs: None,
        artifact_root=root,
        config_hash="fixture",
        data={"target_corpus_root": str(source), "universe_manifest": str(universe)},
        representation={"seeds": []},
        tca=tca,
        evaluation={
            "folds": [
                {
                    "id": "fold-1",
                    "train": ["2024-01-02", "2024-01-30"],
                    "test": ["2024-02-07", "2024-02-09"],
                }
            ]
        },
    )
    config.data_path = lambda name: Path(config.data[name])
    base = root / "evaluation-v2/bases/fold-1"
    base.mkdir(parents=True)
    (base / "manifest.json").write_text("{}")
    for name in ("_require_parameter_freeze", "_require_locked_test_opened"):
        monkeypatch.setattr(orchestration, name, lambda *args: {})
    for name in ("_git_head", "_git_tree"):
        monkeypatch.setattr(orchestration, name, lambda: "fixture")
    monkeypatch.setattr(orchestration, "_learned_ledger_identity", lambda *args: {})

    def forbidden(*args):
        raise AssertionError("TCA must not load the entire corpus")

    monkeypatch.setattr(orchestration, "_load_parquet_corpus", forbidden)
    seen = []

    def workers(tasks):
        for task in tasks:
            current = pd.Timestamp(task.identity["session_date"]).date()
            seen.append(current)
            actual = pd.read_parquet(task.input_directory / "bars.parquet")
            reference = selected.loc[selected["session_date"] == current].sort_values(
                ["instrument_id", "timestamp"]
            )
            pd.testing.assert_frame_equal(actual, reference.reset_index(drop=True)[actual.columns])
            pd.testing.assert_frame_equal(
                pd.read_parquet(task.input_directory / "adv.parquet"),
                daily.loc[daily["session_date"] == current].reset_index(drop=True),
            )
            result = pd.DataFrame(
                {
                    "fold_id": "fold-1",
                    "date": [current],
                    "instrument_id": "A",
                    "method": "ewma",
                    "order_fraction_adv20": 0.03,
                }
            )
            publish_frames(
                task.output_directory,
                identity=task.identity,
                frames={"main.parquet": result, "sensitivity.parquet": result},
            )
        return [task.output_directory for task in tasks]

    monkeypatch.setattr("execsim.ml.paper.tca_workers.run_tca_workers", workers)
    # This fixture isolates corpus/date slicing; ledger preflight has its own
    # fail-closed regression tests and is not populated by this small harness.
    monkeypatch.setattr("execsim.ml.paper.tca_workers.preflight_tca_ledgers", lambda **kwargs: None)
    result = orchestration.run_tca_stage(config, full_run_cli_enabled=True, runtime_approval=None)
    assert result["status"] == "SOFTWARE READY"
    # The production orchestrator must exclude the early-close date before
    # constructing a worker task; valid dates remain unchanged.
    assert seen == [date(2024, 2, 7), date(2024, 2, 9)]


def _adv_fixture(*, post_volume: float = 100.0) -> pd.DataFrame:
    """Build 22 full sessions with a stable raw share-volume transition."""
    frames = []
    for index, day in enumerate(pd.bdate_range("2025-01-02", periods=22)):
        timestamps = pd.date_range(
            day + pd.Timedelta(hours=9, minutes=30),
            periods=390,
            freq="min",
            tz="America/New_York",
        )
        volume = 100.0 if index < 20 else post_volume
        frames.append(
            pd.DataFrame(
                {
                    "instrument_id": "X",
                    "symbol": "X",
                    "timestamp": timestamps,
                    "volume": volume,
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.0,
                    "trade_count": 1,
                    "vwap": 100.0,
                }
            )
        )
    return pd.concat(frames, ignore_index=True)


def _action(*, effective: str, available: str, factor: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "instrument_id": "X",
                "effective_date": effective,
                "available_at": available,
                "factor": factor,
                "source": "fixture",
            }
        ]
    )


def test_causal_adv20_forward_split_uses_target_execution_share_basis() -> None:
    bars = _adv_fixture(post_volume=500.0)
    actions = _action(effective="2025-01-30", available="2025-01-30T00:00:00Z", factor=0.2)
    corrected = causal_adv20(bars, corporate_actions=actions)
    first_post = corrected.loc[corrected["session_date"] == date(2025, 1, 30), "adv20"]
    assert first_post.iloc[0] == pytest.approx(195000.0)
    # The replay remains raw; only historical ADV contributions are restated.
    assert bars.loc[bars["timestamp"].dt.date == date(2025, 1, 30), "volume"].sum() == 195000


def test_causal_adv20_reverse_split_and_multiple_actions_are_cumulative() -> None:
    bars = _adv_fixture(post_volume=10.0)
    reverse = _action(effective="2025-01-30", available="2025-01-30T00:00:00Z", factor=10.0)
    corrected = causal_adv20(bars, corporate_actions=reverse)
    actual = corrected.loc[corrected["session_date"] == date(2025, 1, 30), "adv20"].iloc[0]
    assert actual == pytest.approx(3900.0)
    multiple = pd.concat(
        [
            _action(effective="2025-01-02", available="2025-01-02T00:00:00Z", factor=0.5),
            _action(effective="2025-01-30", available="2025-01-30T00:00:00Z", factor=0.2),
        ],
        ignore_index=True,
    )
    cumulative = causal_adv20(_adv_fixture(), corporate_actions=multiple)
    actual = cumulative.loc[cumulative["session_date"] == date(2025, 1, 30), "adv20"].iloc[0]
    assert actual == pytest.approx(195000.0)


def test_causal_adv20_respects_effective_and_information_clocks() -> None:
    bars = _adv_fixture()
    announced = _action(effective="2025-01-30", available="2025-01-02T00:00:00Z", factor=0.2)
    unknown = _action(effective="2025-01-30", available="2025-01-31T00:00:00Z", factor=0.2)
    not_effective = _action(effective="2025-02-03", available="2025-01-02T00:00:00Z", factor=0.2)
    announced_result = causal_adv20(bars, corporate_actions=announced)
    unknown_result = causal_adv20(bars, corporate_actions=unknown)
    not_effective_result = causal_adv20(bars, corporate_actions=not_effective)
    # Effective and known at the 10:30 decision changes the target basis.
    assert announced_result.loc[
        announced_result["session_date"] == date(2025, 1, 30), "adv20"
    ].iloc[0] == pytest.approx(195000.0)
    # An effective action not known until the following day cannot be used early.
    assert unknown_result.loc[unknown_result["session_date"] == date(2025, 1, 30), "adv20"].iloc[
        0
    ] == pytest.approx(39000.0)
    # An announced action that is not yet effective cannot alter the target basis.
    assert not_effective_result.loc[
        not_effective_result["session_date"] == date(2025, 1, 30), "adv20"
    ].iloc[0] == pytest.approx(39000.0)


def test_causal_adv20_excludes_target_volume() -> None:
    actions = _action(effective="2025-01-30", available="2025-01-30T00:00:00Z", factor=0.2)
    baseline = causal_adv20(_adv_fixture(post_volume=500.0), corporate_actions=actions)
    changed_target = causal_adv20(_adv_fixture(post_volume=999999.0), corporate_actions=actions)
    pd.testing.assert_series_equal(
        baseline.loc[baseline["session_date"] == date(2025, 1, 30), "adv20"].reset_index(drop=True),
        changed_target.loc[
            changed_target["session_date"] == date(2025, 1, 30), "adv20"
        ].reset_index(drop=True),
    )


def test_runtime_corporate_action_manifest_relocation_and_history_identity(tmp_path) -> None:
    import json

    from execsim.data.paper.corporate_action_manifest import write_corporate_action_manifest
    from execsim.data.paper.manifests import file_sha256
    from execsim.ml.paper.orchestration import _verify_frozen_corporate_action_manifest

    runtime = tmp_path / "runtime-data"
    action_dir = runtime / "data/paper/corporate_actions_v2"
    action_dir.mkdir(parents=True)
    source = action_dir / "actions.parquet"
    actions = _action(effective="2025-01-30", available="2025-01-30T00:00:00Z", factor=0.2)
    actions.to_parquet(source, index=False)
    manifest = action_dir / "manifest.json"
    write_corporate_action_manifest(source, actions, manifest, paper_config_hash="fixture")
    artifact = tmp_path / "artifacts"
    sequence = artifact / "sequences/fold-1/sequence-manifest.json"
    sequence.parent.mkdir(parents=True)
    sequence.write_text(json.dumps({"corporate_action_manifest_hash": file_sha256(manifest)}))
    config = SimpleNamespace(
        data={
            "corporate_action_source": "data/paper/corporate_actions_v2/actions.parquet",
            "corporate_action_manifest": "data/paper/corporate_actions_v2/manifest.json",
        },
        runtime_data_root=runtime,
        artifact_root=artifact,
        config_hash="fixture",
        evaluation={"folds": [{"id": "fold-1"}]},
    )
    config.data_path = lambda name: config.runtime_data_root / config.data[name]
    _, loaded, manifest_hash = _verify_frozen_corporate_action_manifest(config)
    assert manifest_hash == file_sha256(manifest)
    assert loaded.iloc[0]["instrument_id"] == "X"
    assert loaded.iloc[0]["effective_date"] == date(2025, 1, 30)
    assert loaded.iloc[0]["factor"] == pytest.approx(0.2)

    market = tmp_path / "market.parquet"
    _adv_fixture().to_parquet(market, index=False)
    history = prepare_tca_history(
        market,
        tmp_path / "history",
        instrument_id="X",
        cutoffs={"fold-1": date(2025, 1, 10)},
        identity={"paper_config_hash": "fixture"},
        corporate_actions=loaded,
        corporate_action_manifest_sha256=manifest_hash,
    )
    assert (
        read_json(history / "manifest.json")["identity"]["corporate_action_manifest_sha256"]
        == manifest_hash
    )
    with pytest.raises(ValueError, match="identity"):
        prepare_tca_history(
            market,
            history,
            instrument_id="X",
            cutoffs={"fold-1": date(2025, 1, 10)},
            identity={"paper_config_hash": "fixture"},
            corporate_actions=loaded,
            corporate_action_manifest_sha256="0" * 64,
        )
