from datetime import date

import numpy as np
import pandas as pd
import pytest

from execsim.ml.paper.evaluation_workers import compact_profile_corpus
from execsim.ml.paper.tca_inputs import prepare_tca_history, read_tca_date


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
