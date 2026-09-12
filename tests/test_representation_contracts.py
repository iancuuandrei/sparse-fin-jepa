"""Exercise regime metadata from real sequence production to frozen diagnostics."""

from dataclasses import asdict
from datetime import date

import pandas as pd
import pytest
from test_paper_representations import _checkpoint_compatibility
from test_paper_sequences import _paper_session

from execsim.data.paper.manifests import file_sha256, read_json, write_json_atomic
from execsim.ml.paper.lightgbm_data import build_historical_baseline_regime_frame
from execsim.ml.paper.orchestration import _stream_embedding_diagnostics
from execsim.ml.paper.regimes import fit_unusual_session_thresholds, label_unusual_sessions
from execsim.ml.representations.checkpoints import save_checkpoint
from execsim.ml.representations.embedding_pipeline import export_embedding_corpus
from execsim.ml.representations.jepa import PredictiveRepresentationModel
from execsim.ml.representations.schemas import CheckpointManifest, RepresentationConfig
from execsim.ml.sequences.corpus import build_fold_sequence_corpus
from execsim.ml.sequences.streaming import PaperSequenceDataset


@pytest.fixture
def regime_corpus(tmp_path):
    dates = [
        date.fromisoformat(value)
        for value in ("2023-12-27", "2023-12-28", "2024-01-03", "2024-04-02", "2024-04-03")
    ]
    instruments = (("asset-2", "BBB"), ("asset-1", "AAA"), ("benchmark-spy", "SPY"))
    bars = pd.concat(
        [
            _paper_session(instrument, symbol, day, index)
            for index, (instrument, symbol) in enumerate(instruments)
            for day in reversed(dates)
        ],
        ignore_index=True,
    )
    build_fold_sequence_corpus(
        bars,
        universe_members=tuple(
            {"instrument_id": instrument, "formation_symbol": symbol}
            for instrument, symbol in instruments[:-1]
        ),
        corporate_actions=pd.DataFrame(),
        fold_id="fold-1",
        output_root=tmp_path / "sequences",
        universe_manifest_hash="a" * 64,
        corporate_action_manifest_hash="b" * 64,
        config_hash="c" * 64,
        spy_instrument_id="benchmark-spy",
        data_classification="synthetic_fixture",
        quality_protocol="resolution-aware-v2",
    )
    return tmp_path / "sequences/fold-1/sequence-manifest.json"


def export_fixture_checkpoint(root, sequence, geometry):
    import torch

    representation = RepresentationConfig(geometry, seed=13)
    model = PredictiveRepresentationModel(representation)
    manifest = CheckpointManifest(
        checkpoint_id=f"fixture-{geometry}",
        geometry=geometry,
        predictor_family=representation.predictor_family,
        fold_id="fold-1",
        seed=13,
        sequence_manifest_hash=file_sha256(sequence),
        normalization_hash="b" * 64,
        cutoff="2023-12-29",
        architecture_hash="d" * 64,
        torch_version=torch.__version__,
        weights_sha256="",
        code_commit="fixture-only",
    )
    checkpoint = root / "representations/fold-1" / geometry / "13"
    save_checkpoint(model, checkpoint / "final", manifest)
    compatibility = _checkpoint_compatibility(manifest)
    write_json_atomic(checkpoint / "compatibility.json", asdict(compatibility))
    output = root / "embeddings/fold-1" / geometry / "13"
    export_embedding_corpus(
        model,
        checkpoint_directory=checkpoint / "final",
        expected_checkpoint=compatibility,
        sequence_manifest_path=sequence,
        output_root=output,
        seed=13,
        geometry=geometry,
        adaptation="none",
        device="cpu",
        batch_size=17,
    )
    return output


@pytest.mark.parametrize("geometry", ["dense", "sparse"])
def test_production_regime_to_embedding_diagnostics(tmp_path, regime_corpus, geometry):
    output = export_fixture_checkpoint(tmp_path, regime_corpus, geometry)
    training = build_historical_baseline_regime_frame(regime_corpus, partition="train")
    test = build_historical_baseline_regime_frame(regime_corpus, partition="test")
    states = label_unusual_sessions(test, fit_unusual_session_thresholds(training))
    embedding = output / "partition=test/embeddings.parquet"
    # Call the real consumer before asserting metadata: a9 reproduces its exact error.
    diagnostics, transitions, counts = _stream_embedding_diagnostics(embedding, states)
    dataset = PaperSequenceDataset(
        regime_corpus, partition="test", seed=13, sample_train_positions=False
    )
    canonical = [dataset[index] for index in range(len(dataset))]
    exported = pd.read_parquet(embedding)
    # The regime builder traverses index rows; the diagnostics consumer orders
    # instrument/date/origin. The export orders canonical session ID/origin.
    ordered_states = states.sort_values(
        ["instrument_id", "session_date", "as_of_token"], kind="stable"
    )
    assert ordered_states["sample_id"].tolist() == exported["sample_id"].tolist()
    assert exported["sample_id"].tolist() == [row["sample_id"] for row in canonical]
    assert ordered_states["session_id"].tolist() == [row["session_id"] for row in canonical]
    assert diagnostics["finite"] == 1.0
    assert transitions
    assert sum(counts.values()) == len(canonical)
    assert read_json(output / "manifest.json")["files"]


def test_diagnostics_reject_mismatched_production_identities(tmp_path, regime_corpus):
    output = export_fixture_checkpoint(tmp_path, regime_corpus, "sparse")
    training = build_historical_baseline_regime_frame(regime_corpus, partition="train")
    states = label_unusual_sessions(
        build_historical_baseline_regime_frame(regime_corpus, partition="test"),
        fit_unusual_session_thresholds(training),
    )
    path = output / "partition=test/embeddings.parquet"
    with pytest.raises(ValueError, match=r"missing columns.*session_id"):
        _stream_embedding_diagnostics(path, states.drop(columns="session_id"))
    with pytest.raises(ValueError, match="duplicates TEST sample"):
        _stream_embedding_diagnostics(path, pd.concat([states, states.iloc[:1]]))
    mismatched = states.copy()
    mismatched.loc[mismatched.index[0], "session_id"] = "another-session"
    with pytest.raises(ValueError, match="session identity contradicts"):
        _stream_embedding_diagnostics(path, mismatched)

    exported = pd.read_parquet(path)
    corruptions = {
        "wrong-order": (exported.iloc[::-1], "frozen sequence-index order"),
        "missing": (exported.iloc[:-1], "exactly match the frozen sequence index"),
        "duplicate": (pd.concat([exported, exported.iloc[:1]]), "duplicate or unexpected"),
        "extra": (
            pd.concat([exported, exported.iloc[:1].assign(sample_id="unexpected-sample")]),
            "duplicate or unexpected",
        ),
    }
    for name, (frame, message) in corruptions.items():
        corrupted = tmp_path / f"{name}.parquet"
        frame.to_parquet(corrupted, index=False)
        with pytest.raises(ValueError, match=message):
            _stream_embedding_diagnostics(corrupted, states)
