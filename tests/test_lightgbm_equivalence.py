from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from reference_lightgbm_builder import (
    build_historical_baseline_regime_frame as old_regime_builder,
)
from reference_lightgbm_builder import build_lightgbm_frames as old_builder
from test_paper_sequences import _bars

from execsim.data.paper.manifests import write_json_atomic
from execsim.ml.paper.lightgbm_data import (
    build_historical_baseline_regime_frame,
    build_lightgbm_frames,
)
from execsim.ml.sequences.builder import build_session_sequence
from execsim.ml.sequences.index import build_sample_index, write_sample_index
from execsim.ml.sequences.manifests import write_sequence_record


@pytest.mark.parametrize("partition", ["train", "validation", "test"])
def test_frozen_builder_matches_raw_and_hybrid_rows(tmp_path: Path, partition: str) -> None:
    record = build_session_sequence(
        _bars(),
        instrument_id="asset-1",
        symbol="AAPL",
        source_sha256="a" * 64,
        cutoff="2024-01-02",
        spy_bars=_bars("SPY"),
        data_classification="synthetic_fixture",
    )
    if partition == "train":
        record = replace(
            record,
            session_date="2023-12-28",
            available_at_ns=record.available_at_ns - 6 * 86400 * 10**9,
            cutoff="2023-12-27",
            training_cutoff="2023-12-29",
            market_information_as_of="2023-12-28T09:30:00-05:00",
            feature_history_end="2023-12-27",
        )
    elif partition == "test":
        record = replace(
            record,
            session_date="2024-04-02",
            available_at_ns=record.available_at_ns + 90 * 86400 * 10**9,
            cutoff="2024-04-01",
            training_cutoff="2023-12-29",
            market_information_as_of="2024-04-02T09:30:00-04:00",
            feature_history_end="2024-04-01",
        )
    samples = build_sample_index(
        record, fold_id="fold-1", partition=partition, source_sequence_hash="b" * 64
    )
    session = write_sequence_record(record, tmp_path / "sessions" / partition)
    second = replace(
        record,
        session_id="asset-2-session",
        instrument_id="asset-2",
        symbol="BBB",
        features=record.features + np.float32(0.25),
    )
    second_path = write_sequence_record(second, tmp_path / "sessions" / partition)
    samples += build_sample_index(
        second, fold_id="fold-1", partition=partition, source_sequence_hash="c" * 64
    )
    index = write_sample_index(samples, tmp_path / "indexes" / partition / "index.parquet")
    manifest = tmp_path / "sequence-manifest.json"
    write_json_atomic(
        manifest,
        {
            "sequence_files": [
                session.relative_to(tmp_path).as_posix(),
                second_path.relative_to(tmp_path).as_posix(),
            ],
            "index_files": [index.relative_to(tmp_path).as_posix()],
        },
    )
    embeddings = tmp_path / "embeddings.parquet"
    rng = np.random.default_rng(13)
    pd.DataFrame(
        {
            "sample_id": [s.sample_id for s in samples],
            "embedding": list(rng.normal(size=(len(samples), 644)).astype(np.float32)),
        }
    ).iloc[::-1].to_parquet(embeddings)
    for embedding_path in (None, embeddings):
        options = dict(
            partition=partition,
            liquidity_groups={"asset-1": 1, "asset-2": 2},
            embedding_path=embedding_path,
        )
        before = old_builder(manifest, **options)
        after = build_lightgbm_frames(manifest, **options)
        for position in (0, 2):
            pd.testing.assert_frame_equal(before[position], after[position], check_exact=True)
        np.testing.assert_array_equal(before[1], after[1])
        np.testing.assert_array_equal(before[3], after[3])
    regime = build_historical_baseline_regime_frame(manifest, partition=partition)
    # Legacy science remains exact; its omitted canonical metadata was a defect,
    # not part of the frozen estimator. Check that identity independently.
    pd.testing.assert_frame_equal(
        old_regime_builder(manifest, partition=partition),
        regime.drop(columns="session_id"),
        check_exact=True,
    )
    sessions = {sample.sample_id: sample.session_id for sample in samples}
    assert regime["session_id"].tolist() == regime["sample_id"].map(sessions).tolist()
