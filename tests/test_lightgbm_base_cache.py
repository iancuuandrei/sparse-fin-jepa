from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from execsim.ml.paper import lightgbm_data


def test_base_cache_reuses_exact_rows_without_rebuilding_and_rejects_corruption(
    tmp_path: Path, monkeypatch
) -> None:
    manifest = tmp_path / "sequence.json"
    manifest.write_text("{}", encoding="utf-8")
    scale = pd.DataFrame({"sample_id": ["b", "a"], "symbol": ["BBB", "AAA"]})
    shape = pd.DataFrame({"sample_id": ["b", "b", "a"], "sample_weight": [0.5, 0.5, 1.0]})
    expected = lightgbm_data.LightGBMFrames(
        scale, np.array([4.0, 8.0]), shape, np.array([0.5, 0.5, 1.0])
    )
    calls = []

    def build(*args, **kwargs):
        calls.append(kwargs["partition"])
        return expected

    monkeypatch.setattr(lightgbm_data, "build_lightgbm_base_frames", build)
    options = dict(
        partition="train",
        liquidity_groups={"a": 1},
        cache_directory=tmp_path / "cache",
        source_commit="a" * 40,
        config_hash="b" * 64,
    )
    lightgbm_data.cached_lightgbm_base_frames(manifest, **options)
    actual = lightgbm_data.cached_lightgbm_base_frames(manifest, **options)
    assert calls == ["train"]
    pd.testing.assert_frame_equal(actual.scale, expected.scale)
    pd.testing.assert_frame_equal(actual.shape, expected.shape)
    np.testing.assert_array_equal(actual.scale_target, expected.scale_target)
    np.testing.assert_array_equal(actual.shape_target, expected.shape_target)
    with pytest.raises(ValueError, match="TRAIN/VALIDATION"):
        lightgbm_data.cached_lightgbm_base_frames(manifest, **{**options, "partition": "test"})
    (tmp_path / "cache" / "shape.npy").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        lightgbm_data.cached_lightgbm_base_frames(manifest, **options)
