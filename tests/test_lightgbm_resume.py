from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from execsim.ml.models.lightgbm_adapter import LightGBMVolumeModel, run_lightgbm_grid


def test_candidate_interrupt_resume_preserves_selection_and_native_outputs(
    tmp_path: Path, monkeypatch
):
    scale = pd.DataFrame(
        {"symbol": ["AAA"] * 16, "x": np.arange(16), "baseline_remaining_volume": np.full(16, 10.0)}
    )
    shape = scale.loc[scale.index.repeat(2)].reset_index(drop=True)
    shape["case_id"] = np.repeat(np.arange(16), 2)
    shape["target_bucket"] = np.tile([0, 1], 16)
    shape["sample_weight"] = 0.5
    frames = (scale, np.arange(16) + 20.0, shape, np.tile([0.3, 0.7], 16))
    original = LightGBMVolumeModel.fit_frames
    calls = []

    def interrupted(self, *args, **kwargs):
        calls.append(self.config)
        if len(calls) == 4:
            raise KeyboardInterrupt("fixture interruption")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(LightGBMVolumeModel, "fit_frames", interrupted)
    options = dict(resume_directory=tmp_path / "candidates", resume_identity={"fold_id": "fold-1"})
    with pytest.raises(KeyboardInterrupt):
        run_lightgbm_grid(frames, frames, **options)
    assert len(list((tmp_path / "candidates").glob("candidate-*/manifest.json"))) == 3
    monkeypatch.setattr(LightGBMVolumeModel, "fit_frames", original)
    resumed, results = run_lightgbm_grid(frames, frames, **options)
    fresh, fresh_results = run_lightgbm_grid(frames, frames)
    assert results == fresh_results
    a, b = resumed.predict_frames(scale, shape, group_columns=("case_id",))
    c, d = fresh.predict_frames(scale, shape, group_columns=("case_id",))
    np.testing.assert_array_equal(a, c)
    pd.testing.assert_frame_equal(b, d)
    resumed.save_native(
        tmp_path / "selected",
        {
            "fold_id": "fold-1",
            "feature_schema_version": "fixture",
            "training_cutoff": "fixture",
            "validation_range": ["fixture"],
            "categorical_features": ["symbol"],
        },
    )
    result_file = next((tmp_path / "candidates").glob("candidate-*/grid-result.json"))
    result_file.write_text("{}")
    with pytest.raises(ValueError, match="checksum"):
        run_lightgbm_grid(frames, frames, **options)
