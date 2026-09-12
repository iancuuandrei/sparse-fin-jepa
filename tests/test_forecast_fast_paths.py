from __future__ import annotations

from time import perf_counter

import numpy as np
import pandas as pd
import pytest

from execsim.ml.models.lightgbm_adapter import (
    LightGBMVolumeModel,
    _contiguous_group_boundaries,
)
from execsim.ml.paper.evaluation_artifacts import (
    _can_use_aligned_shape_keys,
    forecast_metric_frame,
    validate_base,
)
from execsim.ml.paper.lightgbm_data import LightGBMFrames


class _FixedPredictor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        assert len(features) == len(self.values)
        return self.values.copy()


def _model_with_logits(logits: list[float] | np.ndarray) -> LightGBMVolumeModel:
    model = LightGBMVolumeModel()
    model.feature_columns = ("x",)
    model.shape_feature_columns = ("x",)
    model.scale_model = _FixedPredictor(np.zeros(2, dtype=float))
    model.shape_model = _FixedPredictor(np.asarray(logits, dtype=float))
    return model


def _legacy_shape_prediction(
    logits: np.ndarray,
    shape: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
    valid: np.ndarray,
) -> pd.DataFrame:
    output = shape.loc[:, [*group_columns, "target_bucket"]].copy().reset_index(drop=True)
    output["conditional_share"] = 0.0
    grouping: str | list[str] = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    for _, indexes in output.groupby(grouping, sort=False).groups.items():
        positions = np.asarray(list(indexes), dtype=int)
        selected = positions[valid[positions]]
        if not len(selected):
            raise ValueError("A shape case has no valid future target buckets.")
        centered = logits[selected] - np.max(logits[selected])
        probabilities = np.exp(centered)
        output.loc[selected, "conditional_share"] = probabilities / probabilities.sum()
    return output


def test_segmented_softmax_preserves_exact_order_mask_and_reduction() -> None:
    scale = pd.DataFrame({"x": [0.0, 1.0], "baseline_remaining_volume": [10.0, 20.0]})
    shape = pd.DataFrame(
        {
            "case_id": ["case-b", "case-b", "case-b", "case-a", "case-a", "case-c"],
            "fold_id": ["fold-1", "fold-1", "fold-1", "fold-2", "fold-2", "fold-1"],
            "target_bucket": np.asarray([25, 24, 26, 25, 24, 25], dtype=np.int16),
            "target_valid": [True, False, True, True, False, True],
            "x": np.arange(6, dtype=float),
        }
    )
    logits = np.asarray([2.0, 1_000.0, -1.0, 0.25, 9_000.0, -0.5])
    valid = shape["target_valid"].to_numpy(dtype=bool)
    model = _model_with_logits(logits.tolist())

    totals, actual = model.predict_frames(
        scale,
        shape,
        group_columns=("case_id", "fold_id"),
    )
    expected = _legacy_shape_prediction(
        logits,
        shape,
        group_columns=("case_id", "fold_id"),
        valid=valid,
    )

    np.testing.assert_array_equal(totals, [10.0, 20.0])
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    pd.testing.assert_frame_equal(
        actual.loc[:, ["case_id", "fold_id", "target_bucket"]],
        shape.loc[:, ["case_id", "fold_id", "target_bucket"]],
        check_dtype=True,
        check_exact=True,
    )
    np.testing.assert_array_equal(actual.loc[~valid, "conditional_share"], 0.0)
    assert actual["conditional_share"].dtype == np.dtype("float64")
    assert _contiguous_group_boundaries(shape, ("case_id", "fold_id")) is not None


def test_segmented_softmax_falls_back_for_interleaved_groups() -> None:
    scale = pd.DataFrame({"x": [0.0, 1.0], "baseline_remaining_volume": [10.0, 20.0]})
    shape = pd.DataFrame(
        {
            "case_id": ["case-a", "case-b", "case-a"],
            "target_bucket": np.asarray([24, 24, 25], dtype=np.int16),
            "x": [0.0, 1.0, 2.0],
        }
    )
    logits = np.asarray([0.0, 1.0, 2.0])
    valid = np.ones(3, dtype=bool)
    model = _model_with_logits(logits.tolist())

    _, actual = model.predict_frames(scale, shape, group_columns=("case_id",))
    expected = _legacy_shape_prediction(logits, shape, group_columns=("case_id",), valid=valid)

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert _contiguous_group_boundaries(shape, ("case_id",)) is None


def test_segmented_softmax_still_rejects_a_case_without_valid_targets() -> None:
    scale = pd.DataFrame({"x": [0.0], "baseline_remaining_volume": [10.0]})
    shape = pd.DataFrame(
        {
            "case_id": ["case-a", "case-a"],
            "target_bucket": np.asarray([24, 25], dtype=np.int16),
            "target_valid": [False, False],
            "x": [0.0, 1.0],
        }
    )
    model = _model_with_logits([1.0, 2.0])
    model.scale_model = _FixedPredictor(np.zeros(1, dtype=float))

    with pytest.raises(ValueError, match="no valid future target buckets"):
        model.predict_frames(scale, shape, group_columns=("case_id",))


def _metric_base() -> LightGBMFrames:
    scale = pd.DataFrame(
        {
            "sample_id": ["second", "first"],
            "instrument_id": ["I2", "I1"],
            "session_date": ["2023-01-03", "2023-01-03"],
            "as_of": [24, 25],
            "baseline_remaining_volume": [90.0, 180.0],
        }
    )
    shape = pd.DataFrame(
        {
            "case_id": pd.Series(["second", "second", "first"], dtype=object),
            "sample_id": ["second", "second", "first"],
            "target_bucket": np.asarray([24, 25, 25], dtype=np.int16),
        }
    )
    return LightGBMFrames(
        scale,
        np.asarray([100.0, 200.0]),
        shape,
        np.asarray([0.4, 0.6, 1.0]),
    )


def _legacy_metric_frame(
    base: LightGBMFrames,
    totals: np.ndarray,
    predicted_shape: pd.DataFrame,
    *,
    fold_id: str,
    method: str,
    seed: int | None,
) -> pd.DataFrame:
    actual = base.shape.loc[:, ["case_id", "target_bucket"]].copy()
    validate_base(base)
    if len(totals) != len(base.scale) or not np.isfinite(totals).all() or np.any(totals < 0):
        raise ValueError("Forecast totals must be finite nonnegative aligned predictions.")
    actual["actual_share"] = base.shape_target
    joined = actual.merge(
        predicted_shape.loc[:, ["case_id", "target_bucket", "conditional_share"]],
        on=["case_id", "target_bucket"],
        validate="one_to_one",
        how="outer",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        raise ValueError("Forecast shape population differs from the declared base.")
    shares = joined["conditional_share"].to_numpy(dtype=float)
    if not np.isfinite(shares).all() or np.any(shares < 0):
        raise ValueError("Forecast shares must be finite and nonnegative.")
    sums = joined.groupby("case_id", sort=False)["conditional_share"].sum()
    if not np.allclose(sums.to_numpy(), 1.0, rtol=0, atol=1e-12):
        raise ValueError("Forecast conditional shares must sum to one.")
    joined = joined.sort_values(["case_id", "target_bucket"], kind="stable")
    cumulative = joined.groupby("case_id", sort=False)[
        ["actual_share", "conditional_share"]
    ].cumsum()
    distances = (cumulative["actual_share"] - cumulative["conditional_share"]).abs()
    errors = distances.groupby(joined["case_id"], sort=False).mean()
    scale = base.scale.reset_index(drop=True)
    return pd.DataFrame(
        {
            "fold_id": fold_id,
            "method": method,
            "seed": seed,
            "sample_id": scale["sample_id"],
            "instrument_id": scale["instrument_id"],
            "session_date": scale["session_date"],
            "as_of_token": scale["as_of"].astype(int),
            "actual_remaining_volume": base.scale_target,
            "causal_baseline_remaining_volume": scale["baseline_remaining_volume"],
            "predicted_remaining_volume": totals,
            "log_remaining_volume_absolute_error": np.abs(
                np.log1p(totals) - np.log1p(base.scale_target)
            ),
            "conditional_curve_wasserstein": scale["sample_id"].map(errors),
        }
    )


def test_aligned_metric_keys_skip_merge_and_sort_without_changing_results(monkeypatch):
    base = _metric_base()
    totals = np.asarray([120.0, 180.0])
    predicted = base.shape.copy()
    predicted["conditional_share"] = np.asarray([0.7, 0.3, 1.0], dtype=np.float32)
    expected = _legacy_metric_frame(
        base,
        totals,
        predicted,
        fold_id="fold-1",
        method="raw",
        seed=None,
    )
    assert _can_use_aligned_shape_keys(base.shape.loc[:, ["case_id", "target_bucket"]], predicted)

    def unexpected(*args, **kwargs):
        raise AssertionError("Aligned keys should bypass the merge and stable sort.")

    monkeypatch.setattr(pd.DataFrame, "merge", unexpected)
    monkeypatch.setattr(pd.DataFrame, "sort_values", unexpected)
    actual = forecast_metric_frame(
        base,
        totals,
        predicted,
        fold_id="fold-1",
        method="raw",
        seed=None,
    )

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    np.testing.assert_array_equal(actual["sample_id"], ["second", "first"])
    assert actual["sample_id"].dtype == base.scale["sample_id"].dtype


@pytest.mark.parametrize("variant", ["reversed", "unsorted_buckets"])
def test_metric_alignment_falls_back_and_retains_keyed_ordering(variant: str) -> None:
    base = _metric_base()
    totals = np.asarray([120.0, 180.0])
    if variant == "reversed":
        predicted = base.shape.copy()
        predicted["conditional_share"] = [0.7, 0.3, 1.0]
        predicted = predicted.iloc[[2, 1, 0]].reset_index(drop=True)
    else:
        shape = base.shape.iloc[[1, 0, 2]].reset_index(drop=True)
        shares = base.shape_target[[1, 0, 2]]
        base = LightGBMFrames(base.scale, base.scale_target, shape, shares)
        predicted = shape.copy()
        predicted["conditional_share"] = [0.3, 0.7, 1.0]
    assert not _can_use_aligned_shape_keys(
        base.shape.loc[:, ["case_id", "target_bucket"]], predicted
    )
    expected = _legacy_metric_frame(
        base,
        totals,
        predicted,
        fold_id="fold-1",
        method="raw",
        seed=None,
    )

    actual = forecast_metric_frame(
        base,
        totals,
        predicted,
        fold_id="fold-1",
        method="raw",
        seed=None,
    )

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_nonmonotone_near_gate_metric_matches_pre_sort_legacy_order() -> None:
    scale = pd.DataFrame(
        {
            "sample_id": ["edge"],
            "instrument_id": ["I1"],
            "session_date": ["2023-01-03"],
            "as_of": [0],
            "baseline_remaining_volume": [90.0],
        }
    )
    buckets = np.arange(25, -1, -1, dtype=np.int16)
    shape = pd.DataFrame(
        {
            "sample_id": ["edge"] * 26,
            "case_id": ["edge"] * 26,
            "target_bucket": buckets,
        }
    )
    actual_shares = np.zeros(26, dtype=float)
    actual_shares[buckets == 0] = 1.0
    base = LightGBMFrames(scale, np.asarray([100.0]), shape, actual_shares)
    predicted_shares = np.full(26, 1e-16, dtype=float)
    predicted_shares[buckets == 0] = 1.0
    predicted_shares[buckets == 1] = 9.95e-13
    predicted = shape.loc[:, ["case_id", "target_bucket"]].copy()
    predicted["conditional_share"] = predicted_shares
    near_gate_sum = predicted.groupby("case_id", sort=False)["conditional_share"].sum()
    assert np.allclose(near_gate_sum.to_numpy(), 1.0, rtol=0, atol=1e-12)
    assert not _can_use_aligned_shape_keys(
        base.shape.loc[:, ["case_id", "target_bucket"]], predicted
    )
    expected = _legacy_metric_frame(
        base,
        np.asarray([120.0]),
        predicted,
        fold_id="fold-1",
        method="raw",
        seed=None,
    )

    actual = forecast_metric_frame(
        base,
        np.asarray([120.0]),
        predicted,
        fold_id="fold-1",
        method="raw",
        seed=None,
    )

    pd.testing.assert_frame_equal(actual, expected, check_exact=True)


def test_metric_fast_path_does_not_bypass_duplicate_key_rejection() -> None:
    duplicate_scale = pd.DataFrame(
        {
            "sample_id": pd.Series([1, "1"], dtype=object),
            "instrument_id": ["I1", "I2"],
            "session_date": ["2023-01-03", "2023-01-03"],
            "as_of": [25, 25],
            "baseline_remaining_volume": [90.0, 180.0],
        }
    )
    duplicate_shape = pd.DataFrame(
        {
            "sample_id": pd.Series([1, "1"], dtype=object),
            "case_id": ["1", "1"],
            "target_bucket": np.asarray([25, 25], dtype=np.int16),
        }
    )
    duplicate_base = LightGBMFrames(
        duplicate_scale,
        np.asarray([100.0, 200.0]),
        duplicate_shape,
        np.asarray([1.0, 1.0]),
    )
    predicted = duplicate_shape.loc[:, ["case_id", "target_bucket"]].copy()
    predicted["conditional_share"] = [1.0, 1.0]
    assert not _can_use_aligned_shape_keys(
        duplicate_base.shape.loc[:, ["case_id", "target_bucket"]], predicted
    )
    with pytest.raises(pd.errors.MergeError, match="one-to-one"):
        forecast_metric_frame(
            duplicate_base,
            np.asarray([120.0, 180.0]),
            predicted,
            fold_id="fold-1",
            method="raw",
            seed=None,
        )


@pytest.mark.parametrize("variant", ["missing", "extra", "duplicate_prediction"])
def test_metric_fallback_retains_population_and_duplicate_rejection(variant: str) -> None:
    base = _metric_base()
    predicted = base.shape.copy()
    predicted["conditional_share"] = [0.7, 0.3, 1.0]
    if variant == "missing":
        predicted = predicted.iloc[1:].reset_index(drop=True)
        message = "population"
    elif variant == "extra":
        extra = predicted.iloc[[0]].copy()
        extra["case_id"] = "extra"
        extra["target_bucket"] = 24
        predicted = pd.concat([predicted, extra], ignore_index=True)
        message = "population"
    else:
        predicted = pd.concat([predicted, predicted.iloc[[0]]], ignore_index=True)
        message = "one-to-one"

    with pytest.raises(ValueError, match=message):
        forecast_metric_frame(
            base,
            np.asarray([120.0, 180.0]),
            predicted,
            fold_id="fold-1",
            method="raw",
            seed=None,
        )


def _benchmark_aligned_metric_keys(case_count: int = 5_000, repeats: int = 5) -> dict[str, float]:
    """Compare synthetic key validation with the merge/sort work it can bypass."""
    if not 1 <= case_count <= 20_000 or repeats < 1:
        raise ValueError("Benchmark requires 1-20,000 synthetic cases and positive repeats.")
    case_ids = np.repeat(np.arange(case_count, dtype=np.int64), 8)
    buckets = np.tile(np.arange(8, dtype=np.int16), case_count)
    actual = pd.DataFrame({"case_id": case_ids, "target_bucket": buckets})
    predicted = actual.assign(conditional_share=np.full(len(actual), 1 / 8, dtype=np.float64))

    started = perf_counter()
    for _ in range(repeats):
        assert _can_use_aligned_shape_keys(actual, predicted)
    aligned_seconds = (perf_counter() - started) / repeats

    started = perf_counter()
    for _ in range(repeats):
        actual.merge(
            predicted,
            on=["case_id", "target_bucket"],
            validate="one_to_one",
            how="outer",
            indicator=True,
        ).sort_values(["case_id", "target_bucket"], kind="stable")
    merge_sort_seconds = (perf_counter() - started) / repeats
    return {"aligned_key_check_seconds": aligned_seconds, "merge_sort_seconds": merge_sort_seconds}


def _legacy_predict_frames(
    model: LightGBMVolumeModel,
    scale_features: pd.DataFrame,
    shape_features: pd.DataFrame,
    *,
    group_columns: tuple[str, ...],
) -> tuple[np.ndarray, pd.DataFrame]:
    """Run the former full predict_frames path as a benchmark reference."""
    scale = model._prepare_scale(scale_features)
    metadata = shape_features.copy().reset_index(drop=True)
    shape = model._prepare_shape(metadata)
    valid = (
        metadata["target_valid"].astype(bool).to_numpy()
        if "target_valid" in metadata
        else np.ones(len(shape), dtype=bool)
    )
    logits = np.asarray(model.shape_model.predict(shape), dtype=float)
    output = metadata.loc[:, [*group_columns, "target_bucket"]].copy()
    output["conditional_share"] = 0.0
    grouping: str | list[str] = group_columns[0] if len(group_columns) == 1 else list(group_columns)
    for _, indexes in output.groupby(grouping, sort=False).groups.items():
        positions = np.asarray(list(indexes), dtype=int)
        selected = positions[valid[positions]]
        if not len(selected):
            raise ValueError("A shape case has no valid future target buckets.")
        centered = logits[selected] - np.max(logits[selected])
        probabilities = np.exp(centered)
        output.loc[selected, "conditional_share"] = probabilities / probabilities.sum()
    baseline = scale_features["baseline_remaining_volume"].to_numpy(dtype=float)
    residual = np.asarray(model.scale_model.predict(scale), dtype=float)
    totals = np.maximum((1.0 + baseline) * np.exp(residual) - 1.0, 0.0)
    if not np.isfinite(totals).all() or not np.isfinite(logits).all():
        raise ValueError("LightGBM produced a non-finite forecast.")
    return totals, output


def _benchmark_predict_frames(case_count: int = 10_000, repeats: int = 2) -> dict[str, float]:
    """Compare full softmax prediction with the old exact NumPy reduction loop."""
    if not 1 <= case_count <= 20_000 or repeats < 1:
        raise ValueError("Benchmark requires 1-20,000 synthetic cases and positive repeats.")
    horizon = 8
    scale = pd.DataFrame(
        {
            "x": np.zeros(case_count, dtype=float),
            "baseline_remaining_volume": np.full(case_count, 90.0, dtype=float),
        }
    )
    row_count = case_count * horizon
    shape = pd.DataFrame(
        {
            "case_id": np.repeat(np.arange(case_count, dtype=np.int64), horizon),
            "target_bucket": np.tile(np.arange(horizon, dtype=np.int16), case_count),
            "x": np.zeros(row_count, dtype=float),
        }
    )
    logits = np.tile(np.linspace(-2.0, 2.0, horizon), case_count)
    model = _model_with_logits(logits)
    model.scale_model = _FixedPredictor(np.zeros(case_count, dtype=float))
    group_columns = ("case_id",)
    expected = _legacy_predict_frames(model, scale, shape, group_columns=group_columns)
    actual = model.predict_frames(scale, shape, group_columns=group_columns)
    np.testing.assert_array_equal(actual[0], expected[0])
    pd.testing.assert_frame_equal(actual[1], expected[1], check_exact=True)

    started = perf_counter()
    for _ in range(repeats):
        _legacy_predict_frames(model, scale, shape, group_columns=group_columns)
    legacy_seconds = (perf_counter() - started) / repeats
    started = perf_counter()
    for _ in range(repeats):
        model.predict_frames(scale, shape, group_columns=group_columns)
    fast_seconds = (perf_counter() - started) / repeats
    return {
        "cases": float(case_count),
        "shape_rows": float(row_count),
        "legacy_predict_frames_seconds": legacy_seconds,
        "segmented_predict_frames_seconds": fast_seconds,
    }


def _benchmark_forecast_metric_frames(
    case_count: int = 10_000, repeats: int = 2
) -> dict[str, float]:
    """Compare full synthetic forecast metrics against the former join/sort path."""
    if not 1 <= case_count <= 20_000 or repeats < 1:
        raise ValueError("Benchmark requires 1-20,000 synthetic cases and positive repeats.")
    horizon = 8
    case_ids = np.arange(case_count, dtype=np.int64)
    row_count = case_count * horizon
    scale = pd.DataFrame(
        {
            "sample_id": case_ids,
            "instrument_id": np.full(case_count, "synthetic", dtype=object),
            "session_date": np.full(case_count, "2023-01-03", dtype=object),
            "as_of": np.full(case_count, 18, dtype=np.int16),
            "baseline_remaining_volume": np.full(case_count, 90.0, dtype=float),
        }
    )
    shape = pd.DataFrame(
        {
            "sample_id": np.repeat(case_ids, horizon),
            "case_id": np.repeat(case_ids, horizon),
            "target_bucket": np.tile(np.arange(18, 26, dtype=np.int16), case_count),
        }
    )
    shares = np.full(row_count, 1.0 / horizon, dtype=float)
    base = LightGBMFrames(
        scale,
        np.full(case_count, 100.0, dtype=float),
        shape,
        shares,
    )
    totals = np.full(case_count, 110.0, dtype=float)
    predicted = shape.loc[:, ["case_id", "target_bucket"]].copy()
    predicted["conditional_share"] = shares.copy()
    options = dict(fold_id="synthetic", method="fixture", seed=None)
    expected = _legacy_metric_frame(base, totals, predicted, **options)
    actual = forecast_metric_frame(base, totals, predicted, **options)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)

    started = perf_counter()
    for _ in range(repeats):
        _legacy_metric_frame(base, totals, predicted, **options)
    legacy_seconds = (perf_counter() - started) / repeats
    started = perf_counter()
    for _ in range(repeats):
        forecast_metric_frame(base, totals, predicted, **options)
    fast_seconds = (perf_counter() - started) / repeats
    return {
        "cases": float(case_count),
        "shape_rows": float(row_count),
        "legacy_metric_frame_seconds": legacy_seconds,
        "aligned_metric_frame_seconds": fast_seconds,
    }
