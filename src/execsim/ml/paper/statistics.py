"""Date-clustered paired inference for locked paper endpoints."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True, slots=True)
class PairedBlockResult:
    """Summarize one paired candidate-minus-baseline comparison."""

    paired_dates: int
    mean_difference: float
    median_difference: float
    confidence_interval: tuple[float, float]
    date_win_rate: float
    standardized_effect: float
    raw_p_value: float


@dataclass(frozen=True, slots=True)
class CompleteCaseResult:
    """Describe an exact paired intersection before date aggregation."""

    paired_rows: pd.DataFrame
    baseline_rows: int
    candidate_rows: int
    matched_rows: int
    dropped_baseline_rows: int
    dropped_candidate_rows: int


def construct_seed_matched_differences(
    rows: pd.DataFrame,
    *,
    baseline: str,
    candidate: str,
    value_column: str,
    identity_columns: tuple[str, ...],
    method_column: str = "method",
    seed_column: str = "seed",
) -> CompleteCaseResult:
    """Pair each candidate seed with the same-seed or shared baseline cases."""
    required = {method_column, seed_column, value_column, *identity_columns}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Seed-matched ledger missing columns: {sorted(missing)}")
    candidate_rows = rows.loc[rows[method_column] == candidate].copy()
    seeds = tuple(sorted(int(value) for value in candidate_rows[seed_column].dropna().unique()))
    if not seeds:
        raise ValueError("Seed-matched comparison has no candidate seeds.")
    expanded: list[pd.DataFrame] = []
    for seed in seeds:
        seeded_candidate = candidate_rows.loc[candidate_rows[seed_column] == seed].copy()
        available_baseline = rows.loc[rows[method_column] == baseline].copy()
        same_seed = available_baseline.loc[available_baseline[seed_column] == seed]
        if not same_seed.empty:
            seeded_baseline = same_seed.copy()
        else:
            seeded_baseline = available_baseline.loc[available_baseline[seed_column].isna()].copy()
        if seeded_baseline.empty:
            raise ValueError(f"Seed-matched comparison has no baseline for seed {seed}.")
        seeded_candidate["pair_seed"] = seed
        seeded_baseline["pair_seed"] = seed
        expanded.extend((seeded_baseline, seeded_candidate))
    return construct_complete_case_differences(
        pd.concat(expanded, ignore_index=True),
        baseline=baseline,
        candidate=candidate,
        value_column=value_column,
        identity_columns=(*identity_columns, "pair_seed"),
        method_column=method_column,
    )


def construct_complete_case_differences(
    rows: pd.DataFrame,
    *,
    baseline: str,
    candidate: str,
    value_column: str,
    identity_columns: tuple[str, ...],
    method_column: str = "method",
) -> CompleteCaseResult:
    """Intersect exact experiment identities before calculating paired differences."""
    required = {method_column, value_column, *identity_columns}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Complete-case ledger missing columns: {sorted(missing)}")
    selected = rows.loc[rows[method_column].isin((baseline, candidate))].copy()
    if selected.duplicated([method_column, *identity_columns]).any():
        raise ValueError("Complete-case ledger contains duplicated method/case rows.")
    base = selected.loc[selected[method_column] == baseline, [*identity_columns, value_column]]
    other = selected.loc[selected[method_column] == candidate, [*identity_columns, value_column]]
    paired = base.merge(
        other,
        on=list(identity_columns),
        how="inner",
        suffixes=("_baseline", "_candidate"),
        validate="one_to_one",
    )
    paired["difference"] = paired[f"{value_column}_candidate"] - paired[f"{value_column}_baseline"]
    return CompleteCaseResult(
        paired_rows=paired,
        baseline_rows=len(base),
        candidate_rows=len(other),
        matched_rows=len(paired),
        dropped_baseline_rows=len(base) - len(paired),
        dropped_candidate_rows=len(other) - len(paired),
    )


def paper_forecast_metrics(
    actual_remaining: np.ndarray,
    predicted_remaining: np.ndarray,
    actual_shape: np.ndarray,
    predicted_shape: np.ndarray,
) -> dict[str, float]:
    """Compute primary log remaining-volume MAE and conditional-curve W1."""
    actual_total = np.asarray(actual_remaining, dtype=float)
    predicted_total = np.asarray(predicted_remaining, dtype=float)
    actual_curve = np.asarray(actual_shape, dtype=float)
    predicted_curve = np.asarray(predicted_shape, dtype=float)
    if actual_total.shape != predicted_total.shape or actual_curve.shape != predicted_curve.shape:
        raise ValueError("Paper forecast metric arrays must align.")
    if actual_curve.ndim != 2 or len(actual_curve) != len(actual_total):
        raise ValueError("Paper shape metrics require one curve per volume target.")
    if not all(
        np.isfinite(values).all()
        for values in (actual_total, predicted_total, actual_curve, predicted_curve)
    ):
        raise ValueError("Paper forecast metrics require finite values.")
    if (actual_total < 0).any() or (predicted_total < 0).any():
        raise ValueError("Remaining-volume metrics require non-negative values.")
    if (actual_curve < 0).any() or (predicted_curve < 0).any():
        raise ValueError("Conditional curves require non-negative shares.")
    if not np.allclose(actual_curve.sum(axis=1), 1.0) or not np.allclose(
        predicted_curve.sum(axis=1), 1.0
    ):
        raise ValueError("Conditional curves must be row-normalized.")
    return {
        "log_remaining_volume_mae": float(
            np.mean(np.abs(np.log1p(predicted_total) - np.log1p(actual_total)))
        ),
        "conditional_curve_wasserstein": float(
            np.mean(np.abs(np.cumsum(predicted_curve, axis=1) - np.cumsum(actual_curve, axis=1)))
        ),
    }


def average_cases_by_date(
    rows: pd.DataFrame,
    *,
    value_column: str,
    model_columns: tuple[str, ...] = ("model", "fold_id", "seed"),
) -> pd.DataFrame:
    """Give each date equal weight after averaging its symbol-as-of cases."""
    required = {"date", value_column, *model_columns}
    missing = required.difference(rows.columns)
    if missing:
        raise ValueError(f"Date aggregation missing columns: {sorted(missing)}")
    return (
        rows.groupby([*model_columns, "date"], sort=True, as_index=False)[value_column]
        .mean()
        .sort_values([*model_columns, "date"], kind="stable")
        .reset_index(drop=True)
    )


def all_seed_claim_supported(seed_effects: pd.DataFrame) -> bool:
    """Require same-direction seed effects and an averaged interval excluding zero."""
    required = {"seed", "mean_difference", "average_ci_lower", "average_ci_upper"}
    missing = required.difference(seed_effects.columns)
    if missing or seed_effects.empty:
        raise ValueError(f"Seed claim evidence missing columns or rows: {sorted(missing)}")
    effects = seed_effects["mean_difference"].to_numpy(dtype=float)
    same_direction = bool(np.all(effects < 0) or np.all(effects > 0))
    lower = float(seed_effects["average_ci_lower"].iloc[0])
    upper = float(seed_effects["average_ci_upper"].iloc[0])
    return same_direction and (upper < 0 or lower > 0)


def moving_block_bootstrap(
    date_values: pd.DataFrame,
    *,
    value_column: str = "difference",
    fold_column: str = "fold_id",
    block_length: int = 5,
    repetitions: int = 10_000,
    confidence: float = 0.95,
    seed: int = 13,
) -> PairedBlockResult:
    """Bootstrap contiguous dates without permitting blocks to cross folds."""
    required = {"date", value_column, fold_column}
    missing = required.difference(date_values.columns)
    if missing:
        raise ValueError(f"Block bootstrap missing columns: {sorted(missing)}")
    if block_length <= 0 or repetitions <= 0 or not 0 < confidence < 1:
        raise ValueError("Bootstrap parameters are invalid.")
    ordered = date_values.sort_values([fold_column, "date"], kind="stable")
    if ordered.duplicated([fold_column, "date"]).any():
        raise ValueError("Bootstrap input must contain one paired value per fold-date.")
    values = ordered[value_column].to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap values must be non-empty and finite.")
    fold_blocks: list[tuple[int, list[np.ndarray]]] = []
    for _, group in ordered.groupby(fold_column, sort=True):
        group_values = group[value_column].to_numpy(dtype=float)
        width = min(block_length, len(group_values))
        blocks = list(
            group_values[start : start + width] for start in range(len(group_values) - width + 1)
        )
        fold_blocks.append((len(group_values), blocks))
    rng = np.random.default_rng(seed)
    observed_mean = float(np.mean(values))
    means = np.empty(repetitions)
    null_means = np.empty(repetitions)
    for repetition in range(repetitions):
        sampled: list[float] = []
        for fold_size, blocks in fold_blocks:
            fold_sample: list[float] = []
            while len(fold_sample) < fold_size:
                fold_sample.extend(blocks[int(rng.integers(len(blocks)))].tolist())
            sampled.extend(fold_sample[:fold_size])
        means[repetition] = np.mean(sampled)
        null_means[repetition] = np.mean(np.asarray(sampled) - observed_mean)
    alpha = (1 - confidence) / 2
    standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return PairedBlockResult(
        paired_dates=len(values),
        mean_difference=float(np.mean(values)),
        median_difference=float(np.median(values)),
        confidence_interval=(
            float(np.quantile(means, alpha)),
            float(np.quantile(means, 1 - alpha)),
        ),
        date_win_rate=float(np.mean(values < 0)),
        standardized_effect=float(observed_mean / standard_deviation)
        if standard_deviation
        else 0.0,
        raw_p_value=float(
            (1 + np.count_nonzero(np.abs(null_means) >= abs(observed_mean))) / (repetitions + 1)
        ),
    )


def build_confirmatory_inference(
    representation_dates: pd.DataFrame,
    forecast_rows: pd.DataFrame,
    *,
    definitions: tuple[dict[str, str], ...],
    block_length: int,
    sensitivity_block_lengths: tuple[int, ...],
    repetitions: int,
    confidence: float,
    seed: int = 13,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate the five frozen contrasts from exact seed-matched date differences."""
    if len(definitions) != 5:
        raise ValueError("Confirmatory inference requires exactly five frozen definitions.")
    overall_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    sensitivity_rows: list[dict[str, object]] = []
    for contrast_index, definition in enumerate(definitions, start=1):
        stage = definition["stage"]
        candidate = definition["candidate"]
        baseline = definition["baseline"]
        endpoint = definition["endpoint"]
        if stage == "representation":
            if endpoint != "affine_normalized_latent_error":
                raise ValueError(f"Unsupported frozen representation endpoint: {endpoint}")
            ledger = representation_dates.loc[
                representation_dates["probe_capacity"] == "affine_ridge"
            ].rename(columns={"geometry": "method"})
            value_column = "normalized_latent_error"
            identity = (
                "fold_id",
                "date",
                "horizon",
                "probe_capacity",
                "sample_identity_sha256",
            )
        elif stage == "forecasting":
            endpoint_columns = {
                "log_remaining_volume_mae": "log_remaining_volume_absolute_error",
                "conditional_curve_error": "conditional_curve_wasserstein",
            }
            if endpoint not in endpoint_columns:
                raise ValueError(f"Unsupported frozen forecasting endpoint: {endpoint}")
            ledger = forecast_rows.rename(columns={"session_date": "date"})
            value_column = endpoint_columns[endpoint]
            identity = (
                "fold_id",
                "date",
                "instrument_id",
                "sample_id",
                "as_of_token",
            )
        else:
            raise ValueError(f"Unsupported frozen confirmatory stage: {stage}")
        paired = construct_seed_matched_differences(
            ledger,
            baseline=baseline,
            candidate=candidate,
            value_column=value_column,
            identity_columns=identity,
        )
        by_seed_date = (
            paired.paired_rows.groupby(["fold_id", "date", "pair_seed"], sort=True, as_index=False)[
                "difference"
            ]
            .mean()
            .sort_values(["fold_id", "date", "pair_seed"], kind="stable")
        )
        by_date = by_seed_date.groupby(["fold_id", "date"], sort=True, as_index=False)[
            "difference"
        ].mean()
        primary = moving_block_bootstrap(
            by_date,
            block_length=block_length,
            repetitions=repetitions,
            confidence=confidence,
            seed=seed + contrast_index,
        )
        overall_rows.append(
            {
                "contrast_id": contrast_index,
                "stage": stage,
                "candidate": candidate,
                "baseline": baseline,
                "endpoint": endpoint,
                **_block_result_record(primary),
                "matched_cases": paired.matched_rows,
                "dropped_baseline": paired.dropped_baseline_rows,
                "dropped_candidate": paired.dropped_candidate_rows,
            }
        )
        for pair_seed, group in by_seed_date.groupby("pair_seed", sort=True):
            seed_result = moving_block_bootstrap(
                group.loc[:, ["fold_id", "date", "difference"]],
                block_length=block_length,
                repetitions=repetitions,
                confidence=confidence,
                seed=seed + contrast_index + int(pair_seed),
            )
            seed_rows.append(
                {
                    "contrast_id": contrast_index,
                    "stage": stage,
                    "candidate": candidate,
                    "baseline": baseline,
                    "endpoint": endpoint,
                    "seed": int(pair_seed),
                    **_block_result_record(seed_result),
                }
            )
        for sensitivity_length in (block_length, *sensitivity_block_lengths):
            result = moving_block_bootstrap(
                by_date,
                block_length=sensitivity_length,
                repetitions=repetitions,
                confidence=confidence,
                seed=seed + contrast_index,
            )
            sensitivity_rows.append(
                {
                    "contrast_id": contrast_index,
                    "stage": stage,
                    "candidate": candidate,
                    "baseline": baseline,
                    "endpoint": endpoint,
                    "block_length_dates": sensitivity_length,
                    **_block_result_record(result),
                }
            )
    overall = pd.DataFrame(overall_rows).sort_values("contrast_id", kind="stable")
    overall["holm_adjusted_p_value"] = holm_adjust_pvalues(
        overall["raw_p_value"].to_numpy(dtype=float)
    )
    return (
        overall.reset_index(drop=True),
        pd.DataFrame(seed_rows).sort_values(["contrast_id", "seed"], kind="stable"),
        pd.DataFrame(sensitivity_rows).sort_values(
            ["contrast_id", "block_length_dates"], kind="stable"
        ),
    )


def _block_result_record(result: PairedBlockResult) -> dict[str, float | int]:
    return {
        "paired_dates": result.paired_dates,
        "mean_difference": result.mean_difference,
        "median_difference": result.median_difference,
        "ci_lower": result.confidence_interval[0],
        "ci_upper": result.confidence_interval[1],
        "date_win_rate": result.date_win_rate,
        "standardized_effect": result.standardized_effect,
        "raw_p_value": result.raw_p_value,
    }


def holm_adjust_pvalues(pvalues: np.ndarray) -> np.ndarray:
    """Apply the predeclared Holm step-down family-wise adjustment."""
    values = np.asarray(pvalues, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Holm adjustment requires a finite one-dimensional family.")
    if ((values < 0) | (values > 1)).any():
        raise ValueError("P-values must lie in [0, 1].")
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.maximum.accumulate(
        np.asarray([(len(values) - rank) * values[index] for rank, index in enumerate(order)])
    )
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted
