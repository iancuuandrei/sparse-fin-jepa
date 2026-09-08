"""Fair raw, placebo, and hybrid LightGBM feature matrices."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from execsim.ml.models.random_projection import projection_hash, random_projection_matrix

METADATA_COLUMNS = (
    "as_of_bucket",
    "target_bucket",
    "horizon_offset",
    "minutes_remaining",
    "weekday",
    "month",
    "is_month_end",
    "is_quarter_end",
    "symbol",
    "liquidity_group",
)


def build_raw_feature_frame(
    context: np.ndarray, mask: np.ndarray, metadata: pd.DataFrame
) -> pd.DataFrame:
    """Flatten the same causal context and append the locked tabular metadata."""
    values = np.asarray(context, dtype=np.float32)
    masks = np.asarray(mask, dtype=bool)
    missing = set(METADATA_COLUMNS).difference(metadata.columns)
    if values.ndim != 3 or values.shape[1:] != (8, 18) or masks.shape != values.shape[:2]:
        raise ValueError("LightGBM raw context must have shape [row, 8, 18] and matching masks.")
    if len(metadata) != len(values) or missing:
        raise ValueError(f"LightGBM metadata is misaligned or missing: {sorted(missing)}")
    flattened = values.reshape(len(values), -1)
    columns = [
        f"context_t{token:02d}_f{feature:02d}" for token in range(8) for feature in range(18)
    ]
    frame = pd.DataFrame(flattened, columns=columns, index=metadata.index)
    for token in range(8):
        frame[f"context_mask_t{token:02d}"] = masks[:, token].astype(np.int8)
    for column in METADATA_COLUMNS:
        frame[column] = metadata[column]
    frame["symbol"] = frame["symbol"].astype("category")
    return frame


def append_embedding(raw: pd.DataFrame, embedding: np.ndarray) -> pd.DataFrame:
    """Append 640 latent values plus four explicit horizon-availability flags."""
    values = np.asarray(embedding, dtype=np.float32)
    if values.shape != (len(raw), 644) or not np.isfinite(values).all():
        raise ValueError("Hybrid embedding matrix must have shape [row, 644] and be finite.")
    embedding_frame = pd.DataFrame(
        values,
        columns=[f"embedding_{index:03d}" for index in range(644)],
        index=raw.index,
    )
    return pd.concat((raw, embedding_frame), axis=1)


def build_random_control(
    context: np.ndarray, mask: np.ndarray, *, fold_seed: int
) -> tuple[np.ndarray, str]:
    """Project only target-free context and mask into the 640-value placebo space."""
    values = np.asarray(context, dtype=float)
    masks = np.asarray(mask, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (8, 18) or masks.shape != values.shape[:2]:
        raise ValueError("Random-control inputs must match the raw context contract.")
    raw = np.concatenate((values.reshape(len(values), -1), masks), axis=1)
    matrix = random_projection_matrix(raw.shape[1], seed=fold_seed)
    return raw @ matrix, projection_hash(matrix)


def build_untrained_neural_control(
    context: np.ndarray,
    mask: np.ndarray,
    horizon_mask: np.ndarray,
    *,
    fold_seed: int,
    batch_size: int = 8_192,
) -> tuple[np.ndarray, str]:
    """Export a frozen nonlinear target-free placebo in bounded inference batches."""
    import hashlib

    import torch

    from execsim.ml.representations.embeddings import export_frozen_embedding_batch
    from execsim.ml.representations.jepa import PredictiveRepresentationModel
    from execsim.ml.representations.schemas import RepresentationConfig

    values = np.array(context, dtype=np.float32, copy=True, order="C")
    masks = np.array(mask, dtype=bool, copy=True, order="C")
    horizons = np.array(horizon_mask, dtype=bool, copy=True, order="C")
    if values.ndim != 3 or values.shape[1:] != (8, 18) or masks.shape != values.shape[:2]:
        raise ValueError("Untrained neural control inputs must match the raw context contract.")
    if horizons.shape != (len(values), 4):
        raise ValueError("Untrained neural control requires four availability flags per row.")
    if batch_size < 1:
        raise ValueError("Untrained neural control batch_size must be positive.")
    with torch.random.fork_rng():
        torch.manual_seed(fold_seed)
        model = PredictiveRepresentationModel(RepresentationConfig("dense", seed=fold_seed)).eval()
        digest = hashlib.sha256()
        for name, tensor in sorted(model.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
        identity = digest.hexdigest()
        exported = np.empty((len(values), 644), dtype=np.float32)
        for start in range(0, len(values), batch_size):
            stop = min(start + batch_size, len(values))
            exported[start:stop] = export_frozen_embedding_batch(
                model,
                torch.from_numpy(values[start:stop]),
                torch.from_numpy(masks[start:stop]),
                torch.from_numpy(horizons[start:stop]),
            )
    return exported, identity


def append_untrained_neural_control_frames(
    values: tuple[pd.DataFrame, Any, pd.DataFrame, Any], *, fold_seed: int
) -> tuple[pd.DataFrame, Any, pd.DataFrame, Any]:
    """Append one target-free neural placebo consistently to scale and long-shape rows."""
    scale, scale_target, shape, shape_target = values
    feature_columns = [
        f"context_t{token:02d}_f{feature:02d}" for token in range(8) for feature in range(18)
    ]
    mask_columns = [f"context_mask_t{token:02d}" for token in range(8)]
    context = scale[feature_columns].to_numpy(dtype=np.float32).reshape(-1, 8, 18)
    mask = scale[mask_columns].to_numpy(dtype=bool)
    horizons = np.asarray(
        [[int(as_of) + horizon - 1 < 26 for horizon in (1, 2, 4, 8)] for as_of in scale["as_of"]],
        dtype=bool,
    )
    embedding, _identity = build_untrained_neural_control(
        context, mask, horizons, fold_seed=fold_seed
    )
    scale_output = append_embedding(scale, embedding)
    scale_ids = pd.Index(scale["sample_id"].astype(str))
    if scale_ids.has_duplicates:
        raise ValueError("Untrained neural control scale rows duplicate sample identity.")
    shape_positions = scale_ids.get_indexer(shape["sample_id"].astype(str))
    if np.any(shape_positions < 0):
        raise ValueError("Untrained neural control shape rows lack matching scale samples.")
    shape_embedding = embedding[shape_positions]
    shape_output = append_embedding(shape, shape_embedding)
    return scale_output, scale_target, shape_output, shape_target
