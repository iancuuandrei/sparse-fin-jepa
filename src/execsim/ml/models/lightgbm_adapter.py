"""Optional native LightGBM scale and long-form conditional-shape forecaster."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

NON_FEATURE_COLUMNS = {
    "sample_id",
    "case_id",
    "fold_id",
    "instrument_id",
    "session_date",
    "as_of",
    "partition",
    "target_valid",
    "sample_weight",
    "training_cutoff",
    "market_information_as_of",
    "feature_history_end",
    "shape_origin_inclusion_probability",
    "shape_case_weight",
    "shape_row_weight",
}


def _lightgbm() -> Any:
    try:
        import lightgbm
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("Install the 'lightgbm' or 'paper' extra for this model.") from exc
    return lightgbm


@dataclass(frozen=True, slots=True)
class LightGBMConfig:
    """Bound one scientific candidate from the locked validation grid."""

    num_leaves: int = 15
    min_child_samples: int = 50
    reg_lambda: float = 1.0
    learning_rate: float = 0.03
    n_estimators: int = 2_000
    early_stopping_rounds: int = 100
    feature_fraction: float = 0.8
    bagging_fraction: float = 0.8
    bagging_freq: int = 1
    seed: int = 13


@dataclass(frozen=True, slots=True)
class LightGBMExecutionOptions:
    """Bind operational LightGBM resources without changing the scientific grid."""

    device_type: Literal["cpu", "gpu"] = "cpu"
    gpu_platform_id: int | None = None
    gpu_device_id: int | None = None
    gpu_use_dp: bool = False
    num_threads: int = 1

    def __post_init__(self) -> None:
        if self.device_type not in {"cpu", "gpu"}:
            raise ValueError("LightGBM device_type must be 'cpu' or OpenCL 'gpu'.")
        if self.num_threads < 1:
            raise ValueError("LightGBM num_threads must be positive.")
        for name, value in (
            ("gpu_platform_id", self.gpu_platform_id),
            ("gpu_device_id", self.gpu_device_id),
        ):
            if value is not None and value < 0:
                raise ValueError(f"LightGBM {name} must be non-negative when specified.")
        if self.device_type == "cpu":
            if (
                self.gpu_platform_id is not None
                or self.gpu_device_id is not None
                or self.gpu_use_dp
            ):
                raise ValueError("CPU LightGBM execution cannot specify GPU-only options.")
        elif self.gpu_platform_id is None or self.gpu_device_id is None:
            raise ValueError(
                "OpenCL GPU execution requires explicit gpu_platform_id and gpu_device_id."
            )

    def identity(self) -> dict[str, object]:
        """Return the canonical execution identity stored in fitted artifacts."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class LightGBMGridResult:
    """Persist one validation-only scale and shape grid candidate."""

    config: LightGBMConfig
    scale_mae: float
    shape_error: float
    scale_iterations: int
    shape_iterations: int


class LightGBMVolumeModel:
    """Fit one scale model and one long-form, shrinking-horizon shape model."""

    def __init__(
        self,
        config: LightGBMConfig | None = None,
        *,
        execution: LightGBMExecutionOptions | None = None,
    ) -> None:
        self.config = config or LightGBMConfig()
        self.execution = execution or LightGBMExecutionOptions()
        self.scale_config = self.config
        self.shape_config = self.config
        self.scale_model: Any | None = None
        self.shape_model: Any | None = None
        self.feature_columns: tuple[str, ...] = ()
        self.shape_feature_columns: tuple[str, ...] = ()
        self.categorical_features: tuple[str, ...] = ()
        self.category_vocabulary: dict[str, tuple[str, ...]] = {}
        self.output_buckets: int | None = None

    def fit_frames(
        self,
        scale_features: pd.DataFrame,
        remaining_volume: np.ndarray,
        shape_features: pd.DataFrame,
        conditional_share: np.ndarray,
        *,
        categorical_features: tuple[str, ...] = ("symbol",),
        validation: tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray] | None = None,
    ) -> LightGBMVolumeModel:
        """Fit native pandas frames and freeze train-fold categorical vocabularies."""
        lightgbm = _lightgbm()
        scale, total = _validate_scale_frame(scale_features, remaining_volume)
        shape, share = _validate_shape_frame(shape_features, conditional_share)
        scale_target = _scale_residual(scale, total)
        shape_weight = (
            shape["sample_weight"].to_numpy(dtype=float) if "sample_weight" in shape else None
        )
        self.feature_columns = tuple(
            name for name in scale.columns if name not in NON_FEATURE_COLUMNS
        )
        self.shape_feature_columns = tuple(
            name for name in shape.columns if name not in NON_FEATURE_COLUMNS
        )
        self.categorical_features = tuple(categorical_features)
        self.category_vocabulary = _fit_categories((scale, shape), self.categorical_features)
        scale = _apply_categories(
            scale.loc[:, list(self.feature_columns)], self.category_vocabulary, training=True
        )
        shape = _apply_categories(
            shape.loc[:, list(self.shape_feature_columns)], self.category_vocabulary, training=True
        )
        callbacks: list[Any] = []
        scale_eval: tuple[pd.DataFrame, np.ndarray] | None = None
        shape_eval: tuple[pd.DataFrame, np.ndarray] | None = None
        if validation is not None:
            valid_scale, valid_total = _validate_scale_frame(validation[0], validation[1])
            valid_shape, valid_share = _validate_shape_frame(validation[2], validation[3])
            valid_scale = self._prepare_scale(valid_scale)
            valid_shape = self._prepare_shape(valid_shape)
            callbacks = [lightgbm.early_stopping(self.config.early_stopping_rounds, verbose=False)]
            scale_eval = (valid_scale, _scale_residual(validation[0], valid_total))
            shape_eval = (valid_shape, np.log(valid_share + 1e-6))
        parameters = _parameters(self.config, self.execution)
        self.scale_model = lightgbm.LGBMRegressor(**parameters)
        self.scale_model.fit(
            scale,
            scale_target,
            categorical_feature=list(self.categorical_features),
            eval_X=None if scale_eval is None else scale_eval[0],
            eval_y=None if scale_eval is None else scale_eval[1],
            callbacks=callbacks,
        )
        self.shape_model = lightgbm.LGBMRegressor(**parameters)
        self.shape_model.fit(
            shape,
            np.log(share + 1e-6),
            sample_weight=shape_weight,
            categorical_feature=list(self.categorical_features),
            eval_X=None if shape_eval is None else shape_eval[0],
            eval_y=None if shape_eval is None else shape_eval[1],
            callbacks=callbacks,
        )
        return self

    def fit(
        self,
        features: np.ndarray,
        remaining_volume: np.ndarray,
        conditional_shape: np.ndarray,
        *,
        validation: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    ) -> LightGBMVolumeModel:
        """Retain a tiny-array helper while using the long-form production contract."""
        x, total, shape = _validated_arrays(features, remaining_volume, conditional_shape)
        columns = [f"feature_{index}" for index in range(x.shape[1])]
        scale = pd.DataFrame(x, columns=columns)
        scale["baseline_remaining_volume"] = 0.0
        shape_frame, shares = _wide_shape_to_long(scale, shape)
        valid_frames = None
        if validation is not None:
            valid_x, valid_total, valid_shape = _validated_arrays(*validation)
            valid_scale = pd.DataFrame(valid_x, columns=columns)
            valid_scale["baseline_remaining_volume"] = 0.0
            valid_long, valid_shares = _wide_shape_to_long(valid_scale, valid_shape)
            valid_frames = (valid_scale, valid_total, valid_long, valid_shares)
        self.output_buckets = shape.shape[1]
        return self.fit_frames(
            scale,
            total,
            shape_frame,
            shares,
            categorical_features=(),
            validation=valid_frames,
        )

    def predict_frames(
        self,
        scale_features: pd.DataFrame,
        shape_features: pd.DataFrame,
        *,
        group_columns: tuple[str, ...],
        valid_column: str = "target_valid",
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Predict scale and stable-softmax shape only across each case's valid rows."""
        if self.scale_model is None or self.shape_model is None:
            raise RuntimeError("LightGBM volume model is not fitted.")
        scale = self._prepare_scale(scale_features)
        metadata = shape_features.copy().reset_index(drop=True)
        shape = self._prepare_shape(metadata)
        valid = (
            metadata[valid_column].astype(bool).to_numpy()
            if valid_column in metadata
            else np.ones(len(shape), dtype=bool)
        )
        logits = np.asarray(self.shape_model.predict(shape), dtype=float)
        output = metadata.loc[:, [*group_columns, "target_bucket"]].copy()
        output["conditional_share"] = 0.0
        grouping: str | list[str] = (
            group_columns[0] if len(group_columns) == 1 else list(group_columns)
        )
        for _, indexes in output.groupby(grouping, sort=False).groups.items():
            positions = np.asarray(list(indexes), dtype=int)
            selected = positions[valid[positions]]
            if not len(selected):
                raise ValueError("A shape case has no valid future target buckets.")
            centered = logits[selected] - np.max(logits[selected])
            probabilities = np.exp(centered)
            output.loc[selected, "conditional_share"] = probabilities / probabilities.sum()
        baseline = _baseline_remaining(scale_features)
        residual = np.asarray(self.scale_model.predict(scale), dtype=float)
        totals = np.maximum((1.0 + baseline) * np.exp(residual) - 1.0, 0.0)
        if not np.isfinite(totals).all() or not np.isfinite(logits).all():
            raise ValueError("LightGBM produced a non-finite forecast.")
        return totals, output

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Predict the fixed-width tiny-array compatibility representation."""
        if self.output_buckets is None:
            raise RuntimeError("Fixed-width prediction is unavailable for frame-only models.")
        values = np.asarray(features, dtype=float)
        columns = list(self.feature_columns)
        if (
            columns
            and columns[-1] == "baseline_remaining_volume"
            and values.shape[1] == len(columns) - 1
        ):
            scale = pd.DataFrame(values, columns=columns[:-1])
            scale["baseline_remaining_volume"] = 0.0
        else:
            scale = pd.DataFrame(values, columns=columns)
        repeated = scale.loc[scale.index.repeat(self.output_buckets)].reset_index(drop=True)
        repeated["target_bucket"] = np.tile(np.arange(self.output_buckets), len(scale))
        repeated["case_id"] = np.repeat(np.arange(len(scale)), self.output_buckets)
        total, long_shape = self.predict_frames(scale, repeated, group_columns=("case_id",))
        return total, long_shape["conditional_share"].to_numpy().reshape(len(scale), -1)

    def save_native(self, directory: Path, metadata: dict[str, object]) -> Path:
        """Persist two native Booster files, frozen categories, and checksums."""
        if self.scale_model is None or self.shape_model is None:
            raise RuntimeError("Cannot save an unfitted LightGBM model.")
        required = {
            "fold_id",
            "feature_schema_version",
            "training_cutoff",
            "validation_range",
            "categorical_features",
        }
        missing = required.difference(metadata)
        if missing:
            raise ValueError(f"LightGBM artifact metadata missing fields: {sorted(missing)}")
        categorical_metadata = metadata["categorical_features"]
        if not isinstance(categorical_metadata, (list, tuple)):
            raise TypeError("Artifact categorical_features must be a list or tuple.")
        if tuple(str(value) for value in categorical_metadata) != self.categorical_features:
            raise ValueError("Artifact categorical metadata contradicts the fitted vocabulary.")
        directory.mkdir(parents=True, exist_ok=False)
        files = []
        for name, model in (("scale.txt", self.scale_model), ("shape.txt", self.shape_model)):
            path = directory / name
            booster = getattr(model, "booster_", model)
            booster.save_model(path)
            files.append({"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
        payload = {
            **metadata,
            "model_family": "lightgbm-scale-long-shape",
            "lightgbm_version": importlib.metadata.version("lightgbm"),
            "lightgbm_execution": self.execution.identity(),
            "config": asdict(self.config),
            "selected_scale_config": asdict(self.scale_config),
            "selected_shape_config": asdict(self.shape_config),
            "feature_columns": self.feature_columns,
            "shape_feature_columns": self.shape_feature_columns,
            "category_vocabulary": self.category_vocabulary,
            "output_buckets": self.output_buckets,
            "selected_iterations": {
                "scale": int(
                    getattr(self.scale_model, "best_iteration_", 0)
                    or getattr(self.scale_model, "booster_", self.scale_model).current_iteration()
                ),
                "shape": int(
                    getattr(self.shape_model, "best_iteration_", 0)
                    or getattr(self.shape_model, "booster_", self.shape_model).current_iteration()
                ),
            },
            "models": files,
        }
        (directory / "manifest.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        return directory

    @classmethod
    def load_native(
        cls,
        directory: Path,
        *,
        expected_execution: LightGBMExecutionOptions | None = None,
    ) -> tuple[LightGBMVolumeModel, dict[str, object]]:
        """Load checksummed native Boosters and category compatibility state."""
        payload = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        execution = _execution_from_manifest(payload)
        if expected_execution is not None and execution != expected_execution:
            raise ValueError("LightGBM artifact execution identity mismatch.")
        records = payload.get("models")
        if not isinstance(records, list) or len(records) != 2:
            raise TypeError("LightGBM manifest must contain scale and shape models.")
        if [record.get("path") for record in records] != ["scale.txt", "shape.txt"]:
            raise ValueError("LightGBM native model paths must be scale.txt and shape.txt.")
        lightgbm = _lightgbm()
        instance = cls(LightGBMConfig(**payload["config"]), execution=execution)
        instance.scale_config = LightGBMConfig(
            **payload.get("selected_scale_config", payload["config"])
        )
        instance.shape_config = LightGBMConfig(
            **payload.get("selected_shape_config", payload["config"])
        )
        loaded = []
        for record in records:
            path = directory / str(record["path"])
            if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
                raise ValueError(f"LightGBM checksum mismatch: {path.name}")
            loaded.append(lightgbm.Booster(model_file=str(path)))
        instance.scale_model, instance.shape_model = loaded
        instance.feature_columns = tuple(payload["feature_columns"])
        instance.shape_feature_columns = tuple(payload["shape_feature_columns"])
        instance.categorical_features = tuple(payload["categorical_features"])
        instance.category_vocabulary = {
            name: tuple(values) for name, values in payload["category_vocabulary"].items()
        }
        instance.output_buckets = payload.get("output_buckets")
        return instance, payload

    def _prepare_scale(self, frame: pd.DataFrame) -> pd.DataFrame:
        return _prepare_frame(frame, self.feature_columns, self.category_vocabulary)

    def _prepare_shape(self, frame: pd.DataFrame) -> pd.DataFrame:
        return _prepare_frame(frame, self.shape_feature_columns, self.category_vocabulary)


def run_lightgbm_grid(
    training: tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray],
    validation: tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray],
    *,
    categorical_features: tuple[str, ...] = ("symbol",),
    seed: int = 13,
    execution: LightGBMExecutionOptions | None = None,
    candidate_configs: tuple[LightGBMConfig, ...] | None = None,
    resume_directory: Path | None = None,
    resume_identity: dict[str, object] | None = None,
) -> tuple[LightGBMVolumeModel, tuple[LightGBMGridResult, ...]]:
    """Run the exact eight-point validation grid and select without test/TCA data."""
    candidates: list[tuple[LightGBMVolumeModel, LightGBMGridResult]] = []
    persisted: list[tuple[Path, LightGBMGridResult]] = []
    execution_options = execution or LightGBMExecutionOptions()
    configs = candidate_configs or tuple(
        LightGBMConfig(leaves, child, l2, seed=seed)
        for leaves, child, l2 in product((15, 31), (50, 200), (1.0, 10.0))
    )
    _validate_candidate_grid(configs)
    if (resume_directory is None) != (resume_identity is None):
        raise ValueError("Grid resume directory and identity must be supplied together.")
    if resume_directory is not None:
        resume_directory.mkdir(parents=True, exist_ok=True)
    for index, config in enumerate(configs, start=1):
        candidate_path = (
            resume_directory
            / (
                f"candidate-{index:02d}-leaves-{config.num_leaves}"
                f"-child-{config.min_child_samples}-l2-{config.reg_lambda:g}"
            )
            if resume_directory is not None
            else None
        )
        loaded = _load_grid_candidate(
            candidate_path,
            config=config,
            execution=execution_options,
            resume_identity=resume_identity,
        )
        if loaded is not None:
            model, result = loaded
            assert candidate_path is not None
            persisted.append((candidate_path, result))
            del model
            continue
        model = LightGBMVolumeModel(config, execution=execution_options).fit_frames(
            *training, categorical_features=categorical_features, validation=validation
        )
        case_columns = _case_columns(validation[2])
        predicted_total, predicted_shape = model.predict_frames(
            validation[0], validation[2], group_columns=case_columns
        )
        actual = validation[2].loc[:, [*case_columns, "target_bucket"]].copy()
        actual["conditional_share"] = validation[3]
        merged = actual.merge(
            predicted_shape,
            on=[*case_columns, "target_bucket"],
            suffixes=("_actual", "_predicted"),
            validate="one_to_one",
        )
        result = LightGBMGridResult(
            config,
            float(np.mean(np.abs(np.log1p(predicted_total) - np.log1p(validation[1])))),
            _long_curve_error(merged, case_columns),
            int(getattr(model.scale_model, "best_iteration_", 0) or config.n_estimators),
            int(getattr(model.shape_model, "best_iteration_", 0) or config.n_estimators),
        )
        if not np.isfinite([result.scale_mae, result.shape_error]).all():
            raise ValueError("LightGBM grid produced non-finite selection metrics.")
        if candidate_path is None:
            candidates.append((model, result))
        else:
            _persist_grid_candidate(
                candidate_path,
                model=model,
                result=result,
                resume_identity=resume_identity,
            )
            persisted.append((candidate_path, result))
            del model
    if persisted:
        selected_scale_path = min(persisted, key=lambda item: item[1].scale_mae)[0]
        selected_shape_path = min(persisted, key=lambda item: item[1].shape_error)[0]
        selected_scale, _ = LightGBMVolumeModel.load_native(
            selected_scale_path, expected_execution=execution_options
        )
        selected_shape, _ = LightGBMVolumeModel.load_native(
            selected_shape_path, expected_execution=execution_options
        )
        selected_scale.shape_model = selected_shape.shape_model
        selected_scale.scale_config = selected_scale.config
        selected_scale.shape_config = selected_shape.config
        return selected_scale, tuple(item[1] for item in persisted)
    selected_scale = min(candidates, key=lambda item: item[1].scale_mae)[0]
    selected_shape = min(candidates, key=lambda item: item[1].shape_error)[0]
    selected_scale.shape_model = selected_shape.shape_model
    selected_scale.scale_config = selected_scale.config
    selected_scale.shape_config = selected_shape.config
    return selected_scale, tuple(item[1] for item in candidates)


def _persist_grid_candidate(
    path: Path,
    *,
    model: LightGBMVolumeModel,
    result: LightGBMGridResult,
    resume_identity: dict[str, object] | None,
) -> None:
    """Publish one candidate atomically so interruption never creates a reusable partial."""
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}-", dir=path.parent)) / "candidate"
    model.save_native(
        temporary,
        {
            "fold_id": str((resume_identity or {}).get("fold_id", "fixture")),
            "feature_schema_version": "paper-lgbm-grid-candidate-v1",
            "training_cutoff": (resume_identity or {}).get("training_cutoff", "synthetic-fixture"),
            "validation_range": (resume_identity or {}).get(
                "validation_range", ["synthetic-fixture"]
            ),
            "categorical_features": list(model.categorical_features),
            "resume_identity": resume_identity,
        },
    )
    (temporary / "grid-result.json").write_text(
        json.dumps(
            {
                "config": asdict(result.config),
                "scale_mae": result.scale_mae,
                "shape_error": result.shape_error,
                "scale_iterations": result.scale_iterations,
                "shape_iterations": result.shape_iterations,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    manifest_path = temporary / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["grid_result_sha256"] = hashlib.sha256(
        (temporary / "grid-result.json").read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    temporary.parent.rmdir()


def _load_grid_candidate(
    path: Path | None,
    *,
    config: LightGBMConfig,
    execution: LightGBMExecutionOptions,
    resume_identity: dict[str, object] | None,
) -> tuple[LightGBMVolumeModel, LightGBMGridResult] | None:
    """Load a complete compatible candidate or fail closed on conflicting state."""
    if path is None or not path.exists():
        return None
    result_path = path / "grid-result.json"
    manifest_path = path / "manifest.json"
    if not result_path.is_file() or not manifest_path.is_file():
        raise ValueError(f"Incomplete resumable LightGBM candidate: {path}")
    model, manifest = LightGBMVolumeModel.load_native(path, expected_execution=execution)
    if hashlib.sha256(result_path.read_bytes()).hexdigest() != manifest.get("grid_result_sha256"):
        raise ValueError("Resumable LightGBM candidate result checksum mismatch.")
    if manifest.get("resume_identity") != resume_identity or model.config != config:
        raise ValueError(f"Resumable LightGBM candidate identity mismatch: {path}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    if payload.get("config") != asdict(config):
        raise ValueError(f"Resumable LightGBM candidate config mismatch: {path}")
    result = LightGBMGridResult(
        config=config,
        scale_mae=float(payload["scale_mae"]),
        shape_error=float(payload["shape_error"]),
        scale_iterations=int(payload["scale_iterations"]),
        shape_iterations=int(payload["shape_iterations"]),
    )
    if not np.isfinite([result.scale_mae, result.shape_error]).all():
        raise ValueError("Resumable LightGBM candidate has non-finite metrics.")
    return model, result


def _validate_candidate_grid(configs: tuple[LightGBMConfig, ...]) -> None:
    """Reject a missing, duplicated, or protocol-incompatible grid candidate."""
    coordinates = {(item.num_leaves, item.min_child_samples, item.reg_lambda) for item in configs}
    expected = set(product((15, 31), (50, 200), (1.0, 10.0)))
    fixed = {
        (
            item.learning_rate,
            item.n_estimators,
            item.early_stopping_rounds,
            item.feature_fraction,
            item.bagging_fraction,
            item.bagging_freq,
        )
        for item in configs
    }
    if coordinates != expected or len(configs) != 8 or fixed != {(0.03, 2_000, 100, 0.8, 0.8, 1)}:
        raise ValueError("LightGBM candidates do not form one matched locked eight-point grid.")


def _parameters(config: LightGBMConfig, execution: LightGBMExecutionOptions) -> dict[str, object]:
    parameters: dict[str, object] = {
        "num_leaves": config.num_leaves,
        "min_child_samples": config.min_child_samples,
        "reg_lambda": config.reg_lambda,
        "learning_rate": config.learning_rate,
        "n_estimators": config.n_estimators,
        "feature_fraction": config.feature_fraction,
        "bagging_fraction": config.bagging_fraction,
        "bagging_freq": config.bagging_freq,
        "random_state": config.seed,
        "n_jobs": execution.num_threads,
        "verbosity": -1,
    }
    if execution.device_type == "cpu":
        parameters.update({"deterministic": True, "force_col_wise": True})
    else:
        parameters.update(
            {
                "device_type": "gpu",
                "gpu_platform_id": execution.gpu_platform_id,
                "gpu_device_id": execution.gpu_device_id,
                "gpu_use_dp": execution.gpu_use_dp,
            }
        )
    return parameters


def _execution_from_manifest(payload: dict[str, object]) -> LightGBMExecutionOptions:
    recorded = payload.get("lightgbm_execution")
    if not isinstance(recorded, dict):
        raise ValueError("LightGBM manifest is missing execution identity.")
    try:
        return LightGBMExecutionOptions(**recorded)
    except TypeError as exc:
        raise ValueError("LightGBM manifest execution identity is invalid.") from exc


def _fit_categories(
    frames: tuple[pd.DataFrame, ...], categorical: tuple[str, ...]
) -> dict[str, tuple[str, ...]]:
    vocabulary = {}
    for name in categorical:
        values = [frame[name].astype(str) for frame in frames if name in frame]
        if len(values) != len(frames):
            raise ValueError(f"Categorical feature is missing from a training frame: {name}")
        vocabulary[name] = tuple(sorted(pd.concat(values).unique()))
    return vocabulary


def _apply_categories(
    frame: pd.DataFrame,
    vocabulary: dict[str, tuple[str, ...]],
    *,
    training: bool,
) -> pd.DataFrame:
    result = frame.copy()
    for name, categories in vocabulary.items():
        values = result[name].astype(str)
        unknown = sorted(set(values).difference(categories))
        if unknown and not training:
            raise ValueError(f"Incompatible categorical state for {name}: {unknown}")
        result[name] = pd.Categorical(values, categories=categories)
    return result


def _prepare_frame(
    frame: pd.DataFrame,
    columns: tuple[str, ...],
    vocabulary: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    missing = set(columns).difference(frame.columns)
    if missing or set(vocabulary).difference(frame.columns):
        raise ValueError(f"LightGBM frame schema mismatch; missing={sorted(missing)}")
    result = _apply_categories(frame.loc[:, list(columns)], vocabulary, training=False)
    numeric = result.select_dtypes(exclude="category")
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("LightGBM numeric features must be finite.")
    return result


def _validate_scale_frame(
    features: pd.DataFrame, remaining_volume: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray]:
    target = np.asarray(remaining_volume, dtype=float)
    if features.empty or target.shape != (len(features),) or not np.isfinite(target).all():
        raise ValueError("Scale frame and target are empty, incompatible, or non-finite.")
    if (target < 0).any():
        raise ValueError("Remaining-volume targets must be non-negative.")
    return features.reset_index(drop=True).copy(), target


def _baseline_remaining(features: pd.DataFrame) -> np.ndarray:
    if "baseline_remaining_volume" not in features:
        raise ValueError("Scale features require the frozen causal baseline remaining volume.")
    baseline = features["baseline_remaining_volume"].to_numpy(dtype=float)
    if not np.isfinite(baseline).all() or (baseline < 0).any():
        raise ValueError("Causal baseline remaining volume must be finite and non-negative.")
    return baseline


def _scale_residual(features: pd.DataFrame, remaining_volume: np.ndarray) -> np.ndarray:
    baseline = _baseline_remaining(features)
    return np.log1p(remaining_volume) - np.log1p(baseline)


def _validate_shape_frame(
    features: pd.DataFrame, conditional_share: np.ndarray
) -> tuple[pd.DataFrame, np.ndarray]:
    target = np.asarray(conditional_share, dtype=float)
    if (
        features.empty
        or "target_bucket" not in features
        or target.shape != (len(features),)
        or not np.isfinite(target).all()
        or (target < 0).any()
    ):
        raise ValueError("Long-form shape frame or target is invalid.")
    return features.reset_index(drop=True).copy(), target


def _wide_shape_to_long(scale: pd.DataFrame, shape: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    repeated = scale.loc[scale.index.repeat(shape.shape[1])].reset_index(drop=True)
    repeated["target_bucket"] = np.tile(np.arange(shape.shape[1]), len(scale))
    return repeated, shape.reshape(-1)


def _validated_arrays(
    features: np.ndarray, remaining_volume: np.ndarray, conditional_shape: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(features, dtype=float)
    total = np.asarray(remaining_volume, dtype=float)
    shape = np.asarray(conditional_shape, dtype=float)
    if x.ndim != 2 or total.shape != (len(x),) or shape.ndim != 2 or len(shape) != len(x):
        raise ValueError("LightGBM arrays have incompatible shapes.")
    if not all(np.isfinite(value).all() for value in (x, total, shape)):
        raise ValueError("LightGBM arrays must be finite.")
    if (total < 0).any() or (shape < 0).any() or not np.allclose(shape.sum(axis=1), 1.0):
        raise ValueError("Targets require non-negative volume and row-normalized shape.")
    return x, total, shape


def _case_columns(frame: pd.DataFrame) -> tuple[str, ...]:
    columns = tuple(
        name
        for name in ("fold_id", "instrument_id", "session_date", "as_of", "case_id")
        if name in frame
    )
    if not columns:
        raise ValueError("Long-form shape validation requires stable case identity columns.")
    return columns


def _long_curve_error(frame: pd.DataFrame, group_columns: tuple[str, ...]) -> float:
    errors = []
    for _, group in frame.groupby(list(group_columns), sort=False):
        ordered = group.sort_values("target_bucket", kind="stable")
        errors.append(
            np.mean(
                np.abs(
                    np.cumsum(ordered["conditional_share_predicted"])
                    - np.cumsum(ordered["conditional_share_actual"])
                )
            )
        )
    return float(np.mean(errors))


def create_paper_volume_model(
    family: str,
    *,
    config: LightGBMConfig | None = None,
    execution: LightGBMExecutionOptions | None = None,
) -> LightGBMVolumeModel:
    """Create the locked paper forecaster and reject model-family expansion."""
    if family != "lightgbm-scale-shape":
        raise ValueError(f"Unknown paper volume-model family: {family}")
    return LightGBMVolumeModel(config, execution=execution)
