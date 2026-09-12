from __future__ import annotations

import pickle
import random
from dataclasses import asdict, replace
from datetime import datetime

import numpy as np
import pytest
import torch
from torch.nn import functional

from execsim.data.paper.manifests import file_sha256, read_json, stable_hash
from execsim.ml.representations.checkpoints import (
    load_checkpoint,
    load_trusted_resume_state,
    save_checkpoint,
    save_trusted_resume_state,
)
from execsim.ml.representations.diagnostics import representation_diagnostics, sparse_acceptance
from execsim.ml.representations.difficulty import (
    baseline_difficulty_components,
    difficulty_weights,
)
from execsim.ml.representations.embeddings import (
    compose_causal_embedding,
    embedding_cache_identity,
    export_frozen_embedding,
    write_embedding_artifact_manifest,
    write_embedding_parquet,
)
from execsim.ml.representations.evaluation import (
    complete_horizon_origin_mask,
    fit_latent_normalization,
    future_volume_surprise,
    normalized_latent_mse,
    representation_baselines,
)
from execsim.ml.representations.historical_trainer import (
    _resolve_device,
    assert_finite_training_state,
)
from execsim.ml.representations.jepa import PredictiveRepresentationModel
from execsim.ml.representations.links import rep_relu
from execsim.ml.representations.predictors import (
    LinearPredictor,
    create_predictor,
    trainable_parameter_count,
)
from execsim.ml.representations.rdmreg import (
    generalized_gaussian_moments,
    generalized_gaussian_samples,
    rectified_generalized_gaussian_moments,
    sliced_wasserstein_distance,
)
from execsim.ml.representations.schemas import (
    CheckpointCompatibility,
    CheckpointManifest,
    EmbeddingCacheKey,
    RepresentationConfig,
    rectified_gaussian_development_config,
    sparse_location_for_zero_fraction,
)
from execsim.ml.representations.selection import (
    CommonLambdaCandidate,
    select_common_rdm_lambda,
)
from execsim.ml.representations.trainer import (
    RepresentationTrainingOptions,
    calibrate_from_training_batches,
    fit_representation_arrays,
    train_synthetic_fixture,
)


def _checkpoint_compatibility(manifest: CheckpointManifest) -> CheckpointCompatibility:
    return CheckpointCompatibility(
        geometry=manifest.geometry,
        predictor_family=manifest.predictor_family,
        fold_id=manifest.fold_id,
        cutoff=manifest.cutoff,
        universe_manifest_hash=manifest.universe_manifest_hash,
        dataset_manifest_hash=manifest.dataset_manifest_hash,
        sequence_manifest_hash=manifest.sequence_manifest_hash,
        normalization_hash=manifest.normalization_hash,
        architecture_hash=manifest.architecture_hash,
        training_config_hash=manifest.training_config_hash,
        paper_config_hash=manifest.paper_config_hash,
        generalized_gaussian_p=manifest.generalized_gaussian_p,
        generalized_gaussian_mu=manifest.generalized_gaussian_mu,
        generalized_gaussian_sigma=manifest.generalized_gaussian_sigma,
        target_rms=manifest.target_rms,
        target_zero_fraction=manifest.target_zero_fraction,
        rdm_projections=manifest.rdm_projections,
        calibrated_rdm_lambda=manifest.calibrated_rdm_lambda,
        torch_version=manifest.torch_version,
        adaptation=manifest.adaptation,
    )


def test_rep_relu_has_exact_relu_forward_and_gelu_gradient() -> None:
    values = torch.tensor([-1.0, 0.2, 1.0], requires_grad=True)
    linked = rep_relu(values)
    linked.sum().backward()
    actual_gradient = values.grad.detach().clone()
    reference = values.detach().clone().requires_grad_(True)
    functional.gelu(reference).sum().backward()

    assert torch.equal(linked.detach(), functional.relu(values.detach()))
    assert torch.allclose(actual_gradient, reference.grad)


@pytest.mark.parametrize("p,sigma", [(2.0, 1.0), (1.0, 2**-0.5)])
def test_generalized_gaussian_moments_match_deterministic_monte_carlo(
    p: float, sigma: float
) -> None:
    mean, variance = generalized_gaussian_moments(p=p, mu=0.0, sigma=sigma)
    samples = generalized_gaussian_samples((250_000,), p=p, mu=0.0, sigma=sigma, seed=17)

    assert mean == 0
    assert variance == pytest.approx(1.0)
    assert float(samples.mean()) == pytest.approx(0.0, abs=0.01)
    assert float(samples.var()) == pytest.approx(1.0, rel=0.02)


def test_sparse_target_has_quarter_activity_and_half_rms() -> None:
    samples = generalized_gaussian_samples(
        (500_000,), p=1.0, mu=-np.log(2) / np.sqrt(2), sigma=2**-0.5, seed=19
    )
    linked = np.maximum(samples, 0)

    assert float(np.mean(linked > 0)) == pytest.approx(0.25, abs=0.005)
    assert float(np.sqrt(np.mean(linked**2))) == pytest.approx(0.5, abs=0.01)
    assert sparse_location_for_zero_fraction(0.5) == 0.0
    assert sparse_location_for_zero_fraction(0.875) == pytest.approx(-2 * np.log(2) / np.sqrt(2))


def test_primary_sparse_target_is_rectified_gaussian_with_quarter_activity() -> None:
    config = RepresentationConfig("sparse")
    samples = generalized_gaussian_samples(
        (500_000,), p=2.0, mu=-0.6744897501960817, sigma=1.0, seed=127
    )
    linked = np.maximum(samples, 0)

    assert config.target_parameters == pytest.approx((2.0, -0.6744897501960817, 1.0))
    assert config.target_zero_fraction == pytest.approx(0.75)
    assert config.target_rms == pytest.approx(float(np.sqrt(np.mean(linked**2))), abs=0.005)


@pytest.mark.parametrize("zero_fraction", [0.5, 0.75, 0.875])
def test_rectified_laplace_rms_is_derived_and_matches_monte_carlo(zero_fraction: float) -> None:
    mu = sparse_location_for_zero_fraction(zero_fraction)
    first, second = rectified_generalized_gaussian_moments(p=1.0, mu=mu, sigma=2**-0.5)
    samples = generalized_gaussian_samples((500_000,), p=1.0, mu=mu, sigma=2**-0.5, seed=113)
    linked = np.maximum(samples, 0)

    assert first == pytest.approx(float(linked.mean()), abs=0.005)
    assert second == pytest.approx(1.0 - zero_fraction)
    assert np.sqrt(second) == pytest.approx(float(np.sqrt(np.mean(linked**2))), abs=0.01)


@pytest.mark.parametrize("zero_fraction", [0.5, 0.75, 0.875])
def test_rectified_gaussian_sweep_moments_match_monte_carlo(zero_fraction: float) -> None:
    from scipy.special import ndtri

    mu = float(ndtri(1.0 - zero_fraction))
    first, second = rectified_generalized_gaussian_moments(p=2.0, mu=mu, sigma=1.0)
    samples = generalized_gaussian_samples((500_000,), p=2.0, mu=mu, sigma=1.0, seed=173)
    linked = np.maximum(samples, 0)

    assert first == pytest.approx(float(linked.mean()), abs=0.005)
    assert second == pytest.approx(float(np.mean(linked**2)), abs=0.01)
    assert float(np.mean(linked == 0)) == pytest.approx(zero_fraction, abs=0.005)


def test_rectified_gaussian_control_is_sparse_matched_and_development_only() -> None:
    control = rectified_gaussian_development_config(zero_fraction=0.75)
    assert control.geometry == "sparse"
    assert control.target_parameters[0] == 2.0
    assert control.target_zero_fraction == pytest.approx(0.75)
    assert control.target_rms != pytest.approx(0.5)


def test_rdmreg_is_deterministic_and_prefers_its_matched_target() -> None:
    seed = 23
    matched = generalized_gaussian_samples(
        (64, 16), p=2.0, mu=0.0, sigma=1.0, seed=seed + 1_000_003
    )
    matched_tensor = torch.as_tensor(matched)
    matched_loss = sliced_wasserstein_distance(
        matched_tensor, p=2.0, mu=0.0, sigma=1.0, projections=32, seed=seed
    )
    repeated_loss = sliced_wasserstein_distance(
        matched_tensor, p=2.0, mu=0.0, sigma=1.0, projections=32, seed=seed
    )
    mismatched_loss = sliced_wasserstein_distance(
        torch.zeros_like(matched_tensor),
        p=2.0,
        mu=0.0,
        sigma=1.0,
        projections=32,
        seed=seed,
    )

    assert torch.equal(matched_loss, repeated_loss)
    assert matched_loss == 0
    assert mismatched_loss > matched_loss
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        fp32_loss = sliced_wasserstein_distance(
            matched_tensor, p=2.0, mu=0.0, sigma=1.0, projections=8, seed=seed
        )
    assert fp32_loss.dtype == torch.float32


def test_dense_and_sparse_tiny_training_and_predictor_smoke_matrix() -> None:
    dense = train_synthetic_fixture(RepresentationConfig("dense"), steps=2, batch_size=4)
    sparse = train_synthetic_fixture(RepresentationConfig("sparse"), steps=2, batch_size=4)
    context = torch.randn(3, 8, 128)
    mask = torch.ones(3, 8, dtype=torch.bool)
    horizon = torch.tensor([0, 1, 3])

    for family in ("linear", "mlp", "transformer"):
        predictor = create_predictor(family)
        assert predictor(context, mask, horizon).shape == (3, 128)
        assert trainable_parameter_count(predictor) > 0
    torch.manual_seed(101)
    dense_model = PredictiveRepresentationModel(RepresentationConfig("dense"))
    torch.manual_seed(101)
    sparse_model = PredictiveRepresentationModel(RepresentationConfig("sparse"))
    assert trainable_parameter_count(dense_model) == trainable_parameter_count(sparse_model)
    assert trainable_parameter_count(dense_model) < 500_000
    exported = export_frozen_embedding(
        dense_model, torch.zeros(1, 8, 18), torch.ones(1, 8, dtype=torch.bool)
    )
    assert exported.shape == (644,)
    assert np.array_equal(exported[-4:], np.ones(4))
    assert dense.finite_gradients and sparse.finite_gradients
    assert sparse.zero_fraction > dense.zero_fraction
    for name, value in dense_model.encoder.state_dict().items():
        assert torch.equal(value, sparse_model.encoder.state_dict()[name])
    for name, value in dense_model.predictor.state_dict().items():
        assert torch.equal(value, sparse_model.predictor.state_dict()[name])


def test_p0_uses_independent_horizon_specific_affine_coefficients() -> None:
    predictor = LinearPredictor(context_length=2, latent_dim=3, horizons=4, conditioning_dim=5)
    with torch.no_grad():
        for horizon, layer in enumerate(predictor.horizon_linears):
            layer.weight.zero_()
            layer.bias.fill_(float(horizon + 1))
    output = predictor(
        torch.zeros(4, 2, 3),
        torch.ones(4, 2, dtype=torch.bool),
        torch.zeros(4, 5),
        torch.arange(4),
    )
    assert torch.equal(output, torch.arange(1, 5, dtype=torch.float32)[:, None].repeat(1, 3))


def test_padded_raw_values_cannot_affect_any_predictor() -> None:
    for family in ("linear", "mlp", "transformer"):
        torch.manual_seed(107)
        model = PredictiveRepresentationModel(
            RepresentationConfig("sparse", predictor_family=family)
        ).eval()
        context = torch.randn(2, 8, 18)
        mask = torch.tensor([[False, False, False, False, True, True, True, True]] * 2)
        changed = context.clone()
        changed[:, :4] = 1e20
        targets = torch.zeros(2, 4, 18)
        target_mask = torch.ones(2, 4, dtype=torch.bool)

        with torch.no_grad():
            first = model(context, mask, targets, target_mask)["prediction"]
            second = model(changed, mask, targets, target_mask)["prediction"]
        assert torch.equal(first, second)


def test_jepa_uses_one_trainable_encoder_and_masks_unavailable_horizons() -> None:
    torch.manual_seed(31)
    model = PredictiveRepresentationModel(RepresentationConfig("dense", rdm_projections_train=8))
    context = torch.randn(2, 8, 18)
    context_mask = torch.ones(2, 8, dtype=torch.bool)
    targets = torch.randn(2, 4, 18)
    target_mask = torch.tensor([[True, False, False, False], [True, False, False, False]])
    changed_targets = targets.clone()
    changed_targets[:, 1:] = 10_000

    first = model(context, context_mask, targets, target_mask)
    second = model(context, context_mask, changed_targets, target_mask)
    first["target_pre_link"].retain_grad()
    first["prediction_loss"].backward()

    assert torch.equal(first["prediction"], second["prediction"])
    assert torch.equal(first["prediction_loss"], second["prediction_loss"])
    assert first["target_pre_link"].grad is not None
    assert torch.count_nonzero(first["target_pre_link"].grad) > 0
    assert not any("ema" in name or "target_encoder" in name for name in model.state_dict())


def test_prediction_loss_weights_examples_equally_across_variable_horizons() -> None:
    torch.manual_seed(131)
    model = PredictiveRepresentationModel(RepresentationConfig("dense"))
    context = torch.randn(2, 8, 18)
    context_mask = torch.ones(2, 8, dtype=torch.bool)
    targets = torch.randn(2, 4, 18)
    target_mask = torch.tensor([[True, True, True, True], [True, False, False, False]])
    output = model(context, context_mask, targets, target_mask)
    squared = torch.mean((output["prediction"] - output["target"]) ** 2, dim=-1)
    expected = (squared[0].mean() + squared[1, 0]) / 2
    assert torch.allclose(output["prediction_loss"], expected)


def test_difficulty_weights_only_change_predictive_loss() -> None:
    torch.manual_seed(37)
    model = PredictiveRepresentationModel(RepresentationConfig("sparse", rdm_projections_train=8))
    context = torch.randn(3, 8, 18)
    context_mask = torch.ones(3, 8, dtype=torch.bool)
    targets = torch.randn(3, 4, 18)
    target_mask = torch.ones(3, 4, dtype=torch.bool)
    uniform = model.loss(
        context,
        context_mask,
        targets,
        target_mask,
        rdm_lambda=0.1,
        rdm_seed=41,
        sample_weights=torch.ones(3),
    )
    weighted = model.loss(
        context,
        context_mask,
        targets,
        target_mask,
        rdm_lambda=0.1,
        rdm_seed=41,
        sample_weights=torch.tensor([0.5, 1.0, 1.5]),
    )

    assert torch.equal(uniform["rdm_loss"], weighted["rdm_loss"])
    assert uniform["prediction_loss"] != weighted["prediction_loss"]


def test_diagnostics_difficulty_and_embedding_are_deterministic(tmp_path) -> None:
    latents = np.asarray([[0.0, 1.0, 0.0], [1.0, 0.0, 2.0], [0.0, 2.0, 1.0]])
    diagnostics = representation_diagnostics(latents)
    level, shape = baseline_difficulty_components(
        np.asarray([100.0, 200.0]),
        np.asarray([90.0, 180.0]),
        np.asarray([[0.5, 0.5], [0.4, 0.6]]),
        np.asarray([[0.6, 0.4], [0.5, 0.5]]),
    )
    weights = difficulty_weights(np.asarray([3.0, 1.0, 2.0]), np.asarray([0.0, 0.0, 0.0]))
    embedding = compose_causal_embedding(np.zeros(128), np.zeros((4, 128)))

    assert diagnostics["zero_fraction"] == pytest.approx(4 / 9)
    assert (level > 0).all() and (shape > 0).all()
    assert weights.mean() == pytest.approx(1.0)
    assert weights[0] > weights[1]
    assert embedding.shape == (644,)
    output = write_embedding_parquet(
        tmp_path / "embedding.parquet",
        embedding=embedding,
        metadata={
            "instrument_id": "asset-1",
            "symbol": "AAPL",
            "session_date": "2024-01-03",
            "as_of": "2024-01-03T10:30:00-05:00",
            "fold_id": "fold-1",
            "seed": 13,
            "geometry": "sparse",
            "adaptation": "none",
            "predictor_family": "mlp",
            "checkpoint_hash": "a" * 64,
            "sequence_hash": "b" * 64,
            "cutoff": "2023-12-29",
            "component_order": "current,h1,h2,h4,h8,horizon_availability",
        },
    )
    assert output.is_file()
    artifact_manifest = write_embedding_artifact_manifest(
        tmp_path / "embedding-manifest.json",
        artifact_id="embedding-test",
        checkpoint_hash="4" * 64,
        partition_identity="fold-1/test",
        row_count=1,
        training_cutoff="2023-12-29",
        source_hashes=("1" * 64,),
        parquet_path=output,
    )
    assert artifact_manifest.shape == (1, 644)
    assert len(artifact_manifest.parquet_sha256) == 64
    key = EmbeddingCacheKey(
        raw_hash="0" * 64,
        sequence_hash="1" * 64,
        normalization_hash="2" * 64,
        fold_id="fold-1",
        cutoff="2023-12-29",
        architecture_hash="3" * 64,
        geometry="sparse",
        sparsity_target=0.75,
        predictor_family="mlp",
        seed=13,
        checkpoint_hash="4" * 64,
        torch_version=torch.__version__,
    )
    assert embedding_cache_identity(key) == "embedding-" + stable_hash(asdict(key))[:20]
    assert embedding_cache_identity(key) != embedding_cache_identity(replace(key, seed=29))


def test_unavailable_embedding_horizons_are_zero_and_explicit() -> None:
    predicted = np.ones((4, 128))
    embedding = compose_causal_embedding(
        np.ones(128), predicted, np.asarray([True, True, False, False])
    )
    assert np.count_nonzero(embedding[128 + 2 * 128 : 128 + 4 * 128]) == 0
    assert np.array_equal(embedding[-4:], [1, 1, 0, 0])


def test_train_covariance_normalization_and_fixed_observable_probe_contract() -> None:
    train = np.asarray([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]])
    normalization = fit_latent_normalization(train)
    target = np.asarray([[1.0, 1.0], [2.0, 0.0]])
    current = np.asarray([[0.5, 1.0], [1.0, 0.0]])
    error = normalized_latent_mse(target, current, normalization)
    baselines = representation_baselines(train, current, target, normalization)

    assert error > 0
    assert set(baselines) == {"zero_nmse", "train_mean_nmse", "persistence_nmse"}
    assert np.array_equal(
        complete_horizon_origin_mask(np.asarray([3, 4, 17, 18, 25])),
        [False, True, True, True, False],
    )
    assert future_volume_surprise(np.asarray([9.0]), np.asarray([4.0])) == pytest.approx(
        np.log(2.0)
    )


def test_common_rdm_lambda_uses_both_gate_passing_geometries(tmp_path) -> None:
    errors = {0.1: (0.4, 0.5), 1.0: (0.2, 0.3), 10.0: (0.1, 0.9)}
    candidates = tuple(
        CommonLambdaCandidate(
            value,
            geometry,
            "fold-1",
            13,
            errors[value][geometry == "sparse"],
            "PASS",
            f"{index:x}" * 64,
        )
        for index, (value, geometry) in enumerate(
            (value, geometry) for value in (0.1, 1.0, 10.0) for geometry in ("dense", "sparse")
        )
    )
    output = tmp_path / "selection.json"
    assert select_common_rdm_lambda(candidates, output=output) == 1.0
    assert output.is_file()
    blocked = tuple(
        replace(item, collapse_gate_status="FAIL") if item.rdm_lambda == 1.0 else item
        for item in candidates
    )
    assert select_common_rdm_lambda(blocked) == 0.1
    rejected_sparse = tuple(
        replace(
            item,
            observable_probe_error=0.0,
            collapse_gate_status="FAIL",
            checkpoint_hash="",
            failure_reason="no sparse checkpoint passed the collapse gate",
        )
        if item.rdm_lambda == 0.1 and item.geometry == "sparse"
        else item
        for item in candidates
    )
    rejected_output = tmp_path / "selection-with-rejection.json"
    assert select_common_rdm_lambda(rejected_sparse, output=rejected_output) == 1.0
    rejected_payload = read_json(rejected_output)
    rejected_row = next(
        row
        for row in rejected_payload["candidates"]
        if row["rdm_lambda"] == 0.1 and row["geometry"] == "sparse"
    )
    assert rejected_row["failure_reason"] == "no sparse checkpoint passed the collapse gate"


def test_safetensors_checkpoint_is_compatible_and_historical_fit_is_guarded(tmp_path) -> None:
    model = PredictiveRepresentationModel(RepresentationConfig("dense"))
    manifest = CheckpointManifest(
        checkpoint_id="checkpoint-test",
        geometry="dense",
        predictor_family="mlp",
        fold_id="fold-1",
        seed=13,
        sequence_manifest_hash="a" * 64,
        normalization_hash="b" * 64,
        cutoff="2023-12-29",
        architecture_hash="c" * 64,
        torch_version=torch.__version__,
        weights_sha256="",
        checkpoint_role="best",
        dataset_manifest_hash="d" * 64,
        universe_manifest_hash="e" * 64,
        training_config_hash="f" * 64,
        validation_diagnostics=(("effective_rank", 64.0),),
        collapse_gate_status="PASS",
    )
    directory = save_checkpoint(model, tmp_path / "checkpoint", manifest)
    restored = PredictiveRepresentationModel(RepresentationConfig("dense"))
    loaded = load_checkpoint(restored, directory, expected=_checkpoint_compatibility(manifest))

    assert loaded.weights_sha256
    assert loaded.checkpoint_role == "best"
    assert loaded.collapse_gate_status == "PASS"
    with pytest.raises(ValueError, match="fold_id"):
        load_checkpoint(
            restored,
            directory,
            expected=replace(_checkpoint_compatibility(manifest), fold_id="fold-2"),
        )
    with pytest.raises(ValueError, match="geometry"):
        load_checkpoint(
            restored,
            directory,
            expected=replace(_checkpoint_compatibility(manifest), geometry="sparse"),
        )
    fixture = {
        "context": np.zeros((2, 8, 18), dtype=np.float32),
        "context_mask": np.ones((2, 8), dtype=bool),
        "targets": np.zeros((2, 4, 18), dtype=np.float32),
        "target_mask": np.ones((2, 4), dtype=bool),
    }
    with pytest.raises(PermissionError, match="disabled"):
        fit_representation_arrays(
            RepresentationConfig("dense"),
            fixture,
            fixture,
            rdm_lambda=0.01,
            data_classification="historical",
        )
    trained, result = fit_representation_arrays(
        RepresentationConfig("dense", rdm_projections_train=8),
        fixture,
        fixture,
        rdm_lambda=0.01,
        data_classification="synthetic_fixture",
        options=RepresentationTrainingOptions(max_epochs=2, patience=1, use_bfloat16=False),
    )
    assert result.finite_gradients
    assert result.steps >= 1
    assert isinstance(trained, PredictiveRepresentationModel)
    tensor_batch = {
        "context": torch.zeros(2, 8, 18),
        "context_mask": torch.ones(2, 8, dtype=torch.bool),
        "targets": torch.zeros(2, 4, 18),
        "target_mask": torch.ones(2, 4, dtype=torch.bool),
    }
    calibration_model = PredictiveRepresentationModel(
        RepresentationConfig("dense", rdm_projections_train=4)
    )
    calibrated = calibrate_from_training_batches(calibration_model, [tensor_batch] * 32, seed=13)
    assert 1e-3 <= calibrated <= 1e3


def test_resume_rejects_non_allowlisted_objects_with_valid_checksum(tmp_path) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    before = optimizer.state_dict()
    path = tmp_path / "unsupported.pt"
    # Benign unsupported type proves the boundary without an executable payload.
    torch.save({"trusted_local_only": True, "unexpected": datetime(2024, 1, 1)}, path)
    with pytest.raises(pickle.UnpicklingError):
        load_trusted_resume_state(
            path, optimizer, expected_sha256=file_sha256(path), trusted_local=True
        )
    assert optimizer.state_dict() == before


def test_trusted_resume_reproduces_the_next_optimizer_step(tmp_path) -> None:
    torch.manual_seed(43)
    config = RepresentationConfig("dense", rdm_projections_train=4)
    model = PredictiveRepresentationModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    context = torch.randn(2, 8, 18)
    context_mask = torch.ones(2, 8, dtype=torch.bool)
    targets = torch.randn(2, 4, 18)
    target_mask = torch.ones(2, 4, dtype=torch.bool)

    def step(active_model, active_optimizer) -> None:
        active_optimizer.zero_grad(set_to_none=True)
        loss = active_model.loss(
            context,
            context_mask,
            targets,
            target_mask,
            rdm_lambda=0.01,
            rdm_seed=47,
        )["loss"]
        loss.backward()
        active_optimizer.step()

    step(model, optimizer)
    checkpoint = save_checkpoint(
        model,
        tmp_path / "resume-checkpoint",
        CheckpointManifest(
            checkpoint_id="resume-test",
            geometry="dense",
            predictor_family="mlp",
            fold_id="fold-1",
            seed=43,
            sequence_manifest_hash="a" * 64,
            normalization_hash="b" * 64,
            cutoff="2023-12-29",
            architecture_hash="c" * 64,
            torch_version=torch.__version__,
            weights_sha256="",
        ),
    )
    resume_path = tmp_path / "trusted-resume.pt"
    resume_hash = save_trusted_resume_state(resume_path, optimizer, epoch=1)
    step(model, optimizer)
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}

    restored = PredictiveRepresentationModel(config)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=3e-4)
    resume_manifest = CheckpointManifest(
        checkpoint_id="resume-test",
        geometry="dense",
        predictor_family="mlp",
        fold_id="fold-1",
        seed=43,
        sequence_manifest_hash="a" * 64,
        normalization_hash="b" * 64,
        cutoff="2023-12-29",
        architecture_hash="c" * 64,
        torch_version=torch.__version__,
        weights_sha256="",
    )
    load_checkpoint(restored, checkpoint, expected=_checkpoint_compatibility(resume_manifest))
    progress = load_trusted_resume_state(
        resume_path,
        restored_optimizer,
        expected_sha256=resume_hash,
        trusted_local=True,
    )
    assert progress.epoch == 1
    step(restored, restored_optimizer)

    for name, value in restored.state_dict().items():
        assert torch.equal(value, expected[name])


def test_interrupted_resume_matches_continuous_model_scheduler_and_rng(tmp_path) -> None:
    random.seed(211)
    np.random.seed(211)
    torch.manual_seed(211)
    config = RepresentationConfig("dense", rdm_projections_train=4)
    model = PredictiveRepresentationModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=4)

    def random_step(active_model, active_optimizer, active_scheduler) -> None:
        context = torch.randn(2, 8, 18)
        targets = torch.randn(2, 4, 18)
        rdm_seed = random.randrange(1_000_000) + int(np.random.randint(1_000_000))
        active_optimizer.zero_grad(set_to_none=True)
        loss = active_model.loss(
            context,
            torch.ones(2, 8, dtype=torch.bool),
            targets,
            torch.ones(2, 4, dtype=torch.bool),
            rdm_lambda=0.01,
            rdm_seed=rdm_seed,
        )["loss"]
        loss.backward()
        active_optimizer.step()
        active_scheduler.step()

    for _ in range(2):
        random_step(model, optimizer, scheduler)
    manifest = CheckpointManifest(
        checkpoint_id="interrupt-resume",
        geometry="dense",
        predictor_family="mlp",
        fold_id="fold-1",
        seed=211,
        sequence_manifest_hash="a" * 64,
        normalization_hash="b" * 64,
        cutoff="2023-12-29",
        architecture_hash="c" * 64,
        torch_version=torch.__version__,
        weights_sha256="",
    )
    checkpoint = save_checkpoint(model, tmp_path / "interrupt-checkpoint", manifest)
    resume = tmp_path / "interrupt-resume.pt"
    resume_hash = save_trusted_resume_state(
        resume,
        optimizer,
        scheduler=scheduler,
        epoch=2,
        global_step=2,
        sampler_epoch=2,
        rdm_counter=2,
    )
    for _ in range(2):
        random_step(model, optimizer, scheduler)
    expected = {name: value.detach().clone() for name, value in model.state_dict().items()}
    expected_scheduler = scheduler.state_dict()

    random.seed(999)
    np.random.seed(999)
    torch.manual_seed(999)
    restored = PredictiveRepresentationModel(config)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=3e-4)
    restored_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(restored_optimizer, T_max=4)
    load_checkpoint(restored, checkpoint, expected=_checkpoint_compatibility(manifest))
    progress = load_trusted_resume_state(
        resume,
        restored_optimizer,
        scheduler=restored_scheduler,
        expected_sha256=resume_hash,
        trusted_local=True,
    )
    for _ in range(2):
        random_step(restored, restored_optimizer, restored_scheduler)

    assert progress.global_step == progress.sampler_epoch == progress.rdm_counter == 2
    assert restored_scheduler.state_dict() == expected_scheduler
    for name, value in restored.state_dict().items():
        assert torch.equal(value, expected[name])


def test_historical_trainer_fails_immediately_on_nonfinite_loss_or_gradient() -> None:
    model = torch.nn.Linear(2, 1)
    with pytest.raises(FloatingPointError, match="loss"):
        assert_finite_training_state(torch.tensor(float("nan")), model, gradients_required=False)
    model.weight.grad = torch.full_like(model.weight, float("inf"))
    with pytest.raises(FloatingPointError, match="gradient"):
        assert_finite_training_state(torch.tensor(1.0), model, gradients_required=True)


def test_device_resolution_and_sparse_checkpoint_gate_fail_closed() -> None:
    assert _resolve_device(torch, "cpu") == "cpu"
    collapsed = representation_diagnostics(np.ones((8, 128)))
    assert sparse_acceptance(collapsed)
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="CUDA"):
            _resolve_device(torch, "cuda")
