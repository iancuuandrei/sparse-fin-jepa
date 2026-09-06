from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import date, time
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import yaml

from execsim.config import ExecSimConfig, load_config, load_project_dotenv
from execsim.orders import OrderSide, ParentOrder

if TYPE_CHECKING:
    from execsim.ml.models.lightgbm_adapter import LightGBMExecutionOptions

STRATEGIES = ("twap", "vwap", "pov", "almgren-chriss", "optimal", "mpc")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="execsim",
        description="Optimal-execution research CLI.",
        epilog=(
            "Deployable policies use only point-in-time observations and forecasts: "
            "twap, vwap, pov, almgren-chriss, optimal, and mpc. Oracle behavior is "
            "evaluation-only and not exposed by simulate. No trained ML artifact is included; "
            "prepare manifests and inspect a future run with `execsim ml training-plan`, then "
            "launch an authorized fit with `execsim ml train` without --dry-run."
        ),
    )
    commands = parser.add_subparsers(dest="command")
    for name, description in (
        ("show-config", "Load the project configuration."),
        ("smoke", "Run an installation check."),
        ("download-data", "Download and validate Alpaca minute bars."),
        ("build-manifest", "Build the processed-data manifest."),
        ("validate-data", "Validate processed minute bars."),
    ):
        command = commands.add_parser(name, help=description)
        _add_config_argument(command)

    simulate = commands.add_parser("simulate", help="Simulate one execution policy.")
    _add_simulation_arguments(simulate, include_strategy=True)
    legacy = commands.add_parser("simulate-twap", help="Alias for simulate --strategy twap.")
    _add_simulation_arguments(legacy, include_strategy=False)

    experiment = commands.add_parser("experiment", help="Run or inspect an experiment.")
    experiment_commands = experiment.add_subparsers(dest="experiment_command", required=True)
    run = experiment_commands.add_parser("run", help="Run a YAML experiment grid.")
    run.add_argument("--config", type=Path, default=Path("configs/experiment.yaml"))
    report = experiment_commands.add_parser("report", help="Locate an existing report.")
    report.add_argument("--run-id", required=True)
    report.add_argument("--reports-root", type=Path, default=Path("reports/runs"))

    scenario = commands.add_parser("scenario", help="Generate deterministic synthetic bars.")
    scenario.add_argument("--volume-shape", default="u_shaped")
    scenario.add_argument("--price-path", default="constant")
    scenario.add_argument("--seed", type=int, default=17)
    scenario.add_argument("--output", type=Path, required=True)

    ml = commands.add_parser("ml", help="Operate point-in-time ML datasets and training plans.")
    ml_commands = ml.add_subparsers(dest="ml_command", required=True)
    build = ml_commands.add_parser("build-dataset")
    _add_config_argument(build)
    build.add_argument("--mode", choices=("static", "dynamic"), default="static")
    build.add_argument("--bucket-minutes", type=int, choices=(1, 5, 15), default=5)
    build.add_argument("--output-root", type=Path, default=Path("data/ml"))
    build.add_argument("--source", type=Path, action="append", default=[])
    build.add_argument("--allow-incomplete-sessions", action="store_true")
    for command_name in ("validate-dataset", "inspect-dataset"):
        command = ml_commands.add_parser(command_name)
        command.add_argument("--manifest", type=Path, required=True)
    split = ml_commands.add_parser("create-splits")
    split.add_argument("--manifest", type=Path, required=True)
    split.add_argument("--output", type=Path, required=True)
    split.add_argument("--initial-train-sessions", type=int, default=252)
    split.add_argument("--validation-sessions", type=int, default=21)
    split.add_argument("--test-sessions", type=int, default=21)
    split.add_argument("--step-sessions", type=int, default=21)
    split.add_argument("--embargo-sessions", type=int, default=0)
    for command_name in ("training-plan", "train"):
        command = ml_commands.add_parser(command_name)
        command.add_argument("--config", type=Path, default=Path("configs/ml.yaml"))
        command.add_argument("--dry-run", action="store_true")
    paper = ml_commands.add_parser("paper", help="Plan or run the locked sparse-JEPA study.")
    paper_commands = paper.add_subparsers(dest="paper_command", required=True)
    for command_name in (
        "build-universe",
        "download-data",
        "validate-data",
        "build-sequences",
        "validate-sequences",
        "plan",
        "plan-representation",
        "train-representation",
        "export-embeddings",
        "train-volume-model",
        "evaluate-forecast",
        "evaluate-representation",
        "run-tca",
        "report",
        "run",
    ):
        command = paper_commands.add_parser(command_name)
        command.add_argument(
            "--config",
            type=Path,
            default=Path("configs/paper/sparse_jepa/data.yaml"),
        )
        command.add_argument("--dry-run", action="store_true")
        command.add_argument("--enable-network", action="store_true")
        command.add_argument("--enable-historical-training", action="store_true")
        command.add_argument("--enable-full-paper-run", action="store_true")
        command.add_argument("--runtime-approval", type=Path, default=None)
        command.add_argument("--trust-local-resume", action="store_true")
        command.add_argument("--synthetic-fixture", action="store_true")
        command.add_argument("--input", type=Path, default=None)
        command.add_argument("--output", type=Path, default=None)
        command.add_argument("--instrument-id", default=None)
        command.add_argument("--symbol", default=None)
        command.add_argument("--cutoff", default=None)
        command.add_argument("--seasonal-input", type=Path, default=None)
        command.add_argument("--spy-input", type=Path, default=None)
        command.add_argument("--spy-seasonal-input", type=Path, default=None)
        command.add_argument("--previous-close", type=float, default=None)
        if command_name == "train-volume-model":
            command.add_argument("--lightgbm-device", choices=("cpu", "gpu"), default="cpu")
            command.add_argument("--lightgbm-gpu-platform-id", type=int, default=None)
            command.add_argument("--lightgbm-gpu-device-id", type=int, default=None)
            command.add_argument("--lightgbm-gpu-use-dp", action="store_true")
            command.add_argument("--lightgbm-num-threads", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    load_project_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    try:
        return _dispatch(args)
    except (
        FileNotFoundError,
        KeyError,
        PermissionError,
        TypeError,
        ValueError,
        RuntimeError,
    ) as exc:
        parser.error(str(exc))
    return 2


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "show-config":
        print(yaml.safe_dump(load_config(args.config).to_dict(), sort_keys=False).rstrip())
        return 0
    if args.command == "smoke":
        config = load_config(args.config)
        print(f"smoke: ok | project={config.project_name} | timezone={config.timezone}")
        return 0
    if args.command == "download-data":
        from execsim.data.download import download_and_prepare_data

        result = download_and_prepare_data(load_config(args.config))
        for item in result.symbols:
            print(f"{item.symbol}: raw_rows={item.raw_rows} processed_rows={item.processed_rows}")
        print(f"manifest_path={result.manifest_path}")
        return 0
    if args.command == "build-manifest":
        from execsim.data.manifest import build_dataset_manifest

        config = load_config(args.config)
        manifest = build_dataset_manifest(config)
        print(f"manifest: rows={len(manifest)} path={config.resolved_manifest_path}")
        return 0
    if args.command == "validate-data":
        return _validate_market_data(load_config(args.config))
    if args.command in {"simulate", "simulate-twap"}:
        forced = "twap" if args.command == "simulate-twap" else None
        return _simulate(args, forced_strategy=forced)
    if args.command == "experiment":
        return _experiment(args)
    if args.command == "scenario":
        return _scenario(args)
    if args.command == "ml":
        return _ml(args)
    raise ValueError(f"Unknown command: {args.command}")


def _simulate(args: argparse.Namespace, forced_strategy: str | None) -> int:
    from execsim.costs import CostParameter, LinearTemporaryImpactModel, ParameterProvenance
    from execsim.data.loaders import load_processed_symbol_bars, slice_processed_symbol_bars
    from execsim.forecasting import HistoricalProfileForecaster
    from execsim.policies import ExecutionConstraints, create_policy
    from execsim.simulator import simulate_policy

    config = load_config(args.config)
    order, participation = _build_parent_order_from_args(args, config)
    strategy = forced_strategy or args.strategy
    all_bars = load_processed_symbol_bars(config, order.symbol)
    day_bars = slice_processed_symbol_bars(all_bars, order.symbol, order.trade_date)
    provider = None
    if strategy in {"vwap", "optimal", "mpc"}:
        provider = HistoricalProfileForecaster(all_bars, lookback_sessions=args.lookback_sessions)
    result = simulate_policy(
        parent_order=order,
        bars=day_bars,
        policy=create_policy(
            strategy,
            risk_aversion=args.risk_aversion,
            half_spread=args.half_spread,
            temporary_impact=args.temporary_impact,
            volatility=args.volatility,
            pov_target_rate=args.pov_target_rate,
        ),
        constraints=ExecutionConstraints(participation, participation, config.timezone),
        cost_model=LinearTemporaryImpactModel(
            CostParameter(args.half_spread, ParameterProvenance.ASSUMED, "CLI assumption"),
            CostParameter(args.temporary_impact, ParameterProvenance.ASSUMED, "CLI assumption"),
        ),
        forecast_provider=provider,
    )
    _print_simulation_result(result, as_json=args.json)
    return 0


def _experiment(args: argparse.Namespace) -> int:
    if args.experiment_command == "report":
        report = args.reports_root / args.run_id / "REPORT.md"
        if not report.exists():
            raise FileNotFoundError(f"Experiment report not found: {report}")
        print(report.resolve())
        return 0
    from execsim.data.loaders import load_processed_symbol_bars
    from execsim.experiments import ExperimentRunner, ExperimentSpec

    payload = _read_yaml_mapping(args.config)
    project = load_config(payload.get("project_config"))
    values = dict(payload.get("experiment", payload))
    spec = ExperimentSpec(
        symbols=tuple(str(value).upper() for value in values["symbols"]),
        trade_dates=tuple(_parse_date(str(value)) for value in values["trade_dates"]),
        quantities=tuple(int(value) for value in values["quantities"]),
        sides=tuple(values.get("sides", ["buy"])),
        strategies=tuple(values.get("strategies", STRATEGIES)),
        start_time=_parse_time(str(values.get("start_time", "10:00"))),
        end_time=_parse_time(str(values.get("end_time", "11:00"))),
        planned_participation_rate=float(values.get("planned_participation_rate", 0.1)),
        hard_participation_rate=float(values.get("hard_participation_rate", 0.1)),
        pov_target_rate=float(values.get("pov_target_rate", 0.05)),
        half_spread=float(values.get("half_spread", 0.01)),
        temporary_impacts=tuple(float(value) for value in values.get("temporary_impacts", [0.1])),
        volatility=float(values.get("volatility", 0.01)),
        risk_aversions=tuple(float(value) for value in values.get("risk_aversions", [0.0, 0.01])),
        profile_estimator=str(values.get("profile_estimator", "mean")),
        profile_lookback_sessions=int(values.get("profile_lookback_sessions", 20)),
        seed=int(values.get("seed", 17)),
        include_oracle=bool(values.get("include_oracle", False)),
    )
    bars = {symbol: load_processed_symbol_bars(project, symbol) for symbol in spec.symbols}
    outputs = ExperimentRunner(spec, bars, Path(project.reports_dir) / "runs").run()
    print(
        json.dumps(
            {
                "run_id": outputs.run_id,
                "output_dir": str(outputs.output_dir),
                "runs": len(outputs.results),
            },
            indent=2,
        )
    )
    return 0


def _scenario(args: argparse.Namespace) -> int:
    from execsim.data.scenarios import ScenarioConfig, generate_scenario

    bars = generate_scenario(
        ScenarioConfig(
            volume_scenario=args.volume_shape,
            price_scenario=args.price_path,
            seed=args.seed,
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    bars.to_parquet(args.output, index=False)
    print(json.dumps({"output": str(args.output), "rows": len(bars), "seed": args.seed}, indent=2))
    return 0


def _ml(args: argparse.Namespace) -> int:
    if args.ml_command == "paper":
        return _paper(args)
    from execsim.ml.datasets import (
        DatasetBuildConfig,
        WalkForwardConfig,
        build_dataset,
        create_walk_forward_splits,
        load_dataset_manifest,
    )
    from execsim.ml.datasets.validation import load_dataset_rows, validate_dataset_rows

    if args.ml_command == "build-dataset":
        config = load_config(args.config)
        sources = args.source or [config.processed_symbol_path(symbol) for symbol in config.symbols]
        result = build_dataset(
            output_root=args.output_root,
            source_paths=sources,
            materialize_result_rows=False,
            config=DatasetBuildConfig(
                mode=args.mode,
                bucket_minutes=args.bucket_minutes,
                timezone=config.timezone,
                require_calendar_complete=not args.allow_incomplete_sessions,
            ),
        )
        print(
            json.dumps(
                {
                    "dataset_id": result.manifest.dataset_id,
                    "manifest": str(result.manifest_path),
                    "rows": result.manifest.row_count,
                },
                indent=2,
            )
        )
        return 0
    if args.ml_command in {"validate-dataset", "inspect-dataset", "create-splits"}:
        manifest = load_dataset_manifest(args.manifest)
        rows = load_dataset_rows(manifest, args.manifest.parent)
        if args.ml_command == "validate-dataset":
            errors = validate_dataset_rows(rows, manifest)
            print(json.dumps({"valid": not errors, "errors": errors}, indent=2))
            return 1 if errors else 0
        if args.ml_command == "inspect-dataset":
            print(
                json.dumps(
                    {**manifest.stable_payload(), "manifest_sha256": manifest.manifest_hash()},
                    indent=2,
                    default=str,
                )
            )
            return 0
        split = create_walk_forward_splits(
            rows,
            dataset_id=manifest.dataset_id,
            config=WalkForwardConfig(
                args.initial_train_sessions,
                args.validation_sessions,
                args.test_sessions,
                args.step_sessions,
                args.embargo_sessions,
            ),
        )
        split.write(args.output)
        print(
            json.dumps(
                {"split_id": split.split_id, "folds": len(split.folds), "output": str(args.output)},
                indent=2,
            )
        )
        return 0
    from execsim.ml.training import build_training_plan, run_training

    training_config = _training_config(args.config)
    training_result = (
        build_training_plan(training_config)
        if args.ml_command == "training-plan"
        else run_training(training_config, dry_run=args.dry_run)
    )
    print(json.dumps(_to_dict(training_result), indent=2, default=str))
    return 0


def _paper(args: argparse.Namespace) -> int:
    from execsim.ml.paper.benchmark import build_compute_plan
    from execsim.ml.paper.configs import load_paper_config, load_runtime_approval

    config = load_paper_config(args.config)
    runtime_approval = (
        load_runtime_approval(args.runtime_approval, config)
        if args.runtime_approval is not None
        else None
    )
    operation = None
    if args.paper_command == "download-data":
        operation = ("target_acquisition", args.enable_network)
    elif args.paper_command in {"train-representation", "train-volume-model"}:
        if not args.synthetic_fixture:
            operation = ("historical_training", args.enable_historical_training)
    elif args.paper_command in {
        "evaluate-forecast",
        "evaluate-representation",
        "run-tca",
        "report",
    }:
        if not args.dry_run and not args.synthetic_fixture:
            operation = ("locked_result_evaluation", args.enable_full_paper_run)
    if operation is not None:
        config.authorize(operation[0], approval=runtime_approval, cli_enabled=operation[1])
    if not args.dry_run:
        result = _execute_paper_command(args, config, runtime_approval)
        if result is not None:
            print(json.dumps(result, indent=2, default=str))
            return 1 if result.get("valid") is False else 0
    plan = build_compute_plan(
        network_enabled=config.authorization_granted(
            "target_acquisition",
            approval=runtime_approval,
            cli_enabled=args.enable_network and not args.dry_run,
        ),
        historical_training_enabled=(
            config.authorization_granted(
                "historical_training",
                approval=runtime_approval,
                cli_enabled=args.enable_historical_training and not args.dry_run,
            )
        ),
        full_run_enabled=(
            config.authorization_granted(
                "locked_result_evaluation",
                approval=runtime_approval,
                cli_enabled=args.enable_full_paper_run and not args.dry_run,
            )
        ),
    )
    payload = {
        "command": args.paper_command,
        "mode": "dry-run"
        if args.dry_run
        else "synthetic-fixture"
        if args.synthetic_fixture
        else "authorized",
        "plan": asdict(plan),
        "paper_run_id": config.paper_run_id,
        "artifact_root": str(config.artifact_root),
        "report_root": str(config.report_root),
        "warning": (
            "No network acquisition, historical training, or full paper evaluation was executed."
            if args.dry_run
            else "This planning command reports scope and performs no pipeline stage."
        ),
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


def _lightgbm_execution_options(args: argparse.Namespace) -> LightGBMExecutionOptions:
    """Build the runtime-only LightGBM backend selection for the training command."""
    from execsim.ml.models.lightgbm_adapter import LightGBMExecutionOptions

    return LightGBMExecutionOptions(
        device_type=args.lightgbm_device,
        gpu_platform_id=args.lightgbm_gpu_platform_id,
        gpu_device_id=args.lightgbm_gpu_device_id,
        gpu_use_dp=args.lightgbm_gpu_use_dp,
        num_threads=args.lightgbm_num_threads,
    )


def _execute_paper_command(
    args: argparse.Namespace, config: Any, runtime_approval: Any
) -> dict[str, object] | None:
    """Execute bounded paper operations after the command-level safety checks."""
    if args.paper_command == "run":
        from execsim.ml.paper.orchestration import run_authorized_stages

        return run_authorized_stages(
            config,
            network_cli_enabled=args.enable_network,
            training_cli_enabled=args.enable_historical_training,
            full_run_cli_enabled=args.enable_full_paper_run,
            runtime_approval=runtime_approval,
            trusted_local_resume=args.trust_local_resume,
        )
    if args.paper_command == "download-data":
        from execsim.ml.paper.orchestration import download_data_stage

        return download_data_stage(
            config,
            cli_enabled=args.enable_network,
            runtime_approval=runtime_approval,
        )
    if args.paper_command == "build-universe":
        from execsim.ml.paper.orchestration import build_universe_stage

        return build_universe_stage(config)
    if args.paper_command == "validate-data":
        from execsim.ml.paper.orchestration import validate_data_stage

        return validate_data_stage(config, args.input)
    if args.paper_command == "build-sequences":
        from execsim.ml.paper.orchestration import build_sequences_stage

        return build_sequences_stage(config, args.input)
    if args.paper_command == "validate-sequences":
        from execsim.ml.paper.orchestration import validate_sequences_stage

        return validate_sequences_stage(config)
    if args.paper_command == "train-representation" and args.synthetic_fixture:
        from execsim.ml.representations.schemas import RepresentationConfig
        from execsim.ml.representations.trainer import train_synthetic_fixture

        dense = train_synthetic_fixture(RepresentationConfig("dense"), steps=2, batch_size=4)
        sparse = train_synthetic_fixture(RepresentationConfig("sparse"), steps=2, batch_size=4)
        return {
            "data_classification": "synthetic_fixture",
            "dense": asdict(dense),
            "sparse": asdict(sparse),
        }
    if args.paper_command == "train-representation":
        from execsim.ml.paper.orchestration import train_representations_stage

        return train_representations_stage(
            config,
            training_cli_enabled=args.enable_historical_training,
            runtime_approval=runtime_approval,
            trusted_local_resume=args.trust_local_resume,
        )
    if args.paper_command == "train-volume-model" and args.synthetic_fixture:
        import numpy as np

        from execsim.ml.models.lightgbm_adapter import (
            LightGBMConfig,
            LightGBMVolumeModel,
        )

        rng = np.random.default_rng(13)
        features = rng.normal(size=(32, 12))
        total = np.exp(10 + features[:, 0])
        shape = np.exp(features[:, :4])
        shape /= shape.sum(axis=1, keepdims=True)
        execution = _lightgbm_execution_options(args)
        model = LightGBMVolumeModel(
            LightGBMConfig(n_estimators=8, min_child_samples=2),
            execution=execution,
        ).fit(features, total, shape)
        predicted_total, predicted_shape = model.predict(features[:2])
        return {
            "data_classification": "synthetic_fixture",
            "lightgbm_execution": execution.identity(),
            "remaining_volume": predicted_total.tolist(),
            "shape_row_sums": predicted_shape.sum(axis=1).tolist(),
        }
    if args.paper_command == "train-volume-model":
        from execsim.ml.paper.orchestration import train_volume_models_stage

        return train_volume_models_stage(
            config,
            training_cli_enabled=args.enable_historical_training,
            runtime_approval=runtime_approval,
            execution=_lightgbm_execution_options(args),
        )
    if args.paper_command == "export-embeddings" and args.synthetic_fixture:
        import hashlib

        import numpy as np
        import torch

        from execsim.ml.representations.embeddings import (
            export_frozen_embedding,
            write_embedding_artifact_manifest,
            write_embedding_parquet,
        )
        from execsim.ml.representations.jepa import PredictiveRepresentationModel
        from execsim.ml.representations.schemas import RepresentationConfig

        torch.manual_seed(13)
        representation_model = PredictiveRepresentationModel(RepresentationConfig("sparse"))
        context = torch.zeros((1, 8, 18))
        mask = torch.ones((1, 8), dtype=torch.bool)
        embedding = export_frozen_embedding(representation_model, context, mask)
        checkpoint_hash = hashlib.sha256(b"synthetic-sparse-checkpoint").hexdigest()
        sequence_hash = hashlib.sha256(b"synthetic-sequence").hexdigest()
        payload: dict[str, object] = {
            "data_classification": "synthetic_fixture",
            "shape": list(embedding.shape),
            "finite": bool(np.isfinite(embedding).all()),
        }
        if args.output is not None:
            parquet_path = args.output / "synthetic-embedding.parquet"
            write_embedding_parquet(
                parquet_path,
                embedding=embedding,
                metadata={
                    "instrument_id": "synthetic-asset",
                    "symbol": "SYNTH",
                    "session_date": "2024-01-03",
                    "as_of": "2024-01-03T10:30:00-05:00",
                    "fold_id": "fold-1",
                    "seed": 13,
                    "geometry": "sparse",
                    "adaptation": "none",
                    "predictor_family": "mlp",
                    "checkpoint_hash": checkpoint_hash,
                    "sequence_hash": sequence_hash,
                    "cutoff": "2023-12-29",
                    "component_order": "current,h1,h2,h4,h8,horizon_availability",
                },
            )
            manifest = write_embedding_artifact_manifest(
                args.output / "synthetic-embedding-manifest.json",
                artifact_id="synthetic-embedding",
                checkpoint_hash=checkpoint_hash,
                partition_identity="fold-1/test",
                row_count=1,
                training_cutoff="2023-12-29",
                source_hashes=(sequence_hash,),
                parquet_path=parquet_path,
            )
            payload.update(
                {
                    "parquet": str(parquet_path),
                    "manifest": str(args.output / "synthetic-embedding-manifest.json"),
                    "parquet_sha256": manifest.parquet_sha256,
                }
            )
        return payload
    if args.paper_command == "export-embeddings":
        from execsim.ml.paper.orchestration import export_embeddings_stage

        return export_embeddings_stage(config)
    if args.paper_command == "evaluate-representation" and args.synthetic_fixture:
        import numpy as np

        from execsim.ml.paper.benchmark import predictor_capacity_smoke
        from execsim.ml.representations.diagnostics import representation_diagnostics

        rng = np.random.default_rng(13)
        dense_latents = rng.normal(size=(32, 128))
        sparse_latents = np.maximum(rng.laplace(-0.49012907173427356, 2**-0.5, (32, 128)), 0)
        return {
            "data_classification": "synthetic_fixture",
            "dense": representation_diagnostics(dense_latents),
            "sparse": representation_diagnostics(sparse_latents),
            "predictor_capacity": predictor_capacity_smoke(repetitions=1),
        }
    if args.paper_command == "evaluate-representation":
        from execsim.ml.paper.orchestration import evaluate_representations_stage

        return evaluate_representations_stage(
            config,
            full_run_cli_enabled=args.enable_full_paper_run,
            runtime_approval=runtime_approval,
        )
    if args.paper_command == "evaluate-forecast" and args.synthetic_fixture:
        import numpy as np

        from execsim.ml.paper.statistics import paper_forecast_metrics

        actual_total = np.asarray([100.0, 150.0, 220.0])
        predicted_total = np.asarray([105.0, 143.0, 210.0])
        actual_shape = np.asarray([[0.4, 0.6], [0.5, 0.5], [0.3, 0.7]])
        predicted_shape = np.asarray([[0.42, 0.58], [0.48, 0.52], [0.35, 0.65]])
        return {
            "data_classification": "synthetic_fixture",
            "metrics": paper_forecast_metrics(
                actual_total, predicted_total, actual_shape, predicted_shape
            ),
        }
    if args.paper_command == "evaluate-forecast":
        from execsim.ml.paper.orchestration import evaluate_forecasts_stage

        return evaluate_forecasts_stage(
            config,
            full_run_cli_enabled=args.enable_full_paper_run,
            runtime_approval=runtime_approval,
        )
    if args.paper_command == "run-tca" and args.synthetic_fixture:
        from datetime import date, time

        import numpy as np
        import pandas as pd

        from execsim.ml.paper.tca import expand_volume_forecast
        from execsim.orders import ParentOrder
        from execsim.policies import AdaptiveMPCPolicy, ExecutionConstraints
        from execsim.simulator import simulate_policy

        timestamps = pd.date_range(
            "2024-01-03 10:30", periods=15, freq="min", tz="America/New_York"
        )
        bars = pd.DataFrame(
            {
                "symbol": "SYNTH",
                "timestamp": timestamps,
                "open": 100.0,
                "high": 100.1,
                "low": 99.9,
                "close": 100.0,
                "volume": 1_000,
                "trade_count": 100,
                "vwap": 100.0,
            }
        )

        class SyntheticPaperForecaster:
            provider_id = "paper-synthetic"

            def forecast(
                self, *, symbol, session_date, generated_at, bucket_timestamps, observations=None
            ):
                del observations
                buckets = tuple(bucket_timestamps)
                return expand_volume_forecast(
                    symbol=symbol,
                    session_date=session_date,
                    generated_at=generated_at,
                    minute_timestamps=buckets,
                    expected_remaining_volume=1_000.0 * len(buckets),
                    conditional_token_shape=np.asarray([1.0]),
                    within_token_profile=np.full(15, 1 / 15),
                    training_cutoff=date(2023, 12, 29),
                    manifest_hash="a" * 64,
                    forecaster_id=self.provider_id,
                )

        result = simulate_policy(
            parent_order=ParentOrder(
                "SYNTH", "buy", 150, date(2024, 1, 3), time(10, 30), time(10, 45)
            ),
            bars=bars,
            policy=AdaptiveMPCPolicy(temporary_impact=0.001, risk_aversion=0.0),
            constraints=ExecutionConstraints(0.1, 0.1),
            forecast_provider=SyntheticPaperForecaster(),
        )
        return {
            "data_classification": "synthetic_fixture",
            "requested_qty": result.summary.requested_qty,
            "filled_qty": result.summary.filled_qty,
            "optimization_decisions": result.summary.n_optimization_decisions,
            "implementation_shortfall_bps": result.summary.implementation_shortfall_bps,
        }
    if args.paper_command == "run-tca":
        from execsim.ml.paper.orchestration import run_tca_stage

        return run_tca_stage(
            config,
            args.input,
            full_run_cli_enabled=args.enable_full_paper_run,
            runtime_approval=runtime_approval,
        )
    if args.paper_command == "report" and args.synthetic_fixture:
        import pandas as pd

        from execsim.ml.paper.reports import TABLE_NAMES, write_paper_bundle

        tables = {name: pd.DataFrame({"metric": [name], "value": [1.0]}) for name in TABLE_NAMES}
        output = write_paper_bundle(
            args.output or config.report_root,
            paper_run_id=config.paper_run_id + "-synthetic",
            tables=tables,
            provenance={
                "data_classification": "synthetic_fixture",
                "network_acquisition": "NOT RUN",
                "historical_training": "NOT RUN",
            },
        )
        return {"data_classification": "synthetic_fixture", "output": str(output)}
    if args.paper_command == "report":
        from execsim.ml.paper.orchestration import report_stage

        return report_stage(
            config,
            full_run_cli_enabled=args.enable_full_paper_run,
            runtime_approval=runtime_approval,
        )
    if args.paper_command in {"plan", "plan-representation"}:
        from execsim.ml.paper.benchmark import (
            build_compute_plan,
            predictor_capacity_smoke,
            profile_paper_kernels,
        )

        planning_payload: dict[str, object] = {
            "plan": asdict(
                build_compute_plan(
                    network_enabled=config.authorization_granted(
                        "target_acquisition",
                        approval=runtime_approval,
                        cli_enabled=args.enable_network,
                    ),
                    historical_training_enabled=config.authorization_granted(
                        "historical_training",
                        approval=runtime_approval,
                        cli_enabled=args.enable_historical_training,
                    ),
                    full_run_enabled=config.authorization_granted(
                        "locked_result_evaluation",
                        approval=runtime_approval,
                        cli_enabled=args.enable_full_paper_run,
                    ),
                )
            ),
            "measured_profile": asdict(profile_paper_kernels(args.input)),
            "evidence_boundary": "bounded local profile; no historical fit or acquisition",
        }
        if args.paper_command == "plan-representation":
            planning_payload["capacity_kernel_profile"] = predictor_capacity_smoke()
        return planning_payload
    raise ValueError(
        f"{args.paper_command} requires --dry-run, --synthetic-fixture, "
        "or supplied artifact inputs."
    )


def _required_path(value: Path | None, option: str) -> Path:
    if value is None:
        raise ValueError(f"This paper command requires {option}.")
    return value


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _training_config(path: Path) -> Any:
    from execsim.ml.training import TrainingConfig

    values = _read_yaml_mapping(path)
    return TrainingConfig(
        dataset_manifest_path=Path(values["dataset_manifest_path"]),
        split_manifest_path=Path(values["split_manifest_path"]),
        feature_names=tuple(values["feature_names"]),
        target_name=str(values["target_name"]),
        model_family=str(values.get("model_family", "ridge")),
        hyperparameter_grid=tuple(values.get("hyperparameter_grid", [{"alpha": 1.0}])),
        artifact_root=Path(values.get("artifact_root", "artifacts/models")),
        random_seed=int(values.get("random_seed", 17)),
        allow_historical_training=bool(values.get("allow_historical_training", False)),
    )


def _validate_market_data(config: ExecSimConfig) -> int:
    from execsim.data.loaders import load_processed_symbol_bars
    from execsim.data.validation import validate_processed_bars

    reports = [
        validate_processed_bars(load_processed_symbol_bars(config, symbol), symbol=symbol)
        for symbol in config.symbols
    ]
    for report in reports:
        print("\n".join(report.to_lines()))
    return 0 if all(report.is_valid for report in reports) else 1


def _add_config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=None)


def _add_simulation_arguments(parser: argparse.ArgumentParser, *, include_strategy: bool) -> None:
    _add_config_argument(parser)
    if include_strategy:
        parser.add_argument("--strategy", choices=STRATEGIES, default="twap")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--trade-date", default=None)
    parser.add_argument("--side", choices=("buy", "sell"), default=None)
    parser.add_argument("--quantity", type=int, default=None)
    parser.add_argument("--start-time", default=None)
    parser.add_argument("--end-time", default=None)
    parser.add_argument("--max-bar-participation-rate", type=float, default=None)
    parser.add_argument("--pov-target-rate", type=float, default=0.05)
    parser.add_argument("--half-spread", type=float, default=0.01)
    parser.add_argument("--temporary-impact", type=float, default=0.1)
    parser.add_argument("--volatility", type=float, default=0.01)
    parser.add_argument("--risk-aversion", type=float, default=0.0)
    parser.add_argument("--lookback-sessions", type=int, default=20)
    parser.add_argument("--json", action="store_true")


def _build_parent_order_from_args(
    args: argparse.Namespace, config: ExecSimConfig
) -> tuple[ParentOrder, float]:
    defaults = config.demo_twap
    order = ParentOrder(
        symbol=args.symbol or defaults.symbol,
        side=cast(OrderSide, args.side or defaults.side),
        quantity=args.quantity if args.quantity is not None else defaults.quantity,
        trade_date=_parse_date(args.trade_date) if args.trade_date else defaults.trade_date,
        start_time=_parse_time(args.start_time) if args.start_time else defaults.start_time,
        end_time=_parse_time(args.end_time) if args.end_time else defaults.end_time,
    )
    rate = (
        args.max_bar_participation_rate
        if args.max_bar_participation_rate is not None
        else defaults.max_bar_participation_rate
    )
    return order, rate


def _parse_date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid date {value!r}; expected YYYY-MM-DD.") from exc


def _parse_time(value: str) -> time:
    try:
        parsed = time.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"Invalid time {value!r}; expected HH:MM.") from exc
    if parsed.tzinfo is not None:
        raise ValueError("Execution times must be timezone-naive.")
    return parsed


def _print_simulation_result(result: Any, *, as_json: bool) -> None:
    if as_json:
        print(
            json.dumps(
                {
                    "summary": asdict(result.summary),
                    "execution_log_head": result.execution_log.head().to_dict(orient="records"),
                },
                indent=2,
                default=str,
            )
        )
        return
    summary = result.summary
    print(
        f"{summary.strategy}: {summary.symbol} {summary.side} "
        f"requested={summary.requested_qty} filled={summary.filled_qty} "
        f"completion={summary.completion_rate:.4f}"
    )
    print(
        f"implementation_shortfall_bps={summary.implementation_shortfall_bps} "
        f"total_modeled_cost={summary.total_modeled_execution_cost:.6f}"
    )


def _read_yaml_mapping(path: str | Path) -> Mapping[str, Any]:
    resolved = Path(path)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, Mapping):
        raise TypeError(f"YAML must contain a mapping: {resolved}")
    return payload


def _to_dict(value: object) -> dict[str, object]:
    return asdict(cast(Any, value)) if is_dataclass(value) else {"result": str(value)}


if __name__ == "__main__":
    raise SystemExit(main())
