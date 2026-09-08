"""Fold-isolated RunPod transfer, qualification, lifecycle, and integrity commands."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import importlib.metadata
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from execsim.data.paper.manifests import file_sha256, read_json, write_json_atomic
from execsim.ml.paper.configs import load_paper_config

FOLDS = ("fold-1", "fold-2", "fold-3")
COORDINATES = [("raw", None), ("untrained_neural", None)] + [
    (geometry, seed) for geometry in ("dense", "sparse") for seed in (13, 29, 47)
]


def source_identity(repo: Path) -> dict:
    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    if git("status", "--porcelain", "--untracked-files=no"):
        raise ValueError("Execution requires a clean committed source tree.")
    return {
        "source_commit": git("rev-parse", "HEAD"),
        "source_tree": git("rev-parse", "HEAD^{tree}"),
    }


def safe_path(root: Path, name: str) -> Path:
    pure = PurePosixPath(name)
    if pure.is_absolute() or ".." in pure.parts or "\\" in name or ":" in name:
        raise ValueError(f"Unsafe transfer member: {name}")
    path = root.joinpath(*pure.parts)
    path.resolve().relative_to(root.resolve())
    if path.is_symlink():
        raise ValueError("Transfer members cannot be symbolic links.")
    return path


def verify_bundle(root: Path, expected_hash: str | None = None) -> dict:
    manifest = root / "transfer.json"
    if expected_hash is not None and file_sha256(manifest) != expected_hash:
        raise ValueError("Transfer manifest SHA-256 mismatch.")
    payload = read_json(manifest)
    if payload["fold_id"] not in FOLDS or payload["schema"] != "runpod-transfer-v1":
        raise ValueError("Unknown transfer identity.")
    names = [item["path"] for item in payload["files"]]
    if not names or len(names) != len(set(names)):
        raise ValueError("Empty or duplicated transfer members.")
    for item in payload["files"]:
        path = safe_path(root, item["path"])
        if path.stat().st_size != item["bytes"] or file_sha256(path) != item["sha256"]:
            raise ValueError(f"Transfer checksum mismatch: {item['path']}")
    return payload


def package_inputs(args) -> dict:
    """Copy only the requested fold's TRAIN/VALIDATION inputs; leave upstream bytes intact."""
    config = load_paper_config(args.repo / "configs/paper/sparse_jepa_v2")
    identity = source_identity(args.repo)
    root = args.artifacts or args.repo / config.artifact_root
    data_root = args.data_root or args.repo / "data"
    fold = args.fold
    members: dict[str, Path] = {}

    def add(path, destination, expected=None):
        if expected is not None and file_sha256(path) != expected:
            raise ValueError(f"Upstream checksum mismatch: {path}")
        members[destination] = path

    sequence = root / "sequences" / fold / "sequence-manifest.json"
    seq = read_json(sequence)
    if seq["fold_id"] != fold or seq["config_hash"] != config.config_hash:
        raise ValueError("Sequence protocol/fold mismatch.")
    add(sequence, f"sequences/{fold}/sequence-manifest.json")
    import pandas as pd

    sequence_hashes = {}
    for name in seq["index_files"]:
        if any(f"/{p}/" in name.replace("\\", "/") for p in ("train", "validation")):
            index = pd.read_parquet(
                safe_path(sequence.parent, name), columns=["session_id", "source_sequence_hash"]
            )
            for row in index.drop_duplicates().itertuples(index=False):
                if (
                    row.session_id in sequence_hashes
                    and sequence_hashes[row.session_id] != row.source_sequence_hash
                ):
                    raise ValueError("Sequence index hashes disagree.")
                sequence_hashes[row.session_id] = row.source_sequence_hash
    for name in seq["sequence_files"] + seq["index_files"]:
        if any(
            f"/{partition}/" in name.replace("\\", "/") for partition in ("train", "validation")
        ):
            expected = (
                sequence_hashes.get(Path(name).stem) if name in seq["sequence_files"] else None
            )
            if name in seq["sequence_files"] and expected is None:
                raise ValueError("A session lacks an authoritative indexed sequence hash.")
            add(safe_path(sequence.parent, name), f"sequences/{fold}/{name}", expected)
    universe = data_root / "manifests/paper_universe_v2.json"
    add(universe, "data/manifests/paper_universe_v2.json", seq["universe_manifest_hash"])
    upstream_commits = set()
    for geometry in ("dense", "sparse"):
        for seed in (13, 29, 47):
            relative = f"{fold}/{geometry}/{seed}"
            em = root / "embeddings" / relative / "manifest.json"
            payload = read_json(em)
            checkpoint = root / "representations" / relative / "final/manifest.json"
            cp = read_json(checkpoint)
            if cp["calibrated_rdm_lambda"] != 10.0:
                raise ValueError("Upstream lambda is not the frozen selection.")
            if (
                payload["fold_id"],
                payload["geometry"],
                payload["seed"],
                payload["paper_config_hash"],
                payload["sequence_manifest_hash"],
            ) != (fold, geometry, seed, config.config_hash, file_sha256(sequence)) or payload[
                "checkpoint_manifest_hash"
            ] != file_sha256(checkpoint):
                raise ValueError("Embedding compatibility mismatch.")
            upstream_commits.add(cp["code_commit"])
            add(checkpoint, f"representations/{relative}/final/manifest.json")
            add(em, f"embeddings/{relative}/manifest.json")
            for item in payload["files"]:
                if item["partition"] in {"train", "validation"}:
                    add(
                        safe_path(em.parent, item["path"]),
                        f"embeddings/{relative}/{item['path']}",
                        item["sha256"],
                    )
    if len(upstream_commits) != 1:
        raise ValueError("Upstream representation commits disagree.")
    args.output.mkdir(parents=True, exist_ok=False)
    files = []
    for name, path in sorted(members.items()):
        target = safe_path(args.output, name)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        digest = file_sha256(path)
        if file_sha256(target) != digest:
            raise ValueError(f"Copy verification failed: {name}")
        files.append({"path": name, "sha256": digest, "bytes": target.stat().st_size})
    receipt = {
        "schema": "runpod-transfer-v1",
        "kind": "inputs",
        "fold_id": fold,
        "paper_config_hash": config.config_hash,
        **identity,
        "upstream_source_commit": upstream_commits.pop(),
        "files": files,
    }
    write_json_atomic(args.output / "transfer.json", receipt)
    return {
        "status": "PASS",
        "manifest_sha256": file_sha256(args.output / "transfer.json"),
        "fold_id": fold,
        "files": len(files),
    }


def discover() -> dict:
    """Enumerate OpenCL GPU IDs using the installed ICD, then require exactly one RTX 3090."""
    library = ctypes.util.find_library("OpenCL") or (
        "OpenCL.dll" if os.name == "nt" else "libOpenCL.so.1"
    )
    cl = ctypes.CDLL(library)
    count = ctypes.c_uint()
    if cl.clGetPlatformIDs(0, None, ctypes.byref(count)) != 0:
        raise RuntimeError("OpenCL platforms unavailable.")
    platforms = (ctypes.c_void_p * count.value)()
    cl.clGetPlatformIDs(count, platforms, None)

    def info(function, handle, field):
        size = ctypes.c_size_t()
        if function(ctypes.c_void_p(handle), field, 0, None, ctypes.byref(size)) != 0:
            raise RuntimeError("OpenCL info query failed.")
        buffer = ctypes.create_string_buffer(size.value)
        function(ctypes.c_void_p(handle), field, size, buffer, None)
        return buffer.value.decode()

    matches = []
    for pi, handle in enumerate(platforms):
        devices_count = ctypes.c_uint()
        if (
            cl.clGetDeviceIDs(
                ctypes.c_void_p(handle), ctypes.c_ulong(4), 0, None, ctypes.byref(devices_count)
            )
            != 0
        ):
            continue
        devices = (ctypes.c_void_p * devices_count.value)()
        cl.clGetDeviceIDs(ctypes.c_void_p(handle), ctypes.c_ulong(4), devices_count, devices, None)
        for di, device in enumerate(devices):
            name = info(cl.clGetDeviceInfo, device, 0x102B)
            if name in {"NVIDIA GeForce RTX 3090", "NVIDIA RTX 3090"}:
                matches.append(
                    {
                        "platform_id": pi,
                        "device_id": di,
                        "device": name,
                        "platform": info(cl.clGetPlatformInfo, handle, 0x0902),
                        "driver": info(cl.clGetDeviceInfo, device, 0x102D),
                    }
                )
    if len(matches) != 1:
        raise RuntimeError(f"Require exactly one RTX 3090 OpenCL GPU; found {len(matches)}.")
    return matches[0]


def hardware() -> dict:
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    memory = Path("/proc/meminfo")
    cpu_file = Path("/proc/cpuinfo")
    cpu_name = platform.processor()
    if cpu_file.exists():
        cpu_name = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpu_file.read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu_name,
        )
    return {
        "os": platform.platform(),
        "cpu": cpu_name,
        "cpu_count": os.cpu_count(),
        "ram": memory.read_text().splitlines()[0] if memory.exists() else "see host inventory",
        "gpu": gpu,
        "hostname": platform.node(),
        "pod_id": os.environ.get("RUNPOD_POD_ID", platform.node()),
    }


def qualify(args) -> dict:
    import numpy as np
    import pandas as pd

    from execsim.ml.models.lightgbm_adapter import (
        LightGBMConfig,
        LightGBMExecutionOptions,
        LightGBMVolumeModel,
    )

    config = load_paper_config(args.repo / "configs/paper/sparse_jepa_v2")
    source = source_identity(args.repo)
    if importlib.metadata.version("lightgbm") != "4.7.0":
        raise ValueError("Qualification requires LightGBM 4.7.0.")
    device = discover()
    execution = LightGBMExecutionOptions(
        device_type="gpu",
        gpu_platform_id=device["platform_id"],
        gpu_device_id=device["device_id"],
        num_threads=args.threads,
    )
    rng = np.random.default_rng(917)
    x = rng.normal(size=2400)
    scale = pd.DataFrame(
        {
            "case_id": np.arange(2400),
            "symbol": np.tile(["AAA", "BBB"], 1200),
            "x": x,
            "baseline_remaining_volume": np.exp(8 + 0.15 * x),
        }
    )
    total = scale.baseline_remaining_volume.to_numpy() * np.exp(
        0.12 * x + rng.normal(0, 0.03, 2400)
    )
    shape = scale.loc[scale.index.repeat(5)].reset_index(drop=True)
    shape["target_bucket"] = np.tile(np.arange(5), 2400)
    shape["sample_weight"] = 0.75 + 0.5 * rng.random(len(shape))
    logits = (
        0.08 * shape.target_bucket.to_numpy()
        + 0.04 * shape.x.to_numpy() * shape.target_bucket.to_numpy()
    ).reshape(-1, 5)
    shares = np.exp(logits - logits.max(axis=1, keepdims=True))
    shares = (shares / shares.sum(axis=1, keepdims=True)).ravel()
    training = (scale.iloc[:1800], total[:1800], shape.iloc[:9000], shares[:9000])
    valid = (
        scale.iloc[1800:].reset_index(drop=True),
        total[1800:],
        shape.iloc[9000:].reset_index(drop=True),
        shares[9000:],
    )
    results = []
    args.work.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.work) as temporary:
        for repeat in range(2):
            model = LightGBMVolumeModel(
                LightGBMConfig(n_estimators=160, early_stopping_rounds=15), execution=execution
            )
            model.fit_frames(*training, validation=valid)
            predicted, curve = model.predict_frames(valid[0], valid[2], group_columns=("case_id",))
            values = curve.conditional_share.to_numpy()
            if not np.isfinite(predicted).all() or not np.isfinite(values).all():
                raise ValueError("Qualification produced non-finite predictions.")
            path = Path(temporary) / str(repeat)
            model.save_native(
                path,
                {
                    "fold_id": args.fold,
                    "feature_schema_version": "qualification",
                    "training_cutoff": "synthetic",
                    "validation_range": ["synthetic"],
                    "categorical_features": ["symbol"],
                },
            )
            restored, _ = LightGBMVolumeModel.load_native(path, expected_execution=execution)
            a, b = restored.predict_frames(valid[0], valid[2], group_columns=("case_id",))
            np.testing.assert_allclose(a, predicted, rtol=1e-12, atol=1e-12)
            np.testing.assert_allclose(b.conditional_share, values, rtol=1e-12, atol=1e-12)
            results.append((predicted, values))
    for index in (0, 1):
        np.testing.assert_allclose(results[0][index], results[1][index], rtol=0.005, atol=1e-6)
    from lightgbm.basic import _LIB

    build = {
        "status": "PASS",
        "lightgbm_version": "4.7.0",
        "library_sha256": file_sha256(Path(_LIB._name)),
        "backend": "OpenCL",
    }
    receipt = {
        "status": "PASS",
        **source,
        "paper_config_hash": config.config_hash,
        "fold_id": args.fold,
        "hardware": hardware(),
        "opencl": device,
        "lightgbm_version": "4.7.0",
        "build": build,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("lightgbm", "numpy", "pandas", "pyarrow", "torch")
        },
        "execution": {
            "device_type": "gpu",
            "gpu_platform_id": device["platform_id"],
            "gpu_device_id": device["device_id"],
            "gpu_use_dp": False,
            "selected_num_threads": args.threads,
        },
        "repeat_tolerance": {"rtol": 0.005, "atol": 1e-6},
        "evidence": "synthetic only",
    }
    write_json_atomic(args.work / "lightgbm-gpu/build-provenance.json", build)
    write_json_atomic(args.work / "lightgbm-gpu/qualification-receipt.json", receipt)
    return receipt


def ready(args) -> dict:
    receipts = [read_json(path) for path in args.receipts]
    source = source_identity(args.repo)
    config = load_paper_config(args.repo / "configs/paper/sparse_jepa_v2")
    if len(receipts) != 3 or {r["fold_id"] for r in receipts} != set(FOLDS):
        raise ValueError("Launch requires three distinct qualified folds.")
    if len({r["hardware"]["pod_id"] for r in receipts}) != 3:
        raise ValueError("Launch requires three distinct pods.")
    for receipt in receipts:
        if (
            receipt["status"] != "PASS"
            or receipt["source_commit"] != source["source_commit"]
            or receipt["source_tree"] != source["source_tree"]
            or receipt["paper_config_hash"] != config.config_hash
            or receipt["execution"]["device_type"] != "gpu"
            or receipt["execution"]["gpu_use_dp"]
            or receipt["lightgbm_version"] != "4.7.0"
            or receipt["opencl"]["device"] not in {"NVIDIA GeForce RTX 3090", "NVIDIA RTX 3090"}
        ):
            raise ValueError("Incompatible qualification receipt.")
    payload = {
        "status": "THREE_PODS_QUALIFIED",
        **source,
        "paper_config_hash": config.config_hash,
        "qualifications": {
            r["fold_id"]: file_sha256(p) for r, p in zip(receipts, args.receipts, strict=True)
        },
    }
    write_json_atomic(args.output, payload)
    return payload


def train(args) -> dict:
    from execsim.ml.models.lightgbm_adapter import LightGBMExecutionOptions
    from execsim.ml.paper.configs import load_runtime_approval
    from execsim.ml.paper.orchestration import train_volume_models_stage

    receipt = read_json(args.work / "lightgbm-gpu/qualification-receipt.json")
    gate = read_json(args.ready)
    bundle = verify_bundle(args.bundle, args.bundle_sha256)
    config = load_paper_config(args.repo / "configs/paper/sparse_jepa_v2").with_runtime_roots(
        artifact_root=args.bundle,
        data_root=args.bundle / "data",
        cache_root=args.work,
        sequence_root=args.bundle / "sequences",
        embedding_root=args.bundle / "embeddings",
        output_root=args.output,
    )
    identity = source_identity(args.repo)
    if any(bundle[k] != identity[k] or gate[k] != identity[k] for k in identity):
        raise ValueError("Transfer/launch source identity mismatch.")
    if (
        bundle["fold_id"] != args.fold
        or receipt["fold_id"] != args.fold
        or bundle["paper_config_hash"] != config.config_hash
        or gate["qualifications"].get(args.fold)
        != file_sha256(args.work / "lightgbm-gpu/qualification-receipt.json")
        or receipt["opencl"] != discover()
        or receipt["hardware"] != hardware()
    ):
        raise ValueError("Pod, input, or qualification identity mismatch.")
    approval = load_runtime_approval(args.approval, config)
    if approval.approves("locked_result_evaluation"):
        raise ValueError("Pod approval must keep TEST closed.")
    options = dict(receipt["execution"])
    from lightgbm.basic import _LIB

    if file_sha256(Path(_LIB._name)) != receipt["build"]["library_sha256"]:
        raise ValueError("Installed LightGBM library changed after qualification.")
    options["num_threads"] = options.pop("selected_num_threads")
    return train_volume_models_stage(
        config,
        training_cli_enabled=True,
        runtime_approval=approval,
        execution=LightGBMExecutionOptions(**options),
        fold_id=args.fold,
        input_identity={
            "transfer_sha256": file_sha256(args.bundle / "transfer.json"),
            "upstream_source_commit": bundle["upstream_source_commit"],
        },
    )


def process_token(pid: int) -> str | None:
    """Bind stop requests to a Linux PID start time, protecting against PID reuse."""
    path = Path(f"/proc/{pid}/stat")
    try:
        fields = path.read_text().rsplit(")", 1)[1].split()
        return None if fields[0] == "Z" else fields[19]
    except FileNotFoundError:
        return None


def audit_fold(root: Path, fold: str, source: dict, config_hash: str) -> dict:
    from execsim.ml.models.lightgbm_adapter import (
        LightGBMConfig,
        LightGBMVolumeModel,
        _validate_candidate_grid,
    )

    receipt = read_json(root / fold / "execution-receipt.json")
    if (
        receipt["git_commit"] != source["source_commit"]
        or receipt["paper_config_hash"] != config_hash
        or receipt["fold_id"] != fold
    ):
        raise ValueError("Fold execution receipt identity mismatch.")
    found = sorted(
        p.relative_to(root / fold).as_posix() for p in (root / fold).glob("*/*/manifest.json")
    )
    expected = sorted(f"{m}/{s or 'shared'}/manifest.json" for m, s in COORDINATES)
    if found != expected:
        raise ValueError(f"Fold {fold} is incomplete or has extra coordinates: {len(found)}/8")
    for method, seed in COORDINATES:
        path = root / fold / method / str(seed or "shared")
        _, manifest = LightGBMVolumeModel.load_native(path)
        execution = manifest["lightgbm_execution"]
        if (
            manifest["fold_id"] != fold
            or manifest["method"] != method
            or manifest["seed"] != seed
            or manifest["git_commit"] != source["source_commit"]
            or manifest["paper_config_hash"] != config_hash
            or manifest["lightgbm_version"] != "4.7.0"
            or execution["device_type"] != "gpu"
            or execution["gpu_use_dp"]
            or execution != receipt["lightgbm_execution"]
            or manifest["execution_receipt_sha256"]
            != file_sha256(root / fold / "execution-receipt.json")
            or manifest["grid_results_sha256"] != file_sha256(path / "grid-results.json")
        ):
            raise ValueError(f"Coordinate compatibility mismatch: {fold}/{method}/{seed}")
        grid = read_json(path / "grid-results.json")
        _validate_candidate_grid(tuple(LightGBMConfig(**r["config"]) for r in grid["candidates"]))
        if any(row["config"]["seed"] != (seed or 13) for row in grid["candidates"]):
            raise ValueError("Candidate seeds disagree with the coordinate.")
        import math

        if any(
            not math.isfinite(float(row[key]))
            for row in grid["candidates"]
            for key in ("scale_mae", "shape_error")
        ):
            raise ValueError("Candidate metrics must be finite.")
        for role, metric in (("scale", "scale_mae"), ("shape", "shape_error")):
            selected = min(grid["candidates"], key=lambda row: row[metric])["config"]
            if (
                selected != manifest[f"selected_{role}_config"]
                or selected != grid[f"selected_{role}_config"]
            ):
                raise ValueError("Selected configuration does not match the frozen selection rule.")
        if grid["selection_data"] != "validation_only":
            raise ValueError("Invalid model selection population.")
    return {"fold_id": fold, "coordinates": 8, "execution": receipt}


def export_fold(args) -> dict:
    identity = source_identity(args.repo)
    config = load_paper_config(args.repo / "configs/paper/sparse_jepa_v2")
    audit_fold(args.models, args.fold, identity, config.config_hash)
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copytree(
        args.models / args.fold, args.output / args.fold, ignore=shutil.ignore_patterns(".*")
    )
    files = [
        {
            "path": p.relative_to(args.output).as_posix(),
            "bytes": p.stat().st_size,
            "sha256": file_sha256(p),
        }
        for p in sorted(args.output.rglob("*"))
        if p.is_file()
    ]
    payload = {
        "schema": "runpod-transfer-v1",
        "kind": "models",
        "fold_id": args.fold,
        **identity,
        "paper_config_hash": config.config_hash,
        "files": files,
    }
    write_json_atomic(args.output / "transfer.json", payload)
    return {"status": "PASS", "manifest_sha256": file_sha256(args.output / "transfer.json")}


def reassemble(args) -> dict:
    source = source_identity(args.repo)
    config = load_paper_config(args.repo / "configs/paper/sparse_jepa_v2")
    bundles = [
        verify_bundle(path, digest)
        for path, digest in zip(args.inputs, args.input_sha256, strict=True)
    ]
    if len(bundles) != 3 or {p["fold_id"] for p in bundles} != set(FOLDS):
        raise ValueError("Reassembly requires exactly three distinct folds.")
    audits = []
    for path, payload in zip(args.inputs, bundles, strict=True):
        if (
            payload["kind"] != "models"
            or payload["source_commit"] != source["source_commit"]
            or payload["paper_config_hash"] != config.config_hash
        ):
            raise ValueError("Reassembly source/config mismatch.")
        audits.append(audit_fold(path, payload["fold_id"], source, config.config_hash))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise ValueError("Reassembly destination must be new; existing evidence is preserved.")
    with tempfile.TemporaryDirectory(dir=args.output.parent) as temporary:
        target = Path(temporary) / "complete"
        target.mkdir()
        for path, payload in zip(args.inputs, bundles, strict=True):
            shutil.copytree(path / payload["fold_id"], target / payload["fold_id"])
            audit_fold(target, payload["fold_id"], source, config.config_hash)
        result = {
            "status": "INTEGRITY_PASS",
            "coordinates": 24,
            **source,
            "paper_config_hash": config.config_hash,
            "folds": audits,
            "locked_test_opened": False,
        }
        write_json_atomic(target / "reassembly.json", result)
        os.replace(target, args.output)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    commands = parser.add_subparsers(dest="command", required=True)
    package = commands.add_parser("package-inputs")
    package.add_argument("--fold", choices=FOLDS, required=True)
    package.add_argument("--artifacts", type=Path)
    package.add_argument("--data-root", type=Path)
    package.add_argument("--output", type=Path, required=True)
    verify = commands.add_parser("verify")
    verify.add_argument("--bundle", type=Path, required=True)
    verify.add_argument("--bundle-sha256", required=True)
    qualification = commands.add_parser("qualify")
    qualification.add_argument("--fold", choices=FOLDS, required=True)
    qualification.add_argument("--work", type=Path, required=True)
    qualification.add_argument("--threads", type=int, default=16)
    gate = commands.add_parser("ready")
    gate.add_argument("--receipts", type=Path, nargs=3, required=True)
    gate.add_argument("--output", type=Path, required=True)
    for command in ("start", "resume", "_train"):
        launch = commands.add_parser(command)
        launch.add_argument("--fold", choices=FOLDS, required=True)
        launch.add_argument("--bundle", type=Path, required=True)
        launch.add_argument("--bundle-sha256", required=True)
        launch.add_argument("--work", type=Path, required=True)
        launch.add_argument("--output", type=Path, required=True)
        launch.add_argument("--ready", type=Path, required=True)
        launch.add_argument("--approval", type=Path, required=True)
    for command in ("stop", "status"):
        operation = commands.add_parser(command)
        operation.add_argument("--work", type=Path, required=True)
    export = commands.add_parser("export")
    export.add_argument("--fold", choices=FOLDS, required=True)
    export.add_argument("--models", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    merge = commands.add_parser("reassemble")
    merge.add_argument("--inputs", type=Path, nargs=3, required=True)
    merge.add_argument("--input-sha256", nargs=3, required=True)
    merge.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    os.chdir(args.repo)
    if args.command == "package-inputs":
        result = package_inputs(args)
    elif args.command == "verify":
        payload = verify_bundle(args.bundle, args.bundle_sha256)
        result = {"status": "PASS", "fold_id": payload["fold_id"], "files": len(payload["files"])}
    elif args.command == "qualify":
        result = qualify(args)
    elif args.command == "ready":
        result = ready(args)
    elif args.command in {"start", "resume"}:
        if sys.platform != "linux":
            raise RuntimeError("Pod launch/stop requires Linux.")
        args.work.mkdir(parents=True, exist_ok=True)
        state_path = args.work / "process.json"
        if state_path.exists():
            state = read_json(state_path)
            if state["token"] is not None and process_token(state["pid"]) == state["token"]:
                raise ValueError("Fold worker already active.")
        argv = sys.argv[1:]
        argv[argv.index(args.command)] = "_train"
        log = args.work / f"attempt-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%f')}.log"
        with log.open("wb") as stream:
            child = subprocess.Popen(
                [sys.executable, str(Path(__file__).resolve()), *argv],
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        result = {
            "pid": child.pid,
            "token": process_token(child.pid),
            "output": str(args.output.resolve()),
            "log": str(log),
            "fold_id": args.fold,
        }
        write_json_atomic(state_path, result)
    elif args.command == "_train":
        import fcntl

        lock = (args.work / "worker.lock").open("a")
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
        try:
            result = train(args)
        except BaseException as error:
            write_json_atomic(
                args.work / "last-exit.json",
                {
                    "status": "INTERRUPTED" if isinstance(error, KeyboardInterrupt) else "FAILED",
                    "error": str(error),
                },
            )
            raise
        write_json_atomic(args.work / "last-exit.json", result)
    elif args.command in {"stop", "status"}:
        state = read_json(args.work / "process.json")
        live = (
            process_token(state["pid"]) is not None
            and process_token(state["pid"]) == state["token"]
        )
        if args.command == "stop" and live:
            os.kill(state["pid"], signal.SIGTERM)
        result = {**state, "active": live, "stop_requested": args.command == "stop" and live}
        result["completed_coordinates"] = len(
            list((Path(state["output"]) / state["fold_id"]).glob("*/*/manifest.json"))
        )
        result["completed_candidates"] = len(
            list((args.work / "lightgbm-grid-cache").glob("**/candidate-*/manifest.json"))
        )
        result["last_exit"] = (
            read_json(args.work / "last-exit.json")
            if (args.work / "last-exit.json").is_file()
            else None
        )
    elif args.command == "export":
        result = export_fold(args)
    else:
        result = reassemble(args)
    print(json.dumps(result, indent=2, default=str), flush=True)


if __name__ == "__main__":
    main()
