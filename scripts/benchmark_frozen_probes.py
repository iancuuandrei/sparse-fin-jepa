"""Bounded frozen TRAIN/VALIDATION probe benchmark; never consumes TEST.

Run explicitly against a qualified source and immutable artifact root on Linux.
Emits runtime measurements and equivalence digests, not effectiveness values.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

from execsim.data.paper.manifests import file_sha256, read_json
from execsim.ml.representations.probe_runtime import probe_thread_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--mode", choices=["reference", "cached"], required=True)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--rows", type=int, default=2048)
    parser.add_argument("--geometry", choices=["dense", "sparse"], default="dense")
    args = parser.parse_args()
    source_identity = hashlib.sha256(
        json.dumps(
            {
                path.relative_to(args.source).as_posix(): file_sha256(path)
                for path in sorted((args.source / "src").rglob("*.py"))
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()

    with probe_thread_policy(args.threads) as policy:
        import torch

        import execsim

        if not Path(execsim.__file__).resolve().is_relative_to(args.source.resolve()):
            raise RuntimeError("Benchmark import does not match requested source checkout.")
        from torch.utils.data import DataLoader, Subset

        from execsim.ml.representations.checkpoints import load_checkpoint
        from execsim.ml.representations.frozen_evaluation import evaluate_frozen_capacity_streaming
        from execsim.ml.representations.jepa import PredictiveRepresentationModel
        from execsim.ml.representations.schemas import CheckpointCompatibility, RepresentationConfig
        from execsim.ml.sequences.streaming import PaperSequenceDataset

        if not torch.cuda.is_available():
            raise RuntimeError("This benchmark requires the qualified CUDA execution path.")
        checkpoint = (
            args.artifact_root / "upstream-cuda/representations/fold-1" / args.geometry / "13"
        )
        expected = CheckpointCompatibility(**read_json(checkpoint / "compatibility.json"))
        model = PredictiveRepresentationModel(
            RepresentationConfig(
                expected.geometry,
                predictor_family=expected.predictor_family,
                generalized_gaussian_p=expected.generalized_gaussian_p,
                generalized_gaussian_mu=expected.generalized_gaussian_mu,
                generalized_gaussian_sigma=expected.generalized_gaussian_sigma,
                rdm_projections_train=expected.rdm_projections,
                seed=13,
            )
        ).to("cuda")
        load_checkpoint(model, checkpoint / "final", expected=expected)
        sequence = args.artifact_root / "sequences/fold-1/sequence-manifest.json"
        loaders = []
        for partition, count in [("train", args.rows), ("validation", 512), ("validation", 512)]:
            dataset = PaperSequenceDataset(
                sequence, partition=partition, seed=13, cache_size=32, sample_train_positions=False
            )
            loaders.append(
                DataLoader(
                    Subset(dataset, range(count)),
                    batch_size=256,
                    shuffle=False,
                    num_workers=0,
                    pin_memory=True,
                    generator=torch.Generator().manual_seed(13),
                )
            )
        print(
            json.dumps(
                {
                    "event": "START",
                    "mode": args.mode,
                    "policy": policy,
                    "rows": args.rows,
                    "source_sha256": source_identity,
                    "scoring_partition": "validation-not-test",
                }
            ),
            flush=True,
        )
        start = time.perf_counter()
        cpu = time.process_time()
        kwargs = (
            {}
            if args.mode == "reference"
            else {
                "cache_root": args.work / "encoded",
                "cache_identity": {
                    "source_sha256": source_identity,
                    "sequence": file_sha256(sequence),
                    "rows": args.rows,
                    "checkpoint": file_sha256(checkpoint / "final/model.safetensors"),
                    "purpose": "bounded-validation-benchmark",
                },
            }
        )
        output = evaluate_frozen_capacity_streaming(
            model, *loaders, device="cuda", seed=13, **kwargs
        )
        torch.cuda.synchronize()
        canonical = [
            [{k: v for k, v in row.items() if "seconds" not in k} for row in group]
            for group in output
        ]
        # Expose only equivalence digest, never model effectiveness values.
        digest = hashlib.sha256(json.dumps(canonical, sort_keys=True).encode()).hexdigest()
        print(
            json.dumps(
                {
                    "event": "COMPLETE",
                    "mode": args.mode,
                    "wall_seconds": time.perf_counter() - start,
                    "cpu_seconds": time.process_time() - cpu,
                    "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                    "threads": len(list(Path("/proc/self/task").iterdir())),
                    "equivalence_sha256": digest,
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
