# ADR 0016: Separate LightGBM OpenCL execution identity

- Status: Accepted
- Date: 2026-09-06
- Scope: paper LightGBM runtime selection and artifact compatibility

## Context

The frozen paper configuration defines the supervised targets and the exact
eight-coordinate LightGBM search. It does not define the machine that executes
that search. Adding a device field to the six hashed paper YAML files would
therefore change protocol identity for an operational choice.

The Windows research host has integrated AMD graphics and a discrete NVIDIA
GPU. LightGBM's Windows accelerator is the OpenCL `gpu` backend, not its
Linux-only CUDA backend. LightGBM documents explicit OpenCL platform and device
IDs for multi-vendor hosts. It also documents `deterministic` and
`force_col_wise` as CPU-only and uses single-precision GPU histogram
accumulation unless `gpu_use_dp=true`.

The previous adapter embedded one CPU thread in each scientific candidate and
did not record a backend in model manifests. A resumed GPU request could
therefore reuse a CPU artifact without detecting the execution mismatch.

## Decision

Keep `LightGBMConfig` limited to the frozen learner and search coordinates.
Represent backend, OpenCL platform, OpenCL device, GPU precision, and CPU thread
count in a separate immutable `LightGBMExecutionOptions` value.

CPU remains the default and sends no GPU parameters. OpenCL GPU execution
requires explicit non-negative platform and device IDs. It sends
`device_type=gpu`, the two IDs, `gpu_use_dp`, and the selected thread count to
every scale and shape fit in every grid coordinate. The GPU path omits the
CPU-only `deterministic` and `force_col_wise` parameters instead of claiming
that they provide a GPU guarantee.

Store the complete execution identity in every native model manifest, grid
result, stage result, and stage-level execution receipt. Bind the receipt to the
paper configuration and exact Git commit. Artifact loading and stage resume
fail closed when the requested execution identity differs or is absent. The
existing parameter freeze hashes the model manifests and consequently binds
the selected backend without changing the scientific configuration hash.

Record the active runtime contract in this ADR and `docs/ML_DESIGN.md`.
`docs/SPECIFICATIONS.md` and `docs/standards/implementation.md` are themselves
hash-bound by the accepted v2 design freeze, so leave their frozen bytes
unchanged rather than silently recalculating that freeze for an operational
backend addition.

Expose the execution fields only on `execsim ml paper train-volume-model`.
Keep the default binning, grid, targets, categorical handling, and validation
selection unchanged.

## Alternatives considered

- Add GPU fields to `lightgbm.yaml`. Rejected because hardware is operational
  provenance and changing a hashed scientific YAML would invalidate the design
  freeze.
- Use `device_type=cuda` on Windows. Rejected because LightGBM documents its
  CUDA implementation as unsupported on Windows.
- Let OpenCL choose the default device. Rejected because a multi-vendor host
  could silently select the integrated GPU.
- Treat CPU and GPU artifacts as interchangeable. Rejected because backend,
  precision, and thread behavior are part of reproducible fitted-artifact
  provenance.
- Set `max_bin=63` for GPU speed. Rejected because binning changes the learner,
  not merely its execution backend.

## Consequences

- Scientific candidate identity and the canonical paper configuration hash do
  not change.
- A full paper LightGBM stage uses one declared backend for all methods, seeds,
  targets, and grid coordinates.
- CPU artifacts cannot satisfy GPU resumes, GPU artifacts cannot satisfy CPU
  resumes, and a different OpenCL device cannot satisfy the same GPU request.
- GPU execution remains seeded but does not claim LightGBM's CPU deterministic
  guarantee. Repeated-run synthetic qualification must precede historical GPU
  fitting.
- `gpu_use_dp` and thread count are selected from synthetic reproducibility and
  runtime evidence only, never historical validation accuracy.
- The decision changes no target, feature, split, hyperparameter grid,
  selection rule, confirmatory contrast, or estimand.

## Verification

- CPU-only tests verify backend-specific parameter propagation without OpenCL
  hardware.
- Grid tests verify that all eight candidates receive the same execution
  identity.
- Native artifact tests verify execution-identity round trips and fail-closed
  CPU/GPU mismatch handling.
- CLI tests verify that the five execution flags exist only on
  `train-volume-model`.

## Sources

- [LightGBM parameters](https://lightgbm.readthedocs.io/en/latest/Parameters.html)
- [LightGBM GPU device targeting](https://github.com/lightgbm-org/LightGBM/blob/main/docs/GPU-Targets.rst)
- [LightGBM installation guide](https://github.com/lightgbm-org/LightGBM/blob/main/docs/Installation-Guide.rst)
