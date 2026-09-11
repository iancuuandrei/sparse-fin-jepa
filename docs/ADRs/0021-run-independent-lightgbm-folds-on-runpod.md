# ADR 0021: Run independent LightGBM folds on RunPod

- Status: Accepted
- Date: 2026-09-08
- Supplements: ADR 0020

## Problem and evidence

The interrupted local attempt repeatedly constructed the same causal raw frames for eight
variants per fold. The user selected three Community Cloud RTX 3090 pods, with one fold on
each pod. The local attempt ended through Ctrl+C with KeyboardInterrupt and zero selected
coordinates. Its logs and interruption receipt remain untracked operational evidence.

## Decision

Each process runs one explicit fold and its eight sequential coordinates. TRAIN and VALIDATION
bases are cached as checksummed Parquet and NumPy files. Representations are joined by sample
identity only for the active coordinate, preserving the original feature column order.
Candidates and coordinates are published by directory rename after all constituent files
are written. Resume verifies source, config, execution, inputs, native model checksums and
selection results. A missing or corrupt completion record cannot be reused.

RunPod input packages contain the unchanged upstream manifests and only TRAIN/VALIDATION
session/index/embedding files. Missing TEST files are intentional. Packaging verifies indexed
sequence hashes and embedding manifest hashes; import checks every file against the transfer
manifest and its separately supplied SHA-256. The upstream representation commit remains
distinct from the downstream commit. Packages do not require provider credentials or raw bars.

Each pod discovers OpenCL GPU IDs and must qualify exactly one RTX 3090 with LightGBM 4.7.0,
OpenCL GPU, single precision accumulation and the existing repeat tolerances (rtol 0.005,
atol 1e-6). Qualification uses synthetic categories, weights, early stopping and native
roundtrips. Three separate successful qualifications gate launch. No performance selection
occurs. Hosts, drivers and device IDs remain execution provenance.

Reassembly requires eight exact coordinates per fold, matching source/config/backend,
checksummed models and grid results, and the declared validation selection rule. It publishes
an integrity receipt. It does not open TEST or substitute for the later parameter freeze.

## Alternatives and consequences

Further local benchmarking and alternate GPUs were rejected by the user. GPU distributed
training and networking during fitting are unnecessary: folds are independent. A pod needs
enough RAM for one wide coordinate plus common bases; this is not out-of-core LightGBM.
Interruption during a candidate repeats that candidate, while completed candidates survive.
Abrupt interruption can leave hidden staging directories; they are never completion evidence.

The estimand, feature order, observations, targets, weights, seeds, grid and selection rules
are unchanged. A byte-preserved reference of the previous builder supports exact fixture
comparison. No empirical effectiveness claim follows from these software tests.

The existing v2 freeze binds the implementation standard and repository manifest. As in
ADR 0020, this operational direction is recorded here and linked from the unfrozen
implementation report instead of modifying frozen scientific identity documents.

## Specification and verification

The executable procedure is [RUNPOD_EXECUTION.md](../RUNPOD_EXECUTION.md).
Regression coverage includes frame equivalence, candidate interruption/resume, transfer
integrity, distinct-fold launch barriers, native save/load and full repository gates.
Actual Ubuntu setup and RTX 3090 qualification remain NOT RUN until pods are provisioned.

Build source: [LightGBM GPU installation](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html#build-gpu-version).
