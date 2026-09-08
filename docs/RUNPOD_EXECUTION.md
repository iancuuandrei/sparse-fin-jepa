# RunPod fold execution

Use three Ubuntu 22.04/24.04 NVIDIA RTX 3090 Community pods, one GPU and approximately
16 vCPU / 125 GB RAM / 24 GB VRAM each. All pods use the same committed repository tree.
The commands below perform TRAIN/VALIDATION fitting only. TEST files are excluded from input
packages. Ubuntu installation and real RTX 3090 qualification are NOT RUN locally.

## Prepare on the authoritative machine

From the repository, use the committed downstream SHA reported by `git rev-parse HEAD` as
`SOURCE_COMMIT` on all pods. Do not package from a dirty checkout. Using the repository Python:

```powershell
.venv/Scripts/python.exe scripts/runpod.py package-inputs --fold fold-1 --output .runtime/transfer/fold-1
.venv/Scripts/python.exe scripts/runpod.py package-inputs --fold fold-2 --output .runtime/transfer/fold-2
.venv/Scripts/python.exe scripts/runpod.py package-inputs --fold fold-3 --output .runtime/transfer/fold-3
git bundle create .runtime/source.bundle HEAD
```

Record each printed `manifest_sha256` separately as `BUNDLE_SHA`. The output directories
must be new. Transfer the corresponding directory to `/workspace/input` on each pod using
SCP/SFTP or rsync over SSH. Transfer only the matching fold. Interrupted rsync transfers can
resume; the launch command always verifies all bytes again. Do not edit upstream JSON to
change paths: relative sequence paths are resolved below the imported root.

Copy `.runtime/source.bundle` to `/workspace/source.bundle` on each pod; it contains the
committed source and avoids depending on an unpublished GitHub revision.
Copy the already-authorized runtime approval JSON to
`/workspace/work/paper-approvals/approval.json` on each pod. It must match the current config
hash, grant `historical_training`, and deny `locked_result_evaluation`. Never commit it.

## Set up each pod

Set `SOURCE_COMMIT` to the exact SHA of this completed refactor. Authenticate Git using your
normal SSH setup if needed. No RunPod API key is required by these scripts.

```bash
git clone /workspace/source.bundle /workspace/repo
cd /workspace/repo
git checkout --detach "$SOURCE_COMMIT"
bash scripts/setup_runpod.sh
mkdir -p /workspace/work/paper-approvals
```

Setup builds LightGBM 4.7.0 from source with `USE_GPU=ON`, installs repository-declared
dependencies and records package inventories. The pod image must expose NVIDIA's OpenCL
ICD and driver. If `clinfo` or qualification cannot see the RTX 3090, repair the driver/ICD
installation; do not substitute another GPU or CPU backend. Roots shown here are examples,
not compiled-in paths; `--repo`, `--bundle`, `--work` and `--output` accept other locations.

On pod 1, pod 2 and pod 3 respectively:

```bash
.venv/bin/python scripts/runpod.py qualify --fold fold-1 --work /workspace/work --threads 16
.venv/bin/python scripts/runpod.py qualify --fold fold-2 --work /workspace/work --threads 16
.venv/bin/python scripts/runpod.py qualify --fold fold-3 --work /workspace/work --threads 16
```

Run exactly the command matching that pod. Copy each pod's
`/workspace/work/lightgbm-gpu/qualification-receipt.json` back to the authoritative machine
as `.runtime/pods/fold-1.json`, `fold-2.json`, `fold-3.json`. Require all three PASS:

```powershell
.venv/Scripts/python.exe scripts/runpod.py ready --receipts .runtime/pods/fold-1.json .runtime/pods/fold-2.json .runtime/pods/fold-3.json --output .runtime/pods/launch-ready.json
```

Copy `launch-ready.json` to `/workspace/launch-ready.json` on each pod.
This is a three-pod qualification gate, not LOCKED-TEST-READY.

## Launch the three folds

Set `BUNDLE_SHA` on each pod to that fold's separately recorded input manifest hash.
Run the corresponding command on all three pods after the launch gate is created:

```bash
# Pod 1
bash scripts/launch_fold_1.sh --bundle /workspace/input --bundle-sha256 "$BUNDLE_SHA" --work /workspace/work --output /workspace/models --ready /workspace/launch-ready.json --approval /workspace/work/paper-approvals/approval.json
# Pod 2
bash scripts/launch_fold_2.sh --bundle /workspace/input --bundle-sha256 "$BUNDLE_SHA" --work /workspace/work --output /workspace/models --ready /workspace/launch-ready.json --approval /workspace/work/paper-approvals/approval.json
# Pod 3
bash scripts/launch_fold_3.sh --bundle /workspace/input --bundle-sha256 "$BUNDLE_SHA" --work /workspace/work --output /workspace/models --ready /workspace/launch-ready.json --approval /workspace/work/paper-approvals/approval.json
```

The detached process survives SSH disconnection. Keep `/workspace/input`, `/workspace/work`
and `/workspace/models` on storage retained across pod restarts; back up completed outputs
before terminating a Community pod. No network communication is required during fitting.

## Status, graceful stop and resume

```bash
.venv/bin/python scripts/runpod.py status --work /workspace/work
.venv/bin/python scripts/runpod.py stop --work /workspace/work
.venv/bin/python scripts/runpod.py status --work /workspace/work
```

Stop sends SIGTERM to the verified worker PID/start-time. Python records an interruption;
native fitting may finish its current call before delivering the signal. Wait until status
reports `active: false`. No forced kill is performed. Resume with the same launch command
or `runpod.py resume` and the same fold/roots/arguments. Completed candidates and coordinates
are validated and reused. A corrupt completed artifact causes failure and requires an audit;
it is never silently discarded. Logs are separate per attempt. Hidden partial directories
are ignored as completion evidence and preserved for diagnosis.

## Export and reassemble

On each pod, replace `FOLD` with its assigned `fold-1`, `fold-2` or `fold-3`:

```bash
.venv/bin/python scripts/runpod.py export --fold "$FOLD" --models /workspace/models --output /workspace/export
```

Record the printed manifest SHA and copy `/workspace/export` to
`.runtime/results/fold-1`, `fold-2` or `fold-3` on the authoritative machine. Verify each copy
with its separately recorded `RESULT_SHA` before reassembly:

```powershell
.venv/Scripts/python.exe scripts/runpod.py verify --bundle .runtime/results/fold-1 --bundle-sha256 $RESULT_SHA
.venv/Scripts/python.exe scripts/runpod.py reassemble --inputs .runtime/results/fold-1 .runtime/results/fold-2 .runtime/results/fold-3 --input-sha256 $RESULT1_SHA $RESULT2_SHA $RESULT3_SHA --output .runtime/reassembled-lightgbm
```

Repeat `verify` for folds 2 and 3 with their own hashes. Reassembly requires a new output
directory and checks exact 8/8 + 8/8 + 8/8 membership, source/config, backend, native hashes,
candidate grids and validation selection. It preserves per-host provenance. The resulting
tree is a candidate authoritative model tree pending promotion; existing local attempts
remain untouched. No command in this procedure creates LOCKED-TEST-OPENED or evaluates TEST.

## Operational contracts

`execsim ml paper train-volume-model --fold` also accepts `--sequence-root`, `--embedding-root`,
`--model-output-root`, `--paper-data-root`, `--paper-cache-root`, and `--paper-artifact-root`.
Their environment equivalents are `EXECSIM_SEQUENCE_ROOT`, `EXECSIM_EMBEDDING_ROOT`,
`EXECSIM_MODEL_OUTPUT_ROOT`, `EXECSIM_PAPER_DATA_ROOT`, `EXECSIM_PAPER_CACHE_ROOT` and
`EXECSIM_PAPER_ARTIFACT_ROOT`. They do not alter the six scientific YAML files or config hash.
Use the RunPod wrapper for deployment: it adds transfer, host, source and three-pod checks.
One coordinate's wide frames must fit in RAM; LightGBM itself is not out-of-core.
