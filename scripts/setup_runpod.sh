#!/usr/bin/env bash
# Ubuntu 22.04/24.04 NVIDIA pod. Run from the frozen repository checkout.
set -euo pipefail
REPOSITORY="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPOSITORY"
test "$(uname -s)" = Linux
if [ "$(id -u)" -eq 0 ]; then SUDO=(); else SUDO=(sudo); fi
"${SUDO[@]}" apt-get update
"${SUDO[@]}" apt-get install -y python3-venv python3-dev build-essential cmake ninja-build \
  libboost-dev libboost-system-dev libboost-filesystem-dev libboost-chrono-dev \
  ocl-icd-libopencl1 ocl-icd-opencl-dev opencl-headers clinfo rsync git
nvidia-smi
# NVIDIA Container Toolkit exposes the driver library; register its ICD if the image
# did not ship the vendor entry. Do not install or replace the host NVIDIA driver.
NVIDIA_ICD="$(ldconfig -p | awk '/libnvidia-opencl.so.1/{print $NF; exit}')"
if [ -z "$NVIDIA_ICD" ]; then
  printf '%s\n' 'NVIDIA OpenCL driver library is not exposed by this pod image.' >&2
  exit 1
fi
"${SUDO[@]}" install -d /etc/OpenCL/vendors
printf '%s\n' "$NVIDIA_ICD" | "${SUDO[@]}" tee /etc/OpenCL/vendors/nvidia.icd >/dev/null
clinfo --list
python3 -m venv .venv
.venv/bin/python -m pip install -e '.[paper,dev]'
# Force a source GPU/OpenCL build; never accept a CPU-only wheel as qualification.
.venv/bin/python -m pip install --force-reinstall --no-deps --no-cache-dir \
  --no-binary lightgbm --config-settings=cmake.define.USE_GPU=ON 'lightgbm==4.7.0'
.venv/bin/python -m pip check
mkdir -p .runtime/setup
.venv/bin/python -m pip freeze > .runtime/setup/python-packages.txt
dpkg-query -W > .runtime/setup/ubuntu-packages.txt
clinfo > .runtime/setup/opencl.txt
nvidia-smi -q > .runtime/setup/nvidia.txt
printf '%s\n' 'Setup complete. Each pod must pass scripts/runpod.py qualify before fitting.'
