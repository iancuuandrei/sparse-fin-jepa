"""Representation-evaluation-only native thread limits derived from CPU quotas."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


def effective_cpu_capacity(cgroup_root: Path = Path("/sys/fs/cgroup")) -> float:
    """Take the minimum visible affinity and readable v1/v2 ancestor quota."""
    affinity = getattr(os, "sched_getaffinity", None)
    capacity = float(len(affinity(0)) if affinity else (os.cpu_count() or 1))
    candidates = {cgroup_root, cgroup_root / "cpu", cgroup_root / "cpu,cpuacct"}
    membership = Path("/proc/self/cgroup")
    if membership.is_file():
        for line in membership.read_text().splitlines():
            _, controllers, relative = line.split(":", 2)
            prefix = cgroup_root / controllers if controllers else cgroup_root
            path = prefix / relative.lstrip("/")
            if ".." not in path.parts:
                candidates.update(p for p in (path, *path.parents) if p.is_relative_to(cgroup_root))
    for root in candidates:
        try:
            if (root / "cpu.max").is_file():
                quota, period = (root / "cpu.max").read_text().split()
                if quota != "max":
                    capacity = min(capacity, int(quota) / int(period))
            elif (root / "cpu.cfs_quota_us").is_file():
                quota_value = int((root / "cpu.cfs_quota_us").read_text())
                period_value = int((root / "cpu.cfs_period_us").read_text())
                if quota_value > 0:
                    capacity = min(capacity, quota_value / period_value)
        except (OSError, ValueError, ZeroDivisionError) as exc:
            raise RuntimeError(f"Cannot interpret CPU quota at {root}.") from exc
    return capacity


@contextmanager
def probe_thread_policy(threads: int = 4) -> Iterator[dict[str, Any]]:
    """Bound native pools for this stage; leave scientific options untouched."""
    capacity = effective_cpu_capacity()
    if threads < 1 or capacity < 1:
        raise ValueError("Representation probes require at least one effective CPU.")
    limit = min(threads, int(capacity))
    names = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    previous_env = {name: os.environ.get(name) for name in names}
    for name in names:
        os.environ[name] = str(limit)
    import pyarrow as pa
    import torch
    from threadpoolctl import threadpool_limits

    previous = (torch.get_num_threads(), pa.cpu_count(), pa.io_thread_count())
    try:
        # Inter-op may only be configured before its first parallel operation.
        # A late incompatible caller fails before historical work, not silently.
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.set_num_threads(limit)
        pa.set_cpu_count(limit)
        pa.set_io_thread_count(min(2, limit))
        with threadpool_limits(limits=limit):
            yield {
                "effective_cpu_capacity": capacity,
                "native_threads": limit,
                "torch_interop_threads": 1,
                "arrow_io_threads": min(2, limit),
            }
    finally:
        torch.set_num_threads(previous[0])
        pa.set_cpu_count(previous[1])
        pa.set_io_thread_count(previous[2])
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
