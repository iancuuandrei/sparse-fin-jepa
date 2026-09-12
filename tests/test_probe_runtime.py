"""Quota detection must ignore host CPU count when a container has a quota."""

from pathlib import Path

import pytest

from execsim.ml.representations.probe_runtime import effective_cpu_capacity


@pytest.mark.parametrize("kind", ["ewma", "tca"])
def test_evaluation_pool_caps_processes_to_quota(monkeypatch, kind):
    from execsim.ml.paper import evaluation_workers, tca_workers
    from execsim.ml.representations import probe_runtime

    monkeypatch.setattr(probe_runtime, "effective_cpu_capacity", lambda: 2.5)
    seen = []

    class Pool:
        def __init__(self, **kwargs):
            seen.append(kwargs)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, *args, **kwargs):
            return []

    module = evaluation_workers if kind == "ewma" else tca_workers
    monkeypatch.setattr(module, "ProcessPoolExecutor", Pool)
    runner = module.run_ewma_workers if kind == "ewma" else module.run_tca_workers
    assert runner([], workers=16) == []
    assert seen[0]["max_workers"] == 2
    assert seen[0]["initializer"] is evaluation_workers.configure_evaluation_worker


def test_v2_and_v1_cpu_quotas(tmp_path: Path, monkeypatch) -> None:
    import os

    monkeypatch.setattr(os, "cpu_count", lambda: 256)
    monkeypatch.setattr(os, "sched_getaffinity", lambda _: set(range(256)), raising=False)
    (tmp_path / "cpu.max").write_text("2720000 100000")
    assert effective_cpu_capacity(tmp_path) == 27.2
    (tmp_path / "cpu.max").write_text("max 100000")
    cpu = tmp_path / "cpu"
    cpu.mkdir()
    (cpu / "cpu.cfs_quota_us").write_text("150000")
    (cpu / "cpu.cfs_period_us").write_text("100000")
    assert effective_cpu_capacity(tmp_path) == 1.5
    (cpu / "cpu.cfs_period_us").write_text("0")
    with pytest.raises(RuntimeError, match="CPU quota"):
        effective_cpu_capacity(tmp_path)
