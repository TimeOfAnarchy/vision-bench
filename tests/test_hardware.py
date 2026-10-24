from __future__ import annotations

import io
import json
import subprocess

import pytest

from vision_bench.errors import BenchmarkError
from vision_bench.hardware import GpuSampler, MemoryStats
from vision_bench.io import read_object, write_json
from vision_bench.runner import run_benchmark
from vision_bench.worker_process import execute_worker


def test_memory_stats_require_a_real_sampling_span_and_keep_memory_types_separate():
    stats = MemoryStats(123, 250)
    stats.observe(0, None)
    stats.observe(0.1, {"dedicated_bytes": 100, "shared_bytes": 800})
    assert stats.summary()["status"] == "insufficient_samples"
    assert stats.summary()["dedicated_observed_high_water_bytes"] is None
    stats.observe(0.4, {"dedicated_bytes": 300, "shared_bytes": 200})
    result = stats.summary()
    assert result["status"] == "observed"
    assert result["dedicated_observed_high_water_bytes"] == 300
    assert result["shared_observed_high_water_bytes"] == 800
    assert result["worker_pid"] == 123
    assert result["valid_samples"] == 2
    stats.error = "counter failed"
    assert stats.summary()["status"] == "unavailable"
    assert stats.summary()["dedicated_observed_high_water_bytes"] is None


def test_missing_counter_is_unavailable_not_zero(tmp_path):
    def unavailable(pid):
        raise OSError("no counter")

    sampler = GpuSampler(123, tmp_path / "memory.jsonl", reader_factory=unavailable)
    sampler.start()
    summary = sampler.stop()
    assert summary["status"] == "unavailable"
    assert summary["dedicated_observed_high_water_bytes"] is None
    assert "no counter" in summary["reason"]


@pytest.mark.parametrize("timeout", [False, True])
def test_worker_lifecycle_terminates_only_on_failure(tmp_path, monkeypatch, timeout):
    import vision_bench.worker_process as worker_process

    write_json(tmp_path / "request.json", {"start_token": "fixture"})
    calls = []

    class Launcher:
        pid = 101
        args = ["python", "worker"]

        def __init__(self, *args, **kwargs):
            self.stdin, self.stdout = io.StringIO(), io.StringIO()
            self.returncode = None
            write_json(tmp_path / "worker.ready.json", {"pid": 202, "start_token": "fixture"})

        def poll(self):
            return self.returncode

        def communicate(self, **kwargs):
            calls.append(("start", kwargs["input"]))
            if timeout:
                raise subprocess.TimeoutExpired(self.args, 10)
            self.returncode = 0
            return json.dumps({"ok": True, "runtime": {"pid": 202}}), None

        def kill(self):
            calls.append(("launcher_killed", self.pid))
            self.returncode = 1

        def wait(self):
            return self.returncode

    class Owned:
        def __init__(self, pid):
            self.pid = pid
            calls.append(("owned", pid))

        def terminate(self):
            calls.append(("owned_terminated", self.pid))

        def wait(self, deadline):
            calls.append(("owned_waited", self.pid))

        def close(self):
            calls.append(("owned_closed", self.pid))

    monkeypatch.setattr(subprocess, "Popen", Launcher)
    monkeypatch.setattr(worker_process, "_WindowsWorker", Owned)
    # Exercise the owned-handle branch on any test host without touching real processes.
    monkeypatch.setattr(
        worker_process, "os", type("OS", (), {"name": "nt", "getpid": staticmethod(lambda: 999)})
    )
    if timeout:
        with pytest.raises(BenchmarkError, match="deadline"):
            execute_worker(tmp_path, timeout=10, measure_memory=False, interval_ms=250)
        assert calls == [
            ("owned", 202),
            ("start", "fixture\n"),
            ("owned_terminated", 202),
            ("owned_closed", 202),
            ("launcher_killed", 101),
        ]
    else:
        assert execute_worker(tmp_path, timeout=10, measure_memory=False, interval_ms=250)["pid"] == 202
        assert calls == [("owned", 202), ("start", "fixture\n"), ("owned_waited", 202), ("owned_closed", 202)]


def test_shared_only_sampling_keeps_dedicated_unavailable():
    stats = MemoryStats(123, 250)
    stats.observe(0, {"dedicated_bytes": None, "shared_bytes": 800})
    stats.observe(0.3, {"dedicated_bytes": None, "shared_bytes": 900})
    result = stats.summary()
    assert result["status"] == "partial"
    assert result["dedicated_observed_high_water_bytes"] is None
    assert result["shared_observed_high_water_bytes"] == 900


def test_optional_sampler_uses_actual_worker_pid_and_does_not_fail_cpu_run(local_inputs, monkeypatch):
    captured = []

    class Sampler:
        def __init__(self, pid, path, **kwargs):
            captured.append(pid)

        def start(self):
            pass

        def stop(self):
            return {"status": "unavailable", "reason": "fixture_no_counter"}

    monkeypatch.setattr("vision_bench.worker_process.GpuSampler", Sampler)
    values = local_inputs
    path = run_benchmark(
        dataset_path=values["dataset"],
        profile_path=values["profile"],
        model_root=values["root"],
        output_root=values["output"],
        measure_memory=True,
    )
    manifest = read_object(path / "run.json")
    assert captured == [manifest["runtime"]["pid"]]
    assert manifest["runtime"]["gpu_memory"]["status"] == "unavailable"
    assert read_object(path / "metrics.json")["outcomes"]["successful"] == 3
