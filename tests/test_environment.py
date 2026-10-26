from __future__ import annotations

import json
import subprocess

from vision_bench.environment import capture_environment


def test_environment_inventory_keeps_useful_hardware_without_personal_identifiers(monkeypatch):
    monkeypatch.setattr("vision_bench.environment.shutil.which", lambda name: "example-nvidia-smi")
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "Example GPU, 12288, 123.45\n")

    monkeypatch.setattr("vision_bench.environment.subprocess.run", run)
    data = capture_environment()
    assert data["gpu_inventory"]["devices"] == [
        {"name": "Example GPU", "vram_mib": 12288, "driver": "123.45"}
    ]
    assert data["gpu_inventory"]["status"] == "observed"
    assert data["logical_cpus"]
    assert data["os"]
    assert all("uuid" not in " ".join(call) and "serial" not in " ".join(call) for call in calls)
    assert "hostname" not in json.dumps(data)


def test_failed_hardware_probe_is_unavailable_without_leaking_errors(monkeypatch):
    monkeypatch.setattr("vision_bench.environment.shutil.which", lambda name: "example-nvidia-smi")

    def fail(*args, **kwargs):
        raise subprocess.TimeoutExpired("C:/private-user/nvidia-smi", 3)

    monkeypatch.setattr("vision_bench.environment.subprocess.run", fail)
    data = capture_environment()
    assert data["gpu_inventory"]["status"] == "unavailable"
    assert data["gpu_inventory"]["devices"] == []
    assert "private-user" not in json.dumps(data)
