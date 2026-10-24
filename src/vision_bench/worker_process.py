from __future__ import annotations

import ctypes
import json
import os
import subprocess
import sys
import time
from ctypes import wintypes
from pathlib import Path

from .errors import BenchmarkError
from .hardware import GpuSampler
from .io import read_object


class _WindowsWorker:
    """Hold the actual process object, avoiding launcher PID and PID reuse ambiguity."""

    def __init__(self, pid: int):
        self.api = ctypes.WinDLL("kernel32", use_last_error=True)
        self.api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        self.api.OpenProcess.restype = wintypes.HANDLE
        self.api.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
        self.api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        self.api.CloseHandle.argtypes = [wintypes.HANDLE]
        self.handle = self.api.OpenProcess(0x1 | 0x100000, False, pid)
        if not self.handle:
            raise OSError(ctypes.get_last_error(), "Cannot hold the worker process handle.")

    def close(self) -> None:
        self.api.CloseHandle(self.handle)

    def terminate(self) -> None:
        self.api.TerminateProcess(self.handle, 1)
        self.api.WaitForSingleObject(self.handle, 5000)

    def wait(self, timeout: float) -> None:
        status = self.api.WaitForSingleObject(self.handle, max(0, int(timeout * 1000)))
        if status == 258:
            raise subprocess.TimeoutExpired("worker", timeout)
        if status != 0:
            raise OSError("Cannot wait for the worker process.")


def execute_worker(run_dir: Path, *, timeout: float, measure_memory: bool, interval_ms: int) -> dict:
    deadline = time.monotonic() + timeout
    request = read_object(run_dir / "request.json")
    ready_path = run_dir / "worker.ready.json"
    sampler, owned = None, None
    completed = False
    with (run_dir / "worker.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "vision_bench._worker", str(run_dir)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=log,
            text=True,
            encoding="utf-8",
        )
        try:
            while True:
                if time.monotonic() >= deadline:
                    raise subprocess.TimeoutExpired(process.args, timeout)
                if ready_path.exists():
                    try:
                        ready = read_object(ready_path)
                    except BenchmarkError:
                        ready = {}
                    if ready.get("start_token") == request["start_token"]:
                        break
                if process.poll() is not None:
                    raise BenchmarkError("Worker exited before the startup handshake; inspect worker.log.")
                time.sleep(0.01)
            pid = ready.get("pid")
            if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 or pid == os.getpid():
                raise BenchmarkError("Worker returned an invalid process identity.")
            if os.name == "nt":
                owned = _WindowsWorker(pid)
            if measure_memory:
                sampler = GpuSampler(pid, run_dir / "memory.jsonl", interval_ms=interval_ms)
                sampler.start()
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, timeout)
            stdout, _ = process.communicate(input=request["start_token"] + "\n", timeout=remaining)
            if owned is not None:
                owned.wait(deadline - time.monotonic())
            try:
                response = json.loads(stdout)
            except ValueError as exc:
                raise BenchmarkError("Worker returned invalid JSON; inspect worker.log.") from exc
            if process.returncode or response.get("ok") is not True:
                raise BenchmarkError(f"Worker failed: {response.get('error', response)}")
            runtime = response["runtime"]
            if runtime["pid"] != pid:
                raise BenchmarkError("Worker process identity changed after startup.")
            runtime["launcher_pid"] = process.pid
            runtime["gpu_memory"] = (
                sampler.stop()
                if sampler is not None
                else {"status": "unavailable", "reason": "not_requested"}
            )
            sampler = None
            completed = True
            return runtime
        except subprocess.TimeoutExpired as exc:
            raise BenchmarkError(
                f"Worker exceeded its {timeout:g}s deadline; run remains diagnostic."
            ) from exc
        finally:
            if owned is not None:
                if not completed:
                    owned.terminate()
                owned.close()
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
            if process.poll() is None:
                process.kill()
            process.wait()
            if process.stdout is not None:
                process.stdout.close()
            if sampler is not None:
                sampler.stop()
