"""Best-effort run-time hardware inventory, without hostnames, paths or device IDs."""

from __future__ import annotations

import csv
import ctypes
import os
import platform
import shutil
import subprocess


def capture_environment() -> dict:
    cpu, ram = platform.processor() or None, None
    try:
        if os.name == "nt":
            import winreg

            with winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0"
            ) as key:
                cpu = winreg.QueryValueEx(key, "ProcessorNameString")[0].strip()

            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                    (name, ctypes.c_ulonglong)
                    for name in (
                        "physical",
                        "available",
                        "page",
                        "page_available",
                        "virtual",
                        "virtual_available",
                        "extended",
                    )
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                ram = status.physical
        else:
            ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
            if platform.system() == "Linux":
                from pathlib import Path

                for line in Path("/proc/cpuinfo").read_text().splitlines():
                    if line.startswith("model name"):
                        cpu = line.split(":", 1)[1].strip()
                        break
    except (OSError, ValueError, AttributeError):
        pass
    gpu = {
        "status": "unavailable",
        "devices": [],
        "scope": "visible NVIDIA inventory; not session device binding",
    }
    executable = shutil.which("nvidia-smi")
    if executable:
        try:
            result = subprocess.run(
                [executable, "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=3,
                check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            devices = [
                {"name": name.strip(), "vram_mib": int(float(memory.strip())), "driver": driver.strip()}
                for name, memory, driver in csv.reader(result.stdout.splitlines())
            ]
            if devices:
                gpu.update(status="observed", devices=devices)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return {
        "collector": "host-v1",
        "os": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "ram_bytes": ram,
        "gpu_inventory": gpu,
    }
