"""Optional Windows WDDM process counters; no model/runtime imports."""

from __future__ import annotations

import ctypes
import os
import threading
import time
from ctypes import wintypes
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .io import canonical_json


class _ValueUnion(ctypes.Union):
    _fields_ = [("large", ctypes.c_longlong), ("double", ctypes.c_double), ("pointer", ctypes.c_void_p)]


class _Value(ctypes.Structure):
    _fields_ = [("status", wintypes.DWORD), ("value", _ValueUnion)]


class _Item(ctypes.Structure):
    _fields_ = [("name", wintypes.LPWSTR), ("formatted", _Value)]


class PdhGpuReader:
    """PDH English counter names work independently of the Windows UI language."""

    def __init__(self, pid: int):
        if os.name != "nt":
            raise OSError("WDDM GPU counters require Windows.")
        self.pid = pid
        self.api = ctypes.WinDLL("pdh")
        self.api.PdhOpenQueryW.argtypes = [wintypes.LPCWSTR, ctypes.c_size_t, ctypes.POINTER(wintypes.HANDLE)]
        self.api.PdhAddEnglishCounterW.argtypes = [
            wintypes.HANDLE,
            wintypes.LPCWSTR,
            ctypes.c_size_t,
            ctypes.POINTER(wintypes.HANDLE),
        ]
        self.api.PdhCollectQueryData.argtypes = [wintypes.HANDLE]
        self.api.PdhGetFormattedCounterArrayW.argtypes = [
            wintypes.HANDLE,
            wintypes.DWORD,
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(wintypes.DWORD),
            ctypes.c_void_p,
        ]
        self.api.PdhCloseQuery.argtypes = [wintypes.HANDLE]
        for name in (
            "PdhOpenQueryW",
            "PdhAddEnglishCounterW",
            "PdhCollectQueryData",
            "PdhGetFormattedCounterArrayW",
            "PdhCloseQuery",
        ):
            getattr(self.api, name).restype = wintypes.DWORD
        self.query = wintypes.HANDLE()
        self._check(self.api.PdhOpenQueryW(None, 0, ctypes.byref(self.query)))
        self.counters = {}
        try:
            for key, counter_name in (
                ("dedicated_bytes", "Dedicated Usage"),
                ("shared_bytes", "Shared Usage"),
            ):
                counter = wintypes.HANDLE()
                self._check(
                    self.api.PdhAddEnglishCounterW(
                        self.query, rf"\GPU Process Memory(*)\{counter_name}", 0, ctypes.byref(counter)
                    )
                )
                self.counters[key] = counter
        except BaseException:
            self.close()
            raise

    @staticmethod
    def _check(status: int) -> None:
        if status:
            raise OSError(f"PDH counter error 0x{status:08x}")

    def _array(self, counter) -> dict[str, int]:
        # Instances can change between calls; restart the size query on MORE_DATA.
        for _ in range(3):
            size, count = wintypes.DWORD(), wintypes.DWORD()
            status = self.api.PdhGetFormattedCounterArrayW(
                counter, 0x1400, ctypes.byref(size), ctypes.byref(count), None
            )
            if status in (0x800007D5, 0xC0000BC6, 0x800007D1):  # no data / not yet valid / no instance
                return {}
            if status != 0x800007D2:  # PDH_MORE_DATA
                self._check(status)
            if not size.value:
                return {}
            buffer = ctypes.create_string_buffer(size.value)
            status = self.api.PdhGetFormattedCounterArrayW(
                counter, 0x1400, ctypes.byref(size), ctypes.byref(count), buffer
            )
            if status == 0x800007D2:
                continue
            if status in (0x800007D5, 0xC0000BC6, 0x800007D1):
                return {}
            self._check(status)
            items = ctypes.cast(buffer, ctypes.POINTER(_Item))
            return {
                items[i].name: int(items[i].formatted.value.large)
                for i in range(count.value)
                if items[i].name
                and items[i].name.startswith(f"pid_{self.pid}_")
                and items[i].formatted.status in (0, 1)
                and items[i].formatted.value.large >= 0
            }
        raise OSError("PDH instances changed during repeated buffer reads.")

    def read(self) -> dict | None:
        status = self.api.PdhCollectQueryData(self.query)
        if status in (0x800007D5, 0xC0000BC6, 0x800007D1):
            return None
        self._check(status)
        dedicated, shared = (self._array(self.counters[key]) for key in ("dedicated_bytes", "shared_bytes"))
        if not dedicated and not shared:
            return None
        return {
            "dedicated_bytes": sum(dedicated.values()) if dedicated else None,
            "shared_bytes": sum(shared.values()) if shared else None,
            "dedicated_instances": len(dedicated),
            "shared_instances": len(shared),
        }

    def close(self) -> None:
        if self.query:
            self.api.PdhCloseQuery(self.query)
            self.query = wintypes.HANDLE()


@dataclass
class MemoryStats:
    pid: int
    interval_ms: int
    ticks: int = 0
    valid: int = 0
    first: float | None = None
    last: float | None = None
    pool_counts: dict = field(default_factory=dict)
    pool_first: dict = field(default_factory=dict)
    pool_last: dict = field(default_factory=dict)
    maxima: dict = field(default_factory=dict)
    error: str | None = None

    def observe(self, elapsed: float, value: dict | None) -> None:
        self.ticks += 1
        if value is not None and any(
            value.get(key) is not None for key in ("dedicated_bytes", "shared_bytes")
        ):
            self.valid += 1
            self.first = elapsed if self.first is None else self.first
            self.last = elapsed
            for key in ("dedicated_bytes", "shared_bytes"):
                if value.get(key) is not None:
                    self.pool_counts[key] = self.pool_counts.get(key, 0) + 1
                    self.pool_first.setdefault(key, elapsed)
                    self.pool_last[key] = elapsed
                    self.maxima[key] = max(self.maxima.get(key, 0), value[key])

    def summary(self) -> dict:
        span = self.last - self.first if self.first is not None and self.last is not None else 0.0
        observed = {
            key: self.pool_counts.get(key, 0) >= 2
            and self.pool_last[key] - self.pool_first[key] >= self.interval_ms / 1000
            for key in ("dedicated_bytes", "shared_bytes")
        }
        status = (
            "observed"
            if all(observed.values())
            else "partial"
            if any(observed.values())
            else "insufficient_samples"
            if self.valid
            else "unavailable"
        )
        if self.error:
            status = "unavailable"
        return {
            "status": status,
            "method": "windows_pdh_gpu_process_memory",
            "worker_pid": self.pid,
            "requested_interval_ms": self.interval_ms,
            "samples": self.ticks,
            "valid_samples": self.valid,
            "valid_samples_per_pool": self.pool_counts,
            "valid_span_seconds": span,
            "dedicated_observed_high_water_bytes": self.maxima.get("dedicated_bytes")
            if observed["dedicated_bytes"] and not self.error
            else None,
            "shared_observed_high_water_bytes": self.maxima.get("shared_bytes")
            if observed["shared_bytes"] and not self.error
            else None,
            "reason": self.error or (None if all(observed.values()) else "incomplete_observations"),
            "scope": "sampled process memory across GPU instances; not an instantaneous allocator peak",
        }


class GpuSampler:
    def __init__(self, pid: int, path: Path, *, interval_ms: int = 250, reader_factory=PdhGpuReader):
        if isinstance(interval_ms, bool) or not isinstance(interval_ms, int) or interval_ms < 250:
            raise ValueError("GPU sample interval must be at least 250 ms.")
        self.stats = MemoryStats(pid, interval_ms)
        self.path, self.factory = path, reader_factory
        self.stop_event = threading.Event()
        self.started_event = threading.Event()
        self.thread = None

    def start(self) -> None:
        self.thread = threading.Thread(target=self._collect, daemon=True)
        self.thread.start()
        if not self.started_event.wait(5):
            self.stats.error = "sampler_start_timeout"

    def _collect(self) -> None:
        reader = None
        started = time.monotonic()
        try:
            with self.path.open("x", encoding="utf-8", newline="\n") as output:
                reader = self.factory(self.stats.pid)
                while not self.stop_event.is_set():
                    value = reader.read()
                    elapsed = time.monotonic() - started
                    self.stats.observe(elapsed, value)
                    output.write(
                        canonical_json(
                            {
                                "utc": datetime.now(timezone.utc).isoformat(),
                                "elapsed_seconds": elapsed,
                                "worker_pid": self.stats.pid,
                                "value": value,
                            }
                        )
                        + "\n"
                    )
                    output.flush()
                    self.started_event.set()
                    self.stop_event.wait(self.stats.interval_ms / 1000)
        except Exception as exc:
            self.stats.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.started_event.set()
            if reader is not None:
                try:
                    reader.close()
                except Exception as exc:
                    self.stats.error = f"sampler_close_failed: {exc}"

    def stop(self) -> dict:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(5)
            if self.thread.is_alive():
                self.stats.error = "sampler_stop_timeout"
        return self.stats.summary()
