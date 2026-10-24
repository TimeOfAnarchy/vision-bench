from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from .errors import BenchmarkError


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_object(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise BenchmarkError(f"Cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise BenchmarkError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    values = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"line {line_number} is not an object")
                values.append(value)
    except (OSError, ValueError) as exc:
        raise BenchmarkError(f"Cannot read JSONL {path}: {exc}") from exc
    return values


def write_jsonl(path: Path, values: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for value in values:
            handle.write(canonical_json(value) + "\n")


def relative_path(relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative:
        raise BenchmarkError("Artifact/image paths must be nonempty POSIX relative paths.")
    candidate = Path(relative)
    if candidate.is_absolute() or ":" in relative or any(p in {"", ".", ".."} for p in relative.split("/")):
        raise BenchmarkError(f"Unsafe relative path: {relative!r}")
    return candidate


def child_path(root: Path, relative: str) -> Path:
    resolved = (root / relative_path(relative)).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise BenchmarkError(f"Path escapes its root: {relative!r}")
    return resolved


def identifier(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", value):
        raise BenchmarkError(f"Invalid identifier: {value!r}")
    return value


def sha256_value(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise BenchmarkError("Expected a lowercase SHA-256 digest.")
    return value


def positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise BenchmarkError(f"{name} must be a positive integer.")
    return value
