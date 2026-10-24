from __future__ import annotations

import importlib.metadata
import os
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from . import __version__
from .adapters import OnnxTagger
from .io import canonical_json, file_hash, read_jsonl, read_object, write_json
from .preprocess import PREPROCESS_VERSION
from .profiles import load_profile


def execute(run_dir: Path) -> dict:
    request = read_object(run_dir / "request.json")
    if read_object(run_dir / "run.json").get("state") != "running":
        raise ValueError("Worker requires a reserved, unfinished run.")
    with (run_dir / "worker.claim").open("x", encoding="utf-8") as claim:
        claim.write(str(os.getpid()))
    profile = load_profile(run_dir / "profile.json", model_root=Path(request["model_root"]))
    profile.verify()
    started = perf_counter()
    adapter = OnnxTagger(profile)
    model_info = adapter.load()
    load_ms = (perf_counter() - started) * 1000
    samples = read_jsonl(run_dir / "samples.jsonl")
    write_json(run_dir / "vocabulary.json", adapter.vocabulary.to_dict())
    shard_size = request["shard_size"]
    shard_rows, shard_ids, shard_valid = [], [], []
    shard_number = 0

    def flush() -> None:
        nonlocal shard_number
        if not shard_rows:
            return
        path = run_dir / f"scores-{shard_number:06d}.npz"
        with path.open("xb") as output:
            np.savez_compressed(
                output,
                scores=np.asarray(shard_rows, dtype=np.float32),
                sample_ids=np.asarray(shard_ids, dtype=np.str_),
                valid=np.asarray(shard_valid, dtype=np.bool_),
            )
        shard_number += 1
        shard_rows.clear()
        shard_ids.clear()
        shard_valid.clear()

    from .io import child_path

    with (run_dir / "records.jsonl").open("x", encoding="utf-8", newline="\n") as records:
        for row, sample in enumerate(samples):
            error, latency = None, None
            try:
                path = child_path(Path(request["image_root"]), sample["image"])
                if file_hash(path) != sample["sha256"]:
                    raise ValueError("Image SHA-256 differs from dataset.")
                started = perf_counter()
                scores = adapter.infer(path)
                latency = (perf_counter() - started) * 1000
            except Exception as exc:
                scores = np.full(len(adapter.vocabulary.labels), np.nan, dtype=np.float32)
                error = {"type": type(exc).__name__, "message": str(exc)}
            record = {
                "id": sample["id"],
                "row": row,
                "outcome": "ok" if error is None else "error",
                "latency_ms": latency,
                "error": error,
            }
            records.write(canonical_json(record) + "\n")
            records.flush()
            shard_rows.append(scores)
            shard_ids.append(sample["id"])
            shard_valid.append(error is None)
            if len(shard_rows) >= shard_size:
                flush()
        flush()
    versions = {name: importlib.metadata.version(name) for name in ("numpy", "Pillow")}
    versions["vision-bench"] = __version__
    return {
        "pid": os.getpid(),
        "python": sys.version,
        "platform": platform.platform(),
        "model_load_ms": load_ms,
        "libraries": versions,
        "model": model_info,
        "preprocess_version": PREPROCESS_VERSION,
        "preprocess_code_sha256": file_hash(Path(__file__).with_name("preprocess.py")),
    }


def main() -> int:
    run_dir = Path(sys.argv[1]).resolve()
    try:
        runtime = execute(run_dir)
        print(canonical_json({"ok": True, "runtime": runtime}))
        return 0
    except Exception as exc:
        print(canonical_json({"ok": False, "error": {"type": type(exc).__name__, "message": str(exc)}}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
