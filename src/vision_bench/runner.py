from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .adapters import profile_vocabulary
from .artifacts import artifact_metadata, compute_metrics
from .dataset import NORMALIZATION_VERSION, load_dataset
from .errors import BenchmarkError
from .io import identifier, positive_int, read_object, write_json, write_jsonl
from .profiles import load_profile
from .report import write_report
from .thresholds import verify_calibration_split
from .worker_process import execute_worker


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_benchmark(
    *,
    dataset_path: Path,
    profile_path: str | Path,
    model_root: Path,
    output_root: Path,
    provider: str | None = None,
    run_id: str | None = None,
    limit: int | None = None,
    shard_size: int = 16,
    warmup: int = 5,
    timeout: float = 600,
    measure_memory: bool = False,
    memory_interval_ms: int = 250,
    cache_dir: Path | None = None,
) -> Path:
    positive_int(shard_size, name="shard_size")
    if (
        isinstance(warmup, bool)
        or not isinstance(warmup, int)
        or warmup < 0
        or not 0 < timeout < float("inf")
    ):
        raise BenchmarkError("warmup must be nonnegative and timeout must be finite and positive.")
    dataset = load_dataset(dataset_path, limit=limit)
    profile = load_profile(profile_path, model_root=model_root, provider=provider)
    verify_calibration_split(profile.config, dataset.samples)
    profile.verify()
    vocabulary = profile_vocabulary(profile)
    if (
        isinstance(memory_interval_ms, bool)
        or not isinstance(memory_interval_ms, int)
        or memory_interval_ms < 250
    ):
        raise BenchmarkError("Memory interval must be at least 250 ms.")
    if shard_size * len(vocabulary.labels) * 4 > 64 * 1024 * 1024:
        raise BenchmarkError("One recording shard exceeds 64 MiB; reduce shard-size.")
    run_id = identifier(
        run_id or f"{profile.profile_id[:80]}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
    )
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / run_id
    try:
        run_dir.mkdir()
    except FileExistsError as exc:
        raise BenchmarkError(f"Run already exists; choose a new id: {run_id}") from exc
    manifest = {
        "schema_version": 1,
        "state": "running",
        "run_id": run_id,
        "started_at": utc_now(),
        "profile_id": profile.profile_id,
        "profile_fingerprint": profile.fingerprint,
        "warmup": warmup,
        "score_storage": "full-vocabulary-float32-shards",
        "normalization": NORMALIZATION_VERSION,
        "dataset": {
            "id": dataset.manifest["id"],
            "revision": dataset.manifest["revision"],
            "fingerprint": dataset.fingerprint,
            "annotation_kind": dataset.manifest["annotation_kind"],
            "manifest_sha256": dataset.manifest_sha256,
            "samples_sha256": dataset.samples_sha256,
            "selected_count": len(dataset.samples),
            "source": dataset.manifest.get("source"),
        },
    }
    write_json(run_dir / "run.json", manifest)
    try:
        write_json(run_dir / "profile.json", profile.config)
        write_jsonl(run_dir / "samples.jsonl", list(dataset.samples))
        write_json(
            run_dir / "request.json",
            {
                "model_root": str(profile.model_root),
                "image_root": str(dataset.image_root),
                "shard_size": shard_size,
                "start_token": uuid4().hex,
            },
        )
        runtime = execute_worker(
            run_dir, timeout=timeout, measure_memory=measure_memory, interval_ms=memory_interval_ms
        )
        manifest.update(state="completed", runtime=runtime)
        manifest["artifacts"] = artifact_metadata(run_dir)
        metrics = compute_metrics(run_dir, manifest, cache_dir=cache_dir)
        if metrics["outcomes"]["successful"] == 0:
            raise BenchmarkError("No successful predictions; inspect records.jsonl.")
        write_json(run_dir / "metrics.json", metrics)
        write_report(run_dir / "report.html", metrics)
        manifest.update(completed_at=utc_now(), artifacts=artifact_metadata(run_dir))
        write_json(run_dir / "run.json", manifest)
        return run_dir
    except BaseException as exc:
        manifest.update(
            state="failed", completed_at=utc_now(), error={"type": type(exc).__name__, "message": str(exc)}
        )
        write_json(run_dir / "run.json", manifest)
        raise


def recompute(run_dir: Path, *, cache_dir: Path | None = None) -> tuple[dict, bool]:
    metrics = compute_metrics(run_dir, cache_dir=cache_dir)
    return metrics, metrics == read_object(run_dir / "metrics.json")
