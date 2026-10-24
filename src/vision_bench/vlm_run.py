"""Sequential VLM experiments with immutable inputs and complete attempt histories."""

from __future__ import annotations

import math
import os
import platform
import time
from pathlib import Path
from uuid import uuid4

from PIL import Image

from . import __version__
from .dataset import load_dataset
from .errors import BenchmarkError
from .io import (
    canonical_json,
    content_hash,
    file_hash,
    identifier,
    positive_int,
    read_object,
    write_json,
    write_jsonl,
)
from .runner import utc_now
from .vlm_contracts import parse_response, task_validator, validate_profile
from .vlm_transport import (
    IMAGE_PREPROCESS_VERSION,
    TransportError,
    chat_request,
    post_chat,
    prepare_image,
    validate_endpoint,
)

RETRYABLE_HTTP = {429, 500, 502, 503, 504}


def run_vlm(
    *,
    dataset_path: Path,
    task_path: Path,
    profile_path: Path,
    endpoint: str,
    output_root: Path,
    run_id: str | None = None,
    limit: int = 8,
    repeats: int = 2,
    retries: int = 0,
    timeout: float = 120,
    api_key_env: str | None = None,
) -> Path:
    positive_int(limit, name="VLM sample limit")
    if positive_int(repeats, name="repeats") > 10:
        raise BenchmarkError("VLM repeats must be at most 10.")
    if isinstance(retries, bool) or not isinstance(retries, int) or not 0 <= retries <= 3:
        raise BenchmarkError("VLM retries must be between 0 and 3.")
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or not math.isfinite(timeout)
        or timeout <= 0
    ):
        raise BenchmarkError("VLM socket timeout must be finite and positive.")
    endpoint = validate_endpoint(endpoint)
    api_key = os.environ.get(api_key_env) if api_key_env else None
    if api_key_env and not api_key:
        raise BenchmarkError("The requested API key environment variable is unset or empty.")
    task, profile = read_object(task_path), read_object(profile_path)
    validator = task_validator(task)
    validate_profile(profile)
    dataset = load_dataset(dataset_path, limit=limit)
    run_id = identifier(run_id or f"vlm-{profile['id'][:70]}-{uuid4().hex[:12]}")
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / run_id
    try:
        run_dir.mkdir()
    except FileExistsError as exc:
        raise BenchmarkError(f"Run already exists; choose a new id: {run_id}") from exc
    manifest = {
        "schema_version": 1,
        "kind": "vlm",
        "state": "running",
        "run_id": run_id,
        "started_at": utc_now(),
        "task_fingerprint": content_hash(task),
        "profile_fingerprint": content_hash(profile),
        "dataset": {
            "id": dataset.manifest["id"],
            "revision": dataset.manifest["revision"],
            "annotation_kind": dataset.manifest["annotation_kind"],
            "fingerprint": dataset.fingerprint,
            "selected_count": len(dataset.samples),
        },
        "repeats": repeats,
        "retries": retries,
        "socket_timeout_seconds": timeout,
        "runtime": {
            "vision_bench": __version__,
            "python": platform.python_version(),
            "endpoint": endpoint,
            "image_preprocess": IMAGE_PREPROCESS_VERSION,
            "image_preprocess_code_sha256": file_hash(Path(__file__).with_name("vlm_transport.py")),
            "model_identity": "profile-declared revision; response model ID checked; server weights not attested",
            "scope": "external HTTP server; no server VRAM or model-load measurements",
        },
    }
    write_json(run_dir / "run.json", manifest)
    try:
        write_json(run_dir / "task.json", task)
        write_json(run_dir / "profile.json", profile)
        write_jsonl(run_dir / "samples.jsonl", list(dataset.samples))
        with (run_dir / "records.jsonl").open("x", encoding="utf-8", newline="\n") as output:
            for row, sample in enumerate(dataset.samples):
                image_info, payload = None, None
                try:
                    image, image_info = prepare_image(
                        dataset.image_path(sample), sample["sha256"], profile["image"]
                    )
                except (
                    OSError,
                    ValueError,
                    BenchmarkError,
                    Image.DecompressionBombWarning,
                    Image.DecompressionBombError,
                ):
                    image_info = None
                else:
                    image_name = f"input-{row:06}.jpg"
                    (run_dir / image_name).write_bytes(image)
                    image_info["path"] = image_name
                    payload = chat_request(task, profile, image)
                for repeat in range(repeats):
                    started = time.perf_counter()
                    attempts = []
                    result = {"outcome": "image_error", "output": None, "reported_model": None, "usage": None}
                    if payload is not None:
                        for attempt in range(retries + 1):
                            request_started = time.perf_counter()
                            body, status, transport_error = b"", None, None
                            try:
                                status, body = post_chat(endpoint, payload, timeout=timeout, api_key=api_key)
                            except TransportError as exc:
                                body, status, transport_error = exc.body, exc.status, exc.kind
                            elapsed = (time.perf_counter() - request_started) * 1000
                            raw_name = f"response-{row:06}-{repeat:02}-{attempt:02}.bin"
                            (run_dir / raw_name).write_bytes(body)
                            if transport_error:
                                result = {
                                    "outcome": transport_error,
                                    "output": None,
                                    "reported_model": None,
                                    "usage": None,
                                }
                            elif status != 200:
                                result = {
                                    "outcome": "http_error",
                                    "output": None,
                                    "reported_model": None,
                                    "usage": None,
                                }
                            else:
                                result = parse_response(body, task, profile["model"], validator)
                            attempts.append(
                                {
                                    "attempt": attempt,
                                    "http_status": status,
                                    "raw_response": raw_name,
                                    "latency_ms": elapsed,
                                    "outcome": result["outcome"],
                                }
                            )
                            retryable = transport_error in {"network_error", "timeout"} or (
                                transport_error is None and status in RETRYABLE_HTTP
                            )
                            if not retryable or attempt == retries:
                                break
                            time.sleep(0.25 * 2**attempt)
                    record = {
                        "row": row,
                        "id": sample["id"],
                        "repeat": repeat,
                        "prepared_image": image_info,
                        "request_sha256": content_hash(payload) if payload else None,
                        "attempts": attempts,
                        "latency_ms": (time.perf_counter() - started) * 1000,
                        **result,
                    }
                    output.write(canonical_json(record) + "\n")
                    output.flush()
        from .vlm_results import compute_vlm_metrics, vlm_inventory

        manifest.update(state="completed", completed_at=utc_now(), artifacts=vlm_inventory(run_dir))
        write_json(run_dir / "run.json", manifest)
        metrics = compute_vlm_metrics(run_dir)
        write_json(run_dir / "metrics.json", metrics)
        manifest["artifacts"] = vlm_inventory(run_dir)
        write_json(run_dir / "run.json", manifest)
        return run_dir
    except BaseException as exc:
        manifest.update(state="failed", completed_at=utc_now(), error={"type": type(exc).__name__})
        write_json(run_dir / "run.json", manifest)
        raise
