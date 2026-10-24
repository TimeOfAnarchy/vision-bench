"""Offline VLM verification, repeat stability, run comparison and review exports."""

from __future__ import annotations

import itertools
import math
from collections import Counter
from pathlib import Path

import numpy as np

from .dataset import NORMALIZATION_VERSION, normalized_sample
from .errors import BenchmarkError
from .io import canonical_json, child_path, content_hash, file_hash, read_jsonl, read_object
from .vlm_contracts import parse_response, projection, task_validator, validate_profile
from .vlm_transport import MAX_RESPONSE_BYTES, chat_request


def vlm_inventory(run_dir: Path) -> list[dict]:
    return [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": file_hash(path)}
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path.name != "run.json"
    ]


def load_vlm_run(run_dir: Path) -> tuple[dict, dict, dict, list[dict], list[dict]]:
    manifest = read_object(run_dir / "run.json")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("kind") != "vlm"
        or manifest.get("state") != "completed"
    ):
        raise BenchmarkError("Only completed version-1 VLM runs can be evaluated.")
    artifacts = manifest.get("artifacts", [])
    names = [item["path"] for item in artifacts]
    if len(set(names)) != len(names) or not {
        "task.json",
        "profile.json",
        "samples.jsonl",
        "records.jsonl",
    } <= set(names):
        raise BenchmarkError("VLM artifact inventory is incomplete or duplicated.")
    for item in artifacts:
        path = child_path(run_dir, item["path"])
        if path.stat().st_size != item["bytes"] or file_hash(path) != item["sha256"]:
            raise BenchmarkError("VLM artifact integrity check failed.")
    task, profile = read_object(run_dir / "task.json"), read_object(run_dir / "profile.json")
    validator = task_validator(task)
    validate_profile(profile)
    samples = [normalized_sample(item) for item in read_jsonl(run_dir / "samples.jsonl")]
    dataset = manifest["dataset"]
    if (
        not samples
        or len({x["id"] for x in samples}) != len(samples)
        or len(samples) != dataset["selected_count"]
        or any(set(x["tags"]) != set(samples[0]["tags"]) for x in samples)
        or content_hash(
            {
                "dataset_id": dataset["id"],
                "revision": dataset["revision"],
                "annotation_kind": dataset["annotation_kind"],
                "normalization": NORMALIZATION_VERSION,
                "samples": samples,
            }
        )
        != dataset["fingerprint"]
        or content_hash(task) != manifest["task_fingerprint"]
        or content_hash(profile) != manifest["profile_fingerprint"]
    ):
        raise BenchmarkError("VLM saved inputs disagree with their fingerprints.")
    repeats = manifest["repeats"]
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 1 <= repeats <= 10:
        raise BenchmarkError("Invalid recorded VLM repeats.")
    retries = manifest["retries"]
    if isinstance(retries, bool) or not isinstance(retries, int) or not 0 <= retries <= 3:
        raise BenchmarkError("Invalid recorded VLM retry budget.")
    records = read_jsonl(run_dir / "records.jsonl")
    if len(records) != len(samples) * repeats:
        raise BenchmarkError("VLM prediction grid is incomplete.")
    for index, record in enumerate(records):
        row, repeat = divmod(index, repeats)
        if (
            record.get("row") != row
            or record.get("repeat") != repeat
            or record.get("id") != samples[row]["id"]
            or not _latency(record.get("latency_ms"))
        ):
            raise BenchmarkError("VLM prediction order or timing is invalid.")
        attempts = record.get("attempts")
        if not isinstance(attempts, list) or len(attempts) > retries + 1:
            raise BenchmarkError("Invalid VLM attempt history.")
        if not attempts:
            if (
                record["outcome"] != "image_error"
                or record["output"] is not None
                or record["prepared_image"] is not None
                or record.get("request_sha256") is not None
                or record.get("reported_model") is not None
                or record.get("usage") is not None
            ):
                raise BenchmarkError("VLM missing attempts must represent image errors.")
            continue
        image = record["prepared_image"]
        expected_image = f"input-{row:06}.jpg"
        if (
            image["path"] != expected_image
            or expected_image not in names
            or file_hash(child_path(run_dir, expected_image)) != image["sha256"]
        ):
            raise BenchmarkError("Prepared VLM image is missing or changed.")
        prepared = child_path(run_dir, expected_image)
        if prepared.stat().st_size != image["bytes"] or content_hash(
            chat_request(task, profile, prepared.read_bytes())
        ) != record.get("request_sha256"):
            raise BenchmarkError("VLM request does not match its saved image, task and profile.")
        for number, attempt in enumerate(attempts):
            raw_name = f"response-{row:06}-{repeat:02}-{number:02}.bin"
            if (
                attempt.get("attempt") != number
                or attempt.get("raw_response") != raw_name
                or raw_name not in names
                or not _latency(attempt.get("latency_ms"))
            ):
                raise BenchmarkError("Invalid VLM attempt identity or timing.")
            path = child_path(run_dir, raw_name)
            if path.stat().st_size > MAX_RESPONSE_BYTES:
                raise BenchmarkError("Recorded VLM response exceeds the size bound.")
            status = attempt.get("http_status")
            if attempt["outcome"] in {"network_error", "timeout"}:
                if status is not None:
                    raise BenchmarkError("Network failures cannot declare an HTTP status.")
            elif isinstance(status, bool) or not isinstance(status, int) or not 100 <= status <= 599:
                raise BenchmarkError("Invalid recorded HTTP status.")
            if attempt["outcome"] in {"network_error", "timeout", "response_too_large"}:
                parsed = {
                    "outcome": attempt["outcome"],
                    "output": None,
                    "reported_model": None,
                    "usage": None,
                }
            elif attempt.get("http_status") != 200:
                parsed = {"outcome": "http_error", "output": None, "reported_model": None, "usage": None}
            else:
                parsed = parse_response(path.read_bytes(), task, profile["model"], validator)
            if parsed["outcome"] != attempt["outcome"]:
                raise BenchmarkError("VLM attempt does not match its raw response.")
            if number < len(attempts) - 1 and not (
                parsed["outcome"] in {"network_error", "timeout"}
                or parsed["outcome"] == "http_error"
                and status in {429, 500, 502, 503, 504}
            ):
                raise BenchmarkError("VLM history retries a non-retryable outcome.")
        if any(record.get(key) != value for key, value in parsed.items()):
            raise BenchmarkError("VLM final prediction does not match its raw response.")
    return manifest, task, profile, samples, records


def _latency(value) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
    )


def _fraction(numerator: int, denominator: int):
    return numerator / denominator if denominator else None


def _metrics(data: tuple) -> dict:
    manifest, task, profile, samples, records = data
    successes = [record for record in records if record["outcome"] == "ok"]
    latency = [record["latency_ms"] for record in successes]
    eligible, stable, pairs, equal = 0, 0, 0, 0
    for row in range(len(samples)):
        group = records[row * manifest["repeats"] : (row + 1) * manifest["repeats"]]
        values = [projection(record["output"], task) for record in group if record["outcome"] == "ok"]
        if len(values) == len(group) and len(values) >= 2:
            eligible += 1
            stable += len(set(values)) == 1
        for a, b in itertools.combinations(values, 2):
            pairs += 1
            equal += a == b
    planned_pairs = len(samples) * math.comb(manifest["repeats"], 2)
    return {
        "schema_version": 1,
        "kind": "vlm_metrics",
        "run_id": manifest["run_id"],
        "task_fingerprint": manifest["task_fingerprint"],
        "profile_id": profile["id"],
        "samples": len(samples),
        "repeats": manifest["repeats"],
        "predictions": len(records),
        "outcomes": dict(sorted(Counter(record["outcome"] for record in records).items())),
        "attempt_outcomes": dict(
            sorted(
                Counter(attempt["outcome"] for record in records for attempt in record["attempts"]).items()
            )
        ),
        "attempts": sum(len(record["attempts"]) for record in records),
        "retried_predictions": sum(len(record["attempts"]) > 1 for record in records),
        "first_attempt_successes": sum(
            bool(record["attempts"]) and record["attempts"][0]["outcome"] == "ok" for record in records
        ),
        "valid_output_fraction": _fraction(len(successes), len(records)),
        "latency": {
            "scope": "successful prediction including retries/backoff; excludes image preparation and server loading",
            "median_ms": float(np.median(latency)) if latency else None,
            "p95_ms": float(np.percentile(latency, 95)) if latency else None,
        },
        "stability": {
            "comparison_fields": task["comparison_fields"],
            "eligible_samples": eligible,
            "stable_samples": stable,
            "stable_sample_fraction": _fraction(stable, eligible),
            "planned_pairs": planned_pairs,
            "successful_pairs": pairs,
            "equal_pairs": equal,
            "pair_coverage": _fraction(pairs, planned_pairs),
            "agreement_on_successful_pairs": _fraction(equal, pairs),
        },
        "scope": "schema validity and exact field agreement; no application policy or accuracy inference",
    }


def compute_vlm_metrics(run_dir: Path) -> dict:
    return _metrics(load_vlm_run(run_dir))


def compare_vlm_runs(paths: list[Path]) -> dict:
    if len(paths) < 2 or len(set(path.resolve() for path in paths)) != len(paths):
        raise BenchmarkError("VLM comparison needs at least two distinct runs.")
    runs = [load_vlm_run(path) for path in paths]
    first = runs[0][0]
    for manifest, _, _, _, _ in runs[1:]:
        if (
            manifest["dataset"]["fingerprint"] != first["dataset"]["fingerprint"]
            or manifest["task_fingerprint"] != first["task_fingerprint"]
            or manifest["repeats"] != first["repeats"]
        ):
            raise BenchmarkError("VLM comparison requires identical task, dataset ordering and repeats.")
    indices = [
        index for index in range(len(runs[0][4])) if all(run[4][index]["outcome"] == "ok" for run in runs)
    ]
    agree = sum(len({projection(run[4][index]["output"], run[1]) for run in runs}) == 1 for index in indices)
    return {
        "schema_version": 1,
        "kind": "vlm_comparison",
        "task_fingerprint": first["task_fingerprint"],
        "dataset_fingerprint": first["dataset"]["fingerprint"],
        "planned_predictions": len(runs[0][4]),
        "paired_successful": len(indices),
        "equal_predictions": agree,
        "agreement_on_paired_successes": _fraction(agree, len(indices)),
        "runs": [_metrics(run) for run in runs],
        "scope": "aligned sample/repeat positions; exact comparison fields, conditioned on jointly valid responses",
    }


def export_vlm_review(paths: list[Path], output: Path) -> int:
    if not paths or len({path.resolve() for path in paths}) != len(paths):
        raise BenchmarkError("Review export needs distinct VLM runs.")
    if len(paths) > 1:
        compare_vlm_runs(paths)  # Verify alignment before constructing a review packet.
    runs = [load_vlm_run(path) for path in paths]
    rows = []
    for row, sample in enumerate(runs[0][3]):
        predictions = []
        values = set()
        for manifest, task, profile, _, records in runs:
            for record in records[row * manifest["repeats"] : (row + 1) * manifest["repeats"]]:
                if record["outcome"] == "ok":
                    values.add(projection(record["output"], task))
                predictions.append(
                    {
                        "run_id": manifest["run_id"],
                        "profile_id": profile["id"],
                        "repeat": record["repeat"],
                        "outcome": record["outcome"],
                        "proposed_output": record["output"],
                        "prepared_image": record["prepared_image"],
                        "raw_responses": [attempt["raw_response"] for attempt in record["attempts"]],
                    }
                )
        rows.append(
            {
                "schema_version": 1,
                "id": sample["id"],
                "image": sample["image"],
                "sha256": sample["sha256"],
                "task_fingerprint": runs[0][0]["task_fingerprint"],
                "disagreement": len(values) > 1,
                "has_errors": any(item["outcome"] != "ok" for item in predictions),
                "predictions": predictions,
                "human_review": {"reviewer": None, "accepted": None, "corrected_output": None, "notes": ""},
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    return len(rows)
