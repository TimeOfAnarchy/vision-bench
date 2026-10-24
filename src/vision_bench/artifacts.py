from __future__ import annotations

from pathlib import Path

import numpy as np

from .dataset import NORMALIZATION_VERSION, normalized_sample
from .errors import BenchmarkError
from .io import child_path, content_hash, file_hash, read_jsonl, read_object
from .metrics import evaluate_scores, performance_summary


def artifact_metadata(run_dir: Path) -> list[dict]:
    return [
        {"path": path.name, "sha256": file_hash(path), "bytes": path.stat().st_size}
        for path in sorted(run_dir.iterdir())
        if path.is_file() and path.name not in {"run.json", "worker.log"}
    ]


def load_results(
    run_dir: Path, manifest: dict | None = None
) -> tuple[dict, list[dict], dict, np.ndarray, np.ndarray, list[dict]]:
    run_dir = run_dir.resolve()
    manifest = manifest if manifest is not None else read_object(run_dir / "run.json")
    if manifest.get("schema_version") != 1 or manifest.get("state") != "completed":
        raise BenchmarkError("Only completed version-1 runs can be evaluated.")
    artifacts = manifest.get("artifacts", [])
    names = [item["path"] for item in artifacts]
    if len(names) != len(set(names)) or not {
        "samples.jsonl",
        "profile.json",
        "vocabulary.json",
        "records.jsonl",
    } <= set(names):
        raise BenchmarkError("Run artifact inventory is incomplete or duplicated.")
    for item in artifacts:
        path = child_path(run_dir, item["path"])
        if path.stat().st_size != item["bytes"] or file_hash(path) != item["sha256"]:
            raise BenchmarkError(f"Run artifact integrity check failed: {item['path']}")
    samples = [normalized_sample(value) for value in read_jsonl(run_dir / "samples.jsonl")]
    profile = read_object(run_dir / "profile.json")
    expected_fingerprint = content_hash(
        {
            "dataset_id": manifest["dataset"]["id"],
            "revision": manifest["dataset"]["revision"],
            "annotation_kind": manifest["dataset"]["annotation_kind"],
            "normalization": NORMALIZATION_VERSION,
            "samples": samples,
        }
    )
    if (
        expected_fingerprint != manifest["dataset"]["fingerprint"]
        or manifest["normalization"] != NORMALIZATION_VERSION
        or content_hash(profile) != manifest["profile_fingerprint"]
        or profile["id"] != manifest["profile_id"]
    ):
        raise BenchmarkError("Run dataset/profile metadata does not match its saved inputs.")
    vocabulary = read_object(run_dir / "vocabulary.json")
    labels, categories = vocabulary["labels"], vocabulary["categories"]
    named_labels = [label for label in labels if label is not None]
    if (
        not samples
        or not named_labels
        or len(labels) != len(categories)
        or len(set(named_labels)) != len(named_labels)
        or any(
            (label is None) != (category == "unnamed")
            for label, category in zip(labels, categories, strict=True)
        )
    ):
        raise BenchmarkError("Invalid saved samples or vocabulary.")
    matrices, masks, ids = [], [], []
    shards = sorted(name for name in names if name.startswith("scores-") and name.endswith(".npz"))
    if not shards:
        raise BenchmarkError("Run has no score shards.")
    for name in shards:
        with np.load(child_path(run_dir, name), allow_pickle=False) as payload:
            matrix = payload["scores"]
            valid = payload["valid"]
            sample_ids = payload["sample_ids"]
            if (
                matrix.dtype != np.float32
                or matrix.ndim != 2
                or matrix.shape[1] != len(labels)
                or valid.dtype != np.bool_
                or valid.shape != (matrix.shape[0],)
                or sample_ids.shape != valid.shape
                or sample_ids.dtype.kind != "U"
            ):
                raise BenchmarkError("Invalid score shard array contract.")
            if (
                not np.isfinite(matrix[valid]).all()
                or np.any((matrix[valid] < 0) | (matrix[valid] > 1))
                or not np.isnan(matrix[~valid]).all()
            ):
                raise BenchmarkError("Invalid score values or failed-row mask.")
            matrices.append(matrix)
            masks.append(valid)
            ids.extend(sample_ids.tolist())
    if ids != [sample["id"] for sample in samples] or len(set(ids)) != len(ids):
        raise BenchmarkError("Score sample order/identity differs from saved dataset.")
    records = read_jsonl(run_dir / "records.jsonl")
    valid = np.concatenate(masks)
    if len(records) != len(samples) or any(
        record.get("row") != row
        or record.get("id") != ids[row]
        or (record.get("outcome") == "ok") != bool(valid[row])
        for row, record in enumerate(records)
    ):
        raise BenchmarkError("Prediction records and scores disagree.")
    return manifest, samples, vocabulary, np.concatenate(matrices), valid, records


def compute_metrics(run_dir: Path, manifest: dict | None = None) -> dict:
    manifest, samples, vocabulary, scores, valid, records = load_results(run_dir, manifest)
    profile = read_object(run_dir / "profile.json")
    categories = {}
    selected = [sample for sample, keep in zip(samples, valid, strict=True) if keep]
    for category in sorted(samples[0]["tags"]):
        columns = [
            index
            for index, value in enumerate(vocabulary["categories"])
            if value == category and vocabulary["labels"][index] is not None
        ]
        labels = [vocabulary["labels"][index] for index in columns]
        if category not in profile["thresholds"]:
            continue
        truth = [set(sample["tags"][category]) for sample in selected]
        categories[category] = evaluate_scores(
            scores[valid][:, columns], labels, truth, profile["thresholds"][category]
        )
    return {
        "schema_version": 1,
        "metrics_version": 1,
        "profile_id": profile["id"],
        "model_outputs": {
            "total": len(vocabulary["labels"]),
            "named": sum(label is not None for label in vocabulary["labels"]),
            "unnamed": vocabulary["labels"].count(None),
        },
        "dataset_fingerprint": manifest["dataset"]["fingerprint"],
        "annotation_kind": manifest["dataset"]["annotation_kind"],
        "outcomes": {
            "attempted": len(samples),
            "successful": int(valid.sum()),
            "failed": int((~valid).sum()),
        },
        "categories": categories,
        "performance": performance_summary(records, manifest["warmup"]),
        "model_load_ms": manifest["runtime"]["model_load_ms"],
    }
