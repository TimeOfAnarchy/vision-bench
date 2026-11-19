from __future__ import annotations

import shutil
from contextlib import ExitStack
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .artifacts import load_results
from .errors import BenchmarkError
from .io import read_object
from .metrics import evaluate_columns, ratio
from .thresholds import Thresholds


def compare_precision(baseline: Path, candidate: Path, *, cache_dir: Path | None = None) -> dict:
    if baseline.resolve() == candidate.resolve():
        raise BenchmarkError("Precision comparison needs two distinct runs.")
    with ExitStack() as stack:
        left = stack.enter_context(load_results(baseline, cache_dir=cache_dir))
        right = stack.enter_context(load_results(candidate, cache_dir=cache_dir))
        configs = [read_object(path / "profile.json") for path in (baseline, candidate)]
        if left[0]["dataset"]["fingerprint"] != right[0]["dataset"]["fingerprint"]:
            raise BenchmarkError("Precision runs use different dataset subsets or annotations.")
        if left[2]["labels"] != right[2]["labels"] or left[2]["categories"] != right[2]["categories"]:
            raise BenchmarkError("Precision comparison requires identical indexed vocabularies.")
        source = tuple(configs[0].get("source", {}).get(key) for key in ("repository", "revision"))
        if not all(isinstance(value, str) and value for value in source) or source != tuple(
            configs[1].get("source", {}).get(key) for key in ("repository", "revision")
        ):
            raise BenchmarkError("Precision runs must pin the same upstream model repository and revision.")
        for key in ("preprocess", "provider", "thresholds"):
            if configs[0][key] != configs[1][key]:
                raise BenchmarkError(f"Precision comparison requires the same {key}.")
        if configs[0].get("tag_thresholds", {}) != configs[1].get("tag_thresholds", {}):
            raise BenchmarkError("Precision comparison requires the same tag thresholds.")
        policies = {
            category: Thresholds.for_category(configs[0], category) for category in configs[0]["thresholds"]
        }
        if left[0]["runtime"]["preprocess_code_sha256"] != right[0]["runtime"]["preprocess_code_sha256"]:
            raise BenchmarkError("Precision runs use different preprocessing implementations.")
        if not np.array_equal(left[4], right[4]):
            raise BenchmarkError("Precision comparison requires identical successful sample masks.")
        rows = np.flatnonzero(left[4])
        if not len(rows):
            raise BenchmarkError("Precision runs have no paired successful samples.")
        labels, categories = left[2]["labels"], left[2]["categories"]
        count = len(rows) * len(labels)
        changed_images = np.zeros(len(rows), dtype=np.bool_)
        flips_by_label = np.zeros(len(labels), dtype=np.int64)
        maximum, total, maximum_at = 0.0, 0.0, None
        with TemporaryDirectory(prefix="vision-bench-errors-", dir=cache_dir) as temp:
            if shutil.disk_usage(temp).free < count * 4:
                raise BenchmarkError("Not enough temporary disk space for exact error percentiles.")
            errors = np.memmap(
                Path(temp) / "absolute-errors.bin", mode="w+", dtype=np.float32, shape=(count,)
            )
            try:
                offset = 0
                for start in range(0, len(labels), 128):
                    stop = min(start + 128, len(labels))
                    columns = list(range(start, stop))
                    a = left[3][np.ix_(rows, columns)]
                    b = right[3][np.ix_(rows, columns)]
                    delta = np.abs(a - b)
                    errors[offset : offset + delta.size] = delta.ravel()
                    offset += delta.size
                    total += float(delta.sum(dtype=np.float64))
                    index = np.unravel_index(int(np.argmax(delta)), delta.shape)
                    if float(delta[index]) > maximum:
                        maximum = float(delta[index])
                        maximum_at = {
                            "sample_id": left[1][int(rows[index[0]])]["id"],
                            "output_index": start + int(index[1]),
                            "label": labels[start + int(index[1])],
                            "baseline": float(a[index]),
                            "candidate": float(b[index]),
                        }
                    for local, column in enumerate(columns):
                        category = categories[column]
                        if labels[column] is None or category not in policies:
                            continue
                        policy = policies[category]
                        threshold = policy.overrides.get(labels[column], policy.default)
                        flips = (a[:, local] >= threshold) != (b[:, local] >= threshold)
                        flips_by_label[column] = int(flips.sum())
                        changed_images |= flips
                quantiles = np.percentile(errors, [50, 95, 99], overwrite_input=True, method="linear")
            finally:
                errors._mmap.close()
        threshold_changes = {}
        for category in sorted(configs[0]["thresholds"]):
            indices = [i for i, kind in enumerate(categories) if kind == category and labels[i] is not None]
            flips = int(flips_by_label[indices].sum())
            threshold_changes[category] = {
                **policies[category].describe([labels[index] for index in indices]),
                "decisions": len(rows) * len(indices),
                "flips": flips,
                "flip_fraction": ratio(flips, len(rows) * len(indices)),
            }
        quality = {}
        for category in sorted(left[1][0]["tags"]):
            if category not in configs[0]["thresholds"]:
                continue
            columns = [i for i, kind in enumerate(categories) if kind == category and labels[i] is not None]
            names = [labels[i] for i in columns]
            truth = [set(left[1][int(row)]["tags"][category]) for row in rows]
            values = [
                evaluate_columns(data[3], rows, columns, names, truth, policies[category])
                for data in (left, right)
            ]
            quality[category] = {
                role: {key: value for key, value in metrics.items() if key != "per_label"}
                for role, metrics in zip(("baseline", "candidate"), values, strict=True)
            }
        return {
            "schema_version": 1,
            "kind": "precision_pair",
            "baseline_run": left[0]["run_id"],
            "candidate_run": right[0]["run_id"],
            "dataset_fingerprint": left[0]["dataset"]["fingerprint"],
            "provider": configs[0]["provider"],
            "source": {"repository": source[0], "revision": source[1]},
            "baseline_precision": configs[0].get("source", {}).get("precision"),
            "candidate_precision": configs[1].get("source", {}).get("precision"),
            "paired_successful": len(rows),
            "baseline_failed": int((~left[4]).sum()),
            "candidate_failed": int((~right[4]).sum()),
            "compared_scores": count,
            "max_absolute_error": maximum,
            "mean_absolute_error": total / count,
            "p50_absolute_error": float(quantiles[0]),
            "p95_absolute_error": float(quantiles[1]),
            "p99_absolute_error": float(quantiles[2]),
            "maximum_error_location": maximum_at,
            "changed_threshold_samples": int(changed_images.sum()),
            "threshold_changes": threshold_changes,
            "quality": quality,
            "scope": "paired probabilities; identical source, vocabulary, preprocessing, provider and operating thresholds",
        }
