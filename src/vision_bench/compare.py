from __future__ import annotations

from pathlib import Path

import numpy as np

from .artifacts import load_results
from .errors import BenchmarkError
from .io import content_hash, read_object
from .metrics import evaluate_scores


def compare_runs(paths: list[Path]) -> dict:
    if len(paths) < 2 or len({path.resolve() for path in paths}) != len(paths):
        raise BenchmarkError("Comparison needs at least two distinct runs.")
    runs = [load_results(path) for path in paths]
    fingerprint = runs[0][0]["dataset"]["fingerprint"]
    if any(run[0]["dataset"]["fingerprint"] != fingerprint for run in runs):
        raise BenchmarkError("Runs use different dataset subsets, annotation versions, or ordering.")
    paired = np.logical_and.reduce([run[4] for run in runs])
    if not paired.any():
        raise BenchmarkError("No shared successful samples.")
    categories, excluded_categories = {}, {}
    configs = [read_object(path / "profile.json") for path in paths]
    for category in sorted(runs[0][1][0]["tags"]):
        missing = [config["id"] for config in configs if category not in config["thresholds"]]
        if missing:
            excluded_categories[category] = {"reason": "missing_threshold", "profiles": missing}
            continue
        vocabularies = [
            {
                label
                for label, kind in zip(run[2]["labels"], run[2]["categories"], strict=True)
                if kind == category and label is not None
            }
            for run in runs
        ]
        labels = sorted(set.intersection(*vocabularies))
        if not labels:
            excluded_categories[category] = {"reason": "no_common_vocabulary"}
            continue
        profiles = []
        for profile, run in zip(configs, runs, strict=True):
            lookup = {label: index for index, label in enumerate(run[2]["labels"])}
            truth = [
                set(sample["tags"][category]) for sample, keep in zip(run[1], paired, strict=True) if keep
            ]
            values = evaluate_scores(
                run[3][paired][:, [lookup[label] for label in labels]],
                labels,
                truth,
                profile["thresholds"][category],
            )
            profiles.append(
                {
                    "run_id": run[0]["run_id"],
                    "profile_id": profile["id"],
                    "successful": int(run[4].sum()),
                    "failed": int((~run[4]).sum()),
                    "metrics": values,
                }
            )
        categories[category] = {
            "label_count": len(labels),
            "vocabulary_sha256": content_hash(labels),
            "score_scope": "common_outputs_only; full_truth metrics are not native full-vocabulary metrics",
            "runs": profiles,
        }
    if not categories:
        raise BenchmarkError("No category has both common vocabulary and thresholds for every model.")
    return {
        "schema_version": 1,
        "dataset_fingerprint": fingerprint,
        "attempted_per_model": len(paired),
        "paired_successful": int(paired.sum()),
        "categories": categories,
        "excluded_categories": excluded_categories,
        "scope": "category-consistent common vocabulary, paired successful samples",
    }
