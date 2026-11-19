from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .artifacts import load_results
from .bootstrap import CONFIDENCE, resample_weights, summarize, weighted_metrics
from .errors import BenchmarkError
from .io import content_hash, read_object
from .metrics import evaluate_columns
from .sweep import threshold_analysis


def compare_runs(
    paths: list[Path],
    *,
    cache_dir: Path | None = None,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 0,
) -> dict:
    if len(paths) < 2 or len({path.resolve() for path in paths}) != len(paths):
        raise BenchmarkError("Comparison needs at least two distinct runs.")
    for value, name in ((bootstrap_resamples, "resamples"), (bootstrap_seed, "seed")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BenchmarkError(f"Bootstrap {name} must be a non-negative integer.")
    with ExitStack() as stack:
        runs = [stack.enter_context(load_results(path, cache_dir=cache_dir)) for path in paths]
        return _compare(paths, runs, bootstrap_resamples, bootstrap_seed)


def _compare(paths: list[Path], runs: list[tuple], resamples: int, seed: int) -> dict:
    fingerprint = runs[0][0]["dataset"]["fingerprint"]
    if any(run[0]["dataset"]["fingerprint"] != fingerprint for run in runs):
        raise BenchmarkError("Runs use different dataset subsets, annotation versions, or ordering.")
    paired = np.logical_and.reduce([run[4] for run in runs])
    if not paired.any():
        raise BenchmarkError("No shared successful samples.")
    rows = np.flatnonzero(paired)
    weights = resample_weights(len(rows), resamples, seed) if resamples else None
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
        profiles, selections = [], []
        truth = [set(runs[0][1][row]["tags"][category]) for row in rows]
        for profile, run in zip(configs, runs, strict=True):
            lookup = {label: index for index, label in enumerate(run[2]["labels"])}
            columns = [lookup[label] for label in labels]
            threshold = profile["thresholds"][category]
            selections.append((run[3], columns, threshold))
            profiles.append(
                {
                    "run_id": run[0]["run_id"],
                    "profile_id": profile["id"],
                    "successful": int(run[4].sum()),
                    "failed": int((~run[4]).sum()),
                    "metrics": evaluate_columns(run[3], rows, columns, labels, truth, threshold),
                    "threshold_analysis": threshold_analysis(run[3], rows, columns, labels, truth),
                }
            )
        categories[category] = {
            "label_count": len(labels),
            "vocabulary_sha256": content_hash(labels),
            "score_scope": "common_outputs_only; full_truth metrics are not native full-vocabulary metrics",
            "runs": profiles,
        }
        if weights is not None:
            statistics = weighted_metrics(selections, rows, labels, truth, weights)
            categories[category]["bootstrap"] = summarize(
                [profile["run_id"] for profile in profiles], statistics
            )
    if not categories:
        raise BenchmarkError("No category has both common vocabulary and thresholds for every model.")
    return {
        "schema_version": 1,
        "dataset_fingerprint": fingerprint,
        "attempted_per_model": len(paired),
        "paired_successful": int(paired.sum()),
        "categories": categories,
        "excluded_categories": excluded_categories,
        "bootstrap": {
            "method": "paired_percentile",
            "unit": "paired successful sample",
            "resamples": resamples,
            "seed": seed,
            "confidence": CONFIDENCE,
            "reference_run_id": runs[0][0]["run_id"],
            "scope": "sampling variation of this image selection only; labels, thresholds and models are fixed",
        }
        if resamples
        else {"status": "disabled"},
        "scope": "category-consistent common vocabulary, paired successful samples",
    }
