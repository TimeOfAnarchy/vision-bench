from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .artifacts import load_results
from .bootstrap import CONFIDENCE, resample_weights, summarize, weighted_metrics
from .errors import BenchmarkError
from .io import content_hash, read_object
from .metrics import evaluate_columns, support_strata
from .per_image import per_image_counts, per_image_summary, sample_metrics
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


def paired_rows(runs: list[tuple]) -> np.ndarray:
    """Rows that every run completed, after checking that all runs share one dataset selection."""
    fingerprint = runs[0][0]["dataset"]["fingerprint"]
    if any(run[0]["dataset"]["fingerprint"] != fingerprint for run in runs):
        raise BenchmarkError("Runs use different dataset subsets, annotation versions, or ordering.")
    paired = np.logical_and.reduce([run[4] for run in runs])
    if not paired.any():
        raise BenchmarkError("No shared successful samples.")
    return np.flatnonzero(paired)


def category_selection(
    runs: list[tuple], configs: list[dict], category: str
) -> tuple[list[str], list[tuple[np.ndarray, list[int], float]], dict | None]:
    """Common category vocabulary and per-run (scores, columns, threshold), or why it is excluded."""
    missing = [config["id"] for config in configs if category not in config["thresholds"]]
    if missing:
        return [], [], {"reason": "missing_threshold", "profiles": missing}
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
        return [], [], {"reason": "no_common_vocabulary"}
    selections = []
    for config, run in zip(configs, runs, strict=True):
        lookup = {label: index for index, label in enumerate(run[2]["labels"])}
        selections.append((run[3], [lookup[label] for label in labels], config["thresholds"][category]))
    return labels, selections, None


def _compare(paths: list[Path], runs: list[tuple], resamples: int, seed: int) -> dict:
    rows = paired_rows(runs)
    weights = resample_weights(len(rows), resamples, seed) if resamples else None
    categories, excluded_categories = {}, {}
    configs = [read_object(path / "profile.json") for path in paths]
    for category in sorted(runs[0][1][0]["tags"]):
        labels, selections, exclusion = category_selection(runs, configs, category)
        if exclusion:
            excluded_categories[category] = exclusion
            continue
        profiles, image_counts = [], []
        truth = [set(runs[0][1][row]["tags"][category]) for row in rows]
        for profile, run, (scores, columns, threshold) in zip(configs, runs, selections, strict=True):
            image_counts.append(per_image_counts(scores, rows, columns, labels, truth, threshold))
            metrics = evaluate_columns(scores, rows, columns, labels, truth, threshold)
            profiles.append(
                {
                    "run_id": run[0]["run_id"],
                    "profile_id": profile["id"],
                    "successful": int(run[4].sum()),
                    "failed": int((~run[4]).sum()),
                    "metrics": metrics,
                    "by_support": support_strata(metrics["per_label"]),
                    "threshold_analysis": threshold_analysis(scores, rows, columns, labels, truth),
                    "per_image": per_image_summary(image_counts[-1], threshold),
                }
            )
        categories[category] = {
            "label_count": len(labels),
            "vocabulary_sha256": content_hash(labels),
            "score_scope": "common_outputs_only; full_truth metrics are not native full-vocabulary metrics",
            "runs": profiles,
        }
        if weights is not None:
            statistics = [
                {**values, **sample_metrics(counts, weights)}
                for values, counts in zip(
                    weighted_metrics(selections, rows, labels, truth, weights), image_counts, strict=True
                )
            ]
            categories[category]["bootstrap"] = summarize(
                [profile["run_id"] for profile in profiles], statistics
            )
    if not categories:
        raise BenchmarkError("No category has both common vocabulary and thresholds for every model.")
    return {
        "schema_version": 1,
        "dataset_fingerprint": runs[0][0]["dataset"]["fingerprint"],
        "attempted_per_model": len(runs[0][4]),
        "paired_successful": len(rows),
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
