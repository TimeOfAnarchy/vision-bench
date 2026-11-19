from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .artifacts import load_results
from .bootstrap import CONFIDENCE, resample_weights, summarize, weighted_metrics
from .errors import BenchmarkError
from .io import content_hash, read_object
from .metrics import evaluate_columns, ratio, support_strata
from .per_image import per_image_counts, per_image_summary, sample_metrics
from .score_blocks import MappedScores, describe_thresholds
from .sweep import threshold_analysis
from .tag_mapping import TagMapping, TagRules
from .thresholds import Thresholds


def compare_runs(
    paths: list[Path],
    *,
    cache_dir: Path | None = None,
    bootstrap_resamples: int = 1000,
    bootstrap_seed: int = 0,
    tag_mapping: TagMapping | None = None,
) -> dict:
    if len(paths) < 2 or len({path.resolve() for path in paths}) != len(paths):
        raise BenchmarkError("Comparison needs at least two distinct runs.")
    for value, name in ((bootstrap_resamples, "resamples"), (bootstrap_seed, "seed")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise BenchmarkError(f"Bootstrap {name} must be a non-negative integer.")
    with ExitStack() as stack:
        runs = [stack.enter_context(load_results(path, cache_dir=cache_dir)) for path in paths]
        return _compare(paths, runs, bootstrap_resamples, bootstrap_seed, tag_mapping)


def paired_rows(runs: list[tuple]) -> np.ndarray:
    """Rows that every run completed, after checking that all runs share one dataset selection."""
    fingerprint = runs[0][0]["dataset"]["fingerprint"]
    if any(run[0]["dataset"]["fingerprint"] != fingerprint for run in runs):
        raise BenchmarkError("Runs use different dataset subsets, annotation versions, or ordering.")
    paired = np.logical_and.reduce([run[4] for run in runs])
    if not paired.any():
        raise BenchmarkError("No shared successful samples.")
    return np.flatnonzero(paired)


def native_columns(run: tuple, category: str) -> dict[str, int]:
    return {
        label: index
        for index, (label, kind) in enumerate(zip(run[2]["labels"], run[2]["categories"], strict=True))
        if kind == category and label is not None
    }


def category_selection(
    runs: list[tuple], configs: list[dict], category: str, rules: TagRules | None = None
) -> tuple[list[str], list[tuple[np.ndarray | MappedScores, list[int], Thresholds]], dict | None]:
    """Common category vocabulary and per-run (scores, columns, threshold), or why it is excluded."""
    missing = [config["id"] for config in configs if category not in config["thresholds"]]
    if missing:
        return [], [], {"reason": "missing_threshold", "profiles": missing}
    lookups = []
    for run in runs:
        lookup = {}
        for label, index in native_columns(run, category).items():
            for output in rules.expand({label}) if rules is not None else (label,):
                lookup.setdefault(output, []).append(index)
        lookups.append(lookup)
    labels = sorted(set.intersection(*(set(lookup) for lookup in lookups)))
    if not labels:
        return [], [], {"reason": "no_common_vocabulary"}
    selections = []
    for config, run, lookup in zip(configs, runs, lookups, strict=True):
        thresholds = Thresholds.for_category(config, category)
        groups = [lookup[label] for label in labels]
        if rules is None:
            selections.append((run[3], [group[0] for group in groups], thresholds))
            continue
        contributors = {index: run[2]["labels"][index] for group in groups for index in group}
        cutoffs = {
            index: thresholds.overrides.get(label, thresholds.default)
            for index, label in contributors.items()
        }
        scores = MappedScores(run[3], groups, sorted(contributors.values()), cutoffs)
        selections.append((scores, list(range(len(labels))), thresholds))
    return labels, selections, None


def _compare(
    paths: list[Path], runs: list[tuple], resamples: int, seed: int, tag_mapping: TagMapping | None
) -> dict:
    rows = paired_rows(runs)
    weights = resample_weights(len(rows), resamples, seed) if resamples else None
    categories, excluded_categories = {}, {}
    configs = [read_object(path / "profile.json") for path in paths]
    for category in sorted(runs[0][1][0]["tags"]):
        rules = tag_mapping.rules.get(category) if tag_mapping is not None else None
        labels, selections, exclusion = category_selection(runs, configs, category, rules)
        if exclusion:
            excluded_categories[category] = exclusion
            continue
        profiles, image_counts = [], []
        original_truth = [set(runs[0][1][row]["tags"][category]) for row in rows]
        truth = [rules.expand(tags) for tags in original_truth] if rules is not None else original_truth
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
                    "per_image": {
                        **per_image_summary(image_counts[-1], threshold.default),
                        **describe_thresholds(scores, labels, threshold),
                    },
                }
            )
        categories[category] = {
            "label_count": len(labels),
            "vocabulary_sha256": content_hash(labels),
            "score_scope": "common_outputs_only; full_truth metrics are not native full-vocabulary metrics",
            "runs": profiles,
        }
        if rules is not None:
            strict = set.intersection(*(set(native_columns(run, category)) for run in runs))
            annotated = sum(len(tags) for tags in original_truth)
            represented = sum(len(tags & strict) for tags in original_truth)
            categories[category].update(
                score_scope="mapped common outputs; ground truth normalized with the same rules",
                strict_vocabulary={
                    "label_count": len(strict),
                    "positive_occurrences": annotated,
                    "represented_occurrences": represented,
                    "occurrence_recall_ceiling": ratio(represented, annotated),
                },
            )
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
        **({"tag_mapping": tag_mapping.describe()} if tag_mapping is not None else {}),
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
