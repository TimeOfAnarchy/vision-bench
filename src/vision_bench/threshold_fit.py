from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .artifacts import load_results
from .errors import BenchmarkError
from .io import content_hash, identifier, positive_int, read_object
from .metrics import truth_block
from .sweep import GRID_STEPS, f1_curve, reverse_counts, score_levels
from .thresholds import calibration_hashes


def _histograms(levels: np.ndarray, truth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.bincount(levels[truth], minlength=GRID_STEPS + 1),
        np.bincount(levels[~truth], minlength=GRID_STEPS + 1),
    )


def _best_threshold(positive_levels: np.ndarray, negative_levels: np.ndarray) -> float:
    f1 = f1_curve(reverse_counts(positive_levels), reverse_counts(negative_levels))
    return int(np.argmax(f1)) / GRID_STEPS


def _fit_category(
    scores: np.ndarray,
    rows: np.ndarray,
    columns: list[int],
    labels: list[str],
    truth_sets: list[set[str]],
    min_tag_support: int | None,
) -> tuple[float | None, dict[str, float], int]:
    positive_levels = np.zeros(GRID_STEPS + 1, dtype=np.int64)
    negative_levels = np.zeros_like(positive_levels)
    overrides = {}
    for start in range(0, len(labels), 128):
        stop = start + 128
        truth = truth_block(labels[start:stop], truth_sets)
        levels = score_levels(scores[np.ix_(rows, columns[start:stop])])
        positive, negative = _histograms(levels, truth)
        positive_levels += positive
        negative_levels += negative
        if min_tag_support is not None:
            for offset in np.flatnonzero(truth.sum(axis=0) >= min_tag_support):
                overrides[labels[start + offset]] = _best_threshold(
                    *_histograms(levels[:, offset], truth[:, offset])
                )
    positives = int(positive_levels.sum())
    return (_best_threshold(positive_levels, negative_levels) if positives else None), overrides, positives


def fit_thresholds(
    calibration_run: Path,
    *,
    output: Path,
    profile_id: str | None = None,
    per_tag: bool = False,
    min_support: int = 20,
    cache_dir: Path | None = None,
) -> dict:
    """Fit new operating thresholds from one verified, completed calibration run."""
    positive_int(min_support, name="Minimum tag support")
    with load_results(calibration_run, cache_dir=cache_dir) as data:
        manifest, samples, vocabulary, scores, valid, _ = data
        profile = read_object(calibration_run / "profile.json")
        base_id = profile["id"]
        profile["id"] = identifier(profile_id if profile_id is not None else f"{base_id[:121]}-fitted")
        rows = np.flatnonzero(valid)
        if not len(rows):
            raise BenchmarkError("Calibration needs at least one successful image.")
        labels, categories = vocabulary["labels"], vocabulary["categories"]
        tag_thresholds = profile.pop("tag_thresholds", {})
        category_results = {}
        for category in sorted(profile["thresholds"].keys() & samples[0]["tags"].keys()):
            columns = [
                i
                for i, (label, kind) in enumerate(zip(labels, categories, strict=True))
                if label is not None and kind == category
            ]
            truth = [set(samples[row]["tags"][category]) for row in rows]
            threshold, overrides, positives = _fit_category(
                scores, rows, columns, [labels[i] for i in columns], truth, min_support if per_tag else None
            )
            if threshold is None:
                continue
            profile["thresholds"][category] = threshold
            tag_thresholds.pop(category, None)
            if overrides:
                tag_thresholds[category] = overrides
            category_results[category] = {"annotated_occurrences": positives, "fitted_tags": len(overrides)}
        if not category_results:
            raise BenchmarkError(
                "Calibration has no annotated positives in the model's thresholded vocabulary."
            )
        if tag_thresholds:
            profile["tag_thresholds"] = tag_thresholds
        hashes = calibration_hashes(profile) | {sample["sha256"] for sample in samples}
        profile["threshold_fit"] = {
            "objective": "category_micro_f1_and_tag_f1" if per_tag else "category_micro_f1",
            "grid_step": 1 / GRID_STEPS,
            "tie_break": "lowest_threshold",
            "min_tag_support": min_support if per_tag else None,
            "source_run_id": manifest["run_id"],
            "source_run_fingerprint": content_hash(manifest),
            "source_profile_fingerprint": manifest["profile_fingerprint"],
            "dataset_fingerprint": manifest["dataset"]["fingerprint"],
            "annotation_kind": manifest["dataset"]["annotation_kind"],
            "selected_images": len(samples),
            "successful_images": len(rows),
            "fitted_categories": category_results,
            "calibration_image_sha256": sorted(hashes),
        }
    serialized = json.dumps(profile, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(serialized + "\n")
    return profile
