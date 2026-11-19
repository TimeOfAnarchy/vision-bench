"""Threshold-free quality: a micro operating curve and score calibration.

Both use every sample-label pair of the selected columns. Unlabeled pairs count
as negatives, as in the fixed-threshold metrics.
"""

from __future__ import annotations

import numpy as np

from .errors import BenchmarkError
from .metrics import truth_block

GRID_STEPS = 1000
CALIBRATION_BINS = 10
TARGETS = (0.5, 0.6, 0.7, 0.8)


def _point(index: int, precision: np.ndarray, recall: np.ndarray, f1: np.ndarray) -> dict:
    return {
        "threshold": index / GRID_STEPS,
        "precision": float(precision[index]),
        "recall": float(recall[index]),
        "f1": float(f1[index]),
    }


def _best(values: np.ndarray, allowed: np.ndarray) -> int | None:
    """Lowest grid index with the maximum value among allowed thresholds."""
    return int(np.argmax(np.where(allowed, values, -1.0))) if allowed.any() else None


def threshold_analysis(
    scores: np.ndarray,
    rows: np.ndarray,
    columns: list[int],
    labels: list[str],
    truth_sets: list[set[str]],
    *,
    block_size: int = 128,
) -> dict:
    """Counts at every grid threshold (score >= threshold, float32) and calibration bins."""
    if len(rows) != len(truth_sets) or len(columns) != len(labels):
        raise BenchmarkError("Sweep selection and annotations disagree.")
    grid = (np.arange(GRID_STEPS + 1) / GRID_STEPS).astype(np.float32)
    positive_levels = np.zeros(GRID_STEPS + 1, dtype=np.int64)
    negative_levels = np.zeros(GRID_STEPS + 1, dtype=np.int64)
    bin_count = np.zeros(CALIBRATION_BINS, dtype=np.int64)
    bin_positive = np.zeros(CALIBRATION_BINS, dtype=np.int64)
    bin_score = np.zeros(CALIBRATION_BINS)
    for start in range(0, len(labels), block_size):
        stop = min(start + block_size, len(labels))
        block = np.asarray(scores[np.ix_(rows, columns[start:stop])], dtype=np.float32)
        if not np.isfinite(block).all() or np.any((block < 0) | (block > 1)):
            raise BenchmarkError("Invalid probabilities in sweep block.")
        truth = truth_block(labels[start:stop], truth_sets)
        level = np.searchsorted(grid, block, side="right") - 1
        positive_levels += np.bincount(level[truth], minlength=GRID_STEPS + 1)
        negative_levels += np.bincount(level[~truth], minlength=GRID_STEPS + 1)
        bucket = np.minimum(level // (GRID_STEPS // CALIBRATION_BINS), CALIBRATION_BINS - 1)
        bin_count += np.bincount(bucket.ravel(), minlength=CALIBRATION_BINS)
        bin_positive += np.bincount(bucket[truth], minlength=CALIBRATION_BINS)
        bin_score += np.bincount(bucket.ravel(), weights=block.ravel(), minlength=CALIBRATION_BINS)
    tp = np.cumsum(positive_levels[::-1])[::-1]
    fp = np.cumsum(negative_levels[::-1])[::-1]
    positives = int(tp[0])
    predicted = tp + fp
    precision = np.divide(tp, predicted, out=np.zeros(len(tp)), where=predicted > 0)
    recall = tp / positives if positives else np.zeros(len(tp))
    f1 = np.divide(2 * tp, predicted + positives, out=np.zeros(len(tp)), where=(predicted + positives) > 0)

    def target(index: int | None) -> dict | None:
        return None if index is None else _point(index, precision, recall, f1)

    best = _best(f1, np.full(len(tp), positives > 0))

    pairs, upper_pairs = int(bin_count.sum()), int(bin_count[1:].sum())
    gaps = np.abs(bin_score - bin_positive)
    return {
        "scope": "micro over selected sample-label pairs; unlabeled pairs count as negatives",
        "pairs": pairs,
        "annotated_positives": positives,
        "grid": {"step": 1 / GRID_STEPS, "rule": "score >= threshold, compared in float32"},
        "true_positive": tp.tolist(),
        "unlabeled_positive": fp.tolist(),
        "best_f1": None if best is None else {**target(best), "fitted_on_evaluation_set": True},
        "precision_at_recall": {str(value): target(_best(precision, recall >= value)) for value in TARGETS},
        "recall_at_precision": {str(value): target(_best(recall, precision >= value)) for value in TARGETS},
        "calibration": {
            "scope": "annotated positive rate; incomplete silver labels understate the true positive rate",
            "bins": [
                {
                    "low": index / CALIBRATION_BINS,
                    "high": (index + 1) / CALIBRATION_BINS,
                    "count": int(bin_count[index]),
                    "mean_score": float(bin_score[index] / bin_count[index]) if bin_count[index] else None,
                    "annotated_positive_rate": float(bin_positive[index] / bin_count[index])
                    if bin_count[index]
                    else None,
                }
                for index in range(CALIBRATION_BINS)
            ],
            "expected_calibration_error": float(gaps.sum() / pairs) if pairs else None,
            "upper_bins": {
                "scope": "scores >= 0.1; the lowest bin holds most pairs and dominates the overall error",
                "pairs": upper_pairs,
                "expected_calibration_error": float(gaps[1:].sum() / upper_pairs) if upper_pairs else None,
            },
        },
    }
