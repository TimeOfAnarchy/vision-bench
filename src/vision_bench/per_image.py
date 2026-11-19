"""Per-image quality: what one tagged image looks like, averaged over images.

Micro metrics weight images by their tag count; these average each image once.
Unlabeled predictions count as errors for the metric only, as elsewhere.
"""

from __future__ import annotations

import numpy as np

from .errors import BenchmarkError
from .metrics import truth_block
from .score_blocks import MappedScores, score_block, scored_decisions
from .thresholds import Thresholds

RANKS = (5, 10, 20)


def per_image_counts(
    scores: np.ndarray | MappedScores,
    rows: np.ndarray,
    columns: list[int],
    labels: list[str],
    truth_sets: list[set[str]],
    threshold: float | Thresholds,
    *,
    block_size: int = 128,
) -> dict[str, np.ndarray]:
    """Per-image counts at the threshold and expected top-k hits (ties shared evenly)."""
    if len(rows) != len(truth_sets) or len(columns) != len(labels):
        raise BenchmarkError("Per-image selection and annotations disagree.")
    count = len(rows)
    depth = max(RANKS)
    top = np.full((count, depth), -np.inf, dtype=np.float32)

    starts = range(0, len(labels), block_size)
    for start in starts:
        block = np.asarray(score_block(scores, rows, columns[start : start + block_size]), dtype=np.float32)
        if not np.isfinite(block).all() or np.any((block < 0) | (block > 1)):
            raise BenchmarkError("Invalid probabilities in per-image block.")
        merged = np.concatenate([top, block], axis=1)
        top = -np.partition(-merged, depth - 1, axis=1)[:, :depth]
    top = -np.sort(-top, axis=1)
    cutoffs = {rank: top[:, rank - 1 : rank] for rank in RANKS}  # -inf when fewer outputs than rank.
    tp, fp, fn = (np.zeros(count, dtype=np.int64) for _ in range(3))
    above = {rank: np.zeros((2, count), dtype=np.int64) for rank in RANKS}  # all, positive
    tied = {rank: np.zeros((2, count), dtype=np.int64) for rank in RANKS}
    for start in starts:
        stop = start + block_size
        block, predicted = scored_decisions(
            scores, rows, columns[start:stop], labels[start:stop], threshold, dtype=np.float32
        )
        truth = truth_block(labels[start:stop], truth_sets)
        tp += np.count_nonzero(truth & predicted, axis=1)
        fp += np.count_nonzero(~truth & predicted, axis=1)
        fn += np.count_nonzero(truth & ~predicted, axis=1)
        for rank, cutoff in cutoffs.items():
            higher, equal = block > cutoff, block == cutoff
            above[rank] += [np.count_nonzero(higher, axis=1), np.count_nonzero(higher & truth, axis=1)]
            tied[rank] += [np.count_nonzero(equal, axis=1), np.count_nonzero(equal & truth, axis=1)]
    vocabulary = set(labels)
    result = {
        "true_positive": tp,
        "unlabeled_positive": fp,
        "false_negative": fn,
        "oov_positive": np.asarray([len(values - vocabulary) for values in truth_sets], dtype=np.int64),
    }
    for rank in RANKS:
        slots = np.minimum(rank, len(labels)) - above[rank][0]
        share = np.divide(slots, tied[rank][0], out=np.zeros(count), where=tied[rank][0] > 0)
        result[f"hits_at_{rank}"] = above[rank][1] + tied[rank][1] * share
    return result


def _ratios(counts: dict[str, np.ndarray]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per-image numerator and denominator; an image defines a metric when the denominator is positive."""
    tp = counts["true_positive"].astype(np.float64)
    predicted = tp + counts["unlabeled_positive"]
    represented = tp + counts["false_negative"]
    positives = represented + counts["oov_positive"]
    return {
        "sample_precision": (tp, predicted),
        "sample_recall_representable": (tp, represented),
        "sample_recall_full_truth": (tp, positives),
        "sample_f1_representable": (2 * tp, predicted + represented),
        "sample_f1_full_truth": (2 * tp, predicted + positives),
        **{f"sample_recall_at_{rank}": (counts[f"hits_at_{rank}"], represented) for rank in RANKS},
    }


def sample_metrics(counts: dict[str, np.ndarray], weights: np.ndarray) -> dict[str, np.ndarray]:
    """Image-averaged metrics for each weight row; NaN where no image defines a metric."""
    result = {}
    for name, (numerator, denominator) in _ratios(counts).items():
        defined = denominator > 0
        ratio = np.divide(numerator, denominator, out=np.zeros(len(numerator)), where=defined)
        total = weights @ defined.astype(np.float64)
        result[name] = np.divide(weights @ ratio, total, out=np.full(len(weights), np.nan), where=total > 0)
    return result


def per_image_summary(counts: dict[str, np.ndarray], threshold: float) -> dict:
    ratios = _ratios(counts)
    observed = sample_metrics(counts, np.ones((1, len(counts["true_positive"]))))
    numerator, denominator = ratios["sample_f1_full_truth"]
    f1 = numerator[denominator > 0] / denominator[denominator > 0]
    tp, predicted = ratios["sample_precision"]
    represented = ratios["sample_recall_representable"][1]
    positives = ratios["sample_recall_full_truth"][1]
    return {
        "scope": "each image weighted once; common outputs at the profile threshold, top-k over common outputs",
        "threshold": threshold,
        "images": len(tp),
        "metrics": {
            name: {
                "value": None if np.isnan(observed[name][0]) else float(observed[name][0]),
                "images": int((ratios[name][1] > 0).sum()),
            }
            for name in ratios
        },
        "sample_f1_full_truth_quantiles": {
            key: float(np.percentile(f1, q)) if f1.size else None
            for key, q in (("p10", 10), ("p25", 25), ("p50", 50))
        },
        "images_without_predictions": int((predicted == 0).sum()),
        "images_with_common_positives_but_no_hits": int(((represented > 0) & (tp == 0)).sum()),
        "images_with_only_out_of_vocabulary_positives": int(((represented == 0) & (positives > 0)).sum()),
        "recall_at_k_ties": "positives tied at the k-th score share the remaining slots evenly",
    }
