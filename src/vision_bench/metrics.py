from __future__ import annotations

import numpy as np

from .errors import BenchmarkError
from .io import positive_int


def ratio(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def counts_metrics(tp: int, fp: int, fn: int) -> dict:
    return {
        "true_positive": tp,
        "unlabeled_positive": fp,
        "false_negative": fn,
        "precision": ratio(tp, tp + fp),
        "recall": ratio(tp, tp + fn),
        "f1": ratio(2 * tp, 2 * tp + fp + fn),
    }


def average_precision(truth: np.ndarray, scores: np.ndarray) -> float | None:
    if not truth.any():
        return None
    order = np.argsort(-scores, kind="stable")
    truth, scores = truth[order], scores[order]
    boundaries = np.flatnonzero(np.r_[scores[1:] != scores[:-1], True])
    tp = np.cumsum(truth, dtype=np.int64)[boundaries]
    precision = tp / (boundaries + 1)
    recall = tp / truth.sum()
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def evaluate_columns(
    scores: np.ndarray,
    rows: np.ndarray,
    columns: list[int],
    labels: list[str],
    truth_sets: list[set[str]],
    threshold: float,
    *,
    block_size: int = 128,
) -> dict:
    """Exact metrics with at most rows x block_size score/truth work arrays."""
    positive_int(block_size, name="metric block size")
    if len(rows) != len(truth_sets) or len(columns) != len(labels):
        raise BenchmarkError("Metric selection and annotations disagree.")
    if len(set(labels)) != len(labels) or not 0 <= threshold <= 1:
        raise BenchmarkError("Invalid vocabulary or metric threshold.")
    vocabulary = set(labels)
    tp, fp, fn, support = (np.zeros(len(labels), dtype=np.int64) for _ in range(4))
    per_label, ap_values = [], []
    for start in range(0, len(labels), block_size):
        stop = min(start + block_size, len(labels))
        block = scores[np.ix_(rows, columns[start:stop])]
        if not np.isfinite(block).all() or np.any((block < 0) | (block > 1)):
            raise BenchmarkError("Invalid probabilities in metric block.")
        truth = np.asarray(
            [[label in values for label in labels[start:stop]] for values in truth_sets], dtype=np.bool_
        ).reshape(len(rows), stop - start)
        predicted = block >= threshold
        tp[start:stop] = np.count_nonzero(truth & predicted, axis=0)
        fp[start:stop] = np.count_nonzero(~truth & predicted, axis=0)
        fn[start:stop] = np.count_nonzero(truth & ~predicted, axis=0)
        support[start:stop] = truth.sum(axis=0)
        for offset in np.flatnonzero(support[start:stop]):
            index = start + int(offset)
            ap = average_precision(truth[:, offset], block[:, offset])
            ap_values.append(ap)
            per_label.append(
                {
                    "label": labels[index],
                    "support": int(support[index]),
                    "ap": ap,
                    **counts_metrics(int(tp[index]), int(fp[index]), int(fn[index])),
                }
            )
    supported = support > 0
    per_label.sort(key=lambda value: (-value["support"], value["label"]))
    total_occurrences = sum(len(values) for values in truth_sets)
    covered_occurrences = sum(len(values & vocabulary) for values in truth_sets)
    unique_truth = set().union(*truth_sets) if truth_sets else set()
    oov = total_occurrences - covered_occurrences
    macro_f1 = np.divide(
        2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=(2 * tp + fp + fn) > 0
    )
    return {
        "sample_count": len(truth_sets),
        "label_count": len(labels),
        "threshold": threshold,
        "micro_representable": counts_metrics(int(tp.sum()), int(fp.sum()), int(fn.sum())),
        "micro_full_truth": counts_metrics(int(tp.sum()), int(fp.sum()), int(fn.sum()) + oov),
        "macro_f1_supported": float(macro_f1[supported].mean()) if supported.any() else None,
        "mean_average_precision_supported": float(np.mean(ap_values)) if ap_values else None,
        "coverage": {
            "truth_labels": len(unique_truth),
            "represented_truth_labels": len(unique_truth & vocabulary),
            "positive_occurrences": total_occurrences,
            "represented_occurrences": covered_occurrences,
            "oov_occurrences": oov,
            "occurrence_recall_ceiling": ratio(covered_occurrences, total_occurrences),
        },
        "per_label": per_label,
        "absence_semantics": "unlabeled_is_negative_for_metric_only",
    }


def performance_summary(records: list[dict], warmup: int, *, gpu_memory: dict | None = None) -> dict:
    successful = [record for record in records if record["outcome"] == "ok"]
    warm = [record for record in successful if record["row"] >= warmup]

    def timing(values: list[dict]) -> dict:
        times = np.asarray([value["latency_ms"] for value in values], dtype=float)
        return {
            "count": len(values),
            "median_ms": float(np.median(times)) if times.size else None,
            "p95_ms": float(np.percentile(times, 95)) if times.size else None,
            "images_per_second": float(1000 * times.size / times.sum())
            if times.size and times.sum()
            else None,
        }

    return {
        "all_successful": timing(successful),
        "warm": timing(warm),
        "warmup_excluded_rows": warmup,
        "scope": "decode_preprocess_inference",
        "gpu_memory": gpu_memory or {"status": "unavailable", "reason": "not_recorded"},
    }
