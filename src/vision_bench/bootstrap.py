"""Paired bootstrap intervals for tag metrics on a shared sample set.

Each resample draws paired successful samples with replacement and applies the
same draw to every compared run, so differences keep their pairing. Resamples
are integer sample weights; weighted metrics equal metrics on duplicated rows.
"""

from __future__ import annotations

import numpy as np

from .errors import BenchmarkError
from .metrics import truth_block

CONFIDENCE = 0.95
CUTOFF_COLUMNS = 2048


def resample_weights(count: int, resamples: int, seed: int) -> np.ndarray:
    """Row 0 weights every sample once (the observed estimate); rows 1.. are resamples."""
    if count < 1:
        raise BenchmarkError("Bootstrap needs at least one paired sample.")
    draws = np.random.default_rng(seed).integers(0, count, size=(resamples, count))
    draws += np.arange(resamples)[:, None] * count
    counts = np.bincount(draws.ravel(), minlength=resamples * count).reshape(resamples, count)
    return np.vstack([np.ones((1, count)), counts]).astype(np.float64)


def safe_divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def block_average_precision(
    truth: np.ndarray, block: np.ndarray, weights: np.ndarray, support: np.ndarray
) -> np.ndarray:
    """Tie-grouped AP per weight row and label column; NaN where a label has no positive weight.

    Recall rises only at scores held by an observed positive, so precision is needed
    only at those cut-offs: every row scoring at least the cut-off, ties included.
    Cut-offs are evaluated in bounded column groups.
    """
    ap = np.full((len(weights), truth.shape[1]), np.nan)

    def evaluate(group: list[tuple[int, np.ndarray]]) -> None:
        above = np.hstack([block[:, [label]] >= values for label, values in group])
        owners = np.concatenate([np.full(len(values), label) for label, values in group])
        seen = weights @ above.astype(np.float64)
        hits = weights @ (above & truth[:, owners]).astype(np.float64)
        starts = np.flatnonzero(np.r_[True, owners[1:] != owners[:-1]])
        gains = np.diff(hits, axis=1, prepend=0.0)
        gains[:, starts] = hits[:, starts]
        weighted = np.add.reduceat(gains * safe_divide(hits, seen), starts, axis=1)
        labels = owners[starts]
        positives = support[:, labels]
        ap[:, labels] = np.where(positives > 0, safe_divide(weighted, positives), np.nan)

    group, width = [], 0
    for label in np.flatnonzero(truth.any(axis=0)):
        values = np.unique(block[truth[:, label], label])[::-1]
        group.append((label, values))
        width += len(values)
        if width >= CUTOFF_COLUMNS:
            evaluate(group)
            group, width = [], 0
    if group:
        evaluate(group)
    return ap


def weighted_metrics(
    runs: list[tuple[np.ndarray, list[int], float]],
    rows: np.ndarray,
    labels: list[str],
    truth_sets: list[set[str]],
    weights: np.ndarray,
    *,
    block_size: int = 128,
) -> list[dict[str, np.ndarray]]:
    """Metrics for each (scores, columns, threshold) run under each weight row."""
    if weights.shape[1] != len(rows) or len(rows) != len(truth_sets):
        raise BenchmarkError("Bootstrap weights, samples and annotations disagree.")
    draws = len(weights)
    vocabulary = set(labels)
    oov = np.asarray([len(values - vocabulary) for values in truth_sets], dtype=np.float64)
    state = [
        {
            name: np.zeros(draws)
            for name in ("tp", "fp", "fn", "macro_sum", "macro_count", "ap_sum", "ap_count")
        }
        for _ in runs
    ]
    for start in range(0, len(labels), block_size):
        stop = min(start + block_size, len(labels))
        truth = truth_block(labels[start:stop], truth_sets)
        support = weights @ truth.astype(np.float64)
        for (scores, columns, threshold), totals in zip(runs, state, strict=True):
            block = np.asarray(scores[np.ix_(rows, columns[start:stop])], dtype=np.float32)
            predicted = block >= threshold
            tp = weights @ (truth & predicted).astype(np.float64)
            fp = weights @ (~truth & predicted).astype(np.float64)
            fn = support - tp
            totals["tp"] += tp.sum(axis=1)
            totals["fp"] += fp.sum(axis=1)
            totals["fn"] += fn.sum(axis=1)
            present = support > 0
            totals["macro_sum"] += np.where(present, safe_divide(2 * tp, 2 * tp + fp + fn), 0.0).sum(axis=1)
            totals["macro_count"] += present.sum(axis=1)
            ap = block_average_precision(truth, block, weights, support)
            totals["ap_sum"] += np.nansum(ap, axis=1)
            totals["ap_count"] += np.count_nonzero(~np.isnan(ap), axis=1)
    missing = weights @ oov
    results = []
    for totals in state:
        tp, fp, fn = totals["tp"], totals["fp"], totals["fn"]
        full_fn = fn + missing
        results.append(
            {
                "micro_full_truth_f1": safe_divide(2 * tp, 2 * tp + fp + full_fn),
                "micro_full_truth_recall": safe_divide(tp, tp + full_fn),
                "micro_representable_f1": safe_divide(2 * tp, 2 * tp + fp + fn),
                "micro_representable_precision": safe_divide(tp, tp + fp),
                "micro_representable_recall": safe_divide(tp, tp + fn),
                "macro_f1_supported": np.where(
                    totals["macro_count"] > 0, safe_divide(totals["macro_sum"], totals["macro_count"]), np.nan
                ),
                "mean_average_precision_supported": np.where(
                    totals["ap_count"] > 0, safe_divide(totals["ap_sum"], totals["ap_count"]), np.nan
                ),
            }
        )
    return results


def _interval(values: np.ndarray) -> dict:
    """Observed value (row 0) and a percentile interval over finite resamples."""
    estimate, resampled = values[0], values[1:]
    finite = resampled[np.isfinite(resampled)]
    tail = 100 * (1 - CONFIDENCE) / 2
    low, high = np.percentile(finite, [tail, 100 - tail]) if finite.size else (None, None)
    return {
        "estimate": float(estimate) if np.isfinite(estimate) else None,
        "low": None if low is None else float(low),
        "high": None if high is None else float(high),
        "valid_resamples": int(finite.size),
    }


def summarize(run_ids: list[str], statistics: list[dict[str, np.ndarray]]) -> dict:
    """Per-run intervals and paired differences against the first (reference) run."""
    reference = statistics[0]
    return {
        "runs": [
            {"run_id": run_id, "metrics": {name: _interval(value) for name, value in values.items()}}
            for run_id, values in zip(run_ids, statistics, strict=True)
        ],
        "differences_from_reference": [
            {
                "run_id": run_id,
                "metrics": {name: _interval(value - reference[name]) for name, value in values.items()},
            }
            for run_id, values in zip(run_ids[1:], statistics[1:], strict=True)
        ],
    }
