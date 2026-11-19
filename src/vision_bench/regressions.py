"""Per-label regressions between a baseline and a candidate checkpoint.

For every common-output label with enough annotated support, a paired bootstrap
over images gives the AP and F1 differences (candidate minus baseline), percentile
intervals and normal-approximation p-values from the bootstrap standard error.
Benjamini-Hochberg q-values control the false discovery rate across labels.
"""

from __future__ import annotations

import math
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .artifacts import load_results
from .bootstrap import CONFIDENCE, block_average_precision, resample_weights, safe_divide
from .compare import category_selection, paired_rows
from .errors import BenchmarkError
from .io import read_object
from .metrics import truth_block

EXAMPLES = 3


def _label_f1(
    truth: np.ndarray, block: np.ndarray, threshold: float, weights: np.ndarray, support: np.ndarray
):
    predicted = block >= threshold
    tp = weights @ (truth & predicted).astype(np.float64)
    fp = weights @ (~truth & predicted).astype(np.float64)
    f1 = safe_divide(2 * tp, tp + fp + support)
    return np.where(support > 0, f1, np.nan)


def _difference(values: np.ndarray) -> dict:
    """Observed difference (row 0), percentile interval and two-sided p-value over finite resamples."""
    estimate, resampled = float(values[0]), values[1:][np.isfinite(values[1:])]
    if resampled.size < 2:
        return {"estimate": estimate, "low": None, "high": None, "p_value": None, "valid_resamples": 0}
    tail = 100 * (1 - CONFIDENCE) / 2
    low, high = np.percentile(resampled, [tail, 100 - tail])
    error = float(np.std(resampled, ddof=1))
    if error > 0:
        p_value = math.erfc(abs(estimate) / error / math.sqrt(2))
    else:
        p_value = 1.0 if estimate == 0 else 0.0
    return {
        "estimate": estimate,
        "low": float(low),
        "high": float(high),
        "p_value": p_value,
        "valid_resamples": int(resampled.size),
    }


def benjamini_hochberg(p_values: list[float]) -> list[float]:
    """Step-up q-values in input order."""
    count = len(p_values)
    order = np.argsort(p_values, kind="stable")
    ranked = np.asarray(p_values, dtype=np.float64)[order] * count / np.arange(1, count + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q_values = np.empty(count)
    q_values[order] = np.minimum(ranked, 1.0)
    return q_values.tolist()


def _verdict(difference: dict, fdr: float) -> str:
    """A verdict needs both an FDR-controlled q-value and a percentile interval that excludes zero."""
    if difference.get("q_value") is None or difference["q_value"] > fdr:
        return "no_evidence"
    if difference["low"] > 0:
        return "improvement"
    return "regression" if difference["high"] < 0 else "no_evidence"


def _examples(samples, rows, truth_column, baseline, candidate) -> dict:
    change = candidate - baseline

    def pick(mask: np.ndarray, order: np.ndarray) -> list[dict]:
        chosen = [index for index in order if mask[index]][:EXAMPLES]
        return [
            {
                "id": samples[rows[index]]["id"],
                "image": samples[rows[index]]["image"],
                "baseline": float(baseline[index]),
                "candidate": float(candidate[index]),
            }
            for index in chosen
        ]

    return {
        "annotated_with_largest_drop": pick(truth_column & (change < 0), np.argsort(change, kind="stable")),
        "unlabeled_with_largest_rise": pick(~truth_column & (change > 0), np.argsort(-change, kind="stable")),
    }


def _category(runs, configs, rows, category, weights, min_support, fdr, block_size) -> dict | None:
    labels, selections, exclusion = category_selection(runs, configs, category)
    if exclusion:
        return None
    truth_sets = [set(runs[0][1][row]["tags"][category]) for row in rows]
    entries = []
    for start in range(0, len(labels), block_size):
        stop = min(start + block_size, len(labels))
        truth = truth_block(labels[start:stop], truth_sets)
        support = weights @ truth.astype(np.float64)
        blocks = [
            np.asarray(scores[np.ix_(rows, columns[start:stop])], dtype=np.float32)
            for scores, columns, _ in selections
        ]
        ap = [block_average_precision(truth, block, weights, support) for block in blocks]
        f1 = [
            _label_f1(truth, block, threshold, weights, support)
            for block, (_, _, threshold) in zip(blocks, selections, strict=True)
        ]
        for offset in range(stop - start):
            observed = int(support[0, offset])
            if not observed:
                continue
            tested = observed >= min_support
            entry = {
                "label": labels[start + offset],
                "support": observed,
                "baseline": {"ap": float(ap[0][0, offset]), "f1": float(f1[0][0, offset])},
                "candidate": {"ap": float(ap[1][0, offset]), "f1": float(f1[1][0, offset])},
            }
            for metric, values in (("ap", ap), ("f1", f1)):
                entry[f"{metric}_difference"] = (
                    _difference(values[1][:, offset] - values[0][:, offset])
                    if tested
                    else {"estimate": entry["candidate"][metric] - entry["baseline"][metric]}
                )
            entry["_scores"] = (truth[:, offset], blocks[0][:, offset], blocks[1][:, offset])
            entries.append(entry)
    summary = {}
    for metric in ("ap", "f1"):
        key = f"{metric}_difference"
        tested = [entry for entry in entries if entry[key].get("p_value") is not None]
        for entry, q_value in zip(
            tested, benjamini_hochberg([e[key]["p_value"] for e in tested]), strict=True
        ):
            entry[key]["q_value"] = q_value
        for entry in entries:
            entry[key]["verdict"] = (
                _verdict(entry[key], fdr) if entry["support"] >= min_support else "untested"
            )
        verdicts = [entry[key]["verdict"] for entry in entries]
        summary[metric] = {
            name: verdicts.count(name) for name in ("regression", "improvement", "no_evidence", "untested")
        }
    for entry in entries:
        truth_column, baseline, candidate = entry.pop("_scores")
        if "regression" in (entry["ap_difference"]["verdict"], entry["f1_difference"]["verdict"]):
            entry["examples"] = _examples(runs[0][1], rows, truth_column, baseline, candidate)
    entries.sort(key=lambda entry: (entry["ap_difference"]["estimate"], entry["label"]))
    return {
        "label_count": len(labels),
        "supported_labels": len(entries),
        "tested_labels": sum(entry["support"] >= min_support for entry in entries),
        "thresholds": {"baseline": selections[0][2], "candidate": selections[1][2]},
        "summary": summary,
        "labels": entries,
    }


def label_regressions(
    baseline: Path,
    candidate: Path,
    *,
    resamples: int = 1000,
    seed: int = 0,
    min_support: int = 5,
    fdr: float = 0.05,
    cache_dir: Path | None = None,
    block_size: int = 128,
) -> dict:
    """Paired per-label AP/F1 differences, candidate minus baseline, with FDR-controlled verdicts."""
    if baseline.resolve() == candidate.resolve():
        raise BenchmarkError("Regression report needs two distinct runs.")
    for value, name, low in (
        (resamples, "resamples", 2),
        (seed, "seed", 0),
        (min_support, "minimum support", 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < low:
            raise BenchmarkError(f"Regression {name} must be an integer of at least {low}.")
    if not 0 < fdr < 1:
        raise BenchmarkError("False discovery rate must be between 0 and 1.")
    paths = [baseline, candidate]
    with ExitStack() as stack:
        runs = [stack.enter_context(load_results(path, cache_dir=cache_dir)) for path in paths]
        rows = paired_rows(runs)
        configs = [read_object(path / "profile.json") for path in paths]
        weights = resample_weights(len(rows), resamples, seed)
        categories = {}
        for category in sorted(runs[0][1][0]["tags"]):
            result = _category(runs, configs, rows, category, weights, min_support, fdr, block_size)
            if result is not None:
                categories[category] = result
    if not categories:
        raise BenchmarkError("No category has both common vocabulary and thresholds for both runs.")
    return {
        "schema_version": 1,
        "kind": "label_regressions",
        "baseline": {"run_id": runs[0][0]["run_id"], "profile_id": configs[0]["id"]},
        "candidate": {"run_id": runs[1][0]["run_id"], "profile_id": configs[1]["id"]},
        "dataset_fingerprint": runs[0][0]["dataset"]["fingerprint"],
        "paired_successful": len(rows),
        "method": {
            "difference": "candidate minus baseline on paired successful samples and common outputs",
            "bootstrap": {"resamples": resamples, "seed": seed, "confidence": CONFIDENCE},
            "p_value": "two-sided normal approximation from the bootstrap standard error",
            "multiple_testing": "Benjamini-Hochberg across tested labels, separately for AP and F1",
            "verdict": "q-value at most the false discovery rate and a percentile interval excluding zero",
            "false_discovery_rate": fdr,
            "min_support": min_support,
            "f1_thresholds": "each run's own profile threshold",
            "examples": "annotated images with the largest score drop and unlabeled images with the largest rise",
        },
        "categories": categories,
    }
