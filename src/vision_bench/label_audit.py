"""Silver-label audit: review a uniform sample of unlabeled predictions, then correct precision.

Unlabeled predictions are counted as errors by every other metric. Export draws a
uniform random sample of them per run; a reviewer marks each as present, absent
or unsure; scoring estimates which share of all unlabeled predictions is really
present and corrects precision with that rate.
"""

from __future__ import annotations

import math
from contextlib import ExitStack
from pathlib import Path

import numpy as np

from .artifacts import load_results
from .compare import category_selection, paired_rows
from .errors import BenchmarkError
from .io import canonical_json, content_hash, read_jsonl, read_object
from .metrics import truth_block
from .thresholds import Thresholds

VERDICTS = ("present", "absent", "unsure")
Z_95 = 1.959963984540054


def _unlabeled_positives(
    scores: np.ndarray,
    rows: np.ndarray,
    columns: list[int],
    labels: list[str],
    truth_sets: list[set[str]],
    threshold: Thresholds,
    *,
    block_size: int = 128,
) -> tuple[int, np.ndarray]:
    """True-positive count and (row position, label index) of every unlabeled positive."""
    true_positive, pairs = 0, [np.empty((0, 2), dtype=np.int64)]
    for start in range(0, len(labels), block_size):
        stop = min(start + block_size, len(labels))
        block = np.asarray(scores[np.ix_(rows, columns[start:stop])], dtype=np.float32)
        truth = truth_block(labels[start:stop], truth_sets)
        predicted = block >= threshold.for_labels(labels[start:stop])
        true_positive += int(np.count_nonzero(truth & predicted))
        position, label = np.nonzero(~truth & predicted)
        pairs.append(np.stack([position, label + start], axis=1))
    return true_positive, np.concatenate(pairs)


def _plan(paths: list[Path], runs: list[tuple], category: str, per_run: int, seed: int) -> tuple:
    if isinstance(per_run, bool) or not isinstance(per_run, int) or per_run < 1:
        raise BenchmarkError("Audit sample size per run must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise BenchmarkError("Audit seed must be a non-negative integer.")
    rows = paired_rows(runs)
    configs = [read_object(path / "profile.json") for path in paths]
    if category not in runs[0][1][0]["tags"]:
        raise BenchmarkError(f"Dataset has no annotation category {category!r}.")
    labels, selections, exclusion = category_selection(runs, configs, category)
    if exclusion:
        raise BenchmarkError(f"Category {category!r} cannot be audited: {exclusion['reason']}.")
    truth = [set(runs[0][1][row]["tags"][category]) for row in rows]
    params = {
        "category": category,
        "seed": seed,
        "per_run": per_run,
        "dataset_fingerprint": runs[0][0]["dataset"]["fingerprint"],
        "vocabulary_sha256": content_hash(labels),
        "runs": [run[0]["run_id"] for run in runs],
    }
    plans, items = [], {}
    for index, (config, run, (scores, columns, threshold)) in enumerate(
        zip(configs, runs, selections, strict=True)
    ):
        true_positive, pairs = _unlabeled_positives(scores, rows, columns, labels, truth, threshold)
        rng = np.random.default_rng([seed, index])
        chosen = pairs[rng.choice(len(pairs), size=min(per_run, len(pairs)), replace=False)]
        for position, label in chosen.tolist():
            items.setdefault((position, label), []).append(run[0]["run_id"])
        plans.append(
            {
                "run_id": run[0]["run_id"],
                "profile_id": config["id"],
                **threshold.describe(labels),
                "true_positive": true_positive,
                "unlabeled_positive": len(pairs),
                "sampled": len(chosen),
            }
        )
    order = sorted(items)
    order = [order[index] for index in np.random.default_rng([seed, len(runs)]).permutation(len(order))]
    packet = []
    for position, label in order:
        row, sample = int(rows[position]), runs[0][1][int(rows[position])]
        packet.append(
            {
                "schema_version": 1,
                "kind": "silver_label_audit",
                "audit": params,
                "id": sample["id"],
                "image": sample["image"],
                "sha256": sample["sha256"],
                "label": labels[label],
                "annotated": sorted(truth[position]),
                "scores": {
                    run[0]["run_id"]: float(selection[0][row, selection[1][label]])
                    for run, selection in zip(runs, selections, strict=True)
                },
                "sampled_by": items[(position, label)],
                "human_review": {"reviewer": None, "verdict": None, "notes": ""},
            }
        )
    return params, plans, packet


def export_label_audit(
    paths: list[Path],
    *,
    category: str,
    output: Path,
    per_run: int = 100,
    seed: int = 0,
    cache_dir: Path | None = None,
) -> int:
    """Write a shuffled JSONL review packet; reviewing any prefix keeps each run's sample uniform."""
    if not paths or len({path.resolve() for path in paths}) != len(paths):
        raise BenchmarkError("Label audit needs distinct runs.")
    with ExitStack() as stack:
        runs = [stack.enter_context(load_results(path, cache_dir=cache_dir)) for path in paths]
        _, _, packet = _plan(paths, runs, category, per_run, seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as handle:
        for item in packet:
            handle.write(canonical_json(item) + "\n")
    return len(packet)


def _wilson(successes: int, total: int) -> dict | None:
    if not total:
        return None
    rate = successes / total
    scale = 1 + Z_95**2 / total
    center = (rate + Z_95**2 / (2 * total)) / scale
    half = Z_95 * math.sqrt(rate * (1 - rate) / total + Z_95**2 / (4 * total**2)) / scale
    return {"estimate": rate, "low": max(0.0, center - half), "high": min(1.0, center + half)}


def score_label_audit(audit_path: Path, paths: list[Path], *, cache_dir: Path | None = None) -> dict:
    """Verify a reviewed packet against its runs and estimate audited precision per run."""
    reviewed = read_jsonl(audit_path)
    if not reviewed or any(
        item.get("schema_version") != 1 or item.get("kind") != "silver_label_audit" for item in reviewed
    ):
        raise BenchmarkError("Audit file must contain silver_label_audit records.")
    params = reviewed[0]["audit"]
    if any(item["audit"] != params for item in reviewed):
        raise BenchmarkError("Audit records disagree about their sampling parameters.")
    with ExitStack() as stack:
        runs = [stack.enter_context(load_results(path, cache_dir=cache_dir)) for path in paths]
        if [run[0]["run_id"] for run in runs] != params["runs"]:
            raise BenchmarkError("Score the audit with the same runs, in the same order, as the export.")
        expected, plans, packet = _plan(paths, runs, params["category"], params["per_run"], params["seed"])
    identity = [(item["id"], item["label"], item["sampled_by"]) for item in packet]
    if expected != params or identity != [
        (item["id"], item["label"], item["sampled_by"]) for item in reviewed
    ]:
        raise BenchmarkError("Audit packet does not match the runs and sampling parameters.")
    verdicts = [item["human_review"].get("verdict") for item in reviewed]
    if any(verdict is not None and verdict not in VERDICTS for verdict in verdicts):
        raise BenchmarkError(f"Review verdicts must be one of {', '.join(VERDICTS)} or null.")
    results = []
    for plan in plans:
        mine = [
            verdict
            for item, verdict in zip(reviewed, verdicts, strict=True)
            if plan["run_id"] in item["sampled_by"]
        ]
        counts = {verdict: mine.count(verdict) for verdict in VERDICTS} | {"pending": mine.count(None)}
        present_rate = _wilson(counts["present"], counts["present"] + counts["absent"])
        predicted = plan["true_positive"] + plan["unlabeled_positive"]

        def precision(rate: float) -> float:
            return (
                (plan["true_positive"] + plan["unlabeled_positive"] * rate) / predicted if predicted else 0.0
            )

        results.append(
            {
                **plan,
                "verdicts": counts,
                "present_rate": present_rate,
                "silver_precision": precision(0.0),
                "audited_precision": None
                if present_rate is None
                else {key: precision(present_rate[key]) for key in ("estimate", "low", "high")},
            }
        )
    return {
        "schema_version": 1,
        "kind": "silver_label_audit_result",
        "audit": params,
        "items": len(reviewed),
        "runs": results,
        "scope": (
            "common outputs at profile thresholds; present rate is a Wilson 95% interval over reviewed "
            "present/absent verdicts (unsure and pending excluded); recall cannot be corrected because "
            "positives missed by both annotations and the model stay unknown"
        ),
    }
