from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack
from itertools import combinations
from pathlib import Path

import numpy as np

from .artifacts import load_results
from .compare import category_selection, paired_rows
from .errors import BenchmarkError
from .io import canonical_json, content_hash, positive_int, read_object
from .metrics import truth_block
from .score_blocks import MappedScores, describe_thresholds, scored_decisions
from .tag_mapping import TagMapping
from .thresholds import Thresholds

BLOCK_SIZE = 128
Selection = tuple[np.ndarray | MappedScores, list[int], Thresholds]


def _image_statistics(
    selections: list[Selection], rows: np.ndarray, labels: list[str], truth_sets: list[set[str]]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pairs = list(combinations(range(len(selections)), 2))
    predicted = np.zeros((len(selections), len(rows)), dtype=np.int64)
    true_positive = np.zeros_like(predicted)
    intersections = np.zeros((len(pairs), len(rows)), dtype=np.int64)
    for start in range(0, len(labels), BLOCK_SIZE):
        stop = start + BLOCK_SIZE
        block_labels = labels[start:stop]
        truth = truth_block(block_labels, truth_sets)
        masks = [
            scored_decisions(scores, rows, columns[start:stop], block_labels, threshold)[1]
            for scores, columns, threshold in selections
        ]
        for index, mask in enumerate(masks):
            predicted[index] += mask.sum(axis=1)
            true_positive[index] += (mask & truth).sum(axis=1)
        for index, (left, right) in enumerate(pairs):
            intersections[index] += (masks[left] & masks[right]).sum(axis=1)

    disagreement = np.zeros(len(rows))
    for (left, right), shared in zip(pairs, intersections, strict=True):
        union = predicted[left] + predicted[right] - shared
        disagreement += np.divide(union - shared, union, out=np.zeros(len(rows)), where=union > 0)
    return predicted, true_positive, disagreement / len(pairs)


def _f1(true_positive: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(
        2 * true_positive, denominator, out=np.full(denominator.shape, np.nan), where=denominator > 0
    )


def _predictions(selection: Selection, row: int, labels: list[str]) -> list[dict]:
    scores, columns, threshold = selection
    rows = np.asarray([row])
    predictions = []
    for start in range(0, len(labels), BLOCK_SIZE):
        stop = start + BLOCK_SIZE
        block, decisions = scored_decisions(scores, rows, columns[start:stop], labels[start:stop], threshold)
        predictions.extend(
            {"tag": labels[start + offset], "score": float(block[0, offset])}
            for offset in np.flatnonzero(decisions[0])
        )
    return predictions


def _packet(
    runs: list[tuple], configs: list[dict], category: str, top: int, mapping: TagMapping | None
) -> Iterator[dict]:
    rows = paired_rows(runs)
    samples = [runs[0][1][row] for row in rows]
    if category not in samples[0]["tags"]:
        raise BenchmarkError(f"Dataset has no annotation category {category!r}.")
    rules = mapping.rules.get(category) if mapping is not None else None
    labels, selections, exclusion = category_selection(runs, configs, category, rules)
    if exclusion:
        raise BenchmarkError(f"Category {category!r} cannot be exported: {exclusion['reason']}.")
    vocabulary = set(labels)
    annotated = [set(sample["tags"][category]) for sample in samples]
    truth = [rules.expand(tags) for tags in annotated] if rules is not None else annotated
    in_vocabulary = [tags & vocabulary for tags in truth]
    predicted, true_positive, disagreement = _image_statistics(selections, rows, labels, in_vocabulary)
    common_f1 = _f1(true_positive, predicted + np.asarray([len(tags) for tags in in_vocabulary]))
    full_f1 = _f1(true_positive, predicted + np.asarray([len(tags) for tags in truth]))

    def image_id(position: int) -> str:
        return samples[position]["id"]

    reasons = {}
    for rank, position in enumerate(
        sorted(np.flatnonzero(disagreement > 0), key=lambda i: (-disagreement[i], image_id(i)))[:top], 1
    ):
        reasons[position] = [{"kind": "disagreement", "rank": rank}]
    for index, f1_by_image in enumerate(common_f1):
        worst = sorted(np.flatnonzero(np.isfinite(f1_by_image)), key=lambda i: (f1_by_image[i], image_id(i)))
        for rank, position in enumerate(worst[:top], 1):
            reasons.setdefault(position, []).append({"kind": "lowest_f1", "run_index": index, "rank": rank})

    metadata = {
        "category": category,
        "dataset_fingerprint": runs[0][0]["dataset"]["fingerprint"],
        "paired_images": len(rows),
        "vocabulary_sha256": content_hash(labels),
        "top_per_ranking": top,
        **({"tag_mapping": mapping.describe()} if mapping is not None else {}),
    }
    run_metadata = [
        {
            "run_id": run[0]["run_id"],
            "profile_id": profile["id"],
            **describe_thresholds(scores, labels, threshold),
        }
        for run, profile, (scores, _, threshold) in zip(runs, configs, selections, strict=True)
    ]
    for position, ranked_by in reasons.items():
        sample = samples[position]
        model_results = []
        for index, selection in enumerate(selections):
            predictions = _predictions(selection, int(rows[position]), labels)
            tags = {prediction["tag"] for prediction in predictions}
            on_common, on_full = common_f1[index, position], full_f1[index, position]
            model_results.append(
                {
                    **run_metadata[index],
                    "predictions": predictions,
                    "unlabeled_predictions": sorted(tags - in_vocabulary[position]),
                    "missed_annotations": sorted(in_vocabulary[position] - tags),
                    "common_f1": float(on_common) if np.isfinite(on_common) else None,
                    "full_truth_f1": float(on_full) if np.isfinite(on_full) else None,
                }
            )
        yield {
            "schema_version": 1,
            "kind": "hard_image_review",
            "review": metadata,
            "id": sample["id"],
            "image": sample["image"],
            "sha256": sample["sha256"],
            "mean_pairwise_jaccard_distance": float(disagreement[position]),
            "ranked_by": ranked_by,
            "annotated": sorted(annotated[position]),
            "common_annotations": sorted(in_vocabulary[position]),
            "annotations_outside_common_vocabulary": sorted(truth[position] - vocabulary),
            "runs": model_results,
            "human_review": {"notes": ""},
        }


def export_hard_images(
    paths: list[Path],
    *,
    category: str,
    output: Path,
    top: int = 25,
    tag_mapping: TagMapping | None = None,
    cache_dir: Path | None = None,
) -> int:
    """Export images ranked by disagreement and low common-output F1."""
    if len(paths) < 2 or len({path.resolve() for path in paths}) != len(paths):
        raise BenchmarkError("Hard-image review needs at least two distinct runs.")
    positive_int(top, name="Images per ranking")
    with ExitStack() as stack:
        runs = [stack.enter_context(load_results(path, cache_dir=cache_dir)) for path in paths]
        configs = [read_object(path / "profile.json") for path in paths]
        packet = _packet(runs, configs, category, top, tag_mapping)
        first = next(packet, None)
        output.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with output.open("x", encoding="utf-8", newline="\n") as handle:
            if first is not None:
                handle.write(canonical_json(first) + "\n")
                count = 1
            for item in packet:
                handle.write(canonical_json(item) + "\n")
                count += 1
    return count
