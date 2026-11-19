import numpy as np
import pytest

from vision_bench.per_image import RANKS, per_image_counts, per_image_summary, sample_metrics


def test_blocked_counts_match_direct_per_image_computation_with_ties():
    rng = np.random.default_rng(5)
    labels = [f"tag-{index}" for index in range(30)]
    truth = [{label for label in labels if rng.random() < 0.3} | {"outside"} for _ in range(8)]
    scores = np.round(rng.random((9, 31)), 1).astype(np.float32)  # Many ties at the k-th score.
    rows, columns = np.arange(1, 9), list(range(1, 31))
    counts = per_image_counts(scores, rows, columns, labels, truth, 0.5, block_size=7)
    for image, row in enumerate(rows):
        values = scores[row, columns]
        positive = np.asarray([label in truth[image] for label in labels])
        assert counts["true_positive"][image] == np.sum(positive & (values >= 0.5))
        assert counts["unlabeled_positive"][image] == np.sum(~positive & (values >= 0.5))
        assert counts["false_negative"][image] == np.sum(positive & (values < 0.5))
        assert counts["oov_positive"][image] == 1
        for rank in RANKS:
            cutoff = np.sort(values)[::-1][rank - 1]
            higher, equal = values > cutoff, values == cutoff
            expected = (
                np.sum(positive & higher) + np.sum(positive & equal) * (rank - higher.sum()) / equal.sum()
            )
            assert counts[f"hits_at_{rank}"][image] == pytest.approx(expected)


def test_fewer_outputs_than_rank_ranks_every_output():
    scores = np.asarray([[0.2, 0.9, 0.4]], dtype=np.float32)
    counts = per_image_counts(scores, np.arange(1), [0, 1, 2], ["a", "b", "c"], [{"a", "c"}], 0.5)
    assert counts["hits_at_5"].tolist() == [2.0]


def test_weighted_sample_metrics_equal_duplicated_images():
    rng = np.random.default_rng(2)
    counts = {
        "true_positive": rng.integers(0, 4, 10),
        "unlabeled_positive": rng.integers(0, 3, 10),
        "false_negative": rng.integers(0, 3, 10),
        "oov_positive": rng.integers(0, 2, 10),
        **{f"hits_at_{rank}": rng.random(10) * 3 for rank in RANKS},
    }
    weights = np.asarray([[3, 0, 1, 2, 0, 1, 1, 0, 2, 0]], dtype=np.float64)
    duplicated = {name: np.repeat(values, weights[0].astype(int)) for name, values in counts.items()}
    weighted = sample_metrics(counts, weights)
    plain = sample_metrics(duplicated, np.ones((1, int(weights.sum()))))
    for name, value in plain.items():
        assert weighted[name][0] == pytest.approx(value[0])


def test_summary_counts_missed_and_empty_images():
    counts = {
        "true_positive": np.asarray([2, 0, 0]),
        "unlabeled_positive": np.asarray([1, 0, 1]),
        "false_negative": np.asarray([0, 1, 0]),
        "oov_positive": np.asarray([1, 0, 0]),
        **{f"hits_at_{rank}": np.asarray([2.0, 1.0, 0.0]) for rank in RANKS},
    }
    summary = per_image_summary(counts, 0.4)
    metrics = summary["metrics"]
    assert metrics["sample_precision"] == {"value": pytest.approx((2 / 3 + 0) / 2), "images": 2}
    assert metrics["sample_recall_full_truth"] == {"value": pytest.approx((2 / 3 + 0) / 2), "images": 2}
    assert metrics["sample_recall_representable"]["value"] == pytest.approx(0.5)
    assert metrics["sample_recall_at_10"] == {"value": 1.0, "images": 2}
    assert metrics["sample_f1_full_truth"]["images"] == 3
    assert summary["images_without_predictions"] == 1
    assert summary["images_with_common_positives_but_no_hits"] == 1
    assert summary["images_with_only_out_of_vocabulary_positives"] == 0
