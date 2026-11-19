import numpy as np
import pytest

from vision_bench.metrics import evaluate_columns
from vision_bench.sweep import threshold_analysis


def test_sweep_counts_match_fixed_threshold_metrics_including_boundary_ties():
    rng = np.random.default_rng(4)
    labels = [f"tag-{index}" for index in range(7)]
    truth = [{label for label in labels if rng.random() < 0.3} | {"outside"} for _ in range(9)]
    scores = np.round(rng.random((10, 8)), 2).astype(np.float32)
    scores[1, 2] = np.float32(0.35)  # Exactly on a grid threshold: inclusive.
    rows, columns = np.arange(1, 10), [7, 0, 1, 2, 3, 4, 5]
    result = threshold_analysis(scores, rows, columns, labels, truth, block_size=3)
    assert result["pairs"] == 9 * 7
    for threshold in (0.0, 0.17, 0.35, 0.5, 0.99, 1.0):
        index = round(threshold * 1000)
        fixed = evaluate_columns(scores, rows, columns, labels, truth, threshold)["micro_representable"]
        assert result["true_positive"][index] == fixed["true_positive"]
        assert result["unlabeled_positive"][index] == fixed["unlabeled_positive"]
    f1 = [
        evaluate_columns(scores, rows, columns, labels, truth, index / 1000)["micro_representable"]["f1"]
        for index in range(1001)
    ]
    assert result["best_f1"]["f1"] == pytest.approx(max(f1))
    assert result["best_f1"]["threshold"] == f1.index(max(f1)) / 1000
    assert result["best_f1"]["fitted_on_evaluation_set"] is True


def test_operating_targets_and_calibration_bins():
    scores = np.asarray([[0.95, 1.0], [0.15, 0.97], [0.05, 0.62]], dtype=np.float32)
    truth = [{"a"}, {"b"}, set()]
    result = threshold_analysis(scores, np.arange(3), [0, 1], ["a", "b"], truth)
    assert result["annotated_positives"] == 2
    at_recall = result["precision_at_recall"]["0.5"]
    assert at_recall["threshold"] == 0.621
    assert (at_recall["precision"], at_recall["recall"]) == pytest.approx((2 / 3, 1.0))
    assert result["best_f1"]["f1"] == pytest.approx(0.8)
    assert result["recall_at_precision"]["0.8"] is None  # A negative holds the top score.
    assert result["recall_at_precision"]["0.5"]["recall"] == 1.0
    bins = result["calibration"]["bins"]
    assert [item["count"] for item in bins] == [1, 1, 0, 0, 0, 0, 1, 0, 0, 3]
    assert bins[9]["annotated_positive_rate"] == pytest.approx(2 / 3)
    assert bins[9]["mean_score"] == pytest.approx((0.95 + 1.0 + 0.97) / 3)
    assert bins[2]["mean_score"] is None
    assert result["calibration"]["expected_calibration_error"] == pytest.approx(
        (0.05 + 0.15 + 0.62 + 0.92) / 6, abs=1e-6
    )
    upper = result["calibration"]["upper_bins"]
    assert upper["pairs"] == 5
    assert upper["expected_calibration_error"] == pytest.approx((0.15 + 0.62 + 0.92) / 5, abs=1e-6)


def test_no_annotated_positives_has_no_best_threshold():
    result = threshold_analysis(np.asarray([[0.7]], dtype=np.float32), np.arange(1), [0], ["a"], [set()])
    assert result["best_f1"] is None
    assert result["precision_at_recall"]["0.5"] is None
