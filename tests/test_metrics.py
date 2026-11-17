import numpy as np
import pytest

from vision_bench.metrics import average_precision, evaluate_columns


def test_average_precision_groups_tied_predictions():
    assert average_precision(np.asarray([True, False]), np.asarray([0.5, 0.5])) == 0.5
    assert average_precision(np.asarray([False, True]), np.asarray([0.5, 0.5])) == 0.5
    assert average_precision(np.asarray([False]), np.asarray([0.9])) is None


def test_full_truth_counts_out_of_vocabulary_and_threshold_is_inclusive():
    result = evaluate_columns(
        np.asarray([[0.5, 0.4], [0.1, 0.9]], dtype=np.float32),
        np.arange(2),
        [0, 1],
        ["cat", "dog"],
        [{"cat", "outside"}, {"dog"}],
        0.5,
    )
    assert result["micro_representable"]["f1"] == 1
    assert result["micro_full_truth"]["recall"] == pytest.approx(2 / 3)
    assert result["micro_full_truth"]["f1"] == 0.8
    assert result["coverage"]["oov_occurrences"] == 1


def test_unsupported_labels_have_no_ap_but_keep_unlabeled_positives():
    result = evaluate_columns(
        np.asarray([[0.9, 0.8]], dtype=np.float32),
        np.arange(1),
        [0, 1],
        ["known", "unlabeled"],
        [{"known"}],
        0.5,
    )
    assert result["mean_average_precision_supported"] == 1
    assert result["micro_full_truth"]["unlabeled_positive"] == 1


def test_empty_vocabulary_keeps_all_positive_labels_as_misses():
    result = evaluate_columns(np.empty((1, 0), dtype=np.float32), np.arange(1), [], [], [{"outside"}], 0.5)
    assert result["micro_full_truth"]["false_negative"] == 1
    assert result["mean_average_precision_supported"] is None
