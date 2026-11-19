import numpy as np
import pytest

from vision_bench import bootstrap
from vision_bench.bootstrap import resample_weights, summarize, weighted_metrics
from vision_bench.metrics import evaluate_columns


def flat(result: dict) -> dict:
    return {
        "micro_full_truth_f1": result["micro_full_truth"]["f1"],
        "micro_full_truth_recall": result["micro_full_truth"]["recall"],
        "micro_representable_f1": result["micro_representable"]["f1"],
        "micro_representable_precision": result["micro_representable"]["precision"],
        "micro_representable_recall": result["micro_representable"]["recall"],
        "macro_f1_supported": result["macro_f1_supported"],
        "mean_average_precision_supported": result["mean_average_precision_supported"],
    }


def test_resample_weights_keep_observed_row_and_are_seeded():
    weights = resample_weights(5, 20, seed=7)
    assert weights.shape == (21, 5)
    assert weights[0].tolist() == [1] * 5
    assert (weights.sum(axis=1) == 5).all()
    assert np.array_equal(weights, resample_weights(5, 20, seed=7))
    assert not np.array_equal(weights, resample_weights(5, 20, seed=8))


def test_weighted_metrics_equal_metrics_on_duplicated_samples(monkeypatch):
    monkeypatch.setattr(bootstrap, "CUTOFF_COLUMNS", 3)  # Exercise grouped AP cut-offs.
    rng = np.random.default_rng(1)
    labels = [f"tag-{index}" for index in range(9)]
    truth = [
        {label for label in labels if rng.random() < 0.3} | ({"outside"} if rng.random() < 0.3 else set())
        for _ in range(12)
    ]
    scores = np.round(rng.random((14, 11)), 1).astype(np.float32)  # Coarse scores create ties.
    rows, columns = np.arange(2, 14), [10, 0, 1, 2, 3, 4, 5, 6, 7]
    weights = resample_weights(len(rows), 6, seed=3)
    statistics = weighted_metrics([(scores, columns, 0.5)], rows, labels, truth, weights, block_size=4)[0]
    for index, row_weights in enumerate(weights.astype(int)):
        selected = np.repeat(np.arange(len(rows)), row_weights)
        expected = flat(
            evaluate_columns(scores, rows[selected], columns, labels, [truth[i] for i in selected], 0.5)
        )
        assert set(statistics) == set(expected)
        for name, value in expected.items():
            if value is None:
                assert np.isnan(statistics[name][index])
            else:
                assert statistics[name][index] == pytest.approx(value, abs=1e-12)


def test_paired_differences_of_identical_runs_are_exactly_zero():
    scores = np.asarray([[0.9, 0.2], [0.4, 0.7], [0.6, 0.1]], dtype=np.float32)
    run = (scores, [0, 1], 0.5)
    weights = resample_weights(3, 50, seed=0)
    result = summarize(
        ["a", "b"], weighted_metrics([run, run], np.arange(3), ["x", "y"], [{"x"}, {"y"}, set()], weights)
    )
    assert [item["run_id"] for item in result["runs"]] == ["a", "b"]
    difference = result["differences_from_reference"][0]
    assert difference["run_id"] == "b"
    for value in difference["metrics"].values():
        assert (value["estimate"], value["low"], value["high"]) == (0, 0, 0)
    assert result["runs"][0]["metrics"]["micro_representable_f1"]["estimate"] == pytest.approx(0.8)
