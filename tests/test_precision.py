from __future__ import annotations

import numpy as np
import pytest

from vision_bench.errors import BenchmarkError
from vision_bench.io import file_hash, read_object, write_json
from vision_bench.metrics import evaluate_scores
from vision_bench.precision import compare_precision
from vision_bench.runner import run_benchmark


def test_metric_blocks_have_exact_chunk_independent_counts_ap_and_oov():
    scores = np.asarray([[0.5, 0.2, 0.9], [0.5, 0.8, 0.3], [0.1, 0.8, 0.2]], dtype=np.float32)
    truth = [{"a", "outside"}, {"b"}, {"b"}]
    values = [evaluate_scores(scores, ["a", "b", "c"], truth, 0.5, block_size=size) for size in (1, 2, 128)]
    assert values[0] == values[1] == values[2]
    assert values[0]["micro_full_truth"]["true_positive"] == 3
    assert values[0]["micro_full_truth"]["false_negative"] == 1
    assert values[0]["per_label"][0]["ap"] == 1


def pair(values):
    import onnx

    profile = read_object(values["profile"])
    profile["source"] = {"repository": "fixture/tiny", "revision": "v1", "precision": "fp32"}
    write_json(values["profile"], profile)
    kwargs = {
        "dataset_path": values["dataset"],
        "profile_path": values["profile"],
        "model_root": values["root"],
        "output_root": values["output"],
    }
    baseline = run_benchmark(**kwargs, run_id="baseline", shard_size=1)
    model = onnx.load(values["root"] / "model.onnx")
    replacement = onnx.numpy_helper.from_array(np.asarray([[0.4, 0.7, 0.1]], dtype=np.float32), name="values")
    model.graph.initializer[0].CopyFrom(replacement)
    onnx.save(model, values["root"] / "model.onnx")
    profile["source"]["precision"] = "fp16"
    profile["artifacts"]["model"]["sha256"] = file_hash(values["root"] / "model.onnx")
    write_json(values["profile"], profile)
    candidate = run_benchmark(**kwargs, run_id="candidate", shard_size=2)
    return baseline, candidate


def test_precision_pair_reports_numerical_error_threshold_flips_and_quality(
    local_inputs, tmp_path, monkeypatch
):
    baseline, candidate = pair(local_inputs)
    cache = tmp_path / "cache"
    before = {
        str(path): path.read_bytes()
        for root in (baseline, candidate)
        for path in root.iterdir()
        if path.is_file()
    }
    result = compare_precision(baseline, candidate, cache_dir=cache)
    assert result["paired_successful"] == 3
    assert result["max_absolute_error"] == pytest.approx(0.4)
    assert result["mean_absolute_error"] == pytest.approx(0.4 / 3)
    assert result["threshold_changes"]["general"]["flips"] == 3
    assert result["changed_threshold_samples"] == 3
    assert result["quality"]["general"]["candidate"]["micro_full_truth"]["recall"] == 0
    assert list(cache.iterdir()) == []
    assert before == {
        str(path): path.read_bytes()
        for root in (baseline, candidate)
        for path in root.iterdir()
        if path.is_file()
    }
    from contextlib import contextmanager

    from vision_bench.artifacts import load_results

    @contextmanager
    def differing_mask(path, **kwargs):
        with load_results(path, **kwargs) as data:
            if path == candidate:
                data[4][0] = False
            yield data

    monkeypatch.setattr("vision_bench.precision.load_results", differing_mask)
    with pytest.raises(BenchmarkError, match="successful sample masks"):
        compare_precision(baseline, candidate)


def test_precision_pair_refuses_different_model_source(local_inputs):
    from vision_bench.artifacts import artifact_metadata

    baseline, candidate = pair(local_inputs)
    profile = read_object(candidate / "profile.json")
    profile["source"]["revision"] = "different-training-run"
    write_json(candidate / "profile.json", profile)
    from vision_bench.io import content_hash

    manifest = read_object(candidate / "run.json")
    manifest["profile_fingerprint"] = content_hash(profile)
    manifest["artifacts"] = artifact_metadata(candidate)
    write_json(candidate / "run.json", manifest)
    with pytest.raises(BenchmarkError, match="same upstream"):
        compare_precision(baseline, candidate)
