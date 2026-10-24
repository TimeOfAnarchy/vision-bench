from __future__ import annotations

import subprocess

import numpy as np
import pytest

from vision_bench.artifacts import load_results
from vision_bench.compare import compare_runs
from vision_bench.errors import BenchmarkError
from vision_bench.io import file_hash, read_object, write_json
from vision_bench.runner import recompute, run_benchmark


def run(values, **kwargs):
    return run_benchmark(
        dataset_path=values["dataset"],
        profile_path=values["profile"],
        model_root=values["root"],
        output_root=values["output"],
        **kwargs,
    )


def test_real_worker_preserves_full_vectors_and_exact_offline_recomputation(local_inputs, monkeypatch):
    values = local_inputs
    path = run(values, run_id="first", shard_size=2, warmup=1)
    manifest, samples, vocabulary, scores, valid, records = load_results(path)
    assert scores.shape == (3, 3)
    assert vocabulary["labels"] == ["shared", "hero", "extra"]
    assert valid.tolist() == [True, True, True]
    assert len(list(path.glob("scores-*.npz"))) == 2
    assert (path / "report.html").is_file()
    assert manifest["runtime"]["model"]["primary_provider"] == "CPUExecutionProvider"
    before = {item.name: item.read_bytes() for item in path.iterdir() if item.is_file()}
    monkeypatch.setattr(
        subprocess, "run", lambda *a, **k: pytest.fail("Offline recomputation launched a worker")
    )
    metrics, matches = recompute(path)
    assert matches
    assert metrics["categories"]["general"]["coverage"]["oov_occurrences"] == 3
    assert metrics["performance"]["warm"]["count"] == 2
    assert before == {item.name: item.read_bytes() for item in path.iterdir() if item.is_file()}


def test_sample_failure_is_not_a_zero_score_and_completed_runs_cannot_be_reused(local_inputs):
    values = local_inputs
    (values["root"] / "image-1.png").write_bytes(b"not the original image")
    path = run(values, run_id="with-error")
    _, _, _, scores, valid, records = load_results(path)
    assert valid.tolist() == [True, False, True]
    assert np.isnan(scores[1]).all()
    assert records[1]["error"] is not None
    assert read_object(path / "metrics.json")["outcomes"] == {"attempted": 3, "successful": 2, "failed": 1}
    with pytest.raises(BenchmarkError, match="already exists"):
        run(values, run_id="with-error")


def test_compare_refuses_dataset_drift_and_reports_common_character_metrics(local_inputs):
    values = local_inputs
    first = run(values, run_id="model-a")
    second = run(values, run_id="model-b", shard_size=1)
    comparison = compare_runs([first, second])
    assert comparison["paired_successful"] == 3
    assert comparison["categories"]["character"]["label_count"] == 1
    assert comparison["categories"]["general"]["runs"][0]["metrics"]["coverage"]["oov_occurrences"] == 3
    limited = run(values, run_id="prefix", limit=1)
    with pytest.raises(BenchmarkError, match="different dataset subsets"):
        compare_runs([first, limited])


def test_recompute_rejects_tampered_scores(local_inputs):
    path = run(local_inputs)
    shard = next(path.glob("scores-*.npz"))
    with shard.open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(BenchmarkError, match="integrity"):
        recompute(path)


def test_unavailable_provider_creates_failed_diagnostic_run(local_inputs):
    with pytest.raises(BenchmarkError, match="Worker failed"):
        run(local_inputs, run_id="unavailable", provider="CUDAExecutionProvider")
    manifest = read_object(local_inputs["output"] / "unavailable" / "run.json")
    assert manifest["state"] == "failed"
    with pytest.raises(BenchmarkError, match="completed"):
        recompute(local_inputs["output"] / "unavailable")


def test_timeout_leaves_a_failed_run_without_silent_retry(local_inputs, monkeypatch):
    calls = []

    def timed_out(*args, **kwargs):
        calls.append(args)
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", timed_out)
    with pytest.raises(BenchmarkError, match="deadline"):
        run(local_inputs, run_id="timeout", timeout=0.1)
    assert len(calls) == 1
    assert read_object(local_inputs["output"] / "timeout" / "run.json")["state"] == "failed"


def test_matrix_budget_rejects_before_creating_run(local_inputs, monkeypatch):
    import vision_bench.runner as runner

    monkeypatch.setattr(runner, "MAX_SCORE_BYTES", 1)
    with pytest.raises(BenchmarkError, match="512 MiB"):
        run(local_inputs, run_id="oversized")
    assert not local_inputs["output"].exists()


def test_unnamed_model_outputs_survive_worker_shards_and_recompute(local_inputs):
    import onnx

    values = local_inputs
    model_path = values["root"] / "model.onnx"
    model = onnx.load(model_path)
    model.graph.input[0].type.tensor_type.shape.dim[2].dim_value = 448
    model.graph.input[0].type.tensor_type.shape.dim[3].dim_value = 448
    onnx.save(model, model_path)
    (values["root"] / "tags.csv").write_text(
        "id,name,category\n0,shared,0\n1,,0\n2,extra,0\n", encoding="utf-8"
    )
    profile = read_object(values["profile"])
    profile["preprocess"] = "pixai-v0.9"
    profile["artifacts"]["model"]["sha256"] = file_hash(model_path)
    profile["artifacts"]["vocabulary"]["sha256"] = file_hash(values["root"] / "tags.csv")
    write_json(values["profile"], profile)
    path = run(values)
    _, _, vocabulary, scores, valid, _ = load_results(path)
    assert vocabulary["labels"] == ["shared", None, "extra"]
    assert scores.shape == (3, 3)
    assert scores[0].tolist() == pytest.approx([0.8, 0.7, 0.1])
    metrics, matches = recompute(path)
    assert matches
    assert metrics["model_outputs"] == {"total": 3, "named": 2, "unnamed": 1}
    assert metrics["categories"]["character"]["micro_full_truth"]["false_negative"] == 3
