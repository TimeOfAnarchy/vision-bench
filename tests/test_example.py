from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from vision_bench.artifacts import load_results
from vision_bench.dataset import load_dataset
from vision_bench.runner import recompute, run_benchmark


def test_shipped_example_runs_input_dependent_model_and_preserves_generated_inputs(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    script = Path(__file__).resolve().parents[1] / "examples/create_demo.py"
    inputs = tmp_path / "inputs"
    command = [sys.executable, str(script), "--output", str(inputs)]
    subprocess.run(command, check=True)
    originals = {p.relative_to(inputs): p.read_bytes() for p in inputs.rglob("*") if p.is_file()}
    dataset = load_dataset(inputs / "dataset.json", verify_images=True)
    assert len(dataset.samples) == 4
    assert dataset.manifest["annotation_kind"] == "silver"
    run = run_benchmark(
        dataset_path=inputs / "dataset.json",
        profile_path=inputs / "profile.json",
        model_root=inputs / "model",
        output_root=tmp_path / "runs",
        warmup=1,
    )
    with load_results(run) as (_, _, vocabulary, scores, valid, _):
        assert vocabulary["labels"] == ["red", "green", "blue"]
        assert valid.tolist() == [True] * 4
        np.testing.assert_array_equal(
            scores >= 0.5,
            [[True, False, False], [False, True, False], [False, False, True], [True, True, True]],
        )
    assert recompute(run)[1]
    assert (run / "report.html").is_file()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert originals == {p.relative_to(inputs): p.read_bytes() for p in inputs.rglob("*") if p.is_file()}
