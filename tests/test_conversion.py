from __future__ import annotations

import json
import subprocess
import sys
import sysconfig
from pathlib import Path

import numpy as np
import pytest

from vision_bench.io import file_hash


def test_conversion_preserves_source_float32_io_and_numerical_output(tmp_path):
    onnx = pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")
    source, output = tmp_path / "source", tmp_path / "converted"
    source.mkdir()
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Add", ["input", "bias"], ["prediction"])],
        "tiny",
        [onnx.helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, [1, 2])],
        [onnx.helper.make_tensor_value_info("prediction", onnx.TensorProto.FLOAT, [1, 2])],
        [onnx.numpy_helper.from_array(np.asarray([[1, 2]], dtype=np.float32), name="bias")],
    )
    onnx.save(
        onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)], ir_version=10),
        source / "model.onnx",
    )
    before = file_hash(source / "model.onnx")
    script = Path(__file__).resolve().parents[1] / "scripts" / "convert_onnx_fp16.py"
    command = [
        sys.executable,
        str(script),
        "--source-dir",
        str(source),
        "--output-dir",
        str(output),
        "--tools-dir",
        sysconfig.get_path("purelib"),
        "--converter",
        "onnxruntime",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert file_hash(source / "model.onnx") == before
    model = onnx.load(output / "model.onnx")
    onnx.checker.check_model(model, full_check=True)
    assert model.graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    assert model.graph.output[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    session = ort.InferenceSession(str(output / "model.onnx"), providers=["CPUExecutionProvider"])
    assert session.run(None, {"input": np.asarray([[3, 4]], dtype=np.float32)})[0].tolist() == [[4, 6]]
    metadata = json.loads((output / "conversion.json").read_text())
    assert metadata["source_model_sha256"] == before
    assert metadata["artifacts"]["model.onnx"]["sha256"] == file_hash(output / "model.onnx")
    assert any(
        metadata["tools"].get(name) for name in ("onnxruntime", "onnxruntime-directml", "onnxruntime-gpu")
    )
    original = (output / "model.onnx").read_bytes()
    assert subprocess.run(command, capture_output=True).returncode != 0
    assert (output / "model.onnx").read_bytes() == original
