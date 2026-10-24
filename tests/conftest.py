from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vision_bench.io import file_hash, write_json, write_jsonl


@pytest.fixture
def local_inputs(tmp_path: Path) -> dict:
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    root = tmp_path / "inputs"
    root.mkdir()
    samples = []
    for index in range(3):
        path = root / f"image-{index}.png"
        Image.new("RGB", (11, 7), (index * 50, 120, 255)).save(path)
        samples.append(
            {
                "id": f"image:{index}",
                "image": path.name,
                "sha256": file_hash(path),
                "tags": {"general": ["shared", "outside_model"], "character": ["hero"]},
            }
        )
    write_jsonl(root / "samples.jsonl", samples)
    dataset = root / "dataset.json"
    write_json(
        dataset,
        {
            "schema_version": 1,
            "id": "fixture",
            "revision": "v1",
            "annotation_kind": "silver",
            "samples": "samples.jsonl",
            "samples_sha256": file_hash(root / "samples.jsonl"),
        },
    )
    vocabulary = root / "tags.csv"
    vocabulary.write_text("name,category\nshared,0\nhero,4\nextra,0\n", encoding="utf-8")
    graph = onnx.helper.make_graph(
        [onnx.helper.make_node("Identity", ["values"], ["prediction"])],
        "tiny-tagger",
        [onnx.helper.make_tensor_value_info("image", onnx.TensorProto.FLOAT, [1, 3, 384, 384])],
        [onnx.helper.make_tensor_value_info("prediction", onnx.TensorProto.FLOAT, [1, 3])],
        [onnx.numpy_helper.from_array(np.asarray([[0.8, 0.7, 0.1]], dtype=np.float32), name="values")],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)], ir_version=10)
    onnx.save(model, root / "model.onnx")
    profile = root / "profile.json"
    write_json(
        profile,
        {
            "schema_version": 1,
            "id": "tiny",
            "preprocess": "caformer-b36",
            "output_name": "prediction",
            "output_kind": "probabilities",
            "provider": "CPUExecutionProvider",
            "thresholds": {"general": 0.5, "character": 0.5},
            "artifacts": {
                "model": {"path": "model.onnx", "sha256": file_hash(root / "model.onnx")},
                "vocabulary": {"path": "tags.csv", "sha256": file_hash(vocabulary)},
            },
        },
    )
    return {
        "root": root,
        "dataset": dataset,
        "profile": profile,
        "samples": samples,
        "output": tmp_path / "runs",
    }
