"""Generate four synthetic images and an untrained ONNX color example, offline."""

from __future__ import annotations

import argparse
from pathlib import Path

import onnx
from PIL import Image

from vision_bench.io import file_hash, write_json, write_jsonl


def create_demo(output: Path) -> None:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    images, model_root = output / "images", output / "model"
    images.mkdir()
    model_root.mkdir()
    samples = []
    colors = {"red": (255, 0, 0), "green": (0, 255, 0), "blue": (0, 0, 255), "white": (255, 255, 255)}
    for name, color in colors.items():
        image = images / f"{name}.png"
        Image.new("RGB", (32, 32), color).save(image)
        samples.append(
            {
                "id": f"synthetic-{name}",
                "image": f"images/{name}.png",
                "sha256": file_hash(image),
                "tags": {"general": ["red", "green", "blue"] if name == "white" else [name]},
            }
        )
    write_jsonl(output / "samples.jsonl", samples)
    write_json(
        output / "dataset.json",
        {
            "schema_version": 1,
            "id": "synthetic-colors",
            "revision": "v1",
            "annotation_kind": "silver",
            "samples": "samples.jsonl",
            "samples_sha256": file_hash(output / "samples.jsonl"),
            "source": {
                "kind": "generated",
                "note": "Algorithmic color labels; no human review or external images.",
            },
        },
    )
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node("ReduceMean", ["image", "axes"], ["means"], keepdims=0),
            onnx.helper.make_node("Sigmoid", ["means"], ["prediction"]),
        ],
        "untrained-color-demo",
        [onnx.helper.make_tensor_value_info("image", onnx.TensorProto.FLOAT, [1, 3, 384, 384])],
        [onnx.helper.make_tensor_value_info("prediction", onnx.TensorProto.FLOAT, [1, 3])],
        [onnx.helper.make_tensor("axes", onnx.TensorProto.INT64, [2], [2, 3])],
    )
    model = onnx.helper.make_model(graph, opset_imports=[onnx.helper.make_opsetid("", 18)], ir_version=10)
    onnx.checker.check_model(model, full_check=True)
    onnx.save(model, model_root / "model.onnx")
    (model_root / "tags.csv").write_text("name,category\nred,0\ngreen,0\nblue,0\n", encoding="utf-8")
    write_json(
        output / "profile.json",
        {
            "schema_version": 1,
            "id": "untrained-color-demo",
            "preprocess": "caformer-b36",
            "output_name": "prediction",
            "output_kind": "probabilities",
            "provider": "CPUExecutionProvider",
            "thresholds": {"general": 0.5},
            "source": {
                "kind": "generated",
                "note": "Mean normalized RGB channels followed by sigmoid; no trained weights.",
            },
            "artifacts": {
                "model": {"path": "model.onnx", "sha256": file_hash(model_root / "model.onnx")},
                "vocabulary": {"path": "tags.csv", "sha256": file_hash(model_root / "tags.csv")},
            },
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True, help="New directory; existing paths are refused"
    )
    create_demo(parser.parse_args().output)
