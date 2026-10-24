"""Create a separate, provenance-recorded FP16 benchmark pack; preserve originals."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import shutil
import sys
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def installed_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tools-dir", required=True, type=Path)
    parser.add_argument("--prediction-output", help="Retain only this prediction output before conversion.")
    parser.add_argument("--converter", choices=("common", "onnxruntime"), default="common")
    parser.add_argument(
        "--fp32-op", action="append", default=[], help="Keep these sensitive operations in FP32."
    )
    parser.add_argument(
        "--fp32-node-pattern", action="append", default=[], help="Regex for nodes to keep in FP32."
    )
    args = parser.parse_args()
    source = args.source_dir.resolve(strict=True)
    output = args.output_dir.resolve(strict=False)
    if output.exists():
        parser.error(f"Output must be a new directory: {output}")
    sys.path.insert(0, str(args.tools_dir.resolve(strict=True)))
    import onnx

    if args.converter == "onnxruntime":
        from onnxruntime.transformers import float16
    else:
        from onnxconverter_common import float16

    original = source / "model.onnx"
    source_hash = digest(original)
    print(f"Converting {original}", flush=True)
    model = onnx.load(str(original))
    if args.prediction_output:
        retained = [item for item in model.graph.output if item.name == args.prediction_output]
        if len(retained) != 1:
            parser.error("Requested prediction output is missing or ambiguous.")
        del model.graph.output[:]
        model.graph.output.extend(retained)
    blocks = list(dict.fromkeys([*float16.DEFAULT_OP_BLOCK_LIST, *args.fp32_op]))
    patterns = [re.compile(pattern) for pattern in args.fp32_node_pattern]
    nodes = [node.name for node in model.graph.node if any(pattern.search(node.name) for pattern in patterns)]
    converted = float16.convert_float_to_float16(
        model, keep_io_types=True, op_block_list=blocks, node_block_list=nodes
    )
    if args.converter == "onnxruntime":
        from onnxruntime.transformers.onnx_model import OnnxModel

        OnnxModel(converted).topological_sort()
    onnx.checker.check_model(converted, full_check=True)
    output.mkdir(parents=True)
    destination = output / "model.onnx"
    onnx.save(converted, str(destination))
    for name in (
        "selected_tags.csv",
        "categories.json",
        "preprocess.json",
        "thresholds.csv",
        "meta.json",
        "config.json",
        "README.md",
    ):
        if (source / name).is_file():
            shutil.copyfile(source / name, output / name)
    manifest = {
        "source_model_sha256": source_hash,
        "conversion": f"{float16.__name__}.convert_float_to_float16",
        "keep_io_types": True,
        "prediction_output": args.prediction_output,
        "additional_fp32_ops": args.fp32_op,
        "fp32_node_patterns": args.fp32_node_pattern,
        "fp32_nodes": nodes,
        "tools": {
            name: installed_version(name)
            for name in (
                "onnx",
                "onnxconverter-common",
                "onnxruntime",
                "onnxruntime-directml",
                "onnxruntime-gpu",
            )
        },
        "imported_module_versions": {
            name: getattr(sys.modules.get(name), "__version__", None)
            for name in ("onnx", "onnxconverter_common", "onnxruntime", "numpy")
        },
        "artifacts": {
            p.name: {"size_bytes": p.stat().st_size, "sha256": digest(p)}
            for p in sorted(output.iterdir())
            if p.is_file()
        },
    }
    (output / "conversion.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output), "model": manifest["artifacts"]["model.onnx"]}), flush=True)


if __name__ == "__main__":
    main()
