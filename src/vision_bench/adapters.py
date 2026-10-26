from __future__ import annotations

import csv
import os
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .dataset import normalize_tag
from .errors import BenchmarkError
from .io import identifier, positive_int, read_object
from .preprocess import INPUT_SHAPES, image_tensor
from .profiles import ModelProfile

CATEGORIES = {0: "general", 1: "artist", 3: "copyright", 4: "character", 5: "meta", 9: "rating"}


@dataclass(frozen=True)
class Vocabulary:
    labels: tuple[str | None, ...]
    categories: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "labels": self.labels,
            "categories": self.categories,
            "unnamed_outputs": self.labels.count(None),
        }


def load_vocabulary(
    path: Path, *, index_column: str | None = None, categories: dict[int, str] | None = None
) -> Vocabulary:
    categories = CATEGORIES if categories is None else categories
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels, kinds = [], []
    for index, row in enumerate(rows):
        try:
            if index_column is not None and int(row[index_column]) != index:
                raise BenchmarkError("Vocabulary output indices must be contiguous and ordered from zero.")
            raw_label = row["name"]
            if index_column is not None and isinstance(raw_label, str) and not raw_label.strip():
                label, category = None, "unnamed"
            else:
                category = categories[int(row["category"])]
                label = normalize_tag(raw_label)
        except (KeyError, TypeError, ValueError) as exc:
            raise BenchmarkError(f"Invalid vocabulary row {index}: {exc}") from exc
        labels.append(label)
        kinds.append(category)
    named = [label for label in labels if label is not None]
    if not named or len(set(named)) != len(named):
        raise BenchmarkError("Vocabulary must be nonempty without normalized label collisions.")
    return Vocabulary(tuple(labels), tuple(kinds))


def profile_vocabulary(profile: ModelProfile) -> Vocabulary:
    mode = profile.config["preprocess"]
    if mode == "pixai-v1.0":
        return load_vocabulary(
            profile.artifact("vocabulary"), index_column="tag_id", categories={**CATEGORIES, 1: "style"}
        )
    if mode not in {"cl-v2", "camie-v2"}:
        return load_vocabulary(
            profile.artifact("vocabulary"), index_column="id" if mode == "pixai-v0.9" else None
        )
    payload = read_object(profile.artifact("vocabulary"))
    if mode == "cl-v2":
        count = positive_int(payload.get("num_tags"), name="num_tags")
        mapping = payload
        metadata = read_object(profile.artifact("metadata"))
        if (
            metadata.get("num_tags") != count
            or metadata.get("vision_encoder_repo") != "google/siglip2-so400m-patch14-384"
            or metadata.get("is_naflex") is not False
        ):
            raise BenchmarkError("CL metadata does not describe the fixed-size SigLIP2 384 encoder.")
    else:
        dataset = payload.get("dataset_info", {})
        if not isinstance(dataset, dict):
            raise BenchmarkError("Camie dataset_info must be an object.")
        count = positive_int(dataset.get("total_tags"), name="total_tags")
        mapping = dataset.get("tag_mapping", {})
        model, input_spec, output_spec = (
            payload.get(key) for key in ("model_info", "input_spec", "output_spec")
        )
        if (
            not all(isinstance(item, dict) for item in (model, input_spec, output_spec))
            or model.get("img_size") != 512
            or input_spec.get("shape") != [3, 512, 512]
            or not isinstance(output_spec.get("refined_predictions"), dict)
            or output_spec["refined_predictions"].get("shape") != ["batch_size", count]
        ):
            raise BenchmarkError("Camie metadata does not describe the 512px refined-logit model.")
    if not isinstance(mapping, dict) or not all(
        isinstance(mapping.get(key), dict) for key in ("idx_to_tag", "tag_to_category")
    ):
        raise BenchmarkError("JSON vocabulary must declare indexed labels and category mappings.")
    if mode == "cl-v2" and not isinstance(mapping.get("tag_to_idx"), dict):
        raise BenchmarkError("CL vocabulary is missing its backward index mapping.")
    labels, categories = [], []
    for index in range(count):
        raw_label = mapping.get("idx_to_tag", {}).get(str(index))
        label = normalize_tag(raw_label)
        category = mapping.get("tag_to_category", {}).get(raw_label)
        if not isinstance(category, str):
            raise BenchmarkError(f"Missing category at vocabulary index {index}.")
        category = identifier(category.strip().casefold())
        if mode == "cl-v2" and mapping.get("tag_to_idx", {}).get(raw_label) != index:
            raise BenchmarkError("CL vocabulary forward/backward indices disagree.")
        labels.append(label)
        categories.append(category)
    if len(set(labels)) != count:
        raise BenchmarkError("Normalized JSON vocabulary labels collide.")
    return Vocabulary(tuple(labels), tuple(categories))


def create_session(path: Path, provider: str):
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise BenchmarkError("Install one ONNX runtime extra: cpu, dml, or cuda.") from exc
    if provider not in ort.get_available_providers():
        raise BenchmarkError(f"Requested provider is unavailable: {provider}")
    if provider == "CUDAExecutionProvider" and hasattr(ort, "preload_dlls"):
        with redirect_stdout(sys.stderr):
            cudnn_dir = os.environ.get("VISION_BENCH_CUDNN_DIR")
            ort.preload_dlls(cudnn=not bool(cudnn_dir), directory="")
            if cudnn_dir:
                if not Path(cudnn_dir).is_dir():
                    raise BenchmarkError("VISION_BENCH_CUDNN_DIR is not a directory.")
                ort.preload_dlls(cuda=False, directory=str(Path(cudnn_dir).resolve()))
    options = ort.SessionOptions()
    options.enable_mem_pattern = provider != "DmlExecutionProvider"
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    providers = [provider] if provider == "CPUExecutionProvider" else [provider, "CPUExecutionProvider"]
    session = ort.InferenceSession(str(path), sess_options=options, providers=providers)
    session.disable_fallback()
    return session


class OnnxTagger:
    """A single explicit ONNX model, with no global session or environment mutation."""

    def __init__(self, profile: ModelProfile, *, session_factory: Callable = create_session):
        self.profile = profile
        self._factory = session_factory
        self._session = None
        self.vocabulary = profile_vocabulary(profile)

    def load(self) -> dict:
        config = self.profile.config
        session = self._factory(self.profile.artifact("model"), config["provider"])
        if not session.get_providers() or session.get_providers()[0] != config["provider"]:
            raise BenchmarkError("Primary provider fallback is forbidden.")
        inputs = session.get_inputs()
        expected_name = {"cl-v2": "pixel_values", "camie-v2": "input"}.get(config["preprocess"])
        if expected_name is not None and (len(inputs) != 1 or inputs[0].name != expected_name):
            raise BenchmarkError(f"Expected model input {expected_name!r}.")
        expected = INPUT_SHAPES[config["preprocess"]]
        shape = list(inputs[0].shape) if len(inputs) == 1 else []
        if (
            len(shape) != 4
            or inputs[0].type != "tensor(float)"
            or any(
                isinstance(actual, int) and actual != wanted
                for actual, wanted in zip(shape[1:], expected, strict=True)
            )
        ):
            raise BenchmarkError(f"Model input does not match float32 [1, {expected}].")
        channel_axis = 3 if config["preprocess"] == "wd-v3" else 1
        if shape[channel_axis] != 3 or (isinstance(shape[0], int) and shape[0] != 1):
            raise BenchmarkError("Input must declare three channels and accept a batch of one.")
        outputs = [out for out in session.get_outputs() if out.name == config["output_name"]]
        if (
            len(outputs) != 1
            or len(outputs[0].shape) != 2
            or outputs[0].shape[-1] != len(self.vocabulary.labels)
        ):
            raise BenchmarkError("Declared output does not match the complete vocabulary.")
        if "preprocess" in config["artifacts"]:
            # Retain and verify provenance even though inference uses a versioned implementation.
            read_object(self.profile.artifact("preprocess"))
        self._session, self._input = session, inputs[0].name
        import onnxruntime as ort

        return {
            "onnxruntime": ort.__version__,
            "providers": session.get_providers(),
            "primary_provider": config["provider"],
            "input_shape": shape,
            "output_name": config["output_name"],
            "output_kind": config["output_kind"],
            "cpu_graph_partitions_allowed": config["provider"] != "CPUExecutionProvider",
            "cuda_cudnn_directory": os.environ.get("VISION_BENCH_CUDNN_DIR")
            if config["provider"] == "CUDAExecutionProvider"
            else None,
        }

    def infer(self, image_path: Path) -> np.ndarray:
        if self._session is None:
            raise BenchmarkError("Model must be loaded before inference.")
        config = self.profile.config
        tensor = image_tensor(image_path, config["preprocess"])
        output = np.asarray(self._session.run([config["output_name"]], {self._input: tensor})[0])
        if output.shape != (1, len(self.vocabulary.labels)) or not np.isfinite(output).all():
            raise BenchmarkError("Model returned an invalid complete score vector.")
        scores = np.asarray(output[0], dtype=np.float32)
        if config["output_kind"] == "logits":
            scores = np.exp(-np.logaddexp(np.float32(0), -scores))
        if np.any((scores < 0) | (scores > 1)):
            raise BenchmarkError("Declared probabilities must be within [0, 1].")
        return scores
