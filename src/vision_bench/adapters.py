from __future__ import annotations

import csv
import sys
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from .dataset import normalize_tag
from .errors import BenchmarkError
from .io import read_object
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


def load_vocabulary(path: Path, *, index_column: str | None = None) -> Vocabulary:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels, categories = [], []
    for index, row in enumerate(rows):
        try:
            if index_column is not None and int(row[index_column]) != index:
                raise BenchmarkError("Vocabulary output indices must be contiguous and ordered from zero.")
            raw_label = row["name"]
            category = CATEGORIES[int(row["category"])]
            if index_column is not None and isinstance(raw_label, str) and not raw_label.strip():
                label, category = None, "unnamed"
            else:
                label = normalize_tag(raw_label)
        except (KeyError, TypeError, ValueError) as exc:
            raise BenchmarkError(f"Invalid vocabulary row {index}: {exc}") from exc
        labels.append(label)
        categories.append(category)
    named = [label for label in labels if label is not None]
    if not named or len(set(named)) != len(named):
        raise BenchmarkError("Vocabulary must be nonempty without normalized label collisions.")
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
            ort.preload_dlls()
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
        self.vocabulary = load_vocabulary(
            profile.artifact("vocabulary"),
            index_column="id" if profile.config["preprocess"] == "pixai-v0.9" else None,
        )

    def load(self) -> dict:
        config = self.profile.config
        session = self._factory(self.profile.artifact("model"), config["provider"])
        if not session.get_providers() or session.get_providers()[0] != config["provider"]:
            raise BenchmarkError("Primary provider fallback is forbidden.")
        inputs = session.get_inputs()
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
