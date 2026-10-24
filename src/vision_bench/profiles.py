from __future__ import annotations

import math
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path

from .errors import BenchmarkError
from .io import child_path, content_hash, file_hash, identifier, read_object, sha256_value

PREPROCESS_MODES = {"pixai-v0.9", "caformer-b36", "wd-v3", "pixai-v1.0"}
PROVIDERS = {"CPUExecutionProvider", "DmlExecutionProvider", "CUDAExecutionProvider"}


@dataclass(frozen=True)
class ModelProfile:
    config: dict
    model_root: Path

    @property
    def profile_id(self) -> str:
        return self.config["id"]

    @property
    def fingerprint(self) -> str:
        return content_hash(self.config)

    def artifact(self, role: str) -> Path:
        return child_path(self.model_root, self.config["artifacts"][role]["path"])

    def verify(self) -> None:
        for role, artifact in self.config["artifacts"].items():
            path = self.artifact(role)
            if file_hash(path) != artifact["sha256"]:
                raise BenchmarkError(f"Model artifact hash mismatch: {role!r} ({path}).")


def load_profile(path: str | Path, *, model_root: Path, provider: str | None = None) -> ModelProfile:
    if str(path).startswith("builtin:"):
        name = identifier(str(path).removeprefix("builtin:"))
        import json

        try:
            config = json.loads(files("vision_bench").joinpath(f"profiles/{name}.json").read_text("utf-8"))
        except (OSError, ValueError) as exc:
            raise BenchmarkError(f"Unknown/invalid bundled profile: {name}") from exc
    else:
        config = read_object(Path(path))
    if config.get("schema_version") != 1:
        raise BenchmarkError("Model profile schema_version must be 1.")
    identifier(config.get("id"))
    if config.get("preprocess") not in PREPROCESS_MODES:
        raise BenchmarkError("Unknown preprocessing contract.")
    if provider is not None:
        config["provider"] = provider
    if config.get("provider") not in PROVIDERS:
        raise BenchmarkError("An explicit supported execution provider is required.")
    if config.get("output_kind") not in {"probabilities", "logits"}:
        raise BenchmarkError("output_kind must be probabilities or logits.")
    if not isinstance(config.get("output_name"), str) or not config["output_name"]:
        raise BenchmarkError("An explicit output_name is required.")
    thresholds = config.get("thresholds")
    if not isinstance(thresholds, dict) or not thresholds:
        raise BenchmarkError("Explicit per-category thresholds are required.")
    for category, value in thresholds.items():
        identifier(category)
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
            or not 0 <= value <= 1
        ):
            raise BenchmarkError(f"Invalid threshold for {category!r}.")
    artifacts = config.get("artifacts")
    if not isinstance(artifacts, dict) or not {"model", "vocabulary"} <= artifacts.keys():
        raise BenchmarkError("Profile needs model and vocabulary artifacts.")
    for role, artifact in artifacts.items():
        identifier(role)
        if not isinstance(artifact, dict):
            raise BenchmarkError("Artifact descriptors must be objects.")
        child_path(model_root, artifact.get("path"))
        sha256_value(artifact.get("sha256"))
    return ModelProfile(config, model_root.resolve())
