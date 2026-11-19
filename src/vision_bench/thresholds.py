from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from .dataset import normalize_tag
from .errors import BenchmarkError
from .io import sha256_value


def valid_threshold(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0 <= value <= 1


@dataclass(frozen=True)
class Thresholds:
    default: float
    overrides: dict[str, float] = field(default_factory=dict)

    @classmethod
    def for_category(cls, profile: dict, category: str) -> Thresholds:
        return cls(profile["thresholds"][category], profile.get("tag_thresholds", {}).get(category, {}))

    def for_labels(self, labels: list[str]) -> float | np.ndarray:
        if not self.overrides:
            return self.default
        return np.asarray([self.overrides.get(label, self.default) for label in labels], dtype=np.float32)

    def describe(self, labels: list[str]) -> dict:
        result = {"threshold": self.default}
        overrides = {label: self.overrides[label] for label in labels if label in self.overrides}
        if overrides:
            result["tag_thresholds"] = overrides
        return result


def validate_tag_thresholds(profile: dict) -> None:
    overrides = profile.get("tag_thresholds", {})
    if not isinstance(overrides, dict) or not overrides.keys() <= profile["thresholds"].keys():
        raise BenchmarkError("Tag thresholds require their category's fallback threshold.")
    for category, tags in overrides.items():
        if not isinstance(tags, dict) or any(
            not tag or normalize_tag(tag) != tag or not valid_threshold(value) for tag, value in tags.items()
        ):
            raise BenchmarkError(f"Invalid tag thresholds for {category!r}; use normalized tag names.")


def calibration_hashes(profile: dict) -> set[str]:
    if "threshold_fit" not in profile:
        return set()
    fit = profile["threshold_fit"]
    hashes = fit.get("calibration_image_sha256") if isinstance(fit, dict) else None
    if not isinstance(hashes, list) or not hashes:
        raise BenchmarkError("Threshold fit provenance requires calibration image hashes.")
    return {sha256_value(value) for value in hashes}


def verify_calibration_split(profile: dict, samples: Iterable[dict]) -> None:
    hashes = calibration_hashes(profile)
    if hashes:
        overlap = hashes.intersection(sample["sha256"] for sample in samples)
        if overlap:
            raise BenchmarkError(
                f"Calibration and evaluation overlap by {len(overlap)} image SHA-256 hash(es)."
            )
