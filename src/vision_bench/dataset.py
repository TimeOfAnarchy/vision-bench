from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path

from .errors import BenchmarkError
from .io import (
    child_path,
    content_hash,
    file_hash,
    identifier,
    positive_int,
    read_jsonl,
    read_object,
    relative_path,
    sha256_value,
)

NORMALIZATION_VERSION = "danbooru-nfkc-casefold-v1"


def normalize_tag(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BenchmarkError("Tags must be nonempty strings.")
    return "_".join(unicodedata.normalize("NFKC", value).strip().casefold().replace("_", " ").split())


@dataclass(frozen=True)
class Dataset:
    manifest: dict
    image_root: Path
    samples: tuple[dict, ...]
    fingerprint: str
    manifest_sha256: str
    samples_sha256: str

    def image_path(self, sample: dict) -> Path:
        return child_path(self.image_root, sample["image"])


def normalized_sample(value: dict) -> dict:
    sample_id = value.get("id")
    if not isinstance(sample_id, str) or not sample_id.strip() or len(sample_id) > 512:
        raise BenchmarkError("Every sample needs a nonempty id (up to 512 characters).")
    image = value.get("image")
    relative_path(image)
    tags = value.get("tags")
    if not isinstance(tags, dict) or not tags:
        raise BenchmarkError(f"Sample {sample_id!r} needs categorized tags, including empty lists.")
    normalized = {}
    for category, items in tags.items():
        identifier(category)
        if not isinstance(items, list):
            raise BenchmarkError(f"Tags for {category!r} must be a list.")
        normalized[category] = sorted({normalize_tag(tag) for tag in items})
    return {"id": sample_id, "image": image, "sha256": sha256_value(value.get("sha256")), "tags": normalized}


def load_dataset(path: Path, *, limit: int | None = None, verify_images: bool = False) -> Dataset:
    path = path.resolve()
    manifest = read_object(path)
    if manifest.get("schema_version") != 1 or manifest.get("annotation_kind") not in {"silver", "human"}:
        raise BenchmarkError("Dataset needs schema_version=1 and annotation_kind=silver or human.")
    identifier(manifest.get("id"))
    if not isinstance(manifest.get("revision"), str) or not manifest["revision"]:
        raise BenchmarkError("Dataset revision must be nonempty.")
    if limit is not None:
        positive_int(limit, name="Sample limit")
    root_value = manifest.get("image_root", ".")
    if not isinstance(root_value, str):
        raise BenchmarkError("image_root must be a path string.")
    root = (path.parent / root_value).resolve()
    samples_path = child_path(path.parent, manifest.get("samples"))
    samples_sha = file_hash(samples_path)
    if samples_sha != sha256_value(manifest.get("samples_sha256")):
        raise BenchmarkError("Dataset sample file hash mismatch.")
    values = [normalized_sample(value) for value in read_jsonl(samples_path)]
    if not values or len({value["id"] for value in values}) != len(values):
        raise BenchmarkError("Dataset samples must be nonempty with unique ids.")
    samples = tuple(values[:limit])
    categories = set(samples[0]["tags"])
    if any(set(sample["tags"]) != categories for sample in samples):
        raise BenchmarkError("Every sample must declare the same annotation categories.")
    if verify_images:
        for sample in samples:
            if file_hash(child_path(root, sample["image"])) != sample["sha256"]:
                raise BenchmarkError(f"Image hash mismatch for {sample['id']!r}.")
    fingerprint = content_hash(
        {
            "dataset_id": manifest["id"],
            "revision": manifest["revision"],
            "annotation_kind": manifest["annotation_kind"],
            "normalization": NORMALIZATION_VERSION,
            "samples": samples,
        }
    )
    return Dataset(manifest, root, samples, fingerprint, file_hash(path), samples_sha)
