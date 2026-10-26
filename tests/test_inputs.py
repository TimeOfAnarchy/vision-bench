from pathlib import Path

import pytest

from vision_bench.adapters import load_vocabulary
from vision_bench.dataset import load_dataset, normalize_tag
from vision_bench.errors import BenchmarkError
from vision_bench.io import child_path, read_object, write_json
from vision_bench.profiles import load_profile


@pytest.mark.parametrize("relative", ["../outside", "a/../b", "/absolute", "C:/other", "a\\b", "a//b"])
def test_relative_asset_paths_reject_escape(tmp_path, relative):
    with pytest.raises(BenchmarkError):
        child_path(tmp_path, relative)


def test_tag_normalization_does_not_guess_aliases():
    assert normalize_tag("  Ｈｅｒｏ__ Name  ") == "hero_name"
    assert normalize_tag("hero_(franchise)") != normalize_tag("hero")


def test_vocabulary_rejects_normalized_collisions_and_wrong_output_indices(tmp_path):
    path = tmp_path / "tags.csv"
    path.write_text("name,category\nHero Name,4\nhero_name,4\n", encoding="utf-8")
    with pytest.raises(BenchmarkError, match="collisions"):
        load_vocabulary(path)
    path.write_text("id,name,category\n1,hero,4\n", encoding="utf-8")
    with pytest.raises(BenchmarkError, match="ordered"):
        load_vocabulary(path, index_column="id")


def test_dataset_subset_hash_and_image_verification(local_inputs):
    values = local_inputs
    full = load_dataset(values["dataset"], verify_images=True)
    prefix = load_dataset(values["dataset"], limit=1)
    assert full.fingerprint != prefix.fingerprint
    (values["root"] / "image-0.png").write_bytes(b"changed")
    with pytest.raises(BenchmarkError, match="Image hash"):
        load_dataset(values["dataset"], verify_images=True)


def test_dataset_refuses_modified_sample_file(local_inputs):
    values = local_inputs
    with (values["root"] / "samples.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(BenchmarkError, match="sample file hash"):
        load_dataset(values["dataset"])


def test_model_hash_and_threshold_validation(local_inputs):
    values = local_inputs
    profile = load_profile(values["profile"], model_root=values["root"])
    profile.verify()
    (values["root"] / "model.onnx").write_bytes(b"changed")
    with pytest.raises(BenchmarkError, match="hash mismatch"):
        profile.verify()
    config = read_object(values["profile"])
    config["thresholds"]["general"] = True
    write_json(values["profile"], config)
    with pytest.raises(BenchmarkError, match="threshold"):
        load_profile(values["profile"], model_root=values["root"])


def test_bundled_profiles_are_package_resources(tmp_path: Path):
    for name in ("caformer-b36", "pixai-v0.9"):
        profile = load_profile(f"builtin:{name}", model_root=tmp_path)
        assert profile.config["provider"] == "CPUExecutionProvider"


def test_torch_caformer_profile_requires_explicit_runtime_contract(local_inputs):
    config = read_object(local_inputs["profile"])
    config.update(
        backend="torch-caformer",
        provider="TorchCUDA",
        precision="amp-fp16",
        preprocess="caformer-b36-direct-640",
        output_name="logits",
        output_kind="logits",
    )
    # These placeholder hashes test profile validation without importing torch or loading weights.
    config["artifacts"].update(
        {role: {"path": f"{role}.json", "sha256": "0" * 64} for role in ("config", "preprocess")}
    )
    write_json(local_inputs["profile"], config)
    assert (
        load_profile(local_inputs["profile"], model_root=local_inputs["root"]).config["precision"]
        == "amp-fp16"
    )
    for key, value in (
        ("provider", "CUDAExecutionProvider"),
        ("precision", "int8"),
        ("preprocess", "pixai-v1.0"),
        ("output_kind", "probabilities"),
    ):
        write_json(local_inputs["profile"], {**config, key: value})
        with pytest.raises(BenchmarkError):
            load_profile(local_inputs["profile"], model_root=local_inputs["root"])


def test_indexed_vocabulary_preserves_unnamed_output_slots(tmp_path):
    path = tmp_path / "tags.csv"
    path.write_text("id,name,category\n0,known,0\n1,,0\n2,hero,4\n3,,4\n", encoding="utf-8")
    vocabulary = load_vocabulary(path, index_column="id")
    assert vocabulary.labels == ("known", None, "hero", None)
    assert vocabulary.categories == ("general", "unnamed", "character", "unnamed")
    assert vocabulary.to_dict()["unnamed_outputs"] == 2
