from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from vision_bench.adapters import profile_vocabulary
from vision_bench.artifacts import load_results
from vision_bench.errors import BenchmarkError
from vision_bench.io import file_hash, read_object, write_json
from vision_bench.preprocess import preprocess_image
from vision_bench.profiles import load_profile
from vision_bench.runner import recompute, run_benchmark


def test_anonymous_indexed_output_does_not_require_a_category(tmp_path):
    from vision_bench.adapters import load_vocabulary

    path = tmp_path / "tags.csv"
    path.write_text("id,name,category\n0,known,0\n1,,\n", encoding="utf-8")
    vocabulary = load_vocabulary(path, index_column="id")
    assert vocabulary.labels == ("known", None)
    assert vocabulary.categories == ("general", "unnamed")


def configure_json_model(values, mode):
    import onnx

    root = values["root"]
    names = ["shared", "hero", "extra"]
    mapping = {
        "idx_to_tag": {str(i): value for i, value in enumerate(names)},
        "tag_to_idx": {value: i for i, value in enumerate(names)},
        "tag_to_category": {"shared": "general", "hero": "character", "extra": "general"},
    }
    if mode == "cl-v2":
        vocabulary = {"num_tags": 3, **mapping}
        write_json(
            root / "metadata.json",
            {"vision_encoder_repo": "google/siglip2-so400m-patch14-384", "is_naflex": False, "num_tags": 3},
        )
    else:
        vocabulary = {
            "dataset_info": {"total_tags": 3, "tag_mapping": mapping},
            "model_info": {"img_size": 512},
            "input_spec": {"shape": [3, 512, 512]},
            "output_spec": {"refined_predictions": {"shape": ["batch_size", 3]}},
        }
    write_json(root / "vocabulary.json", vocabulary)
    model = onnx.load(root / "model.onnx")
    new_input = "pixel_values" if mode == "cl-v2" else "input"
    model.graph.input[0].name = new_input
    for dimension in (2, 3):
        model.graph.input[0].type.tensor_type.shape.dim[dimension].dim_value = 384 if mode == "cl-v2" else 512
    output = "logits" if mode == "cl-v2" else "refined_predictions"
    model.graph.output[0].name = output
    model.graph.node[0].output[0] = output
    if mode == "cl-v2":
        onnx.save_model(
            model,
            root / "model.onnx",
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="model.onnx.data",
            size_threshold=0,
        )
    else:
        onnx.save(model, root / "model.onnx")
    profile = read_object(values["profile"])
    profile.update(id=mode, preprocess=mode, output_name=output, output_kind="logits")
    profile["artifacts"] = {
        role: {"path": name, "sha256": file_hash(root / name)}
        for role, name in (("model", "model.onnx"), ("vocabulary", "vocabulary.json"))
    }
    if mode == "cl-v2":
        for role, name in (("metadata", "metadata.json"), ("model_data", "model.onnx.data")):
            profile["artifacts"][role] = {"path": name, "sha256": file_hash(root / name)}
    write_json(values["profile"], profile)


@pytest.mark.parametrize("mode", ["cl-v2", "camie-v2"])
def test_json_models_execute_real_workers_with_correct_head_and_full_vectors(local_inputs, mode):
    configure_json_model(local_inputs, mode)
    path = run_benchmark(
        dataset_path=local_inputs["dataset"],
        profile_path=local_inputs["profile"],
        model_root=local_inputs["root"],
        output_root=local_inputs["output"],
    )
    with load_results(path) as (_, _, vocabulary, scores, valid, _):
        assert vocabulary["labels"] == ["shared", "hero", "extra"]
        expected = 1 / (1 + np.exp(-np.asarray([0.8, 0.7, 0.1])))
        assert scores[0].tolist() == pytest.approx(expected.tolist())
        assert valid.all()
    assert recompute(path)[1]


def test_cl_backward_map_and_external_weight_integrity_are_enforced(local_inputs):
    configure_json_model(local_inputs, "cl-v2")
    values = read_object(local_inputs["root"] / "vocabulary.json")
    values["tag_to_idx"]["hero"] = 0
    write_json(local_inputs["root"] / "vocabulary.json", values)
    with pytest.raises(BenchmarkError, match="indices"):
        profile_vocabulary(load_profile(local_inputs["profile"], model_root=local_inputs["root"]))
    (local_inputs["root"] / "model.onnx.data").write_bytes(b"corrupt weights")
    with pytest.raises(BenchmarkError, match="hash"):
        load_profile(local_inputs["profile"], model_root=local_inputs["root"]).verify()


def test_cl_bicubic_and_camie_padding_follow_distinct_preprocessing_contracts():
    image = Image.new("RGB", (20, 10), (255, 0, 0))
    cl = preprocess_image(image, "cl-v2")
    assert cl.shape == (1, 3, 384, 384)
    assert cl[0, :, 0, 0].tolist() == [1, -1, -1]
    camie = preprocess_image(image, "camie-v2")
    assert camie.shape == (1, 3, 512, 512)
    assert camie[0, :, 0, 0].tolist() == pytest.approx(
        ((np.asarray([124, 116, 104]) / 255 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]).tolist(),
        abs=1e-7,
    )


def test_all_six_builtin_profiles_resolve_from_package_resources(tmp_path):
    for name in ("caformer-b36", "pixai-v0.9", "wd-eva02-v3", "cl-v2", "camie-v2", "caformer-b36-amp-fp16"):
        profile = load_profile(f"builtin:{name}", model_root=tmp_path)
        assert profile.config["source"]["revision"]
        json.dumps(profile.config, allow_nan=False)
