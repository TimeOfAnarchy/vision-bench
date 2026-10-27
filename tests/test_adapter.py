import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from vision_bench.adapters import OnnxTagger, create_session
from vision_bench.errors import BenchmarkError
from vision_bench.io import read_object, write_json
from vision_bench.preprocess import image_tensor
from vision_bench.profiles import load_profile


class Session:
    def __init__(self, output, provider="CPUExecutionProvider"):
        self.output, self.provider = output, provider

    def get_providers(self):
        return [self.provider]

    def get_inputs(self):
        return [SimpleNamespace(name="image", shape=[1, 3, 384, 384], type="tensor(float)")]

    def get_outputs(self):
        return [SimpleNamespace(name="prediction", shape=[1, 3])]

    def run(self, outputs, inputs):
        assert inputs["image"].shape == (1, 3, 384, 384)
        return [np.asarray([self.output], dtype=np.float32)]


def adapter(values, output, *, provider="CPUExecutionProvider"):
    profile = load_profile(values["profile"], model_root=values["root"])
    return OnnxTagger(profile, session_factory=lambda *args: Session(output, provider))


def test_native_probabilities_are_not_sigmoided_again(local_inputs):
    model = adapter(local_inputs, [0.9, 0.1, 0.4])
    model.load()
    assert model.infer(local_inputs["root"] / "image-0.png").tolist() == pytest.approx([0.9, 0.1, 0.4])


def test_logits_receive_one_stable_sigmoid(local_inputs):
    config = read_object(local_inputs["profile"])
    config["output_kind"] = "logits"
    write_json(local_inputs["profile"], config)
    model = adapter(local_inputs, [-1000, 0, 1000])
    model.load()
    assert model.infer(local_inputs["root"] / "image-0.png").tolist() == [0, 0.5, 1]


def test_primary_provider_fallback_is_rejected(local_inputs):
    with pytest.raises(BenchmarkError, match="fallback"):
        adapter(local_inputs, [0.9, 0.1, 0.4], provider="OtherProvider").load()


@pytest.mark.parametrize("output", [[-0.1, 0, 1], [0, 1, float("nan")], [0, 1]])
def test_invalid_dense_output_is_rejected(local_inputs, output):
    model = adapter(local_inputs, output)
    model.load()
    with pytest.raises(BenchmarkError):
        model.infer(local_inputs["root"] / "image-0.png")


@pytest.mark.parametrize("threads,spinning", [(None, None), (4, False), (2, True)])
def test_session_cpu_limits_preserve_provider_and_disable_fallback(tmp_path, monkeypatch, threads, spinning):
    captured = {}

    class Options:
        intra_op_num_threads = 0

        def add_session_config_entry(self, key, value):
            captured[key] = value

    class RuntimeSession:
        def __init__(self, path, *, sess_options, providers):
            captured.update(options=sess_options, providers=providers)

        def disable_fallback(self):
            captured["fallback_disabled"] = True

    runtime = SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"],
        SessionOptions=Options,
        ExecutionMode=SimpleNamespace(ORT_SEQUENTIAL="sequential"),
        InferenceSession=RuntimeSession,
    )
    monkeypatch.setitem(sys.modules, "onnxruntime", runtime)
    create_session(
        tmp_path / "model.onnx", "CPUExecutionProvider", cpu_threads=threads, allow_spinning=spinning
    )
    assert captured["options"].intra_op_num_threads == (threads or 0)
    assert captured["providers"] == ["CPUExecutionProvider"] and captured["fallback_disabled"]
    for pool in ("intra_op", "inter_op"):
        key = f"session.{pool}.allow_spinning"
        assert (key not in captured) if spinning is None else captured[key] == str(int(spinning))


@pytest.mark.parametrize("options", [{"cpu_threads": 0}, {"cpu_threads": True}, {"allow_spinning": "false"}])
def test_invalid_session_limits_are_rejected(tmp_path, options):
    with pytest.raises(BenchmarkError):
        create_session(tmp_path / "model.onnx", "CPUExecutionProvider", **options)


def test_optional_tall_image_limit_preserves_the_default_decode_guard(tmp_path):
    path = tmp_path / "tall.png"
    Image.new("RGB", (8, 12275), "blue").save(path)
    with pytest.raises(BenchmarkError, match="8192-edge"):
        image_tensor(path, "caformer-b36")
    assert image_tensor(path, "caformer-b36", max_edge=16384).shape == (1, 3, 384, 384)
    with pytest.raises(BenchmarkError, match="16384"):
        image_tensor(path, "caformer-b36", max_edge=16385)


def test_extended_edge_limit_does_not_relax_pixel_area_guard(tmp_path):
    path = tmp_path / "too-many-pixels.png"
    Image.new("RGB", (4000, 4000), "blue").save(path)
    with pytest.raises(BenchmarkError, match="12MP"):
        image_tensor(path, "caformer-b36", max_edge=16384)
