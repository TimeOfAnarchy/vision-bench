from types import SimpleNamespace

import numpy as np
import pytest

from vision_bench.adapters import OnnxTagger
from vision_bench.errors import BenchmarkError
from vision_bench.io import read_object, write_json
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
