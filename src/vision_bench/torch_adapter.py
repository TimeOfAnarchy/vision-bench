"""Optional local PyTorch CAFormer inference, without downloading weights or code."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import numpy as np

from .adapters import profile_vocabulary
from .dataset import normalize_tag
from .errors import BenchmarkError
from .io import read_object
from .preprocess import INPUT_SHAPES, image_tensor
from .profiles import ModelProfile


class TorchCAFormer:
    def __init__(self, profile: ModelProfile):
        self.profile = profile
        self.vocabulary = profile_vocabulary(profile)
        self.model = None

    def load(self) -> dict:
        try:
            import timm
            import torch
            from safetensors.torch import load_file
        except ImportError as exc:
            raise BenchmarkError(
                "Torch CAFormer requires torch, timm and safetensors in its runtime."
            ) from exc
        if not torch.cuda.is_available():
            raise BenchmarkError("TorchCUDA requires an available CUDA device.")
        config = read_object(self.profile.artifact("config"))
        if (
            config.get("architecture") != "caformer_b36"
            or config.get("num_classes") != len(self.vocabulary.labels)
            or tuple(normalize_tag(tag) for tag in config.get("tags", [])) != self.vocabulary.labels
        ):
            raise BenchmarkError("CAFormer architecture/class count/tag order differs from its vocabulary.")
        # Construct from local configuration only; strict loading forbids silent head changes.
        self.model = timm.create_model(
            "caformer_b36",
            pretrained=False,
            num_classes=config["num_classes"],
            pretrained_cfg=config["pretrained_cfg"],
            **config.get("model_args", {}),
        )
        self.model.load_state_dict(load_file(str(self.profile.artifact("model"))), strict=True)
        self.model.eval().to("cuda")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.cuda.synchronize()
        return {
            "backend": "torch-caformer",
            "torch": torch.__version__,
            "timm": timm.__version__,
            "primary_provider": "TorchCUDA",
            "providers": ["TorchCUDA"],
            "device": torch.cuda.get_device_name(),
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "precision": self.profile.config["precision"],
            "weights_dtype": "float32",
            "strict_state_dict": True,
            "tf32": False,
            "input_shape": [1, *INPUT_SHAPES[self.profile.config["preprocess"]]],
            "output_name": "logits",
            "output_kind": "logits",
        }

    def infer(self, image_path: Path) -> np.ndarray:
        import torch

        if self.model is None:
            raise BenchmarkError("Model must be loaded before inference.")
        tensor = torch.from_numpy(image_tensor(image_path, self.profile.config["preprocess"])).to("cuda")
        context = (
            torch.autocast("cuda", dtype=torch.float16)
            if self.profile.config["precision"] == "amp-fp16"
            else nullcontext()
        )
        with torch.inference_mode(), context:
            logits = self.model(tensor)
        # Match ONNX adapter: sigmoid is calculated in float32, once.
        scores = logits.float().sigmoid().cpu().numpy()
        torch.cuda.synchronize()
        if scores.shape != (1, len(self.vocabulary.labels)) or not np.isfinite(scores).all():
            raise BenchmarkError("Model returned an invalid complete score vector.")
        return scores[0]
