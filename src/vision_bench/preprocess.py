from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
from PIL import Image

from .errors import BenchmarkError

PREPROCESS_VERSION = "image-taggers-v1"
INPUT_SHAPES = {
    "pixai-v0.9": (3, 448, 448),
    "caformer-b36": (3, 384, 384),
    "wd-v3": (448, 448, 3),
    "pixai-v1.0": (3, 1008, 1008),
}


def image_tensor(path: Path, mode: str) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(path) as image:
            if max(image.size) > 8192 or image.width * image.height > 12_000_000:
                raise BenchmarkError("Image exceeds the 8192-edge / 12MP decode limit.")
            if getattr(image, "n_frames", 1) != 1:
                raise BenchmarkError("Only single-frame images are supported.")
            return preprocess_image(image, mode)


def preprocess_image(image: Image.Image, mode: str) -> np.ndarray:
    """Model-specific contracts; original pixel orientation, no implicit EXIF transpose."""
    if mode == "pixai-v0.9":
        rgb = image.convert("RGB").resize((448, 448), Image.Resampling.BILINEAR)
        array = np.asarray(rgb, dtype=np.float32) / 255.0
        return np.ascontiguousarray(((array - 0.5) / 0.5).transpose(2, 0, 1)[None])
    if mode not in INPUT_SHAPES:
        raise BenchmarkError(f"Unknown preprocessing contract: {mode}")
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", image.size, "white")
    background.alpha_composite(rgba)
    rgb = background.convert("RGB")
    if mode == "wd-v3":
        side = max(rgb.size)
        canvas = Image.new("RGB", (side, side), "white")
        canvas.paste(rgb, ((side - rgb.width) // 2, (side - rgb.height) // 2))
        array = np.asarray(canvas.resize((448, 448), Image.Resampling.BICUBIC), dtype=np.float32)
        return np.ascontiguousarray(array[None, :, :, ::-1])
    side = 1008 if mode == "pixai-v1.0" else 512
    interpolation = Image.Resampling.BICUBIC if mode == "pixai-v1.0" else Image.Resampling.BILINEAR
    ratio = side / max(rgb.size)
    resized = rgb.resize((max(1, round(rgb.width * ratio)), max(1, round(rgb.height * ratio))), interpolation)
    canvas = Image.new("RGB", (side, side), "black" if mode == "pixai-v1.0" else "white")
    canvas.paste(resized, ((side - resized.width) // 2, (side - resized.height) // 2))
    if mode == "caformer-b36":
        canvas = canvas.resize((384, 384), Image.Resampling.BICUBIC)
    array = np.asarray(canvas, dtype=np.float32) / 255.0
    mean, std = (
        ((0.5,) * 3, (0.5,) * 3)
        if mode == "pixai-v1.0"
        else (
            (0.485, 0.456, 0.406),
            (0.229, 0.224, 0.225),
        )
    )
    array = (array - np.asarray(mean, dtype=np.float32)) / np.asarray(std, dtype=np.float32)
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None])
