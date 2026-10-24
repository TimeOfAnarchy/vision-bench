import numpy as np
import pytest
from PIL import Image

from vision_bench.preprocess import preprocess_image


def test_pixai_v09_rgb_bilinear_normalization():
    tensor = preprocess_image(Image.new("RGB", (3, 5), (0, 128, 255)), "pixai-v0.9")
    assert tensor.shape == (1, 3, 448, 448)
    assert tensor.dtype == np.float32
    assert tensor[0, :, 0, 0].tolist() == pytest.approx([-1, 128 / 255 * 2 - 1, 1], abs=1e-7)


def test_caformer_keeps_white_padding_and_imagenet_normalization():
    image = Image.new("RGBA", (20, 10), (255, 0, 0, 255))
    tensor = preprocess_image(image, "caformer-b36")
    assert tensor.shape == (1, 3, 384, 384)
    assert tensor[0, :, 0, 0].tolist() == pytest.approx(
        [(1 - 0.485) / 0.229, (1 - 0.456) / 0.224, (1 - 0.406) / 0.225]
    )
    assert tensor[0, :, 192, 192].tolist() == pytest.approx(
        [(1 - 0.485) / 0.229, -0.456 / 0.224, -0.406 / 0.225]
    )


def test_wd_preprocessing_uses_native_bgr_values_without_normalization():
    tensor = preprocess_image(Image.new("RGB", (4, 4), (10, 20, 30)), "wd-v3")
    assert tensor.shape == (1, 448, 448, 3)
    assert tensor[0, 0, 0].tolist() == [30, 20, 10]
