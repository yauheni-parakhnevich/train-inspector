"""Output validation unit tests (review M7)."""

from pathlib import Path

import numpy as np
import pytest

from train_inspector.encode import JPEG_MAX_DIM, OutputError, validate_output, write_image


def test_jpeg_dimension_limit_rejected():
    with pytest.raises(OutputError, match="JPEG limit"):
        validate_output(Path("out.jpg"), predicted_width=70_000, height=1080,
                        scale=1.0, max_width=100_000)


def test_jpeg_ok_after_scale():
    validate_output(Path("out.jpg"), predicted_width=70_000, height=1080,
                    scale=0.5, max_width=100_000)


def test_png_huge_width_ok_under_cap():
    validate_output(Path("out.png"), predicted_width=JPEG_MAX_DIM + 1, height=1080,
                    scale=1.0, max_width=100_000)


def test_max_width_cap():
    with pytest.raises(OutputError, match="max-width"):
        validate_output(Path("out.png"), predicted_width=200_000, height=1080,
                        scale=1.0, max_width=100_000)


def test_unsupported_format():
    with pytest.raises(OutputError, match="unsupported"):
        validate_output(Path("out.webp"), predicted_width=100, height=100,
                        scale=1.0, max_width=100_000)


def test_write_and_scale(tmp_path):
    img = np.random.default_rng(1).integers(0, 255, (100, 400, 3), dtype=np.uint8)
    out = tmp_path / "x.png"
    write_image(out, img, scale=0.5)
    import cv2
    back = cv2.imread(str(out))
    assert back.shape == (50, 200, 3)
