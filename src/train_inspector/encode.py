"""Output encoding per spec FR-5: format dimension limits validated *before*
compositing (review M7), optional downscale, image write.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

JPEG_MAX_DIM = 65_535  # hard format limit

_JPEG_EXTS = {".jpg", ".jpeg"}
_SUPPORTED_EXTS = _JPEG_EXTS | {".png", ".tif", ".tiff"}


class OutputError(Exception):
    """Invalid output configuration (exit code 1)."""


def validate_output(path: Path, predicted_width: int, height: int, scale: float, max_width: int) -> None:
    """Fail early, before pass 2 spends minutes compositing (spec FR-5)."""
    ext = path.suffix.lower()
    if ext not in _SUPPORTED_EXTS:
        raise OutputError(f"unsupported output format '{ext}' (use .png, .jpg, or .tiff)")
    w = int(predicted_width * scale)
    h = int(height * scale)
    if w > max_width:
        raise OutputError(
            f"predicted output width {w}px exceeds --max-width {max_width}px; "
            "use --scale to downsize or raise --max-width"
        )
    if ext in _JPEG_EXTS and (w > JPEG_MAX_DIM or h > JPEG_MAX_DIM):
        raise OutputError(
            f"predicted output {w}x{h}px exceeds the JPEG limit of {JPEG_MAX_DIM}px; "
            "use PNG output or --scale"
        )


def write_image(path: Path, mosaic: np.ndarray, scale: float = 1.0, jpeg_quality: int = 95) -> None:
    if mosaic.size == 0:
        raise OutputError("empty mosaic — nothing to write")
    if scale != 1.0:
        mosaic = cv2.resize(
            mosaic, None, fx=scale, fy=scale,
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4,
        )
    params: list[int] = []
    if path.suffix.lower() in _JPEG_EXTS:
        params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), mosaic, params):
        raise OutputError(f"failed to write image: {path}")
    log.info("wrote %s (%dx%d)", path, mosaic.shape[1], mosaic.shape[0])
