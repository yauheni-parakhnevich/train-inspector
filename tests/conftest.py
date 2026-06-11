"""Synthetic fixtures per spec NFR-5: generated in-repo, lossless (FFV1),
correctness compared against the analytically known source texture."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np
import pytest

FRAME_W, FRAME_H = 640, 360
TRAIN_H = 160
TRAIN_Y = 100  # top row of the train band in every frame

rng_seed = 12345


def make_texture(length: int, height: int = TRAIN_H, seed: int = rng_seed) -> np.ndarray:
    """Random 16x16 colored blocks + mild noise + light blur.

    Band-limited on purpose: this models a real train surface (structure at a
    coarse scale, not per-pixel white noise), so it survives the sub-pixel
    strip resampling and lets SSIM measure geometric fidelity rather than
    high-frequency resampling loss. Block edges still give the motion estimator
    plenty of trackable corners.
    """
    rng = np.random.default_rng(seed)
    bw = bh = 16
    blocks = rng.integers(40, 255, size=(height // bh + 1, length // bw + 1, 3), dtype=np.uint8)
    tex = np.kron(blocks, np.ones((bh, bw, 1), dtype=np.uint8))[:height, :length]
    tex = cv2.add(tex, rng.integers(0, 8, size=tex.shape, dtype=np.uint8))
    return cv2.GaussianBlur(tex, (0, 0), 1.0)


def make_background(textured: bool = False, seed: int = 999) -> np.ndarray:
    if not textured:
        return np.full((FRAME_H, FRAME_W, 3), 96, dtype=np.uint8)
    rng = np.random.default_rng(seed)
    blocks = rng.integers(20, 90, size=(FRAME_H // 16 + 1, FRAME_W // 16 + 1, 3), dtype=np.uint8)
    return np.kron(blocks, np.ones((16, 16, 1), dtype=np.uint8))[:FRAME_H, :FRAME_W]


def render_frames(
    texture: np.ndarray,
    positions: list[float],
    background: np.ndarray | None = None,
    jitter_y: Callable[[int], float] | None = None,
) -> Iterator[np.ndarray]:
    """Yield frames with the train's LEFT edge at positions[i] (px, may be
    negative/over-width). Sub-pixel positions rendered via warpAffine so the
    fixture itself is sub-pixel correct."""
    bg = background if background is not None else make_background()
    th, tl = texture.shape[:2]
    for i, x in enumerate(positions):
        frame = bg.copy()
        jy = jitter_y(i) if jitter_y else 0.0
        # paste texture at continuous x offset (and optional vertical jitter)
        m = np.array([[1.0, 0.0, x], [0.0, 1.0, TRAIN_Y + jy]], dtype=np.float64)
        canvas = cv2.warpAffine(
            texture, m, (FRAME_W, FRAME_H),
            flags=cv2.INTER_LANCZOS4, borderValue=(0, 0, 0),
        )
        mask = cv2.warpAffine(
            np.full((th, tl), 255, dtype=np.uint8), m, (FRAME_W, FRAME_H),
            flags=cv2.INTER_LINEAR, borderValue=0,
        )
        sel = mask > 128
        frame[sel] = canvas[sel]
        yield frame


def write_video(path: Path, frames: Iterator[np.ndarray], fps: float = 30.0) -> Path:
    """FFV1 (lossless) in .avi; assert the build supports it rather than fall
    back to a lossy codec silently."""
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"FFV1"), fps, (FRAME_W, FRAME_H))
    if not wr.isOpened():
        pytest.skip("OpenCV build lacks FFV1 encoder")
    n = 0
    for f in frames:
        wr.write(f)
        n += 1
    wr.release()
    assert n > 0
    return path


def constant_speed_positions(speed: float, length: int, lead_in: int = 10) -> list[float]:
    """Left edge from fully right-of-frame to fully left-of-frame (LTR travel
    is positions increasing — train moves +x entering from the left edge)."""
    start = -length - lead_in * speed
    end = FRAME_W + lead_in * speed
    n = int((end - start) / speed) + 1
    return [start + i * speed for i in range(n)]


@pytest.fixture(scope="session")
def texture_1200() -> np.ndarray:
    return make_texture(1200)


@pytest.fixture(scope="session")
def fixtures_dir(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("fixtures")


def flicker_frames(
    texture: np.ndarray,
    positions: list[float],
    amplitude: float = 0.18,
) -> Iterator[np.ndarray]:
    """Constant-speed train whose per-frame brightness alternates +/-amplitude.

    Models real exposure/auto-gain variation between frames. The wide-strip path
    stamps each frame's brightness into a full strip -> a hard brightness step at
    every boundary (visible vertical lines). The flow cross-dissolve blends
    adjacent frames at each boundary -> the step is smoothed. This is the
    deterministic regression fixture for the seam fix."""
    base = list(render_frames(texture, positions))
    for i, frame in enumerate(base):
        gain = 1.0 + (amplitude if i % 2 else -amplitude)
        yield np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)
