"""Frame decoding: lazy (timestamp_ms, BGR frame) iteration per spec FR-1.

All downstream logic is timestamp-based (VFR-safe). Frame-index seeking is
never used for alignment (spec §8 / review B2).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Transfer characteristics we cannot detect via OpenCV; we warn on 10-bit-ish
# sources by checking the backend's reported pixel format is unavailable, so
# the HDR warning is emitted from CLI based on codec name heuristics instead.


class InputError(Exception):
    """Invalid or unreadable input (exit code 1)."""


@dataclass(frozen=True)
class Roi:
    x: int
    y: int
    w: int
    h: int

    def clamp(self, frame_w: int, frame_h: int) -> "Roi":
        x = max(0, min(self.x, frame_w - 1))
        y = max(0, min(self.y, frame_h - 1))
        w = max(1, min(self.w, frame_w - x))
        h = max(1, min(self.h, frame_h - y))
        return Roi(x, y, w, h)

    def crop(self, frame: np.ndarray) -> np.ndarray:
        return frame[self.y : self.y + self.h, self.x : self.x + self.w]


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    nominal_fps: float
    n_frames_estimate: int  # estimate only; wrong for VFR — never used for logic


def open_capture(path: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise InputError(f"cannot open video: {path}")
    # Honor container rotation metadata (spec FR-1 / review M9).
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 1)
    return cap


def probe(path: str) -> VideoInfo:
    cap = open_capture(path)
    try:
        ok, frame = cap.read()
        if not ok or frame is None:
            raise InputError(f"video has no decodable frames: {path}")
        h, w = frame.shape[:2]
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if not (1.0 <= fps <= 1000.0):
            fps = 30.0  # CAP_PROP_FPS is an estimate; fall back to a sane nominal
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        return VideoInfo(width=w, height=h, nominal_fps=fps, n_frames_estimate=n)
    finally:
        cap.release()


def _normalize(frame: np.ndarray) -> np.ndarray:
    """Normalize any decoded frame to 8-bit 3-channel BGR (spec FR-1)."""
    if frame.dtype == np.uint16:
        frame = (frame >> 8).astype(np.uint8)
    elif frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.ndim == 2:
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    return frame


def iter_frames(
    path: str,
    start_ms: float | None = None,
    end_ms: float | None = None,
    roi: Roi | None = None,
    seek_hint_ms: float | None = None,
) -> Iterator[tuple[float, np.ndarray]]:
    """Yield (timestamp_ms, BGR frame), trimmed to [start_ms, end_ms], ROI-cropped.

    seek_hint_ms: coarse seek target strictly for skipping a long prefix; we
    seek there, verify we landed *before* the region of interest, and reopen
    from t=0 if the keyframe-based seek overshot (spec §8).
    """
    cap = open_capture(path)
    try:
        if seek_hint_ms is not None and seek_hint_ms > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, seek_hint_ms)
            ok, frame = cap.read()
            t = cap.get(cv2.CAP_PROP_POS_MSEC)
            target = start_ms if start_ms is not None else seek_hint_ms
            if not ok or t > target:
                log.debug("coarse seek overshot (landed %.0f ms); reopening from 0", t)
                cap.release()
                cap = open_capture(path)
            else:
                t = cap.get(cv2.CAP_PROP_POS_MSEC)
                frame = _normalize(frame)
                if (start_ms is None or t >= start_ms) and (end_ms is None or t <= end_ms):
                    yield t, roi.crop(frame) if roi else frame

        prev_t = -1.0
        synthetic_dt = None
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                return
            t = cap.get(cv2.CAP_PROP_POS_MSEC)
            if t <= 0 and prev_t >= 0:
                # Some backends return 0 for POS_MSEC on certain containers;
                # synthesize monotonic timestamps from nominal fps.
                if synthetic_dt is None:
                    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    synthetic_dt = 1000.0 / (fps if 1.0 <= fps <= 1000.0 else 30.0)
                t = prev_t + synthetic_dt
            if t <= prev_t:
                t = prev_t + 1e-3  # enforce strict monotonicity
            prev_t = t
            if start_ms is not None and t < start_ms:
                continue
            if end_ms is not None and t > end_ms:
                return
            frame = _normalize(frame)
            yield t, roi.crop(frame) if roi else frame
    finally:
        cap.release()
