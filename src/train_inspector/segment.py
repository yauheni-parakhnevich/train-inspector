"""Train-presence segmentation per spec FR-3.

Hysteresis on smoothed velocity; --min-speed used ONLY here (review B3).
Longest qualifying segment wins; direction reversal is a boundary (review M8).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

from .motion import MotionSample

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Segment:
    start: int  # sample index, inclusive
    end: int  # sample index, exclusive
    direction: int  # +1 = left-to-right (train moves +x), -1 = right-to-left

    def duration_ms(self, samples: list[MotionSample]) -> float:
        return samples[self.end - 1].t_ms - samples[self.start].t_ms

    def predicted_width(self, samples: list[MotionSample]) -> int:
        return int(sum(abs(s.dx_smooth) for s in samples[self.start : self.end]))


def detect_segments(
    samples: list[MotionSample],
    min_speed_px_frame: float,
    nominal_fps: float,
) -> list[Segment]:
    """Hysteresis segmentation: enter at |v| > min_speed for N frames, exit at
    |v| < min_speed/2 for N frames; N = 0.25 s worth. Velocity compared in
    px/frame at nominal fps (spec FR-3)."""
    if len(samples) < 2:
        return []

    dt_s = np.array([max(s.dt_ms, 1e-3) / 1000.0 for s in samples])
    v_pxf = np.array([s.dx_smooth for s in samples]) / dt_s / nominal_fps  # px/frame
    n_hys = max(2, int(round(0.25 * nominal_fps)))
    hi, lo = min_speed_px_frame, min_speed_px_frame / 2.0

    segments: list[Segment] = []
    in_seg = False
    seg_start = 0
    seg_dir = 0
    run = 0

    for i in range(len(samples)):
        v = v_pxf[i]
        if not in_seg:
            if abs(v) > hi:
                run += 1
                if run >= n_hys:
                    in_seg = True
                    seg_start = i - run + 1
                    seg_dir = int(np.sign(v))
                    run = 0
            else:
                run = 0
        else:
            reversed_dir = abs(v) > hi and int(np.sign(v)) != seg_dir
            if abs(v) < lo or reversed_dir:
                run += 1
                if reversed_dir or run >= n_hys:
                    end = i - run + 1
                    if end > seg_start:
                        segments.append(Segment(seg_start, end, seg_dir))
                    in_seg = False
                    run = 0
                    if reversed_dir:
                        # Direction reversal is an immediate boundary; the new
                        # direction may start its own segment from here.
                        run = 1 if abs(v) > hi else 0
            else:
                run = 0

    if in_seg:
        segments.append(Segment(seg_start, len(samples), seg_dir))

    for s in segments:
        log.info(
            "segment: samples [%d, %d) dir=%s duration=%.1fs predicted width=%dpx",
            s.start, s.end, "ltr" if s.direction > 0 else "rtl",
            s.duration_ms(samples) / 1000.0, s.predicted_width(samples),
        )
    return segments


def pick_segment(segments: list[Segment], samples: list[MotionSample]) -> Segment | None:
    """Longest qualifying segment by duration (spec FR-3)."""
    if not segments:
        return None
    return max(segments, key=lambda s: s.duration_ms(samples))
