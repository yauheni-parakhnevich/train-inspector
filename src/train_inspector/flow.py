"""Flow-based sub-time supersampling at the slit (design doc
2026-06-11-slit-flow-supersampling-design.md).

Replaces the single-frame wide strip with a per-row, motion-compensated
cross-dissolve between the two frames bounding each interval, so consecutive
strips meet at a shared frame (no inter-frame jump) and each row tiles at its
own true displacement (perspective seams removed). Localized to a band around
the slit; the caller falls back to its wide-strip path when flow is unreliable.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)

BAND_HALF_W = 96  # base half-width; widened below to always cover the motion
DIS_PRESET = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
RELIABLE_ABS = 2.0   # px; |median Fx - dx_seed| tolerance floor
RELIABLE_FRAC = 0.25  # or this fraction of |dx_seed|


@dataclass
class FlowBand:
    """Per-row horizontal flow sampled at the slit, plus the refined scalar dx."""

    fx_col: np.ndarray  # shape (H,), per-row horizontal displacement at the slit (px)
    dx_refined: float   # median(fx_col), signed - used for the carry/width accounting


def _smooth1d(v: np.ndarray, k: int) -> np.ndarray:
    """Edge-padded moving average. Suppresses per-row flow noise while keeping
    the low-frequency perspective gradient (which we WANT to preserve)."""
    k = max(1, k | 1)
    if k == 1 or len(v) < 3:
        return v.astype(np.float64)
    pad = k // 2
    vp = np.pad(v.astype(np.float64), pad, mode="edge")
    return np.convolve(vp, np.ones(k) / k, mode="valid")


class BandInterpolator:
    """Holds the DIS optical-flow instance and computes per-frame FlowBands."""

    def __init__(self) -> None:
        self._dis = cv2.DISOpticalFlow_create(DIS_PRESET)

    def _band_x(self, slit_x: float, dx_seed: float, width: int) -> tuple[int, int]:
        half = max(BAND_HALF_W, int(math.ceil(abs(dx_seed))) + 16)
        x0 = max(0, int(slit_x) - half)
        x1 = min(width, int(slit_x) + half)
        return x0, x1

    def analyze(
        self, frame_k: np.ndarray, frame_k1: np.ndarray, slit_x: float, dx_seed: float
    ) -> FlowBand | None:
        """Dense flow k->k+1 in a band around the slit. Returns a FlowBand, or
        None when the flow is untrustworthy (sub-pixel motion, or median flow
        disagrees with the pass-1 seed - e.g. a uniform car side). On None the
        caller uses the wide strip, so hard regions match today's quality."""
        if abs(dx_seed) < 1.0:
            return None
        x0, x1 = self._band_x(slit_x, dx_seed, frame_k.shape[1])
        if x0 >= x1:
            return None
        gk = cv2.cvtColor(np.ascontiguousarray(frame_k[:, x0:x1]), cv2.COLOR_BGR2GRAY)
        gk1 = cv2.cvtColor(np.ascontiguousarray(frame_k1[:, x0:x1]), cv2.COLOR_BGR2GRAY)
        flow = self._dis.calc(gk, gk1, None)  # (H, x1-x0, 2)
        col = min(max(int(round(slit_x)) - x0, 0), flow.shape[1] - 1)
        fx_col = _smooth1d(flow[:, col, 0], max(3, frame_k.shape[0] // 20))
        dx_refined = float(np.median(fx_col))
        tol = max(RELIABLE_ABS, RELIABLE_FRAC * abs(dx_seed))
        if abs(dx_refined - dx_seed) > tol:
            log.debug("flow rejected near slit: median Fx %.2f vs seed %.2f (tol %.2f)",
                      dx_refined, dx_seed, tol)
            return None
        return FlowBand(fx_col=fx_col, dx_refined=dx_refined)

    def synthesize(
        self,
        fb: FlowBand,
        frame_k: np.ndarray,
        frame_k1: np.ndarray,
        slit_x: float,
        carry_in: float,
        w: int,
        dy_cum: float,
        direction: int,
        interp_flag: int,
    ) -> np.ndarray:
        """Build the w-column strip as a per-row motion-compensated cross-dissolve.

        S_k  : frame_k  sampled at x = slit - carry_in + c           (the wide strip)
        S_k1 : frame_k1 sampled at x = slit - carry_in + c + fx(y)   (same surface,
               motion-compensated per row -> fixes perspective)
        out[:, c] = (1 - t_c)*S_k[:, c] + t_c*S_k1[:, c]

        t ramps so S_k1's edge abuts the next strip in assembly order (design
        doc): RTL t=(c+0.5)/w, LTR t=1-(c+0.5)/w. Vertical jitter dy_cum is the
        same global translation as the wide path (single resample)."""
        h = frame_k.shape[0]
        cols = np.arange(w, dtype=np.float32)
        xs = (slit_x - carry_in) + cols                       # (w,)
        ys = (np.arange(h, dtype=np.float32) + dy_cum).reshape(h, 1)
        map_y = np.tile(ys, (1, w)).astype(np.float32)        # (h, w)

        map_x_k = np.tile(xs, (h, 1)).astype(np.float32)      # (h, w)
        fx = fb.fx_col.astype(np.float32).reshape(h, 1)
        map_x_k1 = (xs.reshape(1, w) + fx).astype(np.float32)  # (h, w)

        s_k = cv2.remap(frame_k, map_x_k, map_y, interp_flag, borderMode=cv2.BORDER_REPLICATE)
        s_k1 = cv2.remap(frame_k1, map_x_k1, map_y, interp_flag, borderMode=cv2.BORDER_REPLICATE)

        t = (cols + 0.5) / w
        if direction > 0:  # LTR: later-frame edge on the left
            t = 1.0 - t
        t = t.reshape(1, w, 1).astype(np.float32)
        out = (1.0 - t) * s_k.astype(np.float32) + t * s_k1.astype(np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)
