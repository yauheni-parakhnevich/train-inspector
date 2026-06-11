"""Strip extraction and mosaic assembly per spec §10.3 (carry accumulator,
normative) and §10.4 (strip geometry, normative). Vertical jitter alignment
per FR-4 (review M5) applied in the same warp (single interpolation).

By default each strip is a motion-compensated temporal CROSS-DISSOLVE between
the two frames bounding its interval, aligned by the pass-1 displacement: this
makes consecutive strips meet at a shared frame so the per-frame boundary
(exposure / motion blur / sub-pixel) is smoothed away instead of showing as a
vertical seam. --fast keeps the original single-frame wide strip.
"""

from __future__ import annotations

import logging
import math

import cv2
import numpy as np

log = logging.getLogger(__name__)


class Compositor:
    """Accumulates strips in time order; final assembly is strip-granular so
    content orientation within each strip is preserved (spec §10.4).

    Time order produces front→tail content. For LTR travel (train moves +x)
    the train's front is its right end, so strips are concatenated in reverse
    time order; for RTL, in time order. Either way the panorama reads
    left-to-right with the train's front on the side it entered from.
    """

    def __init__(self, height: int, direction: int, slit_x: float, fast: bool = False):
        if direction not in (-1, 1):
            raise ValueError("direction must be +1 (ltr) or -1 (rtl)")
        self.height = height
        self.direction = direction
        self.slit_x = float(slit_x)
        self.interp = cv2.INTER_LINEAR if fast else cv2.INTER_LANCZOS4
        self.blend = not fast  # cross-dissolve unless --fast
        self.carry = 0.0
        self._strips: list[np.ndarray] = []
        self._cols = 0

    @property
    def width(self) -> int:
        return self._cols

    def add(
        self,
        frame: np.ndarray,
        dx_smooth: float,
        dy_cum: float = 0.0,
        frame_next: np.ndarray | None = None,
    ) -> int:
        """Feed one in-segment frame (with its successor for blending). Returns
        strip width taken (may be 0).

        §10.3 accounting (always the pass-1 dx_smooth): total = |dx_smooth| +
        carry_in; w = floor(total); carry_out = total - w.

        When blending is on (default) and `frame_next` is given, the w columns
        are a motion-compensated cross-dissolve between `frame` (k) and
        `frame_next` (k+1) aligned by dx_smooth — removes the inter-frame seam.
        Otherwise (or under --fast) the original single-frame wide strip is
        taken: source region [x_slit - carry_in, x_slit - carry_in + w) of
        `frame`, for BOTH directions; direction is handled purely by assembly
        order in mosaic(). dst(x, y) = src(x + a, y + dy_cum)."""
        carry_in = self.carry
        total = abs(dx_smooth) + carry_in
        w = math.floor(total)
        self.carry = total - w
        if w <= 0:  # sub-pixel step: carry already advanced, no strip this frame
            return 0

        if self.blend and frame_next is not None:
            strip = self._crossfade(frame, frame_next, carry_in, w, dx_smooth, dy_cum)
        else:
            a = self.slit_x - carry_in
            m = np.array([[1.0, 0.0, a], [0.0, 1.0, dy_cum]], dtype=np.float64)
            strip = cv2.warpAffine(
                frame, m, (w, self.height),
                flags=self.interp | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
        self._strips.append(strip)
        self._cols += w
        return w

    def _crossfade(
        self,
        frame_k: np.ndarray,
        frame_k1: np.ndarray,
        carry_in: float,
        w: int,
        dx_shift: float,
        dy_cum: float,
    ) -> np.ndarray:
        """w-column motion-compensated cross-dissolve.

        S_k  : frame_k  at x = slit - carry_in + c             (the wide strip)
        S_k1 : frame_k1 at x = slit - carry_in + c + dx_shift  (the SAME train
               surface one frame later, since content moved dx_shift)
        out[:, c] = (1 - t_c)*S_k[:, c] + t_c*S_k1[:, c]

        t ramps so S_k1's edge abuts the next strip in assembly order: RTL
        t=(c+0.5)/w, LTR t=1-(c+0.5)/w. Consecutive strips then meet at a shared
        frame and the per-frame boundary is smoothed, not a seam. dx_shift is the
        pass-1 displacement, uniform across rows — dense optical flow was tried
        for per-row perspective correction and is unreliable at the tens-of-px/
        frame real footage exhibits (it injects noise instead)."""
        h = self.height
        cols = np.arange(w, dtype=np.float32)
        xs = (self.slit_x - carry_in) + cols
        ys = (np.arange(h, dtype=np.float32) + dy_cum).reshape(h, 1)
        map_y = np.tile(ys, (1, w))
        map_x_k = np.tile(xs, (h, 1))
        map_x_k1 = np.tile(xs + dx_shift, (h, 1)).astype(np.float32)
        s_k = cv2.remap(frame_k, map_x_k, map_y, self.interp, borderMode=cv2.BORDER_REPLICATE)
        s_k1 = cv2.remap(frame_k1, map_x_k1, map_y, self.interp, borderMode=cv2.BORDER_REPLICATE)
        t = (cols + 0.5) / w
        if self.direction > 0:  # LTR: later-frame edge on the left
            t = 1.0 - t
        t = t.reshape(1, w, 1).astype(np.float32)
        out = (1.0 - t) * s_k.astype(np.float32) + t * s_k1.astype(np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)

    def mosaic(self) -> np.ndarray:
        if not self._strips:
            return np.zeros((self.height, 0, 3), dtype=np.uint8)
        ordered = self._strips[::-1] if self.direction > 0 else self._strips
        return np.concatenate(ordered, axis=1)
