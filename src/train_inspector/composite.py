"""Strip extraction and mosaic assembly per spec §10.3 (carry accumulator,
normative) and §10.4 (strip geometry, normative). Vertical jitter alignment
per FR-4 (review M5) applied in the same warp (single interpolation).
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .flow import BandInterpolator

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
        interp: "BandInterpolator | None" = None,
    ) -> int:
        """Feed one in-segment frame pair. Returns strip width taken (may be 0).

        §10.3 accounting is UNCHANGED and always uses the pass-1 dx_smooth:
        total = |dx_smooth| + carry_in; w = floor(total); carry_out = total - w.
        Flow never feeds the width — it is used only to pick the path and to
        motion-compensate the per-row sampling inside synthesize (seam removal).
        When `interp` and `frame_next` are given and the
        flow is reliable, the w columns are a per-row motion-compensated
        cross-dissolve between `frame` (k) and `frame_next` (k+1), removing the
        inter-frame seam (see flow.py). Otherwise the original single-frame wide
        strip is taken: source region [x_slit - carry_in, x_slit - carry_in + w)
        of `frame`, for BOTH directions; direction is handled purely by assembly
        order in mosaic(). dst(x, y) = src(x + a, y + dy_cum)."""
        carry_in = self.carry
        total = abs(dx_smooth) + carry_in
        w = math.floor(total)
        self.carry = total - w
        if w <= 0:  # sub-pixel step: carry already advanced, no strip this frame
            return 0

        flow_band = None
        if interp is not None and frame_next is not None:
            flow_band = interp.analyze(frame, frame_next, self.slit_x, dx_smooth)

        if flow_band is not None:
            strip = interp.synthesize(
                flow_band, frame, frame_next, self.slit_x, carry_in, w,
                dy_cum, self.direction, self.interp,
            )
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

    def mosaic(self) -> np.ndarray:
        if not self._strips:
            return np.zeros((self.height, 0, 3), dtype=np.uint8)
        ordered = self._strips[::-1] if self.direction > 0 else self._strips
        return np.concatenate(ordered, axis=1)
