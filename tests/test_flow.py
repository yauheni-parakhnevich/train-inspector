"""Unit tests for flow-based sub-time supersampling (flow.py)."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from train_inspector.flow import BandInterpolator, FlowBand
from conftest import make_texture

H = 160


def _shifted_pair(dx: float, h: int = H, w: int = 400, seed: int = 7):
    """Two frames: frame_k, and frame_k1 with the SAME texture shifted +dx px."""
    tex = make_texture(w + 40, h, seed)

    def place(off: float) -> np.ndarray:
        m = np.array([[1.0, 0.0, off], [0.0, 1.0, 0.0]], dtype=np.float64)
        return cv2.warpAffine(tex, m, (w, h), flags=cv2.INTER_LANCZOS4)

    return place(0.0), place(dx)


def test_analyze_recovers_dx_and_is_reliable():
    a, b = _shifted_pair(8.0)
    fb = BandInterpolator().analyze(a, b, slit_x=200.0, dx_seed=8.0)
    assert fb is not None
    assert abs(fb.dx_refined - 8.0) < 2.0
    assert fb.fx_col.shape == (H,)


def test_analyze_rejects_when_flow_disagrees_with_seed():
    a, b = _shifted_pair(8.0)  # true motion +8
    fb = BandInterpolator().analyze(a, b, slit_x=200.0, dx_seed=-8.0)  # wrong-sign seed
    assert fb is None


def test_analyze_rejects_subpixel_seed():
    a, b = _shifted_pair(8.0)
    assert BandInterpolator().analyze(a, b, slit_x=200.0, dx_seed=0.4) is None
