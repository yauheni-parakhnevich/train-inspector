"""Compositor unit tests: carry accumulator (§10.3) and strip geometry (§10.4)."""

import numpy as np
import pytest
import cv2
from conftest import make_texture
from train_inspector.flow import BandInterpolator

from train_inspector.composite import Compositor


def gradient_frame(w: int = 200, h: int = 20) -> np.ndarray:
    """Frame whose pixel value encodes its column index (mod 256)."""
    col = (np.arange(w) % 256).astype(np.uint8)
    return np.dstack([np.tile(col, (h, 1))] * 3)


def test_carry_accumulator_no_drift():
    comp = Compositor(height=20, direction=1, slit_x=100.0, fast=True)
    frame = gradient_frame()
    dx = 3.37
    n = 500
    total_w = sum(comp.add(frame, dx) for _ in range(n))
    assert abs(total_w - dx * n) < 1.0  # Σw tracks Σ|dx| within 1 px over the run


def test_zero_and_subpixel_dx_contribute_via_carry():
    comp = Compositor(height=20, direction=1, slit_x=100.0, fast=True)
    frame = gradient_frame()
    widths = [comp.add(frame, 0.25) for _ in range(8)]
    assert sum(widths) == 2  # 8 × 0.25 px accumulates into 2 real columns
    assert comp.add(frame, 0.0) == 0  # stopped train: valid, no strip


@pytest.mark.parametrize("direction", [1, -1])
def test_integer_dx_reproduces_exact_columns(direction):
    """Constant integer dx, fast (bilinear) path: strips must reproduce source
    columns exactly — any off-by-one/half-pixel error shows immediately."""
    h, w_frame = 20, 200
    slit = 100.0
    dx = 5
    comp = Compositor(height=h, direction=direction, slit_x=slit, fast=True)
    frame = gradient_frame(w_frame, h)
    n = 6
    for _ in range(n):
        comp.add(frame, dx * direction)
    out = comp.mosaic()
    assert out.shape == (h, dx * n, 3)
    # With carry == 0 (integer dx) the source region is [slit, slit + dx) for
    # BOTH directions (§10.4: per-frame sampling is direction-independent;
    # direction only flips assembly order). Every strip samples the same region
    # of the same static frame, so the mosaic is that region tiled n times.
    expected_cols = np.arange(int(slit), int(slit) + dx) % 256
    tile = np.tile(expected_cols, n)
    np.testing.assert_array_equal(out[0, :, 0], tile)


def test_ltr_strip_order_reversed_rtl_natural():
    """Mosaic assembly order: LTR reverse-time, RTL time order (§10.4)."""
    h = 4
    frame_a = np.full((h, 50, 3), 10, dtype=np.uint8)
    frame_b = np.full((h, 50, 3), 200, dtype=np.uint8)

    ltr = Compositor(height=h, direction=1, slit_x=25.0, fast=True)
    ltr.add(frame_a, 5)
    ltr.add(frame_b, 5)
    out = ltr.mosaic()
    assert out[0, 0, 0] == 200 and out[0, -1, 0] == 10  # later strip leftmost

    rtl = Compositor(height=h, direction=-1, slit_x=25.0, fast=True)
    rtl.add(frame_a, -5)
    rtl.add(frame_b, -5)
    out = rtl.mosaic()
    assert out[0, 0, 0] == 10 and out[0, -1, 0] == 200  # time order


def test_vertical_jitter_shift():
    h = 30
    frame = np.zeros((h, 60, 3), dtype=np.uint8)
    frame[10] = 255  # bright row at y=10
    comp = Compositor(height=h, direction=1, slit_x=30.0, fast=True)
    # dy_cum = +3: background content has drifted 3 px down since segment
    # start, so the bright row now at y=10 was originally at y=7; the warp
    # (dst(y) = src(y + dy_cum)) must restore it there.
    comp.add(frame, 4.0, dy_cum=3.0)
    out = comp.mosaic()
    assert out[7, :, 0].mean() > 200
    assert out[10, :, 0].mean() < 50


class _StubReject:
    """Interp whose flow is always rejected -> must behave like the wide path."""

    def analyze(self, *a, **k):
        return None


def test_add_falls_back_to_wide_strip_when_flow_rejected():
    frame = gradient_frame()
    comp = Compositor(height=20, direction=1, slit_x=100.0)
    w_flow = comp.add(frame, 5.0, frame_next=frame, interp=_StubReject())
    comp2 = Compositor(height=20, direction=1, slit_x=100.0)
    w_plain = comp2.add(frame, 5.0)
    assert w_flow == w_plain == 5
    np.testing.assert_array_equal(comp.mosaic(), comp2.mosaic())


def test_add_flow_path_preserves_carry_width():
    """Real interp on a moving pair: sum of widths still tracks sum of |dx_refined|."""
    tex = make_texture(440, 160, 3)
    place = lambda off: cv2.warpAffine(
        tex, np.array([[1.0, 0, off], [0, 1.0, 0]]), (400, 160), flags=cv2.INTER_LANCZOS4
    )
    a, b = place(0.0), place(6.0)
    comp = Compositor(height=160, direction=1, slit_x=200.0)
    interp = BandInterpolator()
    total = sum(comp.add(a, 6.0, frame_next=b, interp=interp) for _ in range(20))
    assert abs(total - 6 * 20) <= 20  # within ~1 px/frame of sum|dx|
