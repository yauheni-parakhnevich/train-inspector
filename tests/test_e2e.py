"""End-to-end acceptance tests per spec §11, run against FFV1 fixtures and
compared to the analytically known source texture (no stored goldens)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from train_inspector import decode, pipeline
from train_inspector.encode import OutputError

from conftest import (
    FRAME_H, FRAME_W, TRAIN_H, TRAIN_Y,
    constant_speed_positions, flicker_frames, make_background, make_texture, render_frames, write_video,
)
from helpers import align_and_ssim, extract_train_region, horizontal_seam_energy

TEX_LEN = 1200
# Integer speed renders the fixture train at integer positions → no sub-pixel
# interpolation anywhere, so the slit-scan reconstruction is pixel-exact and
# the strict SSIM ≥ 0.98 criterion is meaningful (§11.1/§11.2).
SPEED = 4
# A non-integer speed exercises the sub-pixel carry accumulator end-to-end; the
# fixture's own sub-pixel rendering plus strip resampling softens the result
# (documented resampling behavior, §10.5/NFR-2), so width is held to ±1% but
# SSIM is checked against a relaxed bound (§11 carry regression).
SPEED_FRAC = 4.37
FRAC_SSIM = 0.95


def _run(video: Path, out: Path, **kw) -> pipeline.Result:
    opts = pipeline.Options(input=video, output=out, **kw)
    return pipeline.run(opts)


@pytest.fixture(scope="module")
def texture() -> np.ndarray:
    return make_texture(TEX_LEN)


@pytest.fixture(scope="module")
def video_constant_ltr(texture, tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("e2e")
    positions = constant_speed_positions(SPEED, TEX_LEN)
    return write_video(d / "constant_ltr.avi", render_frames(texture, positions))


def test_constant_speed_ltr(video_constant_ltr, texture, tmp_path):
    """§11.1: width ±1%, SSIM ≥ 0.98 vs analytic texture."""
    out = tmp_path / "pano.png"
    result = _run(video_constant_ltr, out, fast=True)  # strict SSIM >= 0.98 = pure-geometry path
    pano = cv2.imread(str(out))
    assert pano is not None

    region = extract_train_region(pano, TRAIN_Y, TRAIN_H)
    assert abs(region.shape[1] - TEX_LEN) <= TEX_LEN * 0.01
    assert align_and_ssim(region, texture) >= 0.98
    assert result.direction == "ltr"


def test_constant_speed_rtl(texture, tmp_path):
    """§11.2: reversed travel produces the same left-to-right panorama."""
    positions = constant_speed_positions(SPEED, TEX_LEN)[::-1]
    video = write_video(tmp_path / "rtl.avi", render_frames(texture, positions))
    out = tmp_path / "pano.png"
    result = _run(video, out, fast=True)  # strict SSIM >= 0.98 = pure-geometry path
    assert result.direction == "rtl"

    pano = cv2.imread(str(out))
    region = extract_train_region(pano, TRAIN_Y, TRAIN_H)
    assert abs(region.shape[1] - TEX_LEN) <= TEX_LEN * 0.01
    assert align_and_ssim(region, texture) >= 0.98


def test_subpixel_speed_carry_no_drift(texture, tmp_path):
    """Carry accumulator end-to-end: non-integer speed must hold width to ±1%
    (no cumulative drift over the whole pass) despite sub-pixel resampling."""
    positions = constant_speed_positions(SPEED_FRAC, TEX_LEN)
    video = write_video(tmp_path / "frac.avi", render_frames(texture, positions))
    out = tmp_path / "pano.png"
    _run(video, out)
    pano = cv2.imread(str(out))
    region = extract_train_region(pano, TRAIN_Y, TRAIN_H)
    assert abs(region.shape[1] - TEX_LEN) <= TEX_LEN * 0.01
    assert align_and_ssim(region, texture) >= FRAC_SSIM


def test_acceleration_no_seams(texture, tmp_path):
    """§11.3: 1×→2× linear acceleration, continuity vs analytic texture.

    SSIM threshold relaxed vs constant speed: smoothing lag under constant
    acceleration causes sub-pixel local stretch (documented, §10.5), which
    SSIM penalizes but which is not a seam."""
    positions = []
    x = -TEX_LEN - 40.0
    v = 3.0
    while x < FRAME_W + 40.0:
        positions.append(x)
        x += v
        v = min(6.0, v + 0.01)
    video = write_video(tmp_path / "accel.avi", render_frames(texture, positions))
    out = tmp_path / "pano.png"
    _run(video, out)
    pano = cv2.imread(str(out))
    region = extract_train_region(pano, TRAIN_Y, TRAIN_H)
    assert abs(region.shape[1] - TEX_LEN) <= TEX_LEN * 0.02
    assert align_and_ssim(region, texture) >= 0.90


def test_slow_crawl_inside_segment_keeps_geometry(texture, tmp_path):
    """§11.4 / review B3: mid-pass dip to 0.6 px/frame loses no geometry."""
    positions = []
    x = -TEX_LEN - 40.0
    n_total = 0
    while x < FRAME_W + 40.0:
        positions.append(x)
        third = (TEX_LEN + FRAME_W + 80) / 3
        in_middle = third < (x - (-TEX_LEN - 40.0)) < 2 * third
        x += 0.6 if in_middle else 4.0
        n_total += 1
        if n_total > 4000:
            pytest.fail("fixture generation runaway")
    video = write_video(tmp_path / "crawl.avi", render_frames(texture, positions))
    out = tmp_path / "pano.png"
    _run(video, out)
    pano = cv2.imread(str(out))
    region = extract_train_region(pano, TRAIN_Y, TRAIN_H)
    assert abs(region.shape[1] - TEX_LEN) <= TEX_LEN * 0.015


def test_entry_exit_with_textured_background(texture, tmp_path):
    """§11.5 / review B1: textured static background majority must not zero
    the estimate while the train enters/leaves — nose and tail intact."""
    positions = constant_speed_positions(SPEED, TEX_LEN)
    bg = make_background(textured=True)
    video = write_video(tmp_path / "entry.avi", render_frames(texture, positions, bg))
    out = tmp_path / "pano.png"
    _run(video, out)
    pano = cv2.imread(str(out))
    # textured bg breaks flat-bg column detection; align full band instead
    band = pano[TRAIN_Y : TRAIN_Y + TRAIN_H]
    res = cv2.matchTemplate(band, texture, cv2.TM_CCOEFF_NORMED)
    _, score, _, _ = cv2.minMaxLoc(res)
    assert band.shape[1] >= TEX_LEN  # nothing clipped
    assert score >= 0.9  # full texture, including ends, present once


def test_static_video_exit_code_2(tmp_path):
    bg = make_background()
    video = write_video(tmp_path / "static.avi", iter([bg.copy() for _ in range(60)]))
    with pytest.raises(pipeline.NoMotionError):
        _run(video, tmp_path / "pano.png")


def test_missing_file_exit_code_1(tmp_path):
    with pytest.raises(decode.InputError):
        _run(tmp_path / "nope.mp4", tmp_path / "pano.png")


def test_max_width_guard_fails_before_compositing(video_constant_ltr, tmp_path):
    with pytest.raises(OutputError, match="max-width"):
        _run(video_constant_ltr, tmp_path / "pano.png", max_width=100)


def test_cli_end_to_end(video_constant_ltr, tmp_path):
    """Console entry point: real process, exit code 0, file produced."""
    out = tmp_path / "cli_pano.png"
    proc = subprocess.run(
        [sys.executable, "-m", "train_inspector.cli", str(video_constant_ltr),
         "-o", str(out), "--quiet"],
        capture_output=True, text=True, timeout=300,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()


def test_cli_exit_2_no_motion(tmp_path):
    bg = make_background()
    video = write_video(tmp_path / "static.avi", iter([bg.copy() for _ in range(60)]))
    proc = subprocess.run(
        [sys.executable, "-m", "train_inspector.cli", str(video), "--quiet"],
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 2
    assert "no train motion" in proc.stderr


BLEND_SSIM = 0.97  # default cross-dissolve path: blend softens slightly vs --fast's 0.98


def test_constant_speed_blend_path_no_regression(video_constant_ltr, texture, tmp_path):
    """Default (cross-dissolve) path on a rigid fixture: width +/-1%, SSIM >= 0.97. Blend must
    keep geometry while removing seams."""
    out = tmp_path / "pano.png"
    _run(video_constant_ltr, out)  # default: blend ON
    pano = cv2.imread(str(out))
    region = extract_train_region(pano, TRAIN_Y, TRAIN_H)
    assert abs(region.shape[1] - TEX_LEN) <= TEX_LEN * 0.01
    assert align_and_ssim(region, texture) >= BLEND_SSIM


def test_blend_reduces_seam_energy_vs_fast(texture, tmp_path):
    """Headline regression: on a brightness-flicker fixture the wide-strip
    (--fast) path stamps a hard brightness step at every boundary; the cross-dissolve
    path smooths them, lowering horizontal seam energy."""
    positions = constant_speed_positions(SPEED, TEX_LEN)
    video = write_video(tmp_path / "flicker.avi", flicker_frames(texture, positions))

    out_fast = tmp_path / "fast.png"
    _run(video, out_fast, fast=True)
    out_blend = tmp_path / "blend.png"
    _run(video, out_blend)  # default: blend ON

    reg_fast = extract_train_region(cv2.imread(str(out_fast)), TRAIN_Y, TRAIN_H)
    reg_blend = extract_train_region(cv2.imread(str(out_blend)), TRAIN_Y, TRAIN_H)
    e_fast = horizontal_seam_energy(reg_fast)
    e_blend = horizontal_seam_energy(reg_blend)
    assert e_blend < e_fast * 0.85, f"blend seam energy {e_blend:.2f} not < 0.85x{e_fast:.2f}"


# --- dropped-frame ghost guard (pipeline._capped_dx) -------------------------

def test_capped_dx_clamps_dropped_frame_gap():
    """A dropped-frame gap (dt >> median, dx >> median) is clamped to the cap so
    the wide single-frame strip cannot duplicate surface into a ghost."""
    from train_inspector.pipeline import _capped_dx
    assert _capped_dx(167.0, 152.0, 48.0, 41.0) == pytest.approx(96.0)   # 2.0 * 48
    assert _capped_dx(-167.0, 152.0, 48.0, 41.0) == pytest.approx(-96.0)


def test_capped_dx_leaves_normal_and_fast_frames():
    """Normal frames, and legitimately fast frames at NORMAL dt, are untouched."""
    from train_inspector.pipeline import _capped_dx
    assert _capped_dx(48.0, 41.0, 48.0, 41.0) == 48.0
    assert _capped_dx(150.0, 41.0, 48.0, 41.0) == 150.0   # fast but normal dt -> never capped
    assert _capped_dx(60.0, 152.0, 48.0, 41.0) == 60.0    # big dt but dx within cap
    assert _capped_dx(167.0, 152.0, 0.0, 41.0) == 167.0   # degenerate median -> passthrough
