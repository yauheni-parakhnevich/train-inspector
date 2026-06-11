"""Segment detection unit tests: hysteresis (B3), longest-wins, reversal (M8)."""

from train_inspector.motion import MotionSample
from train_inspector.segment import detect_segments, pick_segment

FPS = 30.0
DT = 1000.0 / FPS


def _series(speeds_px_frame):
    """Build smoothed samples from a px/frame speed profile."""
    samples = [MotionSample(t_ms=0, dt_ms=0, dx=0, dy=0, confidence=1.0)]
    t = 0.0
    for v in speeds_px_frame:
        t += DT
        s = MotionSample(t_ms=t, dt_ms=DT, dx=v, dy=0.0, confidence=1.0)
        s.dx_smooth = v
        samples.append(s)
    return samples


def test_basic_segment_with_hysteresis():
    profile = [0.0] * 30 + [4.0] * 90 + [0.0] * 30
    segs = detect_segments(_series(profile), min_speed_px_frame=1.0, nominal_fps=FPS)
    assert len(segs) == 1
    seg = segs[0]
    assert seg.direction == 1
    # boundaries within hysteresis slack (N = 0.25 s = ~8 frames)
    assert abs(seg.start - 31) <= 9
    assert abs(seg.end - 121) <= 9


def test_dip_above_half_threshold_does_not_split():
    """Review B3: braking train dipping to 0.6 px/frame (> min_speed/2) stays
    one segment; in-segment frames all feed the compositor."""
    profile = [0.0] * 20 + [3.0] * 40 + [0.6] * 30 + [3.0] * 40 + [0.0] * 20
    segs = detect_segments(_series(profile), min_speed_px_frame=1.0, nominal_fps=FPS)
    assert len(segs) == 1


def test_long_stop_splits_and_longest_wins():
    profile = [0.0] * 20 + [3.0] * 30 + [0.0] * 60 + [3.0] * 90 + [0.0] * 20
    samples = _series(profile)
    segs = detect_segments(samples, min_speed_px_frame=1.0, nominal_fps=FPS)
    assert len(segs) == 2
    best = pick_segment(segs, samples)
    assert best.duration_ms(samples) > 2500  # the 90-frame one


def test_direction_reversal_is_boundary():
    profile = [0.0] * 20 + [3.0] * 60 + [-3.0] * 60 + [0.0] * 20
    segs = detect_segments(_series(profile), min_speed_px_frame=1.0, nominal_fps=FPS)
    assert len(segs) == 2
    assert segs[0].direction == 1
    assert segs[1].direction == -1


def test_no_motion_no_segments():
    segs = detect_segments(_series([0.2] * 100), min_speed_px_frame=1.0, nominal_fps=FPS)
    assert segs == []


def test_rtl_direction():
    profile = [0.0] * 20 + [-4.0] * 90 + [0.0] * 20
    segs = detect_segments(_series(profile), min_speed_px_frame=1.0, nominal_fps=FPS)
    assert len(segs) == 1 and segs[0].direction == -1
