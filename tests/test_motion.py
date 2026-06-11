"""Motion estimation unit tests: cluster selection (B1), pairwise accuracy,
VFR-safe smoothing (M4)."""

import numpy as np
import pytest

from train_inspector import motion
from train_inspector.motion import MotionSample, _cluster_1d, _select

from conftest import make_background, make_texture, render_frames


def _samples_from(dxs, dt_ms=33.3, conf=1.0):
    out = [MotionSample(t_ms=0.0, dt_ms=0.0, dx=0.0, dy=0.0, confidence=0.0)]
    t = 0.0
    for dx in dxs:
        t += dt_ms
        out.append(MotionSample(t_ms=t, dt_ms=dt_ms, dx=dx, dy=0.0, confidence=conf))
    return out


class TestClustering:
    def test_minority_moving_cluster_beats_static_majority(self):
        """Review B1: train at 10% of ROI must still win over background."""
        dxs = np.array([0.05, -0.1, 0.0, 0.1, -0.05, 0.02, 0.08, -0.02, 0.0, 0.04,
                        0.01, -0.03, 0.06, 0.0, -0.01, 0.03, 0.02, 0.05, 0.0, -0.04,
                        6.2, 6.3, 6.1, 6.25, 6.35, 6.2, 6.15, 6.3])  # 20 static, 8 train
        dys = np.zeros_like(dxs)
        clusters = _cluster_1d(dxs, dys)
        sel, static = _select(clusters, direction=0)
        assert sel is not None
        assert sel.dx == pytest.approx(6.2, abs=0.2)
        assert static is not None and abs(static.dx) < 0.3

    def test_opposing_train_ignored_with_direction(self):
        """Review Q5: opposing-direction cluster never selected."""
        dxs = np.array([0.0] * 10 + [4.0] * 12 + [-7.0] * 20)
        dys = np.zeros_like(dxs)
        clusters = _cluster_1d(dxs, dys)
        sel, _ = _select(clusters, direction=1)
        assert sel is not None and sel.dx == pytest.approx(4.0, abs=0.3)

    def test_too_few_moving_tracks_not_selected(self):
        dxs = np.array([0.0] * 20 + [5.0] * 3)  # 3 < MIN_TRACKS
        clusters = _cluster_1d(dxs, np.zeros_like(dxs))
        sel, _ = _select(clusters, direction=0)
        assert sel is None


class TestConfidence:
    def test_fast_train_tight_cluster_is_confident(self):
        """A large cluster at high speed has high ABSOLUTE spread but is still
        reliable — confidence must use relative spread (real-footage bug: a
        boxcar at 45 px/frame was wrongly flagged un-estimable)."""
        n = 300
        dxs = np.full(n, 45.0) + np.random.default_rng(0).normal(0, 4.0, n)  # spread ~4
        clusters = _cluster_1d(dxs, np.zeros(n))
        sel, _ = _select(clusters, direction=0)
        tightness = 1.0 / (1.0 + sel.spread / (abs(sel.dx) + 1.0))
        conf = min(1.0, sel.size / 16.0) * tightness
        assert conf > 0.7

    def test_slow_scattered_cluster_is_unconfident(self):
        dxs = np.array([1.0, 2.5, 0.5, 3.0, 1.5, 2.0, 0.8, 2.8, 1.2])  # spread ~ dx
        clusters = _cluster_1d(dxs, np.zeros_like(dxs))
        sel, _ = _select(clusters, direction=0)
        if sel is not None:
            tightness = 1.0 / (1.0 + sel.spread / (abs(sel.dx) + 1.0))
            conf = min(1.0, sel.size / 16.0) * tightness
            assert conf < 0.5


class TestDirectionLocking:
    def test_dominant_direction_from_weighted_sum(self):
        samples = _samples_from([5.0] * 40 + [-30.0] * 5)  # mostly +x, brief big -x
        assert motion.dominant_direction(samples) == 1

    def test_reject_opposing_nulls_wrong_direction(self):
        """Confident backward estimates on a periodic car side (review M3) are
        nulled so smooth() interpolates across them; the pass stays one segment
        instead of fragmenting on a fake reversal."""
        samples = _samples_from([6.0] * 20 + [-25.0] * 8 + [6.0] * 20, conf=0.9)
        motion.reject_opposing(samples, direction=1)
        assert all(s.confidence == 0.0 for s in samples[21:29])
        out = motion.smooth(samples, smooth_s=0.15, nominal_fps=30.0)
        # the nulled run is bridged to ~+6, never composited as -25
        for s in out[21:29]:
            assert s.dx_smooth > 0
            assert s.dx_smooth == pytest.approx(6.0, abs=3.0)

    def test_reject_opposing_keeps_small_noise(self):
        samples = _samples_from([5.0, -0.1, 5.0, -0.2], conf=0.9)
        motion.reject_opposing(samples, direction=1)
        # tiny wrong-sign values are noise, not nulled
        assert samples[2].confidence == 0.9


class TestPairEstimation:
    def test_known_shift_recovered(self):
        """Full estimate_series on a 3-frame synthetic shift, full resolution."""
        tex = make_texture(400, seed=7)
        bg = make_background(textured=True)
        speed = 4.0
        positions = [100.0, 100.0 + speed, 100.0 + 2 * speed]
        frames = [(i * 33.3, f) for i, f in enumerate(render_frames(tex, positions, bg))]
        samples = motion.estimate_series(iter(frames), scale=1.0)
        assert samples[1].dx == pytest.approx(speed, abs=0.3)
        assert samples[2].dx == pytest.approx(speed, abs=0.3)
        assert samples[1].confidence > 0.3


class TestSmoothing:
    def test_vfr_double_interval_not_clipped(self):
        """Review M4: doubled Δt → doubled dx is legitimate, not an outlier."""
        dt = 33.3
        samples = [MotionSample(t_ms=0, dt_ms=0, dx=0, dy=0, confidence=0)]
        t = 0.0
        for i in range(40):
            step = dt * 2 if i == 20 else dt  # one dropped frame
            t += step
            samples.append(MotionSample(t_ms=t, dt_ms=step,
                                        dx=5.0 * step / dt, dy=0.0, confidence=1.0))
        out = motion.smooth(samples, smooth_s=0.15, nominal_fps=30.0)
        doubled = out[21]
        assert doubled.dx_smooth == pytest.approx(10.0, rel=0.1)  # kept, not clamped

    def test_low_confidence_substituted_with_prediction(self):
        dxs = [5.0] * 15 + [40.0] + [5.0] * 15  # one garbage frame, low confidence
        samples = _samples_from(dxs)
        samples[16].confidence = 0.05
        out = motion.smooth(samples, smooth_s=0.15, nominal_fps=30.0)
        assert out[16].dx_smooth == pytest.approx(5.0, rel=0.15)
        assert out[16].substituted

    def test_clustered_low_confidence_bridged_by_interpolation(self):
        """Consecutive failures must be linearly interpolated from confident
        neighbours, not pulled to a median of the garbage values."""
        dxs = [5.0] * 15 + [0.0, 0.0, 0.0, 0.0] + [5.0] * 15  # 4 consecutive zeros
        samples = _samples_from(dxs)
        for s in samples[16:20]:
            s.confidence = 0.05
        out = motion.smooth(samples, smooth_s=0.15, nominal_fps=30.0)
        for i in range(16, 20):
            assert out[i].dx_smooth == pytest.approx(5.0, rel=0.2), i
            assert out[i].substituted

    def test_smooth_marks_substituted_but_does_not_abort(self):
        """The abort is segment-scoped and lives in the pipeline now; smooth()
        only flags substituted frames so the pipeline can count them."""
        dxs = [5.0] * 10 + [5.0] * 20 + [5.0] * 10
        samples = _samples_from(dxs)
        for s in samples[11:31]:
            s.confidence = 0.05
        out = motion.smooth(samples, smooth_s=0.15, nominal_fps=30.0)  # no raise
        assert motion.max_consecutive_substituted(out) >= 20

    def test_static_video_low_confidence_stays_zero(self):
        """Textureless static video → ~0 everywhere (segment detector then
        yields exit 2; never an estimation crash)."""
        samples = _samples_from([0.0] * 60, conf=0.1)
        out = motion.smooth(samples, smooth_s=0.15, nominal_fps=30.0)
        assert all(abs(s.dx_smooth) < 0.1 for s in out)
