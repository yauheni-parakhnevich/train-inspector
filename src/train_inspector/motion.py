"""Motion estimation per spec FR-2 / §10.1–10.2.

Primary: pyramidal Lucas-Kanade on Shi-Tomasi corners, aggregated by
dominant-moving-cluster selection (review B1). Fallback: phase correlation in
a band around the sampling column. Smoothing operates on velocity (px/s)
against real timestamps (review M4).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Iterable, Iterator

import cv2
import numpy as np

log = logging.getLogger(__name__)

MIN_TRACKS = 8
STATIC_DX = 0.3  # px; |median dx| below this = static (background) cluster
CLUSTER_GAP = 1.5  # px; 1-D gap-split merge radius
MIN_CONFIDENCE = 0.3
PHASE_BAND_HALF_W = 64  # px around slit for phase-correlation fallback

_LK_PARAMS = dict(
    winSize=(21, 21),
    maxLevel=3,
    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
)
_FEATURE_PARAMS = dict(maxCorners=400, qualityLevel=0.01, minDistance=7, blockSize=7)


@dataclass
class MotionSample:
    t_ms: float
    dt_ms: float  # interval to previous frame
    dx: float  # raw selected estimate, full-res px (0 for the first frame)
    dy: float  # vertical estimate of the *static* cluster (camera jitter), full-res px
    confidence: float
    substituted: bool = False  # raw estimate discarded, prediction held
    # filled by smooth():
    dx_smooth: float = 0.0
    dy_smooth: float = 0.0


@dataclass(frozen=True)
class _Cluster:
    dx: float  # median dx
    dy: float  # median dy
    size: int
    spread: float  # std of member dx


def _cluster_1d(dxs: np.ndarray, dys: np.ndarray) -> list[_Cluster]:
    """Sort + gap-split 1-D clustering on dx (spec §10.1 step 3)."""
    order = np.argsort(dxs)
    sdx, sdy = dxs[order], dys[order]
    clusters: list[_Cluster] = []
    start = 0
    for i in range(1, len(sdx) + 1):
        if i == len(sdx) or sdx[i] - sdx[i - 1] > CLUSTER_GAP:
            member_dx = sdx[start:i]
            member_dy = sdy[start:i]
            clusters.append(
                _Cluster(
                    dx=float(np.median(member_dx)),
                    dy=float(np.median(member_dy)),
                    size=i - start,
                    spread=float(np.std(member_dx)),
                )
            )
            start = i
    return clusters


def _select(clusters: list[_Cluster], direction: int) -> tuple[_Cluster | None, _Cluster | None]:
    """Return (selected moving cluster, static cluster).

    Largest moving cluster consistent with `direction` (0 = unknown → largest
    moving overall). Static cluster returned separately for jitter estimation.
    The static cluster is never selected as motion (review B1).
    """
    static = max((c for c in clusters if abs(c.dx) < STATIC_DX), key=lambda c: c.size, default=None)
    moving = [c for c in clusters if abs(c.dx) >= STATIC_DX]
    if direction:
        directed = [c for c in moving if np.sign(c.dx) == direction]
        moving = directed or moving
    sel = max(moving, key=lambda c: c.size, default=None)
    if sel is not None and sel.size < MIN_TRACKS:
        sel = None
    return sel, static


def _phase_fallback(
    prev_gray: np.ndarray, gray: np.ndarray, slit_x: int
) -> tuple[float, float, float]:
    """Phase correlation on a band around the slit. Returns (dx, dy, response)."""
    x0 = max(0, slit_x - PHASE_BAND_HALF_W)
    x1 = min(gray.shape[1], slit_x + PHASE_BAND_HALF_W)
    a = prev_gray[:, x0:x1].astype(np.float32)
    b = gray[:, x0:x1].astype(np.float32)
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a, b, win)
    # phaseCorrelate returns the shift of b relative to a as (shift applied to
    # a to get b); content moving right yields positive dx.
    return float(dx), float(dy), float(response)


def estimate_series(
    frames: Iterable[tuple[float, np.ndarray]],
    scale: float = 0.5,
    slit_frac: float = 0.5,
    direction: int = 0,
) -> list[MotionSample]:
    """Pass-1 estimation. `frames` yields (t_ms, BGR). Returns full-res-px samples."""
    samples: list[MotionSample] = []
    prev_gray: np.ndarray | None = None
    prev_t = 0.0
    inv = 1.0 / scale

    for t_ms, frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if scale != 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

        if prev_gray is None:
            samples.append(MotionSample(t_ms=t_ms, dt_ms=0.0, dx=0.0, dy=0.0, confidence=0.0))
            prev_gray, prev_t = gray, t_ms
            continue

        dt_ms = t_ms - prev_t
        dx, dy, conf = _estimate_pair(prev_gray, gray, slit_frac, direction)
        samples.append(
            MotionSample(
                t_ms=t_ms, dt_ms=dt_ms, dx=dx * inv, dy=dy * inv, confidence=conf
            )
        )
        prev_gray, prev_t = gray, t_ms

    return samples


def _estimate_pair(
    prev_gray: np.ndarray, gray: np.ndarray, slit_frac: float, direction: int
) -> tuple[float, float, float]:
    """Estimate (dx, dy_static, confidence) between two grayscale frames (scaled px).

    Estimation is over the full frame (not a slit-centered band): the
    dominant-moving-cluster selection already rejects a textured static
    background, and bridging the occasional un-estimable frame by interpolation
    in smooth() handles entry/exit. A narrow band was tried and is worse — it
    starves the estimator of corners while the train is off-center.
    """
    slit_x = int(gray.shape[1] * slit_frac)
    p0 = cv2.goodFeaturesToTrack(prev_gray, **_FEATURE_PARAMS)

    if p0 is not None and len(p0) >= MIN_TRACKS:
        p1, status, _err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **_LK_PARAMS)
        good = status.ravel() == 1
        if int(good.sum()) >= MIN_TRACKS:
            d = (p1[good] - p0[good]).reshape(-1, 2)
            clusters = _cluster_1d(d[:, 0], d[:, 1])
            sel, static = _select(clusters, direction)
            dy_static = static.dy if static is not None else 0.0
            if sel is not None:
                # Confidence from track COUNT and RELATIVE spread. Absolute
                # spread scales with speed and perspective (a fast or angled
                # train legitimately spreads flow vectors), so normalise spread
                # by the motion magnitude — a 4 px spread on 45 px of motion is
                # tight, not unreliable. Hundreds of agreeing tracks => high
                # confidence regardless of absolute spread.
                tightness = 1.0 / (1.0 + sel.spread / (abs(sel.dx) + 1.0))
                conf = min(1.0, sel.size / 16.0) * tightness
                return sel.dx, dy_static, conf
            # No qualifying moving cluster: report 0 motion, but with LOW
            # confidence. We cannot tell "scene genuinely static" from "train
            # present but its corners lost LK this frame" (common when a
            # textured background supplies the majority of trackable corners).
            # Returning low confidence makes smooth() substitute the velocity
            # prediction mid-segment (so a spurious 0 does not delete a strip of
            # train), while a *genuinely* static clip stays ~0 everywhere and is
            # correctly rejected by the segment detector (exit 2). See review B1.
            if static is not None and static.size >= MIN_TRACKS:
                return 0.0, dy_static, MIN_CONFIDENCE * 0.5
    # LK starved (blank wagon side): phase-correlation fallback near the slit.
    dx, dy, response = _phase_fallback(prev_gray, gray, slit_x)
    return dx, dy, min(1.0, max(0.0, response))


def _rolling_median(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) < 3:
        return x.copy()
    w = min(w | 1, len(x) if len(x) % 2 else len(x) - 1)  # odd, <= len
    pad = w // 2
    xp = np.pad(x, pad, mode="edge")
    return np.median(np.lib.stride_tricks.sliding_window_view(xp, w), axis=1)


def _lowpass(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1 or len(x) < 3:
        return x.copy()
    w = min(w, len(x))
    kernel = np.ones(w) / w
    pad = w // 2
    xp = np.pad(x, (pad, w - 1 - pad), mode="edge")
    return np.convolve(xp, kernel, mode="valid")


def smooth(samples: list[MotionSample], smooth_s: float, nominal_fps: float) -> list[MotionSample]:
    """Velocity-domain smoothing (spec §10.2) + low-confidence substitution (§10.1).

    Returns new samples with dx_smooth/dy_smooth/substituted filled. The
    consecutive-substitution abort (review Q3) is NOT done here — it is
    segment-scoped and lives in the pipeline, because a long run of low
    confidence *before* the train arrives (static scene) is normal, not a
    failure.
    """
    if not samples:
        return []
    out = [replace(s) for s in samples]

    dt_s = np.array([max(s.dt_ms, 1e-3) / 1000.0 for s in out])
    vx = np.array([s.dx for s in out]) / dt_s
    vy = np.array([s.dy for s in out]) / dt_s

    low = np.array([s.confidence < MIN_CONFIDENCE for s in out])
    low[0] = True  # first frame carries no estimate

    # Bridge low-confidence frames by LINEAR INTERPOLATION from the nearest
    # confident neighbours (not a rolling median — clustered failures, common
    # over textured backgrounds, would drag a median to the garbage value).
    idx = np.arange(len(out))
    valid = ~low
    if valid.sum() >= 2:
        vx = np.where(low, np.interp(idx, idx[valid], vx[valid]), vx)
        vy = np.where(low, np.interp(idx, idx[valid], vy[valid]), vy)
    elif valid.sum() == 1:
        vx = np.where(low, vx[valid][0], vx)
        vy = np.where(low, vy[valid][0], vy)
    else:
        vx[:] = 0.0
        vy[:] = 0.0

    # Outlier rejection vs a robust local baseline (spec §10.1 step 7).
    med_w = max(3, int(round(smooth_s * nominal_fps)) | 1)
    vx_med = _rolling_median(vx, med_w)
    dev = np.abs(vx - vx_med)
    tol = np.maximum(3.0 * nominal_fps, 0.3 * np.abs(vx_med))  # 3 px/frame or 30%
    bad = dev > tol
    vx = np.where(bad, vx_med, vx)

    w = max(1, int(round(smooth_s * nominal_fps)))
    vx_s = _lowpass(_rolling_median(vx, w | 1), max(1, w // 2))
    vy_s = _lowpass(_rolling_median(vy, w | 1), max(1, w // 2))

    for i, s in enumerate(out):
        s.dx_smooth = float(vx_s[i] * dt_s[i])
        s.dy_smooth = float(vy_s[i] * dt_s[i])
        s.substituted = bool(low[i] or bad[i])
    out[0].dx_smooth = 0.0
    return out


def dominant_direction(samples: list[MotionSample]) -> int:
    """Global travel direction (+1 ltr / -1 rtl / 0 none) from the
    confidence-weighted sum of displacements over the whole clip. Robust: the
    real, sustained motion outweighs sporadic wrong-direction estimates."""
    s = sum(x.dx * max(x.confidence, 0.05) for x in samples)
    if abs(s) < 1e-6:
        return 0
    return 1 if s > 0 else -1


def reject_opposing(samples: list[MotionSample], direction: int) -> None:
    """Mark estimates that oppose `direction` as un-estimable (confidence 0) so
    smooth() interpolates across them. Only meaningful displacements are
    rejected — a near-zero dx of the wrong sign is just noise and harmless."""
    for s in samples:
        if s.dx * direction < 0 and abs(s.dx) > STATIC_DX:
            s.confidence = 0.0


def max_consecutive_substituted(samples: list[MotionSample]) -> int:
    """Longest run of substituted frames — used by the pipeline to abort a
    segment whose interior was largely un-estimable (review Q3)."""
    return _max_consecutive(np.array([s.substituted for s in samples]))


def _max_consecutive(mask: np.ndarray) -> int:
    best = run = 0
    for m in mask:
        run = run + 1 if m else 0
        best = max(best, run)
    return best


class ProcessingError(Exception):
    """Unrecoverable estimation failure (exit code 3)."""
