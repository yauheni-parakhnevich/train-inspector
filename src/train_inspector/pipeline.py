"""Two-pass pipeline orchestration per spec §8.

Pass 1: reduced-resolution motion estimation over the (trimmed) clip.
Pass 2: full-resolution re-decode of the detected segment, aligned by
TIMESTAMP with tolerance of half the median frame interval — never by frame
index (review B2). Optional per-frame phase-correlation refinement near the
slit (review M6).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import composite, decode, diagnostics, encode, flow, motion, segment

log = logging.getLogger(__name__)

PASS1_SCALE = 0.5
PASS1_FULLRES_ROI_W = 960  # ROI narrower than this → pass 1 runs at full res (spec §8)
BLUR_WARN_DX = 25.0  # px/frame (review N10)
DY_WARN_FRAC = 0.02  # cumulative |dy| beyond 2% of height → "camera not static"


class NoMotionError(Exception):
    """No qualifying train segment (exit code 2)."""


@dataclass
class Options:
    input: Path
    output: Path
    column: float = 0.5
    roi: decode.Roi | None = None
    direction: str = "auto"  # auto | ltr | rtl
    min_speed: float = 1.0
    smooth_s: float = 0.15
    start_ms: float | None = None
    end_ms: float | None = None
    scale: float = 1.0
    max_width: int = 100_000
    jpeg_quality: int = 95
    fast: bool = False
    debug_dir: Path | None = None


@dataclass
class Result:
    output: Path
    width: int
    height: int
    direction: str
    n_frames: int
    mean_dx: float


def run(opts: Options) -> Result:
    info = decode.probe(str(opts.input))
    roi = opts.roi.clamp(info.width, info.height) if opts.roi else None
    roi_w = roi.w if roi else info.width
    roi_h = roi.h if roi else info.height
    slit_x = opts.column * roi_w
    if not (0 <= int(slit_x) < roi_w):
        raise decode.InputError(f"--column {opts.column} falls outside the ROI")

    forced_dir = {"ltr": 1, "rtl": -1}.get(opts.direction, 0)

    # ---- Pass 1: motion series at reduced resolution ----
    scale = 1.0 if roi_w <= PASS1_FULLRES_ROI_W else PASS1_SCALE
    log.info("pass 1: motion estimation (scale %.2f)", scale)
    frames = decode.iter_frames(
        str(opts.input), start_ms=opts.start_ms, end_ms=opts.end_ms, roi=roi
    )
    samples = motion.estimate_series(frames, scale=scale, slit_frac=opts.column,
                                     direction=forced_dir)
    if len(samples) < 2:
        raise decode.InputError("video has fewer than 2 decodable frames")

    # Lock travel direction over the whole clip and reject estimates that
    # oppose it. A train passes one way; a large but WRONG flow cluster on a
    # periodic/low-texture car side (corrugated boxcar, see review M3) can be
    # confident yet point backwards. Nulling the opposers lets smooth()
    # interpolate across them instead of fragmenting the pass or compositing
    # reversed strips. (v1 trade-off: genuine mid-clip shunting is treated as a
    # single direction — acceptable for the stated use case.)
    dom = forced_dir or motion.dominant_direction(samples)
    if dom:
        motion.reject_opposing(samples, dom)

    samples = motion.smooth(samples, opts.smooth_s, info.nominal_fps)

    segments = segment.detect_segments(samples, opts.min_speed, info.nominal_fps)
    seg = segment.pick_segment(segments, samples)

    if opts.debug_dir:
        opts.debug_dir.mkdir(parents=True, exist_ok=True)
        diagnostics.write_motion_csv(opts.debug_dir / "motion.csv", samples)
        diagnostics.write_motion_plot(opts.debug_dir / "motion.png", samples, seg)
        diagnostics.write_segments_txt(opts.debug_dir / "segments.txt", segments, samples)

    if seg is None:
        raise NoMotionError("no train motion detected")
    direction = forced_dir or seg.direction

    seg_samples = samples[seg.start : seg.end]

    # Segment-scoped quality gate (review Q3). Un-estimable frames inside the
    # segment (e.g. a uniform boxcar side that defeats tracking, review M3) are
    # bridged by velocity interpolation in smooth(): the strips are still real
    # train pixels, only their width is a linear guess between confident
    # endpoints — bounded stretch, not a seam (NFR-2). So we ABORT only when the
    # footage is genuinely unusable: a single bridge longer than ~1.5 s of an
    # unknown-speed surface, or more than half the segment un-estimable. A
    # smaller bridge is real but lossy, so we WARN.
    n_sub = motion.max_consecutive_substituted(seg_samples)
    frac_sub = sum(s.substituted for s in seg_samples) / len(seg_samples)
    big_bridge = max(int(round(1.5 * info.nominal_fps)), 12)
    if n_sub > big_bridge or frac_sub > 0.5:
        raise motion.ProcessingError(
            f"segment too un-estimable (longest bridge {n_sub} frames, "
            f"{frac_sub:.0%} substituted); try --roi to isolate the train, or check "
            "focus/motion blur"
        )
    if n_sub > info.nominal_fps * 0.5:
        log.warning(
            "%d consecutive frames un-estimable (a uniform car side?); their width "
            "is interpolated — expect mild local stretch there", n_sub,
        )
    mean_dx = float(np.mean([abs(s.dx_smooth) for s in seg_samples]))
    if mean_dx > BLUR_WARN_DX:
        log.warning(
            "mean displacement %.1f px/frame is high; output sharpness is "
            "bounded by the source motion blur (shutter speed)", mean_dx,
        )

    # Early output validation, before pass 2 spends time compositing (FR-5).
    predicted_w = seg.predicted_width(samples)
    encode.validate_output(opts.output, predicted_w, roi_h, opts.scale, opts.max_width)
    log.info(
        "segment: %.1fs, %d frames, direction %s, predicted width %d px",
        seg.duration_ms(samples) / 1000.0, len(seg_samples),
        "ltr" if direction > 0 else "rtl", predicted_w,
    )

    # ---- Pass 2: full-res composite, timestamp-aligned ----
    # Pass 2 builds each strip as a per-row motion-compensated cross-dissolve
    # between the pair of frames bounding the interval (flow.py), removing the
    # inter-frame seam. The flow's per-row median also subsumes the old
    # phase-correlation refinement. --fast keeps the original wide-strip path.
    interp = None if opts.fast else flow.BandInterpolator()
    comp = composite.Compositor(roi_h, direction, slit_x, fast=opts.fast)
    t_first, t_last = seg_samples[0].t_ms, seg_samples[-1].t_ms
    dt_med = float(np.median([s.dt_ms for s in samples[1:] if s.dt_ms > 0])) or 33.3
    tol = dt_med / 2.0

    frames2 = decode.iter_frames(
        str(opts.input),
        start_ms=None, end_ms=t_last + tol, roi=roi,
        seek_hint_ms=max(0.0, t_first - 2000.0),
    )

    # Timestamp matching: both passes decode sequentially through the same
    # region, so we advance a cursor over seg_samples and match each decoded
    # frame to at most one sample within tolerance (spec §8). Each matched frame
    # forms a pair with the previous matched frame; the first only seeds prev.
    idx = 0
    n_matched = 0
    n_used = 0
    dy_cum = 0.0
    dy_warned = False
    prev_frame: np.ndarray | None = None
    sample_frame = None

    for t_ms, frame in frames2:
        if idx >= len(seg_samples):
            break
        if t_ms < t_first - tol:
            continue
        # advance past samples this frame already overshot (decode hiccups)
        while idx < len(seg_samples) and seg_samples[idx].t_ms < t_ms - tol:
            log.debug("pass-2: no frame matched sample at t=%.1f ms", seg_samples[idx].t_ms)
            idx += 1
        if idx >= len(seg_samples):
            break
        s = seg_samples[idx]
        if abs(s.t_ms - t_ms) > tol:
            continue  # frame between samples (shouldn't happen; be lenient)
        idx += 1
        n_matched += 1

        if sample_frame is None:
            sample_frame = frame.copy()

        dy_cum += s.dy_smooth
        if not dy_warned and abs(dy_cum) > DY_WARN_FRAC * roi_h:
            log.warning("cumulative vertical drift %.1f px — camera does not look static", dy_cum)
            dy_warned = True

        if prev_frame is not None:
            comp.add(prev_frame, s.dx_smooth, dy_cum, frame_next=frame, interp=interp)
        else:
            # First matched frame: emit a single-frame strip (no pair available yet)
            # so total strip count stays identical to the original path.
            comp.add(frame, s.dx_smooth, dy_cum)
        n_used += 1
        prev_frame = frame

    matched_frac = n_matched / len(seg_samples)
    if matched_frac < 0.9:
        raise motion.ProcessingError(
            f"pass-2 alignment matched only {n_matched}/{len(seg_samples)} frames; "
            "decoder timestamps are unstable for this file"
        )

    mosaic = comp.mosaic()
    if opts.debug_dir and sample_frame is not None:
        diagnostics.write_overlays(opts.debug_dir, sample_frame, int(slit_x), None)

    encode.write_image(opts.output, mosaic, opts.scale, opts.jpeg_quality)
    out_w = int(mosaic.shape[1] * opts.scale)
    return Result(
        output=opts.output, width=out_w, height=int(mosaic.shape[0] * opts.scale),
        direction="ltr" if direction > 0 else "rtl", n_frames=n_used, mean_dx=mean_dx,
    )
