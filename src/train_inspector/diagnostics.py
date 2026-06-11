"""Diagnostic artifacts per spec FR-6. CSV always; PNG chart only if
matplotlib (optional [debug] extra) is installed.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import cv2
import numpy as np

from .motion import MotionSample
from .segment import Segment

log = logging.getLogger(__name__)


def write_motion_csv(path: Path, samples: list[MotionSample]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_ms", "dt_ms", "dx_raw", "dy_raw", "dx_smooth", "dy_smooth",
                    "confidence", "substituted"])
        for s in samples:
            w.writerow([f"{s.t_ms:.3f}", f"{s.dt_ms:.3f}", f"{s.dx:.4f}", f"{s.dy:.4f}",
                        f"{s.dx_smooth:.4f}", f"{s.dy_smooth:.4f}",
                        f"{s.confidence:.3f}", int(s.substituted)])


def write_motion_plot(path: Path, samples: list[MotionSample], segment: Segment | None) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        log.info("matplotlib not installed; skipping plot (install with [debug] extra)")
        return
    t = [s.t_ms / 1000.0 for s in samples]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, [s.dx for s in samples], ".", ms=2, alpha=0.4, label="dx raw")
    ax.plot(t, [s.dx_smooth for s in samples], "-", lw=1.5, label="dx smooth")
    if segment is not None:
        ax.axvspan(t[segment.start], t[segment.end - 1], alpha=0.15, color="green",
                   label="segment")
    ax.set_xlabel("time, s")
    ax.set_ylabel("displacement, px/frame")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def write_overlays(
    debug_dir: Path,
    sample_frame: np.ndarray,
    slit_x: int,
    motion_band: np.ndarray | None,
) -> None:
    """ROI/slit overlays (spec FR-6): slit position + slit-quality (static
    high-gradient content at the column) + suggested-ROI motion band."""
    h, w = sample_frame.shape[:2]
    overlay = sample_frame.copy()
    cv2.line(overlay, (slit_x, 0), (slit_x, h - 1), (0, 0, 255), 1)

    # Slit-quality: highlight strong vertical gradients near the slit that are
    # likely static occluders (poles, masts) — review N9.
    gray = cv2.cvtColor(sample_frame, cv2.COLOR_BGR2GRAY)
    band = gray[:, max(0, slit_x - 3) : min(w, slit_x + 4)]
    grad = np.abs(cv2.Sobel(band, cv2.CV_32F, 1, 0, ksize=3)).mean(axis=1)
    strong = grad > np.percentile(grad, 95)
    for y in np.flatnonzero(strong):
        cv2.circle(overlay, (slit_x, int(y)), 2, (0, 255, 255), -1)

    if motion_band is not None and motion_band.size == h:
        # Suggested-ROI: rows with high temporal variance (review OQ1).
        norm = motion_band / (motion_band.max() + 1e-9)
        active = norm > 0.25
        ys = np.flatnonzero(active)
        if len(ys):
            cv2.rectangle(overlay, (0, int(ys[0])), (w - 1, int(ys[-1])), (0, 255, 0), 2)
            cv2.putText(overlay, "suggested --roi band", (8, max(20, int(ys[0]) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imwrite(str(debug_dir / "slit_overlay.png"), overlay)


def write_segments_txt(path: Path, segments: list[Segment], samples: list[MotionSample]) -> None:
    with open(path, "w") as f:
        for i, s in enumerate(segments):
            f.write(
                f"segment {i}: samples [{s.start}, {s.end}) "
                f"t=[{samples[s.start].t_ms:.0f}, {samples[s.end - 1].t_ms:.0f}] ms "
                f"dir={'ltr' if s.direction > 0 else 'rtl'} "
                f"predicted_width={s.predicted_width(samples)}px\n"
            )
