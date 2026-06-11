"""Test metrics: SSIM (global, Gaussian-windowed) and panorama/texture
alignment, per spec §11 acceptance criteria."""

from __future__ import annotations

import cv2
import numpy as np


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Mean SSIM, 11x11 Gaussian window, standard constants."""
    assert a.shape == b.shape, f"shape mismatch {a.shape} vs {b.shape}"
    if a.ndim == 3:
        a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    blur = lambda x: cv2.GaussianBlur(x, (11, 11), 1.5)
    mu_a, mu_b = blur(a), blur(b)
    sa = blur(a * a) - mu_a * mu_a
    sb = blur(b * b) - mu_b * mu_b
    sab = blur(a * b) - mu_a * mu_b
    num = (2 * mu_a * mu_b + c1) * (2 * sab + c2)
    den = (mu_a**2 + mu_b**2 + c1) * (sa + sb + c2)
    return float((num / den).mean())


def extract_train_region(panorama: np.ndarray, train_y: int, train_h: int,
                         bg_value: int = 96, tol: int = 25) -> np.ndarray:
    """Crop the train band rows, then the columns that are not flat background
    (uniform-gray fixtures make train columns trivially separable)."""
    band = panorama[train_y : train_y + train_h]
    gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY).astype(np.int16)
    col_dev = np.abs(gray - bg_value).mean(axis=0)
    train_cols = np.flatnonzero(col_dev > tol)
    assert len(train_cols) > 0, "no train content found in panorama"
    return band[:, train_cols[0] : train_cols[-1] + 1]


def align_and_ssim(region: np.ndarray, texture: np.ndarray, max_shift: int = 8) -> float:
    """Best SSIM of region vs texture over small horizontal alignment shifts
    (compositing is defined up to a sub-pixel global offset)."""
    th, tw = texture.shape[:2]
    rh, rw = region.shape[:2]
    assert rh == th, f"height mismatch {rh} vs {th}"
    best = -1.0
    for shift in range(-max_shift, max_shift + 1):
        lo = max(0, shift)
        hi = min(rw, tw + shift)
        if hi - lo < tw - max_shift:
            continue
        r = region[:, lo:hi]
        t = texture[:, lo - shift : hi - shift]
        if r.shape[1] < 100:
            continue
        best = max(best, ssim(r, t))
    return best


def horizontal_seam_energy(region: np.ndarray) -> float:
    """Mean absolute column-to-column difference. Hard strip seams (an abrupt
    frame/exposure jump every ~dx columns) inflate this; a smooth cross-dissolve
    lowers it. Used as a RELATIVE metric (flow path vs --fast path on the same
    fixture), so texture content cancels out."""
    g = region.astype(np.float64)
    return float(np.abs(np.diff(g, axis=1)).mean())
