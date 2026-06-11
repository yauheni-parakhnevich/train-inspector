# Flow-Based Sub-Time Supersampling at the Slit — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the periodic vertical seams in the panorama by replacing each frame's single wide snapshot strip with a per-row, motion-compensated cross-dissolve between the two frames bounding the interval, so consecutive strips meet at a shared frame and each row tiles at its own displacement.

**Architecture:** New `flow.py` owns a banded dense-optical-flow estimator (`cv2.DISOpticalFlow`) and the strip synthesizer. `composite.py`'s `Compositor.add` gains an optional flow path; the existing `warpAffine` wide strip stays as the automatic fallback (flow rejected) and as `--fast`. `pipeline.py`'s pass-2 loop feeds frame *pairs* and the flow median subsumes the old `_refine_dx` step. Carry/`Σw` geometry is untouched.

**Tech Stack:** Python 3.11+, OpenCV (pinned, `cv2.DISOpticalFlow_create` is core — verified on cv2 4.11), NumPy, pytest with in-repo FFV1 fixtures.

**Design doc:** `docs/superpowers/specs/2026-06-11-slit-flow-supersampling-design.md`

---

## File Structure

| File | Create/Modify | Responsibility |
|---|---|---|
| `src/train_inspector/flow.py` | **Create** | `BandInterpolator` (DIS handle, `analyze`, `synthesize`), `FlowBand` dataclass, constants. Pure over inputs. |
| `src/train_inspector/composite.py` | Modify | `Compositor.add` gains `frame_next`/`interp` kwargs + flow path; wide-strip path unchanged. |
| `src/train_inspector/pipeline.py` | Modify | Pass-2 loop feeds pairs; create `BandInterpolator` when not `--fast`; delete `_refine_dx`/`_slit_band`/`REFINE_BAND_HALF_W`. |
| `src/train_inspector/cli.py` | Modify | `--fast` help text. |
| `tests/test_flow.py` | **Create** | Unit tests: flow recovery, rejection, cross-dissolve, τ direction. |
| `tests/test_composite.py` | Modify | Add fallback-wiring test; existing tests stay green. |
| `tests/conftest.py` | Modify | Add `flicker_frames` fixture generator. |
| `tests/helpers.py` | Modify | Add `horizontal_seam_energy` metric. |
| `tests/test_e2e.py` | Modify | Pin strict SSIM to `--fast`; add default-path SSIM, seam-energy regression. |
| `spec/SPEC.md` | Modify | Fold §8/§10.4/§11/FR-7/NFR-1 changes. |

**Sign / geometry conventions (read before coding):**
- `dx_smooth > 0` ⇒ content moves +x (LTR travel, `direction=+1`); `< 0` ⇒ RTL (`direction=-1`).
- Wide strip samples source columns `[slit_x − carry_in, slit_x − carry_in + w)` of `frame_k`, for **both** directions. Direction only flips assembly order in `mosaic()`.
- Strip `i` is built from the pair `(frame_{i-1}, frame_i)` with `frame_i`'s `dx_smooth` (motion from `i-1` to `i`). `S_k = frame_{i-1}`, `S_k1 = frame_i`.
- Cross-dissolve weight `t` ramps so the **later-frame** (`S_k1`) edge lands on the side that abuts the next strip in final assembly order: `t = (c+0.5)/w` for RTL, `t = 1 − (c+0.5)/w` for LTR. This makes both sides of every interior boundary resolve to the same frame ⇒ the inter-frame jump disappears (derivation in the design doc).

---

## Task 1: `flow.py` — banded flow analysis (`BandInterpolator.analyze`)

**Files:**
- Create: `src/train_inspector/flow.py`
- Test: `tests/test_flow.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_flow.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_flow.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'train_inspector.flow'`

- [ ] **Step 3: Write `flow.py` with `analyze`**

Create `src/train_inspector/flow.py`:

```python
"""Flow-based sub-time supersampling at the slit (design doc
2026-06-11-slit-flow-supersampling-design.md).

Replaces the single-frame wide strip with a per-row, motion-compensated
cross-dissolve between the two frames bounding each interval, so consecutive
strips meet at a shared frame (no inter-frame jump) and each row tiles at its
own true displacement (perspective seams removed). Localized to a band around
the slit; the caller falls back to its wide-strip path when flow is unreliable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

BAND_HALF_W = 96  # base half-width; widened below to always cover the motion
DIS_PRESET = cv2.DISOPTICAL_FLOW_PRESET_MEDIUM
RELIABLE_ABS = 2.0   # px; |median Fx − dx_seed| tolerance floor
RELIABLE_FRAC = 0.25  # or this fraction of |dx_seed|


@dataclass
class FlowBand:
    """Per-row horizontal flow sampled at the slit, plus the refined scalar dx."""

    fx_col: np.ndarray  # shape (H,), per-row horizontal displacement at the slit (px)
    dx_refined: float   # median(fx_col), signed — used for the carry/width accounting


def _smooth1d(v: np.ndarray, k: int) -> np.ndarray:
    """Edge-padded moving average. Suppresses per-row flow noise while keeping
    the low-frequency perspective gradient (which we WANT to preserve)."""
    k = max(1, k | 1)
    if k == 1 or len(v) < 3:
        return v.astype(np.float64)
    pad = k // 2
    vp = np.pad(v.astype(np.float64), pad, mode="edge")
    return np.convolve(vp, np.ones(k) / k, mode="valid")


class BandInterpolator:
    def __init__(self) -> None:
        self._dis = cv2.DISOpticalFlow_create(DIS_PRESET)

    def _band_x(self, slit_x: float, dx_seed: float, width: int) -> tuple[int, int]:
        half = max(BAND_HALF_W, int(math.ceil(abs(dx_seed))) + 16)
        x0 = max(0, int(slit_x) - half)
        x1 = min(width, int(slit_x) + half)
        return x0, x1

    def analyze(
        self, frame_k: np.ndarray, frame_k1: np.ndarray, slit_x: float, dx_seed: float
    ) -> FlowBand | None:
        """Dense flow k→k+1 in a band around the slit. Returns a FlowBand, or
        None when the flow is untrustworthy (sub-pixel motion, or median flow
        disagrees with the pass-1 seed — e.g. a uniform car side). On None the
        caller uses the wide strip, so hard regions match today's quality."""
        if abs(dx_seed) < 1.0:
            return None
        x0, x1 = self._band_x(slit_x, dx_seed, frame_k.shape[1])
        gk = cv2.cvtColor(frame_k[:, x0:x1], cv2.COLOR_BGR2GRAY)
        gk1 = cv2.cvtColor(frame_k1[:, x0:x1], cv2.COLOR_BGR2GRAY)
        flow = self._dis.calc(gk, gk1, None)  # (H, x1-x0, 2)
        col = min(max(int(round(slit_x)) - x0, 0), flow.shape[1] - 1)
        fx_col = _smooth1d(flow[:, col, 0], max(3, frame_k.shape[0] // 20))
        dx_refined = float(np.median(fx_col))
        tol = max(RELIABLE_ABS, RELIABLE_FRAC * abs(dx_seed))
        if abs(dx_refined - dx_seed) > tol:
            return None
        return FlowBand(fx_col=fx_col, dx_refined=dx_refined)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_flow.py -q`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/train_inspector/flow.py tests/test_flow.py
git commit -m "feat(flow): banded DIS optical-flow analysis with reliability gate"
```

---

## Task 2: `flow.py` — strip synthesis (`BandInterpolator.synthesize`)

**Files:**
- Modify: `src/train_inspector/flow.py`
- Test: `tests/test_flow.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_flow.py`:

```python
def test_synthesize_identity_when_frames_equal():
    """Equal frames + zero flow ⇒ the cross-dissolve must reproduce the plain
    wide strip exactly (blend of identical content), regardless of t."""
    tex = make_texture(440, H, 3)
    frame = cv2.warpAffine(
        tex, np.array([[1.0, 0, 0], [0, 1.0, 0]]), (400, H), flags=cv2.INTER_LANCZOS4
    )
    fb = FlowBand(fx_col=np.zeros(H), dx_refined=5.0)
    strip = BandInterpolator().synthesize(
        fb, frame, frame, slit_x=200.0, carry_in=0.0, w=5,
        dy_cum=0.0, direction=-1, interp_flag=cv2.INTER_LINEAR,
    )
    expected = frame[:, 200:205]
    assert strip.shape == (H, 5, 3)
    assert np.abs(strip.astype(int) - expected.astype(int)).mean() < 1.0


def test_synthesize_tau_ramps_opposite_by_direction():
    """t ramps left→right for RTL and right→left for LTR. With a dark frame_k
    and bright frame_k1 (zero flow) the brightness ramp direction must flip."""
    h, w = 8, 6
    a = np.full((h, 400, 3), 50, np.uint8)
    b = np.full((h, 400, 3), 200, np.uint8)
    fb = FlowBand(fx_col=np.zeros(h), dx_refined=6.0)
    bi = BandInterpolator()
    rtl = bi.synthesize(fb, a, b, 200.0, 0.0, w, 0.0, direction=-1, interp_flag=cv2.INTER_LINEAR)
    ltr = bi.synthesize(fb, a, b, 200.0, 0.0, w, 0.0, direction=1, interp_flag=cv2.INTER_LINEAR)
    assert rtl[0, 0, 0] < rtl[0, -1, 0]   # RTL: dark(frame_k) left, bright(frame_k1) right
    assert ltr[0, 0, 0] > ltr[0, -1, 0]   # LTR: reversed
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_flow.py -q -k synthesize`
Expected: FAIL with `AttributeError: 'BandInterpolator' object has no attribute 'synthesize'`

- [ ] **Step 3: Implement `synthesize`**

Append this method to the `BandInterpolator` class in `src/train_inspector/flow.py`:

```python
    def synthesize(
        self,
        fb: FlowBand,
        frame_k: np.ndarray,
        frame_k1: np.ndarray,
        slit_x: float,
        carry_in: float,
        w: int,
        dy_cum: float,
        direction: int,
        interp_flag: int,
    ) -> np.ndarray:
        """Per-row, motion-compensated cross-dissolve over w output columns.

        S_k  : frame_k  sampled at x = slit − carry_in + c           (the wide strip)
        S_k1 : frame_k1 sampled at x = slit − carry_in + c + fx(y)   (same surface,
               motion-compensated per row → fixes perspective)
        out[:, c] = (1 − t_c)·S_k[:, c] + t_c·S_k1[:, c]

        t ramps so S_k1's edge abuts the next strip in assembly order (design
        doc): RTL t=(c+0.5)/w, LTR t=1−(c+0.5)/w. Vertical jitter dy_cum is the
        same global translation as the wide path (single resample)."""
        h = frame_k.shape[0]
        cols = np.arange(w, dtype=np.float32)
        xs = (slit_x - carry_in) + cols                       # (w,)
        ys = (np.arange(h, dtype=np.float32) + dy_cum).reshape(h, 1)
        map_y = np.tile(ys, (1, w)).astype(np.float32)        # (h, w)

        map_x_k = np.tile(xs, (h, 1)).astype(np.float32)      # (h, w)
        fx = fb.fx_col.astype(np.float32).reshape(h, 1)
        map_x_k1 = (xs.reshape(1, w) + fx).astype(np.float32)  # (h, w)

        s_k = cv2.remap(frame_k, map_x_k, map_y, interp_flag, borderMode=cv2.BORDER_REPLICATE)
        s_k1 = cv2.remap(frame_k1, map_x_k1, map_y, interp_flag, borderMode=cv2.BORDER_REPLICATE)

        t = (cols + 0.5) / w
        if direction > 0:  # LTR: later-frame edge on the left
            t = 1.0 - t
        t = t.reshape(1, w, 1).astype(np.float32)
        out = (1.0 - t) * s_k.astype(np.float32) + t * s_k1.astype(np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_flow.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add src/train_inspector/flow.py tests/test_flow.py
git commit -m "feat(flow): per-row motion-compensated cross-dissolve strip synthesis"
```

---

## Task 3: `composite.py` — flow path in `Compositor.add`

**Files:**
- Modify: `src/train_inspector/composite.py:42-78`
- Test: `tests/test_composite.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_composite.py`:

```python
from train_inspector.flow import BandInterpolator, FlowBand


class _StubReject:
    """Interp whose flow is always rejected → must behave like the wide path."""

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
    """Real interp on a moving pair: Σw still tracks Σ|dx_refined| (no drift)."""
    tex = make_texture(440, 160, 3)
    place = lambda off: cv2.warpAffine(
        tex, np.array([[1.0, 0, off], [0, 1.0, 0]]), (400, 160), flags=cv2.INTER_LANCZOS4
    )
    a, b = place(0.0), place(6.0)
    comp = Compositor(height=160, direction=1, slit_x=200.0)
    interp = BandInterpolator()
    total = sum(comp.add(a, 6.0, frame_next=b, interp=interp) for _ in range(20))
    assert abs(total - 6 * 20) <= 20  # within ~1 px/frame of Σ|dx|
```

Add the imports `import cv2` and `from conftest import make_texture` at the top of `tests/test_composite.py` (after the existing `import numpy as np`).

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_composite.py -q -k flow_path`
Expected: FAIL with `TypeError: add() got an unexpected keyword argument 'frame_next'`

- [ ] **Step 3: Modify `Compositor.add`**

In `src/train_inspector/composite.py`, replace the entire `add` method (lines 42-78) with:

```python
    def add(
        self,
        frame: np.ndarray,
        dx_smooth: float,
        dy_cum: float = 0.0,
        frame_next: np.ndarray | None = None,
        interp=None,
    ) -> int:
        """Feed one in-segment frame pair. Returns strip width taken (may be 0).

        §10.3 accounting is UNCHANGED: total = |dx| + carry_in; w = floor(total);
        carry_out = total − w. When `interp` and `frame_next` are given and the
        flow is reliable, the w columns are a per-row motion-compensated
        cross-dissolve between `frame` (k) and `frame_next` (k+1), removing the
        inter-frame seam (see flow.py). Otherwise the original single-frame wide
        strip is taken: source region [x_slit − carry_in, x_slit − carry_in + w)
        of `frame`, for BOTH directions; direction is handled purely by assembly
        order in mosaic(). dst(x, y) = src(x + a, y + dy_cum)."""
        fb = None
        dx_used = dx_smooth
        if interp is not None and frame_next is not None:
            fb = interp.analyze(frame, frame_next, self.slit_x, dx_smooth)
            if fb is not None:
                dx_used = fb.dx_refined

        carry_in = self.carry
        total = abs(dx_used) + carry_in
        w = math.floor(total)
        self.carry = total - w
        if w <= 0:
            return 0

        if fb is not None:
            strip = interp.synthesize(
                fb, frame, frame_next, self.slit_x, carry_in, w,
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
```

- [ ] **Step 4: Run the full composite + flow suite**

Run: `uv run pytest tests/test_composite.py tests/test_flow.py -q`
Expected: PASS (all existing composite tests + new ones). The unchanged-signature tests still pass because `interp` defaults to `None`.

- [ ] **Step 5: Commit**

```bash
git add src/train_inspector/composite.py tests/test_composite.py
git commit -m "feat(composite): optional flow strip path; wide strip as fallback"
```

---

## Task 4: `pipeline.py` — wire flow into pass 2, feed pairs, drop `_refine_dx`

**Files:**
- Modify: `src/train_inspector/pipeline.py` (import line 19; constant line 26; pass-2 block lines 150-224; helpers lines 227-247)

- [ ] **Step 1: Add the `flow` import**

In `src/train_inspector/pipeline.py` line 19, change:

```python
from . import composite, decode, diagnostics, encode, motion, segment
```

to:

```python
from . import composite, decode, diagnostics, encode, flow, motion, segment
```

- [ ] **Step 2: Remove the obsolete refine constant**

In `src/train_inspector/pipeline.py`, delete line 26:

```python
REFINE_BAND_HALF_W = 96
```

(The band width now lives in `flow.BAND_HALF_W`.)

- [ ] **Step 3: Replace the pass-2 setup + loop**

In `src/train_inspector/pipeline.py`, replace the block from the `refine = ...` line through the end of the `for t_ms, frame in frames2:` loop (current lines 151-213, i.e. the comment beginning "# Refinement (review M6)" down to the loop body that ends before `matched_frac = ...`) with:

```python
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
            n_used += 1
        prev_frame = frame
```

- [ ] **Step 4: Update the alignment gate to use `n_matched`**

Immediately after the loop, replace (current lines 208-213):

```python
    matched_frac = n_used / len(seg_samples)
    if matched_frac < 0.9:
        raise motion.ProcessingError(
            f"pass-2 alignment matched only {n_used}/{len(seg_samples)} frames; "
            "decoder timestamps are unstable for this file"
        )
```

with:

```python
    matched_frac = n_matched / len(seg_samples)
    if matched_frac < 0.9:
        raise motion.ProcessingError(
            f"pass-2 alignment matched only {n_matched}/{len(seg_samples)} frames; "
            "decoder timestamps are unstable for this file"
        )
```

- [ ] **Step 5: Delete the dead refinement helpers**

In `src/train_inspector/pipeline.py`, delete the two functions `_slit_band` (current lines 227-230) and `_refine_dx` (current lines 233-247) in their entirety. Verify nothing else references them:

Run: `grep -rn "_refine_dx\|_slit_band\|REFINE_BAND_HALF_W" src/`
Expected: no output.

- [ ] **Step 6: Run the end-to-end suite (pre-existing tests)**

Run: `uv run pytest tests/test_e2e.py -q`
Expected: Most pass. `test_constant_speed_ltr` / `test_constant_speed_rtl` assert SSIM ≥ 0.98 on the default (now flow) path; they may dip slightly below 0.98 due to cross-dissolve softening — that is expected and is fixed in Task 6 (pin them to `--fast`). Do not "fix" by weakening the flow; proceed to Task 6.

- [ ] **Step 7: Commit**

```bash
git add src/train_inspector/pipeline.py
git commit -m "feat(pipeline): flow supersampling in pass 2; subsume phase-corr refine"
```

---

## Task 5: Test infrastructure — flicker fixture + seam-energy metric

**Files:**
- Modify: `tests/conftest.py` (append after `constant_speed_positions`, around line 96)
- Modify: `tests/helpers.py` (append)

- [ ] **Step 1: Add the seam-energy metric**

Append to `tests/helpers.py`:

```python
def horizontal_seam_energy(region: np.ndarray) -> float:
    """Mean absolute column-to-column difference. Hard strip seams (an abrupt
    frame/exposure jump every ~dx columns) inflate this; a smooth cross-dissolve
    lowers it. Used as a RELATIVE metric (flow path vs --fast path on the same
    fixture), so texture content cancels out."""
    g = region.astype(np.float64)
    return float(np.abs(np.diff(g, axis=1)).mean())
```

- [ ] **Step 2: Add the flicker frame generator**

Append to `tests/conftest.py`:

```python
def flicker_frames(
    texture: np.ndarray,
    positions: list[float],
    amplitude: float = 0.18,
) -> Iterator[np.ndarray]:
    """Constant-speed train whose per-frame brightness alternates ±amplitude.

    Models real exposure/auto-gain variation between frames. The wide-strip path
    stamps each frame's brightness into a full strip → a hard brightness step at
    every boundary (visible vertical lines). The flow cross-dissolve blends
    adjacent frames at each boundary → the step is smoothed. This is the
    deterministic regression fixture for the seam fix."""
    base = list(render_frames(texture, positions))
    for i, frame in enumerate(base):
        gain = 1.0 + (amplitude if i % 2 else -amplitude)
        yield np.clip(frame.astype(np.float32) * gain, 0, 255).astype(np.uint8)
```

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py tests/helpers.py
git commit -m "test: flicker fixture + horizontal seam-energy metric"
```

---

## Task 6: `test_e2e.py` — pin strict SSIM to `--fast`, add flow + seam tests

**Files:**
- Modify: `tests/test_e2e.py`

- [ ] **Step 1: Pin the strict-geometry tests to `--fast`**

In `tests/test_e2e.py`, in `test_constant_speed_ltr`, change:

```python
    result = _run(video_constant_ltr, out)
```

to:

```python
    result = _run(video_constant_ltr, out, fast=True)  # strict SSIM ≥ 0.98 = pure-geometry path
```

In `test_constant_speed_rtl`, change:

```python
    result = _run(video, out)
```

to:

```python
    result = _run(video, out, fast=True)  # strict SSIM ≥ 0.98 = pure-geometry path
```

- [ ] **Step 2: Add the failing default-path + seam tests**

Add the import for the new metric — change the `from helpers import ...` line in `tests/test_e2e.py` to:

```python
from helpers import align_and_ssim, extract_train_region, horizontal_seam_energy
```

Add `flicker_frames` to the `from conftest import (...)` block.

Then append these tests to `tests/test_e2e.py`:

```python
FLOW_SSIM = 0.97  # default flow path: cross-dissolve softens slightly vs --fast's 0.98


def test_constant_speed_flow_path_no_regression(video_constant_ltr, texture, tmp_path):
    """Default (flow) path on a rigid fixture: width ±1%, SSIM ≥ 0.97. Flow must
    refine dx accurately enough that geometry is preserved without seams."""
    out = tmp_path / "pano.png"
    _run(video_constant_ltr, out)  # default: flow ON
    pano = cv2.imread(str(out))
    region = extract_train_region(pano, TRAIN_Y, TRAIN_H)
    assert abs(region.shape[1] - TEX_LEN) <= TEX_LEN * 0.01
    assert align_and_ssim(region, texture) >= FLOW_SSIM


def test_flow_reduces_seam_energy_vs_fast(texture, tmp_path):
    """Headline regression: on a brightness-flicker fixture the wide-strip
    (--fast) path stamps a hard brightness step at every boundary; the flow
    cross-dissolve smooths them, lowering horizontal seam energy."""
    positions = constant_speed_positions(SPEED, TEX_LEN)
    video = write_video(tmp_path / "flicker.avi", flicker_frames(texture, positions))

    out_fast = tmp_path / "fast.png"
    _run(video, out_fast, fast=True)
    out_flow = tmp_path / "flow.png"
    _run(video, out_flow)  # default: flow ON

    reg_fast = extract_train_region(cv2.imread(str(out_fast)), TRAIN_Y, TRAIN_H)
    reg_flow = extract_train_region(cv2.imread(str(out_flow)), TRAIN_Y, TRAIN_H)
    e_fast = horizontal_seam_energy(reg_fast)
    e_flow = horizontal_seam_energy(reg_flow)
    assert e_flow < e_fast * 0.85, f"flow seam energy {e_flow:.2f} not < 0.85×{e_fast:.2f}"
```

- [ ] **Step 3: Run the new tests**

Run: `uv run pytest tests/test_e2e.py -q -k "flow or constant_speed"`
Expected: PASS. If `test_flow_reduces_seam_energy_vs_fast` is marginal, the flicker `amplitude` (Task 5) makes the effect stronger — but do NOT relax the 0.85 factor below 0.9 without confirming the seam is genuinely reduced (inspect `e_flow`/`e_fast` printed in the assertion message).

- [ ] **Step 4: Run the FULL suite**

Run: `uv run pytest -q`
Expected: all pass (~45 s + flow overhead). If any pre-existing e2e width assertion now drifts (flow refines dx per-frame), confirm the drift is within the stated ±1–2% tolerances; investigate, do not blindly widen tolerances.

- [ ] **Step 5: Commit**

```bash
git add tests/test_e2e.py
git commit -m "test(e2e): pin strict SSIM to --fast; add flow no-regression + seam-energy tests"
```

---

## Task 7: Docs — `--fast` help, SPEC fold-in, runtime measurement

**Files:**
- Modify: `src/train_inspector/cli.py:67`
- Modify: `spec/SPEC.md` (FR-7 line 106; §8 line 162; §11.1 line 254; NFR-1 line 110)

- [ ] **Step 1: Update `--fast` help text**

In `src/train_inspector/cli.py` line 67, change:

```python
@click.option("--fast", is_flag=True, help="Faster, slightly lower quality (bilinear, no refinement)")
```

to:

```python
@click.option("--fast", is_flag=True, help="Faster (bilinear, no flow supersampling); wide single-frame strips")
```

- [ ] **Step 2: Update SPEC §8 (pass-2 refinement bullet)**

In `spec/SPEC.md`, replace the `- **Pass-2 refinement:**` bullet (line 162) with:

```markdown
  - **Pass-2 flow supersampling (default):** each strip is built as a per-row,
    motion-compensated cross-dissolve between the two frames bounding the
    interval, using dense optical flow (`cv2.DISOpticalFlow`) in a band around
    the slit (`composite.py` + `flow.py`). Consecutive strips meet at a shared
    frame, eliminating the inter-frame seam; per-row flow tiles each row at its
    own displacement, removing perspective seams. The flow's per-row median
    subsumes the former phase-correlation refinement (review M6). Where flow is
    unreliable (sub-pixel motion, or median flow disagreeing with the pass-1
    seed — e.g. a uniform car side), the pair falls back to the single-frame
    wide strip, matching pre-flow quality. `--fast` selects the wide-strip path
    for the whole pass.
```

- [ ] **Step 3: Update SPEC FR-7 (`--fast` row)**

In `spec/SPEC.md` line 106, change:

```markdown
| `--fast` | off | Bilinear instead of Lanczos4 resampling; skips pass-2 refinement (§8). |
```

to:

```markdown
| `--fast` | off | Bilinear resampling; disables flow supersampling — single-frame wide strips (§8). |
```

- [ ] **Step 4: Update SPEC §11.1 (strict criterion is the `--fast` path)**

In `spec/SPEC.md`, in acceptance criterion 1 (line 254), change `pixelwise comparison against the **analytically known source texture** yields SSIM ≥ 0.98` to:

```markdown
pixelwise comparison against the **analytically known source texture** yields SSIM ≥ 0.98 (measured on the `--fast` wide-strip path — the pure-geometry guarantee; the default flow path is held to SSIM ≥ 0.97 with no seam energy regression)
```

- [ ] **Step 5: Update SPEC NFR-1 (re-scope the 2× target)**

In `spec/SPEC.md`, append to the end of the NFR-1 paragraph (line 110, after "...not user-visible (review Q1)."):

```markdown
 The ≥ 2× real-time target is measured on the `--fast` path; the default flow-supersampling path trades speed for seam-free output (one banded dense-flow computation per frame pair).
```

- [ ] **Step 6: Measure and record default-mode runtime**

Run (4K clip, default flow path):

```bash
time uv run train-inspector data/freight_crossing.mp4 -o output/freight_flow.png -v
time uv run train-inspector data/freight_crossing.mp4 -o output/freight_fast.png --fast -v
```

Record both wall-clock numbers in a one-line note appended to NFR-1 in `spec/SPEC.md` (e.g. "Trial: 4K ~11 s freight clip — `--fast` Xs, default Ys."). Visually confirm `output/freight_flow.png` shows reduced vertical lines vs `output/freight_fast.png`.

- [ ] **Step 7: Final full-suite run**

Run: `uv run pytest -q`
Expected: all pass.

- [ ] **Step 8: Commit**

```bash
git add src/train_inspector/cli.py spec/SPEC.md
git commit -m "docs: fold flow supersampling into SPEC; --fast help; runtime note"
```

---

## Self-Review

**Spec coverage (design doc → tasks):**
- §5 sub-time synthesis / carry preserved → Task 2 (`synthesize`), Task 3 (carry untouched in `add`).
- §6 banded DIS flow + reliability fallback → Task 1 (`analyze` gate), Task 3 (fallback branch).
- §6 consolidation (flow median replaces `_refine_dx`) → Task 4 (delete helpers; flow median feeds `dx_used`).
- §7 interfaces (`flow.py`, `Compositor.add`, pass-2 pairs) → Tasks 1–4.
- §8 `--fast` selects wide path; no new flags → Task 3 (`interp=None`), Task 4 (`interp` gated on `opts.fast`), Task 7 (help).
- §9 perf re-scope → Task 7 (NFR-1, runtime measurement).
- §10 testing: strict pinned to `--fast` (T1/§11.1) → Task 6 Step 1; relaxed flow SSIM ≥ 0.97 → Task 6 Step 2; seam regression → Task 6 + Task 5.

**Deviations from the design doc (intentional, noted):**
- The design's two reject signals ("median-vs-seed" + "forward/backward inconsistency") are realized as **median-vs-seed agreement** plus **per-row flow smoothing** (`_smooth1d`) instead of a second backward flow. Rationale: a second DIS pass doubles cost, and a raw per-row variance check would reject the *legitimate* perspective gradient we want to keep. Smoothing suppresses noise while preserving the gradient; the seed-agreement gate already rejects garbage flow (uniform car sides). Faithful to intent (reject untrustworthy flow), lower cost.
- The design's headline perspective regression is implemented as a **brightness-flicker seam-energy** test rather than a sheared-train SSIM test. Rationale: flicker deterministically produces wide-strip seams that the cross-dissolve provably removes, independent of how much per-row gradient DIS recovers in CI; the per-row perspective handling is still exercised by `synthesize` (Task 2) and the real 4K clip (Task 7 Step 6). This avoids a flaky DIS-gradient-recovery threshold.

**Placeholder scan:** none — every code/test step contains full content.

**Type/name consistency:** `BandInterpolator.analyze` → `FlowBand(fx_col, dx_refined)`; `synthesize(fb, frame_k, frame_k1, slit_x, carry_in, w, dy_cum, direction, interp_flag)` used identically in Task 2 tests, Task 3 `Compositor.add`. `horizontal_seam_energy` defined in Task 5, used in Task 6. `flicker_frames` defined in Task 5, imported in Task 6.
