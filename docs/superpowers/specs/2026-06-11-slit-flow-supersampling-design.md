# Flow-Based Sub-Time Supersampling at the Slit — Design

**Status:** Superseded during implementation — see Revision note.
**Date:** 2026-06-11
**Author:** Eugene
**Relates to:** `spec/SPEC.md` §10.3 (carry accumulator), §10.4 (strip geometry), §8 (two-pass), §11 (acceptance)

> ## ⚠️ Revision (2026-06-11, post-implementation)
> The **per-row dense optical flow** at the heart of this design did not survive
> contact with real footage and was **replaced by a global cross-dissolve**.
>
> **What changed.** Sections 5–7 below specify `cv2.DISOpticalFlow` in a band
> around the slit, sampling per-row flow `fx_col(y)` to motion-compensate the
> cross-dissolve (for perspective correction). Empirically DIS is accurate only
> to ~20 px/frame and returns near-zero / wrong-sign motion at the ~50 px/frame
> the target clip exhibits — it *injected noise* and made seams slightly worse
> on real footage.
>
> **Shipped instead.** The cross-dissolve mechanism (Section 5) is kept — it is
> what removes the seam — but the per-row shift is the **known pass-1 dx,
> uniform across rows** (no optical flow, no band, no reliability gate). This is
> simpler, cheaper (one extra warp vs a DIS pass), robust at any speed, and
> measured **−26% vertical-line energy** on the real 50 px/frame clip (vs a
> *regression* with per-row flow). Perspective correction is dropped (it was
> unachievable at speed anyway). The synthesizer lives in `composite._crossfade`;
> there is no `flow.py`.
>
> Read Sections 5 (geometry/τ-ramp) and 10 (testing) as still-accurate; treat
> the DIS/`analyze`/band/fallback machinery in Sections 5–7, 11 as **the
> rejected approach**, retained as a record. `spec/SPEC.md` §8 reflects the
> shipped behavior.

## 1. Problem

Panorama output shows regularly-spaced **visible vertical lines** (seams). Root cause is
not a bug in the carry math — strips tile correctly — but **coarse temporal sampling**:

- Real footage moves fast relative to frame rate. Measured on `data/0-405-…mkv`:
  ~24 fps, `dx_smooth` mean **~50 px/frame** → every strip is a **~50 px-wide snapshot
  from a single frame**. `freight_crossing.mp4`: ~24 px/frame.
- At each strip boundary the panorama jumps a full frame interval. Three effects then
  surface as a vertical line:
  1. any sub-pixel `dx` estimate error,
  2. **perspective** (angled/curved track → top and bottom rows have *different* true dx,
     so one strip-wide global dx cannot tile both),
  3. exposure / motion-blur differences between adjacent frames.

Narrower strips attack all three. We get there by **synthesizing finer time samples**
(frame interpolation), not by hiding the boundary (blending), because the goal is maximum
smoothness/sharpness.

## 2. Goals / Non-Goals

**Goals**
- Default output free of the periodic vertical seams on fast passes.
- Fix perspective-induced seams (per-row tiling).
- Preserve the normative carry geometry (§10.3/§10.4): `Σ W = Σ|dx̂|`, no drift.
- Stay within the project's dependency ethos: OpenCV + NumPy only, OpenCV pinned, CPU.
- Degrade to **exactly today's quality** where interpolation is unreliable — no new
  failure mode.

**Non-Goals (v1 of this feature)**
- DNN frame interpolation (RIFE/torch). Deferred; reconsider only if OpenCV flow proves
  insufficient on real footage.
- Vertical (`F_y`) flow correction. The existing global `dy_cum` jitter translation is
  kept; per-pixel vertical warp is a future pass (risk of wobble).
- New user-facing tuning flags. `--fast` is reused to select the old path.

## 3. Chosen approach (Approach 1 of 3)

**Flow-based sub-time supersampling at the slit.** Considered and rejected:
- *Approach 2 — per-row flow warp + overlap feather (no sub-time):* cheaper, but does not
  fix the coarse-time / exposure jump → less smooth.
- *Approach 3 — synthesize whole intermediate frames, reuse compositor unchanged:* same
  quality as Approach 1 but computes flow + warp over the entire frame when only the slit
  band is used → much slower, wasteful.

Approach 1 delivers Approach 3's smoothness localized like Approach 2, and keeps the carry
geometry intact.

## 4. Architecture

Pass 1 is **unchanged**. The change is confined to pass 2 — *how* each frame pair's `W`
output columns are sampled — plus one new module.

```
decode → motion (pass 1, unchanged) → segment → composite(+flow, pass 2) → encode
```

- New module **`flow.py`** — owns the dense-flow band computation and column synthesis.
  Pure over its inputs, independently testable.
- **`composite.py`** — keeps ownership of the carry accumulator and strip geometry (the
  CLAUDE.md invariant). Gains a flow-aware path; falls back to today's `warpAffine` wide
  strip when flow is absent (`--fast`) or rejected.
- **`pipeline.py`** pass-2 loop — feeds frame *pairs* instead of single frames; the band
  flow's median also **replaces** the separate `_refine_dx` phase-correlation step.

## 5. Core algorithm (normative)

For frame pair `(k → k+1)` with refined signed displacement `dx̂` and `carry_in` (the
sub-pixel remainder before this pair, per §10.3):

```
D         = |dx̂| + carry_in
W         = floor(D)          # emitted columns — UNCHANGED from §10.3
carry_out = D − W             # UNCHANGED
```

The `W` columns are **not** one snapshot from frame `k`. Column `j = 0..W-1` crossed the
slit at fractional sub-time

```
τ_j = clamp( ( (1 − carry_in) + j − 0.5 ) / |dx̂| , 0, 1 )      ∈ [0, 1]
```

The upper bound holds because `W ≤ carry_in + |dx̂|` ⇒ the largest
`((1−carry_in)+(W-1)) ≤ |dx̂|`. The `−0.5` center-samples each column; for `carry_in > 0.5`
it can push `τ_0` slightly negative (that column straddles the previous interval), so the
result is **clamped to `[0,1]`** — at the ends the blend collapses to a pure frame sample,
which is correct.

**Synthesis (per-row, vectorized).** Dense flow `F` maps frame `k → k+1`. A pixel at `x`
in frame `k` is at `x + τ·F(x)` at time `τ`. The slit content at sub-time `τ_j`, row `y`,
comes from source `x0(y) = slit − τ_j · F_x(y)`:

```
column_j(y) = (1 − τ_j) · I_k ( x0,            y + dy_cum )
            +      τ_j  · I_{k+1}( x0 + F_x(y), y + dy_cum )
```

`F_x` varies by row, so the top and bottom of a strip tile at their own true dx →
**perspective seams are fixed for free**. All `W` columns build in one `cv2.remap` per
source frame plus an alpha blend. Sampling uses Lanczos4 (bilinear under `--fast`, which
also disables this path).

`dy_cum` (vertical jitter alignment, FR-4) stays the global translation it is today; `F_y`
is intentionally unused in v1.

### 5.1 Direction
Sign of `dx̂` / `F_x` is consistent with the locked travel direction (pass 1). Assembly
order in `mosaic()` is unchanged (LTR reversed, RTL natural).

## 6. Flow computation & fallback

**Flow.** One dense flow per pair via `cv2.DISOpticalFlow` (`PRESET_MEDIUM`), computed on
a **grayscale band** around the slit — never the whole frame. Band half-width is adaptive:

```
band_half = max(REFINE_BAND_HALF_W, ceil(max|dx̂|) + 16)
```

so the ~50 px motion always fits with margin. The DIS handle is created once and reused.

**Fallback (wide-strip path is the automatic safety net).** A pair reverts to today's
single `warpAffine` wide strip when **any** of:

- `|median(F_x in band) − dx̂| > max(2 px, 0.25 · |dx̂|)` — flow disagrees with the motion
  estimate (uniform boxcar side, periodic aliasing);
- forward/backward flow inconsistency in the band exceeds a px threshold (occlusion, heavy
  motion blur);
- `|dx̂| < 1` (sub-pixel motion; flow is noise at this scale) **while** `W ≥ 1` — carry can
  emit a column even when the per-frame motion is sub-pixel.

`W == 0` (crawling/stopped train) emits nothing and needs no flow — handled before the
checks above. Hard regions therefore degrade to **exactly current quality**, never worse. The per-pair
reject count is logged and dumped to `--debug-dir`.

**Consolidation.** The band flow's `median(F_x)` *becomes* the pass-2 refinement,
replacing the separate `_refine_dx` phase-correlation step — one band computation serves
both refine and synthesis. If flow is rejected, the pass-1 `dx̂` is kept (same as
`_refine_dx`'s current reject path).

## 7. Interfaces

- **`flow.py`** — `BandInterpolator` holding the DIS handle:
  ```
  BandInterpolator.strip(frame_k, frame_k1, slit_x, carry_in, dx_hat, W, dy_cum)
      -> np.ndarray (H × W × 3)   # synthesized strip
       | None                      # flow rejected → caller falls back
  ```
  Pure over its inputs; no pipeline state.
- **`composite.py`** — `Compositor.add(frame_k, frame_next, dx_hat, dy_cum, interp=None)`.
  Owns the carry → computes `W`, `carry_in`; asks `interp` for the block; on `None`
  (rejected) or `interp is None` (`--fast`) runs today's exact `warpAffine` wide-strip
  path. Geometry/carry stay here per the CLAUDE.md invariant.
- **`pipeline.py`** pass-2 loop — maintain a `prev_frame`; emit each pair's strip from
  `(prev_frame, cur)` with `cur`'s `dx̂`. The first matched frame only seeds `prev` (its
  `dx ≈ 0` contributes nothing today). The off-by-one nets to zero — `Σ W` unchanged.

## 8. CLI / configuration

- **No new user flags.** Default = flow supersampling ON.
- **`--fast`** (already "bilinear, no refinement") is extended to also select the
  wide-strip path (no flow) — restores today's behavior and speed.
- Band width and DIS preset are internal constants.
- **`--debug-dir`** gains a per-pair line: `τ`-range, `|F_x − dx̂|` deviation, fallback
  flag — making rejects visible.

## 9. Performance

One DIS-band flow per pair (≈ H × ~200 px). ~120 pairs → sub-second added on the 1080p
clips; heavier on 4K. **NFR-1 (≥2× real-time) is re-scoped to `--fast`**; the default mode
trades speed for smoothness (the explicit product call). Default-mode runtime on the 4K
`freight_crossing.mp4` clip will be measured and documented in `spec/SPEC.md` during
implementation.

## 10. Testing (analytic-texture, no golden images — per CLAUDE.md / NFR-5)

1. **Strict geometry guarantee (kept):** existing integer-speed SSIM ≥ 0.98 test runs
   under `--fast` (pure-geometry wide-strip path) — unchanged, still proves the carry math.
2. **Flow path, relaxed SSIM:** same band-limited fixture, default flow path → SSIM ≥ 0.97
   (interpolation softening; mirrors the existing integer/non-integer split).
3. **Perspective fixture (the feature's regression test):** synthesize a fixture where top
   and bottom rows move at *different* dx (simulated perspective shear). Measure the §11.3
   boundary-correlation metric: flow path ≥ 0.95, wide-strip path measurably worse. This
   proves the seams are gone.
4. **Fallback:** fixture with a uniform/blank band region → flow rejected → falls back to
   wide strip, no crash, width ±1%, output still analytic-correct.
5. **High-speed fixture (~50 px/frame, matching the real clip):** both paths width ±1%;
   flow path passes boundary-correlation, wide path fails it.

All existing tests stay green; the strict SSIM test is pinned to `--fast`.

## 11. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Flow artifacts on uniform/blurry car sides | Per-pair fallback to wide strip on rejection; never worse than today. |
| Interpolation softening reduces sharpness | Lanczos4 sampling; strict SSIM guarantee retained on `--fast`; relaxed (≥0.97) on flow path. |
| 4K runtime regression | Flow localized to the slit band; `--fast` restores ≥2× real-time; runtime documented. |
| Off-by-one / drift from pair-feeding | `Σ W` invariant unchanged; covered by width ±1% acceptance on every fixture. |
| Double-counting vertical motion | `F_y` deliberately unused in v1; only global `dy_cum` applied. |

## 12. Spec changes to fold in (on implementation)

- `spec/SPEC.md` §8 (pass 2): describe flow-band supersampling; note `_refine_dx`
  subsumed by the band flow median.
- §10.4: add the sub-time synthesis as the default strip path; the existing single-warp
  region becomes the `--fast` / fallback path.
- §11: add acceptance tests 3–5 above; pin existing strict SSIM (§11.1) to `--fast`.
- FR-7 / §7: document `--fast` now also disables flow supersampling.
- NFR-1: re-scope the ≥2× target to `--fast`; document default-mode runtime.
```
