# Train Inspector — Specification

**Status:** Draft v0.2
**Date:** 2026-06-10

> v0.2 incorporates the lead-developer review (`feedback/spec-review-lead-dev.md`).
> All blockers (B1–B3), majors (M1–M10), and minors (N1–N12) are addressed; reviewer
> recommendations on the open questions are adopted. Changelog in §14.

## 1. Overview

`train-inspector` is a command-line application that takes a video recording of a train passing by a **static camera** and produces a single **wide panoramic photo** of the entire train as output.

The core technique is *motion-compensated strip stitching* (a software slit-scan / line-scan camera emulation): for each video frame, the train's horizontal displacement relative to the previous frame is estimated, and a vertical strip of pixels whose width equals that displacement is extracted and appended to the output mosaic. The static background blurs away while the moving train is reconstructed at full sharpness across the entire panorama.

## 2. Goals

- Produce a sharp, geometrically consistent wide image of a moving train from a fixed-camera video.
- Work with consumer footage (phone/camera on tripod or handheld-but-steady), common containers and codecs (mp4/mov/mkv, H.264/H.265), including variable-frame-rate (VFR) phone recordings.
- Handle trains moving in either direction (left-to-right or right-to-left), with automatic direction detection.
- Handle non-constant train speed (acceleration, braking, brief stops) via per-frame velocity estimation.
- Compensate minor vertical camera jitter (±few px) by per-strip vertical alignment (see FR-4); "handheld-but-steady" is supported on this basis.
- Be a single, scriptable CLI tool with sensible defaults: `train-inspector input.mp4 -o train.png`.

## 3. Non-Goals (v1)

- Real-time / streaming processing (offline batch only).
- Moving-camera footage (drone fly-bys, pans). Camera must be static (tripod or steadily held).
- Vertical motion compensation beyond the per-strip jitter alignment defined in FR-4 (no rotation, no scale, no full stabilization).
- Object detection/classification of train cars, OCR of wagon numbers (possible future extension).
- Rolling-shutter, perspective, or wheel-distortion correction (documented in §10.5).
- GUI.

## 4. Definitions

| Term | Meaning |
|---|---|
| Slit / sampling column | Fixed vertical column (x-position) in the ROI from which strips are taken. |
| Strip | Vertical slice of pixels extracted from a frame, width derived from per-frame displacement (§10.3). |
| Displacement (dx) | Horizontal motion of the train between consecutive frames, in full-resolution pixels (sub-pixel precision). |
| Velocity (vx) | Train motion in px/second: `dx / Δt` using real frame timestamps. The smoothed quantity (§10.2). |
| Mosaic | The accumulated output panorama. |
| Segment | Contiguous run of frames during which the train is passing (FR-3). |

## 5. Functional Requirements

### FR-1: Input
- Accept a path to a video file. Any format readable by the decoding backend (FFmpeg via OpenCV) is supported.
- Optional `--start` / `--end` timestamps (seconds or `mm:ss`) to trim to the relevant segment.
- Validate input: file exists, is decodable, has at least 2 frames. Fail with a clear error message and exit code 1 otherwise. An empty or inverted `--start/--end` range is exit code 1.
- **Normalization:** all frames are converted to 8-bit BGR for the entire pipeline (OpenCV-native; `cv2.imwrite` consumes BGR directly — no channel swap exists or is needed anywhere).
- **Rotation metadata** (phone portrait/landscape flags) is honored: `CAP_PROP_ORIENTATION_AUTO` is explicitly set to 1, and the pinned OpenCV version's behavior is covered by a test fixture.
- **HDR sources** (HLG / PQ / Dolby Vision transfer characteristics, 10-bit HEVC): v1 processes them through the 8-bit conversion but emits a prominent warning that colors may be washed out; proper tone mapping is a future extension.
- Audio streams are ignored.
- Per-frame **timestamps** (`CAP_PROP_POS_MSEC` read back after each `read()`) are recorded for every decoded frame; all downstream logic is timestamp-based, never frame-count/FPS-based, so VFR files are handled correctly.

### FR-2: Motion estimation
- Estimate per-frame horizontal displacement of the train with sub-pixel accuracy. Full algorithm in §10.1; binding points:
- **Primary method:** sparse pyramidal Lucas–Kanade optical flow on Shi–Tomasi corners within the ROI. **Fallback** (when fewer than `MIN_TRACKS` (default 8) corners survive): phase correlation in a band around the sampling column. "Primary" means LK is attempted every frame; the fallback condition is explicit, not heuristic.
- **Dominant-moving-cluster aggregation, not plain median:** flow vectors are clustered by dx; the selected estimate is the median of the largest *moving* cluster consistent with the segment direction. The static background cluster (dx ≈ 0) is never selected while a segment is active. Rationale and details in §10.1 — this is the entry/exit correctness fix (review B1).
- **Confidence score** per frame (inlier track count, cluster tightness, or phase-corr peak response). Low-confidence frames (blank wagon sides, periodic structure, motion blur) do not use their raw estimate: the smoothed velocity prediction is held instead, and the frame is flagged in diagnostics (review M3).
- Automatically detect motion direction from the sign of the aggregated displacement over the segment.
- **Smoothing operates on velocity in px/second** against real timestamps (rolling median + light low-pass), then converts back to per-frame dx via each frame's actual Δt. A doubled inter-frame interval (dropped frame, VFR) legitimately doubles dx and is not treated as an outlier (review M4).
- Frames with large |dy| or wild dx deviation from the smoothed velocity are rejected and replaced by the constant-velocity prediction. If more than `max(3, round(0.5 s × nominal fps))` consecutive frames are rejected mid-segment, processing aborts with exit code 3 and a diagnostic message (review Q3).

### FR-3: Train presence detection (auto-trim) and segmentation
- Automatically detect segments: sustained, consistent-direction motion within the ROI.
- **Hysteresis:** a segment starts when smoothed |vx| exceeds `--min-speed` for ≥ N consecutive frames (N default: 0.25 s worth), and ends when it stays below `--min-speed / 2` for ≥ N frames. `--min-speed` is expressed in **full-resolution px/frame at the nominal fps** (internally converted to px/s).
- **The threshold is used ONLY here.** Within a detected segment, *every* frame's smoothed dx feeds the compositor accumulator, however small — a train crawling at 0.3 px/frame or briefly stopped contributes correctly sized (≈0-width) strips with no gaps (review B3).
- **Multiple segments** (two trains, a long mid-pass stop, shunting): v1 composites the **longest** qualifying segment, logs all detected segments at `-v`, and documents `--start/--end` as the manual escape hatch. A direction reversal within motion is a segment boundary (review M8).
- An opposing train on a far track during the pass is handled by the direction-consistent cluster selection in FR-2: vectors moving against the segment direction form a separate cluster and are never selected (review Q5).
- If no segment qualifies (no motion, or motion too brief), exit code 2 with message "no train motion detected".

### FR-4: Strip extraction and compositing
- Strip width and placement follow the **single normative formula in §10.3** (floor + carry accumulator; sub-pixel sampling geometry in §10.4). FR-4 intentionally contains no arithmetic of its own (review M1, M2).
- Strips are resampled at sub-pixel offsets via interpolation (Lanczos4 default, bilinear via `--fast`).
- **Vertical jitter alignment:** each strip is shifted vertically by the negated cumulative smoothed dy estimate (sub-pixel, same interpolator) before appending, compensating minor handheld/tripod-bump jitter. Cumulative |dy| beyond ±2% of frame height triggers a "camera not static" warning (review M5).
- Strips are appended in train-travel order so the output always reads left-to-right regardless of travel direction. (The `--no-reorder` flag from v0.1 is **cut** — no use case; review Q4.)
- Output height = ROI height.

### FR-5: Output
- Write the mosaic to the path given by `-o/--output` (default: `<input-stem>_panorama.png`, e.g. `clip.mp4` → `clip_panorama.png`).
- Supported encoders: PNG (default), JPEG (`--quality`), TIFF.
- **Format dimension limits are validated before compositing begins** (predicted width = Σ smoothed |dx| over the segment): JPEG hard limit 65,535 px; PNG/TIFF effectively unbounded but capped by `--max-width` (default 100,000 px). On violation: fail early (exit 1) with a message suggesting PNG and/or `--scale` (review M7).
- Optional `--scale` factor to downscale the final mosaic.

### FR-6: Diagnostics
- `--debug-dir <dir>`: dump intermediate artifacts —
  - displacement/velocity-vs-time CSV (always) and PNG chart (only if `matplotlib` is installed; it is an optional `[debug]` extra, not a core dependency — review N6);
  - ROI overlay sample frames; detected segment boundaries; per-frame confidence and rejection flags;
  - **suggested-ROI overlay**: row-wise temporal-variance motion band computed in pass 1, drawn on a sample frame — helps users pick `--roi`, and is the validation path for promoting auto-ROI in v1.1 (review OQ1);
  - **slit-quality overlay**: static high-gradient content at the chosen column highlighted, so users spot foreground poles/masts occluding the slit (review N9).
- `-v/--verbose` logging (includes all detected segments, mean speed, predicted output width); `--quiet` for errors only.
- Warning when mean |dx| is large (default > 25 px/frame): output sharpness is bounded by source motion blur (review N10).
- Progress bar on stderr by default (TTY only).

### FR-7: Tuning options
| Flag | Default | Purpose |
|---|---|---|
| `--column <float 0..1>` | `0.5` | Sampling column as fraction of **ROI width** (validated to fall inside the ROI; review N3). |
| `--roi <x,y,w,h>` | full frame | Restrict motion estimation and output to a region (crop out platforms, sky, foreground poles). |
| `--direction <auto\|ltr\|rtl>` | `auto` | Override direction detection. |
| `--min-speed <px/frame>` | `1.0` | Segment detection threshold only (FR-3). Full-resolution pixels at nominal fps. |
| `--smooth <float, seconds>` | `0.15` | Velocity smoothing window in **seconds** (timestamp-based; review M4); `0` disables. |
| `--start/--end <time>` | — | Manual trim. |
| `--fast` | off | Bilinear resampling; disables flow supersampling — single-frame wide strips (§8). |

## 6. Non-Functional Requirements

- **NFR-1 Performance:** Process 1080p/60fps **H.264** footage at ≥ 2× real-time on an Apple Silicon laptop (a 60 s clip in ≤ 30 s) (codec pinned per review N12). Memory bounded: mosaic assembled incrementally in chunks; never holds all decoded frames. For long recordings with a short pass (e.g., 20-min clip, 1-min train), pass 1 may use a strided coarse pre-scan (every K-th frame) to bracket candidate motion regions before dense estimation; this is an internal optimization, not user-visible (review Q1). The ≥ 2× real-time target is measured on the `--fast` path; the default flow-supersampling path trades speed for seam-free output (one banded dense-flow computation per frame pair). Trial (4K freight_crossing.mp4): `--fast` 8s, default flow 11s.
- **NFR-2 Quality:** No visible seams or duplicated/missing geometry on constant-speed segments; bounded distortion (mild horizontal stretch/squash) during acceleration is acceptable and documented (§10.5).
- **NFR-3 Portability:** macOS and Linux; Python ≥ 3.11; OpenCV version **pinned** (rotation/seek behavior varies across builds; review M9/B2).
- **NFR-4 UX:** Zero-config happy path; all defaults derived from the footage itself.
- **NFR-5 Testability:** Core pipeline pure-functional over frame iterators; synthetic fixtures are **generated in-repo with lossless encoding** (FFV1 or PNG sequence), and correctness tests compare against the **analytically known source texture**, not stored golden images (review M10).

## 7. CLI Interface

```
train-inspector INPUT [options]

Arguments:
  INPUT                       Path to input video.

Options:
  -o, --output PATH           Output image path [default: <input-stem>_panorama.png]
  --column FLOAT              Sampling column position in ROI, 0..1 [default: 0.5]
  --roi X,Y,W,H               Region of interest in pixels
  --direction [auto|ltr|rtl]  Travel direction [default: auto]
  --min-speed FLOAT           Segment threshold, px/frame at nominal fps [default: 1.0]
  --smooth FLOAT              Velocity smoothing window, seconds [default: 0.15]
  --start TIME, --end TIME    Process only this time range
  --scale FLOAT               Downscale factor for output [default: 1.0]
  --max-width INT             Output width safety cap [default: 100000]
  --quality INT               JPEG quality if output is .jpg [default: 95]
  --fast                      Faster (bilinear, no flow supersampling); wide single-frame strips
  --debug-dir PATH            Write diagnostic artifacts
  -v, --verbose / --quiet
  --version, --help
```

Exit codes:
| Code | Meaning |
|---|---|
| 0 | Success. |
| 1 | Invalid input/arguments: unreadable file, bad flag values, empty/inverted `--start/--end` range, predicted output exceeds format limit or `--max-width`. |
| 2 | No qualifying train segment: no motion, motion too brief, or all motion below threshold. |
| 3 | Processing error: too many consecutive low-confidence/rejected frames (FR-2), decode failure mid-stream, pass-1/pass-2 alignment failure. |

## 8. Architecture

```
┌────────────┐   ┌──────────────┐   ┌──────────────┐   ┌────────────┐   ┌─────────┐
│  Decoder    │──▶│ Motion       │──▶│ Segment      │──▶│ Strip      │──▶│ Encoder │
│ (frame iter)│   │ Estimator    │   │ Detector     │   │ Compositor │   │ (image) │
└────────────┘   └──────────────┘   └──────────────┘   └────────────┘   └─────────┘
```

- **Decoder** — wraps OpenCV `VideoCapture`; yields frames lazily as `(timestamp_ms, frame)`; applies `--start/--end` and ROI crop; sets `CAP_PROP_ORIENTATION_AUTO`; normalizes to 8-bit BGR.
- **Motion Estimator** — two-pass design:
  - **Pass 1** streams frames at reduced resolution (0.5×, or full resolution if the ROI is already ≤ ~960 px wide), computes the raw displacement/velocity series keyed by **timestamp**. Estimates are rescaled to full-resolution pixels. Smoothing and segment detection run on the full series (smoothing needs future frames — hence two passes).
  - **Pass 2** re-decodes the detected segment at full resolution. **Alignment is by timestamp, not frame index**: seek to a point safely before the segment start (or to t=0 as the always-correct fallback), then read sequentially, matching each decoded frame to the pass-1 series by `CAP_PROP_POS_MSEC` with a tolerance of half the median frame interval. Frame-index-based seeking (`CAP_PROP_POS_FRAMES`) is never used for alignment — OpenCV/FFmpeg seeking is keyframe-based and not frame-accurate on HEVC, open-GOP H.264, and VFR files (review B2). An alignment mismatch (no pass-1 timestamp within tolerance) aborts with exit code 3.
  - **Pass-2 flow supersampling (default):** each strip is built as a per-row,
    motion-compensated cross-dissolve between the two frames bounding the
    interval, using dense optical flow (`cv2.DISOpticalFlow`) in a band around
    the slit (`composite.py` + `flow.py`). Consecutive strips meet at a shared
    frame, eliminating the inter-frame seam; per-row flow tiles each row at its
    own displacement, reducing perspective seams. Strip width still comes from
    the pass-1 smoothed dx via the carry accumulator (§10.3) — flow refines only
    the per-row sampling, never the width. The former phase-correlation
    refinement step (review M6) is removed. Where flow is unreliable (sub-pixel
    motion, or median flow disagreeing with the pass-1 value — e.g. a uniform
    car side), the pair falls back to the single-frame wide strip, matching
    pre-flow quality. `--fast` selects the wide-strip path for the whole pass.
- **Segment Detector** — hysteresis thresholding on smoothed velocity; selects the longest qualifying segment; resolves direction (FR-3).
- **Strip Compositor** — maintains the sub-pixel carry accumulator (§10.3); extracts interpolated strips per the geometry in §10.4; applies vertical jitter shift; appends to a preallocated, chunk-grown mosaic buffer.
- **Encoder** — validates format dimension limits (FR-5); writes the final image; applies `--scale`.

Module layout:

```
train_inspector/
  __init__.py
  cli.py            # click-based argument parsing, exit codes
  decode.py         # frame iterator, timestamps, rotation, normalization
  motion.py         # LK + clustering, phase-corr fallback, confidence, smoothing
  segment.py        # hysteresis segmentation, direction, multi-segment policy
  composite.py      # accumulator, strip geometry, vertical alignment, mosaic
  encode.py         # dimension validation, image output
  diagnostics.py    # CSV/plots/overlays (matplotlib optional)
tests/
  fixtures/         # synthetic clips generated in-repo (lossless)
  test_decode.py    # rotation metadata, VFR timestamps
  test_motion.py
  test_segment.py
  test_composite.py
  test_e2e.py       # analytic-texture comparison, alignment tests
```

## 9. Technology Choices

| Concern | Choice | Rationale |
|---|---|---|
| Language | Python 3.11+ | Fast iteration; rich CV ecosystem; CLI distribution via `pipx`/`uv`. |
| Video decode | OpenCV (FFmpeg backend), version pinned | Universal codec support, simple frame iteration. PyAV is the designated fallback backend if timestamp fidelity of OpenCV proves insufficient on real VFR footage (decision point: during pass-2 alignment testing). |
| Motion estimation | OpenCV pyramidal LK + dx-clustering; phase correlation fallback | Battle-tested, sub-pixel capable, fast; clustering fixes entry/exit (B1). |
| Array ops | NumPy | Mosaic assembly, interpolation. |
| CLI | `click` | Declarative options, good help text. |
| Packaging | `pyproject.toml` + `uv`; optional extra `[debug]` → matplotlib | Modern, reproducible; console-script entry point `train-inspector`. |
| Tests | `pytest` | Synthetic fixture generation in-repo. |

## 10. Algorithm Details

### 10.1 Displacement estimation

1. Convert frame to grayscale; pass 1 downscales ×0.5 (skipped for small ROIs, §8).
2. Detect Shi–Tomasi corners in the ROI; track with pyramidal LK from the previous frame.
3. **Cluster** the surviving track displacements by dx (1-D clustering; e.g., sort + gap-split with a 1.5 px merge radius, or small-k k-means). Classify clusters:
   - *static*: |median dx| < 0.3 px — background;
   - *moving*: everything else, signed.
4. **Select** the largest moving cluster whose direction matches the active segment direction (during initial detection, when no direction is established: the largest moving cluster overall). The static cluster is selected only when no moving cluster has ≥ `MIN_TRACKS` (8) members — this is what makes entry/exit correct: even when the train occupies 10% of the ROI, its tracks form the largest *moving* cluster and win over the larger static one (review B1). Opposing-direction traffic forms its own cluster and is ignored (review Q5).
5. **Fallback:** if total surviving tracks < `MIN_TRACKS` (blank wagon side), run phase correlation on a band of ±`64` px around the sampling column between consecutive frames; accept its peak if the response exceeds a threshold.
6. **Confidence:** `min(1, inliers/16) × cluster_tightness` (or normalized phase-corr response on the fallback path). Confidence < 0.3 → discard the raw estimate, hold the smoothed-velocity prediction, flag frame in diagnostics (review M3). Periodic-structure aliasing (windows, corrugation) is caught by the deviation check below.
7. **Sanity checks:** reject frames with |dy| of the selected cluster > 2 px (scaled) or |dx − predicted| > max(3 px, 30% of predicted) → substitute prediction. More than `max(3, 0.5 s × fps)` consecutive substitutions mid-segment → abort, exit 3.
8. Result: series `(t_i, dx_i, dy_i, confidence_i)` in full-resolution pixels.

### 10.2 Smoothing (velocity domain)

Convert to velocity `vx_i = dx_i / Δt_i` (px/s). Apply rolling median (window `--smooth` seconds) then a light low-pass to `vx`. Convert back: `dx̂_i = v̂x_i × Δt_i`. Dropped frames / VFR thus yield legitimately larger dx̂ for longer intervals instead of being clipped as outliers (review M4). The same treatment applies to dy for the vertical alignment series.

### 10.3 Strip width: floor + carry accumulator (normative)

If the train moves `dx` pixels between frames, the novel content sliding past the sampling column is exactly `dx` pixels wide; taking a strip of that width per frame tiles the train surface with no gaps or overlaps — a line-scan camera whose scan rate adapts to train speed.

Per contributing frame, with `carry ∈ [0, 1)` persisted across frames (initialized 0):

```
total  = |dx̂_i| + carry
w_i    = floor(total)          # integer strip width, may be 0
carry  = total − w_i           # sub-pixel remainder, carried forward
```

This is the **only** normative formula (review M1 — the v0.1 `round()` variant in FR-4 is removed; round+carry double-counts). No cumulative drift: Σ w_i tracks Σ |dx̂_i| within 1 px over the entire pass. `w_i = 0` (crawling/stopped train) is valid and contributes nothing that frame while preserving the remainder.

### 10.4 Strip geometry (normative)

Let `x_slit` be the sampling column (continuous coordinate, ROI space), and let `carry_in` / `carry_out` be the accumulator value before/after frame *i* (§10.3). Frame *i* contributes the continuous-coordinate region:

- **Left-to-right travel** (train moves +x; novel content crosses the slit from the left):
  source region `[x_slit − (w_i + carry_out − carry_in)·s … x_slit]` simplified per the accumulator to `[x_slit − total, x_slit − carry_out)` — i.e. a region of continuous width `total − carry_out = w_i`, resampled to `w_i` integer output columns via Lanczos4 (bilinear under `--fast`). The strip is **prepended** to the mosaic (earlier frames show the train's head, which must end up on the right side it entered from — net effect: panorama reads left-to-right).
- **Right-to-left travel:** mirror image — source region `[x_slit + carry_out, x_slit + total)`, strips **appended**.

Equivalently: in both cases the mosaic is built so the train's front is at one end and reads left-to-right in the final output; the implementation may build in either order and flip once at the end. The vertical jitter shift (FR-4) is applied to the strip during the same resampling pass (single interpolation, not two). This paragraph is normative; an implementation deviating by half a pixel per strip produces visible seams (review M2).

### 10.5 Known distortions (documented behavior)

- **Perspective:** parts of the train nearer/farther than the average depth show mild scale differences. Mitigation: column at frame center; future: per-row displacement (shear correction).
- **Acceleration:** smoothing trades responsiveness vs. noise; residual error appears as slight stretch/squash, not seams.
- **Wheels:** rotating geometry sampled over time produces characteristic slit-scan wheel distortion. Inherent; not corrected.
- **Rolling shutter** (CMOS phone sensors): constant shear of the train in each frame, inherited by the panorama as a uniform skew. Not corrected in v1 (review N8).
- **Foreground occlusions** (catenary masts, poles, fences at the slit): a static object at the sampling column is re-stamped into every strip and smears across the panorama. Mitigation: move `--column` or set `--roi`; the slit-quality debug overlay (FR-6) highlights such content (review N9).
- **Motion blur:** at high speed, strips inherit the source exposure blur; output sharpness is bounded by shutter speed. A warning is logged when mean |dx| > 25 px/frame (review N10).

## 11. Acceptance Criteria

1. **Constant speed:** synthetic fixture (textured rectangle "train", lossless FFV1, generated in-repo) crossing at constant speed → output width equals train length in pixels ±1%, and pixelwise comparison against the **analytically known source texture** yields SSIM ≥ 0.98 (measured on the `--fast` wide-strip path — the pure-geometry guarantee; the default flow path is held to SSIM ≥ 0.97 with no seam-energy regression). No stored golden images (review M10).
2. **Direction:** the same fixture reversed (right-to-left) produces a left-to-right panorama meeting criterion 1.
3. **Acceleration:** fixture with linear 1×→2× speed → no visible seams; **boundary-correlation metric** ≥ 0.95, defined as: for each strip boundary, the Pearson correlation between the last source-texture column of strip *k* and first column of strip *k+1* as they appear in the output, averaged over all boundaries (review Q2/M10).
4. **Slow crawl:** fixture at 0.5 px/frame (below `--min-speed`-naive territory) inside a detected segment → full train reconstructed, width ±1% (regression test for B3).
5. **Entry/exit:** fixture where the train occupies ≤ 20% of the ROI at entry → nose and tail fully present, width ±1% (regression test for B1).
6. **Pass alignment:** HEVC and VFR fixtures → pass-1/pass-2 timestamp correspondence verified exactly (every composited frame matched the pass-1 record within tolerance) (regression test for B2).
7. **Rotation:** portrait-metadata fixture decodes upright (regression test for M9).
8. **No motion:** static video exits 2 with "no train motion detected".
9. **Bad input:** corrupt/missing file exits 1 with a clear message; `.jpg` output with predicted width > 65,535 px exits 1 *before* compositing with a message suggesting PNG or `--scale`.
10. **Performance:** 60 s 1080p60 **H.264** real clip processes in ≤ 30 s on an M-series Mac (NFR-1).
11. **Diagnostics:** `--debug-dir` produces velocity CSV, segment boundaries, ROI/slit overlays; plot PNG present iff matplotlib installed.

## 12. Future Extensions (out of scope for v1)

- Automatic ROI detection (promote the pass-1 motion-band diagnostic of FR-6 to default behavior, after validation on real footage).
- `--crop-to-motion` vertical auto-crop (shares the motion-band computation; deferred together with auto-ROI).
- Per-row displacement estimation → shear/perspective correction.
- HDR tone mapping for 10-bit HLG/PQ sources.
- Automatic wagon segmentation and per-car image export; wagon number OCR.
- Exposure normalization across long passes (lighting changes).
- GPU decode path for 4K footage; tiled/BigTIFF output.

## 13. Resolved Questions

Decisions on v0.1's open questions, per review recommendations:

1. **Auto-ROI:** flag-only in v1; the row-wise motion-band computation ships in v1 as a `--debug-dir` diagnostic (suggested-ROI overlay) and is promoted to automatic in v1.1 once validated (review OQ1).
2. **Background fill:** keep the smeared background; honest, self-explanatory output. `--crop-to-motion` deferred to v1.1 (review OQ2).
3. **Large outputs:** PNG default; JPEG dimension limit enforced up front (FR-5). Tiled TIFF/BigTIFF deferred until demanded (review OQ3).
4. **`--no-reorder`:** cut — no use case for a mirrored panorama (review Q4).
5. **Long-clip pass-1 cost:** acceptable for v1; strided coarse pre-scan permitted as an internal optimization (NFR-1, review Q1).

## 14. Changelog

**v0.2 (2026-06-10)** — folded lead-dev review:
- B1 → FR-2/§10.1: dominant-moving-cluster selection replaces plain median; LK primary, phase-corr fallback with explicit condition; entry/exit acceptance test added (§11.5).
- B2 → §8: pass-2 alignment by timestamp with sequential read; frame-index seeking banned; HEVC/VFR alignment test (§11.6); PyAV designated fallback backend.
- B3 → FR-3/§10.3: `--min-speed` confined to segment detection with hysteresis; all in-segment frames feed the accumulator; slow-crawl test (§11.4).
- M1/M2 → §10.3/§10.4 made normative (floor+carry only; exact strip geometry both directions); FR-4 arithmetic removed.
- M3 → confidence score + hold-velocity policy (FR-2, §10.1).
- M4 → velocity-domain smoothing on timestamps; `--smooth` now in seconds (FR-2, §10.2, FR-7).
- M5 → per-strip vertical jitter alignment (FR-4); goal/non-goal contradiction resolved (§2/§3).
- M6 → pass-2 phase-corr refinement near slit; `--fast` opt-out (§8, FR-7).
- M7 → format dimension validation before compositing (FR-5, §11.9).
- M8 → longest-segment policy, reversal = boundary (FR-3).
- M9 → 8-bit BGR normalization, rotation metadata, HDR warning, audio ignored (FR-1); OpenCV pinned (NFR-3).
- M10/Q2 → analytic-texture comparison, lossless in-repo fixtures, boundary-correlation metric defined (§11, NFR-5).
- N1/Q4 → `--no-reorder` cut. N2 → stem naming unified. N3 → `--column` relative to ROI, validated. N4 → exit-code table (§7). N5 → click committed. N6 → matplotlib optional `[debug]` extra. N7 → threshold units defined full-res (FR-3). N8/N9/N10 → distortions documented + slit-quality overlay + blur warning. N11 → BGR end-to-end stated (FR-1). N12 → perf criterion pinned to H.264.
- Q1/Q3/Q5, OQ1–OQ3 → resolved in §13, FR-2, FR-3.

**v0.3 (2026-06-10)** — hardening from first real-footage trial (4K side-on freight crossing):
- **Relative-spread confidence (FR-2/§10.1):** the per-frame confidence now normalises flow-vector spread by the motion magnitude. Absolute spread grows with speed and perspective, so a fast or angled train (hundreds of agreeing tracks at 45 px/frame) was wrongly scored un-estimable. Fixes spurious aborts on real footage.
- **Global direction locking (FR-2):** travel direction is resolved over the whole clip (confidence-weighted) and estimates that oppose it are nulled and interpolated. A uniform/periodic car side (corrugated boxcar) can yield a confident *backward* cluster (the M3 failure); without locking it fragmented one pass into many segments and composited reversed strips. v1 trade-off: genuine mid-clip shunting is treated as one direction (revises M8's unconditional "reversal = boundary").
- **Interpolation bridging of un-estimable runs (§10.1/§10.2):** low-confidence frames are linearly interpolated from confident neighbours (not a rolling median, which clustered failures drag to the garbage value).
- **Segment-scoped, proportional quality gate (review Q3):** the consecutive-substitution abort moved out of smoothing into the pipeline and now fires only on a genuinely unusable segment (bridge > ~1.5 s, or > 50 % substituted); shorter bridges warn (their strip widths are interpolated → mild local stretch, bounded per NFR-2).
- Validated end to end on real footage (`data/README.md`): a 6045×2160 panorama of the full freight train.
