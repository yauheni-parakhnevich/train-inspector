# Spec Review: train-inspector SPEC.md (Draft v0.1)

**Reviewer:** Lead developer (desk review, implementability focus)
**Date:** 2026-06-10

## Verdict: Implementable with changes

The core algorithm (strip width = per-frame displacement, sub-pixel carry accumulator) is
sound and well understood — this is a correct software line-scan emulation, and §10.2/§10.3
show the author understands the math. The architecture (two-pass: low-res motion pass,
full-res composite pass) is the right shape, the module breakdown is sensible, and the tech
stack (OpenCV/NumPy/click) is appropriate. However, the spec under-specifies the three places
where this class of tool actually fails in practice: (1) motion estimation while the train is
entering/leaving the frame, where the static background dominates both phase correlation and
median-of-flows; (2) frame alignment between pass 1 and pass 2 given OpenCV's unreliable
seeking and the prevalence of VFR phone footage; and (3) the interaction between the
min-speed gate and the sub-pixel accumulator, which as written silently discards geometry.
None of these requires a redesign, but each will cause significant rework if discovered
during implementation rather than fixed in the spec now. Fix the blockers, then greenlight.

---

## Blockers

### B1 — Motion estimation fails at train entry/exit; spec's "median" aggregation makes it worse (§5 FR-2, §10.1)

This is the single biggest correctness risk and the spec doesn't mention it. While the train
is entering or leaving the frame, it occupies a minority of the ROI. Consequences:

- **Phase correlation** returns the *dominant* peak. With >50% static background, the
  dominant peak is at (0, 0). dx collapses to ~0 exactly when the nose/tail of the train is
  passing the slit → the front and rear of the train come out clipped, smeared, or stretched.
- **Median of LK track displacements** has the identical failure: when fewer than half the
  tracked corners are on the train, the median is the background's ~0. The spec's "robust
  aggregation (median) to reject outliers" (FR-2) treats the background as inlier and the
  train as outlier at the ends of the pass.

The acceptance criteria (§11.1: width ±1%) cannot be met on real footage without solving
this; on synthetic fixtures it may pass by luck if the fixture "train" fills the ROI.

**What to change:** Add an explicit subsection to §10.1 specifying the strategy, e.g.:
(a) take the *largest non-zero motion mode* — cluster LK flow vectors (or inspect the
second peak of the phase-correlation surface) and select the consistent moving cluster,
falling back to 0 only if no cluster exceeds a track-count threshold; and/or
(b) estimate motion in a narrow band around the sampling column only (the train covers the
slit for the entire useful duration, by definition); and/or
(c) once the segment is established, prefer temporal continuity: reject per-frame estimates
that deviate wildly from the smoothed velocity and substitute the prediction.
Also resolve the "phase correlation AND/OR LK" ambiguity — pick a primary method and define
the fallback condition; "AND/OR" is not implementable.

### B2 — Pass-2 frame alignment relies on OpenCV seeking, which is not frame-accurate (§8 Motion Estimator, §9)

The two-pass design requires that frame *i* in pass 2 is exactly the frame that produced
`dx[i]` in pass 1. Known OpenCV/FFmpeg pitfalls the spec must address:

- `CAP_PROP_POS_FRAMES` / `CAP_PROP_POS_MSEC` seeking is keyframe-based and inaccurate for
  many codecs (H.265, open-GOP H.264, fragmented MP4); landing one frame off shifts the
  entire dx-to-frame mapping and produces systematic duplicated/missing strips.
- `CAP_PROP_FRAME_COUNT` and `CAP_PROP_FPS` are estimates and are wrong for VFR files —
  which is what most phone footage is (§2 explicitly targets phone footage).
- Even `set()` followed by sequential `read()` can behave differently across OpenCV
  versions/backends.

**What to change:** Specify in §8 that pass 2 must align by **timestamp/PTS**
(`CAP_PROP_POS_MSEC` read back per frame), not by frame index, and that the default
implementation is **seek to before the segment, then read sequentially and match frames to
pass-1 timestamps** (decode-and-discard from t=0 as the always-correct fallback; the cost is
bounded since decode is fast relative to the clip). Alternatively allow PyAV as a backend
for PTS-exact access. Add an acceptance test: pass-1/pass-2 frame correspondence verified on
an HEVC and a VFR fixture.

### B3 — `--min-speed` gating inside the segment contradicts the carry accumulator and drops geometry (§5 FR-2 last bullet, FR-7, §10.3)

FR-2: "Frames with |dx| below a threshold ... contribute no strip." With default
`--min-speed 1.0`, a train crawling at 0.8 px/frame contributes *nothing* — but per §10.2
that content genuinely passed the slit and is now permanently missing from the panorama.
Worse, when speed fluctuates around the threshold mid-pass (braking train, noisy estimates),
strips are dropped intermittently → cumulative gaps that violate NFR-2 and §11.3 with no
error reported. The carry accumulator (§10.3) already handles arbitrarily small dx correctly;
the gate fights it.

**What to change:** Redefine the threshold's role: `--min-speed` is used **only by the
Segment Detector** (FR-3) to find the start/end of the pass, with hysteresis (e.g., enter
segment when smoothed |dx| > threshold for N frames, exit when < threshold/2 for N frames).
**Within** the detected segment, every frame's smoothed dx feeds the accumulator, no matter
how small (including dx temporarily ~0 — train briefly stopped contributes ~0-width strips,
which is correct and gap-free). Update FR-2, FR-7, and §10.3 accordingly.

---

## Major concerns

### M1 — `round` vs `floor` inconsistency in the core formula (FR-4 vs §10.3)

FR-4 says strip width = `round(|dx|)` "carrying the sub-pixel remainder forward"; §10.3 says
`w = floor(|dx| + carry)`. These are different algorithms and only §10.3 is drift-free as
stated (round + carry double-counts). Make FR-4 reference §10.3 verbatim. This is the heart
of the tool; the spec must not contain two versions of it.

### M2 — Exact strip geometry is unspecified; this is where seams come from (FR-4, §10.3)

"Extract a strip centered on the sampling column" plus "resampled at the sub-pixel offset"
leaves the implementer to guess: which side of the slit, what the sub-pixel offset is
relative to, and how direction flips the indexing. Off-by-half-pixel here = visible seam
every frame. Specify the formula, e.g.: for left-to-right travel, frame *i* contributes the
region `[x_slit − (|dx_i| + carry_in), x_slit − carry_out)` (continuous coordinates),
resampled to `w` integer columns via the chosen interpolation; mirror for RTL; state the
append order explicitly for both directions. One paragraph of exact math saves a week of
seam debugging and makes §11.1/11.3 deterministic.

### M3 — No fallback for low-texture / periodic train surfaces (§10.1)

Plain tank cars, uniform-color boxcars, or long blank wagon sides give LK nothing to track
and give phase correlation a weak/flat peak. Periodic structure (passenger-car windows,
container corrugation) aliases phase correlation to the wrong multiple of the period. §10.1
step 3 only covers dy-based rejection. Add: a per-frame **confidence score** (phase-corr
peak response / inlier track count), and the policy "low confidence → hold the smoothed
velocity (constant-velocity prediction), flag the frame in diagnostics." Without this,
mid-train garbage strips are guaranteed on real freight footage.

### M4 — VFR and dropped frames break smoothing as specified (FR-2, §8)

Strip-width = inter-frame displacement is actually *robust* to VFR and frame drops (a
doubled inter-frame interval legitimately yields doubled dx and a doubled strip — correct).
But two spec'd mechanisms break it: (a) smoothing "rolling median + low-pass" over the
*frame-indexed* dx series will treat the legitimate 2× dx after a drop as an outlier and
clamp it → missing geometry; (b) `--min-speed` in px/frame is meaningless under VFR.
**Change:** smooth **velocity in px/second** using per-frame timestamps, then convert back
to per-frame dx via each frame's actual Δt; define `--min-speed` against a nominal fps or in
px/s.

### M5 — dy jitter: goal and non-goal contradict, and behavior is undefined (§2, §3, §10.1)

§2 promises "handheld-but-steady" support; §3 excludes "vertical motion compensation
*beyond minor jitter stabilization*" — implying minor stabilization IS in scope — but no FR
defines it. §10.1 only uses dy to *reject* frames. If strips are pasted with no vertical
alignment, even ±2 px of jitter produces visibly wavy roof/underframe lines across the
panorama. **Decide:** either (v1-simple) declare tripod-only, drop "handheld" from §2; or
add an FR: per-strip integer (or sub-pixel) vertical shift by the estimated dy before
appending. The latter is ~10 lines of code and I'd recommend it.

### M6 — Pass-1 half-resolution dx may not be accurate enough for the SSIM ≥ 0.98 target (§8, §11.1)

dx estimated at 0.5× scale carries ~2× the sub-pixel error after upscaling, and pass 2 uses
it unrefined ("hands frames + final per-frame dx to the compositor"). Aggregate drift is
absorbed by the carry, but *per-frame* error of ±0.3–0.5 px shows up as local stretch and
will pressure both the ±1% width and SSIM 0.98 criteria. **Change:** specify optional pass-2
refinement (one phase-correlation step at full res in a narrow band around the slit, seeded
by the pass-1 value) or state that pass 1 runs at full resolution within the ROI when the
ROI is small. At minimum, flag this as a tuning risk and keep the acceptance threshold
revisable.

### M7 — Default `--max-width 100000` exceeds JPEG's hard 65,535 px limit (FR-5, §7, §13 Q3)

JPEG cannot encode images wider than 65,535 px — `cv2.imwrite` will fail or silently
misbehave. With the spec'd defaults, a long train + `.jpg` output is a guaranteed crash path.
**Change:** validate format-specific dimension limits before compositing (JPEG 65,535; PNG
effectively unbounded; TIFF strip/size limits) and fail early with a message suggesting PNG
or `--scale`.

### M8 — Multiple trains / multiple motion segments: policy undefined (FR-3, §8 Segment Detector)

Two trains in one clip, a train that stops for 30 s mid-pass (segment splits in two), or an
opposing train passing on a far track during the pass — the spec says "the contiguous run"
as if there is exactly one. **Define:** v1 selects the **longest** qualifying segment, logs
all detected segments at `-v`, and documents `--start/--end` as the escape hatch. Also
define behavior for direction reversal within a segment (shunting): treat as segment
boundary.

### M9 — Source-format realities unaddressed: rotation metadata, 10-bit/HDR HEVC (FR-1, §9)

Phone footage (the stated target) is frequently: rotated via container metadata (OpenCV
auto-applies it only in newer versions and behavior differs across builds — pin and test),
HEVC 10-bit, and HDR (iPhone HLG/Dolby Vision — OpenCV's 8-bit BGR conversion yields washed
colors). **Change:** FR-1 should state v1 normalizes to 8-bit BGR, honors rotation metadata
(explicitly set `CAP_PROP_ORIENTATION_AUTO`), and *warns* on detected HDR transfer
characteristics rather than silently producing grey output. Audio streams: state they are
ignored (trivial, but say it).

### M10 — Golden-image SSIM tests will be brittle across OpenCV versions (§11.1, NFR-5)

Stored golden PNGs compared at SSIM ≥ 0.98 will break when OpenCV changes interpolation
kernels or the codec used to encode fixtures changes. Since fixtures are synthetic, compare
against the **analytically known texture** (the source strip pattern) instead of a stored
artifact, and generate fixtures with lossless encoding (FFV1/PNG sequence) in-repo, not
checked-in H.264. Also: §11.3's "edge-correlation metric" is undefined — define it or the
criterion isn't testable.

---

## Minor / nitpicks

- **N1** (§5 FR-4 vs §7): `--no-reorder` is referenced in FR-4 but missing from the CLI
  table in §7. Add it (or drop it — see Q4 below).
- **N2** (§5 FR-5 vs §7): default output name is `<input-stem>_panorama.png` in FR-5 but
  `<input>_panorama.png` in §7. Pick one (stem).
- **N3** (§5 FR-7): `--column` is "fraction of frame width" while `--roi` "restricts motion
  estimation and output". If the ROI crops horizontally, column 0.5-of-frame can fall
  outside the ROI. Define `--column` as a fraction of **ROI width** and validate.
- **N4** (§7, §11.4): exit code when `--start/--end` selects an empty/invalid range —
  presumably 1, but say so. Also define exit code when motion exists but no segment passes
  the duration filter (2, presumably).
- **N5** (§8 module layout vs §9): `cli.py` comment says "argparse or click" while §9
  commits to click. Remove the hedge.
- **N6** (§5 FR-6): the displacement *plot* implies matplotlib — an undeclared dependency.
  List it (as an optional `[debug]` extra, ideally), or emit CSV only by default.
- **N7** (§5 FR-7): `--min-speed` is in px/frame, but pass 1 runs at reduced resolution —
  state that the threshold is in **full-resolution** pixels and pass-1 values are rescaled
  before comparison (and see M4 re: VFR units).
- **N8** (§10.4): add **rolling shutter** to documented distortions: on CMOS phone sensors
  it produces a constant shear of the train in each frame, which the slit-scan inherits as a
  uniform skew. Not fixable in v1, but users will ask; one sentence saves a bug report.
- **N9** (§10.4): add **foreground occlusions** (catenary masts, signal poles, fences at the
  slit) — a static object at the sampling column is re-stamped into every strip and smears
  across the entire panorama. Document the mitigation: move `--column` / set `--roi`. A
  debug-dir overlay showing static high-gradient content at the chosen column would help
  users pick a clean slit.
- **N10** (§10.4): motion blur — at high speed (e.g., 30+ px/frame) wide strips are
  horizontally blurred by the source exposure; document that output sharpness is bounded by
  shutter speed, and consider logging a warning when mean dx is large.
- **N11** (§8): color handling — pipeline is OpenCV BGR end-to-end and `cv2.imwrite`
  expects BGR, so it's consistent, but say so explicitly to stop a future contributor from
  "fixing" a nonexistent channel swap (especially if Pillow ever enters `encode.py`).
- **N12** (§11.6): pin the codec for the performance criterion (1080p60 **H.264**); HEVC
  decode is materially slower and makes the criterion ambiguous.

---

## Questions for the author

- **Q1:** For very long recordings (e.g., 20-min clip with a 1-min pass), pass 1 decodes
  the entire file. Is that acceptable for v1, or do you want a coarse pre-scan (strided
  frames) to bracket the active region before the dense pass? Affects NFR-1 expectations.
- **Q2:** §11.3's "edge-correlation metric across strip boundaries" — do you have a
  concrete formula in mind, or is the implementer expected to invent one? (See M10.)
- **Q3:** When dy rejection (§10.1 step 3) interpolates dx from neighbors — over how many
  consecutive bad frames is interpolation allowed before the run aborts with exit code 3?
- **Q4:** What is the actual use case for `--no-reorder` (FR-4)? Outputting a mirrored
  panorama seems like a flag nobody will use; can it be cut?
- **Q5:** Is double-track footage (a second train moving the opposite direction in the
  background during the pass) in scope for "robust aggregation"? It biases the median and is
  common at stations. If in scope, B1's dominant-cluster approach handles it; confirm.

---

## Answers to the spec's Open Questions (§13)

- **OQ1 (auto-ROI in v1?):** Agree with the plan — `--roi` flag only in v1 — **with one
  amendment**: the entry/exit fix for B1 effectively requires knowing *where* motion happens
  horizontally, and a crude vertical motion-activity band (row-wise temporal variance) is
  ~20 lines and nearly free in pass 1. Ship it as a *diagnostic* in `--debug-dir` (suggested
  ROI overlay) in v1, promote to automatic behavior in v1.1 once validated on real footage.
- **OQ2 (background fill / `--crop-to-motion`):** Agree — keep the smeared background in
  v1. It's honest output, visually self-explanatory, and auto-cropping adds a failure mode
  (cropping off pantographs/wheels). Defer to v1.1 alongside auto-ROI since they share the
  motion-band computation.
- **OQ3 (JPEG of multi-10k-px images):** PNG default is right, but this question hides a
  bug, not a preference: JPEG **cannot exceed 65,535 px** in either dimension (see M7), so
  for typical full-train panoramas JPEG output must be either rejected with a clear error or
  auto-scaled. Add the validation in v1; defer tiled TIFF/BigTIFF until someone asks.
