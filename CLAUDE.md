# train-inspector — notes for Claude

CLI that turns a static-camera video of a passing train into a wide slit-scan panorama.

## Build / run / test
- `uv sync` (add `--extra debug` for matplotlib plots). Python ≥ 3.11.
- Run: `uv run train-inspector INPUT -o OUT.png`
- Test: `uv run pytest` (synthetic FFV1 fixtures generated in-repo; ~45 s).

## Architecture (two-pass, see spec/SPEC.md)
`decode → motion → segment → composite → encode`, orchestrated by `pipeline.run`.
- **Pass 1** estimates per-frame dx (LK + dominant-moving-cluster), smooths in the
  velocity domain (px/s) against real timestamps.
- **Pass 2** re-decodes the chosen segment at full res, **timestamp-aligned** (never
  by frame index — OpenCV seek is keyframe-based), stitches sub-pixel strips.

## Invariants that are easy to break
- **Strip geometry is normative** (`composite.py`, spec §10.3/§10.4): source region is
  `[slit − carry_in, slit − carry_in + w)` for BOTH directions; floor+carry accumulator.
  Direction is handled ONLY by assembly order in `mosaic()` (LTR reversed, RTL natural).
  A half-pixel error here = visible seams.
- **`--min-speed` is segment-detection only.** Inside a segment every smoothed dx feeds
  the accumulator, however small. Don't reintroduce per-frame gating (review B3).
- **Low-confidence frames are bridged by linear interpolation** in `motion.smooth`, not a
  rolling median (clustered failures over textured backgrounds defeat a median). This is
  what makes entry/exit over a busy background work (review B1).
- The **consecutive-substitution abort is segment-scoped** in `pipeline.run`, not in
  `smooth` — a long low-confidence run before the train arrives is normal.

## Testing approach
Correctness is checked against the **analytically known source texture**, not stored
golden images. Use **integer train speed** for strict SSIM ≥ 0.98 (no sub-pixel render
anywhere); a separate non-integer-speed test exercises the carry accumulator with a
relaxed SSIM (documented resampling softening). The fixture texture is band-limited on
purpose (models a real train surface; survives strip resampling).
