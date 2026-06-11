# train-inspector

Turn a video of a train passing a **static camera** into a single **wide panoramic photo** of the whole train.

It works like a software line-scan (slit-scan) camera: for every frame it estimates how far the train moved, takes a vertical strip of pixels that wide at a fixed column, and stitches the strips together. The moving train is reconstructed at full sharpness across the whole panorama; the static background smears away.

## Install

```sh
uv sync                 # runtime + dev deps
uv sync --extra debug   # also installs matplotlib for diagnostic plots
```

## Usage

```sh
uv run train-inspector input.mp4 -o train.png
```

Zero-config: travel direction, the train's entry/exit, and output ordering are all detected from the footage. The panorama always reads left-to-right regardless of which way the train travelled.

### Common options

```
--column FLOAT          Sampling column in the ROI, 0..1 [default: 0.5]
--roi X,Y,W,H           Restrict estimation + output to a region (crop platforms, sky, poles)
--direction auto|ltr|rtl  Override direction detection [default: auto]
--start TIME --end TIME   Trim to a time range (seconds or mm:ss)
--scale FLOAT           Downscale the final image
--fast                  Faster, slightly softer (bilinear, no refinement)
--debug-dir DIR         Dump motion CSV/plot + ROI/slit overlays
-v / --quiet            Verbosity
```

Run `uv run train-inspector --help` for the full list.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | Success |
| 1 | Invalid input/arguments (unreadable file, bad range, output exceeds a format/size limit) |
| 2 | No train motion detected |
| 3 | Processing error (segment interior un-estimable, decode/alignment failure) |

## How it works

Two passes (see `spec/SPEC.md` for the full specification):

1. **Motion pass** — reduced-resolution sweep estimates per-frame horizontal displacement (pyramidal Lucas–Kanade, dominant-moving-cluster selection so a busy background can't out-vote the train), smoothed in the velocity domain against real timestamps (variable-frame-rate safe).
2. **Composite pass** — re-decodes the detected segment at full resolution, **aligned by timestamp** (never frame index — OpenCV seeking isn't frame-accurate), and stitches sub-pixel strips with a floor+carry accumulator so there's no cumulative drift over thousands of frames.

Known, documented distortions (wheels, perspective, rolling shutter, motion blur, foreground occluders at the slit) are described in `spec/SPEC.md` §10.5.

## Tests

```sh
uv run pytest
```

Fixtures are synthetic clips generated in-repo (lossless FFV1); correctness is checked against the analytically known source texture (no stored golden images).

## Project layout

```
src/train_inspector/   decode · motion · segment · composite · encode · diagnostics · pipeline · cli
tests/                 unit + end-to-end acceptance tests, synthetic fixtures
spec/SPEC.md           specification (v0.2)
feedback/              lead-developer spec review
```
