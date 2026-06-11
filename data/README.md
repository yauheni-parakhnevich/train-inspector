# Sample data

Test clips are not committed (large binaries — see `.gitignore`). Download a known-good one:

```sh
# Static-camera, side-on freight crossing (4K, ~11 s) — CC0, Pexels
curl -sL -A "Mozilla/5.0" -o data/freight_crossing.mp4 \
  "https://www.pexels.com/download/video/3046683/"

uv run train-inspector data/freight_crossing.mp4 -o freight_panorama.png -v
```

This clip exercises the hard real-world cases: a fast freight (motion blur), a uniform
boxcar side that defeats tracking (interpolation-bridged, logged as a warning), and a
slightly curved/angled track (documented perspective distortion, spec §10.5).

## What makes a clip suitable

- **Static camera** (tripod). Footage shot *from* a moving train looking out the window is
  the opposite of what the tool needs and will be rejected or produce garbage.
- **Train crosses the frame side-on** (perpendicular to the camera), not approaching head-on.
- Anything FFmpeg can decode (mp4/mov/mkv, H.264/H.265), variable frame rate is fine.

Source: <https://www.pexels.com/video/a-cargo-train-passing-through-a-railroad-crossing-3046683/>
