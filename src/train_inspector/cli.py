"""CLI per spec §7. Exit codes: 0 success, 1 invalid input/arguments,
2 no qualifying train segment, 3 processing error."""

from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import click

from . import __version__, decode, encode, motion, pipeline

log = logging.getLogger("train_inspector")

_TIME_RE = re.compile(r"^(?:(\d+):)?(\d+(?:\.\d+)?)$")


def _parse_time(_ctx, _param, value: str | None) -> float | None:
    """Accept seconds ('90', '12.5') or mm:ss ('1:30') → milliseconds."""
    if value is None:
        return None
    m = _TIME_RE.match(value.strip())
    if not m:
        raise click.BadParameter(f"invalid time '{value}' (use seconds or mm:ss)")
    minutes = int(m.group(1) or 0)
    seconds = float(m.group(2))
    return (minutes * 60 + seconds) * 1000.0


def _parse_roi(_ctx, _param, value: str | None) -> decode.Roi | None:
    if value is None:
        return None
    try:
        x, y, w, h = (int(v) for v in value.split(","))
    except ValueError:
        raise click.BadParameter("expected X,Y,W,H integers")
    if w <= 0 or h <= 0:
        raise click.BadParameter("ROI width and height must be positive")
    return decode.Roi(x, y, w, h)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__)
@click.argument("input", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.option("-o", "--output", type=click.Path(dir_okay=False, path_type=Path),
              default=None, help="Output image path [default: <input-stem>_panorama.png]")
@click.option("--column", type=click.FloatRange(0.0, 1.0), default=0.5, show_default=True,
              help="Sampling column position in ROI, 0..1")
@click.option("--roi", callback=_parse_roi, default=None, metavar="X,Y,W,H",
              help="Region of interest in pixels")
@click.option("--direction", type=click.Choice(["auto", "ltr", "rtl"]), default="auto",
              show_default=True, help="Travel direction")
@click.option("--min-speed", type=click.FloatRange(min=0.01), default=1.0, show_default=True,
              help="Segment threshold, px/frame at nominal fps")
@click.option("--smooth", "smooth_s", type=click.FloatRange(min=0.0), default=0.15,
              show_default=True, help="Velocity smoothing window, seconds")
@click.option("--start", callback=_parse_time, default=None, help="Start time (sec or mm:ss)")
@click.option("--end", callback=_parse_time, default=None, help="End time (sec or mm:ss)")
@click.option("--scale", type=click.FloatRange(min=0.01, max=4.0), default=1.0,
              show_default=True, help="Downscale factor for output")
@click.option("--max-width", type=click.IntRange(min=16), default=100_000, show_default=True,
              help="Output width safety cap")
@click.option("--quality", "jpeg_quality", type=click.IntRange(1, 100), default=95,
              show_default=True, help="JPEG quality if output is .jpg")
@click.option("--fast", is_flag=True, help="Faster (bilinear, no cross-dissolve blending); wide single-frame strips")
@click.option("--debug-dir", type=click.Path(file_okay=False, path_type=Path), default=None,
              help="Write diagnostic artifacts")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
@click.option("--quiet", is_flag=True, help="Errors only")
def main(input: Path, output: Path | None, column: float, roi, direction: str,
         min_speed: float, smooth_s: float, start: float | None, end: float | None,
         scale: float, max_width: int, jpeg_quality: int, fast: bool,
         debug_dir: Path | None, verbose: bool, quiet: bool) -> None:
    """Produce a wide panoramic photo of a passing train from INPUT video."""
    level = logging.DEBUG if verbose else logging.ERROR if quiet else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s", stream=sys.stderr)

    if start is not None and end is not None and end <= start:
        click.echo("error: --end must be after --start", err=True)
        sys.exit(1)

    opts = pipeline.Options(
        input=input,
        output=output or input.with_name(input.stem + "_panorama.png"),
        column=column, roi=roi, direction=direction, min_speed=min_speed,
        smooth_s=smooth_s, start_ms=start, end_ms=end, scale=scale,
        max_width=max_width, jpeg_quality=jpeg_quality, fast=fast, debug_dir=debug_dir,
    )

    try:
        result = pipeline.run(opts)
    except (decode.InputError, encode.OutputError) as e:
        click.echo(f"error: {e}", err=True)
        sys.exit(1)
    except pipeline.NoMotionError as e:
        click.echo(f"error: {e}", err=True)
        sys.exit(2)
    except motion.ProcessingError as e:
        click.echo(f"error: {e}", err=True)
        sys.exit(3)

    if not quiet:
        click.echo(
            f"{result.output} ({result.width}x{result.height}, {result.direction}, "
            f"{result.n_frames} frames, mean speed {result.mean_dx:.1f} px/frame)"
        )


if __name__ == "__main__":
    main()
