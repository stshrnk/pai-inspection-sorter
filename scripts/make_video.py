#!/usr/bin/env python3
"""Encode viewport/camera PNG frames without capturing the desktop.

  python scripts/make_video.py outputs/demo/frames outputs/demo.mp4 --fps 20

Frame names must be frame_000000.png, frame_000001.png, ... . Only the frame
images are encoded: terminals, usernames and desktop UI are never captured.
The input frames themselves must also contain no personal information.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("frames", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=float, default=20)
    parser.add_argument("--start-number", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.fps <= 0 or args.start_number < 0:
        parser.error("fps must be positive and start-number nonnegative")
    if not (args.frames / f"frame_{args.start_number:06d}.png").is_file():
        parser.error("First input frame does not exist")
    if args.output.exists() and not args.overwrite:
        parser.error("Output exists; choose a new path or pass --overwrite")
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        parser.error("ffmpeg is unavailable; retain the PNG sequence for later encoding")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, "-hide_banner", "-loglevel", "warning", "-y" if args.overwrite else "-n",
        "-framerate", str(args.fps), "-start_number", str(args.start_number),
        "-i", str(args.frames / "frame_%06d.png"),
        "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(args.output),
    ]
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
