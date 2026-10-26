#!/usr/bin/env python3
"""Encode camera PNG frames with optional measured AI-route captions.

  python3 scripts/make_video.py recordings/demo/frames recordings/demo.mp4 --fps 20
  python3 scripts/make_video.py recordings/demo/frames recordings/demo_captioned.mp4 \
    --metrics artifacts/evaluation20/metrics.json --trials 3 --gamma 1.6
  python3 scripts/make_video.py --submission

Frame names are frame_000000.png, frame_000001.png, ... . Input PNGs remain
unchanged. Gamma is a display-only video adjustment; classifier inputs retain
raw pixels. Captions use only validated trial integers, route enums and finite
scores from measured logs. No arbitrary notes or paths are drawn on screen.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys

LABELS = {'good', 'defect', 'review'}
SAFE_CAPTION = re.compile(r'[A-Za-z0-9 .()=|/+\-]+\Z')
STATIC_CAPTIONS = {'GOOD green / DEFECT red / REVIEW yellow', 'AI input raw | Raw frames retained'}
ROUTE_CAPTION = re.compile(r'Trial [1-9][0-9]* \| AI (GOOD|DEFECT|REVIEW) \| p\(defect\)=[01]\.[0-9]{6}\Z')
GAMMA_CAPTION = re.compile(r'Display gamma [0-9]+(?:\.[0-9]+)? only \| AI input raw \| Raw frames retained\Z')


def load_timeline(path: Path, trials: int | None = None):
    """Return measured episode boundaries without trusting free-form strings."""
    payload = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict) or not isinstance(payload.get('cycles'), list):
        raise ValueError('metrics must contain a cycles list')
    cycles = payload['cycles']
    if trials is not None:
        if trials > len(cycles):
            raise ValueError('requested trial count exceeds recorded cycles')
        cycles = cycles[:trials]
    if not cycles:
        raise ValueError('metrics contain no recorded cycles')
    timeline, cumulative = [], 0.0
    for row in cycles:
        if not isinstance(row, dict):
            raise ValueError('each cycle must be an object')
        duration, trial = row.get('simulation_seconds'), row.get('trial')
        decision = row.get('decision')
        if not isinstance(duration, (int, float)) or isinstance(duration, bool) or not math.isfinite(duration) or duration <= 0:
            raise ValueError('each cycle needs positive finite simulation_seconds')
        if not isinstance(trial, int) or isinstance(trial, bool) or trial < 1:
            raise ValueError('each cycle needs a positive integer trial')
        if not isinstance(decision, dict) or decision.get('method') != 'cnn':
            raise ValueError('AI captions require measured CNN decisions')
        label, score = decision.get('label'), decision.get('defect_score')
        if not isinstance(label, str) or label not in LABELS or row.get('destination', label) != label:
            raise ValueError('decision label and destination must be a matching route enum')
        if not isinstance(score, (int, float)) or isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError('each AI caption requires a finite defect score in [0, 1]')
        timeline.append({'trial': trial, 'label': label, 'score': float(score),
                         'start': cumulative, 'end': cumulative + float(duration)})
        cumulative += float(duration)
        if not math.isfinite(cumulative):
            raise ValueError('invalid cumulative episode duration')
    return timeline, cumulative


def drawtext(text: str, *, y: str, size: int, enable: str | None = None):
    """Text comes only from closed templates, never metrics notes or paths."""
    if not SAFE_CAPTION.fullmatch(text) or not (
        text in STATIC_CAPTIONS or ROUTE_CAPTION.fullmatch(text) or GAMMA_CAPTION.fullmatch(text)
    ):
        raise ValueError('unexpected caption character')
    # expansion=none also disables ffmpeg's %{...} text expansion. Commas in
    # enable are protected by filter-level quotes, not by a shell invocation.
    result = (f"drawtext=font=Sans:text='{text}':expansion=none:x=18:y={y}:fontsize={size}:"
              'fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=7')
    if enable is not None:
        result += f":enable='{enable}'"
    return result


def build_filter(gamma: float, timeline=None, start_seconds: float = 0.0):
    filters = ['pad=ceil(iw/2)*2:ceil(ih/2)*2']
    if gamma != 1.0:
        filters.append(f'eq=gamma={gamma:.6f}')
    if timeline is not None:
        filters.append(drawtext('GOOD green / DEFECT red / REVIEW yellow', y='h-th-54', size=18))
        for episode in timeline:
            start = max(0.0, episode['start'] + 0.7 - start_seconds)
            end = episode['end'] - start_seconds
            if end <= start:
                continue
            # Only validated numeric values and closed label enums enter text.
            caption = (f'Trial {episode["trial"]} | AI {episode["label"].upper()} | '
                       f'p(defect)={episode["score"]:.6f}')
            filters.append(drawtext(caption, y='18', size=25,
                                    enable=f'between(t,{start:.6f},{end:.6f})'))
    if gamma != 1.0 or timeline is not None:
        note = (f'Display gamma {gamma:g} only | AI input raw | Raw frames retained'
                if gamma != 1.0 else 'AI input raw | Raw frames retained')
        filters.append(drawtext(note, y='h-th-16', size=14))
    return ','.join(filters)


def check_filters(ffmpeg, *, gamma: float, captions: bool):
    required = {'pad'}
    if gamma != 1.0:
        required.add('eq')
    if gamma != 1.0 or captions:
        required.add('drawtext')
    result = subprocess.run([ffmpeg, '-hide_banner', '-filters'], capture_output=True, text=True, check=True)
    available = {match.group(1) for match in re.finditer(r'^\s*[TSC.]{3}\s+(\w+)\s', result.stdout, re.MULTILINE)}
    if not required <= available:
        raise ValueError('ffmpeg lacks required filters: ' + ', '.join(sorted(required - available)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('frames', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--fps', type=float, default=20)
    parser.add_argument('--start-number', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--gamma', type=float, default=1.0, help='Display-only gamma; 1.0 preserves raw display levels')
    parser.add_argument('--metrics', type=Path, help='Measured metrics JSON for AI-route captions')
    parser.add_argument('--trials', type=int, help='Encode only the first N measured episodes; requires --metrics')
    args = parser.parse_args()
    if not math.isfinite(args.fps) or args.fps <= 0 or args.start_number < 0:
        parser.error('fps must be finite and positive; start-number must be nonnegative')
    if not math.isfinite(args.gamma) or not 0.1 <= args.gamma <= 10:
        parser.error('gamma must be finite and between 0.1 and 10')
    if args.trials is not None and (args.trials < 1 or args.metrics is None):
        parser.error('a positive --trials requires --metrics')
    first_frame = args.frames / f'frame_{args.start_number:06d}.png'
    if not first_frame.is_file():
        parser.error('First input frame does not exist')
    if args.output.resolve() == first_frame.resolve() or (
        args.output.resolve().parent == args.frames.resolve()
        and re.fullmatch(r'frame_\d+\.png', args.output.resolve().name)
    ):
        parser.error('Output must not replace a raw input frame')
    if args.output.exists() and not args.overwrite:
        parser.error('Output exists; choose a new path or pass --overwrite')
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        parser.error('ffmpeg is unavailable; retain the PNG sequence for later encoding')
    timeline, duration = None, None
    try:
        if args.metrics is not None:
            timeline, duration = load_timeline(args.metrics, args.trials)
        check_filters(ffmpeg, gamma=args.gamma, captions=timeline is not None)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.error(str(error))
    start_seconds = args.start_number / args.fps
    if args.trials is not None and duration <= start_seconds:
        parser.error('start-number is beyond the selected measured episodes')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg, '-hide_banner', '-loglevel', 'warning', '-y' if args.overwrite else '-n',
        '-framerate', str(args.fps), '-start_number', str(args.start_number),
        '-i', str(args.frames / 'frame_%06d.png'),
        '-vf', build_filter(args.gamma, timeline, start_seconds),
    ]
    if args.trials is not None:
        command.extend(['-t', f'{duration - start_seconds:.6f}'])
    command.extend(['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-movflags', '+faststart',
                    '-map_metadata', '-1', str(args.output)])
    subprocess.run(command, check=True)


if __name__ == '__main__':
    if '--submission' in sys.argv[1:]:
        from submission_video import main as build_submission
        build_submission([arg for arg in sys.argv[1:] if arg != '--submission'])
    else:
        main()
