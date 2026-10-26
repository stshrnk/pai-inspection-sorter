"""Compose a Japanese submission video from preserved frames and measured logs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw

from submission_assets import (ROOT, OUT, BG, INK, MUTED, TEAL, BLUE, ROUTES,
                               accuracy, build_assets, camera_display, cnn_pixels, text)

WORK = ROOT / 'artifacts/submission_build'
TITLE = 'PAI最終課題_142245'
SIZE = (1920, 1080)


def canvas(title, section):
    im = Image.new('RGB', SIZE, BG)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 1920, 10), fill=TEAL)
    text(d, (60, 30), section, 26, TEAL, True)
    text(d, (60, 85), title, 60, bold=True)
    text(d, (60, 1020), TITLE + '  |  シミュレーションによる検証', 25, MUTED)
    return im, d


def cards(c, t):
    result = []
    im, d = canvas('AI外観検査・仕分けセル', 'Physical AI 応用1講座  /  最終課題')
    text(d, (65, 280), '画像で判定し、', 70, bold=True)
    text(d, (65, 380), 'ロボットが運び分ける。', 70, bold=True)
    text(d, (65, 550), '製造業の外観検査 → 仕分け', 43, TEAL, True)
    text(d, (65, 640), '良品・不良品・要確認の3経路', 38)
    text(d, (65, 720), 'UR10e + Robotiq 2F-85 / Isaac Sim', 30, MUTED)
    im.paste(camera_display(ROOT / 'recordings/demo/frames/frame_000360.png').resize((800, 600)), (1070, 250))
    result.append(('01_title', im, 7))

    im, d = canvas('撮影から、手を離した後の確認まで', '目的と処理フロー')
    text(d, (65, 230), '固定位置へ供給した部品の「上面の模擬汚れ」を検査', 46)
    flow = Image.open(OUT / 'pipeline.png').resize((1800, 279))
    im.paste(flow, (60, 345))
    text(d, (65, 680), 'AIへ渡すのは画像だけ。正解ラベルや行先は入力しない。', 42, TEAL, True)
    text(d, (65, 770), '確信度が基準未満なら「要確認」として判断を保留。', 42)
    text(d, (65, 870), 'これから3試行を等速で表示  /  試行ごとに初期状態へ戻す', 34, MUTED)
    result.append(('02_flow', im, 13))

    im, d = canvas('CNNと輝度方式を、同じテスト画像で比較', '評価結果  1 / 2  ―  画像分類')
    text(d, (65, 205), '正解率：要確認として除外する前の二値判定', 35, MUTED)
    chart = Image.open(OUT / 'classification.png').resize((1650, 616))
    im.paste(chart, (135, 280))
    text(d, (65, 920), '学習・検証・テストは条件ごとに分離。テスト後の再調整なし。', 34, TEAL, True)
    result.append(('06_classification', im, 14))

    im, d = canvas('工夫と限界を分けて評価する', '評価結果  2 / 2  ―  搬送・今後')
    n = len(t['cycles'])
    success = sum(r['transfer_success'] is True for r in t['cycles'])
    text(d, (65, 215), f'搬送 {success}/{n} 件成功', 72, TEAL, True)
    text(d, (65, 325), '検証用データから選んだ固定3条件。毎回初期状態へリセット。', 38)
    text(d, (65, 390), '部品全体のトレイ内配置と、開放・退避後の静止を確認。', 38)
    text(d, (65, 490), '工夫：画像だけで判定／要確認で保留／条件単位のデータ分離', 38, TEAL, True)
    dark = c['test']['cnn']['lighting']['dark']
    wrong = dark['confusion']['matrix'][0][1]
    passed = dark['routing_confusion']['matrix'][0][1]
    text(d, (65, 595), f'限界：暗照明の良品{wrong}枚を誤判定。うち{passed}枚は保留できず。', 38, '#ac3f3b', True)
    text(d, (65, 675), '固定位置・固定姿勢での供給、シミュレーションのみ。', 38)
    text(d, (65, 780), '今後：照明変化を含む再学習と、実カメラ・実機での検証。', 38)
    text(d, (65, 915), '流用：NVIDIA Isaac Sim UR10eサンプル / Apache-2.0', 27, MUTED)
    text(d, (65, 960), '詳細：THIRD_PARTY_NOTICES.md（ロボットアセットの条件は別）', 25, MUTED)
    result.append(('07_limits', im, 14))
    return result


def trial_panel(row):
    im = Image.new('RGBA', SIZE, BG)
    d = ImageDraw.Draw(im)
    label, color = ROUTES[row['destination']]
    d.rectangle((0, 0, 1920, 10), fill=color)
    text(d, (30, 24), f'試行 {row["trial"]}  |  画像判定 → {label}トレイ', 52, color, True)
    text(d, (34, 1010), '等速（シミュレーション時間）  |  試行ごとに初期状態へ戻す', 29)
    # The transparent window preserves the entire camera view without cropping.
    d.rectangle((24, 124, 1175, 987), fill=(0, 0, 0, 0))
    x = 1230
    text(d, (x, 125), 'AIに入力された検査画像', 38, bold=True)
    text(d, (x, 185), '64 × 64 RGBを拡大・無補正', 29, MUTED)
    im.paste(cnn_pixels(row).resize((384, 384), Image.Resampling.NEAREST), (1340, 245))
    text(d, (x, 648), '当該試行の判定ログ', 25, MUTED)
    dec = row['decision']
    text(d, (x, 695), f'不良スコア p = {dec["defect_score"]:.6f}', 35, bold=True)
    text(d, (x, 750), f'確信度 = {dec["confidence"]:.6f}', 34)
    if row['destination'] == 'review':
        text(d, (x, 815), f'{dec["confidence"]:.6f} < 基準 {dec["review_threshold"]:.6f}', 33, color, True)
        text(d, (x, 865), '検証用データで決めた基準未満', 30, color, True)
        text(d, (x, 912), '不良品と確定せず、要確認へ', 30, color, True)
    else:
        text(d, (x, 815), f'確信度 ≥ 基準 {dec["review_threshold"]:.6f}', 32, color, True)
        text(d, (x, 872), f'行先：{label}トレイ', 42, color, True)
    text(d, (x, 980), '不良スコアは正答確率の保証ではない', 25, MUTED)
    # Caption is an explicit display note, not a change to the inspection pixels.
    text(d, (45, 940), '全景のみ表示の明るさを補正', 25, 'white')
    return im


def reset_boundaries(frames, rows, fps):
    """Locate visible resets near logged episode boundaries, within 0.6 seconds.

    The camera render lags the physics step slightly. Keep the cut on the
    actual scene reset instead of attaching the next log to the previous image.
    """
    cuts, elapsed = [0], 0.
    for row in rows[:-1]:
        elapsed += row['simulation_seconds']
        center = round(elapsed * fps)
        differences = []
        for i in range(max(1, center - 12), min(len(frames), center + 13)):
            a = np.asarray(Image.open(frames[i - 1]).convert('RGB'), dtype=np.float32)
            b = np.asarray(Image.open(frames[i]).convert('RGB'), dtype=np.float32)
            differences.append((float(np.abs(a - b).mean()), i))
        score, boundary = max(differences)
        if score < 1.0:
            raise ValueError('No clear scene reset near the measured boundary.')
        cuts.append(boundary)
    return cuts + [len(frames)]


def encode(args):
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y', *args,
                    '-an', '-c:v', 'libx264', '-threads', '4', '-preset', 'fast', '-crf', '19',
                    '-pix_fmt', 'yuv420p', '-map_metadata', '-1', '-movflags', '+faststart',
                    str(WORK / 'current.mp4')], check=True)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args(argv)
    output = ROOT / 'recordings' / f'{TITLE}.mp4'
    if output.exists() and not args.overwrite:
        p.error('Output exists; use --overwrite to rebuild the submission video.')
    c, t = build_assets()
    WORK.mkdir(parents=True, exist_ok=True)
    fps = t['config']['record_fps']
    frames = sorted((ROOT / 'recordings/demo/frames').glob('frame_*.png'))
    if not frames or any(f.name != f'frame_{i:06d}.png' for i, f in enumerate(frames)):
        raise ValueError('The source frame sequence is missing or incomplete.')
    rows = t['cycles'][:t['config']['record_trials']]
    if len(rows) != 3 or [r['destination'] for r in rows] != ['good', 'defect', 'review']:
        raise ValueError('Expected the three measured demonstration routes.')
    cuts = reset_boundaries(frames, rows, fps)
    segments, timeline, elapsed = [], [], 0.
    def save_segment(name, duration, details):
        nonlocal elapsed
        path = WORK / f'{name}.mp4'
        (WORK / 'current.mp4').replace(path)
        segments.append(path)
        timeline.append({'segment': name, 'start': elapsed, 'end': elapsed + duration, **details})
        elapsed += duration
        print(f'Encoded {name}: {duration:.2f} s', flush=True)
    slides = cards(c, t)
    for name, im, duration in slides[:2]:
        path = WORK / f'{name}.png'
        im.save(path)
        encode(['-loop', '1', '-framerate', str(fps), '-i', str(path), '-t', str(duration)])
        save_segment(name, duration, {'type': 'explanation'})
    for i, row in enumerate(rows):
        panel = WORK / f'panel_{i+1}.png'
        trial_panel(row).save(panel)
        count = cuts[i + 1] - cuts[i]
        encode(['-framerate', str(fps), '-start_number', str(cuts[i]),
                '-i', str(frames[0].parent / 'frame_%06d.png'), '-i', str(panel),
                '-filter_complex', '[0:v]eq=gamma=1.6,scale=1152:864,pad=1920:1080:24:124:white[cam];[cam][1:v]overlay=0:0',
                '-frames:v', str(count)])
        save_segment(f'{i+3:02d}_trial_{i+1}', count / fps,
                     {'type': 'trial', 'trial': row['trial'], 'source_start_frame': cuts[i],
                      'source_end_frame_exclusive': cuts[i+1], 'playback_speed': 1.0,
                      'inspection_image': row['image'], 'decision': row['decision']})
    for name, im, duration in slides[2:]:
        path = WORK / f'{name}.png'
        im.save(path)
        encode(['-loop', '1', '-framerate', str(fps), '-i', str(path), '-t', str(duration)])
        save_segment(name, duration, {'type': 'explanation'})
    concat = WORK / 'concat.txt'
    concat.write_text(''.join(f"file '{path.name}'\n" for path in segments))
    subprocess.run(['ffmpeg', '-hide_banner', '-loglevel', 'warning', '-y', '-f', 'concat',
                    '-safe', '1', '-i', str(concat), '-c', 'copy', '-map_metadata', '-1',
                    '-metadata', f'title={TITLE}', '-movflags', '+faststart', str(output)], check=True)
    (WORK / 'timeline.json').write_text(json.dumps({'title': TITLE, 'duration': elapsed,
        'fps': fps, 'audio': 'none; Japanese visual explanations', 'segments': timeline}, ensure_ascii=False, indent=2))
    print(f'Completed: {TITLE}.mp4 ({elapsed:.2f} s)')


if __name__ == '__main__':
    main()
