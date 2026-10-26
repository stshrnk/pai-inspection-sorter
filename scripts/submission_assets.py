"""Shared, measured figures for the submission PDF and video; no simulation."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "review/submission"
INK = "#183348"
MUTED = "#526b7c"
TEAL = "#087f8c"
BLUE = "#377ac0"
BG = "#f4f7fa"
ROUTES = {"good": ("良品", "#12845e"), "defect": ("不良品", "#c94e4b"),
          "review": ("要確認", "#ad7311")}


def font(size, bold=False):
    spec = "Noto Sans CJK JP" + (":style=Bold" if bold else "")
    path = subprocess.check_output(["fc-match", "-f", "%{file}", spec], text=True)
    return ImageFont.truetype(path, size, index=0)


def text(draw, xy, value, size=32, fill=INK, bold=False):
    draw.text(xy, str(value), font=font(size, bold), fill=fill)


def load_evidence():
    classification = json.loads((ROOT / "docs/classification_metrics.json").read_text())
    transport = json.loads((ROOT / "docs/transport_metrics.json").read_text())
    raw = json.loads((ROOT / "artifacts/evaluation20/metrics.json").read_text())
    if transport != raw:
        raise ValueError("Public and raw transport records differ; reconcile before publishing.")
    for row in transport["cycles"]:
        d = row["decision"]
        p = d["defect_score"]
        expected = "review" if max(p, 1 - p) < d["review_threshold"] else (
            "defect" if p >= .5 else "good")
        if d["label"] != expected or row["destination"] != expected:
            raise ValueError("Recorded decision and threshold disagree.")
    return classification, transport


def accuracy(item):
    matrix = item["confusion"]["matrix"]
    correct = matrix[0][0] + matrix[1][1]
    if abs(correct / item["samples"] - item["accuracy"]) > 1e-10:
        raise ValueError("Confusion matrix and accuracy disagree.")
    return f'{item["accuracy"] * 100:.1f}%（{correct}/{item["samples"]}）'


def camera_display(path):
    image = Image.open(path).convert("RGB")
    # Display adjustment only; inspection inputs below are never brightened.
    lut = [round(255 * (value / 255) ** (1 / 1.6)) for value in range(256)]
    return image.point(lut * 3)


def cnn_pixels(row):
    # Same BILINEAR resize as src/sorter/vision/data.py before /255 normalization.
    path = ROOT / "artifacts/evaluation20" / Path(row["image"]).name
    return Image.open(path).convert("RGB").resize((64, 64), Image.Resampling.BILINEAR)


def build_assets():
    classification, transport = load_evidence()
    OUT.mkdir(parents=True, exist_ok=True)
    im = Image.new("RGB", (1680, 260), "white")
    d = ImageDraw.Draw(im)
    stages = [("カメラ", "RGB画像"), ("CNN", "汚れを判定"), ("判定", "良品・不良品・要確認"),
              ("搬送", "つかむ・運ぶ・離す"), ("確認・ログ", "全体配置と静止")]
    for i, (title, detail) in enumerate(stages):
        x = 8 + i * 336
        d.rounded_rectangle((x, 24, x + 292, 210), radius=18, fill=BG,
                            outline=TEAL, width=3)
        text(d, (x + 20, 48), title, 40, bold=True)
        text(d, (x + 16, 126), detail, 24)
        if i < 4:
            text(d, (x + 296, 80), "→", 34, TEAL)
    im.save(OUT / "pipeline.png")

    im = Image.new("RGB", (1500, 560), "white")
    d = ImageDraw.Draw(im)
    text(d, (28, 15), "独立テストの正解率  |  要確認による保留を適用する前", 32, bold=True)
    text(d, (940, 72), "■ CNN", 26, TEAL, True)
    text(d, (1120, 72), "■ 輝度方式", 26, BLUE, True)
    for i, (key, title) in enumerate([(None, "全体"), ("nominal", "通常照明"), ("dark", "暗照明")]):
        y = 140 + i * 130
        text(d, (30, y + 13), title, 32, bold=True)
        for j, (method, color) in enumerate([("cnn", TEAL), ("brightness", BLUE)]):
            item = classification["test"][method]
            if key:
                item = item["lighting"][key]
            yy = y + j * 48
            d.rectangle((235, yy, 1055, yy + 32), fill="#edf1f5")
            d.rectangle((235, yy, 235 + 820 * item["accuracy"], yy + 32), fill=color)
            text(d, (1080, yy - 5), accuracy(item), 27, color, True)
    for x, label in [(235, "0%"), (625, "50%"), (1010, "100%")]:
        text(d, (x, 532), label, 20, MUTED)
    im.save(OUT / "classification.png")

    im = Image.new("RGB", (1500, 600), "white")
    d = ImageDraw.Draw(im)
    for i, row in enumerate(transport["cycles"][:3]):
        x = i * 500 + 25
        label, color = ROUTES[row["destination"]]
        text(d, (x, 10), f"試行 {row['trial']}  →  {label}", 34, color, True)
        im.paste(cnn_pixels(row).resize((320, 320), Image.Resampling.NEAREST), (x + 65, 75))
        text(d, (x, 415), f"不良スコア  {row['decision']['defect_score']:.6f}", 28)
        text(d, (x, 463), "AI入力：64 × 64 RGB（拡大）", 25)
        text(d, (x, 505), "画素の明るさ・色は無補正", 25, MUTED)
    im.save(OUT / "inspection_inputs.png")
    camera_display(ROOT / "recordings/demo/frames/frame_000360.png").save(OUT / "cell.png")
    return classification, transport


if __name__ == "__main__":
    build_assets()
