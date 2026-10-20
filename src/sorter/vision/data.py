"""Strict condition-group splits and deterministic RGB preprocessing."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_SIZE = 64
LABELS = ("good", "defect")
SPLITS = ("train", "val", "test")


def read_rgb(image) -> Image.Image:
    """Read pixels only. File names and metadata are not model features."""
    if isinstance(image, (str, Path)):
        with Image.open(image) as opened:
            return opened.convert("RGB")
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    pixels = np.asarray(image)
    if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
        raise ValueError("Expected an RGB/RGBA image with shape H x W x 3/4")
    if not np.isfinite(pixels).all():
        raise ValueError("Image contains non-finite pixels")
    if np.issubdtype(pixels.dtype, np.floating):
        pixels = pixels * 255 if pixels.max() <= 1.0 else pixels
    return Image.fromarray(np.clip(pixels[..., :3], 0, 255).astype(np.uint8), "RGB")


def image_array(image) -> np.ndarray:
    rgb = read_rgb(image).resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    return np.asarray(rgb, dtype=np.float32).transpose(2, 0, 1).copy() / 255.0


def brightness_feature(image) -> float:
    """Dark-tail luminance in a fixed central ROI; a simple scalar baseline."""
    rgb = np.asarray(read_rgb(image), dtype=np.float32) / 255.0
    h, w = rgb.shape[:2]
    if h < 4 or w < 4:
        raise ValueError("Image must be at least 4 x 4 pixels")
    # The fixed 0.10 x 0.0515 m top face occupies a ~0.143 m square view.
    # This rectangular ROI stays on that top face, excluding the dark fixture.
    roi = rgb[int(h * 0.35):int(h * 0.65), int(w * 0.2):int(w * 0.8)]
    luminance = roi @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    return float(np.quantile(luminance, 0.1))


def load_manifest(path, *, check_files: bool = True) -> dict:
    """Reject missing classes, image duplicates and cross-split group leakage."""
    path = Path(path)
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("version") != 1:
        raise ValueError("Manifest version must be 1")
    rows = document.get("samples")
    if not isinstance(rows, list) or not rows:
        raise ValueError("Manifest requires a nonempty samples list")
    groups, images, conditions = {}, set(), {}
    root = path.parent.resolve()
    for row in rows:
        if row.get("label") not in LABELS or row.get("split") not in SPLITS:
            raise ValueError("Each sample needs label good/defect and split train/val/test")
        group = row.get("group")
        if not isinstance(group, str) or not group:
            raise ValueError("Each sample needs a nonempty condition group")
        previous = groups.setdefault(group, row["split"])
        if previous != row["split"]:
            raise ValueError(f"Condition group appears in multiple splits: {group}")
        if "condition" in row:
            if not isinstance(row["condition"], dict):
                raise ValueError("Optional condition must be a dictionary")
            signature = json.dumps(row["condition"], sort_keys=True, separators=(",", ":"))
            previous_condition_split = conditions.setdefault(signature, row["split"])
            if previous_condition_split != row["split"]:
                raise ValueError("Identical generation condition appears in multiple splits")
        relative = row.get("image")
        if not isinstance(relative, str) or Path(relative).is_absolute():
            raise ValueError("Image paths must be relative to the manifest")
        image_path = (root / relative).resolve()
        if not image_path.is_relative_to(root):
            raise ValueError("Image must remain within the dataset directory")
        if image_path in images:
            raise ValueError(f"Duplicate image: {relative}")
        images.add(image_path)
        if check_files and not image_path.is_file():
            raise ValueError(f"Missing image: {relative}")
    for split in SPLITS:
        labels = {row["label"] for row in rows if row["split"] == split}
        if labels != set(LABELS):
            raise ValueError(f"Split {split} must contain both classes")
    return document


def split_summary(document: dict) -> dict:
    return {
        split: {
            "images": sum(row["split"] == split for row in document["samples"]),
            "groups": len({row["group"] for row in document["samples"] if row["split"] == split}),
            "classes": {label: sum(row["split"] == split and row["label"] == label
                                   for row in document["samples"]) for label in LABELS},
        } for split in SPLITS
    }
