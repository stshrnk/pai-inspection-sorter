#!/usr/bin/env python3
"""Train on train, select/calibrate on val, and leave test unmeasured."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from torch import nn

from sorter.vision.data import brightness_feature, image_array, load_manifest, split_summary
from sorter.vision.metrics import (brightness_scores, choose_review_threshold, choose_temperature,
                                   evaluate_scores, fit_brightness, sigmoid)
from sorter.vision.model import TinyInspectionCNN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--minimum-review-fraction", type=float, default=0.10)
    parser.add_argument("--maximum-accepted-error", type=float, default=0.05)
    args = parser.parse_args()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.threads < 1:
        parser.error("epochs, patience, batch-size and threads must be positive")
    if not 0 < args.minimum_review_fraction < 1 or not 0 <= args.maximum_accepted_error < 1:
        parser.error("Invalid review policy fractions")
    document = load_manifest(args.manifest)
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if (args.out_dir / "classifier.pt").exists():
        parser.error("Output directory already contains a model; use a new directory")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    tensors, labels, brightness, rows_by_split = {}, {}, {}, {}
    for split in ("train", "val"):
        rows = [row for row in document["samples"] if row["split"] == split]
        rows_by_split[split] = rows
        paths = [args.manifest.parent / row["image"] for row in rows]
        tensors[split] = torch.from_numpy(np.stack([image_array(path) for path in paths])).to(args.device)
        labels[split] = torch.tensor([row["label"] == "defect" for row in rows],
                                     dtype=torch.float32, device=args.device)
        brightness[split] = np.array([brightness_feature(path) for path in paths])
    model = TinyInspectionCNN().to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_loss, best_epoch, best_state = float("inf"), 0, None
    history = []
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(len(labels["train"]), device=args.device)
        loss_sum = 0.0
        for start in range(0, len(order), args.batch_size):
            batch = order[start:start + args.batch_size]
            images = tensors["train"][batch].clone()
            # Stain orientation and mild exposure changes are irrelevant to the class.
            if torch.rand(()).item() < 0.5:
                images = images.flip(3)
            if torch.rand(()).item() < 0.5:
                images = images.flip(2)
            gains = 0.9 + torch.rand((len(batch), 1, 1, 1), device=args.device) * 0.2
            images = (images * gains).clamp(0, 1)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels["train"][batch])
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach().cpu()) * len(batch)
        model.eval()
        with torch.inference_mode():
            val_logits = model(tensors["val"])
            val_loss = float(criterion(val_logits, labels["val"]).cpu())
            val_accuracy = float(((val_logits >= 0) == labels["val"].bool()).float().mean().cpu())
        history.append({"epoch": epoch, "train_loss": loss_sum / len(order),
                        "val_loss": val_loss, "val_accuracy": val_accuracy})
        if val_loss < best_loss - 1e-6:
            best_loss, best_epoch = val_loss, epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0:
            print(f"epoch={epoch} train_loss={loss_sum / len(order):.4f} val_loss={val_loss:.4f} val_accuracy={val_accuracy:.3f}", flush=True)
        if epoch - best_epoch >= args.patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        val_logits = model(tensors["val"]).cpu().numpy()
    val_targets = labels["val"].cpu().numpy().astype(int)
    train_targets = labels["train"].cpu().numpy().astype(int)
    temperature = choose_temperature(val_logits, val_targets)
    val_scores = sigmoid(val_logits / temperature)
    policy_args = {"minimum_review_fraction": args.minimum_review_fraction,
                   "maximum_accepted_error": args.maximum_accepted_error}
    review_policy = choose_review_threshold(val_scores, val_targets, **policy_args)
    baseline = fit_brightness(brightness["train"], train_targets)
    baseline["review_policy"] = choose_review_threshold(
        brightness_scores(brightness["val"], baseline), val_targets, **policy_args)
    metadata = {
        "format_version": 1, "architecture": "TinyInspectionCNN", "input_size": [64, 64],
        "labels": ["good", "defect"], "seed": args.seed, "device": args.device,
        "best_epoch": best_epoch, "temperature": temperature, "review_policy": review_policy,
        "manifest_sha256": manifest_hash,
    }
    torch.save({**metadata, "state_dict": best_state}, args.out_dir / "classifier.pt")
    (args.out_dir / "brightness.json").write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")
    val_groups = [row["group"] for row in rows_by_split["val"]]
    report = {
        **metadata, "split_summary": split_summary(document), "condition_groups_disjoint": True,
        "trainable_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "elapsed_seconds": time.monotonic() - started, "epochs_run": len(history), "history": history,
        "cnn_validation": evaluate_scores(val_scores, val_targets, review_policy["threshold"], val_groups),
        "brightness_validation": evaluate_scores(brightness_scores(brightness["val"], baseline), val_targets,
                                                 baseline["review_policy"]["threshold"], val_groups),
        "test_evaluated": False,
        "limits": ["Synthetic rendered images only; real-world generalization is unmeasured.",
                   "Validation selects epoch, temperature and review policy; only independent test estimates final performance."],
    }
    (args.out_dir / "training.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"best_epoch": best_epoch, "val_accuracy": report["cnn_validation"]["accuracy"],
                      "baseline_val_accuracy": report["brightness_validation"]["accuracy"],
                      "val_coverage": report["cnn_validation"]["coverage"], "test_evaluated": False}), flush=True)


if __name__ == "__main__":
    main()
