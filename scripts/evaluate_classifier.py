#!/usr/bin/env python3
"""Evaluate a frozen CNN and brightness baseline on untouched test groups."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch

from sorter.vision.data import load_manifest, split_summary
from sorter.vision.inference import SorterClassifier
from sorter.vision.metrics import evaluate_scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--overwrite", action="store_true", help="Explicitly repeat an already recorded test evaluation")
    args = parser.parse_args()
    if args.out.exists() and not args.overwrite:
        parser.error("Evaluation already exists; preserve the held-out result or explicitly use --overwrite")
    torch.set_num_threads(args.threads)
    document = load_manifest(args.manifest)
    training = json.loads((args.model_dir / "training.json").read_text(encoding="utf-8"))
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != training["manifest_sha256"]:
        parser.error("Manifest changed after training; evaluation requires the original held-out split")
    rows = [row for row in document["samples"] if row["split"] == "test"]
    targets = np.array([row["label"] == "defect" for row in rows], dtype=int)
    groups = [row["group"] for row in rows]
    report = {"split": "test", "split_summary": split_summary(document),
              "condition_groups_disjoint": True, "manifest_sha256": training["manifest_sha256"],
              "seed": training["seed"], "methods": {}}
    prediction_rows = []
    for method, checkpoint in (("cnn", "classifier.pt"), ("brightness", "brightness.json")):
        classifier = SorterClassifier(args.model_dir / checkpoint, method=method)
        predictions = [classifier.predict(args.manifest.parent / row["image"]) for row in rows]
        scores = np.array([prediction["defect_score"] for prediction in predictions])
        metrics = evaluate_scores(scores, targets, classifier.review_threshold, groups)
        conditions = {}
        condition_keys = sorted({key for row in rows for key in row.get("condition", {})})
        for key in condition_keys:
            values = [json.dumps(row.get("condition", {}).get(key), sort_keys=True) for row in rows]
            conditions[key] = {}
            for value in sorted(set(values)):
                selected = np.array([v == value for v in values])
                conditions[key][value] = evaluate_scores(scores[selected], targets[selected], classifier.review_threshold)
        metrics["by_condition"] = conditions
        metrics["failed_images"] = [row["image"] for row, score, target in zip(rows, scores, targets)
                                    if int(score >= 0.5) != target]
        report["methods"][method] = metrics
        prediction_rows.extend({"image": row["image"], "group": row["group"], "truth": row["label"],
                                **prediction} for row, prediction in zip(rows, predictions))
    report["limits"] = ["Held-out simulation condition groups, not physical camera or real-world data.",
                         "Review coverage and accepted error use validation-frozen thresholds.",
                         "A high review rate can reduce accepted error while reducing throughput."]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with args.out.with_suffix(".predictions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    print(json.dumps({method: {key: metrics[key] for key in ("accuracy", "coverage", "accepted_error", "review_count")}
                      for method, metrics in report["methods"].items()}), flush=True)


if __name__ == "__main__":
    main()
