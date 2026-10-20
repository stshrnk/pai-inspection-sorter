"""Binary quality and validation-only review calibration (no sklearn needed)."""

from __future__ import annotations

import numpy as np


def sigmoid(logits) -> np.ndarray:
    logits = np.clip(np.asarray(logits, dtype=np.float64), -50, 50)
    return 1.0 / (1.0 + np.exp(-logits))


def choose_temperature(logits, targets) -> float:
    logits = np.asarray(logits, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    candidates = np.geomspace(0.1, 10.0, 81)
    losses = [np.mean(np.logaddexp(0, logits / t) - targets * logits / t) for t in candidates]
    return float(candidates[int(np.argmin(losses))])


def choose_review_threshold(scores, targets, *, minimum_review_fraction=0.10,
                            maximum_accepted_error=0.05) -> dict:
    """Choose widest accepted set satisfying validation risk and a review reserve.

    Confidence is max(p, 1-p). The least-confident validation decile defines
    the initial review boundary (ties can reduce the requested fraction). If no
    nonempty accepted set meets the risk bound, reject all at inference.
    This is an empirical validation policy, not a statistical safety guarantee.
    """
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if not len(scores) or scores.shape != targets.shape:
        raise ValueError("Review calibration requires equally-sized nonempty scores and targets")
    confidence = np.maximum(scores, 1 - scores)
    wrong = (scores >= 0.5).astype(int) != targets
    floor = float(np.quantile(confidence, minimum_review_fraction))
    candidates = sorted(set([float(floor), *map(float, confidence[confidence >= floor])]))
    chosen = float(np.nextafter(1.0, np.inf))
    for threshold in candidates:
        accepted = confidence >= threshold
        if accepted.any() and float(wrong[accepted].mean()) <= maximum_accepted_error:
            chosen = threshold
            break
    report = evaluate_scores(scores, targets, chosen)
    return {
        "threshold": chosen,
        "defect_score_review_interval": [max(0.0, 1 - chosen), min(1.0, chosen)],
        "minimum_review_fraction": minimum_review_fraction,
        "review_fraction_tie_policy": "Confidence equal to the boundary is accepted; tied scores can reduce the requested fraction.",
        "maximum_validation_accepted_error": maximum_accepted_error,
        "validation_coverage": report["coverage"],
        "validation_accepted_error": report["accepted_error"],
        "note": "Empirical validation calibration; confidence is not a guaranteed probability.",
    }


def evaluate_scores(scores, targets, review_threshold=0.5, groups=None) -> dict:
    scores = np.asarray(scores, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    if not len(scores) or scores.shape != targets.shape:
        raise ValueError("Evaluation requires equally-sized nonempty scores and targets")
    predicted = (scores >= 0.5).astype(int)
    confidence = np.maximum(scores, 1 - scores)
    accepted = confidence >= review_threshold
    confusion = [[int(np.sum((targets == actual) & (predicted == prediction)))
                  for prediction in (0, 1)] for actual in (0, 1)]
    routed = [[int(np.sum((targets == actual) & accepted & (predicted == prediction)))
               for prediction in (0, 1)] + [int(np.sum((targets == actual) & ~accepted))]
              for actual in (0, 1)]
    tp, fp, fn = confusion[1][1], confusion[0][1], confusion[1][0]
    result = {
        "samples": len(targets),
        "accuracy": float(np.mean(predicted == targets)),
        "defect_precision": tp / (tp + fp) if tp + fp else 0.0,
        "defect_recall": tp / (tp + fn) if tp + fn else 0.0,
        "confusion": {"rows": ["good", "defect"], "columns": ["good", "defect"],
                      "matrix": confusion},
        "routing_confusion": {"rows": ["good", "defect"],
                              "columns": ["good", "defect", "review"], "matrix": routed},
        "review_count": int(np.sum(~accepted)),
        "coverage": float(accepted.mean()),
        "accepted_error": float(np.mean(predicted[accepted] != targets[accepted])) if accepted.any() else None,
        "review_threshold": float(review_threshold),
        "brier_score": float(np.mean((scores - targets) ** 2)),
    }
    if groups is not None:
        groups = np.asarray(groups)
        result["by_group"] = {
            str(group): evaluate_scores(scores[groups == group], targets[groups == group], review_threshold)
            for group in sorted(set(groups.tolist()))
        }
    return result


def fit_brightness(features, targets) -> dict:
    """Fit a single luminance threshold on training images only."""
    features, targets = np.asarray(features), np.asarray(targets)
    unique = np.unique(features)
    boundaries = np.r_[unique[0] - 1e-6, (unique[:-1] + unique[1:]) / 2, unique[-1] + 1e-6]
    best = None
    for polarity in (-1.0, 1.0):
        for boundary in boundaries:
            error = float(np.mean(((features - boundary) * polarity >= 0).astype(int) != targets))
            if best is None or error < best[0]:
                best = (error, float(boundary), polarity)
    scale = max(float(np.std(features)) * 0.25, 0.005)
    return {"threshold": best[1], "polarity": best[2], "score_scale": scale,
            "feature": "10th percentile luminance in fixed central width-60-percent height-30-percent ROI",
            "training_error": best[0]}


def brightness_scores(features, config) -> np.ndarray:
    return sigmoid((np.asarray(features) - config["threshold"]) * config["polarity"] / config["score_scale"])
