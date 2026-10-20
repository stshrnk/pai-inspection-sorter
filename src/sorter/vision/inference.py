"""Frozen model inference accepts only image pixels, never a manifest row."""

from __future__ import annotations

import json
from pathlib import Path

from .data import brightness_feature, image_array
from .metrics import brightness_scores, sigmoid


class SorterClassifier:
    def __init__(self, checkpoint, *, method="cnn", device="cpu"):
        self.method = method
        self.device = device
        checkpoint = Path(checkpoint)
        if method == "cnn":
            import torch
            from .model import TinyInspectionCNN

            self._torch = torch
            saved = torch.load(checkpoint, map_location=device, weights_only=True)
            self.model = TinyInspectionCNN().to(device)
            self.model.load_state_dict(saved["state_dict"])
            self.model.eval()
            self.temperature = saved["temperature"]
            self.review_threshold = saved["review_policy"]["threshold"]
        elif method == "brightness":
            self.config = json.loads(checkpoint.read_text(encoding="utf-8"))
            self.review_threshold = self.config["review_policy"]["threshold"]
        else:
            raise ValueError("Method must be cnn or brightness")

    def predict(self, image) -> dict:
        if self.method == "cnn":
            tensor = self._torch.from_numpy(image_array(image)).unsqueeze(0).to(self.device)
            with self._torch.inference_mode():
                logit = float(self.model(tensor).cpu().item())
            score = float(sigmoid(logit / self.temperature))
        else:
            score = float(brightness_scores([brightness_feature(image)], self.config)[0])
        confidence = max(score, 1 - score)
        label = "review" if confidence < self.review_threshold else ("defect" if score >= 0.5 else "good")
        return {"label": label, "defect_score": score, "confidence": confidence,
                "review_threshold": self.review_threshold, "method": self.method}
