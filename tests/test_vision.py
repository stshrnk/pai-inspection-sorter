"""Meaningful guards against condition leakage and unsafe confidence routing."""

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sorter.vision.data import brightness_feature, image_array, load_manifest
from sorter.vision.inference import SorterClassifier
from sorter.vision.metrics import choose_review_threshold, evaluate_scores, fit_brightness, brightness_scores


class VisionTests(unittest.TestCase):
    def manifest(self):
        return {"version": 1, "samples": [
            {"image": f"{split}-{label}.png", "label": label, "split": split, "group": split}
            for split in ("train", "val", "test") for label in ("good", "defect")
        ]}

    def load(self, document):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "manifest.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            return load_manifest(path, check_files=False)

    def test_group_leakage_rejected(self):
        document = self.manifest()
        document["samples"][-1]["group"] = "train"
        with self.assertRaisesRegex(ValueError, "multiple splits"):
            self.load(document)

    def test_duplicate_image_rejected(self):
        document = self.manifest()
        document["samples"][-1]["image"] = document["samples"][0]["image"]
        with self.assertRaisesRegex(ValueError, "Duplicate image"):
            self.load(document)

    def test_identical_condition_with_different_group_names_rejected(self):
        document = self.manifest()
        document["samples"][0]["condition"] = {"light": 300}
        document["samples"][-1]["condition"] = {"light": 300}
        with self.assertRaisesRegex(ValueError, "Identical generation condition"):
            self.load(document)

    def test_path_escape_rejected(self):
        document = self.manifest()
        document["samples"][0]["image"] = "../elsewhere.png"
        with self.assertRaisesRegex(ValueError, "within the dataset"):
            self.load(document)

    def test_condition_pairs_within_same_split_allowed(self):
        self.assertEqual(len(self.load(self.manifest())["samples"]), 6)

    def test_low_confidence_error_routed_to_review(self):
        scores, targets = [0.05, 0.95, 0.55, 0.1], [0, 1, 0, 0]
        policy = choose_review_threshold(scores, targets)
        metrics = evaluate_scores(scores, targets, policy["threshold"])
        self.assertEqual(metrics["review_count"], 1)
        self.assertEqual(metrics["accepted_error"], 0.0)
        self.assertEqual(metrics["coverage"], 0.75)

    def test_no_trustworthy_validation_set_rejects_everything(self):
        scores, targets = [0.99, 0.95, 0.9], [0, 0, 0]
        policy = choose_review_threshold(scores, targets)
        metrics = evaluate_scores(scores, targets, policy["threshold"])
        self.assertEqual(metrics["coverage"], 0)
        self.assertIsNone(metrics["accepted_error"])

    def test_perfect_confidence_ties_do_not_force_all_to_review(self):
        scores, targets = [0.0, 1.0, 0.0, 1.0], [0, 1, 0, 1]
        policy = choose_review_threshold(scores, targets)
        self.assertEqual(evaluate_scores(scores, targets, policy["threshold"])["coverage"], 1.0)

    def test_inference_uses_pixels_only(self):
        config = fit_brightness([0.8, 0.9, 0.1, 0.2], [0, 0, 1, 1])
        config["review_policy"] = {"threshold": 0.7}
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            path = Path(directory) / "brightness.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            classifier = SorterClassifier(path, method="brightness")
            result = classifier.predict(np.full((64, 64, 3), 30, dtype=np.uint8))
        self.assertEqual(result["label"], "defect")
        self.assertGreater(result["defect_score"], 0.9)

    def test_brightness_fits_darker_defects(self):
        features, targets = [0.8, 0.9, 0.1, 0.2], [0, 0, 1, 1]
        baseline = fit_brightness(features, targets)
        np.testing.assert_array_equal(brightness_scores(features, baseline) >= 0.5, np.array(targets, bool))

    def test_rgb_and_rgba_preprocessing_agree(self):
        rgb = np.full((80, 80, 3), 180, dtype=np.uint8)
        rgba = np.concatenate((rgb, np.zeros((80, 80, 1), np.uint8)), axis=2)
        np.testing.assert_equal(image_array(rgb), image_array(rgba))
        self.assertEqual(image_array(rgb).shape, (3, 64, 64))
        self.assertAlmostEqual(brightness_feature(rgb), 180 / 255, places=5)


if __name__ == "__main__":
    unittest.main()
