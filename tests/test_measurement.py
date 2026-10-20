"""Meaningful checks against false-positive placement and lost log outcomes."""

import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

from src.sorter.measurement import (
    InspectionLog, PlacementMonitor, TrayBounds, atomic_write_json,
    evaluate_placement, summarize_records,
)


class PlacementTests(unittest.TestCase):
    def setUp(self):
        self.tray = TrayBounds("good", (-0.1, -0.1, 0.0), (0.1, 0.1, 0.2), 0.0)
        self.sample = dict(position=(0, 0, 0.025), size=(0.08, 0.06, 0.05),
                           linear_velocity=(0, 0, 0), angular_velocity=(0, 0, 0), released=True)

    def check(self, **updates):
        return evaluate_placement(self.tray, **(self.sample | updates))

    def test_center_inside_does_not_allow_overhang(self):
        result = self.check(position=(0.08, 0, 0.025))
        self.assertFalse(result["contained"])
        self.assertFalse(result["eligible"])

    def test_hovering_and_gripped_parts_do_not_succeed(self):
        self.assertEqual(self.check(position=(0, 0, 0.07))["reason"], "not_on_floor")
        self.assertEqual(self.check(released=False)["reason"], "not_released")
        self.assertEqual(self.check(gripper_clear=False)["reason"], "gripper_not_clear")

    def test_rotated_extent_is_included(self):
        angle = math.pi / 4
        result = self.check(position=(0.055, 0, 0.025),
                            orientation_wxyz=(math.cos(angle / 2), 0, 0, math.sin(angle / 2)))
        self.assertFalse(result["contained"])

    def test_speed_norm_and_angular_speed_prevent_false_success(self):
        self.assertFalse(self.check(linear_velocity=(0.01, 0.01, 0.01))["stationary"])
        self.assertFalse(self.check(angular_velocity=(0, 0, 0.16))["stationary"])

    def test_invalid_measurement_fails_closed(self):
        self.assertEqual(self.check(linear_velocity=None)["reason"], "invalid_measurement")
        self.assertFalse(self.check(position=(math.nan, 0, 0))["eligible"])
        self.assertFalse(self.check(orientation_wxyz=(0, 0, 0, 0))["eligible"])

    def test_requires_consecutive_samples_after_release(self):
        monitor = PlacementMonitor(self.tray, required_steps=3)
        self.assertFalse(monitor.update(**(self.sample | {"released": False}))["success"])
        self.assertFalse(monitor.update(**self.sample)["success"])
        self.assertFalse(monitor.update(**self.sample)["success"])
        moving = monitor.update(**(self.sample | {"linear_velocity": (0.02, 0, 0)}))
        self.assertEqual(moving["settled_steps"], 0)
        self.assertFalse(monitor.update(**self.sample)["success"])
        self.assertFalse(monitor.update(**self.sample)["success"])
        self.assertTrue(monitor.update(**self.sample)["success"])
        self.assertFalse(monitor.update(**(self.sample | {"released": False}))["success"])


class LoggingTests(unittest.TestCase):
    def test_append_keeps_failed_trials_and_full_details(self):
        with tempfile.TemporaryDirectory() as temporary:
            logger = InspectionLog(temporary)
            logger.write({"trial": 1, "success": True, "decision": "good", "checks": {"contained": True}})
            InspectionLog(temporary).write({"trial": 2, "success": False, "reason": "outside_tray"})
            lines = logger.jsonl_path.read_text().splitlines()
            self.assertEqual(len(lines), 2)
            with logger.csv_path.open(newline="") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(json.loads(rows[0]["record_json"])["checks"], {"contained": True})
            self.assertFalse(json.loads(lines[1])["success"])

    def test_atomic_metrics_do_not_replace_valid_result_with_nonfinite_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.json"
            atomic_write_json(path, {"success": False})
            with self.assertRaises(ValueError):
                atomic_write_json(path, {"speed": math.nan})
            self.assertEqual(json.loads(path.read_text()), {"success": False})
            self.assertEqual(list(Path(temporary).glob("*.tmp")), [])

    def test_summary_does_not_treat_pending_or_string_as_measured_success(self):
        summary = summarize_records([
            {"success": True, "cycle_time": 2},
            {"success": False, "reason": "outside_tray", "cycle_time": 4},
            {"success": "false"}, {"status": "pending"},
        ])
        self.assertEqual(summary["measured_trials"], 2)
        self.assertEqual(summary["pending_or_unknown"], 2)
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["mean_cycle_time_s"], 3)
        self.assertEqual(summary["failure_reasons"], {"outside_tray": 1})
        self.assertIsNone(summarize_records([])["success_rate"])


if __name__ == "__main__":
    unittest.main()
