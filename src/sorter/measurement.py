"""Simulator-independent placement checks and incremental inspection logs.

Geometry is expressed in metres, velocities in m/s and rad/s.  Quaternions use
Isaac Sim's (w, x, y, z) order.  The tray bounds must describe its *interior*,
excluding the walls.  These checks use physical state only for measurement;
they do not provide labels or object names to the image classifier.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence


def _vector(value: Sequence[float], length: int, name: str) -> tuple[float, ...]:
    result = tuple(float(x) for x in value)
    if len(result) != length or not all(math.isfinite(x) for x in result):
        raise ValueError(f"{name} must contain {length} finite numbers")
    return result


@dataclass(frozen=True)
class TrayBounds:
    """Axis-aligned interior bounds; floor_z is the physical support surface."""

    name: str
    minimum: Sequence[float]
    maximum: Sequence[float]
    floor_z: float

    def __post_init__(self) -> None:
        lower = _vector(self.minimum, 3, "minimum")
        upper = _vector(self.maximum, 3, "maximum")
        if any(lo >= hi for lo, hi in zip(lower, upper)):
            raise ValueError("Each tray minimum must be smaller than its maximum")
        if not math.isfinite(self.floor_z) or not lower[2] <= self.floor_z <= upper[2]:
            raise ValueError("floor_z must be finite and inside the tray bounds")
        object.__setattr__(self, "minimum", lower)
        object.__setattr__(self, "maximum", upper)


def world_aabb(
    position: Sequence[float],
    size: Sequence[float],
    orientation_wxyz: Sequence[float] | None = None,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return the world AABB of a box, including its complete rotated extent."""
    center = _vector(position, 3, "position")
    dimensions = _vector(size, 3, "size")
    if any(value <= 0 for value in dimensions):
        raise ValueError("All size components must be positive")
    half = tuple(value / 2 for value in dimensions)
    if orientation_wxyz is not None:
        quat = _vector(orientation_wxyz, 4, "orientation_wxyz")
        norm = math.sqrt(sum(x * x for x in quat))
        if norm < 1e-12:
            raise ValueError("orientation_wxyz cannot be a zero quaternion")
        w, x, y, z = (value / norm for value in quat)
        rotation = (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        )
        half = tuple(sum(abs(row[i]) * half[i] for i in range(3)) for row in rotation)
    return (
        tuple(center[i] - half[i] for i in range(3)),
        tuple(center[i] + half[i] for i in range(3)),
    )


def evaluate_placement(
    tray: TrayBounds,
    position: Sequence[float],
    size: Sequence[float],
    linear_velocity: Sequence[float],
    angular_velocity: Sequence[float],
    *,
    released: bool,
    orientation_wxyz: Sequence[float] | None = None,
    linear_limit: float = 0.015,
    angular_limit: float = 0.15,
    floor_tolerance: float = 0.012,
    containment_tolerance: float = 0.001,
    gripper_clear: bool = True,
) -> dict[str, Any]:
    """Check one sample; success still requires a PlacementMonitor duration.

    Missing/nonfinite measurements fail closed instead of being counted as a
    successful placement.  Velocity limits are strict (<), as specified.
    """
    if min(linear_limit, angular_limit) <= 0 or min(floor_tolerance, containment_tolerance) < 0:
        raise ValueError("Velocity limits must be positive and tolerances nonnegative")
    try:
        lower, upper = world_aabb(position, size, orientation_wxyz)
        linear = _vector(linear_velocity, 3, "linear_velocity")
        angular = _vector(angular_velocity, 3, "angular_velocity")
    except (TypeError, ValueError, OverflowError) as exc:
        return {
            "success": False, "eligible": False, "contained": False,
            "on_floor": False, "stationary": False, "released": bool(released),
            "gripper_clear": bool(gripper_clear), "reason": "invalid_measurement",
            "detail": str(exc), "tray": tray.name,
        }
    linear_speed = math.sqrt(sum(value * value for value in linear))
    angular_speed = math.sqrt(sum(value * value for value in angular))
    contained = all(
        lower[i] >= tray.minimum[i] - containment_tolerance
        and upper[i] <= tray.maximum[i] + containment_tolerance
        for i in range(3)
    )
    on_floor = abs(lower[2] - tray.floor_z) <= floor_tolerance
    stationary = linear_speed < linear_limit and angular_speed < angular_limit
    checks = (
        (bool(released), "not_released"),
        (bool(gripper_clear), "gripper_not_clear"),
        (contained, "outside_tray"),
        (on_floor, "not_on_floor"),
        (stationary, "moving"),
    )
    reason = next((reason for passed, reason in checks if not passed), "settling")
    eligible = all(passed for passed, _ in checks)
    return {
        "success": False, "eligible": eligible, "contained": contained,
        "on_floor": on_floor, "stationary": stationary,
        "released": bool(released), "gripper_clear": bool(gripper_clear),
        "reason": reason, "tray": tray.name,
        "aabb_min": list(lower), "aabb_max": list(upper),
        "linear_speed": linear_speed, "angular_speed": angular_speed,
        "floor_offset": lower[2] - tray.floor_z,
    }


class PlacementMonitor:
    """Require consecutive released, contained, supported and stationary steps.

    Call update once per *physics step*.  At 200 Hz, required_steps=60 means
    0.30 s. Any failed sample resets the consecutive count, including a sample
    after success. Instantiate or reset for every new trial.
    """

    def __init__(self, tray: TrayBounds, required_steps: int = 60, **limits: Any) -> None:
        if not isinstance(required_steps, int) or isinstance(required_steps, bool) or required_steps < 1:
            raise ValueError("required_steps must be a positive integer")
        self.tray = tray
        self.required_steps = required_steps
        self.limits = limits
        self.reset()

    def reset(self) -> None:
        self.settled_steps = 0
        self.samples = 0

    def update(
        self,
        position: Sequence[float],
        size: Sequence[float],
        linear_velocity: Sequence[float],
        angular_velocity: Sequence[float],
        *,
        released: bool,
        orientation_wxyz: Sequence[float] | None = None,
        gripper_clear: bool = True,
    ) -> dict[str, Any]:
        result = evaluate_placement(
            self.tray, position, size, linear_velocity, angular_velocity,
            released=released, orientation_wxyz=orientation_wxyz,
            gripper_clear=gripper_clear, **self.limits,
        )
        self.samples += 1
        self.settled_steps = self.settled_steps + 1 if result["eligible"] else 0
        result.update(
            success=self.settled_steps >= self.required_steps,
            settled_steps=self.settled_steps,
            required_steps=self.required_steps,
            samples=self.samples,
        )
        if result["success"]:
            result["reason"] = "placed_and_settled"
        return result


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def atomic_write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    """Replace metrics atomically, preserving the previous file on write errors."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent, suffix=".tmp", delete=False)
    temporary = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
    finally:
        if temporary.exists():
            temporary.unlink()


class InspectionLog:
    """Append each completed/failed trial immediately to JSONL and CSV.

    JSONL retains the full record. CSV uses stable columns plus record_json for
    detailed physical checks. Existing logs are appended, never overwritten.
    """

    COLUMNS = (
        "timestamp_utc", "run_id", "trial", "image", "defect_score", "confidence",
        "decision", "destination", "expected_destination", "success", "reason",
        "cycle_time", "settled_steps", "stop_reason", "record_json",
    )

    def __init__(self, output_dir: str | Path, stem: str = "inspections") -> None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = directory / f"{stem}.jsonl"
        self.csv_path = directory / f"{stem}.csv"

    def write(self, record: Mapping[str, Any]) -> dict[str, Any]:
        item = dict(record)
        item.setdefault("timestamp_utc", datetime.now(timezone.utc).isoformat())
        encoded = json.dumps(item, ensure_ascii=False, default=_json_default, allow_nan=False)
        with self.jsonl_path.open("a", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        needs_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        row = {key: item.get(key, "") for key in self.COLUMNS}
        row["record_json"] = encoded
        with self.csv_path.open("a", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=self.COLUMNS)
            if needs_header:
                writer.writeheader()
            writer.writerow(row)
            stream.flush()
            os.fsync(stream.fileno())
        return item


def summarize_records(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate actual boolean outcomes; pending/unknown results stay separate."""
    rows = list(records)
    measured = [row for row in rows if isinstance(row.get("success"), bool)]
    succeeded = sum(row["success"] for row in measured)
    failures: dict[str, int] = {}
    destinations: dict[str, dict[str, int]] = {}
    durations = []
    for row in measured:
        destination = str(row.get("destination", row.get("tray", "unspecified")))
        group = destinations.setdefault(destination, {"trials": 0, "success": 0})
        group["trials"] += 1
        group["success"] += int(row["success"])
        if not row["success"]:
            reason = str(row.get("reason", row.get("failure_reason", "unspecified")))
            failures[reason] = failures.get(reason, 0) + 1
        duration = row.get("cycle_time", row.get("duration_s"))
        if isinstance(duration, (float, int)) and not isinstance(duration, bool) and math.isfinite(duration) and duration >= 0:
            durations.append(float(duration))
    return {
        "records": len(rows), "measured_trials": len(measured),
        "pending_or_unknown": len(rows) - len(measured), "success": succeeded,
        "failed": len(measured) - succeeded,
        "success_rate": succeeded / len(measured) if measured else None,
        "mean_cycle_time_s": sum(durations) / len(durations) if durations else None,
        "failure_reasons": failures, "by_destination": destinations,
    }
