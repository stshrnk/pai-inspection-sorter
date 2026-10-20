"""Top-object detector using point-cloud heuristics.

Finds the highest cluster of points inside the source bin and estimates its
centroid and yaw. The yaw is recovered with PCA on the top-surface points so
the gripper can be aligned with the object's shorter side.

This is a minimal, Sim2Real-friendly stand-in for foundation-model grasp
prediction (e.g. FoundationPose / AnyGrasp). Drop-in replacement targets:
  - centroid in world frame (meters)
  - yaw about the world Z axis (radians, wraps to [-pi/2, pi/2])
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import BIN_POSITION, BIN_SIZE


@dataclass
class ObjectDetection:
    """Result of top-object detection."""
    position: np.ndarray  # (3,) world-frame centroid
    yaw: float            # rotation about Z axis (radians) of object's long axis
    num_points: int       # points used in the fit


class TopObjectDetector:
    """Detects the top-most object inside the source bin from a point cloud."""

    # Margin inside the bin walls (meters) — ignore points too close to walls.
    BIN_MARGIN = 0.06
    # Height slab around the top-most point to consider "top surface" (meters).
    TOP_SLAB_HEIGHT = 0.008
    # Minimum points required to trust a PCA yaw estimate.
    MIN_POINTS_FOR_PCA = 15
    # Max Z above bin floor for a candidate object. Excludes bin walls/rim.
    Z_MIN_ABOVE_FLOOR = 0.02   # ignore the bin floor itself
    Z_MAX_ABOVE_FLOOR = 0.10   # cube ~5cm tall; reject walls/rim (>0.10)

    def __init__(self) -> None:
        self._bin_center = BIN_POSITION
        self._bin_half = BIN_SIZE / 2.0

    def _points_in_bin(self, pc: np.ndarray) -> np.ndarray:
        cx, cy, cz = self._bin_center
        hx, hy, hz = self._bin_half
        m = self.BIN_MARGIN
        mask = (
            (pc[:, 0] > cx - hx + m) & (pc[:, 0] < cx + hx - m) &
            (pc[:, 1] > cy - hy + m) & (pc[:, 1] < cy + hy - m) &
            (pc[:, 2] > cz + self.Z_MIN_ABOVE_FLOOR) &
            (pc[:, 2] < cz + self.Z_MAX_ABOVE_FLOOR)
        )
        return pc[mask]

    def detect_labeled_object(
        self,
        points: np.ndarray,
        ids: np.ndarray,
        id_to_labels: dict,
        label: str,
    ) -> ObjectDetection | None:
        """Detect an object by semantic label (preferred when available).

        Filters the point cloud to points whose pixel had the requested
        semantic label, then runs the same top-slab / PCA routine to
        estimate centroid and yaw.
        """
        if points is None or ids is None or id_to_labels is None:
            return None
        # id_to_labels maps id (str) -> {"class": "<label>"}.
        matching_ids = [
            int(i) for i, meta in id_to_labels.items()
            if isinstance(meta, dict) and meta.get("class") == label
        ]
        if not matching_ids:
            return None
        mask = np.isin(ids, matching_ids)
        labelled = points[mask]
        if len(labelled) == 0:
            return None
        return self._fit_detection(labelled)

    def detect_top_object(self, pointcloud: np.ndarray) -> ObjectDetection | None:
        """Return centroid + yaw of the top object in the bin (label-free)."""
        if pointcloud is None or len(pointcloud) == 0:
            return None
        bin_points = self._points_in_bin(np.asarray(pointcloud))
        if len(bin_points) == 0:
            return None
        return self._fit_detection(bin_points)

    def _fit_detection(self, points: np.ndarray) -> ObjectDetection | None:
        """Take candidate points, extract the top slab, and fit centroid+yaw."""
        if points is None or len(points) == 0:
            return None
        z_max = float(points[:, 2].max())
        slab_mask = points[:, 2] > (z_max - self.TOP_SLAB_HEIGHT)
        top_points = points[slab_mask]
        if len(top_points) == 0:
            return None

        centroid = top_points.mean(axis=0)

        yaw = 0.0
        if len(top_points) >= self.MIN_POINTS_FOR_PCA:
            xy = top_points[:, :2] - centroid[:2]
            cov = np.cov(xy.T)
            _, eigvecs = np.linalg.eigh(cov)
            principal = eigvecs[:, -1]
            yaw = float(np.arctan2(principal[1], principal[0]))
            # Wrap to [-pi/2, pi/2] (axis is 180°-symmetric).
            if yaw > np.pi / 2:
                yaw -= np.pi
            elif yaw < -np.pi / 2:
                yaw += np.pi

        return ObjectDetection(
            position=centroid.astype(np.float64),
            yaw=yaw,
            num_points=int(len(top_points)),
        )
