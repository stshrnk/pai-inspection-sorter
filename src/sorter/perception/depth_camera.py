"""3D depth camera setup for bin picking.

Mounts a depth camera above the bin and provides depth image + point cloud
for detecting pick targets.
"""

import numpy as np
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.sensors.camera import Camera

from ..config import (
    CAMERA_FREQUENCY,
    CAMERA_ORIENTATION_EULER,
    CAMERA_POSITION,
    CAMERA_PRIM_PATH,
    CAMERA_RESOLUTION,
)


class DepthCameraManager:
    """Manages a depth camera sensor above the bin."""

    def __init__(self) -> None:
        self._camera = Camera(
            prim_path=CAMERA_PRIM_PATH,
            position=CAMERA_POSITION,
            orientation=euler_angles_to_quat(CAMERA_ORIENTATION_EULER, degrees=True),
            frequency=CAMERA_FREQUENCY,
            resolution=CAMERA_RESOLUTION,
        )

    def initialize(self) -> None:
        """Call after world.reset() to initialize the render product and attach annotators."""
        self._camera.initialize()
        self._camera.add_distance_to_image_plane_to_frame()
        self._camera.add_pointcloud_to_frame(include_unlabelled=True)
        self._camera.add_semantic_segmentation_to_frame()

    def get_semantic_segmentation(self) -> tuple[np.ndarray | None, dict | None]:
        """Return (segmentation_image, id_to_label). Image shape: (H, W) uint32.

        The id_to_label dict maps pixel IDs to dicts like {"class": "red_cube"}.
        Returns (None, None) if no frame available.
        """
        data = self._camera.get_current_frame().get("semantic_segmentation")
        if not data:
            return None, None
        return data.get("data"), data.get("info", {}).get("idToLabels")

    def get_depth_image(self) -> np.ndarray:
        """Return the depth image (H x W) in meters."""
        return self._camera.get_depth()

    def get_rgb_image(self) -> np.ndarray:
        """Return the RGB image (H x W x 3)."""
        return self._camera.get_rgb()

    def get_pointcloud(self) -> np.ndarray:
        """Return the 3D point cloud from the depth image (N x 3) in world frame.

        Uses depth-to-pointcloud conversion via camera intrinsics as fallback
        if the annotator returns empty data.
        """
        pc = self._camera.get_pointcloud()
        if pc is not None and pc.size > 3:
            return pc

        # Fallback: compute from depth + intrinsics
        return self._depth_to_pointcloud()

    def get_labeled_pointcloud(self) -> tuple[np.ndarray | None, np.ndarray | None, dict | None]:
        """Return (points_world, per-point semantic ids, id_to_label_dict).

        Uses Isaac Sim's built-in pinhole inverse projection so the world
        transform matches the camera's actual view matrix. Each 3D point
        keeps a 1-to-1 mapping with its source pixel, letting us look up
        the pixel's semantic id. Returns (None, None, None) on failure.
        """
        depth = self.get_depth_image()
        if depth is None or depth.size == 0:
            return None, None, None

        seg_img, id_to_labels = self.get_semantic_segmentation()
        if seg_img is None or id_to_labels is None:
            return None, None, None

        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        valid = (depth > 0) & (depth < 100)
        u_v = u[valid]
        v_v = v[valid]
        d_v = depth[valid]
        seg_flat = seg_img.reshape(h, w) if seg_img.ndim == 2 else seg_img[..., 0]
        ids = seg_flat[valid]

        points_2d = np.stack([u_v, v_v], axis=-1).astype(np.float32)
        try:
            pts_world = self._camera.get_world_points_from_image_coords(
                points_2d, d_v.astype(np.float32)
            )
        except Exception:
            return None, None, None
        pts_world = np.asarray(pts_world)

        return pts_world.astype(np.float32), ids.astype(np.int32), id_to_labels

    def _depth_to_pointcloud(self) -> np.ndarray | None:
        """Convert depth image to 3D point cloud using camera intrinsics."""
        depth = self.get_depth_image()
        if depth is None or depth.size == 0:
            return None

        h, w = depth.shape
        intrinsics = self._camera.get_intrinsics_matrix()
        if intrinsics is None:
            return None

        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        # Create pixel coordinate grid
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        # Filter out invalid depth values
        valid = (depth > 0) & (depth < 100)  # reasonable range in meters
        u_valid = u[valid]
        v_valid = v[valid]
        d_valid = depth[valid]

        # Back-project to camera frame
        x_cam = (u_valid - cx) * d_valid / fx
        y_cam = (v_valid - cy) * d_valid / fy
        z_cam = d_valid

        points_cam = np.stack([x_cam, y_cam, z_cam], axis=-1)

        # Transform to world frame
        world_pose = self._camera.get_world_pose()
        if world_pose is None:
            return points_cam

        cam_pos, cam_quat = world_pose
        rot_matrix = self._quat_to_rotation_matrix(cam_quat)
        points_world = (rot_matrix @ points_cam.T).T + cam_pos

        return points_world.astype(np.float32)

    @staticmethod
    def _quat_to_rotation_matrix(quat: np.ndarray) -> np.ndarray:
        """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
        w, x, y, z = quat
        return np.array([
            [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
            [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
            [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
        ])

    @property
    def camera(self) -> Camera:
        return self._camera
