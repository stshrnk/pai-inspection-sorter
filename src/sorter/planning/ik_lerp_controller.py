"""IK + Linear Interpolation controller for UR10e."""

import os

import numpy as np
from isaacsim.core.api.controllers.base_controller import BaseController
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.types import ArticulationAction
import isaacsim.robot_motion.motion_generation as mg


class IKLerpController(BaseController):
    """Computes target joint angles via IK, then linearly interpolates toward them.

    Drop-in replacement for RMPFlowController as cspace_controller in PickPlaceController.
    Inherits from BaseController to match the interface expected by the parent PickPlaceController.
    """

    RMPFLOW_DIR = os.path.join(os.path.dirname(__file__), "rmpflow")

    # IK seed: elbow-up pose (upper arm upright, elbow above workspace)
    # [shoulder_pan, shoulder_lift, elbow, wrist_1, wrist_2, wrist_3]
    ELBOW_UP_SEED = np.array([0.0, -np.pi / 2, np.pi / 2, -np.pi, -np.pi / 2, 0.0])
    # Joint indices in cspace
    _SHOULDER_LIFT_IDX = 1
    _ELBOW_IDX = 2

    def __init__(
        self,
        name: str,
        robot_articulation: Articulation,
        physics_dt: float = 1.0 / 60.0,
        interpolation_speed: float = 0.05,
    ) -> None:
        super().__init__(name=name)
        self._robot_articulation = robot_articulation
        self._interpolation_speed = interpolation_speed

        self._kinematics_solver = mg.lula.LulaKinematicsSolver(
            robot_description_path=os.path.join(self.RMPFLOW_DIR, "robot_descriptor.yaml"),
            urdf_path=os.path.join(self.RMPFLOW_DIR, "ur10e.urdf"),
        )

        self._kinematics_solver.set_default_cspace_seeds([])

        self._articulation_kinematics = mg.ArticulationKinematicsSolver(
            robot_articulation, self._kinematics_solver, "ee_link_robotiq_arg2f_base_link"
        )

        self._joints_subset = self._articulation_kinematics.get_joints_subset()

        robot_position, robot_orientation = robot_articulation.get_world_pose()
        self._kinematics_solver.set_robot_base_pose(
            robot_position=robot_position, robot_orientation=robot_orientation
        )

        self._target_cspace_positions = None
        self._keep_elbow_up = False

    def forward(
        self,
        target_end_effector_position: np.ndarray,
        target_end_effector_orientation: np.ndarray | None = None,
    ) -> ArticulationAction:
        """Compute IK for target pose and interpolate toward solution."""
        action, success = self._articulation_kinematics.compute_inverse_kinematics(
            target_position=target_end_effector_position,
            target_orientation=target_end_effector_orientation,
        )

        if success:
            target = action.joint_positions[action.joint_indices].copy()
            if self._keep_elbow_up:
                # Keep shoulder upright (-180° to -45°)
                target[self._SHOULDER_LIFT_IDX] = np.clip(
                    target[self._SHOULDER_LIFT_IDX], -np.pi, -np.pi / 4
                )
                # Keep elbow up (positive) so arm doesn't cover the workpiece
                target[self._ELBOW_IDX] = np.clip(
                    target[self._ELBOW_IDX], 0.0, np.pi
                )
            self._target_cspace_positions = target

        if self._target_cspace_positions is None:
            num_dof = self._robot_articulation.num_dof
            return ArticulationAction(joint_positions=[None] * num_dof)

        current_cspace = self._joints_subset.get_joint_positions()

        interpolated_cspace = current_cspace + self._interpolation_speed * (
            self._target_cspace_positions - current_cspace
        )

        return self._joints_subset.make_articulation_action(interpolated_cspace, None)

    def compute_end_effector_pose(self):
        """Return current EE (position, orientation_quat[w,x,y,z]) via FK."""
        from isaacsim.core.utils.numpy.rotations import rot_matrices_to_quats

        joint_positions = self._joints_subset.get_joint_positions()
        pos, rot_mat = self._kinematics_solver.compute_forward_kinematics(
            frame_name="ee_link_robotiq_arg2f_base_link",
            joint_positions=joint_positions,
        )
        quat = rot_matrices_to_quats(np.asarray(rot_mat))  # [w, x, y, z]
        return np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64)

    def set_interpolation_speed(self, speed: float) -> None:
        """Update the joint-space interpolation speed."""
        self._interpolation_speed = speed

    def use_elbow_up(self, enabled: bool) -> None:
        """Enable/disable elbow-up constraint to keep arm above workpiece."""
        self._keep_elbow_up = enabled
        if enabled:
            self._kinematics_solver.set_default_cspace_seeds([self.ELBOW_UP_SEED])
        else:
            self._kinematics_solver.set_default_cspace_seeds([])

    def reset(self) -> None:
        super().reset()
        self._target_cspace_positions = None
        robot_position, robot_orientation = self._robot_articulation.get_world_pose()
        self._kinematics_solver.set_robot_base_pose(
            robot_position=robot_position, robot_orientation=robot_orientation
        )
