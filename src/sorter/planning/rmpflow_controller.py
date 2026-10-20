"""RMPFlow-based motion controller for UR10e."""

import os

import isaacsim.robot_motion.motion_generation as mg
from isaacsim.core.prims import Articulation


class RMPFlowController(mg.MotionPolicyController):
    """Wraps RMPFlow motion policy for UR10e with Robotiq gripper."""

    RMPFLOW_DIR = os.path.join(os.path.dirname(__file__), "rmpflow")

    def __init__(self, name: str, robot_articulation: Articulation, physics_dt: float = 1.0 / 60.0) -> None:
        self.rmpflow = mg.lula.motion_policies.RmpFlow(
            robot_description_path=os.path.join(self.RMPFLOW_DIR, "robot_descriptor.yaml"),
            rmpflow_config_path=os.path.join(self.RMPFLOW_DIR, "ur10e_rmpflow_common.yaml"),
            urdf_path=os.path.join(self.RMPFLOW_DIR, "ur10e.urdf"),
            end_effector_frame_name="ee_link_robotiq_arg2f_base_link",
            maximum_substep_size=0.00334,
        )

        self.articulation_rmp = mg.ArticulationMotionPolicy(robot_articulation, self.rmpflow, physics_dt)

        super().__init__(name=name, articulation_motion_policy=self.articulation_rmp)
        self._default_position, self._default_orientation = (
            self._articulation_motion_policy._robot_articulation.get_world_pose()
        )
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )

    def reset(self):
        super().reset()
        self._motion_policy.set_robot_base_pose(
            robot_position=self._default_position, robot_orientation=self._default_orientation
        )
