"""Reference-only robot assembly profiles; import after SimulationApp starts.

The 85 profile uses NVIDIA's assembled UR10e accessory variant, rather than
silently relabelling the older tutorial's 2F-140 geometry as a 2F-85.
"""
from __future__ import annotations

import math
import os
from pathlib import Path

ASSET_SUFFIX_85 = '/Isaac/Robots_Multiphysics/UniversalRobots/ur10e/ur10e.usda'
ROBOT_PRIM = '/ur'
GRIPPER_PRIM_85 = ROBOT_PRIM + '/Robotiq_2F_85'
END_EFFECTOR_85 = GRIPPER_PRIM_85 + '/base_link'
FINGER_JOINT_85 = GRIPPER_PRIM_85 + '/Joints/finger_joint'


def kinematics_directory(profile: str = 'robotiq85') -> Path:
    """Directory accepted as IKLerpController.RMPFLOW_DIR for this profile."""
    if profile != 'robotiq85':
        raise ValueError('Unsupported robot profile')
    return Path(__file__).resolve().parent / 'planning' / 'profiles' / 'ur10e_85'


def create_robot(world, asset_root: str, profile: str = 'robotiq85'):
    """Create a referenced UR10e + real 85 variant; world.reset initializes it.

    The existing Lula controller must separately use kinematics_directory().
    Its legacy EE frame name is retained in the profile URDF. This function
    does not mutate shared controller configuration or start SimulationApp.
    """
    kinematics_directory(profile)  # Reject an unknown profile before mutation.
    import numpy as np
    from pxr import UsdPhysics
    from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
    from isaacsim.robot.manipulators.grippers import ParallelGripper
    from isaacsim.robot.manipulators.manipulators import SingleManipulator

    stage = get_current_stage()
    if stage.GetPrimAtPath(ROBOT_PRIM).IsValid():
        raise RuntimeError('Robot prim already exists; create the profile in an empty scene')
    robot_prim = add_reference_to_stage(os.environ.get('ROBOT_USD',asset_root.rstrip('/') + ASSET_SUFFIX_85), ROBOT_PRIM)

    def select_variant(prim, name, value):
        variants = prim.GetVariantSet(name)
        if value not in variants.GetVariantNames():
            raise RuntimeError(f'Required robot variant unavailable: {name}={value}')
        if not variants.SetVariantSelection(value):
            raise RuntimeError(f'Unable to select robot variant: {name}={value}')

    select_variant(robot_prim, 'Physics', 'physx')
    select_variant(robot_prim, 'Gripper', 'Robotiq_2F_85')
    gripper_prim = stage.GetPrimAtPath(GRIPPER_PRIM_85)
    if not gripper_prim.IsValid():
        raise RuntimeError('Robotiq 2F-85 accessory did not compose')
    select_variant(gripper_prim, 'Physics', 'Physx_Mimic')
    if not stage.GetPrimAtPath(END_EFFECTOR_85).IsValid():
        raise RuntimeError('Robotiq 2F-85 base link unavailable')
    finger_prim = stage.GetPrimAtPath(FINGER_JOINT_85)
    if not finger_prim.IsValid():
        raise RuntimeError('Robotiq 2F-85 finger joint unavailable')
    joint = UsdPhysics.RevoluteJoint(finger_prim)
    lower = float(joint.GetLowerLimitAttr().Get())
    upper = float(joint.GetUpperLimitAttr().Get())
    # USD revolute limits are degrees; ArticulationAction targets are radians.
    closed = min(0.8, math.radians(upper))
    if not math.isfinite(lower + upper) or not lower <= 0 < math.degrees(closed) <= upper:
        raise RuntimeError('Robotiq 2F-85 finger limits do not support the profile')
    gripper = ParallelGripper(
        end_effector_prim_path=END_EFFECTOR_85,
        joint_prim_names=['finger_joint'],
        joint_opened_positions=np.array([0.]),
        joint_closed_positions=np.array([closed]),
        action_deltas=None,
        use_mimic_joints=True,
    )
    robot = world.scene.add(SingleManipulator(
        prim_path=ROBOT_PRIM, name='robot', end_effector_prim_path=END_EFFECTOR_85,
        gripper=gripper,
    ))
    robot.sorter_kinematics_dir = str(kinematics_directory(profile))
    robot.sorter_profile = profile
    robot.sorter_robot_profile = profile
    return robot
