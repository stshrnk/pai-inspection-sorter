# Portions derived from isaac-sim/IsaacSim standalone_examples/api/isaacsim.robot.manipulators/ur10e/tasks/pick_place.py
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for this project. See THIRD_PARTY_NOTICES.md.
"""PickPlace task for UR10e with Robotiq 2F-85 parallel gripper.

Based on the official UR10e standalone example.
Adds source and destination KLT bins to the scene.
"""

from typing import Optional

import isaacsim.core.api.tasks as tasks
import numpy as np
from isaacsim.core.api.scenes.scene import Scene
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.semantics import add_labels
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.robot.manipulators.grippers import ParallelGripper
from isaacsim.robot.manipulators.manipulators import SingleManipulator
from isaacsim.storage.native import get_assets_root_path

from ..config import (
    BIN_ORIENTATION,
    BIN_POSITION,
    BIN_PRIM_PATH,
    BIN_SCALE,
    BIN_SIZE,
    BIN_USD_PATH,
    DEST_BIN_POSITION,
    DEST_BIN_PRIM_PATH,
    DEST_BIN_ORIENTATION,
    DEST_BIN_SCALE,
)


class PickPlace(tasks.PickPlace):
    def __init__(
        self,
        name: str = "ur10e_pick_place",
        cube_initial_position: Optional[np.ndarray] = None,
        cube_initial_orientation: Optional[np.ndarray] = None,
        target_position: Optional[np.ndarray] = None,
        offset: Optional[np.ndarray] = None,
        cube_size: Optional[np.ndarray] = np.array([0.0515, 0.0515, 0.0515]),
    ) -> None:
        tasks.PickPlace.__init__(
            self,
            name=name,
            cube_initial_position=cube_initial_position,
            cube_initial_orientation=cube_initial_orientation,
            target_position=target_position,
            cube_size=cube_size,
            offset=offset,
        )

    def set_up_scene(self, scene: Scene) -> None:
        super().set_up_scene(scene)
        assets_root_path = get_assets_root_path()

        # Source bin
        add_reference_to_stage(
            usd_path=assets_root_path + BIN_USD_PATH,
            prim_path=BIN_PRIM_PATH,
        )
        scene.add(
            SingleRigidPrim(
                prim_path=BIN_PRIM_PATH,
                name="source_bin",
                position=BIN_POSITION,
                orientation=euler_angles_to_quat(BIN_ORIENTATION),
                scale=BIN_SCALE,
            )
        )

        # Destination bin
        add_reference_to_stage(
            usd_path=assets_root_path + BIN_USD_PATH,
            prim_path=DEST_BIN_PRIM_PATH,
        )
        scene.add(
            SingleRigidPrim(
                prim_path=DEST_BIN_PRIM_PATH,
                name="dest_bin",
                position=DEST_BIN_POSITION,
                orientation=euler_angles_to_quat(DEST_BIN_ORIENTATION),
                scale=DEST_BIN_SCALE,
            )
        )

        # Single extra cube offset diagonally (+X and +Y) from the bin center
        # so picking the main cube (at center) does not sweep the arm through
        # red's position. Drop height kept low for fast settling.
        px, py, pz = BIN_POSITION
        extra_cubes = [
            {"name": "red_cube", "color": np.array([1.0, 0.0, 0.0]),
             "pos": np.array([px + 0.15, py + 0.15, pz + 0.15])},
        ]
        stage = get_current_stage()
        for c in extra_cubes:
            scene.add(
                DynamicCuboid(
                    prim_path=f"/World/{c['name']}",
                    name=c["name"],
                    position=c["pos"],
                    size=0.1,
                    scale=np.array([1.0, 0.515, 1.0]),
                    color=c["color"],
                )
            )
            prim = stage.GetPrimAtPath(f"/World/{c['name']}")
            if prim.IsValid():
                add_labels(prim, labels=[c["name"]])

        # Label the main cube as well so perception can pick it out.
        main_prim = stage.GetPrimAtPath("/World/Cube")
        if main_prim.IsValid():
            add_labels(main_prim, labels=["cube"])

    def set_robot(self) -> SingleManipulator:
        assets_root_path = get_assets_root_path()
        asset_path = (
            assets_root_path + "/Isaac/Samples/Rigging/Manipulator/configure_manipulator/ur10e/ur/ur_gripper.usd"
        )
        add_reference_to_stage(usd_path=asset_path, prim_path="/ur")
        gripper = ParallelGripper(
            end_effector_prim_path="/ur/ee_link/robotiq_arg2f_base_link",
            joint_prim_names=["finger_joint"],
            joint_opened_positions=np.array([0]),
            joint_closed_positions=np.array([40]),
            action_deltas=np.array([-40]),
            use_mimic_joints=True,
        )
        manipulator = SingleManipulator(
            prim_path="/ur",
            name="ur10_robot",
            end_effector_prim_path="/ur/ee_link/robotiq_arg2f_base_link",
            gripper=gripper,
        )
        return manipulator
