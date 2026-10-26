# Portions derived from isaac-sim/IsaacSim standalone_examples/api/isaacsim.robot.manipulators/ur10e/controller/pick_place.py
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Modified for this project. See THIRD_PARTY_NOTICES.md.
"""Pick and place controller for UR10e with Robotiq 2F-85 parallel gripper."""

import os
import typing

import numpy as np
import isaacsim.robot.manipulators.controllers as manipulators_controllers

_DEBUG_PHASE = os.environ.get("PICK_PHASE_DEBUG", "0") == "1"
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.grippers import ParallelGripper

from .ik_lerp_controller import IKLerpController
from .rmpflow_controller import RMPFlowController

# cuRobo is imported lazily only when requested (--curobo flag)
# to avoid pulling in heavy GPU dependencies at startup.

# EE downward orientation quaternion [w, x, y, z] used by IK+LERP.
_EE_DOWN_QUAT = euler_angles_to_quat(np.array([0, np.pi, 0]))


def _yaw_down_quat(yaw: float) -> np.ndarray:
    """Downward gripper orientation with an additional yaw about world Z.

    The detector's PCA gives the angle of the object's long axis; rotating
    the gripper by this yaw makes the jaws close along the short axis.
    """
    return euler_angles_to_quat(np.array([0.0, np.pi, float(yaw)]))

# cuRobo's EE frame is tool0 (parent of the gripper) rather than the gripper
# base link Isaac Sim uses for IK+LERP. The two frames are at the same point
# but one is rotated 90° about Z, so the jaws need a matching yaw to close
# along the cube's short (Y) side.
_EE_DOWN_QUAT_CUROBO = euler_angles_to_quat(np.array([0, np.pi, np.pi / 2]))


def _slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between quaternions [w,x,y,z]."""
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)
    dot = np.dot(q0, q1)
    # Take shorter arc
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    # Fall back to LERP when quaternions are very close
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    a = np.sin((1.0 - t) * theta) / sin_theta
    b = np.sin(t * theta) / sin_theta
    return a * q0 + b * q1


class PickPlaceController(manipulators_controllers.PickPlaceController):
    """PickPlace controller using RMPFlow or IK+LERP for UR10e with ParallelGripper."""

    # Number of steps to hold for EE orientation settling after Phase 0
    ORIENT_SETTLE_STEPS = 120

    def __init__(
        self,
        name: str,
        gripper: ParallelGripper,
        robot_articulation: SingleArticulation,
        events_dt: list[float] | None = None,
        use_ik_lerp: bool = False,
        use_curobo: bool = False,
        interpolation_speed: float = 0.05,
    ) -> None:
        if events_dt is None:
            if use_curobo:
                # cuRobo motion phases (0,1,4,5,6,8,9) are trajectory-done
                # driven so their events_dt acts only as an upper bound.
                # Gripper/settle phases (2,3,7) need physical convergence time.
                # Phases: 0=approach, 1=descend_pick, 2=settle, 3=grip_close,
                #         4=lift, 5=transport_xy, 6=descend_place, 7=grip_open,
                #         8=lift_after_place, 9=return
                events_dt = [0.004, 0.004, 0.01, 0.02, 0.004, 0.004, 0.004, 0.02, 0.004, 0.004]
            else:
                # Tuned via experiments/run_sweep.sh: best cycle-time /
                # smoothness trade-off at matching success rate (iklerp_fast).
                events_dt = [0.03, 0.015, 0.5, 0.1, 0.012, 0.024, 0.009, 0.2, 0.015, 0.03]

        # Hybrid layout: when use_curobo is set, keep an IKLerpController alongside
        # cuRobo. Long-distance motions (approach, lift, transport, return) use
        # cuRobo; precise descent phases use IK+LERP for reliable slow approach.
        self._ik_lerp_fallback: IKLerpController | None = None
        if use_curobo:
            from .curobo_controller import CuroboController
            cspace_controller = CuroboController(
                name=name + "_cspace_controller",
                robot_articulation=robot_articulation,
            )
            self._ik_lerp_fallback = IKLerpController(
                name=name + "_ik_lerp_fallback",
                robot_articulation=robot_articulation,
                interpolation_speed=interpolation_speed,
            )
        elif use_ik_lerp:
            cspace_controller = IKLerpController(
                name=name + "_cspace_controller",
                robot_articulation=robot_articulation,
                interpolation_speed=interpolation_speed,
            )
        else:
            cspace_controller = RMPFlowController(
                name=name + "_cspace_controller",
                robot_articulation=robot_articulation,
            )

        self._use_curobo = use_curobo
        self._use_ik_lerp = use_ik_lerp and not use_curobo
        self._orienting_down = False
        self._orient_steps = 0
        # Cartesian interpolation state for Phase 0
        self._phase0_start_pos = None
        self._phase0_start_ori = None
        self._default_lerp_speed = interpolation_speed
        # cuRobo: cache pick pose at cycle start so later phases do not drift
        # with cube physics (esp. after a failed grip that kicks the cube).
        self._curobo_pick_snapshot = None
        self._curobo_prev_event = -1
        # Arm joint positions captured at the end of a motion phase. Used to
        # explicitly hold the arm still during gripper-close/open phases so the
        # fingers can finish closing before the arm starts its next motion
        # (I4M-style: separate arm and gripper control steps).
        self._arm_hold_positions: list[float] | None = None

        super().__init__(
            name=name,
            cspace_controller=cspace_controller,
            gripper=gripper,
            events_dt=events_dt,
            end_effector_initial_height=0.6,
        )

    def _compute_curobo_phase_target(
        self,
        event: int,
        picking_position: np.ndarray,
        placing_position: np.ndarray,
        end_effector_offset: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the EE position+orientation (downward) for each motion phase."""
        pick = np.asarray(picking_position, dtype=np.float64)
        place = np.asarray(placing_position, dtype=np.float64)
        off = np.asarray(end_effector_offset, dtype=np.float64)
        h1 = self._h1  # approach / transport height
        h0 = pick[2]   # cube Z
        pose_above_pick = np.array([pick[0] + off[0], pick[1] + off[1], h1 + off[2]])
        pose_at_pick    = np.array([pick[0] + off[0], pick[1] + off[1], h0 + off[2]])
        pose_above_place = np.array([place[0] + off[0], place[1] + off[1], h1 + off[2]])
        pose_at_place    = np.array([place[0] + off[0], place[1] + off[1], place[2] + off[2]])

        ori = _EE_DOWN_QUAT_CUROBO
        if event == 0:
            return pose_above_pick, ori
        if event == 1:
            return pose_at_pick, ori
        if event == 4:
            return pose_above_pick, ori
        if event == 5:
            return pose_above_place, ori
        if event == 6:
            return pose_at_place, ori
        if event == 8:
            return pose_above_place, ori
        if event == 9:
            return pose_above_pick, ori
        # Gripper / settle phases fall back to the last approach height
        return pose_above_pick, ori

    def forward(
        self,
        picking_position: np.ndarray,
        placing_position: np.ndarray,
        current_joint_positions: np.ndarray,
        end_effector_offset: typing.Optional[np.ndarray] = None,
        end_effector_orientation: typing.Optional[np.ndarray] = None,
        pick_yaw: float = 0.0,
    ) -> ArticulationAction:
        if end_effector_offset is None:
            end_effector_offset = np.array([0, 0, 0])

        # Apply the detected yaw only while approaching / descending onto the
        # pick target (events 0-3). After the gripper has closed, keep a
        # constant orientation for the rest of the cycle so the wrist does
        # not wrap through a singularity during transport.
        use_pick_yaw = (pick_yaw != 0.0) and (self._event <= 3)
        if end_effector_orientation is None and use_pick_yaw:
            end_effector_orientation = _yaw_down_quat(pick_yaw)

        # --- cuRobo mode: plan once per phase to the phase-final target ---
        if self._use_curobo:
            num_dof = current_joint_positions.shape[0]

            if self._pause or self.is_done():
                self.pause()
                return ArticulationAction(joint_positions=[None] * num_dof)

            # Snapshot the pick pose on the first call of a new cycle so that
            # later phases don't track a cube that physics has moved.
            if self._event == 0 and self._curobo_prev_event != 0:
                self._curobo_pick_snapshot = np.asarray(picking_position, dtype=np.float64).copy()
            self._curobo_prev_event = self._event
            pick_for_phase = (
                self._curobo_pick_snapshot
                if self._curobo_pick_snapshot is not None
                else picking_position
            )

            # Simplified single-cube validation: use cuRobo for every motion
            # phase so this run exercises the full cuRobo pipeline.
            motion_phase = self._event in (0, 1, 4, 5, 6, 8, 9)
            curobo_phase = motion_phase
            iklerp_phase = False

            if self._event == 2:
                # Inertia settle — explicitly hold arm at the previous endpoint.
                joints = [None] * num_dof
                if self._arm_hold_positions is not None:
                    for i, v in enumerate(self._arm_hold_positions[:6]):
                        joints[i] = v
                action = ArticulationAction(joint_positions=joints)
            elif self._event == 3:
                action = self._gripper.forward(action="close")
            elif self._event == 7:
                action = self._gripper.forward(action="open")
            elif iklerp_phase and self._ik_lerp_fallback is not None:
                # Slow precise descent with IK+LERP (more reliable for gripping
                # than cuRobo's time-optimal trajectory).
                target_pos, target_ori = self._compute_curobo_phase_target(
                    self._event, pick_for_phase, placing_position, end_effector_offset
                )
                self._ik_lerp_fallback.use_elbow_up(True)
                action = self._ik_lerp_fallback.forward(
                    target_end_effector_position=target_pos,
                    target_end_effector_orientation=target_ori,
                )
                if action.joint_positions is not None:
                    self._arm_hold_positions = [
                        float(v) if v is not None else None
                        for v in action.joint_positions[:6]
                    ]
            else:
                target_pos, target_ori = self._compute_curobo_phase_target(
                    self._event, pick_for_phase, placing_position, end_effector_offset
                )
                action = self._cspace_controller.forward(
                    target_end_effector_position=target_pos,
                    target_end_effector_orientation=target_ori,
                )
                if action.joint_positions is not None:
                    self._arm_hold_positions = [
                        float(v) if v is not None else None
                        for v in action.joint_positions[:6]
                    ]

            # Advance phase when motion trajectory is done (cuRobo motion phases),
            # otherwise use events_dt-based timing. IK+LERP descent phases use
            # events_dt because the LERP convergence is gradual.
            advance_now = False
            if curobo_phase and hasattr(self._cspace_controller, "is_trajectory_done"):
                if self._cspace_controller.is_trajectory_done():
                    advance_now = True
                else:
                    self._t += self._events_dt[self._event]
            else:
                self._t += self._events_dt[self._event]
                if self._t >= 1.0:
                    advance_now = True

            if advance_now:
                if _DEBUG_PHASE:
                    try:
                        j = current_joint_positions
                        print(f"[phase {self._event}->{self._event + 1}] "
                              f"joints[pan,lift,elbow,w1,w2,w3]="
                              f"[{j[0]:.2f},{j[1]:.2f},{j[2]:.2f},"
                              f"{j[3]:.2f},{j[4]:.2f},{j[5]:.2f}] "
                              f"finger={j[6]:.3f}")
                    except Exception:
                        pass
                self._event += 1
                self._t = 0
            return action

        if self._use_ik_lerp and isinstance(self._cspace_controller, IKLerpController):
            # --- Phase 0: Cartesian LERP (position) + SLERP (orientation) ---
            if self._event == 0:
                # Record start EE pose on first call
                if self._phase0_start_pos is None:
                    self._phase0_start_pos, self._phase0_start_ori = (
                        self._cspace_controller.compute_end_effector_pose()
                    )
                    # Use direct IK during Cartesian interpolation
                    self._cspace_controller.set_interpolation_speed(1.0)

                # Goal: approach position above picking target
                goal_pos = np.array([
                    picking_position[0] + end_effector_offset[0],
                    picking_position[1] + end_effector_offset[1],
                    self._h1 + end_effector_offset[2],
                ])
                goal_ori = _yaw_down_quat(pick_yaw) if use_pick_yaw else _EE_DOWN_QUAT

                alpha = self._t

                # Position LERP + Orientation SLERP
                interp_pos = self._phase0_start_pos + alpha * (goal_pos - self._phase0_start_pos)
                interp_ori = _slerp(self._phase0_start_ori, goal_ori, alpha)

                self._cspace_controller.use_elbow_up(True)
                action = self._cspace_controller.forward(
                    target_end_effector_position=interp_pos,
                    target_end_effector_orientation=interp_ori,
                )

                # Advance time (mirrors parent logic)
                self._t += self._events_dt[self._event]
                if self._t >= 1.0:
                    self._event += 1
                    self._t = 0
                    # Restore default interpolation speed for remaining phases
                    self._cspace_controller.set_interpolation_speed(self._default_lerp_speed)
                return action

            # Phase 1+: elbow-up to keep arm clear of workpiece
            self._cspace_controller.use_elbow_up(self._event == 1 or self._orienting_down)

            # After Phase 0 completes, pause and orient EE downward
            if self._event == 1 and not self._orienting_down and self._orient_steps == 0:
                self._orienting_down = True
                self._orient_steps = 0

            if self._orienting_down:
                self._orient_steps += 1
                # Hold position above pick target, orient EE downward (+ yaw)
                hold_position = np.array([
                    picking_position[0] + end_effector_offset[0],
                    picking_position[1] + end_effector_offset[1],
                    self._h1 + end_effector_offset[2],
                ])
                hold_ori = _yaw_down_quat(pick_yaw) if use_pick_yaw else _EE_DOWN_QUAT
                action = self._cspace_controller.forward(
                    target_end_effector_position=hold_position,
                    target_end_effector_orientation=hold_ori,
                )
                if self._orient_steps >= self.ORIENT_SETTLE_STEPS:
                    self._orienting_down = False
                return action

        # For non-cuRobo paths, log phase transitions here for side-by-side
        # comparison with cuRobo runs.
        prev_event = self._event
        action = super().forward(
            picking_position=picking_position,
            placing_position=placing_position,
            current_joint_positions=current_joint_positions,
            end_effector_offset=end_effector_offset,
            end_effector_orientation=end_effector_orientation,
        )
        if _DEBUG_PHASE and self._event != prev_event:
            j = current_joint_positions
            try:
                print(f"[phase {prev_event}->{self._event}] "
                      f"joints[pan,lift,elbow,w1,w2,w3]="
                      f"[{j[0]:.2f},{j[1]:.2f},{j[2]:.2f},"
                      f"{j[3]:.2f},{j[4]:.2f},{j[5]:.2f}] "
                      f"finger={j[6]:.3f}")
            except Exception:
                pass
        return action

    def reset(self, **kwargs) -> None:
        super().reset(**kwargs)
        self._orienting_down = False
        self._orient_steps = 0
        self._phase0_start_pos = None
        self._phase0_start_ori = None
        self._curobo_pick_snapshot = None
        self._curobo_prev_event = -1
        self._arm_hold_positions = None
