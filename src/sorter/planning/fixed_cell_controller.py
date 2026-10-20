"""Convergence checked, dynamic pick and place for a fixed inspection cell.

The controller only commands robot joints. It never moves, attaches, or reads
the workpiece. Pick/place coordinates are copied once from the cell fixture.
"""

from __future__ import annotations

import os

import numpy as np
from isaacsim.core.utils.rotations import euler_angles_to_quat
from isaacsim.core.utils.types import ArticulationAction

from .ik_lerp_controller import IKLerpController


class _FixedCellIK(IKLerpController):
    """Use the existing Lula configuration without reindexing compact actions.

    ArticulationSubset.make_articulation_action already returns joint positions
    in subset order, alongside articulation joint indices. Reindexing those
    positions by articulation indices would mix the two coordinate orders.
    """

    def __init__(self, robot, physics_dt: float, max_joint_speed: float):
        super().__init__("fixed_cell_ik", robot, physics_dt, interpolation_speed=0.1)
        self._physics_dt = physics_dt
        self.max_joint_speed = max_joint_speed
        self.ik_ok = False
        self._commanded_cspace_positions = None
        self.tracking_limited_elapsed = 0.0
        self.tracking_error = 0.0
        self._kinematics_solver.set_default_cspace_seeds([self.ELBOW_UP_SEED])

    def forward(self, target_end_effector_position, target_end_effector_orientation=None):
        result, self.ik_ok = self._articulation_kinematics.compute_inverse_kinematics(
            target_position=target_end_effector_position,
            target_orientation=target_end_effector_orientation,
        )
        current = np.asarray(self._joints_subset.get_joint_positions(), dtype=float)
        if self.ik_ok:
            target = np.asarray(result.joint_positions, dtype=float)
            if target.shape != current.shape or not np.all(np.isfinite(target)):
                raise RuntimeError("IK returned an invalid joint vector")
            self._target_cspace_positions = target.copy()
        if self._target_cspace_positions is None:
            return self._joints_subset.make_articulation_action(current, None)
        if self._commanded_cspace_positions is None:
            self._commanded_cspace_positions = current.copy()
        tracking = self._commanded_cspace_positions - current
        tracking = np.arctan2(np.sin(tracking), np.cos(tracking))
        self.tracking_error = float(np.max(np.abs(tracking)))
        if self.tracking_error > 0.30:
            # Do not integrate farther into a collision or a stalled drive.
            self.tracking_limited_elapsed += self._physics_dt
            return self._joints_subset.make_articulation_action(self._commanded_cspace_positions.copy(), None)
        self.tracking_limited_elapsed = 0.0
        error = self._target_cspace_positions - self._commanded_cspace_positions
        error = np.arctan2(np.sin(error), np.cos(error))
        limit = self.max_joint_speed * self._physics_dt
        delta = np.clip(0.12 * error, -limit, limit)
        # Integrate from the previous COMMAND, not measured joint position.
        # Otherwise current+tiny_delta yields a weak effective position drive
        # that can settle with a large gravity-induced end-effector error.
        self._commanded_cspace_positions += delta
        return self._joints_subset.make_articulation_action(self._commanded_cspace_positions.copy(), None)


class FixedCellController:
    """A finite, fail-closed pick/place sequence with position feedback.

    Call ``step()`` once per physics tick and apply its returned action to the
    robot. A new instance is required for each workpiece. ``done`` means the
    robot sequence finished; the caller must separately verify tray containment
    and workpiece stillness. ``released`` means opening has been commanded.

    ``EE_OFFSET`` overrides the vertical distance from workpiece center to the
    Robotiq base frame in metres. No simulator label or workpiece state enters
    this controller. The gripper must use radian setpoints and no action deltas.
    """

    PHASES = (
        "approach", "descend", "close", "lift", "transport", "lower", "open", "retreat"
    )

    def __init__(
        self,
        robot,
        pick_position,
        place_position,
        physics_dt: float = 1 / 200,
        ee_offset: float = 0.20,
        clearance: float = 0.24,
        max_joint_speed: float = 0.9,
        phase_timeout: float = 15.0,
    ):
        self.robot = robot
        self.physics_dt = float(physics_dt)
        if self.physics_dt <= 0:
            raise ValueError("physics_dt must be positive")
        self.pick_position = np.array(pick_position, dtype=float, copy=True)
        self.place_position = np.array(place_position, dtype=float, copy=True)
        if self.pick_position.shape != (3,) or self.place_position.shape != (3,):
            raise ValueError("Pick and place positions must each contain three coordinates")
        if not np.all(np.isfinite([self.pick_position, self.place_position])):
            raise ValueError("Pick and place positions must be finite")
        self.ee_offset = float(os.environ.get("EE_OFFSET", ee_offset))
        self.clearance = float(clearance)
        self.phase_timeout = float(phase_timeout)
        self.orientation = euler_angles_to_quat(np.array([0.0, np.pi, 0.0]))
        self._ik = _FixedCellIK(robot, self.physics_dt, float(max_joint_speed))
        self._arm_indices = np.array(self._ik._joints_subset.joint_indices, dtype=int)
        self._finger_indices = np.array(robot.gripper.active_joint_indices, dtype=int)
        self._index = 0
        self._phase_elapsed = 0.0
        self._stable_elapsed = 0.0
        self._ik_failed_elapsed = 0.0
        self._elapsed = 0.0
        self._pose_logged = False
        self._hold_positions = np.array(robot.get_joint_positions(), dtype=float, copy=True)
        self._failed = False
        self._released = False
        self.failure_reason = ""
        self.last_position_error = float("inf")
        self.last_orientation_error = float("inf")
        self.phase_history = []
        pick_ee = self.pick_position + np.array([0.0, 0.0, self.ee_offset])
        place_ee = self.place_position + np.array([0.0, 0.0, self.ee_offset])
        safe_z = max(pick_ee[2], place_ee[2]) + self.clearance
        above_pick = np.array([pick_ee[0], pick_ee[1], safe_z])
        above_place = np.array([place_ee[0], place_ee[1], safe_z])
        self._targets = (
            above_pick, pick_ee, pick_ee, above_pick,
            above_place, place_ee, place_ee, above_place,
        )
        # Gripper holds allow real contact dynamics to settle. Motion holds
        # additionally require measured end-effector convergence.
        self._minimum_times = (1.0, 0.8, 1.0, 0.8, 1.0, 0.8, 0.8, 0.8)

    @property
    def phase(self) -> str:
        if self.failed:
            return "failed"
        return "done" if self.done else self.PHASES[self._index]

    @property
    def done(self) -> bool:
        return self._index >= len(self.PHASES) and not self.failed

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def released(self) -> bool:
        return self._released

    @property
    def elapsed(self) -> float:
        return self._elapsed

    @property
    def phase_elapsed(self) -> float:
        return self._phase_elapsed

    def abort(self, reason: str) -> None:
        """Stop the sequence and hold the robot at its measured joint state."""
        if not self._failed:
            self.failure_reason = str(reason)
            self._hold_positions = np.array(self.robot.get_joint_positions(), dtype=float, copy=True)
            self._failed = True
            print(f"[cell] stop: {self.failure_reason}", flush=True)

    def _hold_action(self) -> ArticulationAction:
        return ArticulationAction(joint_positions=self._hold_positions.copy())

    def _with_gripper(self, arm_action, command: str) -> ArticulationAction:
        finger_action = self.robot.gripper.forward(action=command)
        positions = list(finger_action.joint_positions)
        if arm_action.joint_positions is not None:
            indices = arm_action.joint_indices
            if indices is None:
                indices = range(len(arm_action.joint_positions))
            for index, value in zip(indices, arm_action.joint_positions):
                if value is not None:
                    positions[int(index)] = float(value)
        # All moving/holding phases include an explicit gripper target so it
        # remains closed during lift and transit.
        return ArticulationAction(joint_positions=positions)

    def _pose_errors(self, target) -> tuple[float, float]:
        position, orientation = self.robot.end_effector.get_world_pose()
        position = np.asarray(position, dtype=float)
        orientation = np.asarray(orientation, dtype=float)
        orientation /= np.linalg.norm(orientation)
        if not self._pose_logged and (self._phase_elapsed >= 2.0 or self._phase_elapsed <= self.physics_dt):
            fk_position, fk_orientation = self._ik.compute_end_effector_pose()
            print(
                f"[cell-pose] phase={self.phase} t={self._phase_elapsed:.3f}s "
                f"actual_position={np.round(position,5).tolist()} "
                f"actual_quat={np.round(orientation,5).tolist()} "
                f"fk_position={np.round(fk_position,5).tolist()} "
                f"fk_quat={np.round(fk_orientation,5).tolist()} "
                f"target_quat={np.round(self.orientation,5).tolist()}",
                flush=True,
            )
            current=np.asarray(self._ik._joints_subset.get_joint_positions(),dtype=float)
            target_q=self._ik._target_cspace_positions
            command_q=self._ik._commanded_cspace_positions
            velocity=np.asarray(self.robot.get_joint_velocities(),dtype=float)[self._arm_indices]
            print(
                f"[cell-joints] phase={self.phase} "
                f"current={np.round(current,4).tolist()} "
                f"ik_target={None if target_q is None else np.round(target_q,4).tolist()} "
                f"command={None if command_q is None else np.round(command_q,4).tolist()} "
                f"velocity={np.round(velocity,4).tolist()} "
                f"tracking_error={self._ik.tracking_error:.4f}rad",
                flush=True,
            )
            self._pose_logged = self._phase_elapsed >= 2.0
        cosine = abs(float(np.dot(orientation, self.orientation)))
        return float(np.linalg.norm(position - target)), float(2 * np.arccos(np.clip(cosine, 0, 1)))

    def _advance(self) -> None:
        name = self.PHASES[self._index]
        self.phase_history.append({
            "phase": name,
            "simulation_seconds": self._phase_elapsed,
            "position_error_m": self.last_position_error,
            "orientation_error_rad": self.last_orientation_error,
        })
        print(
            f"[cell] {name} complete t={self._phase_elapsed:.2f}s "
            f"position_error={self.last_position_error:.4f}m "
            f"orientation_error={self.last_orientation_error:.3f}rad",
            flush=True,
        )
        self._hold_positions = np.array(self.robot.get_joint_positions(), dtype=float, copy=True)
        self._index += 1
        self._phase_elapsed = self._stable_elapsed = self._ik_failed_elapsed = 0.0
        self._pose_logged = False

    def step(self) -> ArticulationAction:
        if self.failed or self.done:
            return self._hold_action()
        self._elapsed += self.physics_dt
        self._phase_elapsed += self.physics_dt
        phase = self.phase
        target = self._targets[self._index]
        self.last_position_error, self.last_orientation_error = self._pose_errors(target)
        if not np.isfinite(self.last_position_error + self.last_orientation_error):
            self.abort("non-finite robot pose")
            return self._hold_action()
        if self._phase_elapsed > self.phase_timeout:
            self.abort(
                f"{phase} timeout: position_error={self.last_position_error:.4f}m "
                f"orientation_error={self.last_orientation_error:.3f}rad"
            )
            return self._hold_action()
        closing = 2 <= self._index < 6
        command = "close" if closing else "open"
        if phase in ("close", "open"):
            # Keep only arm joints from the snapshot. The gripper target below
            # must not be overwritten by the full articulation hold vector.
            arm = ArticulationAction(
                joint_positions=self._hold_positions[self._arm_indices],
                joint_indices=self._arm_indices,
            )
            if phase == "open":
                self._released = True
            pose_ok = self.last_position_error < 0.015 and self.last_orientation_error < 0.15
            if not pose_ok and self._phase_elapsed > 0.35:
                self.abort(f"arm moved during {phase}")
                return self._hold_action()
            finger_speed = np.asarray(self.robot.get_joint_velocities(), dtype=float)[self._finger_indices]
            settled = bool(np.max(np.abs(finger_speed)) < 0.15)
            if phase == "open":
                finger_position = np.asarray(self.robot.get_joint_positions(), dtype=float)[self._finger_indices]
                settled = settled and bool(np.max(np.abs(finger_position)) < 0.08)
        else:
            arm = self._ik.forward(target, self.orientation)
            if self._ik.tracking_limited_elapsed > 0.5:
                self.abort(f"joint tracking stalled in {phase}: error={self._ik.tracking_error:.3f}rad")
                return self._hold_action()
            if not self._ik.ik_ok:
                self._ik_failed_elapsed += self.physics_dt
                if self._ik_failed_elapsed > 1.0:
                    self.abort(f"IK did not converge in {phase}")
                    return self._hold_action()
            else:
                self._ik_failed_elapsed = 0.0
            tolerance = 0.006 if phase in ("descend", "lower") else 0.012
            settled = self.last_position_error < tolerance and self.last_orientation_error < 0.10
        result = self._with_gripper(arm, command)
        self._stable_elapsed = self._stable_elapsed + self.physics_dt if settled else 0.0
        if self._phase_elapsed >= self._minimum_times[self._index] and self._stable_elapsed >= 0.12:
            self._advance()
        return result
