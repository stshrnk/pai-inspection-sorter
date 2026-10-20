"""CPU-only controller checks; Isaac imports are stubbed in an isolated context."""
from contextlib import redirect_stdout
import importlib.util
import io
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np


class Action:
    def __init__(self, joint_positions=None, joint_indices=None):
        self.joint_positions = joint_positions
        self.joint_indices = joint_indices


def quaternion(euler):
    roll, pitch, yaw = np.asarray(euler) / 2
    cr, cp, cy = np.cos([roll, pitch, yaw])
    sr, sp, sy = np.sin([roll, pitch, yaw])
    return np.array([cr * cp * cy + sr * sp * sy,
                     sr * cp * cy - cr * sp * sy,
                     cr * sp * cy + sr * cp * sy,
                     cr * cp * sy - sr * sp * cy])


def load_controller():
    """Load a fresh module without leaving fake Isaac modules in sys.modules."""
    names = ['isaacsim', 'isaacsim.core', 'isaacsim.core.utils',
             'isaacsim.core.utils.rotations', 'isaacsim.core.utils.types',
             '_sorter_controller_test', '_sorter_controller_test.ik_lerp_controller']
    stubs = {name: ModuleType(name) for name in names}
    for module in stubs.values():
        module.__path__ = []
    stubs['isaacsim.core.utils.rotations'].euler_angles_to_quat = quaternion
    stubs['isaacsim.core.utils.types'].ArticulationAction = Action
    stubs['_sorter_controller_test.ik_lerp_controller'].IKLerpController = type(
        'UnusedBaseIK', (), {})
    source = Path(__file__).resolve().parents[1] / 'src/sorter/planning/fixed_cell_controller.py'
    spec = importlib.util.spec_from_file_location('_sorter_controller_test.controller', source)
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, stubs):
        spec.loader.exec_module(module)
    return module


class FakeRobot:
    """Robot observations and command sink, with no scene/teleport operations."""
    arm_indices = np.array([0, 2, 3, 4, 5, 6])
    finger_index = 1

    def __init__(self):
        self.q = np.zeros(7)
        self.velocity = np.zeros(7)
        self.position = np.array([0.4, 0.55, 0.5])
        self.orientation = quaternion([0, np.pi, 0])
        self.last_waypoint = None
        self.end_effector = SimpleNamespace(
            get_world_pose=lambda: (self.position.copy(), self.orientation.copy()))
        self.gripper = SimpleNamespace(
            active_joint_indices=[self.finger_index],
            joint_closed_positions=np.array([0.8]),
            joint_opened_positions=np.array([0.0]),
            forward=self.grip_action)
        self.dof_properties = np.zeros(7, dtype=[
            ('lower', float), ('upper', float), ('stiffness', float),
            ('damping', float), ('maxEffort', float)])
        self.dof_properties['upper'] = np.pi
        self.dof_properties['upper'][self.finger_index] = np.deg2rad(47)

    def grip_action(self, action):
        q = [None] * 7
        q[self.finger_index] = 0.8 if action == 'close' else 0.0
        return Action(q)

    def get_joint_positions(self):
        return self.q.copy()

    def get_joint_velocities(self):
        return self.velocity.copy()

    def apply(self, action, finger_contact=False):
        for index, value in enumerate(action.joint_positions):
            if value is not None:
                self.q[index] = value
        if finger_contact:
            self.q[self.finger_index] = min(self.q[self.finger_index], 0.25)
        if self.last_waypoint is not None:
            self.position = self.last_waypoint.copy()

    def set_world_pose(self, *args, **kwargs):
        raise AssertionError('Controller must not teleport a body')

    def set_joint_positions(self, *args, **kwargs):
        raise AssertionError('Controller must only return articulation actions')

    @property
    def scene(self):
        raise AssertionError('Controller must not access or manipulate workpieces')


class FakeCartesianIK:
    def __init__(self, robot, *args):
        self.robot = robot
        self._joints_subset = SimpleNamespace(
            joint_indices=robot.arm_indices,
            get_joint_positions=lambda: robot.q[robot.arm_indices].copy())
        self._target_cspace_positions = None
        self._commanded_cspace_positions = None
        self.tracking_limited_elapsed = 0.0
        self.tracking_error = 0.0
        self.ik_ok = True

    def forward(self, target, orientation):
        self.robot.last_waypoint = np.array(target, copy=True)
        return Action(np.zeros(6), self.robot.arm_indices)

    def compute_end_effector_pose(self):
        return self.robot.end_effector.get_world_pose()


class FixedCellControllerTests(unittest.TestCase):
    def setUp(self):
        quiet = redirect_stdout(io.StringIO())
        quiet.__enter__()
        self.addCleanup(quiet.__exit__, None, None, None)
        self.module = load_controller()

    def controller(self, **kwargs):
        self.module._FixedCellIK = FakeCartesianIK
        robot = FakeRobot()
        control = self.module.FixedCellController(
            robot, [0.4, 0.55, 0.045], [-0.4, 0.55, 0.045],
            physics_dt=0.05, ee_offset=0.13, **kwargs)
        return robot, control

    def test_stub_imports_do_not_escape(self):
        keys = ['isaacsim.core.utils.types', '_sorter_controller_test.ik_lerp_controller']
        before = {key: sys.modules.get(key) for key in keys}
        load_controller()
        self.assertEqual(before, {key: sys.modules.get(key) for key in keys})

    def test_cartesian_descent_is_vertical_and_speed_bounded(self):
        robot, control = self.controller()
        robot.position = np.array([0.4, 0.55, 0.415])
        target = np.array([0.4, 0.55, 0.175])
        points = []
        for step in range(1, 140):
            control._phase_elapsed = step * 0.05
            points.append(control._cartesian_waypoint('descend', target))
            robot.position[:] = 99  # Must not change the captured start.
        points = np.asarray(points)
        np.testing.assert_allclose(points[:, :2], np.tile([0.4, 0.55], (len(points), 1)))
        np.testing.assert_allclose(points[-1], target)
        np.testing.assert_allclose(control._path_start, [0.4, 0.55, 0.415])
        self.assertAlmostEqual(control._path_duration, 6.0)
        self.assertLessEqual(np.linalg.norm(np.diff(points, axis=0), axis=1).max(), 0.04 * 0.05 + 1e-12)

    def make_integrator(self):
        actual = np.zeros(6)
        ik = self.module._FixedCellIK.__new__(self.module._FixedCellIK)
        ik._joints_subset = SimpleNamespace(
            get_joint_positions=lambda: actual.copy(),
            make_articulation_action=lambda q, v: Action(q.copy()))
        ik._articulation_kinematics = SimpleNamespace(
            compute_inverse_kinematics=lambda **kwargs: (Action(np.ones(6)), True))
        ik._target_cspace_positions = None
        ik._commanded_cspace_positions = None
        ik.max_joint_speed = 0.9
        ik._physics_dt = 0.005
        ik.tracking_limited_elapsed = 0.0
        ik.tracking_error = 0.0
        return actual, ik

    def test_joint_commands_integrate_through_servo_lag(self):
        actual, ik = self.make_integrator()
        commands = []
        for _ in range(400):
            command = ik.forward(np.zeros(3)).joint_positions
            commands.append(command.copy())
            actual += 0.2 * (command - actual)
        np.testing.assert_allclose(actual, np.ones(6), atol=1e-6)
        self.assertLessEqual(np.abs(np.diff(commands, axis=0)).max(), 0.9 * 0.005 + 1e-12)

    def test_tracking_guard_stops_integrating_into_stall(self):
        actual, ik = self.make_integrator()
        for _ in range(300):
            ik.forward(np.zeros(3))
        np.testing.assert_array_equal(actual, np.zeros(6))
        self.assertGreater(ik._commanded_cspace_positions.max(), 0.30)
        self.assertLess(ik._commanded_cspace_positions.max(), 0.31)
        self.assertGreater(ik.tracking_limited_elapsed, 0.5)
        robot, control = self.controller()
        control._ik.tracking_limited_elapsed = 0.6
        control._ik.tracking_error = 0.31
        control.step()
        self.assertTrue(control.failed)
        self.assertIn('tracking stalled', control.failure_reason)

    def test_close_can_advance_despite_loaded_finger_velocity(self):
        robot, control = self.controller()
        control._index = 2
        robot.position = control._targets[2].copy()
        commands = []
        for _ in range(100):
            robot.velocity[robot.finger_index] = 3.0
            action = control.step()
            commands.append(action.joint_positions[robot.finger_index])
            robot.apply(action, finger_contact=True)
            if control.phase != 'close':
                break
        self.assertEqual(control.phase, 'lift')
        self.assertFalse(control.failed)
        self.assertGreaterEqual(control.elapsed, 2.0)
        self.assertAlmostEqual(commands[-1], 0.8)
        self.assertLessEqual(np.abs(np.diff(commands)).max(), 0.4 * 0.05 + 1e-12)
        # Non-contiguous arm indices must remain held while the finger moves.
        np.testing.assert_array_equal(robot.q[robot.arm_indices], np.zeros(6))

    def test_open_requires_actual_open_position_and_velocity(self):
        robot, control = self.controller()
        control._index = 6
        robot.position = control._targets[6].copy()
        robot.q[robot.finger_index] = 0.25
        for _ in range(30):
            control.step()
        self.assertEqual(control.phase, 'open')
        robot.q[robot.finger_index] = 0.0
        robot.velocity[robot.finger_index] = 1.0
        for _ in range(5):
            control.step()
        self.assertEqual(control.phase, 'open')
        robot.velocity[robot.finger_index] = 0.0
        for _ in range(4):
            control.step()
        self.assertEqual(control.phase, 'retreat')
        self.assertTrue(control.released)

    def test_phase_timeout_fails_without_reporting_completion(self):
        robot, control = self.controller(phase_timeout=0.2)
        robot.position = np.array([0.0, 0.0, 0.8])
        for _ in range(5):
            control.step()
        self.assertTrue(control.failed)
        self.assertFalse(control.done)
        self.assertFalse(control.released)
        self.assertIn('timeout', control.failure_reason)

    def test_completed_action_sequence_does_not_claim_transfer_success(self):
        robot, control = self.controller()
        for _ in range(1500):
            action = control.step()
            robot.apply(action)
            if control.done or control.failed:
                break
        self.assertTrue(control.done)
        self.assertFalse(control.failed)
        self.assertTrue(control.released)
        self.assertEqual(len(control.phase_history), 8)
        self.assertFalse(hasattr(control, 'success'))


if __name__ == '__main__':
    unittest.main()
