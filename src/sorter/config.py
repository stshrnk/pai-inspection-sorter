"""Configuration for random bin picking simulation."""

import numpy as np

# --- Simulation ---
PHYSICS_DT = 1 / 200
RENDERING_DT = 20 / 200
HEADLESS = False

# --- Robot (UR10e + Robotiq 2F-85) ---
ROBOT_PRIM_PATH = "/World/ur10e"
ROBOT_USD_PATH = "/Isaac/Samples/Rigging/Manipulator/configure_manipulator/ur10e/ur/ur_gripper.usd"
END_EFFECTOR_PRIM = "ee_link/robotiq_arg2f_base_link"
GRIPPER_JOINT_NAMES = ["finger_joint"]
GRIPPER_OPEN = np.array([0])
GRIPPER_CLOSED = np.array([40])
GRIPPER_DELTAS = np.array([-40])
END_EFFECTOR_OFFSET = np.array([0.0, 0.0, 0.20])

# --- Bin (KLT container) ---
BIN_PRIM_PATH = "/World/bin"
BIN_USD_PATH = "/Isaac/Props/KLT_Bin/small_KLT.usd"
BIN_POSITION = np.array([0.4, 0.8, 0.0])
BIN_ORIENTATION = np.array([0, 0, np.pi / 2])  # rotated 90 deg
BIN_SCALE = np.array([2.0, 2.0, 2.0])  # 2x scale
BIN_SIZE = np.array([0.6, 0.8, 0.4])  # approximate KLT bin interior at 2x (m)

# --- Objects to pick (Flip_Stack parts from Nucleus) ---
NUM_OBJECTS = 8
OBJECT_USD_PATHS = [
    "/Isaac/Props/Flip_Stack/large_corner_bracket_physics.usd",
    "/Isaac/Props/Flip_Stack/screw_95_physics.usd",
    "/Isaac/Props/Flip_Stack/screw_99_physics.usd",
    "/Isaac/Props/Flip_Stack/small_corner_bracket_physics.usd",
    "/Isaac/Props/Flip_Stack/t_connector_physics.usd",
]
OBJECT_DROP_HEIGHT = 0.3  # height above bin to drop objects (m)

# --- Safety Fence ---
FENCE_USD_PATH = "/NVIDIA/Assets/ArchVis/Industrial/Railing/MetalFencing_A2.usd"
FENCE_SCALE = np.array([0.01, 0.01, 0.01])  # assets are in cm, scale to meters
FENCE_POSITIONS = [
    # Surround the bin area on 3 sides (leaving robot side open)
    np.array([0.5, -0.5, 0.0]),   # front
    np.array([0.5, 0.5, 0.0]),    # back
    np.array([1.0, 0.0, 0.0]),    # right side
]
FENCE_ORIENTATIONS = [
    np.array([0, 0, 0]),           # front (parallel to X)
    np.array([0, 0, 0]),           # back (parallel to X)
    np.array([0, 0, np.pi / 2]),   # right (rotated 90 deg)
]

# --- 3D Camera (mounted above the source bin) ---
CAMERA_PRIM_PATH = "/World/depth_camera"
CAMERA_POSITION = np.array([0.4, 0.8, 1.3])  # above source bin (looks down into it)
CAMERA_ORIENTATION_EULER = np.array([0, 90, 0])  # looking down (camera Z -> world -Z)
CAMERA_RESOLUTION = (640, 480)
CAMERA_FREQUENCY = 30

# --- Destination Bin ---
DEST_BIN_PRIM_PATH = "/World/dest_bin"
DEST_BIN_POSITION = np.array([-0.4, 0.8, 0.0])
DEST_BIN_ORIENTATION = np.array([0, 0, np.pi / 2])  # same rotation as source bin
DEST_BIN_SCALE = np.array([2.0, 2.0, 2.0])  # 2x scale

# --- Place target (above destination bin floor) ---
PLACE_POSITION = DEST_BIN_POSITION.copy()
