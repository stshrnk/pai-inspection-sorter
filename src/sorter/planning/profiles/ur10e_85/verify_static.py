"""Compare profile FK against small, in-memory NVIDIA USDA metadata.

Run with Isaac Python plus omni.usd.libs available on PYTHONPATH/LD_LIBRARY_PATH.
This uses Sdf only: it does not start SimulationApp, compose/render a stage,
load mesh files, or write downloaded assets. --out writes numeric evidence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import math
from pathlib import Path
from urllib.request import urlopen
import xml.etree.ElementTree as ET

import numpy as np
from pxr import Sdf

ROOT = 'https://omniverse-content-staging.s3-us-west-2.amazonaws.com/Assets/Isaac/6.0'
ARM = '/Isaac/Robots_Multiphysics/UniversalRobots/ur10e'
HAND = '/Isaac/Robots_Multiphysics/Robotiq/2F-85'
SOURCES = {
    'arm_base': ARM + '/payloads/base.usda',
    'arm_physics': ARM + '/payloads/Physics/physics.usda',
    'assembly': ARM + '/payloads/Gripper/Robotiq_2F_85.usda',
    'hand_base': HAND + '/payloads/base.usda',
    'hand_robot': HAND + '/payloads/robot.usda',
    'hand_physics': HAND + '/payloads/Physics/Physics.usda',
}
JOINTS = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
          'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']


def quaternion_matrix(quat):
    if hasattr(quat, 'GetReal'):
        quat = [quat.GetReal(), *quat.GetImaginary()]
    w, x, y, z = np.asarray(quat, float) / np.linalg.norm(quat)
    return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                     [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                     [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def rpy_matrix(rpy):
    r, p, y = rpy
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def transform(position, rotation):
    matrix = np.eye(4)
    matrix[:3, :3], matrix[:3, 3] = rotation, position
    return matrix


def attrs(layer, path):
    prim = layer.GetPrimAtPath(path)
    if prim is None:
        raise ValueError(f'Missing metadata prim: {path}')
    return {name: attribute.default for name, attribute in prim.attributes.items()}


def xform(properties):
    position = properties.get('xformOp:translate', (0, 0, 0))
    if 'xformOp:orient' in properties:
        rotation = quaternion_matrix(properties['xformOp:orient'])
    elif 'xformOp:rotateZYX' in properties:
        # USD rotateZYX applies Z, then Y, then X; column-vector form Rx*Ry*Rz.
        angles = np.deg2rad(properties['xformOp:rotateZYX'])
        rotation = rpy_matrix(-angles).T
    else:
        rotation = np.eye(3)
    if not np.allclose(properties.get('xformOp:scale', (1, 1, 1)), (1, 1, 1)):
        raise ValueError('Expected metre-scale, unscaled kinematics')
    return transform(position, rotation)


def joint_transform(properties, side):
    return transform(properties[f'physics:localPos{side}'],
                     quaternion_matrix(properties[f'physics:localRot{side}']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--asset-root', default=ROOT)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()

    def fetch(item):
        key, suffix = item
        url = args.asset_root.rstrip('/') + suffix
        with urlopen(url, timeout=30) as response:
            data = response.read(800000)
        layer = Sdf.Layer.CreateAnonymous()
        if not layer.ImportFromString(data.decode('utf-8')):
            raise ValueError('Unable to parse source metadata')
        return key, layer, {'url': url, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()}

    fetched = list(ThreadPoolExecutor(max_workers=4).map(fetch, SOURCES.items()))
    layers = {key: layer for key, layer, _ in fetched}
    source_metadata = {key: metadata for key, _, metadata in fetched}
    profile = ET.parse(Path(__file__).with_name('ur10e.urdf')).getroot()
    joints, q0, errors = {}, np.eye(4), {}
    for name in JOINTS:
        node = profile.find(f"joint[@name='{name}']")
        position = np.fromstring(node.find('origin').get('xyz'), sep=' ')
        angles = np.fromstring(node.find('origin').get('rpy'), sep=' ')
        urdf_local = transform(position, rpy_matrix(angles))
        properties = attrs(layers['arm_physics'], '/ur10e/joints/' + name)
        usd_local = joint_transform(properties, 0) @ np.linalg.inv(joint_transform(properties, 1))
        if properties['physics:axis'] != 'Z' or node.find('axis').get('xyz') != '0. 0. 1.':
            raise AssertionError('Joint axis mismatch')
        q0 = q0 @ urdf_local
        child = node.find('child').get('link')
        usd_world = xform(attrs(layers['arm_base'], '/ur10e/' + child))
        errors[name] = {
            'local_position_max_abs_m': float(np.max(np.abs(urdf_local[:3, 3] - usd_local[:3, 3]))),
            'local_rotation_matrix_max_abs': float(np.max(np.abs(urdf_local[:3, :3] - usd_local[:3, :3]))),
            'q0_world_position_max_abs_m': float(np.max(np.abs(q0[:3, 3] - usd_world[:3, 3]))),
            'q0_world_rotation_matrix_max_abs': float(np.max(np.abs(q0[:3, :3] - usd_world[:3, :3]))),
        }
        joints[name] = properties
    ft = xform(attrs(layers['arm_base'], '/ur10e/wrist_3_link/ft_frame'))
    fixed = attrs(layers['assembly'], '/ur10e/Robotiq_2F_85/base_link/AssemblerFixedJoint')
    mount_from_fixed = ft @ joint_transform(fixed, 0) @ np.linalg.inv(joint_transform(fixed, 1))
    hand_base = attrs(layers['hand_robot'], '/Robotiq_2F_85/base_link')
    hand_base.update(attrs(layers['hand_base'], '/Robotiq_2F_85/base_link'))
    root_hand = xform(attrs(layers['assembly'], '/ur10e/Robotiq_2F_85'))
    world_hand = root_hand @ xform(hand_base)
    world_wrist = xform(attrs(layers['arm_base'], '/ur10e/wrist_3_link'))
    mount_from_pose = np.linalg.inv(world_wrist) @ world_hand
    node = profile.find("joint[@name='ee_link_root_joint']")
    mount_urdf = transform(np.fromstring(node.find('origin').get('xyz'), sep=' '),
                           rpy_matrix(np.fromstring(node.find('origin').get('rpy'), sep=' ')))
    finger = attrs(layers['hand_physics'], '/Robotiq_2F_85/Joints/finger_joint')
    max_error = max(value for row in errors.values() for value in row.values())
    fixed_error = float(np.max(np.abs(mount_from_fixed - mount_urdf)))
    pose_error = float(np.max(np.abs(mount_from_pose - mount_urdf)))
    result = {
        'verification': 'static USD metadata and zero-pose FK only; no simulation',
        'sources': source_metadata, 'arm_joint_errors': errors,
        'max_arm_error': max_error,
        'mount_from_fixed_joint': mount_from_fixed.tolist(),
        'mount_from_authored_pose': mount_from_pose.tolist(),
        'mount_urdf': mount_urdf.tolist(),
        'mount_fixed_max_abs_error': fixed_error, 'mount_pose_max_abs_error': pose_error,
        'finger_limits_degrees': [float(finger['physics:lowerLimit']), float(finger['physics:upperLimit'])],
        'tolerance': 1e-6,
        'passed': max(max_error, fixed_error, pose_error) < 1e-6,
        'limits': ['Finger contact dynamics and grasp success require a separate Isaac Sim run.',
                   'Profile contains no collision-avoidance geometry.'],
    }
    if args.out:
        args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
