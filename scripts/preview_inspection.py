"""Recapture demo conditions in the robot cell with frozen, RGB-only inference.

This preview performs no robot transport and is excluded from benchmarks.
Run with the same Isaac Python interpreter used for the main cell.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sorter.measurement import atomic_write_json


APPEARANCE_DEFAULTS = {
    'gray': .65, 'dome': 65., 'key': 180., 'patch_x': 0., 'patch_y': 0.,
    'patch_w': .032, 'patch_h': .021, 'patch_gray': .06,
}
INITIAL_JOINTS = {
    'shoulder_pan_joint': 0., 'shoulder_lift_joint': -1.57,
    'elbow_joint': 1.57, 'wrist_1_joint': -1.57,
    'wrist_2_joint': -1.57, 'wrist_3_joint': 0., 'finger_joint': 0.,
}


def read_conditions(path: Path) -> list[dict]:
    rows = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(rows, list) or not rows:
        raise ValueError('Conditions must be a nonempty JSON list')
    for row in rows:
        if not isinstance(row, dict) or row.get('truth') not in ('good', 'defect'):
            raise ValueError('Each condition needs good/defect truth for appearance generation')
        if 'note' in row and not isinstance(row['note'], str):
            raise ValueError('Optional condition note must be text')
        condition = row.get('condition')
        if not isinstance(condition, dict):
            raise ValueError('Each row needs an appearance condition dictionary')
        for key, value in condition.items():
            if (key not in APPEARANCE_DEFAULTS or isinstance(value, bool)
                    or not isinstance(value, (int, float)) or not math.isfinite(value)):
                raise ValueError('Invalid appearance condition field')
        if any(condition.get(key, 1.) <= 0 for key in
               ('gray', 'dome', 'key', 'patch_w', 'patch_h', 'patch_gray')):
            raise ValueError('Appearance intensities and dimensions must be positive')
    return rows


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--out-dir', type=Path, default=Path('artifacts/preview'))
    parser.add_argument('--conditions', type=Path, default=Path('config/demo_conditions.json'))
    parser.add_argument('--model', type=Path, default=Path('models/inspection/classifier.pt'))
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    # Every preview owns a new directory. Never overwrite an earlier image,
    # prediction, or diagnostic, including a previous failed preview.
    try:
        args.out_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print('[preview-error] Output directory already exists; choose a new one.', file=sys.stderr)
        return 2
    except OSError as error:
        print(f'[preview-error] output_initialization: {type(error).__name__}', file=sys.stderr)
        return 2

    output_file = args.out_dir / 'predictions.json'
    payload = {
        'version': 1,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'status': 'starting',
        'purpose': 'demo_condition_recapture_preview',
        'benchmark_included': False,
        'transport_attempted': False,
        'transport_result': 'not_attempted',
        'model_frozen': True,
        'classification_inputs': 'RGB pixels only',
        'config': {
            'robot_profile': 'robotiq85', 'renderer': 'RaytracedLighting',
            'anti_aliasing': 0, 'physics_dt': 1 / 200, 'rendering_dt': 1 / 20,
            'warmup_physics_steps': 80, 'warmup_render_every': 20,
            'capture_render_frames': 60,
            'conditions_file': args.conditions.name, 'checkpoint_file': args.model.name,
            'initial_joint_positions_rad': INITIAL_JOINTS,
        },
        'predictions': [],
    }
    app = None
    stage = 'initial_output'
    exit_code = 1

    def persist():
        atomic_write_json(output_file, payload)

    try:
        persist()
        stage = 'conditions_validation'
        rows = read_conditions(args.conditions)
        payload['config']['conditions_sha256'] = hashlib.sha256(args.conditions.read_bytes()).hexdigest()
        payload['config']['condition_count'] = len(rows)
        stage = 'checkpoint_validation'
        if not args.model.is_file():
            raise FileNotFoundError('Frozen checkpoint is unavailable')
        checkpoint_hash = hashlib.sha256(args.model.read_bytes()).hexdigest()
        payload['config']['model_sha256'] = checkpoint_hash
        persist()

        stage = 'simulation_startup'
        from isaacsim import SimulationApp
        app = SimulationApp({
            'headless': args.headless, 'width': 960, 'height': 720,
            'renderer': 'RaytracedLighting', 'anti_aliasing': 0,
        })
        import numpy as np
        from PIL import Image
        from isaacsim.core.api import World
        from isaacsim.core.utils.types import ArticulationAction
        from src.sorter.cell_scene import (
            SOURCE, create_scene, initialize_camera, set_appearance, capture_rgb,
        )
        from src.sorter.vision import SorterClassifier

        stage = 'scene_initialization'
        world = World(stage_units_in_meters=1., physics_dt=1 / 200, rendering_dt=1 / 20)
        scene = create_scene(world, robot_profile='robotiq85')
        world.reset()
        robot, part = scene['robot'], scene['part']
        initial = robot.get_joint_positions().copy()
        for name, value in INITIAL_JOINTS.items():
            initial[robot.get_dof_index(name)] = value
        initialize_camera(scene)
        stage = 'frozen_classifier_initialization'
        classifier = SorterClassifier(args.model)
        payload['status'] = 'running'
        persist()

        for index, source in enumerate(rows):
            condition = dict(APPEARANCE_DEFAULTS)
            condition.update(source['condition'])
            record = {
                'index': index + 1, 'truth': source['truth'],
                'condition': condition, 'note': source.get('note', ''),
                'status': 'starting', 'image': None, 'image_quality': None,
                'decision': None, 'transport_attempted': False,
                'benchmark_included': False,
            }
            payload['predictions'].append(record)
            stage = f'condition_{index + 1}_reset'
            # Reproduce main.py's fixed supply and robot starting pose. There
            # are no motion controllers and no robot transport in this tool.
            robot.set_joint_positions(initial.copy())
            robot.set_joint_velocities(np.zeros(robot.num_dof))
            robot.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=initial.copy()))
            part.set_world_pose(position=SOURCE.copy(), orientation=np.array([1., 0, 0, 0]))
            part.set_linear_velocity(np.zeros(3))
            part.set_angular_velocity(np.zeros(3))
            set_appearance(scene, condition, source['truth'] == 'defect')
            print('[preview-condition] ' + json.dumps({'index': index + 1, 'condition': condition}), flush=True)
            for tick in range(80):
                if not app.is_running() or not world.is_playing():
                    raise RuntimeError('Simulation stopped before inspection')
                before = world.current_time_step_index
                world.step(render=False, update_fabric=True)
                if world.current_time_step_index != before + 1:
                    raise RuntimeError('Physics did not advance by exactly one step')
                if (tick + 1) % 20 == 0:
                    world.render()

            stage = f'condition_{index + 1}_capture'
            image = np.asarray(capture_rgb(world, scene['camera'], frames=60))
            if image.ndim != 3 or image.shape[2] < 3 or min(image.shape[:2]) < 4 or not np.isfinite(image).all():
                raise RuntimeError('Inspection camera returned an invalid image')
            image = image[:, :, :3].astype(np.uint8, copy=True)
            record['image_quality'] = {
                'shape': list(image.shape), 'min': float(image.min()),
                'max': float(image.max()), 'std': float(image.std()),
                'spatial_std_max': float(image.astype(np.float64).std(axis=(0, 1)).max()),
            }
            record['image'] = f'inspection_{index:03d}.png'
            Image.fromarray(image).save(args.out_dir / record['image'])
            if record['image_quality']['spatial_std_max'] <= 1e-6:
                raise RuntimeError('Inspection image is spatially constant')

            stage = f'condition_{index + 1}_classification'
            # Only the captured pixel array enters inference. Generation truth,
            # condition metadata and filenames are never supplied to the model.
            decision = classifier.predict(image)
            if decision.get('label') not in ('good', 'defect', 'review') or decision.get('method') != 'cnn':
                raise ValueError('Classifier returned an invalid label or method')
            if not all(isinstance(decision.get(key), (float, int))
                       and math.isfinite(decision[key]) and 0 <= decision[key] <= 1
                       for key in ('defect_score', 'confidence', 'review_threshold')):
                raise ValueError('Classifier returned invalid scores')
            record.update(status='classified', decision=decision)
            persist()
            print('[preview] ' + json.dumps({'index': index + 1, 'decision': decision}), flush=True)

        stage = 'checkpoint_integrity_check'
        if hashlib.sha256(args.model.read_bytes()).hexdigest() != checkpoint_hash:
            raise RuntimeError('Frozen checkpoint changed during preview')
        payload['checkpoint_unchanged'] = True
        payload['status'] = 'completed'
        persist()
        print(f'[preview] completed {len(rows)} image classifications; transport not attempted.', flush=True)
        exit_code = 0
    except Exception as error:
        # Exception paths can contain personal filesystem details. Persist a
        # stage and exception type sufficient to locate the failed operation.
        failure = {'stage': stage, 'type': type(error).__name__}
        payload.update(status='failed', error=failure)
        if payload['predictions'] and payload['predictions'][-1]['status'] != 'classified':
            payload['predictions'][-1]['status'] = 'failed'
        try:
            persist()
            (args.out_dir / 'error.log').write_text(json.dumps(failure) + '\n', encoding='utf-8')
        except OSError:
            print('[preview-error] Unable to persist failure diagnostics.', file=sys.stderr)
        print('[preview-error] ' + json.dumps(failure), file=sys.stderr, flush=True)
    finally:
        if app is not None:
            try:
                app.close()
            except Exception as error:
                exit_code = 1
                failure = {'stage': 'simulation_shutdown', 'type': type(error).__name__}
                payload.update(status='failed', error=failure)
                try:
                    persist()
                    (args.out_dir / 'error.log').write_text(json.dumps(failure) + '\n', encoding='utf-8')
                except OSError:
                    pass
                print('[preview-error] ' + json.dumps(failure), file=sys.stderr, flush=True)
    return exit_code


if __name__ == '__main__':
    sys.exit(main())
