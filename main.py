"""Image-only inspection and physically simulated UR10e pick/place cell."""
from __future__ import annotations

import argparse
import faulthandler
import json
import math
import os
from pathlib import Path
import sys
import time
import uuid

from src.sorter.measurement import InspectionLog, atomic_write_json

sys.stdout.reconfigure(line_buffering=True)
faulthandler.enable()

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--headless', action='store_true')
p.add_argument('--trials', type=int, default=2)
p.add_argument('--model', default='models/inspection/classifier.pt')
p.add_argument('--sequence', default='good,defect', help='Appearance sequence; truth is used only by evaluation')
p.add_argument('--conditions', type=Path, help='Appearance plan JSON; no destinations are supplied to the classifier')
p.add_argument('--motion-only', choices=['good', 'defect', 'review'], help='Fixed destination for mechanical tests; bypasses AI')
p.add_argument('--output-dir', default='artifacts/run')
p.add_argument('--record-dir', help='Save camera-only frames, no desktop or terminal')
p.add_argument('--record-trials',type=int,help='Record only this many initial trials')
p.add_argument('--record-every', type=int, default=10, help='Physics steps per frame (dt=0.005)')
p.add_argument('--render-every', type=int, default=20, help='Physics steps per non-recording render')
p.add_argument('--continue-after-failure', action='store_true', help='Evaluation only: reset each failed trial; default stops')
p.add_argument('--phase-timeout', type=float, default=15.)
p.add_argument('--robot-profile', choices=['robotiq85','legacy140'], default='robotiq85')
p.add_argument('--ee-offset', type=float, default=.13)
p.add_argument('--joint-speed', type=float, default=.9)
a = p.parse_args()
if a.trials < 1 or a.record_every < 1 or a.render_every < 1:
    p.error('trials and render/record intervals must be positive')
if a.record_trials is not None and a.record_trials<1:
    p.error('record-trials must be positive')
if not all(math.isfinite(value) and value > 0 for value in (a.phase_timeout, a.ee_offset, a.joint_speed)):
    p.error('phase-timeout, ee-offset and joint-speed must be finite and positive')
sequence = a.sequence.split(',')
if any(value not in ('good', 'defect') for value in sequence):
    p.error('sequence must contain good/defect')
appearance_plan=None
if a.conditions:
    try:
        appearance_plan=json.loads(a.conditions.read_text())
        if not isinstance(appearance_plan,list) or not appearance_plan:
            raise ValueError('empty conditions')
        allowed={'gray','dome','key','patch_x','patch_y','patch_w','patch_h','patch_gray'}
        for row in appearance_plan:
            if row.get('truth') not in ('good','defect') or not isinstance(row.get('condition'),dict):
                raise ValueError('invalid condition row')
            for key,value in row['condition'].items():
                if key not in allowed or not isinstance(value,(int,float)) or not math.isfinite(value):
                    raise ValueError('invalid condition field')
            if any(row['condition'].get(key,1)<=0 for key in ('gray','dome','key','patch_w','patch_h','patch_gray')):
                raise ValueError('nonpositive condition')
    except (OSError,ValueError,TypeError,AttributeError):
        p.error('conditions must be a nonempty JSON list of truth and numeric appearance conditions')



def _safe_error(stage, error):
    """Persist useful error context without exception paths or personal details."""
    return f'{stage}: {type(error).__name__}'


def main():
    output = Path(a.output_dir)
    metrics_path = Path(os.environ.get('METRICS_OUT', str(output / 'metrics.json')))
    # Logs append, while metrics and images use fixed names. Refuse reuse so a
    # previous inspection can never silently point at a newly overwritten image.
    protected = [output / name for name in ('metrics.json', 'inspections.csv', 'inspections.jsonl')]
    if metrics_path.exists() or any(path.exists() for path in protected) or any(output.glob('inspection_*.png')):
        print('[error] Output already contains inspection results; use a new output directory.', file=sys.stderr)
        return 2
    if a.record_dir and any(Path(a.record_dir).glob('frame_*.png')):
        print('[error] Recording frames already exist; use a new recording directory.', file=sys.stderr)
        return 2

    output.mkdir(parents=True, exist_ok=True)
    logger = InspectionLog(output)
    run_id = uuid.uuid4().hex
    cycles = []
    payload = {
        'run_id': run_id,
        'config': {
            'controller': 'fixed-cell IK feedback', 'motion_only': a.motion_only,
            'robot_profile': a.robot_profile,
            'physics_dt': 1 / 200, 'ee_offset': a.ee_offset,
            'joint_speed': a.joint_speed, 'sequence': sequence, 'trials_requested': a.trials,
            'stop_on_failure': not a.continue_after_failure,
            'renderer': 'RaytracedLighting', 'anti_aliasing': 0,
            'record_every': a.record_every, 'record_fps': 200 / a.record_every,
            'render_every':a.render_every,'record_trials':a.record_trials,
        },
        'cycles': cycles,
        'appearance_plan':appearance_plan,
    }

    def persist(status='running', stop_reason=None):
        n = len(cycles)
        successes = sum(record['success'] for record in cycles)
        ai_decisions = [record for record in cycles if record.get('decision', {}).get('method') == 'cnn']
        accepted = [record for record in ai_decisions if record.get('destination') in ('good', 'defect')]
        payload['status'] = status
        if stop_reason is not None:
            payload['stop_reason'] = stop_reason
        payload['summary'] = {
            'cycles': n, 'success': successes, 'success_rate': successes / n if n else None,
            'transfer_success': sum(record['transfer_success'] for record in cycles),
            'review': sum(record.get('destination') == 'review' for record in cycles),
            'ai_decisions': len(ai_decisions), 'accepted_decisions': len(accepted),
            'classification_coverage': len(accepted) / len(ai_decisions) if ai_decisions else None,
            'accepted_route_accuracy': sum(record['correct_route'] for record in accepted) / len(accepted) if accepted else None,
        }
        atomic_write_json(metrics_path, payload)

    app = None
    stage = 'initial_metrics'
    exit_code = 1
    try:
        # A durable, empty run exists before model loading and simulator startup.
        persist('starting')
        stage = 'configuration'
        effective_offset = float(os.environ.get('EE_OFFSET', a.ee_offset))
        if not math.isfinite(effective_offset) or effective_offset <= 0:
            raise ValueError('invalid end-effector offset')
        payload['config']['ee_offset'] = effective_offset
        stage = 'model_checkpoint_check'
        if not a.motion_only and not Path(a.model).is_file():
            raise FileNotFoundError('model checkpoint unavailable')
        if not a.motion_only:
            import hashlib
            payload['config']['model_sha256']=hashlib.sha256(Path(a.model).read_bytes()).hexdigest()
        stage = 'simulation_startup'
        print('[startup] SimulationApp begin')
        from isaacsim import SimulationApp
        app = SimulationApp({
            'headless': a.headless, 'width': 960, 'height': 720,
            'renderer': 'RaytracedLighting', 'anti_aliasing': 0,
        })
        print('[startup] SimulationApp ready')
        stage = 'simulation_imports'
        import numpy as np
        from PIL import Image
        from isaacsim.core.api import World
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.sensors.camera import Camera
        from isaacsim.core.utils.viewports import set_camera_view
        from src.sorter.cell_scene import (
            create_scene, initialize_camera, set_appearance, capture_rgb,
            SOURCE, PART_SIZE, PART_MASS, TRAYS, FLOOR_Z, TRAY_WIDTH,
        )
        from src.sorter.measurement import TrayBounds, PlacementMonitor
        from src.sorter.planning.fixed_cell_controller import FixedCellController

        payload['config']['part_mass_kg']=PART_MASS
        stage = 'world_creation'
        print('[startup] World begin')
        world = World(stage_units_in_meters=1., physics_dt=1 / 200, rendering_dt=1 / 20)
        stage = 'scene_creation'
        scene = create_scene(world, robot_profile=a.robot_profile)
        stage = 'world_reset'
        print('[startup] reset begin')
        world.reset()
        print('[startup] reset ready')
        robot, part = scene['robot'], scene['part']
        initial = robot.get_joint_positions().copy()
        print(f'[cell] robot DOF={robot.num_dof} joints={robot.dof_names}')
        names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                 'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint', 'finger_joint']
        for name, value in zip(names, [0., -1.57, 1.57, -1.57, -1.57, 0., 0.]):
            initial[robot.get_dof_index(name)] = value
        stage = 'camera_initialization'
        initialize_camera(scene)
        overview = None
        if a.record_dir:
            Path(a.record_dir).mkdir(parents=True, exist_ok=True)
            overview = Camera('/World/overview', name='overview', resolution=(960, 720), frequency=20)
            overview.initialize()
            overview.set_frequency(-1)
            overview.set_clipping_range(.01, 20.)
            overview.set_focal_length(.040)
            overview.set_horizontal_aperture(.036)
            set_camera_view(eye=np.array([1.35, 1.65, 1.45]), target=np.array([0., .38, .22]), camera_prim_path='/World/overview')
        set_camera_view(eye=np.array([1.35, 1.65, 1.45]), target=np.array([0., .38, .22]))
        classifier = None
        if not a.motion_only:
            stage = 'classifier_initialization'
            from src.sorter.vision import SorterClassifier
            classifier = SorterClassifier(a.model)
        frame_index, tick = 0, 0
        recording_enabled=True

        def step():
            nonlocal tick, frame_index
            if not app.is_running() or not world.is_playing():
                raise RuntimeError('simulation is not playing')
            before = world.current_time_step_index
            # render=True may advance a rendering interval rather than exactly
            # one physics step. Advance physics once, then render separately.
            world.step(render=False, update_fabric=True)
            if world.current_time_step_index != before + 1:
                raise RuntimeError('physics step did not advance exactly once')
            tick += 1
            if tick % a.render_every == 0 or (overview is not None and recording_enabled and tick % a.record_every == 0):
                world.render()
            if overview is not None and recording_enabled and tick % a.record_every == 0:
                rgb = np.asarray(overview.get_rgb())
                if rgb.ndim == 3 and rgb.shape[2] >= 3 and np.isfinite(rgb).all():
                    Image.fromarray(rgb[:, :, :3].astype(np.uint8)).save(Path(a.record_dir) / f'frame_{frame_index:06d}.png')
                    frame_index += 1

        persist()
        for trial in range(a.trials):
            recording_enabled=a.record_trials is None or trial<a.record_trials
            start = time.monotonic()
            simulation_start = world.current_time
            truth = (appearance_plan[trial % len(appearance_plan)]['truth'] if appearance_plan else sequence[trial % len(sequence)])
            control = None
            fatal_trial_error = False
            record = {
                'run_id': run_id, 'trial': trial + 1, 'image': None, 'truth': truth,
                'destination': None, 'decision': {}, 'success': False, 'transfer_success': False,
                'correct_route': False, 'review': False, 'max_part_z': None,
                'cycle_time': 0., 'simulation_seconds': 0., 'motion_simulation_seconds': 0.,
                'final_position': None, 'placement': {}, 'phases': [],
                'defect_score': None, 'confidence': None, 'settled_steps': 0,
                'image_quality': None, 'failure_reason': '', 'reason': '',
            }
            placement, max_height = {}, None
            try:
                stage = 'trial_reset'
                # Explicit supply reset only. No object pose commands occur
                # during the subsequent robot-controlled transportation.
                robot.set_joint_positions(initial.copy())
                robot.set_joint_velocities(np.zeros(robot.num_dof))
                robot.get_articulation_controller().apply_action(ArticulationAction(joint_positions=initial.copy()))
                part.set_world_pose(position=SOURCE.copy(), orientation=np.array([1., 0, 0, 0]))
                part.set_linear_velocity(np.zeros(3))
                part.set_angular_velocity(np.zeros(3))
                condition = {'gray': .65, 'dome': 65., 'key': 180., 'patch_x': 0., 'patch_y': 0.,
                             'patch_w': .032, 'patch_h': .021, 'patch_gray': .06}
                if appearance_plan:
                    condition.update(appearance_plan[trial % len(appearance_plan)]['condition'])
                record['condition']=condition.copy()
                set_appearance(scene, condition, truth == 'defect')
                for _ in range(80):
                    step()
                stage = 'inspection_capture'
                image = np.asarray(capture_rgb(world, scene['camera'], frames=10, physics_step=step))
                if image.ndim != 3 or image.shape[2] < 3 or not np.isfinite(image).all():
                    raise RuntimeError('invalid inspection image')
                image = image[:, :, :3]
                pixel_std = np.std(image.astype(np.float64), axis=(0, 1))
                record['image_quality'] = {
                    'min': float(image.min()), 'max': float(image.max()),
                    'range': float(image.max()) - float(image.min()),
                    'std': float(image.std()), 'spatial_std_max': float(pixel_std.max()),
                    'shape': list(image.shape),
                }
                image_relative = f'inspection_{trial:03d}.png'
                Image.fromarray(image.astype(np.uint8)).save(output / image_relative)
                record['image'] = image_relative
                if float(pixel_std.max()) <= 1e-6:
                    stage = 'blank_inspection_image'
                    raise RuntimeError('inspection image is spatially constant')
                stage = 'image_classification'
                if classifier is None:
                    decision = {'label': a.motion_only, 'defect_score': None,
                                'confidence': None, 'method': 'motion-only'}
                else:
                    decision = classifier.predict(image)  # Pixels only: no truth, names or physical state.
                record.update(decision=decision, destination=decision['label'],
                              defect_score=decision.get('defect_score'), confidence=decision.get('confidence'))
                destination = decision['label']
                if destination not in TRAYS:
                    raise ValueError('classifier returned an invalid destination')
                if classifier is not None and not all(
                    isinstance(decision.get(key), (float, int))
                    and math.isfinite(decision[key]) and 0 <= decision[key] <= 1
                    for key in ('defect_score', 'confidence')
                ):
                    # Invalid model output must not leak NaN into the metrics
                    # and prevent even the failure row from being saved.
                    record.update(defect_score=None, confidence=None,
                                  decision={'method': 'cnn', 'error': 'invalid_scores'})
                    raise ValueError('classifier returned invalid scores')
                record['review'] = destination == 'review'
                print(f'[inspection] trial={trial + 1} destination={destination} score={decision.get("defect_score")}')
                center = TRAYS[destination]
                tray = TrayBounds(
                    destination,
                    [center[0] - TRAY_WIDTH / 2, center[1] - TRAY_WIDTH / 2, FLOOR_Z - .005],
                    [center[0] + TRAY_WIDTH / 2, center[1] + TRAY_WIDTH / 2, FLOOR_Z + .12],
                    floor_z=FLOOR_Z,
                )
                monitor = PlacementMonitor(tray, required_steps=60)
                place = np.array([center[0], center[1], FLOOR_Z + PART_SIZE[2] / 2])
                stage = 'controller_initialization'
                control = FixedCellController(robot, SOURCE.copy(), place, ee_offset=a.ee_offset,
                                              max_joint_speed=a.joint_speed, phase_timeout=a.phase_timeout)
                phase, sample_count = '', 0
                stage = 'transportation'
                while app.is_running() and not control.done and not control.failed:
                    action = control.step()
                    robot.get_articulation_controller().apply_action(action)
                    step()
                    pos, quat = part.get_world_pose()
                    if not np.isfinite(pos).all() or not np.isfinite(quat).all():
                        raise RuntimeError('non-finite workpiece pose')
                    max_height = float(pos[2]) if max_height is None else max(max_height, float(pos[2]))
                    if control.phase == 'transport' and phase != 'transport' and float(pos[2]) <= float(SOURCE[2]) + .06:
                        control.abort('workpiece not lifted')
                    placement = monitor.update(
                        pos, PART_SIZE, part.get_linear_velocity(), part.get_angular_velocity(),
                        released=control.released, orientation_wxyz=quat,
                        gripper_clear=control.phase in ('retreat', 'done'),
                    )
                    if control.phase != phase:
                        phase = control.phase
                        print(f'[trial {trial + 1}] phase={phase} part={np.round(pos, 3).tolist()} finger={robot.get_joint_positions()[robot.get_dof_index("finger_joint")]:.3f}')
                    sample_count += 1
                    if sample_count % 600 == 0:
                        print(f'[trial {trial + 1}] t={control.elapsed:.1f}s error={control.last_position_error:.4f} orientation={control.last_orientation_error:.3f}')
                if control.done:
                    stage = 'placement_settling'
                    for _ in range(160):
                        step()
                        pos, quat = part.get_world_pose()
                        placement = monitor.update(
                            pos, PART_SIZE, part.get_linear_velocity(), part.get_angular_velocity(),
                            released=True, orientation_wxyz=quat, gripper_clear=True,
                        )
                if not control.done and not control.failed:
                    control.abort('simulation stopped')
                transfer_success = bool(control.done and placement.get('success', False))
                # review is an explicit deferral, reported separately from the
                # good/defect accepted-route accuracy in the summary.
                correct_route = classifier is None or destination == 'review' or destination == truth
                success = bool(transfer_success and correct_route)
                failure_reason = control.failure_reason if control.failed else (
                    '' if success else ('wrong destination' if not correct_route else placement.get('reason', 'placement check failed'))
                )
                record.update(success=success, transfer_success=transfer_success,
                              correct_route=correct_route, failure_reason=failure_reason,
                              reason=failure_reason or placement.get('reason', ''))
            except BaseException as error:
                # Capture/classification/reset errors also become failed trial
                # rows. They stop even an evaluation run, because the input or
                # simulator state is no longer reliable.
                reason = _safe_error(stage, error)
                fatal_trial_error = True
                record.update(failure_reason=reason, reason=reason, stop_reason=reason)
                if control is not None:
                    control.abort(reason)
                print(f'[error] {reason}', file=sys.stderr)
                if os.environ.get('SORTER_DEBUG') == '1':
                    import traceback
                    traceback.print_exc()
            record['cycle_time'] = time.monotonic() - start
            record['simulation_seconds'] = max(0., float(world.current_time - simulation_start))
            record['max_part_z'] = max_height
            record['placement'] = placement
            record['settled_steps'] = placement.get('settled_steps', 0)
            if control is not None:
                record['motion_simulation_seconds'] = control.elapsed
                record['phases'] = control.phase_history
            try:
                position, _ = part.get_world_pose()
                values = np.asarray(position, dtype=float)
                if np.isfinite(values).all():
                    record['final_position'] = values.tolist()
            except Exception:
                pass
            cycles.append(record)
            stage = 'inspection_log_write'
            logger.write(record)
            if fatal_trial_error:
                persist('error', record['reason'])
                return 1
            persist()
            print(f'[result] trial={trial + 1} success={record["success"]} transfer={record["transfer_success"]} reason={record["failure_reason"]}')
            if not record['success'] and not a.continue_after_failure:
                persist('stopped_on_failure', record['reason'])
                break
        else:
            persist('completed')
        print('[result] ' + json.dumps(payload['summary']))
        exit_code = 0 if len(cycles) == a.trials and all(record['success'] for record in cycles) else 1
    except BaseException as error:
        reason = _safe_error(stage, error)
        print(f'[error] {reason}', file=sys.stderr)
        if os.environ.get('SORTER_DEBUG') == '1':
            import traceback
            traceback.print_exc()
        try:
            persist('error', reason)
        except Exception:
            print('[error] Unable to save error metrics.', file=sys.stderr)
        exit_code = 1
    finally:
        if app is not None:
            try:
                app.close()
            except Exception as error:
                print(f'[error] {_safe_error("simulation_shutdown", error)}', file=sys.stderr)
                exit_code = 1
    return exit_code


if __name__ == '__main__':
    try:
        sys.exit(main())
    except Exception as error:
        print(f'[error] {_safe_error("output_initialization", error)}', file=sys.stderr)
        sys.exit(1)
