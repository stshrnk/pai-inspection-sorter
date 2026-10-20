"""UR10e Pick and Place with Robotiq 2F-85 Parallel Gripper + Overhead 3D Camera.

Uses UR10e with a physical parallel gripper (Robotiq 2F-85).
An overhead depth camera captures RGB, depth, and point cloud data.
"""

import os
import sys
import time
import json
import argparse
import faulthandler
faulthandler.enable()
faulthandler.dump_traceback_later(120, repeat=True)

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaacsim import SimulationApp

_p = argparse.ArgumentParser()
_p.add_argument("--headless", action="store_true")
_p.add_argument("--curobo", action="store_true", help="Use cuRobo GPU motion planner (default: IK+LERP)")
_p.add_argument("--rmpflow", action="store_true", help="Use RMPFlow instead of IK+LERP")
_p.add_argument("--lerp-speed", type=float, default=0.07, help="Interpolation speed for IK+LERP (0-1)")
_p.add_argument("--perception", action="store_true",
                help="Use 3D camera + top-object detection for pick targets "
                     "(instead of simulator ground-truth cube positions)")
_a = _p.parse_args()
print("[startup] SimulationApp begin", flush=True)
simulation_app = SimulationApp({"headless": _a.headless})

print("[startup] SimulationApp ready", flush=True)

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.utils.viewports import set_camera_view

from src.sorter.planning.pick_place_controller import PickPlaceController
from src.sorter.perception.depth_camera import DepthCameraManager
from src.sorter.perception.object_detector import TopObjectDetector
from src.sorter.tasks.pick_place import PickPlace

WARMUP_FRAMES = int(os.environ.get("WARMUP_FRAMES", "60"))
SAVE_INTERVAL = 200  # save camera images every N steps


def _write_metrics(cycles: list[dict], mode: str, events_dt, args) -> None:
    out_path = os.environ.get("METRICS_OUT")
    if not out_path:
        return
    n = len(cycles)
    succ = sum(1 for c in cycles if c["success"])
    avg_cycle = sum(c["cycle_time"] for c in cycles) / n if n else 0.0
    avg_vel_rms = sum(c["vel_rms"] for c in cycles) / n if n else 0.0
    max_vel = max((c["vel_max"] for c in cycles), default=0.0)
    payload = {
        "config": {
            "mode": mode,
            "perception": bool(args.perception),
            "lerp_speed": args.lerp_speed,
            "events_dt": events_dt,
            "warmup_frames": WARMUP_FRAMES,
            "label": os.environ.get("CONFIG_LABEL", ""),
        },
        "summary": {
            "cycles": n,
            "success": succ,
            "success_rate": succ / n if n else 0.0,
            "avg_cycle_time": avg_cycle,
            "avg_vel_rms": avg_vel_rms,
            "max_vel": max_vel,
        },
        "cycles": cycles,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[metrics] wrote {out_path}")


def _parse_events_dt(raw: str | None) -> list[float] | None:
    if not raw:
        return None
    vals = [float(x) for x in raw.split(",")]
    if len(vals) != 10:
        raise ValueError(f"EVENTS_DT must have 10 comma-separated floats, got {len(vals)}")
    return vals


def main() -> None:
    print("[startup] World begin", flush=True)
    my_world = World(stage_units_in_meters=1.0, physics_dt=1 / 200, rendering_dt=20 / 200)

    print("[startup] World ready", flush=True)
    from src.sorter.config import BIN_POSITION, BIN_SIZE, DEST_BIN_POSITION
    cube_half_h = 0.0515 / 2.0
    # Main cube at the source bin center (most reliable pick location).
    cube_start = BIN_POSITION.copy()
    cube_start[2] = 0.4  # above 2x-scaled bin walls
    # Place target inside destination bin
    target_position = DEST_BIN_POSITION.copy()
    target_position[2] = cube_half_h
    my_task = PickPlace(
        name="ur10e_pick_place",
        cube_initial_position=cube_start,
        target_position=target_position,
        cube_size=np.array([0.1, 0.0515, 0.1]),
    )
    print("[startup] add_task begin", flush=True)
    my_world.add_task(my_task)
    print("[startup] reset begin", flush=True)
    my_world.reset()

    print("[startup] reset ready", flush=True)
    # Viewport camera: top-down view
    set_camera_view(eye=[0, 0.8, 3], target=[0, 0.4, 0])

    # Overhead 3D camera + top-object detector (enabled via --perception).
    camera_mgr: DepthCameraManager | None = None
    detector: TopObjectDetector | None = None
    if _a.perception:
        camera_mgr = DepthCameraManager()
        camera_mgr.initialize()
        detector = TopObjectDetector()

    print(f"Warming up ({WARMUP_FRAMES} frames)...")
    for _ in range(WARMUP_FRAMES):
        my_world.step(render=True)

    task_params = my_world.get_task("ur10e_pick_place").get_params()
    ur10e_name = task_params["robot_name"]["value"]
    my_ur10e = my_world.scene.get_object(ur10e_name)

    events_dt_override = _parse_events_dt(os.environ.get("EVENTS_DT"))
    my_controller = PickPlaceController(
        name="controller",
        robot_articulation=my_ur10e,
        gripper=my_ur10e.gripper,
        use_curobo=_a.curobo,
        use_ik_lerp=not _a.curobo and not _a.rmpflow,
        interpolation_speed=_a.lerp_speed,
        events_dt=events_dt_override,
    )
    articulation_controller = my_ur10e.get_articulation_controller()

    reset_needed = False
    step_count = 0

    # Queue of cubes to pick — simplified to 1 cube for cuRobo validation.
    # 2 cubes: main at bin center (picked first), then red at +Y offset.
    cube_names = [task_params["cube_name"]["value"], "red_cube"]
    cube_index = 0
    current_done = False
    # Perception override for the pick position/yaw for the current cycle.
    perception_pick_pos: np.ndarray | None = None
    perception_pick_yaw: float = 0.0

    mode = "cuRobo" if _a.curobo else ("RMPFlow" if _a.rmpflow else "IK+LERP")
    perception_tag = " +perception" if _a.perception else ""
    print(f"=== UR10e Pick and Place ({mode}{perception_tag}) — {len(cube_names)} cubes ===")
    t_start = time.time()
    cycle_start = t_start

    # Smoothness metrics: running sum of |v|^2 and max |v| per cycle.
    vel_sq_sum = 0.0
    vel_max = 0.0
    vel_samples = 0
    cycles: list[dict] = []

    while simulation_app.is_running():
        my_world.step(render=True)

        if my_world.is_playing():
            if reset_needed:
                my_world.reset()
                for _ in range(WARMUP_FRAMES):
                    my_world.step(render=True)
                my_controller.reset()
                reset_needed = False
                cube_index = 0
                current_done = False
                perception_pick_pos = None
                perception_pick_yaw = 0.0
                step_count = 0
            if my_world.current_time_step_index == 0:
                my_controller.reset()

            step_count += 1

            # All cubes done — wait for user to stop/reset
            if cube_index >= len(cube_names):
                continue

            # Ground-truth cube position (always queried for success check).
            cube_name = cube_names[cube_index]
            cube_obj = my_world.scene.get_object(cube_name)
            cube_pos, _ = cube_obj.get_world_pose()

            # If perception is enabled, run detection once at the start of
            # each new cycle and use the detected centroid as the pick pose.
            if detector is not None and not current_done and perception_pick_pos is None:
                label = cube_name if cube_name != task_params["cube_name"]["value"] else "cube"
                pts, ids, id_to_labels = camera_mgr.get_labeled_pointcloud()
                det = detector.detect_labeled_object(pts, ids, id_to_labels, label) \
                    if pts is not None else None
                if det is None:
                    # Fall back to label-free top-object detection.
                    pc = camera_mgr.get_pointcloud()
                    det = detector.detect_top_object(pc) if pc is not None else None
                if det is not None:
                    # Use camera only for (X, Y); Z comes from the known cube
                    # half-height (cube rests on the ground plane).
                    adjusted = np.array(
                        [det.position[0], det.position[1], cube_half_h],
                        dtype=np.float64,
                    )
                    perception_pick_pos = adjusted
                    perception_pick_yaw = det.yaw
                    print(f"  [perception] label={label} xy=[{det.position[0]:.3f},"
                          f"{det.position[1]:.3f}] z={cube_half_h:.3f} (known) "
                          f"-> pick=[{adjusted[0]:.3f},{adjusted[1]:.3f},{adjusted[2]:.3f}] "
                          f"yaw={det.yaw:+.2f} rad, pts={det.num_points}")
                else:
                    print(f"  [perception] no object detected for label={label}; "
                          f"falling back to ground truth")
                    perception_pick_yaw = 0.0

            pick_pos_for_controller = (
                perception_pick_pos if perception_pick_pos is not None else cube_pos
            )

            observations = my_world.get_observations()
            # Joint velocity sampling for smoothness RMS.
            try:
                jv = my_ur10e.get_joint_velocities()
                if jv is not None:
                    jv_arr = np.asarray(jv, dtype=np.float64)
                    vel_sq_sum += float(np.sum(jv_arr * jv_arr))
                    vmax = float(np.max(np.abs(jv_arr)))
                    if vmax > vel_max:
                        vel_max = vmax
                    vel_samples += jv_arr.size
            except Exception:
                pass
            actions = my_controller.forward(
                picking_position=pick_pos_for_controller,
                placing_position=target_position,
                current_joint_positions=observations[task_params["robot_name"]["value"]]["joint_positions"],
                end_effector_offset=np.array([0, 0, 0.20]),
                pick_yaw=perception_pick_yaw,
            )

            if my_controller.is_done() and not current_done:
                now = time.time()
                elapsed = now - t_start
                cycle_time = now - cycle_start
                dist = float(np.linalg.norm(cube_pos[:2] - target_position[:2]))
                success = dist < 0.15
                vel_rms = (vel_sq_sum / vel_samples) ** 0.5 if vel_samples > 0 else 0.0
                cycles.append({
                    "cube": cube_name,
                    "success": bool(success),
                    "cycle_time": cycle_time,
                    "dist": dist,
                    "vel_rms": vel_rms,
                    "vel_max": vel_max,
                })
                print(f"[{cube_index+1}/{len(cube_names)}] {cube_name}: "
                      f"pos=[{cube_pos[0]:.3f},{cube_pos[1]:.3f},{cube_pos[2]:.3f}], "
                      f"dist={dist:.3f}, SUCCESS={success}, "
                      f"cycle={cycle_time:.2f}s, vel_rms={vel_rms:.3f}, "
                      f"vel_max={vel_max:.3f}, total={elapsed:.2f}s")
                current_done = True
                # Reset per-cycle smoothness accumulators.
                vel_sq_sum = 0.0
                vel_max = 0.0
                vel_samples = 0
                cycle_start = now

                cube_index += 1
                if cube_index < len(cube_names):
                    # Reset controller for next cube
                    my_controller.reset()
                    current_done = False
                    perception_pick_pos = None  # force re-detection next cycle
                    perception_pick_yaw = 0.0
                    print(f"  → Next: {cube_names[cube_index]}")
                else:
                    print(f"=== All {len(cube_names)} cubes done! total_time={elapsed:.2f}s ===")
                    _write_metrics(cycles, mode, events_dt_override, _a)
                    if _a.headless or os.environ.get("METRICS_OUT"):
                        break

            articulation_controller.apply_action(actions)

        if my_world.is_stopped():
            reset_needed = True

    simulation_app.close()


if __name__ == "__main__":
    main()
