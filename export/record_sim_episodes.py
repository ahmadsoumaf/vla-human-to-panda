#!/usr/bin/env python3
"""Record accepted human-demo replays as raw robot episodes for the LeRobot/SmolVLA export.

For every demo with "vla_dataset_ok": true in data/real/processed/<demo>_report.json, the saved
retargeted plan (<demo>_panda_plan.npz) is executed with the frozen PandaTask controller exactly
as replay_human_demo.py does, and every --frame-stride physics steps it records:

  observation.images.front   RGB from the scene camera /World/Camera
  observation.state          7 arm joint positions + gripper opening (mean finger joint, m)
  action                     7 commanded arm joint targets + gripper command (finger joint, m)

Output: data/lerobot_raw/<demo>/{episode.npz, meta.json, frames/000000.jpg ...}
Nothing in the frozen pipeline is modified: recording hooks into PandaTask by subclassing and by
wrapping its articulation controller.

  ./run_record_episodes.sh            (headless; the camera is still rendered)
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "isaac"))
sys.path.insert(0, str(PROJECT_ROOT / "retargeting"))

from isaacsim import SimulationApp  # noqa: E402

from task_geometry import DEFAULT_SCENE, GRIPPER_OPEN  # noqa: E402

PROCESSED = PROJECT_ROOT / "data" / "real" / "processed"
DEFAULT_OUT = PROJECT_ROOT / "data" / "lerobot_raw"
TASK = "Pick up the blue triangle and place it on the target."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record accepted demo replays for the LeRobot export.")
    parser.add_argument("--demos", nargs="*", help="Demo names (default: all with vla_dataset_ok=true).")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--frame-stride", type=int, default=2, help="Record every N physics steps (60 Hz / 2 = 30 fps).")
    parser.add_argument("--gui", action="store_true", help="Show the Isaac window while recording.")
    return parser.parse_args()


ARGS = parse_args()


def accepted_demos() -> list[str]:
    names = []
    for path in sorted(PROCESSED.glob("*_report.json")):
        if json.loads(path.read_text()).get("vla_dataset_ok"):
            names.append(path.name.removesuffix("_report.json"))
    return names


DEMOS = ARGS.demos or accepted_demos()
if not DEMOS:
    print("[error] no accepted demos (vla_dataset_ok=true) to record", file=sys.stderr)
    raise SystemExit(1)
for demo in DEMOS:
    if not (PROCESSED / f"{demo}_panda_plan.npz").exists():
        print(f"[error] missing plan {demo}_panda_plan.npz; replay the demo first", file=sys.stderr)
        raise SystemExit(1)

simulation_app = SimulationApp(
    {"headless": not ARGS.gui, "width": 1440, "height": 900, "renderer": "RaytracedLighting", "sync_loads": True}
)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.sensors.camera import Camera  # noqa: E402

from human_to_panda import PHASES  # noqa: E402
from panda_common import CAMERA_PATH, ControlConfig, PandaTask, inspect_scene, open_scene  # noqa: E402

ARM_DOF = 7


class RecordingController:
    """Wraps the articulation controller to remember the last commanded arm joint targets."""

    def __init__(self, inner) -> None:
        self._inner = inner
        self.last_arm_target: np.ndarray | None = None

    def apply_action(self, action) -> None:
        if action.joint_positions is not None:
            targets = np.asarray(action.joint_positions, dtype=np.float64).ravel()
            indices = action.joint_indices
            if indices is None:
                self.last_arm_target = targets[:ARM_DOF].copy()
            else:
                full = np.full(9, np.nan) if self.last_arm_target is None else np.concatenate([self.last_arm_target, [np.nan] * 2])
                full[np.asarray(indices, dtype=np.int64).ravel()] = targets
                self.last_arm_target = full[:ARM_DOF].copy()
        self._inner.apply_action(action)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class RecordingPandaTask(PandaTask):
    """PandaTask that records camera/state/action every `stride` physics steps."""

    def start_recording(self, camera: Camera, stride: int, frames_dir: Path) -> None:
        self.camera = camera
        self.stride = stride
        self.frames_dir = frames_dir
        self.recorder = RecordingController(self.controller)
        self.controller = self.recorder
        self.states: list[np.ndarray] = []
        self.actions: list[np.ndarray] = []
        self.ee_positions: list[np.ndarray] = []
        self.object_positions: list[np.ndarray] = []
        self.recording = True

    def step(self, gripper_command: float):
        result = super().step(gripper_command)
        if getattr(self, "recording", False) and self.step_counter % self.stride == 0:
            self._capture(gripper_command)
        return result

    def _capture(self, gripper_command: float) -> None:
        rgb = self.camera.get_rgb()
        if rgb is None or rgb.size == 0:
            return  # renderer not producing frames yet
        joints = np.asarray(self.franka.get_joint_positions(), dtype=np.float64)
        arm = joints[:ARM_DOF]
        target = self.recorder.last_arm_target
        if target is None or not np.all(np.isfinite(target)):
            target = arm
        cv2.imwrite(
            str(self.frames_dir / f"{len(self.states):06d}.jpg"),
            cv2.cvtColor(np.asarray(rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR),
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )
        self.states.append(np.concatenate([arm, [float(np.mean(joints[ARM_DOF:ARM_DOF + 2]))]]))
        self.actions.append(np.concatenate([target, [float(gripper_command)]]))
        self.ee_positions.append(self.ee_pose()[0])
        self.object_positions.append(self.object_center())


def load_plan(path: Path):
    data = np.load(path)
    phase = np.asarray(data["phase"])
    boundaries = np.flatnonzero(np.diff(phase)) + 1
    segments = []
    for chunk in np.split(np.arange(len(phase)), boundaries):
        segments.append((PHASES[int(phase[chunk[0]])], data["positions"][chunk], data["gripper"][chunk]))
    return segments, np.asarray(data["place_object_center"], dtype=np.float64)


def record_demo(demo: str) -> dict:
    stage = open_scene(simulation_app, ARGS.scene)
    info = inspect_scene(stage)
    task = RecordingPandaTask(info, ControlConfig(render=True, log_interval=0))
    camera = Camera(prim_path=CAMERA_PATH, resolution=(ARGS.width, ARGS.height))
    camera.initialize()
    for _ in range(10):  # let the render product produce frames before recording
        task.world.step(render=True)

    out_dir = ARGS.output / demo
    if out_dir.exists():
        shutil.rmtree(out_dir)
    (out_dir / "frames").mkdir(parents=True)
    task.start_recording(camera, ARGS.frame_stride, out_dir / "frames")

    segments, place_object_center = load_plan(PROCESSED / f"{demo}_panda_plan.npz")
    # Same execution as replay_human_demo.py (frozen): move to start, then every plan segment.
    ok = task.move_to_pose("[record] moving to plan start", segments[0][1][0], GRIPPER_OPEN, max_steps=400)
    for name, positions, gripper in segments:
        if not ok:
            break
        ok = task.follow_path(f"[record] {name}", positions, gripper)
        if ok and name in ("descend_grasp", "descend_place"):
            ok = task.move_to_pose(f"  settle at {name.split('_')[1]} pose", positions[-1], float(gripper[-1]), max_steps=180, min_steps=10)
            if name == "descend_place":
                task.detach(place_object_center)
        elif name == "close":
            task.check_grasp(grasp_assist=False)
    task.settle(30)
    success = bool(ok and task.check_success(place_object_center[2]))
    final_error = float(np.linalg.norm(task.object_center()[:2] - task.target_center[:2]))

    fps = 60.0 / ARGS.frame_stride
    n = len(task.states)
    np.savez_compressed(
        out_dir / "episode.npz",
        state=np.array(task.states, dtype=np.float32),
        action=np.array(task.actions, dtype=np.float32),
        ee_position=np.array(task.ee_positions, dtype=np.float32),
        object_position=np.array(task.object_positions, dtype=np.float32),
        timestamp=(np.arange(n) / fps).astype(np.float32),
    )
    meta = {
        "demo": demo,
        "task": TASK,
        "fps": fps,
        "frames": n,
        "image_size": [ARGS.height, ARGS.width],
        "camera": CAMERA_PATH,
        "state_names": [f"panda_joint{i}" for i in range(1, 8)] + ["gripper_opening_m"],
        "action_names": [f"panda_joint{i}_target" for i in range(1, 8)] + ["gripper_command_m"],
        "replay_success": success,
        "final_target_error_mm": round(final_error * 1000, 1),
        "source_report": str((PROCESSED / f"{demo}_report.json").relative_to(PROJECT_ROOT)),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[record] {demo}: {n} frames @ {fps:.0f} fps, success={success}, error={final_error * 1000:.1f} mm -> {out_dir}", flush=True)

    World.clear_instance()
    return meta


def main() -> int:
    results = [record_demo(demo) for demo in DEMOS]
    failed = [m["demo"] for m in results if not m["replay_success"]]
    if failed:
        print(f"[error] replay failed while recording: {failed}; these episodes must not be exported", file=sys.stderr)
        return 2
    print(f"[record] {len(results)} episodes recorded in {ARGS.output}", flush=True)
    return 0


try:
    EXIT_CODE = main()
finally:
    simulation_app.close()
raise SystemExit(EXIT_CODE)
