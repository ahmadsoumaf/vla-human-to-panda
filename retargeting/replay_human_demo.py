#!/usr/bin/env python3
"""Replay a processed human demonstration with the Panda in the Isaac Sim FLS scene.

processed demo .npz -> human_to_panda.retarget() (anchored on the live scene) -> Panda TCP
plan -> IK tracking with the same PandaTask controller as the scripted demo.

  ./run_human_replay.sh data/real/processed/demo_001.npz
  ./run_human_replay.sh data/real/processed/synthetic_demo.npz --headless
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "isaac"))
sys.path.insert(0, str(PROJECT_ROOT / "retargeting"))

from isaacsim import SimulationApp  # noqa: E402

from human_to_panda import RetargetError, add_config_args, config_from_args, load_demo, retarget  # noqa: E402
from task_geometry import DEFAULT_SCENE, GRIPPER_OPEN  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a processed human FLS demo with the Panda in Isaac Sim.")
    parser.add_argument("demo", type=Path, help="Processed demo .npz from process_demo.py")
    parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE, help="USD scene to load.")
    parser.add_argument("--output", type=Path, help="Sim trajectory .npz (default: data/sim/<name>_replay.npz)")
    parser.add_argument("--plan-output", type=Path, help="Retargeted plan .npz (default: data/real/processed/<name>_panda_plan.npz)")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim without a GUI.")
    parser.add_argument("--grasp-assist", action="store_true", help="Glue the object to the TCP after a successful pinch.")
    parser.add_argument("--keep-open", dest="keep_open", action="store_true", default=None, help="Keep the window open afterwards.")
    parser.add_argument("--close-on-complete", dest="keep_open", action="store_false", help="Close Isaac Sim when done.")
    parser.add_argument("--log-interval", type=int, default=60, help="Print state every N simulation steps.")
    parser.add_argument("--max-steps", type=int, default=20000, help="Hard stop for the replay.")
    add_config_args(parser)
    args = parser.parse_args()
    if args.keep_open is None:
        args.keep_open = not args.headless
    return args


ARGS = parse_args()

# Check the demo before paying for an Isaac Sim start-up.
try:
    DEMO = load_demo(ARGS.demo)
except (RetargetError, FileNotFoundError) as exc:
    print(f"[error] {exc}", file=sys.stderr, flush=True)
    raise SystemExit(1)

simulation_app = SimulationApp(
    {
        "headless": ARGS.headless,
        "width": 1440,
        "height": 900,
        "renderer": "RaytracedLighting",
        "sync_loads": True,
    }
)

import numpy as np  # noqa: E402

from panda_common import ControlConfig, PandaTask, inspect_scene, open_scene, print_scene_info  # noqa: E402


def run_replay() -> bool:
    stage = open_scene(simulation_app, ARGS.scene)
    info = inspect_scene(stage)
    print_scene_info(info)
    task = PandaTask(
        info,
        ControlConfig(render=not ARGS.headless, log_interval=ARGS.log_interval, max_steps=ARGS.max_steps),
    )
    task.print_robot_info()

    cfg = config_from_args(ARGS)
    try:
        plan = retarget(DEMO, task.anchors(), cfg)
    except RetargetError as exc:
        print(f"[error] cannot retarget {ARGS.demo}: {exc}", file=sys.stderr, flush=True)
        return False
    print(f"\n[retarget] {ARGS.demo}\n{plan.summary()}", flush=True)
    plan_path = ARGS.plan_output or PROJECT_ROOT / "data" / "real" / "processed" / f"{ARGS.demo.stem}_panda_plan.npz"
    plan.save(plan_path, cfg)
    print(f"[data] saved plan {plan_path}", flush=True)

    # Move to the start of the plan at safe height before tracking the human path.
    if not task.move_to_pose("[replay] moving to plan start", plan.positions[0], GRIPPER_OPEN, max_steps=400):
        return False

    for name, positions, gripper in plan.segments():
        if not task.follow_path(f"[replay] {name}", positions, gripper):
            print("[error] step budget exhausted (--max-steps)", flush=True)
            return False
        if name in ("descend_grasp", "descend_place"):
            # Settle exactly on the grasp/place pose before the gripper moves.
            if not task.move_to_pose(f"  settle at {name.split('_')[1]} pose", positions[-1], float(gripper[-1]), max_steps=180, min_steps=10):
                return False
            if name == "descend_place":
                task.detach(plan.place_object_center)
        elif name == "close":
            task.check_grasp(ARGS.grasp_assist)
        elif name == "lift":
            lifted = task.object_center()[2] - info.table_top_z
            print(f"  lift check: object {lifted * 100:.1f} cm above the table", flush=True)
            if lifted < 0.03:
                print("  [warn] object was not lifted; the grasp failed", flush=True)

    task.settle(60)
    success = task.check_success(plan.place_object_center[2])
    final_error = float(np.linalg.norm(task.object_center()[:2] - task.target_center[:2]))
    write_report(plan, success, final_error)
    output = ARGS.output or PROJECT_ROOT / "data" / "sim" / f"{ARGS.demo.stem}_replay.npz"
    task.log.save(
        output,
        plan_times=plan.times,
        plan_positions=plan.positions,
        plan_gripper=plan.gripper,
        plan_phase=plan.phase,
        source_demo=np.array(str(ARGS.demo)),
    )
    print(f"[data] saved sim replay {output}", flush=True)
    print("SUCCESS" if success else "FAILED", flush=True)
    return success


def write_report(plan, success: bool, final_error_m: float) -> None:
    """Per-demo curation record: replay success and demonstration quality are kept separate."""
    q = plan.quality
    target_px = DEMO.get("demonstrated_target_xy", np.full(2, np.nan))
    report = {
        "demo": str(ARGS.demo),
        "target_mode": str(DEMO.get("target_mode", "marker")),
        "grasp_t": plan.human_grasp_t,
        "release_t": plan.human_release_t,
        "pick_observed": q["pick_observed"],
        "release_observed": q["release_observed"],
        "approach_observed": q["approach_observed"],
        "demonstrated_target_px": [float(v) for v in np.asarray(target_px).ravel()],
        "grasp_correction_cm": round(q["grasp_correction_m"] * 100, 2),
        "release_correction_cm": round(q["release_correction_m"] * 100, 2),
        "replay_success": bool(success),
        "final_target_error_mm": round(final_error_m * 1000, 1),
        "good_demonstration": q["good_demonstration"],
        "quality_issues": q["issues"],
        "warnings": plan.warnings,
        # Only demos that replay AND were correctly observed go into the VLA dataset.
        "vla_dataset_ok": bool(success and q["good_demonstration"]),
    }
    path = PROJECT_ROOT / "data" / "real" / "processed" / f"{ARGS.demo.stem}_report.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    verdict = "ACCEPT for VLA dataset" if report["vla_dataset_ok"] else (
        "REPLAY SUCCESS but POOR demonstration - not for the VLA dataset" if success else "REPLAY FAILED"
    )
    print(f"[dataset] {verdict}  ({path})", flush=True)


def main() -> int:
    success = run_replay()
    if ARGS.keep_open and not ARGS.headless:
        print("[app] Replay complete; keeping Isaac Sim open.", flush=True)
        while simulation_app.is_running():
            simulation_app.update()
    return 0 if success else 2


try:
    EXIT_CODE = main()
finally:
    if not (ARGS.keep_open and not ARGS.headless):
        simulation_app.close()

raise SystemExit(EXIT_CODE)
