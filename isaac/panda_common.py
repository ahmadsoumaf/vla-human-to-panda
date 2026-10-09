"""Panda + FLS scene control shared by the scripted demo and the human-demo replay.

Import this module only after `SimulationApp(...)` has been created: it pulls in the
Isaac Sim / omni modules at import time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import carb
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.numpy.rotations import quats_to_rot_matrices, rot_matrices_to_quats
from isaacsim.core.utils.rotations import euler_angles_to_quat, quat_to_euler_angles
from isaacsim.core.utils.stage import is_stage_loading, open_stage
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.manipulators.examples.franka import Franka, KinematicsSolver
from pxr import Gf, Usd, UsdGeom
import time
from task_geometry import EMPTY_GRIP_THRESHOLD, GRIPPER_CLOSED, GRIPPER_OPEN, SceneAnchors

PANDA_ROOT = "/World/Panda"
TABLE_PATH = "/World/Table"
FLS_PATH = "/World/FLSObject"
TARGET_PATH = "/World/TargetRegion"
CAMERA_PATH = "/World/Camera"


@dataclass
class Bounds:
    minimum: np.ndarray
    maximum: np.ndarray
    center: np.ndarray
    size: np.ndarray


@dataclass
class SceneInfo:
    panda_articulation: str
    panda_hand: str
    left_finger: str
    right_finger: str
    fls: str
    target: str
    table: str
    camera: str
    table_top_z: float
    fls_bounds: Bounds
    table_bounds: Bounds
    target_center: np.ndarray
    target_radius: float
    object_center_local: np.ndarray


@dataclass
class ControlConfig:
    render: bool = True
    position_tolerance: float = 0.002
    orientation_tolerance: float = 0.05
    log_interval: int = 30
    max_steps: int = 5000


class DemoLog:
    def __init__(self) -> None:
        self.time: list[float] = []
        self.ee_position: list[np.ndarray] = []
        self.ee_orientation: list[np.ndarray] = []
        self.gripper: list[np.ndarray] = []
        self.object_position: list[np.ndarray] = []
        self.target_position: list[np.ndarray] = []

    def append(
        self,
        t: float,
        ee_position: np.ndarray,
        ee_orientation: np.ndarray,
        gripper: np.ndarray,
        object_position: np.ndarray,
        target_position: np.ndarray,
    ) -> None:
        self.time.append(float(t))
        self.ee_position.append(np.array(ee_position, dtype=np.float64))
        self.ee_orientation.append(np.array(ee_orientation, dtype=np.float64))
        self.gripper.append(np.array(gripper, dtype=np.float64))
        self.object_position.append(np.array(object_position, dtype=np.float64))
        self.target_position.append(np.array(target_position, dtype=np.float64))

    def save(self, path: Path, **extra: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            time=np.array(self.time, dtype=np.float64),
            ee_position=np.array(self.ee_position, dtype=np.float64),
            ee_orientation=np.array(self.ee_orientation, dtype=np.float64),
            gripper=np.array(self.gripper, dtype=np.float64),
            object_position=np.array(self.object_position, dtype=np.float64),
            target_position=np.array(self.target_position, dtype=np.float64),
            **extra,
        )


def fmt_vec(vec: np.ndarray) -> str:
    return f"({vec[0]: .4f}, {vec[1]: .4f}, {vec[2]: .4f})"


def open_scene(simulation_app, scene_path: Path) -> Usd.Stage:
    scene_path = scene_path.expanduser().resolve()
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene USD does not exist: {scene_path}")
    print(f"[app] Opening scene: {scene_path}", flush=True)
    if not open_stage(str(scene_path)):
        raise RuntimeError(f"Could not open scene: {scene_path}")
    simulation_app.update()
    simulation_app.update()
    while is_stage_loading():
        simulation_app.update()
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("No USD stage is open after loading the scene.")
    print("[app] Stage loaded; inspecting scene.", flush=True)
    use_scene_camera(simulation_app, stage)
    return stage


def use_scene_camera(simulation_app, stage: Usd.Stage, camera_path: str = CAMERA_PATH) -> None:
    """Show the scene's task camera in the viewport instead of the default Perspective view."""
    if not stage.GetPrimAtPath(camera_path).IsValid():
        print(f"[warn] {camera_path} not in the scene; keeping the Perspective view", flush=True)
        return
    try:
        from isaacsim.core.utils.viewports import set_active_viewport_camera

        set_active_viewport_camera(camera_path)
        simulation_app.update()
        print(f"[app] Viewport camera: {camera_path}", flush=True)
    except Exception as exc:  # no viewport (e.g. some headless setups)
        print(f"[warn] Could not switch the viewport to {camera_path}: {exc}", flush=True)


def compute_bounds(stage: Usd.Stage, prim_path: str) -> Bounds:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Missing prim for bounds: {prim_path}")
    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=True)
    box = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
    minimum = np.array(box.GetMin(), dtype=np.float64)
    maximum = np.array(box.GetMax(), dtype=np.float64)
    center = 0.5 * (minimum + maximum)
    return Bounds(minimum=minimum, maximum=maximum, center=center, size=maximum - minimum)


def world_matrix(stage: Usd.Stage, prim_path: str) -> Gf.Matrix4d:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Missing prim for transform: {prim_path}")
    return UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(prim)


def find_descendant_by_name(stage: Usd.Stage, root_path: str, name: str) -> str | None:
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return None
    for prim in Usd.PrimRange(root):
        if prim.GetName() == name:
            return str(prim.GetPath())
    return None


def find_articulation_root(stage: Usd.Stage) -> str:
    root = stage.GetPrimAtPath(PANDA_ROOT)
    if not root.IsValid():
        raise RuntimeError(f"Missing Panda root: {PANDA_ROOT}")
    for prim in Usd.PrimRange(root):
        if "PhysicsArticulationRootAPI" in prim.GetAppliedSchemas():
            return str(prim.GetPath())
    raise RuntimeError(f"No PhysicsArticulationRootAPI found under {PANDA_ROOT}")


def inspect_scene(stage: Usd.Stage) -> SceneInfo:
    panda_articulation = find_articulation_root(stage)
    panda_hand = find_descendant_by_name(stage, panda_articulation, "panda_hand")
    left_finger = find_descendant_by_name(stage, panda_articulation, "panda_leftfinger")
    right_finger = find_descendant_by_name(stage, panda_articulation, "panda_rightfinger")
    missing = [
        label
        for label, value in [
            ("panda_hand", panda_hand),
            ("panda_leftfinger", left_finger),
            ("panda_rightfinger", right_finger),
        ]
        if value is None
    ]
    if missing:
        raise RuntimeError(f"Could not find Panda prim(s): {', '.join(missing)}")

    for required in [TABLE_PATH, FLS_PATH, TARGET_PATH, CAMERA_PATH]:
        if not stage.GetPrimAtPath(required).IsValid():
            raise RuntimeError(f"Missing required scene prim: {required}")

    table_bounds = compute_bounds(stage, TABLE_PATH)
    fls_bounds = compute_bounds(stage, FLS_PATH)
    fls_root_inv = world_matrix(stage, FLS_PATH).GetInverse()
    object_center_local = np.array(fls_root_inv.Transform(Gf.Vec3d(*fls_bounds.center)), dtype=np.float64)

    target_center = np.array(world_matrix(stage, TARGET_PATH).ExtractTranslation(), dtype=np.float64)
    radius_attr = stage.GetPrimAtPath(TARGET_PATH).GetAttribute("task:radius")
    if not radius_attr or radius_attr.Get() is None:
        raise RuntimeError(f"{TARGET_PATH} has no task:radius attribute; rebuild the scene.")

    return SceneInfo(
        panda_articulation=panda_articulation,
        panda_hand=panda_hand or "",
        left_finger=left_finger or "",
        right_finger=right_finger or "",
        fls=FLS_PATH,
        target=TARGET_PATH,
        table=TABLE_PATH,
        camera=CAMERA_PATH,
        table_top_z=float(table_bounds.maximum[2]),
        fls_bounds=fls_bounds,
        table_bounds=table_bounds,
        target_center=target_center,
        target_radius=float(radius_attr.Get()),
        object_center_local=object_center_local,
    )


def print_scene_info(info: SceneInfo) -> None:
    print("\n[scene] Discovered prim paths", flush=True)
    print(f"  Panda articulation: {info.panda_articulation}", flush=True)
    print(f"  Panda hand/end-effector: {info.panda_hand}", flush=True)
    print(f"  left finger: {info.left_finger}", flush=True)
    print(f"  right finger: {info.right_finger}", flush=True)
    print(f"  FLS object: {info.fls}", flush=True)
    print(f"  target: {info.target}", flush=True)
    print(f"  table: {info.table}", flush=True)
    print(f"  camera: {info.camera}", flush=True)
    print("\n[scene] Bounds/transforms", flush=True)
    print(f"  table top z: {info.table_top_z:.4f}", flush=True)
    print(f"  table center/size: {fmt_vec(info.table_bounds.center)} / {fmt_vec(info.table_bounds.size)}", flush=True)
    print(f"  FLS center/size: {fmt_vec(info.fls_bounds.center)} / {fmt_vec(info.fls_bounds.size)}", flush=True)
    print(f"  target center/radius: {fmt_vec(info.target_center)} / {info.target_radius:.4f}", flush=True)


def quat_rotate(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    return quats_to_rot_matrices(np.array(quat_wxyz, dtype=np.float64)) @ np.array(vec, dtype=np.float64)


def object_center_from_pose(root_position: np.ndarray, root_orientation: np.ndarray, center_local: np.ndarray) -> np.ndarray:
    return np.array(root_position, dtype=np.float64) + quat_rotate(root_orientation, center_local)


def root_position_for_center(center: np.ndarray, orientation: np.ndarray, center_local: np.ndarray) -> np.ndarray:
    return np.array(center, dtype=np.float64) - quat_rotate(orientation, center_local)


class PandaTask:
    """World + Panda + FLS object with IK, gripper and logging helpers."""

    def __init__(self, info: SceneInfo, cfg: ControlConfig) -> None:
        self.info = info
        self.cfg = cfg
        self.world = World(
            physics_dt=1.0 / 60.0,
            rendering_dt=1.0 / 60.0,
            stage_units_in_meters=1.0,
            physics_prim_path="/World/PhysicsScene",
        )
        self.franka = self.world.scene.add(
            Franka(
                prim_path=info.panda_articulation,
                name="panda",
                end_effector_prim_name="panda_hand",
                gripper_open_position=np.array([GRIPPER_OPEN, GRIPPER_OPEN], dtype=np.float64),
                gripper_closed_position=np.array([GRIPPER_CLOSED, GRIPPER_CLOSED], dtype=np.float64),
                deltas=np.array([0.004, 0.004], dtype=np.float64),
            )
        )
        self.fls = self.world.scene.add(SingleRigidPrim(prim_path=info.fls, name="fls_object", reset_xform_properties=False))

        self.world.reset()
        self.reset_robot()
        for _ in range(20):
            self.command_gripper(GRIPPER_OPEN)
            self.world.step(render=cfg.render)

        self.ik_solver = KinematicsSolver(self.franka, end_effector_frame_name="right_gripper")
        base_position, base_orientation = self.franka.get_world_pose()
        self.base_position = np.array(base_position, dtype=np.float64)
        self.ik_solver.get_kinematics_solver().set_robot_base_pose(base_position, base_orientation)
        self.controller = self.franka.get_articulation_controller()

        self.log = DemoLog()
        self.step_counter = 0
        self.target_center = np.array(info.target_center, dtype=np.float64)
        self.attached_root_offset: np.ndarray | None = None
        self.attached_orientation: np.ndarray | None = None

        _, obj_root_orientation = self.fls.get_world_pose()
        # The FLS block is a triangular prism with no parallel side faces.  Close the jaws along
        # the object's local X so one pad lands on the flat back face and the other on the apex
        # (an antipodal grasp); closing across the two slanted faces squeezes the block out.
        object_yaw = float(quat_to_euler_angles(np.array(obj_root_orientation, dtype=np.float64))[2])
        self.down_orientation = euler_angles_to_quat(
            np.array([0.0, math.pi, object_yaw + math.pi / 2.0], dtype=np.float64)
        )

    def print_robot_info(self) -> None:
        print("\n[robot] DOF names", flush=True)
        for idx, name in enumerate(self.franka.dof_names):
            marker = " (finger)" if name in {"panda_finger_joint1", "panda_finger_joint2"} else ""
            print(f"  [{idx}] {name}{marker}", flush=True)
        print(f"[robot] IK frame: {self.ik_solver.get_end_effector_frame()}", flush=True)
        print(f"[robot] Base position: {fmt_vec(self.base_position)}", flush=True)

    def anchors(self) -> SceneAnchors:
        """Live task anchors (object settled on the table, robot base from the articulation)."""
        info = self.info
        return SceneAnchors(
            object_start=[float(v) for v in self.object_center()],
            target_center=[float(v) for v in self.target_center],
            target_radius=info.target_radius,
            table_top_z=info.table_top_z,
            robot_base=[float(v) for v in self.base_position],
            table_min=[float(v) for v in info.table_bounds.minimum],
            table_max=[float(v) for v in info.table_bounds.maximum],
            object_height=float(info.fls_bounds.size[2]),
        )

    # ------------------------------------------------------------------ robot primitives

    def reset_robot(self) -> None:
        dof_count = len(self.franka.dof_names)
        home = np.zeros(dof_count, dtype=np.float64)
        default = np.array([0.012, -0.568, 0.0, -2.811, 0.0, 3.037, 0.741, GRIPPER_OPEN, GRIPPER_OPEN], dtype=np.float64)
        home[: min(dof_count, len(default))] = default[: min(dof_count, len(default))]
        self.franka.set_joint_positions(home)
        self.franka.set_joint_velocities(np.zeros(dof_count, dtype=np.float64))
        self.franka.gripper.set_joint_positions(np.array([GRIPPER_OPEN, GRIPPER_OPEN], dtype=np.float64))

    def command_gripper(self, value: float) -> None:
        indices = np.array(self.franka.gripper.active_joint_indices, dtype=np.int64)
        positions = np.array([value] * len(indices), dtype=np.float64)
        self.franka.apply_action(ArticulationAction(joint_positions=positions, joint_indices=indices))

    def gripper_state(self) -> np.ndarray:
        try:
            return np.array(self.franka.gripper.get_joint_positions(), dtype=np.float64)
        except Exception:
            return np.array([np.nan, np.nan], dtype=np.float64)

    def ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        ee_position, ee_rotation = self.ik_solver.compute_end_effector_pose()
        ee_quat = rot_matrices_to_quats(np.array(ee_rotation, dtype=np.float64))
        return np.array(ee_position, dtype=np.float64), np.array(ee_quat, dtype=np.float64)

    def object_center(self) -> np.ndarray:
        root_position, root_orientation = self.fls.get_world_pose()
        return object_center_from_pose(root_position, root_orientation, self.info.object_center_local)

    def apply_ik(self, target_position: np.ndarray, target_orientation: np.ndarray | None = None) -> bool:
        """Command the arm toward a TCP pose. Returns True if the position-only fallback was used."""
        orientation = self.down_orientation if target_orientation is None else target_orientation
        action, success = self.ik_solver.compute_inverse_kinematics(
            target_position=np.array(target_position, dtype=np.float64),
            target_orientation=np.array(orientation, dtype=np.float64),
            position_tolerance=self.cfg.position_tolerance,
            orientation_tolerance=self.cfg.orientation_tolerance,
        )
        used_position_only = False
        if not success:
            action, success = self.ik_solver.compute_inverse_kinematics(
                target_position=np.array(target_position, dtype=np.float64),
                target_orientation=None,
                position_tolerance=self.cfg.position_tolerance,
            )
            used_position_only = True
        if success:
            self.controller.apply_action(action)
        else:
            carb.log_warn(f"IK did not converge for target {target_position}; holding previous action.")
        return used_position_only

    def step(self, gripper_command: float) -> tuple[np.ndarray, np.ndarray]:
        """Step physics once, log, and keep an assisted object glued to the TCP."""
        self.command_gripper(gripper_command)
        step_started = time.perf_counter()
        self.world.step(render=self.cfg.render)

        if self.cfg.render:
            remaining = self.world.get_physics_dt() - (
                time.perf_counter() - step_started
            )
            if remaining > 0:
                time.sleep(remaining)

        ee_position, ee_orientation = self.ee_pose()
        if self.attached_root_offset is not None and self.attached_orientation is not None:
            self.fls.set_world_pose(position=ee_position + self.attached_root_offset, orientation=self.attached_orientation)
            self.fls.set_linear_velocity(np.zeros(3, dtype=np.float64))
            self.fls.set_angular_velocity(np.zeros(3, dtype=np.float64))
        object_position = self.object_center()
        gripper_state = self.gripper_state()
        self.log.append(self.world.current_time, ee_position, ee_orientation, gripper_state, object_position, self.target_center)
        if self.cfg.log_interval > 0 and self.step_counter % self.cfg.log_interval == 0:
            print(
                f"[log] t={self.world.current_time: .3f} ee={fmt_vec(ee_position)} "
                f"obj={fmt_vec(object_position)} target={fmt_vec(self.target_center)} "
                f"grip_cmd={gripper_command:.3f} grip_state={np.array2string(gripper_state, precision=4)}",
                flush=True,
            )
        self.step_counter += 1
        return ee_position, object_position

    def out_of_steps(self) -> bool:
        return self.step_counter >= self.cfg.max_steps

    # ------------------------------------------------------------------ motion segments

    def move_to_pose(
        self,
        label: str,
        target_position: np.ndarray,
        gripper_command: float,
        max_steps: int,
        min_steps: int = 45,
        settle_steps_required: int = 16,
    ) -> bool:
        """Drive the TCP to a pose and wait until it has settled there. False if the step budget ran out."""
        print(label, flush=True)
        settled = 0
        used_position_only = False
        target_position = np.array(target_position, dtype=np.float64)
        for local_step in range(max_steps):
            used_position_only |= self.apply_ik(target_position)
            ee_position, _ = self.step(gripper_command)
            distance = float(np.linalg.norm(ee_position - target_position))
            if local_step >= min_steps and distance < max(self.cfg.position_tolerance * 1.5, 0.004):
                settled += 1
            else:
                settled = 0
            if settled >= settle_steps_required:
                break
            if self.out_of_steps():
                return False
        if used_position_only:
            print("  [warn] Used position-only IK fallback for part of this segment.", flush=True)
        return True

    def follow_path(self, label: str, positions: np.ndarray, gripper_commands: np.ndarray, steps_per_point: int = 1) -> bool:
        """Track a dense TCP path (one IK target per physics step), without waiting to settle."""
        print(f"{label} ({len(positions)} points)", flush=True)
        used_position_only = False
        for position, gripper in zip(positions, gripper_commands):
            for _ in range(steps_per_point):
                used_position_only |= self.apply_ik(position)
                self.step(float(gripper))
                if self.out_of_steps():
                    return False
        if used_position_only:
            print("  [warn] Used position-only IK fallback for part of this path.", flush=True)
        return True

    def ramp_gripper(self, label: str, start: float, end: float, steps: int = 90) -> None:
        print(label, flush=True)
        for i in range(steps):
            self.step(start + (end - start) * (i + 1) / steps)

    def check_grasp(self, grasp_assist: bool) -> bool:
        """After closing: report whether the block is between the fingers and optionally attach it."""
        ee_position, _ = self.ee_pose()
        root_position, root_orientation = self.fls.get_world_pose()
        grasp_distance = float(np.linalg.norm(self.object_center() - ee_position))
        finger_opening = float(np.nanmin(self.gripper_state()))
        holding = finger_opening > EMPTY_GRIP_THRESHOLD
        print(
            f"  grasp check: ee-object distance={grasp_distance:.4f} m finger_opening={finger_opening:.4f} m "
            f"({'object between fingers' if holding else 'fingers closed on nothing'})",
            flush=True,
        )
        if grasp_assist and holding and grasp_distance < 0.03:
            self.attached_orientation = np.array(root_orientation, dtype=np.float64)
            self.attached_root_offset = np.array(root_position, dtype=np.float64) - ee_position
            print("  grasp assist: enabled (disable with --no-grasp-assist).", flush=True)
        elif grasp_assist:
            print("  [warn] Grasp assist not attached: the fingers are not closed on the object.", flush=True)
        return holding

    def detach(self, place_object_center: np.ndarray | None = None) -> None:
        """Drop an assisted object, optionally snapping it to its placement pose first."""
        if self.attached_orientation is not None and place_object_center is not None:
            root = root_position_for_center(place_object_center, self.attached_orientation, self.info.object_center_local)
            self.fls.set_world_pose(position=root, orientation=self.attached_orientation)
            self.fls.set_linear_velocity(np.zeros(3, dtype=np.float64))
            self.fls.set_angular_velocity(np.zeros(3, dtype=np.float64))
        self.attached_root_offset = None
        self.attached_orientation = None

    def settle(self, count: int, gripper_command: float = GRIPPER_OPEN) -> None:
        for _ in range(count):
            self.step(gripper_command)

    def check_success(self, expected_object_z: float) -> bool:
        final_object = self.object_center()
        xy_error = float(np.linalg.norm(final_object[:2] - self.target_center[:2]))
        z_error = abs(float(final_object[2] - expected_object_z))
        success_radius = max(self.info.target_radius, 0.06)
        success = xy_error <= success_radius and z_error <= 0.08
        print(
            f"[check] final object={fmt_vec(final_object)} target={fmt_vec(self.target_center)} "
            f"xy_error={xy_error:.4f} z_error={z_error:.4f} threshold={success_radius:.4f}",
            flush=True,
        )
        return success
