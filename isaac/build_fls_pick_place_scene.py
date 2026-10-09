#!/usr/bin/env python3
"""Build the FLS Panda pick/place scene for Isaac Sim.

This script intentionally stops at reliable scene construction.  The controller
layer can be added later without changing the asset/layout plumbing here.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

from isaacsim import SimulationApp

from task_geometry import SceneAnchors


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSET_ROOT = PROJECT_ROOT / "assets" / "fls"
DEFAULT_OUTPUT = PROJECT_ROOT / "isaac" / "fls_pick_place_scene.usd"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the FLS Panda pick/place USD scene.")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim without a GUI and exit after saving.")
    parser.add_argument("--build-only", action="store_true", help="Save the USD and exit instead of keeping the GUI open.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="USD scene path to write.")
    parser.add_argument("--fls-mass", type=float, default=0.05, help="Mass in kg for the movable FLS object.")
    parser.add_argument("--gripper-force", type=float, default=40.0, help="Max force in N for each Panda finger drive.")
    parser.add_argument("--no-open", action="store_true", help="Do not reopen the saved scene in the active Isaac stage.")
    return parser.parse_args()


ARGS = parse_args()

simulation_app = SimulationApp(
    {
        "headless": ARGS.headless,
        "width": 1440,
        "height": 900,
    }
)

import omni.timeline
import omni.usd
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade


@dataclass(frozen=True)
class Bounds:
    min: Gf.Vec3d
    max: Gf.Vec3d
    size: Gf.Vec3d
    center: Gf.Vec3d


@dataclass(frozen=True)
class AssetInfo:
    path: Path
    default_prim: str
    meters_per_unit: float
    up_axis: str
    bounds: Bounds
    articulation_suffix: str | None = None
    hand_suffix: str | None = None
    left_finger_suffix: str | None = None
    right_finger_suffix: str | None = None


@dataclass(frozen=True)
class ScenePaths:
    world: str = "/World"
    panda: str = "/World/Panda"
    table: str = "/World/Table"
    fls: str = "/World/FLSObject"
    target: str = "/World/TargetRegion"
    camera: str = "/World/Camera"
    looks: str = "/World/Looks"
    lights: str = "/World/Lights"
    physics_scene: str = "/World/PhysicsScene"
    task: str = "/World/Task"


@dataclass(frozen=True)
class Layout:
    panda_translation: Gf.Vec3d
    table_translation: Gf.Vec3d
    fls_translation: Gf.Vec3d
    target_center: Gf.Vec3d
    camera_eye: Gf.Vec3d
    camera_target: Gf.Vec3d
    table_top_z: float
    table_min_world: Gf.Vec3d
    table_max_world: Gf.Vec3d
    fls_center_world: Gf.Vec3d


PATHS = ScenePaths()


def fmt_vec(vec: Gf.Vec3d | Gf.Vec3f) -> str:
    return f"({float(vec[0]):.4f}, {float(vec[1]):.4f}, {float(vec[2]):.4f})"


def shifted(bounds: Bounds, translation: Gf.Vec3d) -> Bounds:
    return Bounds(
        min=bounds.min + translation,
        max=bounds.max + translation,
        size=bounds.size,
        center=bounds.center + translation,
    )


def inspect_asset(asset_path: Path) -> AssetInfo:
    stage = Usd.Stage.Open(str(asset_path))
    if stage is None:
        raise RuntimeError(f"Could not open USD asset: {asset_path}")

    default_prim = stage.GetDefaultPrim()
    if not default_prim:
        raise RuntimeError(f"Asset has no default prim: {asset_path}")

    purposes = [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy]
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes, useExtentsHint=True)
    box = bbox_cache.ComputeWorldBound(default_prim).ComputeAlignedBox()
    mn = Gf.Vec3d(box.GetMin())
    mx = Gf.Vec3d(box.GetMax())
    bounds = Bounds(min=mn, max=mx, size=mx - mn, center=(mn + mx) * 0.5)

    default_path = str(default_prim.GetPath())
    articulation_suffix = None
    hand_suffix = None
    left_finger_suffix = None
    right_finger_suffix = None

    for prim in stage.Traverse():
        prim_path = str(prim.GetPath())
        name = prim.GetName()
        suffix = prim_path.removeprefix(default_path)

        if "PhysicsArticulationRootAPI" in prim.GetAppliedSchemas():
            articulation_suffix = suffix
        if name == "panda_hand":
            hand_suffix = suffix
        elif name == "panda_leftfinger":
            left_finger_suffix = suffix
        elif name == "panda_rightfinger":
            right_finger_suffix = suffix

    info = AssetInfo(
        path=asset_path,
        default_prim=default_path,
        meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage),
        up_axis=str(UsdGeom.GetStageUpAxis(stage)),
        bounds=bounds,
        articulation_suffix=articulation_suffix,
        hand_suffix=hand_suffix,
        left_finger_suffix=left_finger_suffix,
        right_finger_suffix=right_finger_suffix,
    )
    print(
        f"[asset] {asset_path.name}: default={info.default_prim} "
        f"units={info.meters_per_unit} up={info.up_axis} "
        f"min={fmt_vec(info.bounds.min)} max={fmt_vec(info.bounds.max)} size={fmt_vec(info.bounds.size)}",
        flush=True,
    )
    return info


def compute_layout(panda: AssetInfo, table: AssetInfo, fls: AssetInfo) -> Layout:
    table_translation = Gf.Vec3d(0.0, 0.0, -table.bounds.min[2])
    table_world = shifted(table.bounds, table_translation)
    table_width = float(table_world.size[0])
    table_depth = float(table_world.size[1])
    table_top_z = float(table_world.max[2])

    # Keep the object and target well inside the Panda's ~0.85 m reach (both
    # ~0.45-0.55 m from the base) so IK can hold a top-down gripper orientation.
    panda_center_xy = Gf.Vec3d(
        float(table_world.min[0] + table_width * 0.25),
        float(table_world.min[1] + table_depth * 0.50),
        0.0,
    )
    panda_tabletop_clearance = 0.002
    panda_translation = Gf.Vec3d(
        float(panda_center_xy[0] - panda.bounds.center[0]),
        float(panda_center_xy[1] - panda.bounds.center[1]),
        float(table_top_z + panda_tabletop_clearance - panda.bounds.min[2]),
    )

    fls_center_xy = Gf.Vec3d(
        float(table_world.min[0] + table_width * 0.50),
        float(table_world.min[1] + table_depth * 0.35),
        0.0,
    )
    fls_clearance = 0.002
    fls_translation = Gf.Vec3d(
        float(fls_center_xy[0] - fls.bounds.center[0]),
        float(fls_center_xy[1] - fls.bounds.center[1]),
        float(table_top_z + fls_clearance - fls.bounds.min[2]),
    )
    fls_center_world = fls.bounds.center + fls_translation

    target_center = Gf.Vec3d(
        float(table_world.min[0] + table_width * 0.53),
        float(table_world.min[1] + table_depth * 0.75),
        float(table_top_z + 0.004),
    )

    camera_target = Gf.Vec3d(
        float((fls_center_world[0] + target_center[0]) * 0.5),
        float((fls_center_world[1] + target_center[1]) * 0.5),
        float(table_top_z + 0.10),
    )
    camera_eye = Gf.Vec3d(
        float(table_world.center[0] + table_width * 0.70),
        float(table_world.min[1] - table_depth * 1.15),
        float(table_top_z + max(table_depth, 0.65) * 0.95),
    )

    return Layout(
        panda_translation=panda_translation,
        table_translation=table_translation,
        fls_translation=fls_translation,
        target_center=target_center,
        camera_eye=camera_eye,
        camera_target=camera_target,
        table_top_z=table_top_z,
        table_min_world=table_world.min,
        table_max_world=table_world.max,
        fls_center_world=fls_center_world,
    )


def set_xform(
    prim: Usd.Prim,
    translate: Gf.Vec3d | None = None,
    scale: Gf.Vec3f | None = None,
    matrix: Gf.Matrix4d | None = None,
) -> None:
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    if matrix is not None:
        xform.AddTransformOp().Set(matrix)
        return
    if translate is not None:
        xform.AddTranslateOp().Set(translate)
    if scale is not None:
        xform.AddScaleOp().Set(scale)


def add_reference(stage: Usd.Stage, prim_path: str, asset_path: Path, translation: Gf.Vec3d) -> Usd.Prim:
    prim = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    rel_asset = os.path.relpath(asset_path, ARGS.output.resolve().parent)
    prim.GetReferences().AddReference(rel_asset)
    set_xform(prim, translate=translation)
    return prim


def create_preview_material(
    stage: Usd.Stage,
    material_path: str,
    color: tuple[float, float, float],
    roughness: float = 0.55,
    metallic: float = 0.0,
    opacity: float = 1.0,
) -> UsdShade.Material:
    UsdGeom.Scope.Define(stage, PATHS.looks)
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, f"{material_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(opacity)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return material


def bind_material_to_gprims(stage: Usd.Stage, root_path: str, material: UsdShade.Material) -> None:
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"Cannot bind material; missing root prim: {root_path}")

    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Gprim):
            binding = UsdShade.MaterialBindingAPI.Apply(prim)
            binding.Bind(material, UsdShade.Tokens.strongerThanDescendants)


def ensure_static_colliders(stage: Usd.Stage, root_path: str) -> None:
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"Cannot add table colliders; missing root prim: {root_path}")

    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rb = UsdPhysics.RigidBodyAPI(prim)
            rb.CreateRigidBodyEnabledAttr(False)
            rb.CreateKinematicEnabledAttr(True)

        if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Gprim):
            collision = UsdPhysics.CollisionAPI.Apply(prim)
            collision.CreateCollisionEnabledAttr(True)
            physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            physx_collision.CreateContactOffsetAttr(0.005)
            physx_collision.CreateRestOffsetAttr(0.0)

        if prim.IsA(UsdGeom.Mesh) and not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_collision.CreateApproximationAttr("convexHull")


def ensure_dynamic_fls(stage: Usd.Stage, root_path: str, mass_kg: float) -> None:
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        raise RuntimeError(f"Cannot add FLS physics; missing root prim: {root_path}")

    has_collider = False
    for prim in Usd.PrimRange(root):
        if prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Gprim):
            collision = UsdPhysics.CollisionAPI.Apply(prim)
            collision.CreateCollisionEnabledAttr(True)
            physx_collision = PhysxSchema.PhysxCollisionAPI.Apply(prim)
            physx_collision.CreateContactOffsetAttr(0.002)
            physx_collision.CreateRestOffsetAttr(0.0)
            has_collider = True

        if prim.IsA(UsdGeom.Mesh):
            mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
            mesh_collision.CreateApproximationAttr("convexHull")

    if not has_collider:
        raise RuntimeError(f"FLS object has no mesh/Gprim descendants for collision: {root_path}")

    rb = UsdPhysics.RigidBodyAPI.Apply(root)
    rb.CreateRigidBodyEnabledAttr(True)
    rb.CreateKinematicEnabledAttr(False)

    physx_rb = PhysxSchema.PhysxRigidBodyAPI.Apply(root)
    physx_rb.CreateDisableGravityAttr(False)
    physx_rb.CreateLinearDampingAttr(0.05)
    physx_rb.CreateAngularDampingAttr(0.05)
    # The block is tiny and light; without extra iterations/CCD the finger pads tunnel into it.
    physx_rb.CreateSolverPositionIterationCountAttr(32)
    physx_rb.CreateSolverVelocityIterationCountAttr(4)
    physx_rb.CreateEnableCCDAttr(True)

    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(mass_kg))

    # Grippy material so the parallel-jaw grasp holds the small block (PhysX default mu is 0.5).
    physics_material_path = f"{PATHS.looks}/FLS_PhysicsMaterial"
    physics_material = UsdShade.Material.Define(stage, physics_material_path)
    material_api = UsdPhysics.MaterialAPI.Apply(physics_material.GetPrim())
    material_api.CreateStaticFrictionAttr(1.0)
    material_api.CreateDynamicFrictionAttr(1.0)
    material_api.CreateRestitutionAttr(0.0)
    physx_material = PhysxSchema.PhysxMaterialAPI.Apply(physics_material.GetPrim())
    physx_material.CreateFrictionCombineModeAttr("max")
    for prim in Usd.PrimRange(root):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                physics_material, UsdShade.Tokens.weakerThanDescendants, "physics"
            )


def limit_gripper_force(stage: Usd.Stage, panda_info: AssetInfo, max_force_n: float) -> None:
    """Override the asset's 200 N finger drives; at that force the pads tunnel through the small block."""
    hand_path = suffix_path(PATHS.panda, panda_info.hand_suffix)
    for joint_name in ("panda_finger_joint1", "panda_finger_joint2"):
        joint = stage.GetPrimAtPath(f"{hand_path}/{joint_name}")
        if not joint.IsValid():
            raise RuntimeError(f"Missing Panda finger joint: {hand_path}/{joint_name}")
        UsdPhysics.DriveAPI.Apply(joint, "linear").CreateMaxForceAttr(float(max_force_n))


def preapply_gripper_rigid_body_api(stage: Usd.Stage, panda_info: AssetInfo) -> None:
    """Author PhysxRigidBodyAPI on the hand/fingers ahead of time.

    The Franka/gripper wrappers wrap these links in SingleRigidPrim, which applies
    PhysxRigidBodyAPI at runtime if it is missing.  Adding an API schema to a link whose
    visuals are instanceable makes the renderer drop that link's mesh at the world origin
    (the hand ends up on the floor under the table and the fingers float below the wrist).
    """
    for suffix in (panda_info.hand_suffix, panda_info.left_finger_suffix, panda_info.right_finger_suffix):
        prim = stage.GetPrimAtPath(suffix_path(PATHS.panda, suffix))
        PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateSleepThresholdAttr(0.0)


def create_target(stage: Usd.Stage, layout: Layout) -> None:
    """Invisible target anchor: a bare Xform at the target center carrying the success radius."""
    table_span = layout.table_max_world - layout.table_min_world
    radius = max(0.055, min(float(table_span[0]), float(table_span[1])) * 0.11)

    target = UsdGeom.Xform.Define(stage, PATHS.target)
    set_xform(target.GetPrim(), translate=layout.target_center)
    target.GetPrim().CreateAttribute("task:radius", Sdf.ValueTypeNames.Double, custom=True).Set(radius)


def create_ground(stage: Usd.Stage, layout: Layout, material: UsdShade.Material) -> None:
    span = layout.table_max_world - layout.table_min_world
    ground_size_x = max(float(span[0]) * 1.7, 2.5)
    ground_size_y = max(float(span[1]) * 3.0, 2.0)
    thickness = 0.02
    cube = UsdGeom.Cube.Define(stage, "/World/GroundPlane")
    cube.CreateSizeAttr(1.0)
    set_xform(
        cube.GetPrim(),
        translate=Gf.Vec3d(0.0, 0.0, -thickness * 0.5),
        scale=Gf.Vec3f(ground_size_x, ground_size_y, thickness),
    )
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr(True)
    PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim()).CreateRestOffsetAttr(0.0)
    UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)


def look_at_matrix(eye: Gf.Vec3d, target: Gf.Vec3d, up: Gf.Vec3d = Gf.Vec3d(0.0, 0.0, 1.0)) -> Gf.Matrix4d:
    forward = target - eye
    if forward.GetLength() < 1e-6:
        raise ValueError("Camera eye and target are too close")
    forward.Normalize()

    z_axis = forward * -1.0
    x_axis = Gf.Cross(up, z_axis)
    if x_axis.GetLength() < 1e-6:
        x_axis = Gf.Vec3d(1.0, 0.0, 0.0)
    else:
        x_axis.Normalize()
    y_axis = Gf.Cross(z_axis, x_axis)
    y_axis.Normalize()

    return Gf.Matrix4d(
        x_axis[0],
        x_axis[1],
        x_axis[2],
        0.0,
        y_axis[0],
        y_axis[1],
        y_axis[2],
        0.0,
        z_axis[0],
        z_axis[1],
        z_axis[2],
        0.0,
        eye[0],
        eye[1],
        eye[2],
        1.0,
    )


def create_camera(stage: Usd.Stage, layout: Layout) -> None:
    camera = UsdGeom.Camera.Define(stage, PATHS.camera)
    set_xform(camera.GetPrim(), matrix=look_at_matrix(layout.camera_eye, layout.camera_target))
    camera.CreateFocalLengthAttr(28.0)
    camera.CreateFocusDistanceAttr(float((layout.camera_eye - layout.camera_target).GetLength()))
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))


def add_lighting(stage: Usd.Stage, layout: Layout) -> None:
    UsdGeom.Scope.Define(stage, PATHS.lights)

    dome = UsdLux.DomeLight.Define(stage, f"{PATHS.lights}/Dome")
    dome.CreateIntensityAttr(250.0)
    dome.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
    # Lift the light gizmo off the world origin (it otherwise sits on the floor under the table).
    set_xform(dome.GetPrim(), translate=Gf.Vec3d(0.0, 0.0, layout.table_top_z + 1.8))

    rect = UsdLux.RectLight.Define(stage, f"{PATHS.lights}/WorkspaceSoftbox")
    rect.CreateIntensityAttr(850.0)
    rect.CreateWidthAttr(1.8)
    rect.CreateHeightAttr(1.2)
    set_xform(
        rect.GetPrim(),
        translate=Gf.Vec3d(0.0, -0.15, layout.table_top_z + 1.35),
    )

    key = UsdLux.DistantLight.Define(stage, f"{PATHS.lights}/Key")
    key.CreateIntensityAttr(450.0)
    key.CreateAngleAttr(0.35)
    # Distant lights only use rotation; the translation just moves the gizmo off the floor.
    key_xform = UsdGeom.Xformable(key.GetPrim())
    key_xform.ClearXformOpOrder()
    key_xform.AddTranslateOp().Set(Gf.Vec3d(0.6, -0.6, layout.table_top_z + 1.6))
    key_xform.AddRotateXYZOp().Set(Gf.Vec3f(35.0, 0.0, 30.0))


def create_physics_scene(stage: Usd.Stage) -> None:
    scene = UsdPhysics.Scene.Define(stage, PATHS.physics_scene)
    scene.CreateGravityDirectionAttr(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr(9.81)
    physx_scene = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
    physx_scene.CreateSolverTypeAttr("TGS")
    physx_scene.CreateEnableCCDAttr(True)


def create_task_metadata(stage: Usd.Stage, layout: Layout, panda_info: AssetInfo) -> None:
    prim = UsdGeom.Xform.Define(stage, PATHS.task).GetPrim()
    attrs = {
        "task:description": "blue FLS triangle pick -> lift -> move -> place",
        "task:flsPrimPath": PATHS.fls,
        "task:targetPrimPath": PATHS.target,
        "task:pandaArticulationPath": suffix_path(PATHS.panda, panda_info.articulation_suffix),
        "task:initialObjectCenter": fmt_vec(layout.fls_center_world),
        "task:targetCenter": fmt_vec(layout.target_center),
    }
    for name, value in attrs.items():
        prim.CreateAttribute(name, Sdf.ValueTypeNames.String, custom=True).Set(value)


def suffix_path(root_path: str, suffix: str | None) -> str:
    if not suffix:
        return root_path
    return root_path + suffix


def require_panda_paths(panda_info: AssetInfo) -> None:
    missing = [
        label
        for label, suffix in {
            "articulation": panda_info.articulation_suffix,
            "hand/end-effector": panda_info.hand_suffix,
            "left finger": panda_info.left_finger_suffix,
            "right finger": panda_info.right_finger_suffix,
        }.items()
        if suffix is None
    ]
    if missing:
        raise RuntimeError(f"Panda asset is missing expected prims: {', '.join(missing)}")


def validate_prims(stage: Usd.Stage, panda_info: AssetInfo) -> dict[str, str]:
    require_panda_paths(panda_info)
    important = {
        "Panda articulation": suffix_path(PATHS.panda, panda_info.articulation_suffix),
        "Panda hand/end-effector": suffix_path(PATHS.panda, panda_info.hand_suffix),
        "left finger": suffix_path(PATHS.panda, panda_info.left_finger_suffix),
        "right finger": suffix_path(PATHS.panda, panda_info.right_finger_suffix),
        "FLS object": PATHS.fls,
        "target": PATHS.target,
        "table": PATHS.table,
        "camera": PATHS.camera,
    }
    missing = [label for label, path in important.items() if not stage.GetPrimAtPath(path).IsValid()]
    if missing:
        details = ", ".join(f"{label}={important[label]}" for label in missing)
        raise RuntimeError(f"Scene validation failed; missing prims: {details}")
    return important


def write_anchors(stage: Usd.Stage, layout: Layout, fls_info: AssetInfo, output: Path) -> Path:
    """Save world-frame task anchors next to the USD for tools that run outside Isaac Sim."""
    link0 = None
    for prim in Usd.PrimRange(stage.GetPrimAtPath(PATHS.panda)):
        if prim.GetName() == "panda_link0":
            link0 = prim
            break
    if link0 is None:
        raise RuntimeError(f"Could not find panda_link0 under {PATHS.panda}")
    base = UsdGeom.XformCache(Usd.TimeCode.Default()).GetLocalToWorldTransform(link0).ExtractTranslation()
    radius = float(stage.GetPrimAtPath(PATHS.target).GetAttribute("task:radius").Get())
    anchors = SceneAnchors(
        object_start=[float(v) for v in layout.fls_center_world],
        target_center=[float(v) for v in layout.target_center],
        target_radius=radius,
        table_top_z=float(layout.table_top_z),
        robot_base=[float(v) for v in base],
        table_min=[float(v) for v in layout.table_min_world],
        table_max=[float(v) for v in layout.table_max_world],
        object_height=float(fls_info.bounds.size[2]),
    )
    path = output.with_suffix(".anchors.json")
    anchors.save(path)
    return path


def build_scene() -> tuple[Usd.Stage, AssetInfo, Layout, dict[str, str]]:
    output = ARGS.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    panda_asset = ASSET_ROOT / "panda_instanceable.usd"
    table_asset = ASSET_ROOT / "sm_table.usd"
    fls_asset = ASSET_ROOT / "fls_block.usd"
    for asset in [panda_asset, table_asset, fls_asset]:
        if not asset.exists():
            raise FileNotFoundError(asset)

    panda_info = inspect_asset(panda_asset)
    require_panda_paths(panda_info)
    table_info = inspect_asset(table_asset)
    fls_info = inspect_asset(fls_asset)
    layout = compute_layout(panda_info, table_info, fls_info)

    print(f"[layout] Panda translation: {fmt_vec(layout.panda_translation)}", flush=True)
    print(f"[layout] Table top z: {layout.table_top_z:.4f}", flush=True)
    print(f"[layout] FLS center: {fmt_vec(layout.fls_center_world)}", flush=True)
    print(f"[layout] Target center: {fmt_vec(layout.target_center)}", flush=True)

    stage = Usd.Stage.CreateNew(str(output))
    stage.SetStartTimeCode(0)
    stage.SetEndTimeCode(240)
    stage.SetFramesPerSecond(60)
    stage.SetTimeCodesPerSecond(60)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, PATHS.world).GetPrim()
    stage.SetDefaultPrim(world)
    create_physics_scene(stage)

    blue = create_preview_material(stage, f"{PATHS.looks}/FLS_BrightBlue", (0.0, 0.22, 1.0), roughness=0.35)
    table_mat = create_preview_material(stage, f"{PATHS.looks}/Table_LightNeutral", (0.78, 0.80, 0.78), roughness=0.65)
    ground_mat = create_preview_material(stage, f"{PATHS.looks}/Ground_MatteGray", (0.62, 0.64, 0.62), roughness=0.8)

    add_reference(stage, PATHS.panda, panda_asset, layout.panda_translation)
    add_reference(stage, PATHS.table, table_asset, layout.table_translation)
    add_reference(stage, PATHS.fls, fls_asset, layout.fls_translation)

    bind_material_to_gprims(stage, PATHS.fls, blue)
    bind_material_to_gprims(stage, PATHS.table, table_mat)
    ensure_static_colliders(stage, PATHS.table)
    ensure_dynamic_fls(stage, PATHS.fls, ARGS.fls_mass)
    limit_gripper_force(stage, panda_info, ARGS.gripper_force)
    preapply_gripper_rigid_body_api(stage, panda_info)

    create_target(stage, layout)
    create_ground(stage, layout, ground_mat)
    add_lighting(stage, layout)
    create_camera(stage, layout)
    create_task_metadata(stage, layout, panda_info)

    important = validate_prims(stage, panda_info)
    stage.GetRootLayer().Save()
    print(f"[save] Wrote {output}", flush=True)
    print(f"[save] Wrote {write_anchors(stage, layout, fls_info, output)}", flush=True)
    return stage, panda_info, layout, important


def reopen_in_isaac(output: Path, layout: Layout) -> None:
    context = omni.usd.get_context()
    context.open_stage(str(output))
    for _ in range(5):
        simulation_app.update()
    try:
        from isaacsim.core.utils.viewports import set_active_viewport_camera, set_camera_view

        set_camera_view(
            eye=[float(v) for v in layout.camera_eye],
            target=[float(v) for v in layout.camera_target],
            camera_prim_path=PATHS.camera,
        )
        set_active_viewport_camera(PATHS.camera)
    except Exception as exc:
        print(f"[warn] Could not set active viewport camera: {exc}", flush=True)


def print_summary(important: dict[str, str], layout: Layout) -> None:
    print("\nImportant prim paths", flush=True)
    for label, path in important.items():
        print(f"  {label}: {path}", flush=True)
    print("\nTask anchors", flush=True)
    print(f"  Initial blue FLS center: {fmt_vec(layout.fls_center_world)}", flush=True)
    print(f"  Target center: {fmt_vec(layout.target_center)}", flush=True)
    print(f"  Camera eye -> target: {fmt_vec(layout.camera_eye)} -> {fmt_vec(layout.camera_target)}", flush=True)


if __name__ == "__main__":
    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing():
        timeline.stop()

    stage, panda_info, layout, important = build_scene()
    print_summary(important, layout)

    output_path = ARGS.output.resolve()
    if not ARGS.no_open:
        reopen_in_isaac(output_path, layout)

    keep_open = not ARGS.headless and not ARGS.build_only
    if keep_open:
        print("\n[ready] Scene is open in Isaac Sim. Close the Isaac window to end this process.", flush=True)
        while simulation_app.is_running():
            simulation_app.update()

    simulation_app.close()
