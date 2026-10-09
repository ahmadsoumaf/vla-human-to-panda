#!/usr/bin/env python3
"""Retarget a processed 2D human demonstration onto the Panda in the Isaac FLS scene.

Version 1 (2D, no depth):
  * Image -> table plane: a 2D similarity transform (rotation + uniform scale + translation)
    fitted so the real object position at grasp time lands on the sim object and the real
    target lands on the sim target.  This uses the object/target relationship instead of
    absolute pixels, so camera resolution, zoom and in-plane rotation do not matter.
    The camera is assumed to look down at the table (image y is flipped to get a
    right-handed table frame); perspective distortion is ignored.
  * The human grasp point (thumb/index midpoint) drives Panda X/Y.  It is gap-filled,
    smoothed, then corrected with a slowly varying offset so the path ends exactly on the
    sim object at the grasp and on the human's placement point at the release.
  * Z is not estimated from video.  It comes from the task phase:
      approach / transport / retreat at a safe height, vertical descents to the known grasp
      and place heights, gripper closed between the human's pinch and release.
  * Paths are retimed so the TCP never exceeds a maximum speed, and clamped to the
    reachable workspace.

Usable as a library (`retarget(...)`, used by replay_human_demo.py) or from the command line
to inspect a plan without starting Isaac Sim:
  python3 retargeting/human_to_panda.py data/real/processed/demo_001.npz --plot
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "isaac"))

from task_geometry import (  # noqa: E402
    DEFAULT_ANCHORS,
    GRIPPER_CLOSED,
    GRIPPER_OPEN,
    PANDA_MAX_REACH,
    PANDA_MIN_REACH,
    SceneAnchors,
    min_tcp_z,
)

PHASES = [
    "approach",
    "descend_grasp",
    "close",
    "lift",
    "transport",
    "descend_place",
    "open",
    "lift_after_place",
    "retreat",
]
PATH_PHASES = {"approach", "transport", "retreat"}
REQUIRED_KEYS = ["timestamps", "image_size", "grasp_point_xy", "object_xy", "object_visible", "target_xy", "grasp_state"]


VALIDATION_PHASES = ("approach", "grasp", "carry", "release", "post-release")


class RetargetError(RuntimeError):
    """The demonstration cannot be turned into a safe Panda plan.

    `phase` names the task phase whose validation failed (one of VALIDATION_PHASES), if any.
    """

    def __init__(self, message: str, phase: str | None = None) -> None:
        self.phase = phase
        super().__init__(f"[{phase}] {message}" if phase else message)


@dataclass
class RetargetConfig:
    control_dt: float = 1.0 / 60.0
    smoothing_sigma_s: float = 0.10
    max_speed: float = 0.20  # m/s, TCP speed limit along the X/Y paths
    time_scale: float = 1.0  # > 1 replays slower than the human
    safe_height: float = 0.15  # above the grasp height
    grasp_z_offset: float = 0.0
    fingertip_clearance: float = 0.004
    min_closed_s: float = 0.25
    anchor_window_s: float = 0.3
    max_hand_gap_s: float = 0.6
    # Fraction of the hand's deviation perpendicular to the object->target line that is kept.
    # From a low/oblique camera, image motion off that line is mostly lifting (height and depth
    # are not separable in 2D), so it is dropped (0); an overhead camera can keep it (1).
    lateral_gain: float = 0.0
    min_anchor_separation_fraction: float = 0.05  # of the image diagonal
    # Demonstration quality (for dataset curation, not for whether a replay is possible).
    max_quality_correction_m: float = 0.05  # hand-to-object jump at grasp/release
    max_quality_object_gap_before_grasp_s: float = 0.3  # object / hand last seen this close to grasp / release
    workspace_margin: float = 0.05
    descend_s: float = 1.2
    gripper_s: float = 1.5
    lift_s: float = 1.2


@dataclass
class PandaPlan:
    times: np.ndarray
    positions: np.ndarray  # (M, 3) TCP (Lula right_gripper) targets in world frame
    gripper: np.ndarray  # (M,) finger joint targets
    phase: np.ndarray  # (M,) index into PHASES
    grasp_position: np.ndarray
    place_position: np.ndarray
    place_object_center: np.ndarray
    safe_z: float
    image_to_table: np.ndarray  # (2, 3) affine on pixel coords -> world x/y
    meters_per_pixel: float
    human_grasp_t: float
    human_release_t: float
    human_hand_xy_world: np.ndarray  # mapped + smoothed hand path before correction
    clamped_points: int
    lateral_gain: float = 1.0
    warnings: list[str] = field(default_factory=list)
    quality: dict = field(default_factory=dict)

    def segments(self):
        """Yield (phase_name, positions, gripper) for consecutive runs of the same phase."""
        boundaries = np.flatnonzero(np.diff(self.phase)) + 1
        for chunk in np.split(np.arange(len(self.phase)), boundaries):
            yield PHASES[int(self.phase[chunk[0]])], self.positions[chunk], self.gripper[chunk]

    def save(self, path: Path, config: RetargetConfig | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        extra = {f"config_{k}": np.float64(v) for k, v in asdict(config).items()} if config else {}
        np.savez_compressed(
            path,
            times=self.times,
            positions=self.positions,
            gripper=self.gripper,
            phase=self.phase,
            phase_names=np.array(PHASES),
            grasp_position=self.grasp_position,
            place_position=self.place_position,
            place_object_center=self.place_object_center,
            safe_z=np.float64(self.safe_z),
            image_to_table=self.image_to_table,
            meters_per_pixel=np.float64(self.meters_per_pixel),
            human_grasp_t=np.float64(self.human_grasp_t),
            human_release_t=np.float64(self.human_release_t),
            human_hand_xy_world=self.human_hand_xy_world,
            clamped_points=np.int64(self.clamped_points),
            warnings=np.array(self.warnings, dtype=str),
            **{f"quality_{k}": np.array(v) for k, v in self.quality.items()},
            **extra,
        )

    def summary(self) -> str:
        lines = [
            f"plan: {len(self.times)} samples, {self.times[-1]:.2f} s at {1.0 / np.median(np.diff(self.times)):.0f} Hz",
            f"  image->table scale: {self.meters_per_pixel * 1000:.3f} mm/px   lateral gain: {self.lateral_gain:g}",
            f"  human grasp at {self.human_grasp_t:.2f} s, release at {self.human_release_t:.2f} s",
            f"  grasp TCP: ({self.grasp_position[0]: .4f}, {self.grasp_position[1]: .4f}, {self.grasp_position[2]: .4f})",
            f"  place TCP: ({self.place_position[0]: .4f}, {self.place_position[1]: .4f}, {self.place_position[2]: .4f})",
            f"  safe z: {self.safe_z:.4f}   clamped points: {self.clamped_points}",
        ]
        for name, positions, _ in self.segments():
            lines.append(f"  {name:<17s} {len(positions):5d} samples ({len(positions) * (self.times[1] - self.times[0]):5.2f} s)")
        q = self.quality
        if q:
            lines.append(
                f"  quality: pick observed={q['pick_observed']}  grasp jump={q['grasp_correction_m'] * 100:.1f} cm  "
                f"release jump={q['release_correction_m'] * 100:.1f} cm  -> "
                + ("GOOD demonstration" if q["good_demonstration"] else "POOR demonstration: " + "; ".join(q["issues"]))
            )
        for w in self.warnings:
            lines.append(f"  [warn] {w}")
        return "\n".join(lines)


# ----------------------------------------------------------------------------- helpers


def load_demo(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Processed demo not found: {path}")
    data = dict(np.load(path, allow_pickle=False))
    missing = [k for k in REQUIRED_KEYS if k not in data]
    if missing:
        raise RetargetError(f"{path} is missing keys {missing}; re-run process_demo.py")
    return data


def to_complex(xy: np.ndarray) -> np.ndarray:
    """Pixel coords -> complex numbers in a right-handed frame (image y points down)."""
    xy = np.asarray(xy, dtype=np.float64)
    return xy[..., 0] - 1j * xy[..., 1]


def fit_similarity(src: tuple[complex, complex], dst: tuple[complex, complex]) -> tuple[complex, complex]:
    """w = a * z + b mapping two source points onto two destination points."""
    a = (dst[1] - dst[0]) / (src[1] - src[0])
    return a, dst[0] - a * src[0]


def similarity_matrix(a: complex) -> np.ndarray:
    """2x2 linear part of w = a*z (with the image y flip folded in) for pixel (x, y)."""
    # w = a * (x - i y): real = ar x + ai y, imag = ai x - ar y
    return np.array([[a.real, a.imag], [a.imag, -a.real]])


def gaussian_smooth(values: np.ndarray, sigma_samples: float) -> np.ndarray:
    if sigma_samples <= 0.5 or len(values) < 3:
        return values.copy()
    radius = int(np.ceil(3 * sigma_samples))
    kernel = np.exp(-0.5 * (np.arange(-radius, radius + 1) / sigma_samples) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, ((radius, radius), (0, 0)), mode="edge")
    return np.stack([np.convolve(padded[:, k], kernel, mode="valid") for k in range(values.shape[1])], axis=1)


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def tracked_chain(t: np.ndarray, ok: np.ndarray, start: int, max_gap_s: float, step: int) -> int:
    """Walk from tracked frame `start` in direction `step` while consecutive tracked frames are at
    most `max_gap_s` apart; return the last frame reached."""
    frames = np.flatnonzero(ok)
    frames = frames[frames < start][::-1] if step < 0 else frames[frames > start]
    current = start
    for frame in frames:
        if abs(t[frame] - t[current]) > max_gap_s:
            break
        current = int(frame)
    return current


def longest_gap(t: np.ndarray, ok: np.ndarray, lo: int, hi: int) -> tuple[float, float, float]:
    """Longest untracked stretch inside [lo, hi] as (duration, start_t, end_t), endpoints included."""
    frames = np.flatnonzero(ok[lo : hi + 1]) + lo
    if len(frames) == 0:
        return float(t[hi] - t[lo]), float(t[lo]), float(t[hi])
    times = np.concatenate([[t[lo]], t[frames], [t[hi]]])
    gaps = np.diff(times)
    k = int(np.argmax(gaps))
    return float(gaps[k]), float(times[k]), float(times[k + 1])


def fill_range(t: np.ndarray, xy: np.ndarray, ok: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Linear interpolation of tracked rows over [lo, hi] (gap lengths validated by the caller)."""
    idx = np.flatnonzero(ok[lo : hi + 1]) + lo
    seg_t = t[lo : hi + 1]
    return np.stack([np.interp(seg_t, t[idx], xy[idx, k]) for k in range(2)], axis=1)


def windowed_median(xy: np.ndarray, visible: np.ndarray, lo: int, hi: int) -> np.ndarray | None:
    lo, hi = max(lo, 0), min(hi, len(xy))
    rows = xy[lo:hi][visible[lo:hi] & np.all(np.isfinite(xy[lo:hi]), axis=1)]
    return np.median(rows, axis=0) if len(rows) else None


def retime(points: np.ndarray, t: np.ndarray, cfg: RetargetConfig) -> np.ndarray:
    """Resample a timed XY path at the control rate, keeping the human's timing except where
    a step would exceed cfg.max_speed; only those steps are stretched (local time warp)."""
    if len(points) < 2 or t[-1] <= t[0]:
        return points[:1].copy()
    step_len = np.linalg.norm(np.diff(points, axis=0), axis=1)
    step_dt = np.maximum(np.diff(t) * cfg.time_scale, step_len / cfg.max_speed)
    new_t = np.concatenate([[0.0], np.cumsum(step_dt)])
    n = max(2, int(np.ceil(new_t[-1] / cfg.control_dt)) + 1)
    grid = np.linspace(0.0, new_t[-1], n)
    return np.stack([np.interp(grid, new_t, points[:, k]) for k in range(points.shape[1])], axis=1)


def clamp_to_workspace(xy: np.ndarray, anchors: SceneAnchors, cfg: RetargetConfig) -> tuple[np.ndarray, np.ndarray]:
    base = np.array(anchors.robot_base[:2])
    lo = np.array(anchors.table_min[:2]) + cfg.workspace_margin
    hi = np.array(anchors.table_max[:2]) - cfg.workspace_margin
    out = np.clip(xy, lo, hi)
    rel = out - base
    radius = np.linalg.norm(rel, axis=1, keepdims=True)
    clipped_radius = np.clip(radius, PANDA_MIN_REACH, PANDA_MAX_REACH)
    out = base + rel * (clipped_radius / np.maximum(radius, 1e-9))
    changed = np.linalg.norm(out - xy, axis=1) > 1e-6
    return out, changed


# ----------------------------------------------------------------------------- main entry


def retarget(demo: dict[str, np.ndarray], anchors: SceneAnchors, cfg: RetargetConfig | None = None) -> PandaPlan:
    cfg = cfg or RetargetConfig()
    warnings: list[str] = []
    t = np.asarray(demo["timestamps"], dtype=np.float64)
    fps = 1.0 / float(np.median(np.diff(t)))
    width, height = (float(v) for v in demo["image_size"])
    diagonal = float(np.hypot(width, height))
    grasp_state = np.asarray(demo["grasp_state"]).astype(bool)
    object_xy = np.asarray(demo["object_xy"], dtype=np.float64)
    object_visible = np.asarray(demo["object_visible"]).astype(bool)

    # 1. Grasp / release events: the longest closed interval of the human pinch.
    padded = np.concatenate([[False], grasp_state, [False]]).astype(np.int8)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1) - 1
    if len(starts) == 0:
        raise RetargetError("no grasp in the demo (grasp_state never closes); check the pinch thresholds", phase="grasp")
    longest = int(np.argmax(ends - starts))
    i_grasp, i_release = int(starts[longest]), int(ends[longest])
    if t[i_release] - t[i_grasp] < cfg.min_closed_s:
        raise RetargetError(
            f"longest grasp lasts only {t[i_release] - t[i_grasp]:.2f} s (< {cfg.min_closed_s} s); likely tracking noise",
            phase="grasp",
        )
    if len(starts) > 1:
        warnings.append(f"{len(starts)} grasp intervals found; using the longest ({t[i_grasp]:.2f}-{t[i_release]:.2f} s)")
    if i_release >= len(t) - 1:
        warnings.append("demo ends while the hand is still closed; release placed at the last frame")

    # 2. Phase-aware tracking validation.  Hand data is only required where the robot copies it:
    #      approach      copied only over the continuously tracked stretch right before the grasp
    #      grasp         strict: the hand must be seen at the grasp (no guessing about the pick)
    #      carry         hand OR carried object, gaps <= max_hand_gap
    #      release       hand just before it OR object just after it
    #      post-release  nothing required; the retreat is copied only while the hand is tracked
    max_gap = cfg.max_hand_gap_s
    near = max_gap / 2
    hand_raw = np.array(demo["grasp_point_xy"], dtype=np.float64)
    hand_ok = np.all(np.isfinite(hand_raw), axis=1)
    obj_ok = object_visible & np.all(np.isfinite(object_xy), axis=1)

    around_grasp = np.flatnonzero(hand_ok & (np.abs(t - t[i_grasp]) <= near))
    if len(around_grasp) == 0:
        raise RetargetError(
            f"hand not visible within {near:.2f} s of the grasp at {t[i_grasp]:.2f} s; "
            "cannot tell whether the object was picked up",
            phase="grasp",
        )

    # Carry: frames without the hand but with the object follow the object (+ hand-object offset).
    held = np.zeros(len(t), dtype=bool)
    held[i_grasp : i_release + 1] = True
    both = held & hand_ok & obj_ok
    if both.any():
        from_object = held & ~hand_ok & obj_ok
        if from_object.any():
            hand_raw[from_object] = object_xy[from_object] + np.median(hand_raw[both] - object_xy[both], axis=0)
            hand_ok = hand_ok | from_object
            warnings.append(f"{int(from_object.sum())} carry frames follow the object (hand out of view)")
    gap, gap_start, gap_end = longest_gap(t, hand_ok, i_grasp, i_release)
    if gap > max_gap:
        raise RetargetError(
            f"neither hand nor object tracked for {gap:.2f} s ({gap_start:.2f}-{gap_end:.2f} s), "
            f"more than --max-hand-gap {max_gap} s; the carried path there would be a guess",
            phase="carry",
        )

    release_window = (t >= t[i_release] - near) & (t <= t[i_release] + near)
    if not ((hand_ok & release_window & (t <= t[i_release])).any() or (obj_ok & release_window & (t >= t[i_release])).any()):
        raise RetargetError(
            f"neither the hand before nor the object after the release at {t[i_release]:.2f} s is tracked",
            phase="release",
        )

    # Approach: the tracked stretch leading into the grasp (earlier, separated motion is not copied).
    before = np.flatnonzero(hand_ok[: i_grasp + 1] & (t[: i_grasp + 1] >= t[i_grasp] - near))
    approach_start = tracked_chain(t, hand_ok, int(before[-1]), max_gap, -1) if len(before) else i_grasp
    approach_start = min(approach_start, i_grasp)
    if approach_start >= i_grasp:
        warnings.append("approach not observed (hand entered view already grasping); robot goes straight to the object")
    elif hand_ok[:approach_start].any():
        warnings.append(
            f"approach copied from {t[approach_start]:.2f} s; earlier hand motion is separated by a tracking gap "
            f"> {max_gap} s and is not copied"
        )

    # Post-release: copy the retreat only while the hand stays tracked; its absence is fine.
    after = np.flatnonzero(hand_ok & (t >= t[i_release]) & (t <= t[i_release] + max_gap))
    retreat_end = tracked_chain(t, hand_ok, int(after[0]), max_gap, +1) if len(after) else i_release
    retreat_end = max(retreat_end, i_release)
    if retreat_end < len(t) - 1 and hand_ok[retreat_end + 1 :].any():
        warnings.append(f"post-release hand tracking ends at {t[retreat_end]:.2f} s; later motion is not needed")

    lo, hi = approach_start, retreat_end
    hand_px = np.full_like(hand_raw, np.nan)
    hand_px[lo : hi + 1] = fill_range(t, hand_raw, hand_ok, lo, hi)

    # 3. Real anchors: object just before the grasp, target, and object just after release.
    window = max(1, int(round(cfg.anchor_window_s * fps)))
    obj_at_grasp = windowed_median(object_xy, object_visible, i_grasp - window, i_grasp + 1)
    anchor_right_before_grasp = obj_at_grasp is not None
    if obj_at_grasp is None:
        obj_at_grasp = windowed_median(object_xy, object_visible, 0, i_grasp + 1)
        if obj_at_grasp is None:
            raise RetargetError("blue object never detected before the grasp; cannot anchor the mapping", phase="grasp")
        warnings.append("object not visible right before the grasp; using its earlier position")
    target_px = np.median(np.asarray(demo["target_xy"], dtype=np.float64), axis=0)
    if not np.all(np.isfinite(target_px)):
        raise RetargetError("target position missing (no stable object position after the release?)", phase="post-release")
    if np.linalg.norm(target_px - obj_at_grasp) < cfg.min_anchor_separation_fraction * diagonal:
        raise RetargetError("object and target are almost at the same image position; cannot fit the mapping", phase="release")

    # 4. Image -> table similarity transform anchored on object and target.
    sim_obj = complex(anchors.object_start[0], anchors.object_start[1])
    sim_tgt = complex(anchors.target_center[0], anchors.target_center[1])
    a, b = fit_similarity((to_complex(obj_at_grasp), to_complex(target_px)), (sim_obj, sim_tgt))
    to_world = lambda px: (lambda w: np.stack([w.real, w.imag], axis=-1))(a * to_complex(px) + b)  # noqa: E731
    meters_per_pixel = abs(a)

    hand_world = np.full_like(hand_px, np.nan)
    hand_world[lo : hi + 1] = gaussian_smooth(to_world(hand_px[lo : hi + 1]), cfg.smoothing_sigma_s * fps)
    if cfg.lateral_gain != 1.0:
        origin = np.array(anchors.object_start[:2])
        axis = np.array(anchors.target_center[:2]) - origin
        axis /= np.linalg.norm(axis)
        normal = np.array([-axis[1], axis[0]])
        rel = hand_world - origin
        hand_world = origin + np.outer(rel @ axis, axis) + cfg.lateral_gain * np.outer(rel @ normal, normal)

    # Where the human put the object relative to the target (falls back to the target center).
    # In release mode the demonstrated target *is* the resting place, so place exactly on it.
    release_mode = str(demo.get("target_mode", "marker")) == "release"
    obj_after = None if release_mode else windowed_median(object_xy, object_visible, i_release, i_release + 2 * window)
    if release_mode:
        place_xy = np.array(anchors.target_center[:2])
    elif obj_after is None:
        place_xy = np.array(anchors.target_center[:2])
        warnings.append("object not visible after release; placing at the target center")
    else:
        place_xy = to_world(obj_after)
        offset = float(np.linalg.norm(place_xy - np.array(anchors.target_center[:2])))
        if offset > anchors.target_radius:
            warnings.append(f"human placed the object {offset * 100:.1f} cm from the target center (outside the target)")

    # 5. Offset correction so the path hits the sim object at grasp and the placement at release.
    grasp_xy = np.array(anchors.object_start[:2])
    e_grasp = grasp_xy - hand_world[i_grasp]
    e_release = place_xy - hand_world[i_release]
    idx = np.arange(len(t))
    w_approach = smoothstep((idx - lo) / max(i_grasp - lo, 1))[:, None]
    w_transport = np.clip((idx - i_grasp) / max(i_release - i_grasp, 1), 0.0, 1.0)[:, None]
    offset = np.where(
        (idx <= i_grasp)[:, None],
        w_approach * e_grasp,
        (1 - w_transport) * e_grasp + w_transport * e_release,
    )
    corrected = hand_world + offset
    corrected[i_grasp] = grasp_xy
    corrected[i_release] = place_xy
    for label, err in (("grasp", e_grasp), ("release", e_release)):
        if np.linalg.norm(err) > 0.10:
            warnings.append(f"hand-to-object offset at {label} is {np.linalg.norm(err) * 100:.1f} cm (large correction)")

    # Demonstration quality: was the real pick observed, and how much had to be corrected?
    seen_before = np.flatnonzero(obj_ok[:i_grasp])
    object_gap_before_grasp = float(t[i_grasp] - t[seen_before[-1]]) if len(seen_before) else float("inf")
    grasp_jump = float(np.linalg.norm(e_grasp))
    release_jump = float(np.linalg.norm(e_release))
    pick_observed = (
        anchor_right_before_grasp
        and object_gap_before_grasp <= cfg.max_quality_object_gap_before_grasp_s
        and grasp_jump <= cfg.max_quality_correction_m
    )
    issues = []
    if not pick_observed:
        issues.append(
            f"real pick not observed (object last seen {object_gap_before_grasp:.2f} s before the detected grasp, "
            f"grasp jump {grasp_jump * 100:.1f} cm)"
        )
    # The hand-to-object jump at release is reported but not gated: releases are often inferred
    # when the hand leaves view, by which time it is already rising (image-vertical from an
    # oblique camera).  What matters is that the hand was tracked up to the release.
    hand_seen_before_release = np.flatnonzero(np.all(np.isfinite(demo["grasp_point_xy"][: i_release + 1]), axis=1))
    hand_gap_before_release = float(t[i_release] - t[hand_seen_before_release[-1]]) if len(hand_seen_before_release) else float("inf")
    release_observed = hand_gap_before_release <= cfg.max_quality_object_gap_before_grasp_s
    if not release_observed:
        issues.append(f"release not observed (hand last seen {hand_gap_before_release:.2f} s before it)")
    quality = dict(
        pick_observed=bool(pick_observed),
        approach_observed=bool(lo < i_grasp),
        object_gap_before_grasp_s=object_gap_before_grasp,
        grasp_correction_m=grasp_jump,
        release_correction_m=release_jump,
        release_observed=bool(release_observed),
        good_demonstration=not issues,
        issues=issues,
    )

    # 6. Workspace limits; the grasp and place points themselves must be reachable.
    corrected[:lo] = corrected[lo]
    corrected[hi + 1 :] = corrected[hi]
    clamped, changed = clamp_to_workspace(corrected, anchors, cfg)
    changed[:lo] = False
    changed[hi + 1 :] = False
    for label, i in (("grasp", i_grasp), ("place", i_release)):
        if changed[i]:
            raise RetargetError(
                f"{label} point {corrected[i]} is outside the reachable workspace",
                phase="grasp" if label == "grasp" else "release",
            )
    if changed.any():
        warnings.append(f"{int(changed.sum())} path samples clamped to the reachable workspace")
        issues.append(f"{int(changed.sum())} path samples clamped to the workspace")
        quality["good_demonstration"] = False

    # 7. Heights from the task phase (no depth from video).
    table_top = anchors.table_top_z
    lowest = min_tcp_z(table_top, cfg.fingertip_clearance)
    grasp_z = max(anchors.object_start[2] + cfg.grasp_z_offset, lowest)
    place_object_z = table_top + anchors.object_height * 0.5 + 0.003
    place_z = max(place_object_z + cfg.grasp_z_offset, lowest)
    safe_z = grasp_z + cfg.safe_height

    # 8. Assemble the timed plan.
    pieces: list[tuple[str, np.ndarray, np.ndarray]] = []

    def path(name: str, lo: int, hi: int, z: float, grip: float) -> None:
        xy = retime(clamped[lo : hi + 1], t[lo : hi + 1], cfg)
        pieces.append((name, np.column_stack([xy, np.full(len(xy), z)]), np.full(len(xy), grip)))

    def vertical(name: str, xy: np.ndarray, z0: float, z1: float, grip: float, seconds: float) -> None:
        n = max(2, int(round(seconds * cfg.time_scale / cfg.control_dt)))
        z = z0 + (z1 - z0) * smoothstep(np.linspace(0.0, 1.0, n))
        pieces.append((name, np.column_stack([np.repeat(xy[None, :], n, axis=0), z]), np.full(n, grip)))

    def gripper(name: str, xy: np.ndarray, z: float, g0: float, g1: float) -> None:
        n = max(2, int(round(cfg.gripper_s * cfg.time_scale / cfg.control_dt)))
        pos = np.repeat(np.array([[xy[0], xy[1], z]]), n, axis=0)
        pieces.append((name, pos, g0 + (g1 - g0) * np.linspace(0.0, 1.0, n)))

    path("approach", lo, i_grasp, safe_z, GRIPPER_OPEN)
    vertical("descend_grasp", grasp_xy, safe_z, grasp_z, GRIPPER_OPEN, cfg.descend_s)
    gripper("close", grasp_xy, grasp_z, GRIPPER_OPEN, GRIPPER_CLOSED)
    vertical("lift", grasp_xy, grasp_z, safe_z, GRIPPER_CLOSED, cfg.lift_s)
    path("transport", i_grasp, i_release, safe_z, GRIPPER_CLOSED)
    vertical("descend_place", place_xy, safe_z, place_z, GRIPPER_CLOSED, cfg.descend_s)
    gripper("open", place_xy, place_z, GRIPPER_CLOSED, GRIPPER_OPEN)
    vertical("lift_after_place", place_xy, place_z, safe_z, GRIPPER_OPEN, cfg.lift_s)
    if hi > i_release:
        path("retreat", i_release, hi, safe_z, GRIPPER_OPEN)

    positions = np.concatenate([p for _, p, _ in pieces])
    grippers = np.concatenate([g for _, _, g in pieces])
    phases = np.concatenate([np.full(len(p), PHASES.index(name)) for name, p, _ in pieces])
    times = np.arange(len(positions)) * cfg.control_dt

    return PandaPlan(
        times=times,
        positions=positions,
        gripper=grippers,
        phase=phases,
        grasp_position=np.array([grasp_xy[0], grasp_xy[1], grasp_z]),
        place_position=np.array([place_xy[0], place_xy[1], place_z]),
        place_object_center=np.array([place_xy[0], place_xy[1], place_object_z]),
        safe_z=float(safe_z),
        image_to_table=np.column_stack([similarity_matrix(a), [b.real, b.imag]]),
        meters_per_pixel=float(meters_per_pixel),
        human_grasp_t=float(t[i_grasp]),
        human_release_t=float(t[i_release]),
        human_hand_xy_world=hand_world[lo : hi + 1],
        clamped_points=int(changed.sum()),
        quality=quality,
        lateral_gain=cfg.lateral_gain,
        warnings=warnings,
    )


# ----------------------------------------------------------------------------- top-down plot


def plot_plan(plan: PandaPlan, anchors: SceneAnchors, path: Path) -> bool:
    """Top-down sketch of the retargeted plan on the table (needs OpenCV)."""
    try:
        import cv2
    except ImportError:
        print("[plot] OpenCV not available; skipping the plan image", flush=True)
        return False
    px_per_m = 600
    lo = np.array(anchors.table_min[:2]) - 0.1
    hi = np.array(anchors.table_max[:2]) + 0.1
    size = ((hi - lo) * px_per_m).astype(int)
    img = np.full((size[1], size[0], 3), 255, np.uint8)
    to_px = lambda xy: np.column_stack([(xy[:, 0] - lo[0]) * px_per_m, (hi[1] - xy[:, 1]) * px_per_m]).astype(np.int32)  # noqa: E731

    table = to_px(np.array([anchors.table_min[:2], [anchors.table_max[0], anchors.table_min[1]], anchors.table_max[:2], [anchors.table_min[0], anchors.table_max[1]]]))
    cv2.polylines(img, [table.reshape(-1, 1, 2)], True, (150, 150, 150), 2)
    base = to_px(np.array([anchors.robot_base[:2]]))[0]
    for r in (PANDA_MIN_REACH, PANDA_MAX_REACH):
        cv2.circle(img, tuple(base), int(r * px_per_m), (210, 210, 210), 1, cv2.LINE_AA)
    cv2.circle(img, tuple(base), 10, (80, 80, 80), -1)
    tgt = to_px(np.array([anchors.target_center[:2]]))[0]
    cv2.circle(img, tuple(tgt), int(anchors.target_radius * px_per_m), (0, 180, 0), 2, cv2.LINE_AA)

    raw = to_px(plan.human_hand_xy_world)
    cv2.polylines(img, [raw.reshape(-1, 1, 2)], False, (200, 200, 255), 1, cv2.LINE_AA)
    colors = {"approach": (0, 160, 0), "transport": (0, 0, 220), "retreat": (130, 130, 130)}
    for name, positions, _ in plan.segments():
        if name in colors:
            pts = to_px(positions[:, :2])
            cv2.polylines(img, [pts.reshape(-1, 1, 2)], False, colors[name], 2, cv2.LINE_AA)
    obj = to_px(np.array([anchors.object_start[:2]]))[0]
    cv2.drawMarker(img, tuple(obj), (255, 0, 0), cv2.MARKER_TRIANGLE_UP, 18, 2)
    cv2.drawMarker(img, tuple(to_px(plan.place_position[None, :2])[0]), (0, 0, 220), cv2.MARKER_TILTED_CROSS, 16, 2)
    legend = [
        ("approach (open)", colors["approach"]),
        ("transport (closed)", colors["transport"]),
        ("retreat", colors["retreat"]),
        ("raw mapped hand", (200, 200, 255)),
        ("target", (0, 180, 0)),
        ("object (blue triangle)", (255, 0, 0)),
    ]
    for k, (text, color) in enumerate(legend):
        cv2.putText(img, text, (10, 22 + 20 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)
    print(f"[plot] saved {path}", flush=True)
    return True


def add_config_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("retargeting")
    defaults = RetargetConfig()
    group.add_argument("--max-speed", type=float, default=defaults.max_speed, help="Max TCP speed along X/Y paths (m/s).")
    group.add_argument("--time-scale", type=float, default=defaults.time_scale, help="> 1 replays slower than the human.")
    group.add_argument("--smoothing", type=float, default=defaults.smoothing_sigma_s, help="Gaussian smoothing sigma (s).")
    group.add_argument("--safe-height", type=float, default=defaults.safe_height, help="Transport height above the grasp (m).")
    group.add_argument(
        "--camera-view",
        choices=["oblique", "overhead"],
        default="oblique",
        help="oblique (low side view): ignore image motion off the object->target line, which is mostly "
        "lifting; overhead: keep it as real sideways motion.",
    )
    group.add_argument("--lateral-gain", type=float, help="Override the kept fraction of off-line motion (0..1).")
    group.add_argument("--max-hand-gap", type=float, default=defaults.max_hand_gap_s, help="Longest tolerated hand-tracking gap (s).")


def config_from_args(args: argparse.Namespace) -> RetargetConfig:
    return RetargetConfig(
        max_speed=args.max_speed,
        time_scale=args.time_scale,
        smoothing_sigma_s=args.smoothing,
        safe_height=args.safe_height,
        max_hand_gap_s=args.max_hand_gap,
        lateral_gain=args.lateral_gain if args.lateral_gain is not None else (0.0 if args.camera_view == "oblique" else 1.0),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Retarget a processed human demo to a Panda TCP plan (no Isaac needed).")
    parser.add_argument("demo", type=Path, help="Processed demo .npz from process_demo.py")
    parser.add_argument("--anchors", type=Path, default=DEFAULT_ANCHORS, help="Scene anchors JSON written by the scene builder.")
    parser.add_argument("--output", type=Path, help="Plan .npz (default: data/real/processed/<name>_panda_plan.npz)")
    parser.add_argument("--plot", action="store_true", help="Also save a top-down plan image to data/real/debug/")
    add_config_args(parser)
    args = parser.parse_args(argv)

    cfg = config_from_args(args)
    try:
        plan = retarget(load_demo(args.demo), SceneAnchors.load(args.anchors), cfg)
    except (RetargetError, FileNotFoundError) as exc:
        print(f"[error] {exc}", file=sys.stderr, flush=True)
        return 1
    print(plan.summary(), flush=True)
    output = args.output or PROJECT_ROOT / "data" / "real" / "processed" / f"{args.demo.stem}_panda_plan.npz"
    plan.save(output, cfg)
    print(f"[data] saved {output}", flush=True)
    if args.plot:
        plot_plan(plan, SceneAnchors.load(args.anchors), PROJECT_ROOT / "data" / "real" / "debug" / f"{args.demo.stem}_retarget.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
