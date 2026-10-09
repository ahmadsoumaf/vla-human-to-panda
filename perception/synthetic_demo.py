"""Generate a synthetic FLS demonstration video for testing the pipeline without a real recording.

Draws a neutral table, a blue triangle, a white target square with a black border and a
simple hand skeleton that approaches, pinches, carries, places and releases the triangle.
The object/target detectors run on the rendered pixels; MediaPipe cannot detect a drawn hand,
so the hand landmarks (with tracker-like jitter and a few dropped frames) are saved alongside
the video and replayed by the "synthetic" hand backend.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from hand_tracker import NUM_LANDMARKS, HAND_CONNECTIONS

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VIDEO = PROJECT_ROOT / "data" / "real" / "videos" / "synthetic_demo.mp4"


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def hand_landmarks(grasp_point: np.ndarray, direction: float, aperture: float, scale: float) -> np.ndarray:
    """Rough 21-point hand whose thumb and index tips straddle `grasp_point`."""
    forward = np.array([np.cos(direction), np.sin(direction)])
    side = np.array([-forward[1], forward[0]])
    pts = np.zeros((NUM_LANDMARKS, 2))
    wrist = grasp_point - forward * 1.9 * scale
    pts[0] = wrist
    # Thumb (1-4) and index (5-8) converge on the grasp point from either side.
    thumb_tip = grasp_point - side * aperture / 2
    index_tip = grasp_point + side * aperture / 2
    pts[1] = wrist + forward * 0.35 * scale - side * 0.35 * scale
    pts[2] = wrist + forward * 0.75 * scale - side * 0.55 * scale
    pts[3] = 0.5 * (pts[2] + thumb_tip) - side * 0.1 * scale
    pts[4] = thumb_tip
    pts[5] = wrist + forward * 1.0 * scale + side * 0.25 * scale
    pts[6] = pts[5] + (index_tip - pts[5]) * 0.4 + side * 0.15 * scale
    pts[7] = pts[5] + (index_tip - pts[5]) * 0.75 + side * 0.1 * scale
    pts[8] = index_tip
    # Middle, ring and little fingers curled behind the index finger.
    for finger, offset in enumerate([0.0, -0.25, -0.5]):
        mcp = wrist + forward * (1.0 - 0.05 * finger) * scale + side * offset * scale
        base = 9 + 4 * finger
        pts[base] = mcp
        for k in range(1, 4):
            pts[base + k] = mcp + forward * 0.25 * k * scale - side * 0.08 * k * scale
    return pts


def render(args: argparse.Namespace) -> None:
    width, height, fps = args.width, args.height, args.fps
    n_frames = int(round(args.duration * fps))
    rng = np.random.default_rng(args.seed)
    t = np.arange(n_frames) / fps

    # Key positions as fractions of the frame, so any resolution works.
    obj_start = np.array([0.30 * width, 0.62 * height])
    target_center = np.array([0.70 * width, 0.40 * height])
    hand_start = np.array([0.18 * width, 0.92 * height])
    hand_end = np.array([0.88 * width, 0.85 * height])
    scale = 0.07 * min(width, height)
    obj_size = 0.025 * min(width, height)
    target_half = 0.07 * min(width, height)

    # Timeline (seconds): approach, pinch, carry, release, retreat.
    t_grasp, t_closed, t_place, t_open = 2.4, 2.9, 5.2, 5.7

    grasp_point = np.zeros((n_frames, 2))
    aperture = np.zeros(n_frames)
    obj_xy = np.zeros((n_frames, 2))
    # Pinch ratio (thumb-index / wrist-middle MCP) ~1.1 open and ~0.15 when holding the object.
    open_ap, closed_ap = 1.1 * scale, 0.35 * obj_size
    for i, ti in enumerate(t):
        if ti < t_grasp:
            s = smoothstep(np.array(ti / t_grasp))
            arc = np.array([0.0, -0.10 * height]) * np.sin(np.pi * s)
            grasp_point[i] = hand_start + (obj_start - hand_start) * s + arc
            aperture[i] = open_ap
            obj_xy[i] = obj_start
        elif ti < t_closed:
            grasp_point[i] = obj_start
            aperture[i] = open_ap + (closed_ap - open_ap) * smoothstep(np.array((ti - t_grasp) / (t_closed - t_grasp)))
            obj_xy[i] = obj_start
        elif ti < t_place:
            s = smoothstep(np.array((ti - t_closed) / (t_place - t_closed)))
            arc = np.array([0.0, -0.08 * height]) * np.sin(np.pi * s)
            grasp_point[i] = obj_start + (target_center - obj_start) * s + arc
            aperture[i] = closed_ap
            obj_xy[i] = grasp_point[i]
        elif ti < t_open:
            grasp_point[i] = target_center
            aperture[i] = closed_ap + (open_ap - closed_ap) * smoothstep(np.array((ti - t_place) / (t_open - t_place)))
            obj_xy[i] = target_center
        else:
            s = smoothstep(np.array((ti - t_open) / (args.duration - t_open)))
            grasp_point[i] = target_center + (hand_end - target_center) * s
            aperture[i] = open_ap
            obj_xy[i] = target_center

    landmarks = np.zeros((n_frames, NUM_LANDMARKS, 2))
    for i in range(n_frames):
        direction = -np.pi / 2 - 0.35 + 0.2 * np.sin(0.7 * t[i])
        landmarks[i] = hand_landmarks(grasp_point[i], direction, aperture[i], scale)

    # Tracker-like jitter plus a few short dropouts (hand not detected).
    observed = landmarks + rng.normal(0.0, args.jitter_px, landmarks.shape)
    for start in rng.choice(np.arange(5, n_frames - 10), size=args.dropouts, replace=False):
        observed[start : start + rng.integers(2, 6)] = np.nan

    args.output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {args.output}")
    background = np.full((height, width, 3), (178, 182, 180), np.uint8)
    noise = rng.normal(0, 4, (height, width, 1))
    background = np.clip(background + noise, 0, 255).astype(np.uint8)
    tri_local = np.array([[-0.5, -0.55], [0.6, 0.0], [-0.5, 0.55]]) * obj_size

    for i in range(n_frames):
        frame = background.copy()
        tl = (target_center - target_half).astype(int)
        br = (target_center + target_half).astype(int)
        cv2.rectangle(frame, tuple(tl), tuple(br), (20, 20, 20), thickness=-1)
        border = int(0.18 * target_half)
        cv2.rectangle(frame, tuple(tl + border), tuple(br - border), (245, 245, 245), thickness=-1)

        tri = (obj_xy[i] + tri_local).astype(np.int32)
        cv2.fillPoly(frame, [tri], (200, 60, 10))  # strong blue in BGR

        pts = landmarks[i].astype(int)
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), (120, 160, 205), max(2, int(scale * 0.12)), cv2.LINE_AA)
        for p in pts:
            cv2.circle(frame, tuple(p), max(2, int(scale * 0.07)), (100, 140, 190), -1, cv2.LINE_AA)
        writer.write(frame)
    writer.release()

    landmarks_path = args.output.with_name(args.output.stem + "_landmarks.npz")
    np.savez_compressed(landmarks_path, landmarks_xy=observed, true_landmarks_xy=landmarks, object_xy=obj_xy, fps=fps)
    print(f"[synthetic] video: {args.output} ({width}x{height}, {n_frames} frames @ {fps} fps)", flush=True)
    print(f"[synthetic] hand landmarks: {landmarks_path}", flush=True)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--jitter-px", type=float, default=1.5)
    parser.add_argument("--dropouts", type=int, default=4, help="Number of short hand-tracking dropouts.")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    render(parse_args())
