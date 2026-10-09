#!/usr/bin/env python3
"""Process a fixed-camera video of a human FLS pick-and-place into a 2D demonstration.

video -> per-frame hand / blue object / target detections -> gap filling, grasp state,
normalized + relative features -> data/real/processed/<name>.npz + debug overlay video.

Examples:
  ./run_process_demo.sh data/real/videos/demo_001.mp4
  ./run_process_demo.sh --synthetic            # generate and process a synthetic test video
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path

import cv2
import numpy as np

from hand_tracker import (
    DEFAULT_HAND_MODEL,
    HAND_CONNECTIONS,
    NUM_LANDMARKS,
    HandDetection,
    estimate_grasp_state,
    make_hand_tracker,
)
from object_tracker import BlueObjectTracker, ObjectTrackerConfig
from target_tracker import (
    ReleaseTargetConfig,
    TargetDetection,
    TargetTrackerConfig,
    corners_from_roi,
    detect_fixed_target,
    infer_release_target,
    select_target_interactively,
    target_from_corners,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_DIR = PROJECT_ROOT / "data" / "real"
DEFAULT_CONFIG = Path(__file__).resolve().parent / "default_config.json"


# ----------------------------------------------------------------------------- arguments


def parse_int_list(text: str, count: int, name: str) -> list[int]:
    values = [int(round(float(v))) for v in text.replace(" ", "").split(",") if v]
    if len(values) != count:
        raise argparse.ArgumentTypeError(f"{name} needs {count} comma-separated numbers, got {text!r}")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Process a human FLS demo video into a 2D trajectory.")
    parser.add_argument("video", nargs="?", type=Path, help="Input video (e.g. data/real/videos/demo_001.mp4).")
    parser.add_argument("--synthetic", action="store_true", help="Generate a synthetic test video and process it.")
    parser.add_argument("--output", type=Path, help="Output .npz (default: data/real/processed/<video name>.npz).")
    parser.add_argument("--debug-video", type=Path, help="Debug overlay video (default: data/real/debug/<name>_debug.mp4).")
    parser.add_argument("--no-debug-video", action="store_true", help="Skip writing the debug overlay video.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="JSON config with HSV/target/grasp settings.")

    hand = parser.add_argument_group("hand")
    hand.add_argument("--hand-backend", choices=["mediapipe", "synthetic"], help="Default: mediapipe (synthetic with --synthetic).")
    hand.add_argument("--hand-model", type=Path, default=DEFAULT_HAND_MODEL, help="MediaPipe hand_landmarker.task file.")
    hand.add_argument("--synthetic-landmarks", type=Path, help="Landmarks .npz for the synthetic hand backend.")
    hand.add_argument("--grasp-close-ratio", type=float, help="Pinch ratio below which the hand counts as closed.")
    hand.add_argument("--grasp-open-ratio", type=float, help="Pinch ratio above which the hand counts as open.")

    obj = parser.add_argument_group("object (HSV, OpenCV hue 0-179)")
    obj.add_argument("--hsv-lower", help="Lower HSV bound h,s,v (default from config: 95,120,50).")
    obj.add_argument("--hsv-upper", help="Upper HSV bound h,s,v (default from config: 130,255,255).")
    obj.add_argument("--object-roi", help="Only search for the object inside x,y,w,h (pixels).")
    obj.add_argument("--save-mask-frame", type=int, help="Also save the blue mask of this frame to data/real/debug/.")

    target = parser.add_argument_group("target")
    target.add_argument(
        "--target-mode",
        choices=["marker", "release"],
        default="marker",
        help="marker: detect a visible target area; release: the object's stable position after the "
        "human lets go is the demonstrated target (no physical target needed).",
    )
    target.add_argument("--target-roi", help="Fixed target box x,y,w,h in pixels (skips auto detection).")
    target.add_argument("--target-corners", help="Fixed target corners x1,y1,x2,y2,x3,y3,x4,y4 in pixels.")
    target.add_argument("--select-target", action="store_true", help="Drag the target box on the first frame (needs a display).")

    video = parser.add_argument_group("video")
    video.add_argument("--max-width", type=int, default=1920, help="Downscale frames wider than this before processing.")
    video.add_argument("--max-frames", type=int, help="Only process the first N frames.")
    video.add_argument("--no-validate", action="store_true", help="Save even if detection-quality checks fail.")
    args = parser.parse_args(argv)
    if not args.synthetic and args.video is None:
        parser.error("give a video path or --synthetic")
    return args


def load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    return json.loads(path.read_text())


def dataclass_from_dict(cls, values: dict):
    names = {f.name for f in fields(cls)}
    unknown = set(values) - names
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} keys in config: {sorted(unknown)}")
    converted = {k: tuple(v) if isinstance(v, list) else v for k, v in values.items()}
    return cls(**converted)


# ----------------------------------------------------------------------------- video io


class VideoReader:
    def __init__(self, path: Path, max_width: int | None) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            raise RuntimeError(f"OpenCV could not open video: {path}")
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS)) or 30.0
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.max_width = max_width
        self.scale = 1.0

    def read(self) -> np.ndarray | None:
        ok, frame = self.cap.read()
        if not ok:
            return None
        if self.max_width and frame.shape[1] > self.max_width:
            self.scale = self.max_width / frame.shape[1]
            frame = cv2.resize(frame, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        return frame

    def close(self) -> None:
        self.cap.release()


def iter_frames(path: Path, max_width: int | None, max_frames: int | None):
    reader = VideoReader(path, max_width)
    try:
        count = 0
        while max_frames is None or count < max_frames:
            frame = reader.read()
            if frame is None:
                return
            yield frame
            count += 1
    finally:
        reader.close()


# ----------------------------------------------------------------------------- post-processing


def fill_short_gaps(values: np.ndarray, visible: np.ndarray, max_gap: int) -> np.ndarray:
    """Linearly interpolate gaps of at most `max_gap` frames; longer gaps stay NaN."""
    values = np.array(values, dtype=np.float64)
    out = values.copy()
    out[~visible] = np.nan
    idx = np.flatnonzero(visible)
    if len(idx) < 2:
        return out
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            w = (np.arange(1, gap + 1) / (gap + 1))[:, None]
            out[a + 1 : b] = (1 - w) * values[a] + w * values[b]
    return out


def first_release_on_hand_loss(
    grasp_state: np.ndarray,
    hand_visible: np.ndarray,
    object_xy: np.ndarray,
    object_visible: np.ndarray,
    fps: float,
    image_diagonal: float,
    min_absent_s: float,
    max_extent_fraction: float,
    max_step_fraction: float = 0.004,
    search_from: int = 0,
    min_object_fraction: float = 0.8,
) -> int | None:
    """First frame (>= search_from) where a closed grasp ends because the hand vanished.

    People often let go and pull the hand out of view in the same motion, so the tracker never
    sees the fingers reopen.  A closed grasp with no hand for `min_absent_s` while the object is
    visible and at rest is a release; the hand leaving the frame while carrying the object moves
    the object and is not.  "At rest" = median per-frame step below `max_step_fraction` of the
    image diagonal (tolerates slow hand-held camera drift) and total extent below
    `max_extent_fraction` (rejects carries).
    """
    min_absent = max(1, int(round(min_absent_s * fps)))
    frame = search_from
    n = len(grasp_state)
    while frame < n:
        if not (grasp_state[frame] and not hand_visible[frame]):
            frame += 1
            continue
        stretch_end = frame
        while stretch_end < n and grasp_state[stretch_end] and not hand_visible[stretch_end]:
            stretch_end += 1
        if stretch_end - frame >= min_absent:
            window = slice(frame, min(stretch_end, frame + 2 * min_absent))
            seen = object_visible[window] & np.all(np.isfinite(object_xy[window]), axis=1)
            if seen.mean() >= min_object_fraction:
                pts = object_xy[window][seen]
                extent = float(np.max(np.ptp(pts, axis=0)))
                step = float(np.median(np.linalg.norm(np.diff(pts, axis=0), axis=1))) if len(pts) > 1 else 0.0
                if extent <= max_extent_fraction * image_diagonal and step <= max_step_fraction * image_diagonal:
                    return frame
        frame = stretch_end
    return None


def estimate_grasp_with_hand_loss(
    pinch: np.ndarray,
    close_ratio: float,
    open_ratio: float,
    min_hold_frames: int,
    hand_loss: dict | None,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Pinch hysteresis plus releases inferred from hand loss.

    After an inferred release the state machine restarts from "open" on the remaining frames,
    so a later pinch (e.g. the real pick after setting the object down) starts a new grasp.
    """
    state, closure = estimate_grasp_state(pinch, close_ratio, open_ratio, min_hold_frames)
    releases: list[int] = []
    if hand_loss is None:
        return state, closure, releases
    search_from = 0
    while True:
        frame = first_release_on_hand_loss(state, search_from=search_from, **hand_loss)
        if frame is None:
            return state, closure, releases
        releases.append(frame)
        state[frame:], _ = estimate_grasp_state(pinch[frame:], close_ratio, open_ratio, min_hold_frames)
        search_from = frame + 1


def phase_validation(
    events: list[dict],
    timestamps: np.ndarray,
    hand_visible: np.ndarray,
    object_xy: np.ndarray,
    object_visible: np.ndarray,
    validation: dict,
) -> list[str]:
    """Check that each task phase of the main grasp (the longest one) has the data it needs.

    approach/grasp  the object was tracked before the grasp (source anchor) and the hand is
                    visible at the grasp
    carry           hand OR object tracked; the object may be hidden by the fingers
    release         the object is recovered shortly after the release
    (post-release   enough stable object frames for the target: checked by infer_release_target)
    """
    if not events:
        return []
    main = max(events, key=lambda e: e["end"] - e["start"])
    start, end = main["start"], main["end"]
    t = timestamps
    obj_ok = object_visible & np.all(np.isfinite(object_xy), axis=1)
    max_gap = validation.get("max_carry_gap_s", 0.6)
    problems = []

    min_pre = int(validation.get("min_pre_grasp_object_frames", 5))
    pre = int(obj_ok[:start].sum())
    if pre < min_pre:
        problems.append(
            f"[grasp] blue object tracked in only {pre} frames before the grasp at {t[start]:.2f}s "
            f"(need {min_pre} to anchor its start position; check --hsv-lower/--hsv-upper with --save-mask-frame N)"
        )
    near = max_gap / 2
    if not (hand_visible & (np.abs(t - t[start]) <= near)).any():
        problems.append(f"[grasp] hand not visible within {near:.2f}s of the grasp at {t[start]:.2f}s")

    tracked = (hand_visible | obj_ok)[start : end + 1]
    times = np.concatenate([[t[start]], t[start : end + 1][tracked], [t[end]]])
    gap = float(np.max(np.diff(times))) if len(times) > 1 else 0.0
    if gap > max_gap:
        problems.append(f"[carry] neither hand nor object tracked for {gap:.2f}s (limit {max_gap}s)")

    recover_s = validation.get("release_recover_s", 1.0)
    window = (t >= t[end]) & (t <= t[end] + recover_s)
    if not (obj_ok & window).any():
        problems.append(f"[release] object not recovered within {recover_s}s after the release at {t[end]:.2f}s")
    return problems


def find_grasp_events(grasp_state: np.ndarray, timestamps: np.ndarray) -> list[dict]:
    """Closed intervals of the binary grasp state as [{start, end, start_t, end_t}]."""
    events = []
    padded = np.concatenate([[0], grasp_state.astype(np.int8), [0]])
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1) - 1
    for s, e in zip(starts, ends):
        events.append({"start": int(s), "end": int(e), "start_t": float(timestamps[s]), "end_t": float(timestamps[e])})
    return events


# ----------------------------------------------------------------------------- debug overlay


def draw_trail(frame: np.ndarray, points: np.ndarray, color: tuple[int, int, int]) -> None:
    pts = points[np.all(np.isfinite(points), axis=1)].astype(np.int32)
    if len(pts) >= 2:
        cv2.polylines(frame, [pts.reshape(-1, 1, 2)], False, color, 2, cv2.LINE_AA)


def draw_overlay(
    frame: np.ndarray,
    i: int,
    t: float,
    hand: HandDetection,
    obj_xy: np.ndarray,
    obj_contour: np.ndarray | None,
    target: TargetDetection,
    grasp_closed: bool,
    pinch_ratio: float,
    hand_trail: np.ndarray,
    obj_trail: np.ndarray,
) -> np.ndarray:
    out = frame.copy()
    thickness = max(1, int(round(out.shape[1] / 640)))
    if target.corners is not None:
        cv2.polylines(out, [target.corners.astype(np.int32).reshape(-1, 1, 2)], True, (0, 200, 255), 2 * thickness)
    if np.all(np.isfinite(target.xy)):
        center = tuple(target.xy.astype(int))
        cv2.drawMarker(out, center, (0, 200, 255), cv2.MARKER_TILTED_CROSS, 18 * thickness, 2 * thickness)
        if target.corners is None:
            cv2.circle(out, center, 16 * thickness, (0, 200, 255), 2 * thickness, cv2.LINE_AA)
            label_pos = (center[0] - 95 * thickness, center[1] - 24 * thickness)
            cv2.putText(out, "DEMONSTRATED TARGET", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * thickness, (0, 0, 0), 4 * thickness, cv2.LINE_AA)
            cv2.putText(out, "DEMONSTRATED TARGET", label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * thickness, (0, 200, 255), thickness, cv2.LINE_AA)

    draw_trail(out, obj_trail, (255, 120, 0))
    draw_trail(out, hand_trail, (0, 0, 255))

    if obj_contour is not None:
        cv2.drawContours(out, [obj_contour], -1, (255, 255, 0), thickness)
    if np.all(np.isfinite(obj_xy)):
        cv2.circle(out, tuple(obj_xy.astype(int)), 6 * thickness, (255, 255, 0), -1)

    if hand.visible:
        pts = hand.landmarks_xy.astype(int)
        for a, b in HAND_CONNECTIONS:
            cv2.line(out, tuple(pts[a]), tuple(pts[b]), (0, 255, 0), thickness, cv2.LINE_AA)
        for p in pts:
            cv2.circle(out, tuple(p), 2 * thickness, (0, 180, 0), -1)
        for p, color in [(hand.thumb_xy, (255, 0, 255)), (hand.index_xy, (255, 0, 255)), (hand.wrist_xy, (0, 255, 255))]:
            cv2.circle(out, tuple(p.astype(int)), 4 * thickness, color, -1)
        gp = hand.grasp_point_xy.astype(int)
        cv2.circle(out, tuple(gp), 7 * thickness, (0, 0, 255) if grasp_closed else (255, 255, 255), 2 * thickness)

    label = "CLOSED" if grasp_closed else "OPEN"
    color = (0, 0, 255) if grasp_closed else (0, 160, 0)
    lines = [
        (f"frame {i}  t={t:6.2f}s", (255, 255, 255)),
        (f"grasp: {label}  pinch={pinch_ratio:.2f}" if np.isfinite(pinch_ratio) else f"grasp: {label}  (no hand)", color),
        ("hand: visible" if hand.visible else "hand: MISSING", (255, 255, 255) if hand.visible else (0, 0, 255)),
        ("object: visible" if obj_contour is not None else "object: MISSING", (255, 255, 255) if obj_contour is not None else (0, 0, 255)),
    ]
    scale = 0.6 * thickness
    for k, (text, c) in enumerate(lines):
        y = int((28 + 28 * k) * thickness)
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4 * thickness, cv2.LINE_AA)
        cv2.putText(out, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, scale, c, thickness, cv2.LINE_AA)
    return out


# ----------------------------------------------------------------------------- main pipeline


def resolve_target(args: argparse.Namespace, first_frames: list[np.ndarray], cfg: TargetTrackerConfig, scale: float) -> TargetDetection:
    if args.target_corners:
        corners = np.array(parse_int_list(args.target_corners, 8, "--target-corners"), dtype=np.float64).reshape(4, 2)
        return target_from_corners(corners * scale, "corners")
    if args.target_roi:
        x, y, w, h = parse_int_list(args.target_roi, 4, "--target-roi")
        return target_from_corners(corners_from_roi(x, y, w, h) * scale, "roi")
    if args.select_target:
        return select_target_interactively(first_frames[0])
    return detect_fixed_target(first_frames[: cfg.detection_frames], cfg)


def process(args: argparse.Namespace) -> Path:
    config = load_config(args.config)

    if args.synthetic:
        import synthetic_demo

        synth_args = synthetic_demo.parse_args([])
        synthetic_demo.render(synth_args)
        args.video = synth_args.output
        args.synthetic_landmarks = synth_args.output.with_name(synth_args.output.stem + "_landmarks.npz")
        args.hand_backend = args.hand_backend or "synthetic"
    args.hand_backend = args.hand_backend or "mediapipe"

    name = args.video.stem
    output = args.output or REAL_DIR / "processed" / f"{name}.npz"
    debug_path = args.debug_video or REAL_DIR / "debug" / f"{name}_debug.mp4"

    obj_values = dict(config.get("object", {}))
    if args.hsv_lower:
        obj_values["hsv_lower"] = parse_int_list(args.hsv_lower, 3, "--hsv-lower")
    if args.hsv_upper:
        obj_values["hsv_upper"] = parse_int_list(args.hsv_upper, 3, "--hsv-upper")
    if args.object_roi:
        obj_values["roi"] = parse_int_list(args.object_roi, 4, "--object-roi")
    obj_cfg = dataclass_from_dict(ObjectTrackerConfig, obj_values)
    target_cfg = dataclass_from_dict(TargetTrackerConfig, config.get("target", {}))
    grasp_cfg = config.get("grasp", {})
    close_ratio = args.grasp_close_ratio if args.grasp_close_ratio is not None else grasp_cfg.get("close_ratio", 0.35)
    open_ratio = args.grasp_open_ratio if args.grasp_open_ratio is not None else grasp_cfg.get("open_ratio", 0.55)
    validation = config.get("validation", {})

    probe = VideoReader(args.video, args.max_width)
    fps = probe.fps
    print(f"[video] {args.video}  fps={fps:.2f}  frames={probe.frame_count}", flush=True)
    # The camera is fixed, so the target only needs the first few frames.
    first_frames = [f for f in (probe.read() for _ in range(max(1, target_cfg.detection_frames))) if f is not None]
    scale = probe.scale
    probe.close()
    if not first_frames:
        raise RuntimeError(f"No frames could be read from {args.video}")
    height, width = first_frames[0].shape[:2]
    if scale != 1.0:
        print(f"[video] frames downscaled by {scale:.3f} to {width}x{height} (--max-width)", flush=True)

    if args.target_mode == "marker":
        target = resolve_target(args, first_frames, target_cfg, scale)
        print(f"[target] {target.source}: center=({target.xy[0]:.1f}, {target.xy[1]:.1f}) px", flush=True)
    else:
        target = None  # inferred from the release once the grasp events are known
        print("[target] mode=release: the object's resting place after release defines the target", flush=True)
    del first_frames

    tracker = make_hand_tracker(args.hand_backend, args.hand_model, args.synthetic_landmarks)
    object_tracker = BlueObjectTracker(obj_cfg)

    landmark_rows: list[np.ndarray] = []
    pinch_rows: list[float] = []
    object_rows: list[np.ndarray] = []
    area_rows: list[float] = []
    conf_rows: list[float] = []
    contours: list[np.ndarray | None] = []
    hands: list[HandDetection] = []

    # Stream the video (phone videos are too large to hold in memory).
    for i, frame in enumerate(iter_frames(args.video, args.max_width, args.max_frames)):
        hand = tracker.track(frame, i, i / fps)
        hands.append(hand)
        landmark_rows.append(hand.landmarks_xy)
        pinch_rows.append(hand.pinch_ratio)
        det = object_tracker.detect(frame)
        contours.append(det.contour)
        object_rows.append(det.xy)
        area_rows.append(det.area)
        conf_rows.append(det.confidence)
        if args.save_mask_frame is not None and i == args.save_mask_frame:
            mask_path = REAL_DIR / "debug" / f"{name}_mask_{i:05d}.png"
            mask_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(mask_path), object_tracker.mask(frame))
            print(f"[debug] blue mask of frame {i}: {mask_path}", flush=True)
        if (i + 1) % 100 == 0:
            print(f"[track] {i + 1} frames", flush=True)
    tracker.close()

    n = len(hands)
    timestamps = np.arange(n, dtype=np.float64) / fps
    landmarks = np.array(landmark_rows, dtype=np.float64)
    hand_visible = np.array([h.visible for h in hands], dtype=bool)
    pinch_ratio = np.array(pinch_rows, dtype=np.float64)
    object_raw = np.array(object_rows, dtype=np.float64)
    object_area = np.array(area_rows, dtype=np.float64)
    object_conf = np.array(conf_rows, dtype=np.float64)
    object_visible = np.array([c is not None for c in contours], dtype=bool)

    max_gap = max(1, int(round(validation.get("max_fill_gap_s", 0.3) * fps)))
    wrist = fill_short_gaps(landmarks[:, 0], hand_visible, max_gap)
    thumb = fill_short_gaps(landmarks[:, 4], hand_visible, max_gap)
    index = fill_short_gaps(landmarks[:, 8], hand_visible, max_gap)
    grasp_point = 0.5 * (thumb + index)
    object_xy = fill_short_gaps(object_raw, object_visible, max_gap)
    pinch_filled = fill_short_gaps(pinch_ratio[:, None], hand_visible, max_gap)[:, 0]
    hand_loss = None
    if grasp_cfg.get("release_on_hand_loss", True):
        hand_loss = dict(
            hand_visible=hand_visible,
            object_xy=object_raw,
            object_visible=object_visible,
            fps=fps,
            image_diagonal=float(np.hypot(width, height)),
            min_absent_s=grasp_cfg.get("hand_loss_min_absent_s", 0.4),
            max_extent_fraction=grasp_cfg.get("hand_loss_max_object_extent_fraction", 0.03),
            max_step_fraction=grasp_cfg.get("hand_loss_max_object_step_fraction", 0.004),
        )
    grasp_state, grasp_closure, hand_loss_releases = estimate_grasp_with_hand_loss(
        pinch_filled, close_ratio, open_ratio, int(grasp_cfg.get("min_hold_frames", 3)), hand_loss
    )
    for frame in hand_loss_releases:
        print(f"[grasp] release inferred at {frame / fps:.2f}s: hand left view and the object came to rest", flush=True)
    size = np.array([width, height], dtype=np.float64)
    events = find_grasp_events(grasp_state, timestamps)

    target_problem = None
    release_frame = -1
    if target is None:
        if events:
            longest = max(events, key=lambda e: e["end"] - e["start"])
            release_frame = longest["end"]
            release_cfg = dataclass_from_dict(ReleaseTargetConfig, config.get("release_target", {}))
            if release_frame >= n - 1:
                target_problem = (
                    f"[release] the grasp that starts at {timestamps[longest['start']]:.2f}s never ends: the fingers "
                    f"never reopen past the open threshold ({open_ratio}) and the hand never leaves view before the "
                    "video ends, so the release (and the demonstrated target) cannot be determined"
                )
            try:
                if target_problem:
                    raise LookupError
                target = infer_release_target(
                    object_raw, object_visible, release_frame, fps, float(np.hypot(width, height)), release_cfg
                )
                print(
                    f"[target] demonstrated target from release at {timestamps[release_frame]:.2f}s: "
                    f"center=({target.xy[0]:.1f}, {target.xy[1]:.1f}) px from {len(target.stable_frames)} stable frames",
                    flush=True,
                )
            except LookupError:
                pass
            except RuntimeError as exc:
                target_problem = f"[post-release] cannot infer the demonstrated target: {exc}"
        else:
            target_problem = "[grasp] cannot infer the demonstrated target without a grasp/release"
        if target is None:
            target = TargetDetection(xy=np.full(2, np.nan), corners=None, source="release")
    target_xy = np.repeat(target.xy[None, :], n, axis=0)

    hand_fraction = float(hand_visible.mean())
    object_fraction = float(object_visible.mean())
    print(
        f"[summary] frames={n} hand_visible={hand_fraction:.0%} object_visible={object_fraction:.0%} "
        f"grasp_events={len(events)}",
        flush=True,
    )
    for k, ev in enumerate(events):
        print(f"  grasp {k}: closed {ev['start_t']:.2f}s -> {ev['end_t']:.2f}s", flush=True)

    # Phase-aware validation: no global visibility fractions (the fingers legitimately hide the
    # object while carrying, and the hand is usually out of view before/after the task).
    problems = phase_validation(events, timestamps, hand_visible, object_raw, object_visible, validation)
    if target_problem:
        problems.append(target_problem)
    if not events:
        problems.append(
            "[grasp] no grasp detected (pinch ratio never below the close threshold); "
            f"min pinch ratio seen={np.nanmin(pinch_ratio) if np.any(hand_visible) else float('nan'):.2f}, "
            f"close threshold={close_ratio}"
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    norm = lambda xy: xy / size  # noqa: E731
    arrays = dict(
        timestamps=timestamps,
        fps=np.float64(fps),
        image_size=size,
        hand_xy=wrist,
        hand_xy_norm=norm(wrist),
        wrist_xy=wrist,
        thumb_xy=thumb,
        index_xy=index,
        grasp_point_xy=grasp_point,
        grasp_point_xy_norm=norm(grasp_point),
        hand_landmarks_xy=landmarks,
        pinch_ratio=pinch_ratio,
        grasp_state=grasp_state,
        grasp_closure=grasp_closure,
        grasp_close_ratio=np.float64(close_ratio),
        grasp_open_ratio=np.float64(open_ratio),
        object_xy=object_xy,
        object_xy_norm=norm(object_xy),
        object_xy_raw=object_raw,
        object_area=object_area,
        object_confidence=object_conf,
        target_xy=target_xy,
        target_xy_norm=norm(target_xy),
        target_corners_xy=target.corners if target.corners is not None else np.full((4, 2), np.nan),
        target_mode=np.array(args.target_mode),
        demonstrated_target_xy=target.xy if target.source == "release" else np.full(2, np.nan),
        target_stable_frames=target.stable_frames if target.stable_frames is not None else np.zeros(0, np.int64),
        release_frame=np.int64(release_frame),
        hand_relative_to_object=norm(grasp_point) - norm(object_xy),
        object_relative_to_target=norm(object_xy) - norm(target_xy),
        hand_visible=hand_visible,
        object_visible=object_visible,
        release_from_hand_loss=np.array(hand_loss_releases, dtype=np.int64),
        grasp_events=np.array([[e["start"], e["end"]] for e in events], dtype=np.int64).reshape(-1, 2),
        source_video=np.array(str(args.video)),
        target_source=np.array(target.source),
        hand_backend=np.array(args.hand_backend),
    )

    if not args.no_debug_video:
        write_debug_video(debug_path, args, hands, contours, target, arrays, fps)

    if problems and not args.no_validate:
        message = "\n  - ".join(problems)
        raise RuntimeError(
            f"Demo failed validation, so {output.name} was NOT saved:\n  - {message}\n"
            "Inspect the debug video, fix the settings, or re-run with --no-validate to save anyway."
        )
    if problems:
        print("[warn] validation problems (saved anyway because of --no-validate):", flush=True)
        for p in problems:
            print(f"  - {p}", flush=True)

    np.savez_compressed(output, **arrays)
    print(f"[data] saved {output}", flush=True)
    return output


def write_debug_video(
    path: Path,
    args: argparse.Namespace,
    hands: list[HandDetection],
    contours: list[np.ndarray | None],
    target: TargetDetection,
    arrays: dict,
    fps: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    width, height = (int(v) for v in arrays["image_size"])
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open debug video writer: {path}")
    for i, frame in enumerate(iter_frames(args.video, args.max_width, len(hands))):
        overlay = draw_overlay(
            frame,
            i,
            float(arrays["timestamps"][i]),
            hands[i],
            arrays["object_xy"][i],
            contours[i],
            target,
            bool(arrays["grasp_state"][i]),
            float(arrays["pinch_ratio"][i]),
            arrays["grasp_point_xy"][: i + 1],
            arrays["object_xy"][: i + 1],
        )
        writer.write(overlay)
    writer.release()
    print(f"[debug] saved {path}", flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        process(args)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
