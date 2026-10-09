"""Target region for a fixed camera: a visible marker, or the demonstrated release position.

Target modes (process_demo.py --target-mode):
  marker   a visible target area (default, below)
  release  no physical target: the FLS object's stable position after the human releases it
           becomes the demonstrated target (see infer_release_target)

The default marker is a white square with a black border.  Automatic detection runs on the
first frames and the median result is reused for the whole video (the camera is fixed).
If automatic detection is unreliable, pass the target once:
  --target-roi x,y,w,h            axis-aligned box in pixels
  --target-corners x1,y1,...,x4,y4 four corners in pixels
  --select-target                 drag a box on the first frame (needs a display)
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class TargetTrackerConfig:
    min_area_fraction: float = 0.002
    max_area_fraction: float = 0.25
    max_aspect_ratio: float = 2.0
    # Interior of the marker must be bright and fill a reasonable part of the outer border.
    min_interior_brightness: float = 150.0
    min_hole_fraction: float = 0.25
    max_hole_fraction: float = 0.95
    detection_frames: int = 15
    min_detections: int = 5
    max_center_spread_fraction: float = 0.02


@dataclass
class ReleaseTargetConfig:
    # Skip this long after the release so the fingers have left the object.
    settle_s: float = 0.2
    # Search window after the release (s); None searches to the end of the video.
    search_s: float | None = 3.0
    # A frame is "stable" if the object moved less than this fraction of the image diagonal.
    max_step_fraction: float = 0.004
    min_stable_frames: int = 8
    # Stable detections must agree to within this fraction of the image diagonal.
    max_spread_fraction: float = 0.015


@dataclass
class TargetDetection:
    xy: np.ndarray  # center in pixels
    corners: np.ndarray | None  # (4, 2) pixels; None for a demonstrated (release) target
    source: str  # "auto", "roi", "corners", "selected" or "release"
    stable_frames: np.ndarray | None = None  # frames used for a release target


def corners_from_roi(x: float, y: float, w: float, h: float) -> np.ndarray:
    return np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.float64)


def target_from_corners(corners: np.ndarray, source: str) -> TargetDetection:
    corners = np.asarray(corners, dtype=np.float64).reshape(4, 2)
    return TargetDetection(xy=corners.mean(axis=0), corners=corners, source=source)


def detect_marker(frame_bgr: np.ndarray, cfg: TargetTrackerConfig) -> np.ndarray | None:
    """Find a black-bordered white quadrilateral; returns its 4 outer corners or None."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, dark = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, hierarchy = cv2.findContours(dark, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return None

    image_area = float(gray.shape[0] * gray.shape[1])
    best_corners = None
    best_area = 0.0
    for idx, contour in enumerate(contours):
        parent, child = hierarchy[0][idx][3], hierarchy[0][idx][2]
        if parent != -1 or child == -1:
            continue  # need an outer dark border with a hole (the white interior)
        area = float(cv2.contourArea(contour))
        if not (cfg.min_area_fraction * image_area <= area <= cfg.max_area_fraction * image_area):
            continue
        approx = cv2.approxPolyDP(contour, 0.04 * cv2.arcLength(contour, True), True)
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            continue
        (_, _), (rw, rh), _ = cv2.minAreaRect(approx)
        if min(rw, rh) <= 0 or max(rw, rh) / min(rw, rh) > cfg.max_aspect_ratio:
            continue

        hole = max((contours[c] for c in _children(hierarchy, idx)), key=cv2.contourArea)
        hole_fraction = float(cv2.contourArea(hole)) / area
        if not (cfg.min_hole_fraction <= hole_fraction <= cfg.max_hole_fraction):
            continue
        hole_mask = np.zeros_like(gray)
        cv2.drawContours(hole_mask, [hole], -1, 255, thickness=-1)
        if cv2.mean(gray, mask=hole_mask)[0] < cfg.min_interior_brightness:
            continue
        if area > best_area:
            best_area = area
            best_corners = approx.reshape(4, 2).astype(np.float64)
    return best_corners


def _children(hierarchy: np.ndarray, idx: int):
    child = hierarchy[0][idx][2]
    while child != -1:
        yield child
        child = hierarchy[0][child][0]


def detect_fixed_target(frames: list[np.ndarray], cfg: TargetTrackerConfig) -> TargetDetection:
    """Detect the marker on several early frames and return the median (camera is fixed)."""
    found = [corners for corners in (detect_marker(frame, cfg) for frame in frames) if corners is not None]
    if len(found) < min(cfg.min_detections, len(frames)):
        raise RuntimeError(
            f"Target marker found in only {len(found)}/{len(frames)} early frames. "
            "Use a white square with a black border, or pass --target-roi x,y,w,h, "
            "--target-corners x1,y1,x2,y2,x3,y3,x4,y4, or --select-target."
        )
    centers = np.array([c.mean(axis=0) for c in found])
    center = np.median(centers, axis=0)
    diagonal = float(np.hypot(*frames[0].shape[:2]))
    spread = float(np.max(np.linalg.norm(centers - center, axis=1)))
    if spread > cfg.max_center_spread_fraction * diagonal:
        raise RuntimeError(
            f"Target detections disagree by {spread:.1f} px across early frames (camera moving or a "
            "different square is being picked up). Pass --target-roi or --target-corners instead."
        )
    # Use the detection closest to the median as the representative quadrilateral.
    corners = found[int(np.argmin(np.linalg.norm(centers - center, axis=1)))]
    return TargetDetection(xy=center, corners=corners, source="auto")


def select_target_interactively(frame_bgr: np.ndarray) -> TargetDetection:
    window = "Select target region, then press ENTER (c to cancel)"
    x, y, w, h = cv2.selectROI(window, frame_bgr, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(window)
    if w <= 0 or h <= 0:
        raise RuntimeError("No target region selected.")
    print(f"[target] selected ROI: --target-roi {x},{y},{w},{h}", flush=True)
    return target_from_corners(corners_from_roi(x, y, w, h), "selected")


def infer_release_target(
    object_xy: np.ndarray,
    object_visible: np.ndarray,
    release_index: int,
    fps: float,
    image_diagonal: float,
    cfg: ReleaseTargetConfig,
) -> TargetDetection:
    """Demonstrated target = median of the object's stable detections after the release."""
    start = release_index + int(round(cfg.settle_s * fps))
    stop = len(object_xy) if cfg.search_s is None else min(len(object_xy), start + int(round(cfg.search_s * fps)))
    frames = np.arange(start, stop)
    frames = frames[object_visible[frames] & np.all(np.isfinite(object_xy[frames]), axis=1)]
    if len(frames) < cfg.min_stable_frames:
        raise RuntimeError(
            f"only {len(frames)} object detections in the {stop - start} frames after the release "
            f"(need {cfg.min_stable_frames} stable ones to infer the demonstrated target). Keep the blue piece "
            "in view and the hand away from it for ~0.5 s after letting go, or use --target-mode marker."
        )
    # Stable = barely moved since the previous detection (hand gone, object at rest).
    steps = np.linalg.norm(np.diff(object_xy[frames], axis=0), axis=1)
    stable = frames[1:][steps < cfg.max_step_fraction * image_diagonal]
    if len(stable) < cfg.min_stable_frames:
        raise RuntimeError(
            f"the object is visible after the release but only {len(stable)} frames are stable "
            f"(need {cfg.min_stable_frames}); it may still be moving or the hand is occluding it."
        )
    # Use the position right after the object came to rest: detections within the spread limit
    # of the first stable frame.  Later frames can drift with a hand-held camera.
    stable = stable[np.linalg.norm(object_xy[stable] - object_xy[stable[0]], axis=1) < cfg.max_spread_fraction * image_diagonal]
    if len(stable) < cfg.min_stable_frames:
        raise RuntimeError(
            f"only {len(stable)} detections agree with the first resting position after the release "
            f"(need {cfg.min_stable_frames}); the object did not come to rest."
        )
    center = np.median(object_xy[stable], axis=0)
    return TargetDetection(xy=center, corners=None, source="release", stable_frames=stable)
