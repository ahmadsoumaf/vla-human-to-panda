"""Blue FLS object tracking by HSV colour segmentation.

RGB frame -> HSV -> blue mask -> morphology cleanup -> largest valid contour -> centroid.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class ObjectTrackerConfig:
    # OpenCV hue is 0-179; saturated blue sits around 100-130.
    hsv_lower: tuple[int, int, int] = (95, 120, 50)
    hsv_upper: tuple[int, int, int] = (130, 255, 255)
    kernel_size: int = 5
    open_iterations: int = 1
    close_iterations: int = 2
    # Valid contour area as a fraction of the image area (resolution independent).
    min_area_fraction: float = 0.00005
    max_area_fraction: float = 0.05
    min_solidity: float = 0.6
    # Optional search region (x, y, w, h) in pixels; None searches the whole frame.
    roi: tuple[int, int, int, int] | None = None


@dataclass
class ObjectDetection:
    visible: bool
    xy: np.ndarray = field(default_factory=lambda: np.full(2, np.nan))
    area: float = 0.0
    confidence: float = 0.0
    contour: np.ndarray | None = None


class BlueObjectTracker:
    def __init__(self, cfg: ObjectTrackerConfig) -> None:
        self.cfg = cfg
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.kernel_size, cfg.kernel_size))

    def mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(self.cfg.hsv_lower, np.uint8), np.array(self.cfg.hsv_upper, np.uint8))
        if self.cfg.open_iterations > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel, iterations=self.cfg.open_iterations)
        if self.cfg.close_iterations > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=self.cfg.close_iterations)
        if self.cfg.roi is not None:
            x, y, w, h = self.cfg.roi
            roi_mask = np.zeros_like(mask)
            roi_mask[y : y + h, x : x + w] = 255
            mask = cv2.bitwise_and(mask, roi_mask)
        return mask

    def detect(self, frame_bgr: np.ndarray) -> ObjectDetection:
        mask = self.mask(frame_bgr)
        image_area = float(frame_bgr.shape[0] * frame_bgr.shape[1])
        min_area = self.cfg.min_area_fraction * image_area
        max_area = self.cfg.max_area_fraction * image_area

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        best: ObjectDetection | None = None
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < min_area or area > max_area:
                continue
            hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
            solidity = area / hull_area if hull_area > 0 else 0.0
            if solidity < self.cfg.min_solidity:
                continue
            if best is not None and area <= best.area:
                continue
            moments = cv2.moments(contour)
            if moments["m00"] <= 0:
                continue
            centroid = np.array([moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]], dtype=np.float64)
            # Confidence: compact blobs well above the minimum size score close to 1.
            size_score = min(1.0, area / (4.0 * min_area))
            best = ObjectDetection(True, centroid, area, float(solidity * size_score), contour)
        return best if best is not None else ObjectDetection(False)
