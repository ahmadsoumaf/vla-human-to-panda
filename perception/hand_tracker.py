"""2D hand tracking: wrist, thumb tip, index tip, pinch grasp point and grasp state.

Backends:
  mediapipe  - MediaPipe Tasks HandLandmarker (needs assets/models/hand_landmarker.task)
  synthetic  - ground-truth landmarks saved by synthetic_demo.py (for testing the pipeline)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HAND_MODEL = PROJECT_ROOT / "assets" / "models" / "hand_landmarker.task"
HAND_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

NUM_LANDMARKS = 21
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9

# MediaPipe hand skeleton, used for the debug overlay.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]


@dataclass
class HandDetection:
    visible: bool
    landmarks_xy: np.ndarray  # (21, 2) pixels, NaN when not visible
    score: float = 0.0

    @classmethod
    def missing(cls) -> "HandDetection":
        return cls(visible=False, landmarks_xy=np.full((NUM_LANDMARKS, 2), np.nan))

    @property
    def wrist_xy(self) -> np.ndarray:
        return self.landmarks_xy[WRIST]

    @property
    def thumb_xy(self) -> np.ndarray:
        return self.landmarks_xy[THUMB_TIP]

    @property
    def index_xy(self) -> np.ndarray:
        return self.landmarks_xy[INDEX_TIP]

    @property
    def grasp_point_xy(self) -> np.ndarray:
        """Human grasp point: midpoint between the thumb tip and the index fingertip."""
        return 0.5 * (self.thumb_xy + self.index_xy)

    @property
    def hand_scale(self) -> float:
        """Wrist -> middle-finger MCP length in pixels; makes the pinch measure scale-invariant."""
        return float(np.linalg.norm(self.landmarks_xy[MIDDLE_MCP] - self.wrist_xy))

    @property
    def pinch_ratio(self) -> float:
        """Thumb-index distance divided by hand scale (NaN when the hand is not visible)."""
        scale = self.hand_scale
        if not self.visible or not np.isfinite(scale) or scale < 1e-6:
            return float("nan")
        return float(np.linalg.norm(self.thumb_xy - self.index_xy) / scale)


class MediaPipeHandTracker:
    def __init__(
        self,
        model_path: Path = DEFAULT_HAND_MODEL,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks.python import BaseOptions, vision
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe is not installed in this Python. Run ./setup_perception_env.sh "
                "and use ./run_process_demo.sh (it uses .venv-perception)."
            ) from exc
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Hand landmarker model not found: {model_path}\n"
                f"Download it with ./setup_perception_env.sh or from {HAND_MODEL_URL}"
            )
        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_timestamp_ms = -1

    def track(self, frame_bgr: np.ndarray, frame_index: int, timestamp_s: float) -> HandDetection:
        height, width = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(frame_bgr[:, :, ::-1])
        # VIDEO mode requires strictly increasing integer timestamps.
        timestamp_ms = max(int(round(timestamp_s * 1000.0)), self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp_ms
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        if not result.hand_landmarks:
            return HandDetection.missing()
        landmarks = result.hand_landmarks[0]
        xy = np.array([[lm.x * width, lm.y * height] for lm in landmarks], dtype=np.float64)
        score = float(result.handedness[0][0].score) if result.handedness else 1.0
        return HandDetection(visible=True, landmarks_xy=xy, score=score)

    def close(self) -> None:
        self._landmarker.close()


class SyntheticHandTracker:
    """Replays ground-truth landmarks written by synthetic_demo.py (NaN rows = hand missing)."""

    def __init__(self, landmarks_path: Path) -> None:
        data = np.load(landmarks_path)
        self._landmarks = np.array(data["landmarks_xy"], dtype=np.float64)

    def track(self, frame_bgr: np.ndarray, frame_index: int, timestamp_s: float) -> HandDetection:
        if frame_index >= len(self._landmarks):
            return HandDetection.missing()
        xy = self._landmarks[frame_index]
        if not np.all(np.isfinite(xy)):
            return HandDetection.missing()
        return HandDetection(visible=True, landmarks_xy=xy.copy(), score=1.0)

    def close(self) -> None:
        pass


def make_hand_tracker(backend: str, model_path: Path = DEFAULT_HAND_MODEL, synthetic_landmarks: Path | None = None):
    if backend == "mediapipe":
        return MediaPipeHandTracker(model_path)
    if backend == "synthetic":
        if synthetic_landmarks is None:
            raise ValueError("--hand-backend synthetic needs --synthetic-landmarks")
        return SyntheticHandTracker(synthetic_landmarks)
    raise ValueError(f"Unknown hand backend: {backend}")


def estimate_grasp_state(
    pinch_ratio: np.ndarray,
    close_threshold: float,
    open_threshold: float,
    min_hold_frames: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Binary grasp state (1 = closed) with hysteresis, plus a continuous closure in [0, 1].

    The state switches to closed when the pinch ratio stays below `close_threshold`, and back
    to open when it stays above `open_threshold`, each for `min_hold_frames` frames.  Frames
    without a visible hand keep the previous state.
    """
    if close_threshold >= open_threshold:
        raise ValueError("grasp close threshold must be smaller than the open threshold")
    closure = np.clip((open_threshold - pinch_ratio) / (open_threshold - close_threshold), 0.0, 1.0)
    closure = np.where(np.isfinite(pinch_ratio), closure, np.nan)

    state = np.zeros(len(pinch_ratio), dtype=np.int8)
    current = 0
    streak = 0
    for i, ratio in enumerate(pinch_ratio):
        if np.isfinite(ratio):
            wants_switch = (current == 0 and ratio < close_threshold) or (current == 1 and ratio > open_threshold)
            streak = streak + 1 if wants_switch else 0
            if streak >= min_hold_frames:
                current = 1 - current
                state[i - streak + 1 : i + 1] = current
                streak = 0
        state[i] = current
    return state, closure
