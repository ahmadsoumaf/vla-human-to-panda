"""Task constants shared by the scripted controller, the retargeter and the replay.

Pure Python (no Isaac imports) so perception/retargeting code can use it outside Isaac Sim.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE = PROJECT_ROOT / "isaac" / "fls_pick_place_scene.usd"
DEFAULT_ANCHORS = PROJECT_ROOT / "isaac" / "fls_pick_place_scene.anchors.json"

# Distance from Lula's right_gripper frame (panda_hand + 0.100 m) down to the bottom of the
# finger collision hulls in panda_instanceable.usd (panda_hand + 0.1122 m).
FINGERTIP_BELOW_TCP = 0.0122
# Finger opening (per finger) below which the gripper is considered closed on nothing.
EMPTY_GRIP_THRESHOLD = 0.002
GRIPPER_OPEN = 0.04
GRIPPER_CLOSED = 0.0
# Comfortable top-down reach for the Panda right_gripper frame, measured from the base axis.
PANDA_MIN_REACH = 0.30
PANDA_MAX_REACH = 0.72


def min_tcp_z(table_top_z: float, fingertip_clearance: float = 0.004) -> float:
    """Lowest TCP height that keeps the fingertips `fingertip_clearance` above the table."""
    return table_top_z + FINGERTIP_BELOW_TCP + fingertip_clearance


@dataclass
class SceneAnchors:
    """World-frame anchors of the sim task, used to place a retargeted human demo."""

    object_start: list[float]
    target_center: list[float]
    target_radius: float
    table_top_z: float
    robot_base: list[float]
    table_min: list[float]
    table_max: list[float]
    object_height: float

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self), indent=2) + "\n")

    @classmethod
    def load(cls, path: Path) -> "SceneAnchors":
        if not path.exists():
            raise FileNotFoundError(
                f"Scene anchors not found: {path}. Rebuild the scene with ./run_scene.sh --headless --build-only"
            )
        return cls(**json.loads(path.read_text()))
