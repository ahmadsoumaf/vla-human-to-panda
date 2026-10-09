"""Phase-aware perception validation and demonstration-quality flags.

Run with:  python3 -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "perception"))
sys.path.insert(0, str(PROJECT_ROOT / "isaac"))
sys.path.insert(0, str(PROJECT_ROOT / "retargeting"))

pytest.importorskip("cv2")
from process_demo import phase_validation  # noqa: E402
from human_to_panda import load_demo, retarget  # noqa: E402
from task_geometry import DEFAULT_ANCHORS, SceneAnchors  # noqa: E402

FPS = 30.0
VALIDATION = {"min_pre_grasp_object_frames": 5, "max_carry_gap_s": 0.6, "release_recover_s": 1.0}


def scenario(n: int = 150, grasp: tuple[int, int] = (40, 100)):
    """Object resting, picked at frame 40, carried, released at frame 100 and resting again."""
    t = np.arange(n) / FPS
    hand = np.zeros(n, bool)
    hand[35:101] = True
    obj = np.ones(n, bool)
    obj_xy = np.tile([100.0, 100.0], (n, 1))
    events = [{"start": grasp[0], "end": grasp[1], "start_t": t[grasp[0]], "end_t": t[grasp[1]]}]
    return events, t, hand, obj_xy, obj


def problems(events, t, hand, obj_xy, obj):
    return phase_validation(events, t, hand, obj_xy, obj, VALIDATION)


def test_object_hidden_during_carry_is_fine_when_hand_tracked():
    events, t, hand, obj_xy, obj = scenario()
    obj[42:99] = False  # the fingers hide the block for the whole carry
    assert problems(events, t, hand, obj_xy, obj) == []


def test_carry_without_hand_or_object_fails_in_carry():
    events, t, hand, obj_xy, obj = scenario()
    obj[50:80] = False
    hand[50:80] = False  # 1.0 s with nothing tracked
    assert [p for p in problems(events, t, hand, obj_xy, obj) if p.startswith("[carry]")]


def test_no_object_before_grasp_fails_in_grasp():
    events, t, hand, obj_xy, obj = scenario()
    obj[:40] = False
    assert [p for p in problems(events, t, hand, obj_xy, obj) if p.startswith("[grasp]")]


def test_hand_missing_at_grasp_fails_in_grasp():
    events, t, hand, obj_xy, obj = scenario()
    hand[25:60] = False
    assert [p for p in problems(events, t, hand, obj_xy, obj) if "hand not visible" in p]


def test_object_not_recovered_after_release_fails_in_release():
    events, t, hand, obj_xy, obj = scenario()
    obj[95:] = False
    assert [p for p in problems(events, t, hand, obj_xy, obj) if p.startswith("[release]")]


@pytest.mark.skipif(not (PROJECT_ROOT / "data/real/processed/synthetic_demo.npz").exists(), reason="run ./run_process_demo.sh --synthetic")
def test_missed_pick_is_flagged_as_poor_demonstration():
    """Object not seen during the last second before the detected grasp (like the rejected demo_003)."""
    anchors = SceneAnchors.load(DEFAULT_ANCHORS)
    demo = load_demo(PROJECT_ROOT / "data/real/processed/synthetic_demo.npz")
    grasp_t = retarget(demo, anchors).human_grasp_t
    hidden = (demo["timestamps"] > grasp_t - 1.0) & (demo["timestamps"] <= grasp_t)
    demo = dict(demo)
    demo["object_visible"] = demo["object_visible"].copy()
    demo["object_visible"][hidden] = False
    demo["object_xy"] = demo["object_xy"].copy()
    demo["object_xy"][hidden] = np.nan
    plan = retarget(demo, anchors)
    assert plan.quality["pick_observed"] is False
    assert plan.quality["good_demonstration"] is False


@pytest.mark.skipif(not (PROJECT_ROOT / "data/real/processed/demo_002.npz").exists(), reason="demo_002 not processed")
def test_demo_002_is_a_good_demonstration():
    plan = retarget(load_demo(PROJECT_ROOT / "data/real/processed/demo_002.npz"), SceneAnchors.load(DEFAULT_ANCHORS))
    assert plan.quality["pick_observed"] and plan.quality["good_demonstration"]
