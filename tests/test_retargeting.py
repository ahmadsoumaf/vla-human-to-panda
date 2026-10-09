"""Regression tests for phase-aware validation in retargeting/human_to_panda.py.

Run with:  python3 -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "isaac"))
sys.path.insert(0, str(PROJECT_ROOT / "retargeting"))

from human_to_panda import RetargetError, RetargetConfig, load_demo, retarget  # noqa: E402
from task_geometry import DEFAULT_ANCHORS, SceneAnchors  # noqa: E402

PROCESSED = PROJECT_ROOT / "data" / "real" / "processed"
DEMO_002 = PROCESSED / "demo_002.npz"
SYNTHETIC = PROCESSED / "synthetic_demo.npz"
SYNTHETIC_TRUTH = PROJECT_ROOT / "data" / "real" / "videos" / "synthetic_demo_landmarks.npz"


@pytest.fixture(scope="module")
def anchors() -> SceneAnchors:
    return SceneAnchors.load(DEFAULT_ANCHORS)


def frames_between(demo: dict, t0: float, t1: float) -> np.ndarray:
    t = demo["timestamps"]
    return (t >= t0) & (t <= t1)


def drop_hand(demo: dict, mask: np.ndarray) -> dict:
    demo = dict(demo)
    for key in ("grasp_point_xy", "hand_xy", "thumb_xy", "index_xy"):
        if key in demo:
            demo[key] = demo[key].copy()
            demo[key][mask] = np.nan
    return demo


def drop_object(demo: dict, mask: np.ndarray) -> dict:
    demo = dict(demo)
    demo["object_visible"] = demo["object_visible"].copy()
    demo["object_visible"][mask] = False
    demo["object_xy"] = demo["object_xy"].copy()
    demo["object_xy"][mask] = np.nan
    return demo


# ----------------------------------------------------------------------------- demo_002 (real)


@pytest.mark.skipif(not DEMO_002.exists(), reason="run ./run_process_demo.sh data/real/videos/demo_002.mp4 --target-mode release")
def test_demo_002_post_release_hand_disappearance_is_accepted(anchors):
    demo = load_demo(DEMO_002)
    plan = retarget(demo, anchors)

    # The real pick-and-place (not the initial set-down at ~0.6-1.0 s) is the one retargeted.
    assert 1.8 < plan.human_grasp_t < 2.3
    assert 3.9 < plan.human_release_t < 4.4
    # The hand really is gone after the release for longer than the allowed gap...
    after = demo["timestamps"] > plan.human_release_t
    hand_after = np.all(np.isfinite(demo["grasp_point_xy"][after]), axis=1)
    assert not hand_after[: int(RetargetConfig().max_hand_gap_s * demo["fps"]) + 1].any()
    # ...and the plan still ends on the demonstrated target.
    np.testing.assert_allclose(plan.place_position[:2], anchors.target_center[:2], atol=1e-9)


@pytest.mark.skipif(not DEMO_002.exists(), reason="processed demo_002 not available")
def test_demo_002_set_down_before_pick_is_not_copied_as_approach(anchors):
    plan = retarget(load_demo(DEMO_002), anchors)
    assert any("approach copied from" in w for w in plan.warnings)


# ----------------------------------------------------------------------------- synthetic phases


@pytest.fixture(scope="module")
def synthetic() -> dict:
    if not SYNTHETIC.exists():
        pytest.skip("run ./run_process_demo.sh --synthetic")
    return load_demo(SYNTHETIC)


@pytest.fixture(scope="module")
def events(synthetic, anchors) -> tuple[float, float]:
    plan = retarget(synthetic, anchors)
    return plan.human_grasp_t, plan.human_release_t


def test_post_release_hand_loss_is_accepted(synthetic, anchors, events):
    _, release = events
    demo = drop_hand(synthetic, synthetic["timestamps"] > release + 0.05)
    plan = retarget(demo, anchors)
    assert plan.human_release_t == pytest.approx(release)


def test_carry_gap_filled_by_object_is_accepted(synthetic, anchors, events):
    grasp, release = events
    mid = 0.5 * (grasp + release)
    gap = frames_between(synthetic, mid - 0.5, mid + 0.5)
    demo = drop_hand(synthetic, gap)
    # Hand lost but the carried object reliably tracked (the drawn hand occludes it in the video,
    # so use the generator's ground truth for those frames).
    truth = np.load(SYNTHETIC_TRUTH)["object_xy"]
    demo["object_visible"] = demo["object_visible"].copy()
    demo["object_visible"][gap] = True
    demo["object_xy"] = demo["object_xy"].copy()
    demo["object_xy"][gap] = truth[: len(gap)][gap]
    plan = retarget(demo, anchors)
    assert any("follow the object" in w for w in plan.warnings)


def test_carry_gap_without_hand_and_object_fails_in_carry(synthetic, anchors, events):
    grasp, release = events
    mid = 0.5 * (grasp + release)
    gap = frames_between(synthetic, mid - 0.5, mid + 0.5)
    demo = drop_object(drop_hand(synthetic, gap), gap)
    with pytest.raises(RetargetError) as err:
        retarget(demo, anchors)
    assert err.value.phase == "carry"


def test_hand_missing_at_grasp_fails_in_grasp(synthetic, anchors, events):
    grasp, _ = events
    demo = drop_hand(synthetic, frames_between(synthetic, grasp - 0.5, grasp + 0.5))
    with pytest.raises(RetargetError) as err:
        retarget(demo, anchors)
    assert err.value.phase == "grasp"


def test_early_approach_gap_trims_the_copied_approach(synthetic, anchors, events):
    grasp, _ = events
    demo = drop_hand(synthetic, frames_between(synthetic, grasp - 1.8, grasp - 0.9))
    plan = retarget(demo, anchors)
    assert any("approach copied from" in w for w in plan.warnings)
