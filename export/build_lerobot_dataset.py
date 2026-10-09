#!/usr/bin/env python3
"""Build a LeRobot (v3.0) dataset for SmolVLA from the recorded accepted episodes.

Input:  data/lerobot_raw/<demo>/{episode.npz, meta.json, frames/*.jpg}  (record_sim_episodes.py)
Output: data/lerobot/<name>/  (LeRobotDataset: parquet data, mp4 video, meta/)

Only episodes whose replay succeeded during recording AND whose source report has
"vla_dataset_ok": true are exported; anything else is refused.

  ./run_build_lerobot_dataset.sh
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW = PROJECT_ROOT / "data" / "lerobot_raw"
OUT = PROJECT_ROOT / "data" / "lerobot"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--repo-id", default="local/fls_panda_pick_place")
    parser.add_argument("--name", default="fls_panda_pick_place", help="Folder under data/lerobot/.")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing dataset folder.")
    return parser.parse_args()


def eligible_episodes() -> list[Path]:
    episodes = []
    for meta_path in sorted(RAW.glob("*/meta.json")):
        meta = json.loads(meta_path.read_text())
        report = json.loads((PROJECT_ROOT / meta["source_report"]).read_text())
        if not report.get("vla_dataset_ok"):
            print(f"[skip] {meta['demo']}: report says vla_dataset_ok=false", flush=True)
            continue
        if not meta.get("replay_success"):
            print(f"[skip] {meta['demo']}: replay failed while recording", flush=True)
            continue
        episodes.append(meta_path.parent)
    return episodes


def main() -> int:
    args = parse_args()
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    episodes = eligible_episodes()
    if not episodes:
        print("[error] no eligible recorded episodes; run ./run_record_episodes.sh first", file=sys.stderr)
        return 1

    metas = [json.loads((ep / "meta.json").read_text()) for ep in episodes]
    fps = {m["fps"] for m in metas}
    sizes = {tuple(m["image_size"]) for m in metas}
    if len(fps) != 1 or len(sizes) != 1:
        print(f"[error] episodes disagree on fps {fps} or image size {sizes}", file=sys.stderr)
        return 1
    height, width = sizes.pop()
    first = metas[0]

    root = OUT / args.name
    if root.exists():
        if not args.overwrite:
            print(f"[error] {root} exists; pass --overwrite to rebuild it", file=sys.stderr)
            return 1
        shutil.rmtree(root)

    features = {
        "observation.images.front": {
            "dtype": "video",
            "shape": (height, width, 3),
            "names": ["height", "width", "channels"],
        },
        "observation.state": {"dtype": "float32", "shape": (len(first["state_names"]),), "names": first["state_names"]},
        "action": {"dtype": "float32", "shape": (len(first["action_names"]),), "names": first["action_names"]},
    }
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=int(fps.pop()),
        features=features,
        root=root,
        robot_type="franka_panda",
        use_videos=True,
    )

    for ep_dir, meta in zip(episodes, metas):
        data = np.load(ep_dir / "episode.npz")
        frames = sorted((ep_dir / "frames").glob("*.jpg"))
        if len(frames) != len(data["state"]):
            print(f"[error] {meta['demo']}: {len(frames)} frames but {len(data['state'])} states", file=sys.stderr)
            return 1
        for i, frame_path in enumerate(frames):
            image = cv2.cvtColor(cv2.imread(str(frame_path)), cv2.COLOR_BGR2RGB)
            dataset.add_frame(
                {
                    "observation.images.front": image,
                    "observation.state": data["state"][i].astype(np.float32),
                    "action": data["action"][i].astype(np.float32),
                    "task": meta["task"],
                }
            )
        dataset.save_episode()
        print(f"[export] {meta['demo']}: {len(frames)} frames", flush=True)
    dataset.finalize()

    # Reload from disk to make sure the dataset is readable the way training will read it.
    check = LeRobotDataset(args.repo_id, root=root)
    sample = check[0]
    print(
        f"[export] {root}: {check.num_episodes} episodes, {check.num_frames} frames @ {check.fps} fps; "
        f"sample image {tuple(sample['observation.images.front'].shape)}, state {tuple(sample['observation.state'].shape)}, "
        f"action {tuple(sample['action'].shape)}, task '{sample['task']}'",
        flush=True,
    )
    (root / "source_demos.json").write_text(
        json.dumps({"episodes": [m["demo"] for m in metas], "recordings": [str(e.relative_to(PROJECT_ROOT)) for e in episodes]}, indent=2)
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
