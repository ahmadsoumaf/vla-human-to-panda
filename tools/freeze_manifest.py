#!/usr/bin/env python3
"""Freeze manifest for the real-demo pipeline (perception, retargeting, Panda control, config).

  python3 tools/freeze_manifest.py           # write FROZEN_PIPELINE.json
  python3 tools/freeze_manifest.py --check   # exit 1 if any frozen file changed
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "FROZEN_PIPELINE.json"
FROZEN = [
    "perception/process_demo.py",
    "perception/hand_tracker.py",
    "perception/object_tracker.py",
    "perception/target_tracker.py",
    "perception/default_config.json",
    "retargeting/human_to_panda.py",
    "retargeting/replay_human_demo.py",
    "isaac/panda_common.py",
    "isaac/task_geometry.py",
    "isaac/fls_pick_place_scene.anchors.json",
    "assets/models/hand_landmarker.task",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    current = {rel: sha256(PROJECT_ROOT / rel) for rel in FROZEN}
    if args.check:
        frozen = json.loads(MANIFEST.read_text())["files"]
        changed = [rel for rel in FROZEN if frozen.get(rel) != current[rel]]
        if changed:
            print("Frozen pipeline files changed:\n  " + "\n  ".join(changed), file=sys.stderr)
            return 1
        print(f"Frozen pipeline unchanged ({len(FROZEN)} files).")
        return 0
    accepted = sorted(
        json.loads(p.read_text())["demo"]
        for p in (PROJECT_ROOT / "data/real/processed").glob("*_report.json")
        if json.loads(p.read_text())["vla_dataset_ok"]
    )
    MANIFEST.write_text(
        json.dumps(
            {
                "frozen_on": date.today().isoformat(),
                "note": "Perception/retargeting/controller frozen for VLA dataset export; "
                "process new videos with these exact files and thresholds.",
                "accepted_demos": accepted,
                "files": current,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"Wrote {MANIFEST} ({len(FROZEN)} files, {len(accepted)} accepted demos)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
