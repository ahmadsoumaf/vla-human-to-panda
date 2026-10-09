#!/usr/bin/env bash
# Retarget a processed human demo and replay it with the Panda in Isaac Sim.
#   ./run_human_replay.sh data/real/processed/demo_001.npz [--headless] [--grasp-assist]
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_SIM_DIR="${ISAAC_SIM_DIR:-$HOME/isaac-sim}"
ISAACLAB_DIR="${ISAACLAB_DIR:-$HOME/IsaacLab}"

if [[ ! -x "$ISAAC_SIM_DIR/python.sh" ]]; then
    echo "Isaac Sim python.sh not found or not executable: $ISAAC_SIM_DIR/python.sh" >&2
    exit 1
fi

if [[ -d "$ISAACLAB_DIR/source" ]]; then
    export PYTHONPATH="$ISAACLAB_DIR/source:${PYTHONPATH:-}"
fi

cd "$PROJECT_DIR"
exec "$ISAAC_SIM_DIR/python.sh" "$PROJECT_DIR/retargeting/replay_human_demo.py" "$@"
