#!/usr/bin/env bash
# Record accepted demo replays (camera + state + action) into data/lerobot_raw/ for the LeRobot export.
#   ./run_record_episodes.sh                 # all demos with vla_dataset_ok=true, headless
#   ./run_record_episodes.sh --demos demo_002
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ISAAC_SIM_DIR="${ISAAC_SIM_DIR:-$HOME/isaac-sim}"
if [[ ! -x "$ISAAC_SIM_DIR/python.sh" ]]; then
    echo "Isaac Sim python.sh not found or not executable: $ISAAC_SIM_DIR/python.sh" >&2
    exit 1
fi
cd "$PROJECT_DIR"
python3 tools/freeze_manifest.py --check
exec "$ISAAC_SIM_DIR/python.sh" "$PROJECT_DIR/export/record_sim_episodes.py" "$@"
