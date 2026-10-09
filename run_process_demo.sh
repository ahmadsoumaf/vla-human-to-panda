#!/usr/bin/env bash
# Process a human demo video into data/real/processed/<name>.npz (+ debug video).
#   ./run_process_demo.sh data/real/videos/demo_001.mp4
#   ./run_process_demo.sh --synthetic
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$PROJECT_DIR/.venv-perception/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "Perception environment missing. Run ./setup_perception_env.sh first." >&2
    exit 1
fi

cd "$PROJECT_DIR"
# Silence TensorFlow Lite / glog start-up chatter from MediaPipe.
export GLOG_minloglevel="${GLOG_minloglevel:-2}"
export TF_CPP_MIN_LOG_LEVEL="${TF_CPP_MIN_LOG_LEVEL:-2}"
exec "$PYTHON" "$PROJECT_DIR/perception/process_demo.py" "$@"
