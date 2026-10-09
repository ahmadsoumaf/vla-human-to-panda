#!/usr/bin/env bash
# Build data/lerobot/<name>/ (LeRobot v3.0) from the recorded accepted episodes.
#   ./run_build_lerobot_dataset.sh [--overwrite]
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$PROJECT_DIR/.venv-lerobot/bin/python"
if [[ ! -x "$PYTHON" ]]; then
    echo "LeRobot environment missing (.venv-lerobot). Create it with: uv venv .venv-lerobot --python 3.12 && uv pip install --python .venv-lerobot/bin/python 'lerobot[dataset]'" >&2
    exit 1
fi
cd "$PROJECT_DIR"
exec "$PYTHON" "$PROJECT_DIR/export/build_lerobot_dataset.py" "$@"
