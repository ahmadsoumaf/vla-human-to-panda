#!/usr/bin/env bash
# Create .venv-perception (MediaPipe + OpenCV) and download the MediaPipe hand model.
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$PROJECT_DIR/.venv-perception"
MODEL="$PROJECT_DIR/assets/models/hand_landmarker.task"
MODEL_URL="https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"

if command -v uv >/dev/null 2>&1; then
    [[ -x "$VENV/bin/python" ]] || uv venv "$VENV" --python 3.12
    uv pip install --python "$VENV/bin/python" -r "$PROJECT_DIR/perception/requirements.txt"
else
    [[ -x "$VENV/bin/python" ]] || python3 -m venv "$VENV"
    "$VENV/bin/pip" install -r "$PROJECT_DIR/perception/requirements.txt"
fi

if [[ ! -f "$MODEL" ]]; then
    mkdir -p "$(dirname "$MODEL")"
    curl -sSfL -o "$MODEL" "$MODEL_URL"
fi
"$VENV/bin/python" -c "import cv2, mediapipe; print('perception env ready: mediapipe', mediapipe.__version__, 'opencv', cv2.__version__)"
