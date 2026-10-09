#!/usr/bin/env bash
# Fine-tune SmolVLA (lerobot/smolvla_base) on the exported FLS dataset.
#   ./run_train_smolvla.sh                         # 500-step smoke test -> outputs/train/smolvla_smoke
#   STEPS=20000 NAME=smolvla_fls ./run_train_smolvla.sh
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEPS="${STEPS:-500}"
NAME="${NAME:-smolvla_smoke}"
BATCH="${BATCH:-4}"
cd "$PROJECT_DIR"
exec .venv-lerobot/bin/lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=local/fls_panda_pick_place \
  --dataset.root=data/lerobot/fls_panda_pick_place \
  --rename_map='{"observation.images.front": "observation.images.camera1"}' \
  --batch_size="$BATCH" --steps="$STEPS" --log_freq=25 --save_freq="$STEPS" --num_workers=2 \
  --output_dir="outputs/train/$NAME" --job_name="$NAME" \
  --policy.device=cuda --policy.push_to_hub=false --wandb.enable=false "$@"
