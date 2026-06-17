#!/usr/bin/env bash
set -euo pipefail

# Dated training series for architecture comparison and longer follow-up.
# Runs sequentially (one after another) and writes logs per run.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

MODEL_DIR="scripts/models"
DATA_DIR="data/Training_data_manually_Drawn_labels/atv_data_label"

mkdir -p "$MODEL_DIR"

run() {
  local name="$1"
  shift
  local log_path="$MODEL_DIR/${name}.log"

  echo "============================================================"
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] START: $name"
  echo "Log: $log_path"
  echo "============================================================"

  # stdbuf helps keep live output flowing in screen/tee.
  stdbuf -oL -eL conda run -n deeplogger python scripts/run_training.py "$@" 2>&1 | tee "$log_path"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] DONE: $name"
  echo
}

# 1) Fair head-to-head under identical settings.
run "compare_aug_focal_tversky_e200_20260617" \
  --data-dir "$DATA_DIR" \
  --model-dir "$MODEL_DIR" \
  --data-type atv \
  --models attention_unet_atv unet_atv_v2 \
  --loss focal_tversky \
  --batch-size 4 \
  --epochs 200 \
  --run-name compare_aug_focal_tversky_e200_20260617

# 2) Optional legacy baseline for 3-way architecture comparison.
run "compare_aug_focal_tversky_v1_e200_20260617" \
  --data-dir "$DATA_DIR" \
  --model-dir "$MODEL_DIR" \
  --data-type atv \
  --models unet_atv_v1 \
  --loss focal_tversky \
  --batch-size 4 \
  --epochs 200 \
  --run-name compare_aug_focal_tversky_v1_e200_20260617

# 3) Longer run on attention model (post-comparison reference).
run "winner_attention_aug_focal_tversky_e300_20260617" \
  --data-dir "$DATA_DIR" \
  --model-dir "$MODEL_DIR" \
  --data-type atv \
  --models attention_unet_atv \
  --loss focal_tversky \
  --batch-size 4 \
  --epochs 300 \
  --run-name winner_attention_aug_focal_tversky_e300_20260617

echo "All scheduled runs completed."
