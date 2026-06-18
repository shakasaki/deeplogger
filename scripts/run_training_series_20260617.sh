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

  # Use a pseudo-terminal so tqdm progress bars render live in screen while
  # still recording output to a log file.
  local cmd=(conda run --no-capture-output -n deeplogger python scripts/run_training.py "$@")
  local cmd_str
  printf -v cmd_str '%q ' "${cmd[@]}"

  if command -v script >/dev/null 2>&1; then
    script -q -f "$log_path" -c "$cmd_str"
  else
    # Fallback for environments without util-linux script(1).
    stdbuf -oL -eL "${cmd[@]}" 2>&1 | tee "$log_path"
  fi

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
