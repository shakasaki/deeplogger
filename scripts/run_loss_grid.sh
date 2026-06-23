#!/usr/bin/env bash
# Loss grid on the UNetATV v2 backbone, under the borehole-stratified CV.
#
# Trains one v2 model per (loss, held-out borehole) on the seeded split, then
# evaluates on the held-out borehole. Compare results to focal_tversky on v2
# (already run: mean F1 53.7), which uses the identical seeded split.
#
# Reproducible + resumable: a (loss, fold) whose report already exists is
# skipped, so re-running picks up where it left off. Reports land in
# scripts/models/grid_<loss>_<fold>_e200_report.md and the registry
# scripts/models/training_runs.csv.
#
# Usage (background, survives logout):
#   nohup bash scripts/run_loss_grid.sh > scripts/models/grid_driver.log 2>&1 &
# Check progress:
#   tail -f scripts/models/grid_driver.log
#   grep -H 'Epoch .*complete' scripts/models/grid_*_e200.log | tail
set -u

PY="${PY:-$HOME/.conda/envs/deeplogger/bin/python}"
DATA="data/Training_data_manually_Drawn_labels/atv_data_label"
MD="scripts/models"
EPOCHS="${EPOCHS:-200}"
BATCH="${BATCH:-4}"

# Most-informative losses first so an early stop still yields complete losses.
# focal_tversky is omitted: already run on v2 on this seeded split (the baseline).
LOSSES=(dice_topk dice_focal rce bce_dice bce tversky)
FOLDS=(ST2 MB8)

cd "$(dirname "$0")/.." || exit 1

for loss in "${LOSSES[@]}"; do
  for fold in "${FOLDS[@]}"; do
    run="grid_${loss}_${fold}_e200"
    if [ -f "$MD/${run}_report.md" ]; then
      echo "=== SKIP $run (report exists) $(date) ==="
      continue
    fi
    echo "=== loss=$loss fold=$fold START $(date) ==="
    "$PY" scripts/run_training.py \
      --data-dir "$DATA" --model-dir "$MD" --data-type atv \
      --models unet_atv_v2 --loss "$loss" --batch-size "$BATCH" --epochs "$EPOCHS" \
      --test-borehole "$fold" --run-name "$run" \
      > "$MD/${run}.log" 2>&1
    echo "=== loss=$loss fold=$fold DONE rc=$? $(date) ==="
  done
done
echo "=== GRID COMPLETE $(date) ==="
