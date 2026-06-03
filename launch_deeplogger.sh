#!/usr/bin/env bash
# Launch the DeepLogger GUI in the 'deeplogger' conda environment.
# Double-click this script (or the .desktop file) to open the viewer.
set -e

CONDA_BASE="$HOME/miniconda3"
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate deeplogger

cd "$(dirname "$(realpath "$0")")"
exec python -m deeplogger.gui.viewer "$@"
