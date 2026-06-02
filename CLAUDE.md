# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepLogger is a deep learning tool for borehole televiewer data interpretation. It uses U-Net neural networks to detect and characterize fractures in borehole images from ATV (Acoustic Televiewer) and OTV (Optical Televiewer) data. The project is research-stage, developed at ETH Zürich for the VALTER/Bedretto underground laboratory.

## Setup & Commands

```bash
# Install (editable mode)
python -m pip install -e .

# Download data from Polybox
python deeplogger/download_data.py

# Run all tests
pytest test/

# Run a single test
pytest test/test_filters.py -v

# Launch the Streamlit GUI (interactive preprocessing + inference)
streamlit run app.py

# Run the end-to-end CLI inference demo
python examples/inference_demo.py --sample-id 0 --threshold 0.5
```

Recommended: conda environment with Python 3.11.

## Architecture

**Data pipeline:**
LAS files → `importLASv3.py` (parse) → `labels.py` (rasterize fractures) → `image_processing.py` / `filters.py` (preprocess) → `dataloader.py` (PyTorch Dataset) → U-Net model → trained `.pt` weights

**Two data streams with shared architecture:**
- **ATV:** single-channel amplitude data → `model_architectures_ATV.py` (`UNetOTV`, 1-channel input)
- **OTV:** 3-channel RGB optical data → `model_architectures_OTV.py` (`UNetOTV`, 3-channel input)

Footgun: both modules export a class named `UNetOTV`. Disambiguate by import alias (the GUI imports the ATV variant `as UNetATV`) and by detecting input channels from the saved state dict (`encoder1.enc1conv1.weight.shape[1]`) — see `app.py:detect_model_channels`.

**Key modules in `deeplogger/`:**
- `common_helpers.py` — file I/O, data validation, label reading, tensor inspection
- `config.py` — dataclasses (`Borehole`, `Fracture`, `TrainingConfig`) and enums (`DataType`, `LossType`, `OptimizerType`) for structured configuration with serialization
- `train.py` — unified `train(config)` function that supersedes the per-loss/per-architecture scripts in `model/`
- `labels.py` — fracture label generation (rasterization from depth/azimuth/dip parameters)
- `image_processing.py` — SVD removal, mean removal, FFT high-pass filtering, Radon transforms
- `filters.py` — Gaussian blur, neighbor dilation filter, custom convolution
- `loss_functions.py` — DiceLoss, smooth Gaussian-weighted pooling (PyTorch + JAX variants)
- `dataloader.py` — `BoreholeDataset` PyTorch Dataset (with `from_directory()` classmethod); `Dataset` and `Dataset_np` are kept as backward-compatible aliases
- `download_data.py` — automated data download via pooch from ETH Polybox
- `plotting.py` — ATV/OTV visualization and comparison plots
- `utils/logger.py` — custom logger. There is no `logger` PyPI package; do not add one (the name conflicts with stdlib `logging`)

**Directory path constants** are defined in `deeplogger/__init__.py`: `DATA_DIR`, `OUTPUT_DIR`, `MODEL_DIR`, `TUTORIAL_DIR`, `TEST_DIR`.

**Training scripts** in `model/` are legacy per-loss/per-architecture variants (BCE, Dice, BCE+Dice, ResNet101). New training work should extend `deeplogger/train.py` and pass a `TrainingConfig`, not copy-paste from `model/`.

**Notebooks** in `notebooks/` are used for exploratory analysis and data preparation.

**GUI entry point** is `app.py` (Streamlit). It loads weights from `models/` at the repo root, auto-detects ATV vs OTV from the state dict, and stacks single-channel ATV into pseudo-RGB when feeding a 3-channel model.

## Testing

Tests use pytest (68 tests across `test_config.py`, `test_dataloader.py`, `test_filters.py`, `test_image_processing.py`, `test_labels.py`, `test_loss_functions.py`). `test/conftest.py` forces the non-interactive matplotlib backend. Test data are `.pt` files (PyTorch tensor bundles of `[image, mask]`); tests validate mathematical properties like sum preservation and NaN absence, and `test_image_processing.py` writes before/after images to `test/output/` for manual inspection. Files prefixed `exploratory_*.py` in `test/` are analysis scripts with hardcoded paths — they are intentionally not collected by pytest; don't treat them as tests. There is no CI/CD pipeline.

## Dependencies

Core: torch, numpy, scipy, pandas, scikit-image, scikit-learn, matplotlib, rasterio, shapely, openpyxl, pooch, pexpect. Some modules optionally use JAX (`loss_functions.py`) and OpenCV (`filters.py`). Packaging is via `pyproject.toml`; optional extras are `[dev]` (pytest), `[jax]`, and `[gui]` (streamlit).

## Project state & docs

- `CHANGELOG.md` is the source of truth for what changed and *why*. It doubles as source material for the planned scientific manuscript, so log methodological decisions (not just code changes) there.
- `TODO.md` tracks open work items, dated.
- Repository is hosted on GitLab (`gitlab.com/shakasa/deeplogger`), not GitHub.
