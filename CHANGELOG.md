# Changelog

All notable changes to DeepLogger will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/). This log also serves as source material for the DeepLogger scientific manuscript — entries capture not just what changed but why, including methodological decisions.

## [Unreleased]

### Added
- `CLAUDE.md` — development guidance for AI-assisted coding sessions
- `CHANGELOG.md` — change tracking, doubles as manuscript development narrative
- `examples/inference_demo.py` — end-to-end CLI demo: loads prepared sample, runs trained U-Net, outputs 4-panel visualization (image, ground truth, prediction probability, overlay with Dice score) + preprocessing comparison. Supports `--sample-id`, `--threshold`, `--model-path` args
- `app.py` — Streamlit GUI for interactive exploration: dataset/sample selector, preprocessing controls (SVD removal, mean removal, FFT high-pass, Gaussian blur) with live preview, model inference with Dice score, prediction overlay. Run with `streamlit run app.py`
- `streamlit` added as optional `gui` dependency in `pyproject.toml`
- Streamlit GUI: auto-detects model input channels from weights, handles ATV/OTV data format differences, adapts single-channel ATV data for 3-ch OTV models (pseudo-RGB stacking), searches `models/` directory at project root for model weights
- `deeplogger/config.py` — dataclasses for structured configuration:
  - `Borehole`: bundles diameter, data_path, data_type, azimuth_values — eliminates repeated loose parameters
  - `Fracture`: bundles azimuth, dip, depth, aperture with unit conversion (`aperture_m` property) and corrections — prevents argument-order bugs in label generation
  - `TrainingConfig`: captures all hyperparameters (loss, optimizer, epochs, LR, augmentation, etc.) with `to_dict()`/`from_dict()` for serialization — replaces the 6+ near-identical training scripts
  - `DataType`, `LossType`, `OptimizerType` enums for type-safe configuration
- `deeplogger/train.py` — unified training module with a single `train(config)` function that handles model selection (ATV/OTV), loss function, optimizer, data splitting, augmentation, validation, and checkpoint saving. Replaces copy-pasted training scripts in `model/`
- `BoreholeDataset` class in `dataloader.py` — richer Dataset with `from_directory()` classmethod, configurable device/transforms, proper docstrings. Backward-compatible aliases `Dataset` and `Dataset_np` preserved
- `deeplogger/las_reader.py` — modern polars-based LAS 3.0 reader for the new GUI (legacy pandas `importLASv3.py` retained for notebooks/preparation). Provides:
  - `read_las_header(path)` → `LasHeader` dataclass (depth bounds, step, null sentinel, delimiter, data type, azimuth count, data offset, version); infers ATV/OTV and azimuth count from the first data row
  - `read_las_data(path, header, row_slice)` → `(depth, data)`; ATV `(N, n_az)` float32 with NULL→NaN, OTV `(N, n_az, 3)` uint8 (dot-packed RGB split); supports windowed reads for lazy viewing
  - `BoreholeLog` container — per-log façade over the readers: `open()`, `read(row_slice)`, cached `depth_vector`/`n_rows`, `depth_to_row`/`row_to_depth`, header passthroughs, and `to_zarr(cache_dir)` (the "Convert & cache" action)
- `deeplogger/pyramid.py` — multiscale pyramid + zarr cache + navigation:
  - `build_depth_pyramid(image, min_rows, factor)`: downsamples the depth axis only (×2 block mean, NaN-aware for ATV, uint8-preserving for OTV), keeping azimuth full-resolution
  - `write_zarr_pyramid` / `read_zarr_pyramid`: persist/load the pyramid (+ depth vector + metadata) as a plain zarr group; levels read back as lazy arrays
  - `select_pyramid_level` / `level_window`: pure helpers that choose the level and row window for the current scroll/zoom (the viewer's navigation logic)
- `deeplogger/gui/viewer.py` — fast **pyqtgraph** browse viewer (`LogViewer`): real-depth Y axis with azimuth (X) locked, depth-only scroll/zoom that swaps pyramid level/window on the fly so large logs stay responsive. ATV controls bar: colormap dropdown, Auto-contrast, and a `ColorBarItem` (two contrast handles); plus a **Remove-SVD** spinbox that destripes the full-resolution log (in-memory re-pyramid) for live preview. `launch_viewer(path)` opens a `.zarr` cache or converts and opens a `.las`. Run: `python -m deeplogger.gui.viewer <log.las|cache.zarr>`
- `deeplogger/image_processing.py:remove_svd_components(image, n_components)` — economy-SVD removal of the first N singular components to suppress coherent vertical stripes; uses `full_matrices=False` so it is practical on full-resolution logs (unlike the existing `remove_svd`, whose full `U` is impractical for tall logs), NaN-safe, dtype-preserving
- `LasHeader.data_type` uses the existing `DataType` enum (consistency with `config.py`)

### Fixed
- Resolved merge conflict in `deeplogger/plotting.py` — kept `plot_atv_tt` (single travel-time plot) and removed conflicting `plot_comparison` stub that was superseded by `plot_comparison_am`/`plot_comparison_tt`
- Resolved merge conflict in `.gitignore` — cleaned up nested conflict markers from prior stash/merge operations
- Removed duplicate `matplotlib` imports in `plotting.py`
- Fixed broken function call in `importLASv3.py`: `get_data_subset_from_start_depth` was passing concatenated path as single arg to `get_depth_only` (which expects two args) and missing required `depth_range` parameter for `get_index_from_start_depth`
- **Fixed `high_pass_FFT_2D` bug** — the slice `[int(r*keep):int(r*(1-keep))]` was empty for `cutoff_frequency < 0.5`, meaning the filter silently did nothing for all typical cutoff values. The function now correctly zeroes low-frequency bins at both ends of the FFT array (DC + positive low-freq at start, negative low-freq mirrors at end). This means any prior results using `high_pass_FFT_2D` with cutoff < 0.5 had **no FFT filtering applied**
- Corrected the `pyproject.toml` build-backend from the non-existent `setuptools.backends._legacy:_Backend` to `setuptools.build_meta` — the invalid value made `pip install` abort before installing anything, blocking clean environment creation

### Changed
- Migrated from `setup.py` to `pyproject.toml` — modern Python packaging with `[build-system]`, proper metadata, and separated `[project.optional-dependencies]` for dev and JAX extras
- Fixed dependencies: removed invalid `logger` PyPI package (was conflicting with built-in `logging`; project uses custom `deeplogger/utils/logger.py`), removed duplicate `rasterio`/`shapely` entries, added missing `opencv-python`, added minimum version pins for all deps
- Version bumped to `0.1.0` (from `0.0.1`) to mark the start of structured development
- Renamed exploratory scripts in `test/` (`test_model.py` → `exploratory_model.py`, `test_transforms.py` → `exploratory_transforms.py`, `Labelling_issues.py` → `exploratory_labelling.py`) — these were analysis scripts, not tests; they had hardcoded paths, no assertions, and script-level code that broke pytest collection
- Added `polars` to core dependencies (LAS reader) and `napari`, `magicgui`, `zarr` to the `[gui]` extra for the new desktop GUI
- Rebuilt the `deeplogger` conda environment on Python 3.12 (was 3.11) — newest version with full wheel support across the stack (torch, napari, rasterio, jax, polars, zarr); 3.13/3.14 not yet viable for the whole set

### Removed
- Untracked `__pycache__/` directories (6 `.pyc` files across Python 3.8/3.9/3.10)
- Untracked `deeplogger/backup/` (14 deprecated files — old versions of dataloader, model architectures, LAS readers, etc.)
- Untracked `scrap/radon_transform.py` (experimental file)
- Untracked `deeplogger.log` and `main.log`
- Updated `.gitignore` to prevent re-tracking of `backup/`, `scrap/`, `*.log`
- Deleted `setup.py` (replaced by `pyproject.toml`)
- Deleted `deeplogger/labels_filtering_plotting.py` — 100% duplicate of functions already in `labels.py` and `filters.py`, never imported anywhere
- Stripped `deeplogger/common_helpers.py` from 371 lines to ~60 — removed: duplicated filter functions (canonical in `filters.py`), duplicated `crop_depth` (canonical in `labels.py`), duplicated `find_nearest` (canonical in `importLASv3.py`), broken old label pipeline with undefined variables (`color_map`, `image_columns`), hardcoded label file paths. Kept: `create_directory`, `save_obj`, `check_if_file_exists`, `check_tensors_for_nans`
- Removed unused imports: `jax.numpy`, `jax.vmap`, `torch.autograd.Variable` from `loss_functions.py`; duplicate `import torch`; unused `matplotlib.pyplot` from `image_processing.py`

### Testing
- Added `pyproject.toml` pytest config: `testpaths = ["test"]`, verbose output, short tracebacks
- Added `test/conftest.py` with `matplotlib.use("Agg")` (non-interactive backend) and shared fixtures
- Rewrote `test_filters.py` with 6 proper pytest tests covering `neighbor_filter`, `gaussian_kernel`, and `gaussian_blur` — original had a wrong assertion (`kernel_size * sum` doesn't hold for dilation on diagonal matrices due to boundary effects)
- Added `test_config.py` — 6 tests for dataclass construction, defaults, serialization round-trip
- Added `test_labels.py` — 7 tests for label rasterization (binary output, correct shape, non-zero pixels, out-of-range handling, depth clipping, DataFrame filtering)
- Added `test_image_processing.py` — 28 tests using synthetic borehole-like data (horizontal banding + sinusoidal fracture trace). Covers `replace_empty_measurements`, `remove_svd`, `remove_mean`, `high_pass_FFT_2D`, `high_pass_2D_kernel`, `radon_transform`. Visual tests output before/after comparison images to `test/output/` for manual inspection
- Added `test_loss_functions.py` — 13 tests for `DiceLoss` (boundary values, symmetry, gradient flow, smoothing), `smoothf1_loss` (perfect/no overlap, negativity, gradients), and `reduce_loss` (mean, sum, invalid)
- Added `test_dataloader.py` — 8 tests for `BoreholeDataset` using temporary `.pt` files. Covers `from_directory()`, dtype, transforms, empty dirs, non-.pt filtering, backward-compatible aliases
- Total: 68 tests, all passing

### Design Decisions
- **GUI viewer split (revised 2026-05-29): fast pyqtgraph browse viewer + napari for labeling only.** The fast scroll/zoom/colormap browsing of the whole log is handled by a **pyqtgraph** viewer (GPU-fast, built-in LUT/contrast, ROI selection), *not* napari — to keep browsing snappy and avoid loading large arrays into napari. **Napari is invoked only for labeling**, receiving the small windowed array the user selects (via a pyqtgraph ROI). The data layer (LAS reader, `BoreholeLog`, zarr pyramid) feeds both. This supersedes the original "napari-centric" framing below; the multiscale pyramid now feeds the pyqtgraph viewer (we manage level/window selection) rather than napari's automatic multiscale.
- **GUI architecture pivot: Napari-centric desktop app (replaces the Streamlit prototype).** The interpretation tool is being redesigned around a lightweight Napari (Qt) desktop application instead of the browser-based Streamlit `app.py`. Rationale:
  - **Scale.** Raw LAS logs are large (ATV 90 MB–534 MB; OTV up to 2.5 GB of ASCII). They cannot be re-parsed or held fully in RAM for interactive browsing. Napari natively supports lazily-loaded, chunked, multiscale images — the standard solution for data of this size. Streamlit (browser) cannot.
  - **Human-in-the-loop.** The downstream goal is a reinforcement loop where the user corrects model predictions. Napari's Labels layer is purpose-built for painting/editing masks; Streamlit is not.
- **Two routes, one app.** A start menu offers (1) *View / Pick data* and (2) *Inference + Correct*. Route 1 is built first. Model **training stays out of the GUI**; the inference route will later add prediction → user correction → **retraining (fine-tune)**, not full training.
- **Data ingest: polars as reader, zarr as store — complementary, not alternatives.** LAS 3.0 files are comma-delimited (`DLM. COMMA`, `WRAP. NO`), one row per depth step — an ideal fit for `polars` (fast, lazy `scan_csv`). On import the user explicitly chooses:
  - **Convert & cache** (recommended for large logs): polars parses once → written as a chunked, compressed, **multiscale** zarr pyramid in a cache dir (`data/.cache/`). Fast chunked browsing thereafter.
  - **Open directly (lazy)**: polars row-range scans on the LAS, no extra disk, single-resolution, somewhat slower scrolling.
- **Multiscale only on the convert path.** The pyramid (level 0 full-res; each level downsampled ×2 along the depth axis via block-mean to avoid aliasing the fracture sinusoids) is precomputed during conversion. Napari auto-selects the level for the current zoom and streams only visible chunks. The lazy path stays single-resolution by design.
- **Depth from the LAS itself.** Real depth comes from the LAS index column (`STRT`/`STOP`/`STEP`, column 0), surfaced via Napari layer `scale`/`translate` so axes read in meters. The `Bedretto_metadata.csv` join idea is dropped — that file is local-only and does not ship with the data.
- **Dual picking, both feeding the existing label pipeline.** (1) Freehand **mask** painting via the Labels layer; (2) **parametric sinusoid** picking via a `magicgui` widget whose sliders *are* the `Fracture` parameters — the curve is exactly `labels.py:get_label`'s geometry (`depth(az) = depth + cos(az + azimuth)·(1/cos(dip))·(D/2)`). Slider→parameter map: center depth↔`depth`, amplitude↔`dip` (amplitude = `(D/2)/cos(dip)`), phase↔`azimuth`, band thickness↔`aperture`. Sliders are driven in **geological units** (dip/azimuth/depth/aperture) with image-space amplitude/phase shown as derived readouts. "Add pick" both rasterizes the curve into the mask (`apply_label`) and appends a structured `Fracture` record to a pick table — reusing `config.Fracture` and `labels.py` unchanged. The borehole diameter `D` is a per-log setting needed for the dip↔amplitude conversion.

### Planned
- CI/CD pipeline (GitLab CI)
- Train dedicated ATV model
