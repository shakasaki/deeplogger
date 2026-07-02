# Changelog

All notable changes to DeepLogger will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/). This log also serves as source material for the DeepLogger scientific manuscript — entries capture not just what changed but why, including methodological decisions.

## [Unreleased]

### Fixed (2026-07-02 — wiki `.gitignore` never actually matched, code graph wired up)

The research wiki dropped its `.wiki` submodule on 2026-06-04 (`7d61672`) in favour of a plain Syncthing-synced `deeplogger.wiki/` folder, but `.gitignore` still had the old submodule-era pattern `.wiki/` — which only matches a directory literally named `.wiki`, not one ending in `.wiki`. The wiki had been showing as untracked ever since. Fixed to `*.wiki/` (+ `.codebase-memory`, added below).
- Indexed the repo with `codebase-memory-mcp` (1339 nodes, 4285 edges, 85 files) and relocated the artifact from `.codebase-memory/` into `deeplogger.wiki/codebase-memory/`, symlinked back from `deeplogger/.codebase-memory`, so the graph index travels via Syncthing rather than git. `.codebase-memory` added to `.gitignore` (the symlink itself needs the pattern without a trailing slash to match).
- Added a package-level Mermaid module diagram to the wiki's `code.md` (Graph index section), generated from `get_architecture()` / `query_graph()` over `File-[:IMPORTS]->*` edges: data pipeline → config → models/losses → `train.py`, the `deeplogger/gui/` package, entry-point scripts, and the legacy `model/` scripts.
- Corrected a stale `code.md` claim: the loss-functions row said `DiceFocalLoss`/`DiceTopKLoss` were "to port," but they (plus `RCELoss`) were already implemented in `e334670` (2026-06-23). Re-verified against `deeplogger/loss_functions.py` and narrowed the still-pending item to boundary/soft-clDice (Phase-2 connectivity ablation).

### Fixed (2026-06-05 — napari labeler usable: aspect, depth ruler, crisp pixels)

Verified live over VNC on this box. The labeler window (`deeplogger/gui/labeler.py`) was unusable for picking — the unrolled borehole rendered as a thin horizontal stripe because the image was added to napari with no `scale`, so its square pixels crushed depth.
- **Aspect** — a window is `~step_m` (0.00417 m) per depth row vs `π·D/n_azimuth` (~0.00087 m) per azimuth column, so depth was squashed ~4.8×, flattening fracture sinusoids into a band. The image/labels/shapes layers now share a depth-axis `scale = (step_m / col_m, 1.0)` (an **order-1** ratio) so sinusoids display true-to-aspect. The pick gesture is in data coords (`world_to_data`) and is unaffected.
- **napari sub-unit-scale footgun** — an earlier attempt scaled in absolute metres `(0.00417, 0.00087)` to get a real-depth cursor readout. This makes napari's "new labels" extent math (`world_extent / scale`) try to allocate a `(245761, 586710)` → **134 GiB** array and crash the app. Reverted to the order-1 ratio; never scale napari layers by sub-unit values.
- **Depth on the image** — napari has no labelled tick axis, so a non-metric scale loses the depth readout. Added a `depth_ticks(lo, hi, target)` helper (nice 1/2/2.5/5×10ⁿ steps) and a "depth ruler" points layer: round-number depth labels in metres down the left edge, independent of display scale. Axis labels set to `("depth", "azimuth")`.
- **Resolution** — the image is full-res (`_levels[0]`); the softness was napari interpolation under the stretch. ATV image now uses `interpolation2d="nearest"` for crisp pixels.
- **No default sinusoid** — the candidate curve is no longer drawn on open (removed the initial `_redraw()`); it appears only on a drag-pick or slider edit. Sinusoid geometry is still WIP.
- Tests: `test/test_labeler.py::TestDepthTicks` (4) plus a headless offscreen check that `launch_labeler` builds all four layers, opens with an empty curve, and yields a sane (`239×359`) new-labels extent. Suite 209 passing.

### Fixed (2026-06-05 — GUI nested event loop, take two: single application loop)

Implements the loss the 2026-06-04 literature review identified as the best-evidenced lever for extreme class imbalance (see `.wiki/methods.md §2.4`), unblocking the revised training grid.
- **`FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=4/3, smooth=1.0)`** in `deeplogger/loss_functions.py`. Tversky generalises Dice by decoupling false-positive (α) and false-negative (β) penalties; β>α emphasises recall of the sparse fracture class. The focal exponent γ>1 down-weights easy pixels, steepening the gradient on the hard tail. One class serves both grid rows: γ=1 → plain Tversky, γ=4/3 → Focal-Tversky. Refs: Salehi et al. 2017 (`salehi2017tversky`); Abraham & Khan 2019 (`abraham2019focal`); focal mechanism after Lin et al. 2017 (`lin2017focal`).
- **Methodological note:** Tversky(α=β=0.5) equals Dice only in the smoothing-free limit. With the shared `smooth=1`, the two place the constant differently in the ratio, so they converge only as foreground pixel count grows (gap <1e-3 on a 2×128×128 batch). A smoothing artefact, not a modelling difference — recorded so the equivalence is not mistaken for a bug.
- `LossType.TVERSKY` / `LossType.FOCAL_TVERSKY` added to `config.py`; wired in `train.py:_build_loss`. 14 tests in `test/test_loss_functions.py` (bounds, gradient flow, recall asymmetry, focusing inequality, Dice limit, `_build_loss` wiring). Full suite 205 passing.

### Fixed (2026-06-05 — GUI nested event loop, take two: single application loop)

The `QCoreApplication::exec: The event loop is already running` warning and the unresponsive viewer **persisted** after the launcher refactor in commit `367052c` and the cache fix below. Root cause: the launcher ran a modal `QDialog.exec()` loop and then `launch_viewer` ran a second `pg.exec()` loop. Even sequenced (dialog accepts, *then* the viewer opens), PyQt6/pyqtgraph treats the second `exec()` as nested and returns from it immediately, so the viewer never becomes interactive — the launcher window vanishes and nothing responds.
- Fix: collapse to **one** event loop. `launcher.launch()` now shows the launcher **non-modally** and dispatches the user's choice on its `accepted` signal, opening the viewer/inspector inside the single `pg.exec()` loop. Split `viewer.launch_viewer` into `open_viewer` (build + show, no loop) and `launch_viewer` (standalone `python -m deeplogger.gui.viewer <path>` entry: mkQApp + open + loop) so both paths share the build logic without nesting.
- Top-level windows are held in a list so they outlive the dispatch slot (else garbage-collected on return).
- Headless smoke test (offscreen Qt, `deeplogger` env): `accept()` emits `accepted` synchronously, the launcher hides, no nested `exec()`. Needs interactive confirmation on the remote box.
- This **supersedes** the "earlier event-loop fixes were correct; re-conversion was the remaining cause" conclusion in the entry below — the re-conversion was *a* stall on large logs, but the nesting was the cause of the warning and the no-response.

### Fixed (2026-06-05 — GUI data load re-converted the whole log every open)

`viewer.launch_viewer` called `BoreholeLog.open(path).to_zarr(...)` unconditionally for `.las` inputs, and `to_zarr` writes with zarr `mode="w"` — so every open re-read, re-pyramided and re-wrote the entire multi-GB log on the main thread, blocking the UI for minutes and presenting as a "stuck" data load. Now reuses an existing `<dir>/.cache/<stem>.zarr` if present (repeat opens are instant); the first, unavoidable conversion is wrapped in a wait cursor so it is no longer silent. Delete the cache to force a rebuild after the source LAS changes. The earlier PyQt6 event-loop fixes were correct; this synchronous re-conversion was the remaining cause. Needs on-machine confirmation (GUI + data live on the remote box).

### Fixed (2026-06-04 — Napari GUI runs under PyQt6)

The `deeplogger-gui` environment uses **PyQt6**, which broke the GUI launcher on three counts (the code was written for PyQt5):
- **Scoped enums** — PyQt6 removed flat enum access. Migrated 8 sites in `deeplogger/gui/launcher.py` and `deeplogger/gui/bundle_inspector.py` to scoped form (`Qt.WindowType.WindowContextHelpButtonHint`, `Qt.AlignmentFlag.AlignCenter`, `Qt.Key.Key_Left/Right`; forward-compatible with modern PyQt5/PySide). Fixes `AttributeError: type object 'Qt' has no attribute 'WindowContextHelpButtonHint'` on launch.
- **`exec_()` → `exec()`** — PyQt6 renamed the `QDialog`/`QApplication` event-loop method (`launcher.launch()`).
- **Nested event loop** — `Launcher` ran a modal `exec()` loop whose Browse slot then called `launch_viewer()`, starting a second `pg.exec()` loop while the first was still on the stack → `QCoreApplication::exec: The event loop is already running`, hanging on data load. Refactored `launcher.py`: slots now only record the user's choice (`_path` / `_inspector`) and `accept()`; `launch()` dispatches the viewer/inspector **after** the modal loop unwinds, running a single event loop. `viewer.launch_viewer` left unchanged (self-contained for the standalone `python -m deeplogger.gui.viewer <path>` entry point).
- **Not yet runtime-confirmed** on the user's machine — data load was still reported stuck after these fixes; user took it local to verify with the standalone-viewer isolation test. See TODO 2026-06-04.

### Changed (2026-06-04 — wiki wired to private remote + literature review folded in)

- `.wiki/` **wired as a proper git submodule** of `git@github.com:shakasaki/deeplogger-wiki.git` (parent commits `1591933`, `ab59583`): `.wiki` removed from `.gitignore`, `.gitmodules` registered. Supersedes the "local-only, remote to be wired" status from the migration entry below.
- **Deep-research literature review** (21 sources; 16/25 claims survived 3-vote adversarial verification; crack-detection + geoscience analogues) folded into the wiki (`.wiki` commit `f5660c9`). Methodological outcomes for the manuscript:
  - **Loss is the evidenced lever, not the attention architecture.** Under extreme imbalance (<1 % foreground), plain Dice has unstable gradients and plain BCE over-weights background; the **Focal-Tversky / Unified Focal family** is recommended. Evidence: 12-loss crack-segmentation benchmark (Nguyen & Thai 2023, `doi:10.1016/j.engstruct.2023.116988`); Unified Focal at 0.2 % foreground DSC 0.634 vs Dice 0.536 vs CE 0.336 (Yeung et al. 2022, `doi:10.1016/j.compmedimag.2021.102026`). Recommended Focal-Tversky α=0.3, β=0.7, γ=4/3.
  - **Attention gates have no published support** for binary thin-structure segmentation (RQ1 unresolved either way) → `AttentionUNetATV` demoted from "best expected" to an ablation in the training grid.
  - SAM/SAM2 foundation models ruled out as the primary segmenter (fail on low-contrast thin structures); domain-specific augmentation (flips, colour jitter, blur, mixup) confirmed effective for the small-data regime.
  - Added 6 Crossref-verified-DOI references to `.wiki/references.bib` (`salehi2017tversky`, `yeung2022unified`, `nguyen2023crack`, `wu2019faultseg`, `yu2024u2net`, `zhang2024sam`); revised `methods.md` (§2.4 new), `experiments.md` grid, `open_problems.md`, `index.md`.

### Changed (2026-06-04 — wiki migrated to research-wiki submodule)

**Knowledge base relocated from `docs/wiki/` to a private `.wiki/` git submodule**, managed by the `/research-wiki` skill. Rationale: compounding research wiki that syncs across machines and stays out of the public project history; structured so pages flow directly into the manuscript.
- Page mapping: `INDEX.md`→`index.md`, `dataset.md`→`data.md`, `methods.md`→`methods.md` (+ decisions log), `results.md`→`experiments.md`, `background.md` + paper outline + figure inventory→`manuscript.md`, "Key Implementation Files"→`code.md` (new), "Open Questions"→`open_problems.md` (new), added `log.md` (dated session narrative).
- `references.bib` carried over verbatim (Pandoc `[@key]` citations preserved). Redundant `references.md` (rendered bibliography) dropped — regenerable from the `.bib` via pandoc.
- `.wiki/` is currently a local-only git repo; private remote still to be wired as a proper submodule (TODO 2026-06-04).
- `docs/wiki/` removed from the parent repo.

### Added (2026-06-03 — design document, training script)

**Technical design document (`docs/DESIGN.md`):**
- New 9-section document covering: Problem & Data (ATV/OTV, Bedretto Lab, dataset format), Preprocessing (LAS ingest, pyramid, SVD destriping, per-sample normalisation), Model Architectures (all four U-Net variants with comparison table), Loss Functions (BCE, Dice, BCE+Dice with formulas), Training Strategy (config table, dataset split, augmentation, optimizer choice, thesis best-result metrics), Inference & GUI Pipeline (full pipeline diagram, padding constraint explanation), Fine-tuning Loop, GPU Reproduction Guide, References (28 entries from codebase + thesis bibliography).
- Section 2.3 explicitly documents that ATV training data was **SVD-filtered with k=3 before snippet creation** (Perritaz, 2024, §4.3.2, PDF p. 14) — inference on new ATV data must apply the same filter for distribution consistency.
- Section 8 (GPU Reproduction Guide): exact conda + CUDA torch install commands, training command for both the recommended BCE+Dice run and the thesis baseline, expected runtime, scp copy-back steps.

**Training script (`scripts/run_training.py`):**
- CLI script wrapping `train.py:train()` with argparse interface (model, loss, optimizer, lr, epochs, batch size, thresholds).
- Supports training one or more architectures in sequence (`--models unet_atv_v2 attention_unet_atv`) on the same data split for direct comparison.
- Post-training evaluation on the held-out test set at configurable thresholds (default 0.5 and 0.75): accuracy, sensitivity, specificity, precision, F1, mean per-image Dice.
- Writes a markdown report to `<model-dir>/<run-name>_report.md` with config table, per-epoch loss table, and test evaluation table.

**Architecture diagram (`docs/architecture_diagram.png`):**
- Generated by `scripts/plot_architectures.py` (added previous session, not yet committed). Side-by-side block diagrams of UNetATV and AttentionUNetATV.

### Added (2026-06-03 — inference pipeline, v2 architectures, GUI improvements)

**Inference pipeline (Route 2):**
- `deeplogger/gui/inferencer.py` — full napari inference UI: model panel (auto-discover `.pt` files, detect ATV/OTV from state dict), inference panel (run U-Net, probability map layer, threshold slider, live binary overlay, Dice vs ground truth), save panel (save raw prediction), correction panel (save user-edited overlay as `[image, mask]` training bundle + fine-tune from corrections in a background thread).
- `deeplogger/train.py:finetune()` — fine-tune an existing model on a directory of correction bundles. Adam + BCE, configurable epochs/lr, progress callback for UI updates, saves `<stem>_finetuned.pt`.
- `run_predict()` now pads **both H and W** to satisfy `H%16==8, W%16==8` (OTV model uses `pool4(stride=2, kernel=3, no-padding)` which requires both spatial dims to satisfy this constraint — only H was padded before, causing tensor cat failures on non-standard widths). Output is cropped back to original shape.
- `_valid_height()` — computes smallest valid spatial dim ≥ H satisfying `%16==8`. Required because OTV's `pool4(k=3,s=2)` + `upconv4(k=3,s=2)` only round-trips cleanly when dim/8 is odd.
- `test/test_inferencer.py` — 19 tests: `_valid_height` constraint, `run_predict` output shape (ATV 1-ch, OTV 3-ch, arbitrary H/W, channel coercion), `compute_dice` boundary cases.

**V2 model architectures (`deeplogger/model_architectures_v2.py`):**
- `UNetATV` — faithful Perritaz (2024) ATV U-Net with per-sample min-max normalisation added to `forward()`. All skip connections use `F.interpolate` (no more hard divisibility constraint on input size). pool4 retained as size-preserving `MaxPool(k=3,s=1,p=1)`.
- `AttentionUNetATV` — extended version: attention gates (Oktay et al., 2018) on all four skip connections, spatial `Dropout2d` on enc3/enc4/bottleneck, pool4 changed to `MaxPool(k=2,s=2)` (was a no-op at stride=1), all decoder steps use `F.interpolate` for size-robustness.
- `AttentionGate` — soft spatial gate: gating signal from decoder modulates encoder skip features via learned α ∈ (0,1). Directly addresses <1% foreground imbalance.
- `_norm_input()` — per-sample min-max normalisation to [0,1], applied inside `forward()` so inference is invariant to ATV amplitude scale differences between boreholes.
- Both models accept any (H, W) without padding constraints. Param counts: UNetATV 7.76M, AttentionUNetATV 7.89M (+130K for gates).

**`ModelType` enum + training wiring:**
- `deeplogger/config.py` — `ModelType` enum: `UNET_ATV_V1` (original), `UNET_ATV_V2` (v2 + norm), `ATTENTION_UNET_ATV`, `UNET_OTV`. Added `model_type: ModelType = ModelType.UNET_ATV_V2` field to `TrainingConfig`; `from_dict` deserialises it; backward-compatible with old pickled configs via `getattr(config, 'model_type', None)`.
- `deeplogger/train.py:_build_model()` — dispatches on `ModelType`; falls back to legacy `data_type` logic when `model_type` is absent (old configs).

**`BCEDiceLoss` (`deeplogger/loss_functions.py`):**
- Weighted BCE + Dice combination. Shape-agnostic (handles both `(B,H,W)` from v2 models and `(B,1,H,W)` from v1). Cites Sudre et al. 2017 (arXiv:1707.03237) and Alexakis & Armenakis 2020 (doi:10.3390/rs12101672). Replaces the inline anonymous class in `_build_loss()`.
- `deeplogger/train.py:_build_loss()` — now uses the proper `BCEDiceLoss` class.

**GUI launcher + bundle inspector:**
- `deeplogger/gui/launcher.py` — Qt start screen with "Browse / Label" and "Inspect Bundles" buttons. Shown when the viewer starts with no file argument.
- `deeplogger/gui/bundle_inspector.py` — pyqtgraph widget that browses a directory of `[image, mask]` `.pt` bundles: side-by-side image/mask display, ← → keyboard navigation, status bar showing filename / index / mask coverage %.
- `deeplogger/gui/viewer.py:__main__` — now shows launcher when called with no args; direct path arg still opens viewer immediately.
- `launch_deeplogger.sh` + `deeplogger.desktop` — double-click desktop launcher: activates `deeplogger` conda env and opens the GUI via the launcher.

**Architecture diagram:**
- `scripts/plot_architectures.py` — matplotlib block diagram of both v2 architectures (U-shape layout, channel counts at each stage, attention gates marked in red, dropout badges in purple, pool4 annotation). Saves to `docs/architecture_diagram.png`.

**Tests:**
- Total test count: 162 → 193 (+31). New: `test_inferencer.py` (19), `TestBCEDiceLoss` (8), `TestModelType` (4).

### Investigation findings (2026-06-03)
- **All models in `models/` are 3-channel OTV models** trained on `Bedretto_Output/` (RGB [0,1] normalized optical data). The thesis best ATV model (Adam, epoch 75, 1-channel) is not in the repo — it was never committed. Training data for ATV (1709 manually-drawn snippets) exists at `~/DATA/Bedretto varia/Data_Msc_Thesis_Perritaz/data/Training_data_manually_Drawn_labels/atv_data_label/`. To reproduce the thesis best model, run `model/train_2D_Unet_BCELoss_ATV.py` (Adam + BCE + 1-ch UNetOTV from `model_architectures_ATV`) on that data, or use the new `TrainingConfig(model_type=ModelType.UNET_ATV_V2, ...)` with `deeplogger/train.py`.
- **`detect_model_channels` is insufficient** for disambiguating ATV vs OTV architecture: can check `upconv4.weight.shape` (kernel 2×2 = ATV, 3×3 = OTV). The 3-ch models in `models/` use OTV architecture (pool4 stride=2) despite being trained on stacked-grayscale ATV images.

### Added
- `deeplogger/gui/inferencer.py` — napari inference viewer, parallel to `labeler.py`. `launch_inferencer(image, depth, *, data_type, n_azimuth, mask, source_name, output_dir)` opens a napari window with three dock panels:
  - **Model panel**: discovers `.pt` state-dict files from `models/`, `data/IDs_and_model/`, `data/models/`; loads with auto-detected ATV/OTV channels (same `encoder1.enc1conv1.weight.shape[1]` heuristic as `app.py`); shows device and channel type in a status label.
  - **Inference panel**: threshold slider (0–1) + "Run inference" button → updates a `hot`-colormap probability layer and a binary overlay layer live. If ground truth mask is provided, shows Dice coefficient.
  - **Save panel**: saves `[image, prediction]` as a `.pt` bundle to the output directory.
  - Launched from `viewer.py` via a new "Infer window…" button (sits next to "Label window…" in a button row).
- `viewer.py:LogViewer._on_infer_clicked` — hands the current depth window to `launch_inferencer`; closes any previous inferencer window first (same pattern as `_on_label_clicked`).
- `inferencer.py:find_models`, `detect_model_channels`, `load_model` — model discovery and loading helpers, extracted from `app.py` logic (no streamlit dependency).
- `inferencer.py:_valid_height(H)` — computes the smallest H' ≥ H satisfying `H' % 16 == 8`, the exact condition for all four UNetOTV skip connections to round-trip without size mismatch. **Key finding:** UNetOTV pool4 uses `kernel_size=3` (not 2), so enc4→pool4→upconv4 only round-trips when enc4 = H/8 is odd, which combined with the standard stride-2 pooling requirement gives H ≡ 8 (mod 16). Any viewer window with other heights failed at the enc3/dec3 or enc1/dec1 concat. Input is padded with `np.pad(mode="edge")` to the next valid height; output is cropped back to the original size.

### Fixed
- `inferencer.py:run_predict` — both `UNetOTV` and `UNetATV` perform channel reordering **inside** `forward()` (OTV: `x.permute(0,3,1,2)`; ATV: `x.unsqueeze(0).permute(1,0,2,3)`), so the correct input format is channel-last `(1,H,W,3)` for OTV and `(1,H,W)` for ATV — no manual transpose. Initial implementation wrongly transposed OTV input to `(1,3,H,W)` before the model's own permute, causing a "got 360 channels, expected 3" error. Fixed to pass `image.unsqueeze(0)` directly; grayscale images fed to an OTV model are stacked to 3 channels with `np.stack`.


- napari labeling bridge from the browse viewer. A "Label window…" button hands the **currently visible depth window** (full resolution, including any SVD destriping) to napari via `deeplogger/gui/labeler.py:launch_labeler`. Windows above `MAX_LABEL_ROWS` (20k full-res rows) are refused with a "zoom in" prompt; selection uses the display bounds, not a separate ROI drag. Two label modes share one Labels layer:
  - **Mask painting** — napari's native freehand brush.
  - **Interactive sinusoid picking** — "Pick structure" arms a mouse mode on the image: the click row sets the centre depth, horizontal drag sets the azimuth (one image width = 360° of phase), vertical drag sets the dip via the curve's amplitude (`amplitude = (D/2)/cos(dip)`, using the per-log diameter). A live curve tracks the gesture on a Shapes layer. "Save pick" rasterizes the band (`labels.py:get_label`, at an editable aperture) into the Labels layer and stages the pick. This replaces the earlier slider-based picker.
  - **Saving** (direct to `OUTPUT_DIR` by default, editable in an Output settings panel): *Save label* → `[image, mask]` `.pt` bundle named `<borehole>_<start>m_<end>m.pt`; *Save picks* → appends staged picks to `<borehole>_picks.csv` (columns Borehole / Depth (m) / Dip (deg) / Azimuth (deg)).
  - Pure helpers (`build_label_mask`, `gesture_to_fracture`, `sinusoid_curve`, `default_label_filename`, `append_picks_csv`, `save_label_bundle`) are unit-tested; the napari interaction shell is manual-verify. Labeler uses `get_label` geometry, which differs from `apply_label` (band thickness, corrections) — to be reconciled. The picks CSV omits aperture by design.
- Save processed logs to a new zarr from the browse viewer: `LogViewer.save_processed(out_path)` + a "Save processed…" button (file dialog) in the ATV controls bar. Writes the currently displayed pyramid — including any live SVD destriping — so a processed log reopens directly without reprocessing. The number of SVD components removed is recorded as `svd_removed` in the store attributes for provenance, and `diameter`/depth metadata is carried through. Supported by a new optional `extra_attrs` argument on `pyramid.write_zarr_pyramid` (provenance merged into the zarr group attrs) and by making the writer materialize lazy zarr levels, so an opened store can be re-saved.
- Reproducible conda environments (no env files existed before): `environment.yml` (`deeplogger` — training/dev, CUDA torch + `[dev,gui]`) and `environment-gui.yml` (`deeplogger-gui` — lightweight GUI/inference, CPU-only torch + `[gui]`, skips ~3–4 GB of CUDA wheels). Both Python 3.12. Matching fully-pinned snapshots `requirements-lock-train.txt` (Linux + CUDA 13) and `requirements-lock-gui.txt` (CPU-only, with the PyTorch CPU index header) for bit-for-bit rebuilds. Setup documented in `README.md` and `pyproject.toml`. Standardized on **Python 3.12** across env files and `CLAUDE.md` (resolves a 3.11/3.12 mismatch — the 3.12 decision was already logged below; the env files now match).
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
- Labeler interactive picking made live: the candidate sinusoid is now drawn from a persistent "pick" Shapes layer that redraws on every parameter or drag change (and on open), so it previews and is editable *before* saving — previously the curve only appeared after "Save pick" (as the rasterized band). Added depth/dip/azimuth sliders that drive the curve; the drag gesture sets those sliders rather than rendering mid-drag.
- Labeler pick tool was dead: the mouse-drag callback was on the image layer, which only fires when that layer is selected, but the later-added shapes layer stole selection. Moved the callback to the viewer level and froze camera pan + selected the image layer while picking.
- "Label window…" opened a new `napari.Viewer` each click without closing the prior one, so windows stacked and only the top was reachable — now closes the previous labeler first.
- Resolved merge conflict in `deeplogger/plotting.py` — kept `plot_atv_tt` (single travel-time plot) and removed conflicting `plot_comparison` stub that was superseded by `plot_comparison_am`/`plot_comparison_tt`
- Resolved merge conflict in `.gitignore` — cleaned up nested conflict markers from prior stash/merge operations
- Removed duplicate `matplotlib` imports in `plotting.py`
- Fixed broken function call in `importLASv3.py`: `get_data_subset_from_start_depth` was passing concatenated path as single arg to `get_depth_only` (which expects two args) and missing required `depth_range` parameter for `get_index_from_start_depth`
- **Fixed `high_pass_FFT_2D` bug** — the slice `[int(r*keep):int(r*(1-keep))]` was empty for `cutoff_frequency < 0.5`, meaning the filter silently did nothing for all typical cutoff values. The function now correctly zeroes low-frequency bins at both ends of the FFT array (DC + positive low-freq at start, negative low-freq mirrors at end). This means any prior results using `high_pass_FFT_2D` with cutoff < 0.5 had **no FFT filtering applied**
- Corrected the `pyproject.toml` build-backend from the non-existent `setuptools.backends._legacy:_Backend` to `setuptools.build_meta` — the invalid value made `pip install` abort before installing anything, blocking clean environment creation

### Changed
- Viewer ATV colormaps reordered to lead with warm/heat maps that best match acoustic amplitude (`afmhot`, `copper`, `hot`, `gist_heat`), with `afmhot` as the default (was `inferno`); the previous perceptual maps remain available below them. Chosen on visual review of real ATV logs.
- Repository moved from GitLab to GitHub (`github.com/shakasaki/deeplogger`); updated clone URL, project metadata, and docs.
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
