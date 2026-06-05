## 2026-06-04 — GUI PyQt6 fixes + literature review (session close)

- [fix-pending-verify] **GUI stuck / "event loop already running"** — two causes fixed 2026-06-05: (1) **nested event loop** still live despite `367052c` — launcher's modal `QDialog.exec()` + viewer's `pg.exec()` nest; PyQt6/pyqtgraph returns from the 2nd `exec()` immediately so the viewer never goes interactive. Fixed by a single loop: non-modal launcher, dispatch on `accepted`, `open_viewer()` (no loop) inside one `pg.exec()`. (2) re-conversion of the whole LAS every open (`to_zarr` `mode="w"`) — fixed by reusing `<dir>/.cache/<stem>.zarr` + wait cursor. **Verify on remote box:** launcher → pick file → viewer should open interactive, no warning. If a *first* convert still hangs, isolate inside `read()`/`build_depth_pyramid` via `python -m deeplogger.gui.viewer /path/to/log.las`.
- [x] **Implement Tversky / Focal-Tversky** in `deeplogger/loss_functions.py` (α=0.3, β=0.7, γ=4/3). Done 2026-06-05: `FocalTverskyLoss` (γ=1 → Tversky, γ=4/3 → Focal-Tversky), `LossType.TVERSKY`/`FOCAL_TVERSKY`, wired in `train.py`, 14 tests. See `.wiki/methods.md §2.4`.
- [ ] **Exercise napari labeler + inferencer under PyQt6** — only `launcher.py`/`bundle_inspector.py` were fixed/scanned; `viewer.py`, `labeler.py`, `inferencer.py` run downstream code not yet exercised under PyQt6 (may hit more enum/`exec_` renames).

## 2026-06-04 — Wiki migrated to .wiki/ research-wiki submodule

- [x] **Wire `.wiki/` private remote.** Done 2026-06-04: `.wiki/` registered as submodule of `git@github.com:shakasaki/deeplogger-wiki.git` (full migrated wiki pulled from remote). `.wiki` removed from `.gitignore`; `.gitmodules` added.
- [x] Commit parent gitlink. Done: `.gitignore`/`.gitmodules` committed (`1591933`, `ab59583`); `.wiki` gitlink bumped to wiki commit `f5660c9` in the session-close commit.

## 2026-06-02

- [issue] Sine/sinusoid picker not functioning properly — drag gesture and/or curve preview unreliable on real data; needs manual testing on actual log to diagnose (interaction with napari event loop? coordinate mapping issue?).
- [issue] Labeler + inferencer napari UI layout not user-friendly — dock panel arrangement, widget sizing, and workflow order need UX pass.
- [issue] Inference predictions poor quality — likely a training data / model limitation rather than code bug; all available models are OTV 3-ch, trained on limited Bedretto dataset. Need better-trained models before the inferencer is useful for science.
- [ ] Install DeepLogger on cluster and run proper model training — better GPU, more epochs, possibly more data. See `environment.yml` for deps; `deeplogger/train.py:train(TrainingConfig)` is the entry point.

## 2026-05-29 — GUI redesign: Napari desktop app (design finalized, see CHANGELOG "Design Decisions")

Data layer (DONE — `deeplogger/las_reader.py`, `deeplogger/pyramid.py`, 117 tests):
- [x] GUI deps in `[gui]` extra: `pyqtgraph`, `napari`, `magicgui`, `zarr` (+ `polars` core)
- [x] LAS reader via polars: `read_las_header` (STRT/STOP/STEP/null/delim, ATV/OTV + azimuth from first data row) and `read_las_data` (ATV float32 NULL→NaN, OTV uint8 RGB, windowed reads)
- [x] `BoreholeLog` container: open/read/depth_vector/n_rows/depth↔row mapping; `to_zarr` convert action
- [x] Multiscale pyramid: `build_depth_pyramid` (×2 depth-axis block mean) + `write_zarr_pyramid`/`read_zarr_pyramid` (lazy zarr levels) into cache dir

Route 1 — View / Pick data (GUI, in progress):
- [x] **Fast browse viewer = pyqtgraph** (`deeplogger/gui/viewer.py`, `LogViewer`): real-depth Y axis, azimuth (X) locked, depth-only scroll/zoom swapping pyramid level/window, colormap dropdown + auto-contrast (ColorBarItem). Interactively confirmed good.
- [x] Processing in viewer — SVD destripe spinbox (`remove_svd_components`, economy SVD); reprocesses full-res + re-pyramids for live preview (~1.6 s on real ATV)
- [x] Save processed data to a new zarr (so processed logs can be reloaded directly) — `LogViewer.save_processed()` + "Save processed…" button (file dialog); records `svd_removed` provenance in the store attrs
- [ ] More processing steps if needed (median, bandpass/"bumper") — reuse in-repo `image_processing.py`/`filters.py`, port from `gdp` only what's missing
- [ ] Lazy-LAS (no-convert) viewer source (currently zarr only)
- [ ] Import flow with explicit user choice: **Convert & cache** (→ to_zarr) vs **Open directly (lazy)** (window-on-demand reads)
- [x] "Label window…" button → hands the **visible depth window** (full-res, cap 20k rows, warn to zoom in) to napari (`deeplogger/gui/labeler.py:launch_labeler`). Uses display bounds, not a separate ROI drag.
- [ ] Start menu: [View / Pick data] | [Inference + Correct]
- [x] Napari labeling (small window only) — mask mode: Labels layer freehand painting
- [x] Napari labeling — sinusoid mode: **interactive** mouse picking ("Pick structure": click=depth, drag x=azimuth/phase, drag y=dip/amplitude), live curve on a Shapes layer; per-log diameter + aperture fields. Replaced the slider interface.
- [x] Save the label as a `[image, mask]` .pt bundle (`<borehole>_<start>m_<end>m.pt`) and save picks to `<borehole>_picks.csv` (Borehole/Depth/Dip/Azimuth, append). Direct-to-OUTPUT_DIR by default; editable in the Output settings panel.
- [ ] Manual-verify the napari flow on a real log (Pick structure drag gesture, paint, save label + picks); headless tests cover only the pure helpers + window cap. Verify gesture sensitivity feels right (azimuth full-width=360°, dip from vertical swing).
- [ ] Add aperture column to the picks CSV? (interactive pick has an aperture field but CSV omits it by design)
- [ ] Reconcile `get_label` vs `apply_label` geometry mismatch (band = ±aperture/2 + corrections vs ±aperture, no corrections); labeler uses `get_label`

Route 2 — Inference + Correct (DONE 2026-06-03):
- [x] Run inference on a log window → probability map + binary overlay in napari (`deeplogger/gui/inferencer.py`)
- [x] Human correction loop: user edits overlay → "Save correction" saves `[image, mask]` bundle
- [x] Fine-tune from corrections: `train.py:finetune()` + "Fine-tune model" button in inferencer
- [issue] All saved models are 3-ch OTV architecture, not the ATV 1-ch model from the thesis. ATV training data (1709 snippets) is at `~/DATA/Bedretto varia/Data_Msc_Thesis_Perritaz/data/Training_data_manually_Drawn_labels/atv_data_label/`. Must retrain before inference is useful on ATV data.

## 2026-06-03

- [ ] Write `docs/DESIGN.md` — technical design document covering: problem & data, preprocessing, model architectures (UNetATV v1/v2, AttentionUNetATV), loss functions (BCE, Dice, BCE+Dice), training strategy (splits, optimizer, scheduler), inference & GUI pipeline, fine-tuning loop. Section 8 = complete reference list (all papers cited in code + Perritaz thesis).
- [ ] Retrain ATV model: `TrainingConfig(model_type=ModelType.UNET_ATV_V2, data_dir="~/DATA/.../atv_data_label/", loss_type=LossType.BCE_DICE, optimizer_type=OptimizerType.ADAM)` on GPU. Also train `AttentionUNetATV` and compare validation Dice + F1. Best model from thesis used Adam + BCE, epoch 75 — now reproducible via `train.py`.
- [ ] Add dataset split by borehole (not random) to `train.py` to prevent spatial-correlation leakage — test set should be whole boreholes (SB/BFE), not random snippets.
- [ ] Start menu item "Inspect Bundles" verified working end-to-end (launcher → bundle_inspector).
- [ ] Manual-verify napari labeler flow on a real log (sinusoid pick gesture, paint, save).

## 2026-04-03
- [ ] Fix GUI to run models successfully on all data types (ATV inference still failing in Streamlit)
- [ ] ~~Fix UI layout — plot all panels in one line~~ (superseded by Napari pivot — Streamlit retired)
- [ ] ~~Allow loading raw LAS data with a depth range selector~~ (folded into Napari Route 1 above)
- [ ] ~~Add depth mode toggle (real depth vs pixel index)~~ (resolved by design: depth read from LAS, real-meter axes)
- [ ] CI/CD pipeline (GitLab CI)
- [ ] Train dedicated ATV model
