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
- [x] Napari labeling — sinusoid mode: magicgui geological-unit sliders (depth/dip/azimuth/aperture) + per-log diameter; "Add pick" rasterizes `labels.py:get_label` into the Labels layer
- [x] Save the label as a `[image, mask]` .pt bundle (`labeler.save_label_bundle`). NOTE: not yet saving a structured `Fracture` pick table alongside — picks list is in-memory only; persist next.
- [ ] Manual-verify the napari flow on a real log (button + paint + sinusoid + save); headless tests cover only the pure helpers + window cap
- [ ] Reconcile `get_label` vs `apply_label` geometry mismatch (band = ±aperture/2 + corrections vs ±aperture, no corrections); labeler uses `get_label`

Route 2 — Inference + Correct (later):
- [ ] Run inference on a (converted/lazy) log → prediction mask as editable Labels layer
- [ ] Human reinforcement loop: user corrects predicted picks
- [ ] Retrain (fine-tune) from corrections — not full training

## 2026-04-03
- [ ] Fix GUI to run models successfully on all data types (ATV inference still failing in Streamlit)
- [ ] ~~Fix UI layout — plot all panels in one line~~ (superseded by Napari pivot — Streamlit retired)
- [ ] ~~Allow loading raw LAS data with a depth range selector~~ (folded into Napari Route 1 above)
- [ ] ~~Add depth mode toggle (real depth vs pixel index)~~ (resolved by design: depth read from LAS, real-meter axes)
- [ ] CI/CD pipeline (GitLab CI)
- [ ] Train dedicated ATV model
