"""Fast pyqtgraph browse viewer for borehole logs.

Displays a multiscale zarr pyramid with a real-depth axis, colormap/contrast
controls, and fluid scrolling. Only the depth (Y) axis zooms/pans; azimuth (X)
is pinned to the full 0..n_azimuth span. As the depth window changes, the
matching pyramid level and row window are loaded, so multi-gigabyte logs scroll
smoothly without ever holding the whole log in memory. napari is intentionally
not used here; it is reserved for labeling a small window selected from this
viewer.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

from deeplogger.config import DataType
from deeplogger.image_processing import remove_svd_components
from deeplogger.pyramid import (
    build_depth_pyramid,
    level_window,
    read_zarr_pyramid,
    select_pyramid_level,
)

# Display arrays as (row, col) = (depth, azimuth) with depth on the vertical axis.
pg.setConfigOption("imageAxisOrder", "row-major")

# Colormaps offered in the dropdown (ATV only); filtered to those available.
# Warm/heat maps first — best match for ATV amplitude; afmhot is the default.
_COLORMAP_CHOICES = [
    "afmhot", "plasma", "inferno", "copper", "hot", "gist_heat",
    "viridis", "magma", "cividis", "turbo", "gray",
]
_DEFAULT_COLORMAP = "afmhot"


def _get_colormap(name: str):
    """Fetch a pyqtgraph ColorMap by name, trying the matplotlib source too."""
    try:
        return pg.colormap.get(name)
    except Exception:
        return pg.colormap.get(name, source="matplotlib")


class LogViewer(QtWidgets.QWidget):
    """A depth-scrollable view of a borehole log backed by a zarr pyramid.

    Attributes:
        n_levels: Number of pyramid levels available.
        data_type: ATV (single-channel amplitude) or OTV (RGB).
    """

    def __init__(self, levels, depth, attrs, parent=None):
        """Build the viewer over an opened pyramid.

        Args:
            levels: Pyramid levels, finest first (lazy arrays are fine).
            depth: Full-resolution depth vector in meters.
            attrs: Metadata dict from :func:`read_zarr_pyramid` (uses
                ``data_type`` and ``n_azimuth``).
            parent: Optional Qt parent widget.
        """
        super().__init__(parent)
        self._levels = levels
        self._depth = np.asarray(depth)
        self.n_levels = len(levels)
        self.data_type = attrs["data_type"]
        self._is_atv = self.data_type is DataType.ATV
        self._n_azimuth = int(attrs["n_azimuth"])
        self._diameter = attrs.get("diameter")  # carried through on save
        self._source_path = None  # set by from_zarr; seeds the save dialog dir
        self._svd_removed = 0  # components currently destriped (provenance on save)
        self._start = float(self._depth[0])
        self._stop = float(self._depth[-1])
        self._factor = int(attrs.get("factor", 2))
        self._min_rows = 2048  # pyramid coarsest target (matches the default build)
        self._updating = False  # guards against range-change feedback loops
        self._first = True  # set contrast from the data on the first frame

        main = QtWidgets.QVBoxLayout(self)

        # "Label window" and "Infer window" apply to both ATV and OTV.
        btn_row = QtWidgets.QHBoxLayout()
        label_btn = QtWidgets.QPushButton("Label window…")
        label_btn.setToolTip("Open the visible depth window in napari to draw a label")
        label_btn.clicked.connect(self._on_label_clicked)
        btn_row.addWidget(label_btn)

        infer_btn = QtWidgets.QPushButton("Infer window…")
        infer_btn.setToolTip("Open the visible depth window in napari to run inference")
        infer_btn.clicked.connect(self._on_infer_clicked)
        btn_row.addWidget(infer_btn)
        main.addLayout(btn_row)

        glw = pg.GraphicsLayoutWidget()
        self._plot = glw.addPlot()
        self._plot.setLabel("left", "Depth", units="m")
        self._plot.setLabel("bottom", "Azimuth", units="deg")
        self._plot.invertY(True)  # depth increases downward
        self._img = pg.ImageItem()
        self._plot.addItem(self._img)

        # Azimuth is fixed: only depth scrolls/zooms.
        vb = self._plot.getViewBox()
        vb.setMouseEnabled(x=False, y=True)
        vb.setLimits(xMin=0, xMax=self._n_azimuth, yMin=self._start, yMax=self._stop)

        # Clean colorbar (two contrast handles) + colormap + SVD, for ATV only.
        if self._is_atv:
            self._raw = np.asarray(self._levels[0][:])  # full-res, for processing
            self._cbar = pg.ColorBarItem(colorMap=_get_colormap(_DEFAULT_COLORMAP))
            self._cbar.setImageItem(self._img)
            glw.addItem(self._cbar)
            main.addLayout(self._build_controls())
        else:
            self._cbar = None

        main.addWidget(glw)

        self._plot.setXRange(0, self._n_azimuth, padding=0)
        self._plot.setYRange(self._start, self._stop, padding=0)
        self._plot.sigRangeChanged.connect(self._on_range_changed)
        self._refresh()

    @classmethod
    def from_zarr(cls, store_path, parent=None) -> "LogViewer":
        """Open a viewer directly from a zarr pyramid on disk."""
        levels, depth, attrs = read_zarr_pyramid(store_path)
        viewer = cls(levels, depth, attrs, parent=parent)
        viewer._source_path = Path(store_path)
        return viewer

    # --- controls ----------------------------------------------------------

    def _build_controls(self) -> QtWidgets.QHBoxLayout:
        """Build the colormap dropdown + auto-contrast button (ATV)."""
        bar = QtWidgets.QHBoxLayout()
        bar.addWidget(QtWidgets.QLabel("Colormap:"))
        self._cmap_box = QtWidgets.QComboBox()
        available = []
        for name in _COLORMAP_CHOICES:
            try:
                _get_colormap(name)
                available.append(name)
            except Exception:
                pass
        self._cmap_box.addItems(available)
        if _DEFAULT_COLORMAP in available:
            self._cmap_box.setCurrentText(_DEFAULT_COLORMAP)
        # Connect after populating so the initial fill doesn't fire the handler.
        self._cmap_box.currentTextChanged.connect(self._set_colormap)
        bar.addWidget(self._cmap_box)

        auto = QtWidgets.QPushButton("Auto contrast")
        auto.clicked.connect(self._auto_contrast)
        bar.addWidget(auto)

        bar.addSpacing(16)
        bar.addWidget(QtWidgets.QLabel("Remove SVD:"))
        self._svd_box = QtWidgets.QSpinBox()
        self._svd_box.setRange(0, min(20, min(self._raw.shape)))
        self._svd_box.setValue(0)
        self._svd_box.setToolTip(
            "Number of dominant SVD components to remove (suppresses vertical stripes)"
        )
        self._svd_box.valueChanged.connect(self._apply_svd)
        bar.addWidget(self._svd_box)

        bar.addSpacing(16)
        save = QtWidgets.QPushButton("Save processed…")
        save.setToolTip("Write the currently displayed (processed) log to a new .zarr")
        save.clicked.connect(self._on_save_clicked)
        bar.addWidget(save)

        bar.addStretch(1)
        return bar

    def _apply_svd(self, n_components: int):
        """Reprocess the full-resolution log with SVD-stripe removal and rebuild
        the pyramid in memory. ``n_components == 0`` restores the raw data."""
        if not self._is_atv:
            return
        if n_components == 0:
            processed = self._raw
        else:
            processed = remove_svd_components(self._raw, n_components)
        self._svd_removed = n_components
        self._levels = build_depth_pyramid(
            processed, min_rows=self._min_rows, factor=self._factor
        )
        self.n_levels = len(self._levels)
        self._refresh()
        self._auto_contrast()  # amplitudes shrink after destriping; rescale

    def save_processed(self, out_path: str | Path) -> Path:
        """Write the currently displayed pyramid to a new zarr group.

        Persists ``self._levels`` as they are right now — i.e. including any SVD
        destriping applied via the controls — so a processed log can be reopened
        directly without reprocessing. The number of SVD components removed is
        recorded in the store's attributes for provenance.

        Args:
            out_path: Destination ``.zarr`` path (created/overwritten).

        Returns:
            The path to the written zarr group.

        See Also:
            deeplogger.pyramid.write_zarr_pyramid: The underlying writer.
        """
        from deeplogger.pyramid import write_zarr_pyramid

        svd_n = self._svd_removed
        return write_zarr_pyramid(
            out_path,
            self._levels,
            self._depth,
            data_type=self.data_type,
            n_azimuth=self._n_azimuth,
            diameter=self._diameter,
            start_depth=self._start,
            stop_depth=self._stop,
            factor=self._factor,
            extra_attrs={"svd_removed": svd_n} if svd_n else None,
        )

    def _on_save_clicked(self):
        """Prompt for a path and save the processed log there."""
        start_dir = str(self._source_path.parent) if self._source_path else ""
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save processed log", start_dir, "Zarr group (*.zarr)"
        )
        if not path:
            return
        if not path.endswith(".zarr"):
            path += ".zarr"
        self.save_processed(path)

    def _set_colormap(self, name: str):
        if self._cbar is not None and name:
            self._cbar.setColorMap(_get_colormap(name))

    def _auto_contrast(self):
        """Set contrast to the 1st-99th percentile of the visible data."""
        if self._cbar is None:
            return
        arr = self._img.image
        if arr is None:
            return
        finite = arr[np.isfinite(arr)]
        if finite.size:
            lo, hi = np.percentile(finite, [1, 99])
            if hi > lo:
                self._cbar.setLevels((float(lo), float(hi)))

    # --- view / data -------------------------------------------------------

    def _nearest_row(self, depth_m: float) -> int:
        """Nearest full-resolution row index for a depth (monotonic vector)."""
        d = self._depth
        idx = int(np.searchsorted(d, depth_m))
        if idx <= 0:
            return 0
        if idx >= d.shape[0]:
            return d.shape[0] - 1
        return idx if (d[idx] - depth_m) < (depth_m - d[idx - 1]) else idx - 1

    def _viewport_px(self) -> int:
        """Pixel height of the plot area (>= 1)."""
        return max(1, int(self._plot.getViewBox().height()))

    def _on_range_changed(self, *_args):
        if not self._updating:
            self._refresh()

    def _refresh(self):
        """Load the level/window matching the current depth view and display it."""
        (_x0, _x1), (y0, y1) = self._plot.viewRange()
        top = max(self._start, min(y0, y1))
        bottom = min(self._stop, max(y0, y1))
        r0 = self._nearest_row(top)
        r1 = self._nearest_row(bottom)
        if r1 <= r0:
            r1 = min(r0 + 1, self._depth.shape[0])

        level = select_pyramid_level(r1 - r0, self._viewport_px(), self.n_levels)
        self._current_level = level  # exposed for tests/diagnostics
        lr0, lr1 = level_window(r0, r1, level)
        arr = np.asarray(self._levels[level][lr0:lr1])

        self._updating = True
        try:
            self._img.setImage(arr, autoLevels=False, autoRange=False)
            # Place the (possibly decimated) window so it spans its real depth
            # extent: switching levels changes sharpness, not position.
            d_top = float(self._depth[r0])
            d_bottom = float(self._depth[min(r1, self._depth.shape[0]) - 1])
            self._img.setRect(
                QtCore.QRectF(0.0, d_top, float(self._n_azimuth), d_bottom - d_top)
            )
        finally:
            self._updating = False

        if self._first:
            self._auto_contrast()  # sensible initial contrast from real data
            self._first = False

    # --- labeling ----------------------------------------------------------

    def _visible_row_range(self) -> tuple[int, int]:
        """Full-resolution row range currently in view (mirrors _refresh)."""
        (_x0, _x1), (y0, y1) = self._plot.viewRange()
        top = max(self._start, min(y0, y1))
        bottom = min(self._stop, max(y0, y1))
        r0 = self._nearest_row(top)
        r1 = self._nearest_row(bottom)
        if r1 <= r0:
            r1 = min(r0 + 1, self._depth.shape[0])
        return r0, r1

    def current_window(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the visible window as ``(image, depth)`` at full resolution.

        The image reflects any processing applied (e.g. SVD destriping), since
        it reads level 0 of the current pyramid.

        Returns:
            ``(image, depth)``: the full-resolution image for the visible depth
            range and its depth vector (meters).

        Raises:
            ValueError: If the visible range exceeds ``MAX_LABEL_ROWS`` rows.
        """
        from deeplogger.gui.labeler import MAX_LABEL_ROWS

        r0, r1 = self._visible_row_range()
        if r1 - r0 > MAX_LABEL_ROWS:
            raise ValueError(
                f"Visible window is {r1 - r0} rows (max {MAX_LABEL_ROWS} for "
                "labeling). Zoom in to a smaller depth range first."
            )
        image = np.asarray(self._levels[0][r0:r1])
        depth = self._depth[r0:r1]
        return image, depth

    def _on_label_clicked(self):
        """Open the visible window in napari for labeling (cap-guarded)."""
        try:
            image, depth = self.current_window()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Window too large", str(exc))
            return
        from deeplogger.gui.labeler import launch_labeler

        # Close any previous labeler window so they don't stack and overlap.
        prev = getattr(self, "_labeler", None)
        if prev is not None:
            try:
                prev.close()
            except Exception:
                pass

        name = self._source_path.stem if self._source_path else "window"
        # Keep a reference so the napari viewer is not garbage-collected.
        self._labeler = launch_labeler(
            image,
            depth,
            data_type=self.data_type,
            n_azimuth=self._n_azimuth,
            diameter=self._diameter,
            source_name=name,
        )

    def _on_infer_clicked(self):
        """Open the visible window in napari for inference."""
        try:
            image, depth = self.current_window()
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Window too large", str(exc))
            return
        from deeplogger.gui.inferencer import launch_inferencer

        prev = getattr(self, "_inferencer", None)
        if prev is not None:
            try:
                prev.close()
            except Exception:
                pass

        name = self._source_path.stem if self._source_path else "window"
        self._inferencer = launch_inferencer(
            image,
            depth,
            data_type=self.data_type,
            n_azimuth=self._n_azimuth,
            source_name=name,
        )


def launch_viewer(path: str | Path) -> None:
    """Open the browse viewer on a ``.zarr`` cache or a ``.las`` file.

    A ``.las`` file is converted to a cached pyramid (under ``<dir>/.cache``)
    before viewing; a ``.zarr`` path is opened directly.

    Args:
        path: Path to a ``.las`` log or a ``.zarr`` pyramid.
    """
    path = Path(path)
    app = pg.mkQApp("DeepLogger viewer")  # noqa: F841 (keeps the app alive)
    if path.suffix == ".las":
        from deeplogger.las_reader import BoreholeLog

        store = BoreholeLog.open(path).to_zarr(path.parent / ".cache")
    else:
        store = path
    viewer = LogViewer.from_zarr(store)
    viewer.setWindowTitle(f"DeepLogger — {path.name}")
    viewer.resize(500, 900)
    viewer.show()
    pg.exec()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python -m deeplogger.gui.viewer <log.las | cache.zarr>")
        sys.exit(1)
    launch_viewer(sys.argv[1])
