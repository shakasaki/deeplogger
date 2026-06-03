"""Bundle Inspector — browse a directory of [image, mask] .pt training pairs.

Opens every ``*.pt`` file in a chosen directory, interprets each as a
``[image, mask]`` bundle, and shows them side-by-side in a pyqtgraph window.
Supports ATV (H, W) and OTV (H, W, 3) images.

Navigation: Prev / Next buttons or left / right arrow keys.
Status bar: filename, index / total, and mask coverage %.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

pg.setConfigOption("imageAxisOrder", "row-major")


class BundleInspector(QtWidgets.QWidget):
    """Pyqtgraph widget for browsing [image, mask] .pt bundle files.

    Args:
        directory: Path to a directory containing ``*.pt`` bundles.
            Each bundle must be a list or tuple ``[image, mask]``.
    """

    def __init__(self, directory: str | Path):
        super().__init__()
        self._dir = Path(directory)
        self._files: list[str] = sorted(
            glob.glob(str(self._dir / "*.pt"))
        )
        if not self._files:
            raise ValueError(f"No .pt files found in {directory}")
        self._idx: int = 0
        self._build_ui()
        self.setWindowTitle(f"Bundle Inspector — {self._dir.name}")
        self.resize(1100, 600)
        self._load(0)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 4)

        # Top row: two image panels
        panels = QtWidgets.QHBoxLayout()

        self._img_view = pg.ImageView(name="Image")
        self._img_view.setColorMap(pg.colormap.get("afmhot", source="matplotlib"))
        self._img_view.ui.roiBtn.hide()
        self._img_view.ui.menuBtn.hide()

        self._mask_view = pg.ImageView(name="Mask")
        self._mask_view.setColorMap(pg.colormap.get("viridis", source="matplotlib"))
        self._mask_view.ui.roiBtn.hide()
        self._mask_view.ui.menuBtn.hide()

        # Labels above panels
        img_group  = QtWidgets.QGroupBox("Image")
        mask_group = QtWidgets.QGroupBox("Mask / Label")
        img_group.setLayout(QtWidgets.QVBoxLayout())
        mask_group.setLayout(QtWidgets.QVBoxLayout())
        img_group.layout().addWidget(self._img_view)
        mask_group.layout().addWidget(self._mask_view)

        panels.addWidget(img_group)
        panels.addWidget(mask_group)
        layout.addLayout(panels)

        # Navigation row
        nav = QtWidgets.QHBoxLayout()
        self._prev_btn = QtWidgets.QPushButton("◀  Prev")
        self._next_btn = QtWidgets.QPushButton("Next  ▶")
        self._prev_btn.setShortcut(QtCore.Qt.Key_Left)
        self._next_btn.setShortcut(QtCore.Qt.Key_Right)
        self._prev_btn.clicked.connect(self._prev)
        self._next_btn.clicked.connect(self._next)

        self._status = QtWidgets.QLabel()
        self._status.setAlignment(QtCore.Qt.AlignCenter)

        nav.addWidget(self._prev_btn)
        nav.addStretch()
        nav.addWidget(self._status)
        nav.addStretch()
        nav.addWidget(self._next_btn)
        layout.addLayout(nav)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def _prev(self) -> None:
        if self._idx > 0:
            self._load(self._idx - 1)

    def _next(self) -> None:
        if self._idx < len(self._files) - 1:
            self._load(self._idx + 1)

    def keyPressEvent(self, event) -> None:
        if event.key() == QtCore.Qt.Key_Left:
            self._prev()
        elif event.key() == QtCore.Qt.Key_Right:
            self._next()
        else:
            super().keyPressEvent(event)

    # ------------------------------------------------------------------
    # Loading and display
    # ------------------------------------------------------------------

    def _load(self, idx: int) -> None:
        import torch

        self._idx = idx
        path = self._files[idx]
        bundle = torch.load(path, map_location="cpu", weights_only=False)

        img  = np.asarray(bundle[0], dtype=np.float32)
        mask = np.asarray(bundle[1], dtype=np.float32)

        # For OTV (H, W, 3) convert to grayscale for the image panel
        # by taking the luminance-weighted mean so structure is visible.
        if img.ndim == 3 and img.shape[-1] == 3:
            display_img = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
        else:
            display_img = img.squeeze()

        # Normalise image to [0, 1] for display
        lo, hi = display_img.min(), display_img.max()
        if hi > lo:
            display_img = (display_img - lo) / (hi - lo)

        self._img_view.setImage(display_img, autoRange=True, autoLevels=True)
        self._mask_view.setImage(mask.squeeze(), autoRange=True, autoLevels=True)

        n_total = mask.size
        coverage = float(np.count_nonzero(mask)) / max(n_total, 1) * 100.0

        fname = os.path.basename(path)
        self._status.setText(
            f"{fname}   [{idx + 1} / {len(self._files)}]   "
            f"mask: {coverage:.2f}%   img shape: {img.shape}"
        )
        self._prev_btn.setEnabled(idx > 0)
        self._next_btn.setEnabled(idx < len(self._files) - 1)


def launch_bundle_inspector(directory: Optional[str | Path] = None) -> BundleInspector:
    """Open the bundle inspector, optionally prompting for a directory.

    If *directory* is ``None`` a folder-selection dialog is shown.

    Args:
        directory: Path to a directory of ``.pt`` bundles, or ``None``.

    Returns:
        The :class:`BundleInspector` widget.

    Raises:
        ValueError: If no ``.pt`` files are found in the chosen directory.
    """
    if directory is None:
        chosen = QtWidgets.QFileDialog.getExistingDirectory(
            None, "Select bundle directory"
        )
        if not chosen:
            raise ValueError("No directory selected.")
        directory = chosen

    inspector = BundleInspector(directory)
    inspector.show()
    return inspector
