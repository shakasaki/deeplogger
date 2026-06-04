"""DeepLogger main launcher window.

Presents three entry points so the user can choose a workflow before any
data is loaded.  Replaces the bare file-open dialog that was previously
shown when the viewer was started with no arguments.

Entry points
------------
Browse / Label
    Opens a ``.las`` or ``.zarr`` file in the :class:`LogViewer`.
    From there the user can scroll the log, apply SVD destriping, open
    a depth window in napari for labelling, and run inference.

Inspect Bundles
    Opens a directory of ``[image, mask]`` ``.pt`` bundles in the
    :class:`BundleInspector` for quality-checking training data.
"""

from __future__ import annotations

from pathlib import Path

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets


class Launcher(QtWidgets.QDialog):
    """Start-screen dialog that routes the user to the right workflow.

    Args:
        parent: Optional parent widget.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("DeepLogger")
        self.setWindowFlags(
            self.windowFlags() & ~QtCore.Qt.WindowType.WindowContextHelpButtonHint
        )
        self.setMinimumWidth(340)
        self._build_ui()
        self._viewer = None
        self._inspector = None
        self._path = None

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(14)
        layout.setContentsMargins(28, 24, 28, 24)

        title = QtWidgets.QLabel("<b>DeepLogger</b>")
        title.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("font-size: 18px; margin-bottom: 4px;")
        layout.addWidget(title)

        subtitle = QtWidgets.QLabel(
            "Borehole televiewer log viewer,\n"
            "fracture labeler and deep-learning inference tool"
        )
        subtitle.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(subtitle)

        layout.addSpacing(8)

        browse_btn = self._make_btn(
            "Browse / Label",
            "Open a LAS or zarr log.\n"
            "Scroll, destripe (SVD), label windows in napari, run inference.",
            self._on_browse,
        )
        layout.addWidget(browse_btn)

        inspect_btn = self._make_btn(
            "Inspect Bundles",
            "Browse a directory of [image, mask] .pt training bundles.\n"
            "Verify label quality before training.",
            self._on_inspect,
        )
        layout.addWidget(inspect_btn)

    @staticmethod
    def _make_btn(title: str, tooltip: str, slot) -> QtWidgets.QPushButton:
        btn = QtWidgets.QPushButton(title)
        btn.setToolTip(tooltip)
        btn.setMinimumHeight(48)
        btn.setStyleSheet(
            "QPushButton { font-size: 13px; font-weight: bold; border-radius: 6px; }"
            "QPushButton:hover { background: #3a7dcf; color: white; }"
        )
        btn.clicked.connect(slot)
        return btn

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_browse(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open borehole log",
            str(Path.home()),
            "Borehole logs (*.las *.zarr);;All files (*)",
        )
        if not path:
            return
        # Record the choice and close; the viewer is launched at the top level
        # in launch() so we don't start a nested event loop inside this slot.
        self._path = path
        self.accept()

    def _on_inspect(self) -> None:
        directory = QtWidgets.QFileDialog.getExistingDirectory(
            self, "Select bundle directory"
        )
        if not directory:
            return
        try:
            from deeplogger.gui.bundle_inspector import BundleInspector
            self._inspector = BundleInspector(directory)
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "No bundles found", str(exc))
            return
        # Shown at the top level in launch(); see _on_browse.
        self.accept()


def launch() -> None:
    """Start the application: show the launcher, then hand off to user choice."""
    app = pg.mkQApp("DeepLogger")  # noqa: F841 (keeps the app alive)
    launcher = Launcher()
    launcher.exec()
    # Dispatch the user's choice AFTER the launcher's modal loop has unwound,
    # so the viewer/inspector runs the only event loop (no nesting).
    if launcher._path is not None:
        from deeplogger.gui.viewer import launch_viewer
        launch_viewer(launcher._path)
    elif launcher._inspector is not None:
        launcher._inspector.show()
        pg.exec()
    # else: user closed the launcher without choosing — just exit.
