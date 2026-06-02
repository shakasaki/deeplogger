"""napari labeling for a windowed borehole log selection.

The browse viewer (``viewer.py``) hands a small depth window — the image
currently in view, at full resolution — to :func:`launch_labeler`, which opens
a napari window beside it for creating a fracture label. Two label modes share
one Labels layer:

- **Mask painting:** napari's native Labels brush, for freehand masks.
- **Parametric sinusoid:** a magicgui panel whose sliders are the geological
  ``Fracture`` parameters (depth / dip / azimuth / aperture, plus the per-log
  diameter). "Add pick" rasterizes the fracture's sinusoidal trace into the
  same Labels layer via :func:`deeplogger.labels.get_label`.

The result saves as a ``[image, mask]`` tensor bundle (the training-data
format). The heavy napari/magicgui imports are deferred into ``launch_labeler``
so the pure helpers here stay importable without a display.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from deeplogger.config import DataType, Fracture
from deeplogger.labels import get_label

# A window wider than this many full-resolution depth rows is refused for
# labeling — napari would struggle and labeling is meant for small windows.
MAX_LABEL_ROWS = 20_000


def build_label_mask(
    fractures: list[Fracture],
    depth: np.ndarray,
    n_azimuth: int,
    diameter: float,
) -> np.ndarray:
    """Rasterize a set of fractures into one binary mask.

    Each fracture's sinusoidal band (from :func:`deeplogger.labels.get_label`)
    is OR-ed together, so overlapping picks merge into a single mask.

    Args:
        fractures: Fracture picks to rasterize.
        depth: Depth vector (meters) for each mask row.
        n_azimuth: Number of azimuth columns.
        diameter: Borehole diameter in meters (for the dip→amplitude geometry).

    Returns:
        Binary uint8 mask of shape ``(len(depth), n_azimuth)``.
    """
    mask = np.zeros((depth.shape[0], n_azimuth), dtype=np.uint8)
    for fracture in fractures:
        band = get_label(fracture, depth, diameter, n_azimuth)
        mask[band > 0] = 1
    return mask


def save_label_bundle(
    path: str | Path, image: np.ndarray, mask: np.ndarray
) -> Path:
    """Save an ``[image, mask]`` tensor bundle (the training-data format).

    Args:
        path: Destination ``.pt`` file.
        image: Window image (ATV ``(H, W)`` float32 or OTV ``(H, W, 3)`` uint8).
        mask: Binary label of shape ``(H, W)``.

    Returns:
        The path written.
    """
    import torch

    path = Path(path)
    bundle = [
        torch.from_numpy(np.ascontiguousarray(image)),
        torch.from_numpy(np.ascontiguousarray(mask.astype(np.uint8))),
    ]
    torch.save(bundle, path)
    return path


def launch_labeler(
    image: np.ndarray,
    depth: np.ndarray,
    *,
    data_type: DataType,
    n_azimuth: int,
    diameter: float | None,
    source_name: str = "window",
):
    """Open a napari window to label a windowed log selection.

    Adds the window image and an empty Labels layer (for freehand painting),
    plus a magicgui sinusoid-picker panel docked on the right. Returns the
    napari viewer so the caller can keep a reference (preventing GC).

    Args:
        image: Window image — ATV ``(H, W)`` or OTV ``(H, W, 3)``.
        depth: Depth vector (meters) for the window rows.
        data_type: ATV or OTV (selects RGB vs single-channel display).
        n_azimuth: Number of azimuth columns.
        diameter: Borehole diameter in meters; seeds the picker (editable).
        source_name: Label shown in the window title and save dialog default.

    Returns:
        The :class:`napari.Viewer` instance.
    """
    import napari
    from magicgui.widgets import Container, FloatSlider, FloatSpinBox, Label, PushButton
    from qtpy.QtWidgets import QFileDialog

    viewer = napari.Viewer(title=f"DeepLogger label — {source_name}")
    if data_type is DataType.OTV:
        viewer.add_image(image, name="log", rgb=True)
    else:
        viewer.add_image(image, name="log", colormap="afmhot")
    mask_layer = viewer.add_labels(
        np.zeros(image.shape[:2], dtype=np.uint8), name="label"
    )

    fractures: list[Fracture] = []
    d_lo, d_hi = float(depth.min()), float(depth.max())

    depth_w = FloatSlider(min=d_lo, max=d_hi, value=(d_lo + d_hi) / 2, label="depth [m]")
    dip_w = FloatSlider(min=0.0, max=89.0, value=30.0, label="dip [deg]")
    az_w = FloatSlider(min=0.0, max=360.0, value=0.0, label="azimuth [deg]")
    ap_w = FloatSpinBox(min=0.0, max=100.0, value=5.0, label="aperture [mm]")
    diam_w = FloatSpinBox(
        min=0.01, max=1.0, value=float(diameter or 0.1), label="diameter [m]"
    )
    add_btn = PushButton(text="Add pick")
    save_btn = PushButton(text="Save .pt")

    def _current_fracture() -> Fracture:
        return Fracture(
            azimuth=az_w.value, dip=dip_w.value, depth=depth_w.value, aperture=ap_w.value
        )

    def _add_pick():
        fracture = _current_fracture()
        fractures.append(fracture)
        band = get_label(fracture, depth, diam_w.value, n_azimuth)
        data = np.asarray(mask_layer.data)
        data[band > 0] = 1  # merge with any freehand painting
        mask_layer.data = data
        mask_layer.refresh()

    def _save():
        path, _ = QFileDialog.getSaveFileName(
            None, "Save label", f"{source_name}_label.pt", "PyTorch bundle (*.pt)"
        )
        if not path:
            return
        if not path.endswith(".pt"):
            path += ".pt"
        save_label_bundle(path, image, np.asarray(mask_layer.data))

    add_btn.clicked.connect(_add_pick)
    save_btn.clicked.connect(_save)
    panel = Container(
        widgets=[
            Label(value="Sinusoid pick (or paint the label layer)"),
            depth_w, dip_w, az_w, ap_w, diam_w, add_btn, save_btn,
        ]
    )
    viewer.window.add_dock_widget(panel, area="right", name="Fracture picker")
    return viewer
