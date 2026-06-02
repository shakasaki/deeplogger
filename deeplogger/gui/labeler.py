"""napari labeling for a windowed borehole log selection.

The browse viewer (``viewer.py``) hands a small depth window — the image
currently in view, at full resolution — to :func:`launch_labeler`, which opens
a napari window beside it for picking fractures and drawing labels.

**Interactive sinusoid picking.** "Pick structure" arms a mouse mode on the
image: the click row sets the fracture's centre depth; dragging horizontally
shifts the sinusoid's phase (azimuth, full image width = 360°); dragging
vertically grows its amplitude (dip, via the borehole diameter). A live curve
tracks the gesture on a Shapes layer. "Save pick" commits the current curve —
rasterising its band into the Labels layer and appending it to the session pick
list. The Labels layer also accepts freehand brush painting.

**Saving** (all to the output directory, no dialog by default — editable in the
Output settings panel):
- *Save label* → ``[image, mask]`` ``.pt`` bundle named
  ``<borehole>_<start>m_<end>m.pt``.
- *Save picks* → appends the session's picks to ``<borehole>_picks.csv``
  (columns: Borehole, Depth (m), Dip (deg), Azimuth (deg)).

The heavy napari/magicgui imports are deferred into :func:`launch_labeler` so
the pure helpers here stay importable without a display.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from deeplogger import OUTPUT_DIR
from deeplogger.config import DataType, Fracture
from deeplogger.labels import get_label

# A window wider than this many full-resolution depth rows is refused for
# labeling — napari would struggle and labeling is meant for small windows.
MAX_LABEL_ROWS = 20_000

# CSV columns for the picks table.
PICKS_COLUMNS = ["Borehole", "Depth (m)", "Dip (deg)", "Azimuth (deg)"]


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


def gesture_to_fracture(
    click_row: int,
    dx_cols: float,
    dy_rows: float,
    depth: np.ndarray,
    n_azimuth: int,
    diameter: float,
    aperture: float,
) -> Fracture:
    """Map a click-and-drag gesture to a :class:`Fracture`.

    The click row sets the centre depth; horizontal drag sets the azimuth
    (phase) with one full image width spanning 360°; vertical drag sets the dip
    via the curve's amplitude — ``amplitude = (D/2)/cos(dip)`` inverted, so a
    bigger vertical swing means a steeper dip.

    Args:
        click_row: Row index where the drag started.
        dx_cols: Horizontal drag in azimuth columns.
        dy_rows: Vertical drag in depth rows.
        depth: Window depth vector (meters), monotonically increasing.
        n_azimuth: Number of azimuth columns.
        diameter: Borehole diameter in meters.
        aperture: Fracture aperture in millimeters (band thickness).

    Returns:
        The Fracture described by the gesture.
    """
    n_rows = depth.shape[0]
    row = int(np.clip(click_row, 0, n_rows - 1))
    centre_depth = float(depth[row])
    azimuth = float((dx_cols / n_azimuth) * 360.0) % 360.0

    step_m = (float(depth[-1]) - float(depth[0])) / max(n_rows - 1, 1)
    swing_m = abs(dy_rows) * abs(step_m)
    half_d = diameter / 2.0
    amplitude = max(swing_m, half_d)  # amplitude cannot be below D/2 (dip 0)
    dip = float(np.degrees(np.arccos(np.clip(half_d / amplitude, 0.0, 1.0))))
    dip = float(np.clip(dip, 0.0, 89.0))
    return Fracture(azimuth=azimuth, dip=dip, depth=centre_depth, aperture=aperture)


def sinusoid_curve(
    fracture: Fracture, depth: np.ndarray, n_azimuth: int, diameter: float
) -> np.ndarray:
    """Compute the fracture's sinusoidal trace as ``(row, col)`` points.

    Mirrors the centre line of :func:`deeplogger.labels.get_label` (no aperture
    band), mapped from depth-meters into fractional row coordinates so it can be
    drawn on a napari Shapes layer over the window image.

    Args:
        fracture: The fracture to trace.
        depth: Window depth vector (meters), monotonically increasing.
        n_azimuth: Number of azimuth columns.
        diameter: Borehole diameter in meters.

    Returns:
        Array of shape ``(n_azimuth, 2)`` with ``(row, col)`` per azimuth.
    """
    n_rows = depth.shape[0]
    az_rad = np.linspace(0.0, 2.0 * np.pi, n_azimuth)
    amplitude = fracture.depth + np.cos(
        az_rad + np.deg2rad(fracture.azimuth)
    ) * (1.0 / np.cos(np.deg2rad(fracture.dip))) * (diameter / 2.0)
    rows = np.interp(amplitude, depth, np.arange(n_rows))
    cols = np.arange(n_azimuth)
    return np.column_stack([rows, cols])


def default_label_filename(borehole: str, depth: np.ndarray) -> str:
    """Default ``.pt`` filename: ``<borehole>_<start>m_<end>m.pt``."""
    return f"{borehole}_{float(depth[0]):.2f}m_{float(depth[-1]):.2f}m.pt"


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
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = [
        torch.from_numpy(np.ascontiguousarray(image)),
        torch.from_numpy(np.ascontiguousarray(mask.astype(np.uint8))),
    ]
    torch.save(bundle, path)
    return path


def append_picks_csv(
    path: str | Path, borehole: str, picks: list[Fracture]
) -> Path:
    """Append fracture picks to a per-borehole CSV (created with a header).

    Args:
        path: CSV path (e.g. ``<output>/<borehole>_picks.csv``).
        borehole: Borehole name written into the ``Borehole`` column.
        picks: Fractures to append as rows.

    Returns:
        The CSV path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(PICKS_COLUMNS)
        for fr in picks:
            writer.writerow(
                [borehole, f"{fr.depth:.4f}", f"{fr.dip:.2f}", f"{fr.azimuth:.2f}"]
            )
    return path


def launch_labeler(
    image: np.ndarray,
    depth: np.ndarray,
    *,
    data_type: DataType,
    n_azimuth: int,
    diameter: float | None,
    source_name: str = "window",
    output_dir: str | Path | None = None,
):
    """Open a napari window to label a windowed log selection.

    See the module docstring for the interaction and saving behaviour. Returns
    the napari viewer so the caller can keep a reference (preventing GC).

    Args:
        image: Window image — ATV ``(H, W)`` or OTV ``(H, W, 3)``.
        depth: Depth vector (meters) for the window rows.
        data_type: ATV or OTV (selects RGB vs single-channel display).
        n_azimuth: Number of azimuth columns.
        diameter: Borehole diameter in meters; seeds the picker (editable).
        source_name: Default borehole name (title, filenames, CSV).
        output_dir: Default save directory (defaults to ``deeplogger.OUTPUT_DIR``).

    Returns:
        The :class:`napari.Viewer` instance.
    """
    import napari
    from magicgui.widgets import Container, FloatSpinBox, Label, LineEdit, PushButton

    viewer = napari.Viewer(title=f"DeepLogger label — {source_name}")
    if data_type is DataType.OTV:
        image_layer = viewer.add_image(image, name="log", rgb=True)
    else:
        image_layer = viewer.add_image(image, name="log", colormap="afmhot")
    mask_layer = viewer.add_labels(
        np.zeros(image.shape[:2], dtype=np.uint8), name="label"
    )
    curve_layer = viewer.add_shapes(
        name="pick", edge_color="cyan", face_color="transparent", edge_width=1.0
    )

    state = {"picks": [], "saved_n": 0, "picking": False, "current": None}

    # --- picker panel ------------------------------------------------------
    pick_btn = PushButton(text="Pick structure")
    diam_w = FloatSpinBox(min=0.01, max=1.0, value=float(diameter or 0.1), label="diameter [m]")
    aperture_w = FloatSpinBox(min=0.0, max=100.0, value=5.0, label="aperture [mm]")
    readout = Label(value="(drag on the image to pick)")
    save_pick_btn = PushButton(text="Save pick")

    # --- save panel --------------------------------------------------------
    save_label_btn = PushButton(text="Save label")
    save_picks_btn = PushButton(text="Save picks")
    status = Label(value="")

    # --- output settings panel --------------------------------------------
    borehole_w = LineEdit(value=source_name, label="Borehole")
    out_dir_w = LineEdit(value=str(output_dir or OUTPUT_DIR), label="Output dir")

    def _set_curve(points: np.ndarray):
        curve_layer.data = [points]
        curve_layer.shape_type = "path"

    def _toggle_pick():
        state["picking"] = not state["picking"]
        pick_btn.text = "Picking… (drag image)" if state["picking"] else "Pick structure"

    def _on_drag(layer, event):
        if not state["picking"]:
            return
        r0, c0 = layer.world_to_data(event.position)[:2]
        click_row = int(round(r0))
        yield
        while event.type == "mouse_move":
            r, c = layer.world_to_data(event.position)[:2]
            fr = gesture_to_fracture(
                click_row, c - c0, r - r0, depth, n_azimuth, diam_w.value, aperture_w.value
            )
            state["current"] = fr
            _set_curve(sinusoid_curve(fr, depth, n_azimuth, diam_w.value))
            readout.value = f"depth {fr.depth:.2f} m | dip {fr.dip:.1f}° | az {fr.azimuth:.1f}°"
            yield

    image_layer.mouse_drag_callbacks.append(_on_drag)

    def _save_pick():
        fr = state["current"]
        if fr is None:
            status.value = "No pick to save — drag on the image first."
            return
        state["picks"].append(fr)
        band = get_label(fr, depth, diam_w.value, n_azimuth)
        data = np.asarray(mask_layer.data)
        data[band > 0] = 1
        mask_layer.data = data
        mask_layer.refresh()
        state["current"] = None
        curve_layer.data = []
        status.value = f"{len(state['picks'])} pick(s) staged."

    def _save_label():
        out = Path(out_dir_w.value)
        fname = default_label_filename(borehole_w.value, depth)
        path = save_label_bundle(out / fname, image, np.asarray(mask_layer.data))
        status.value = f"Saved label → {path}"

    def _save_picks():
        new = state["picks"][state["saved_n"]:]
        if not new:
            status.value = "No new picks to save."
            return
        out = Path(out_dir_w.value)
        path = append_picks_csv(out / f"{borehole_w.value}_picks.csv", borehole_w.value, new)
        state["saved_n"] = len(state["picks"])
        status.value = f"Appended {len(new)} pick(s) → {path}"

    pick_btn.clicked.connect(_toggle_pick)
    save_pick_btn.clicked.connect(_save_pick)
    save_label_btn.clicked.connect(_save_label)
    save_picks_btn.clicked.connect(_save_picks)

    picker = Container(widgets=[
        Label(value="Pick structure (or paint the label layer)"),
        pick_btn, diam_w, aperture_w, readout, save_pick_btn,
    ])
    save_panel = Container(widgets=[Label(value="Save"), save_label_btn, save_picks_btn, status])
    settings = Container(widgets=[Label(value="Output settings"), borehole_w, out_dir_w])
    viewer.window.add_dock_widget(picker, area="left", name="Picker")
    viewer.window.add_dock_widget(save_panel, area="left", name="Save")
    viewer.window.add_dock_widget(settings, area="left", name="Output settings")
    return viewer
