"""napari inference viewer for a windowed borehole log selection.

Mirrors the labeler (``labeler.py``) but replaces fracture picking with U-Net
model loading and inference.

**Model panel**: select a ``.pt`` model from discovered dirs or type a path;
load it to see ATV/OTV type and compute device.

**Inference panel**: "Run inference" runs the loaded model on the window image
and displays the probability map. The threshold slider updates the binary
overlay live. If ground truth is available, shows the Dice coefficient.

**Save panel**: saves ``[image, prediction]`` as a ``.pt`` bundle.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np

from deeplogger import DATA_DIR, OUTPUT_DIR
from deeplogger.config import DataType

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

_MODEL_SEARCH_DIRS = [
    os.path.join(_REPO_ROOT, "models"),
    os.path.join(DATA_DIR, "IDs_and_model"),
    os.path.join(DATA_DIR, "models"),
]


def find_models() -> list[str]:
    """Return paths to .pt state-dict files in the default model directories."""
    import torch

    found = []
    for d in _MODEL_SEARCH_DIRS:
        if not os.path.isdir(d):
            continue
        for f in sorted(glob.glob(os.path.join(d, "*.pt"))):
            try:
                peek = torch.load(f, map_location="cpu", weights_only=False)
                if isinstance(peek, dict):
                    found.append(f)
            except Exception:
                continue
    return found


def detect_model_channels(model_path: str) -> int:
    """Detect U-Net input channels from a saved state dict."""
    import torch

    sd = torch.load(model_path, map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "encoder1.enc1conv1.weight" in sd:
        return int(sd["encoder1.enc1conv1.weight"].shape[1])
    return 3


def load_model(model_path: str):
    """Load a U-Net from a state-dict file. Returns ``(model, device, in_channels)``."""
    import torch
    from deeplogger.model_architectures_ATV import UNetOTV as UNetATV
    from deeplogger.model_architectures_OTV import UNetOTV

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    in_ch = detect_model_channels(model_path)
    if in_ch == 3:
        model = UNetOTV(in_channels=3, out_channels=1, init_features=32)
    else:
        model = UNetATV(in_channels=1, out_channels=1, init_features=32)
    sd = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(sd)
    model.to(device).eval()
    return model, device, in_ch


def run_predict(model, image: np.ndarray, device, in_channels: int) -> np.ndarray:
    """Run inference on a single window image. Returns a (H, W) probability map.

    Both UNetOTV and UNetATV handle channel reordering internally in forward():
    - OTV forward() does permute(0,3,1,2), so expects (1,H,W,3) — channel-last.
    - ATV forward() does unsqueeze(0).permute(1,0,2,3), so expects (1,H,W).
    """
    import torch

    img = np.asarray(image, dtype=np.float32)
    if in_channels == 3:
        # OTV needs (1, H, W, 3)
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.shape[-1] != 3:
            img = img[:, :, :3]
    else:
        # ATV needs (1, H, W)
        if img.ndim == 3:
            img = img[:, :, 0]
    tensor = torch.from_numpy(img).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(tensor)
    return pred.squeeze().cpu().numpy()


def compute_dice(prediction: np.ndarray, ground_truth: np.ndarray, threshold: float) -> float:
    """Dice coefficient between thresholded prediction and binary ground truth."""
    pred = (prediction > threshold).astype(float)
    gt = (ground_truth > 0).astype(float)
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return 2.0 * float(np.sum(pred * gt)) / denom


def launch_inferencer(
    image: np.ndarray,
    depth: np.ndarray,
    *,
    data_type: DataType,
    n_azimuth: int,
    mask: np.ndarray | None = None,
    source_name: str = "window",
    output_dir: str | Path | None = None,
):
    """Open a napari window to run inference on a windowed log selection.

    Args:
        image: Window image — ATV ``(H, W)`` float or OTV ``(H, W, 3)`` uint8.
        depth: Depth vector (meters) for the window rows.
        data_type: ATV or OTV.
        n_azimuth: Number of azimuth columns.
        mask: Optional ground truth binary mask ``(H, W)`` for Dice display.
        source_name: Title and default filename stem.
        output_dir: Default save directory (defaults to ``deeplogger.OUTPUT_DIR``).

    Returns:
        The :class:`napari.Viewer` instance.
    """
    import napari
    from magicgui.widgets import (
        ComboBox,
        Container,
        FloatSlider,
        Label,
        LineEdit,
        PushButton,
    )

    viewer = napari.Viewer(title=f"DeepLogger infer — {source_name}")

    if data_type is DataType.OTV:
        viewer.add_image(image, name="log", rgb=True)
    else:
        viewer.add_image(image, name="log", colormap="afmhot")

    has_gt = mask is not None and mask.sum() > 0
    if has_gt:
        viewer.add_labels(mask.astype(np.uint8), name="ground truth", opacity=0.4)

    pred_layer = viewer.add_image(
        np.zeros(image.shape[:2], dtype=np.float32),
        name="prediction",
        colormap="hot",
        opacity=0.6,
        contrast_limits=[0.0, 1.0],
        visible=False,
    )
    overlay_layer = viewer.add_labels(
        np.zeros(image.shape[:2], dtype=np.uint8),
        name="overlay",
        opacity=0.4,
        visible=False,
    )

    state = {"model": None, "device": None, "in_channels": None, "prediction": None}

    # --- model panel --------------------------------------------------------
    available = find_models()
    if available:
        model_combo = ComboBox(choices=available, label="Model", value=available[0])
    else:
        model_combo = ComboBox(choices=["(none found)"], label="Model")
    load_btn = PushButton(text="Load model")
    model_status = Label(value="Not loaded")

    def _load_model():
        path = model_combo.value
        if not os.path.isfile(path):
            model_status.value = f"Not found: {path}"
            return
        try:
            model, device, in_ch = load_model(path)
        except Exception as exc:
            model_status.value = f"Error: {exc}"
            return
        state.update(model=model, device=device, in_channels=in_ch, prediction=None)
        kind = "OTV 3-ch" if in_ch == 3 else "ATV 1-ch"
        model_status.value = f"{kind} | {device}"

    load_btn.clicked.connect(_load_model)

    model_panel = Container(widgets=[
        Label(value="Select and load a .pt model"),
        model_combo, load_btn, model_status,
    ])

    # --- inference panel ----------------------------------------------------
    threshold_w = FloatSlider(min=0.0, max=1.0, value=0.5, label="threshold")
    run_btn = PushButton(text="Run inference")
    dice_label = Label(value="")
    infer_status = Label(value="")

    def _update_overlay(*_):
        if state["prediction"] is None:
            return
        binary = (state["prediction"] > threshold_w.value).astype(np.uint8)
        overlay_layer.data = binary
        overlay_layer.visible = True
        if has_gt:
            dice = compute_dice(state["prediction"], mask, threshold_w.value)
            dice_label.value = f"Dice: {dice:.3f}"

    def _run_inference():
        if state["model"] is None:
            infer_status.value = "Load a model first."
            return
        infer_status.value = "Running…"
        try:
            pred = run_predict(state["model"], image, state["device"], state["in_channels"])
        except Exception as exc:
            infer_status.value = f"Error: {exc}"
            return
        state["prediction"] = pred
        pred_layer.data = pred.astype(np.float32)
        pred_layer.visible = True
        _update_overlay()
        infer_status.value = "Done."

    run_btn.clicked.connect(_run_inference)
    threshold_w.changed.connect(_update_overlay)

    infer_panel = Container(widgets=[
        Label(value="Run inference, then adjust threshold"),
        threshold_w, run_btn, dice_label, infer_status,
    ])

    # --- save panel ---------------------------------------------------------
    out_dir_w = LineEdit(value=str(output_dir or OUTPUT_DIR), label="Output dir")
    save_btn = PushButton(text="Save prediction (.pt)")
    save_status = Label(value="")

    def _save_prediction():
        if state["prediction"] is None:
            save_status.value = "No prediction to save."
            return
        import torch

        out = Path(out_dir_w.value)
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{source_name}_prediction.pt"
        bundle = [
            torch.from_numpy(np.ascontiguousarray(image)),
            torch.from_numpy(np.ascontiguousarray(state["prediction"].astype(np.float32))),
        ]
        torch.save(bundle, path)
        save_status.value = f"Saved → {path}"

    save_btn.clicked.connect(_save_prediction)
    save_panel = Container(widgets=[
        Label(value="Save"), save_btn, out_dir_w, save_status,
    ])

    viewer.window.add_dock_widget(model_panel, area="left", name="Model")
    viewer.window.add_dock_widget(infer_panel, area="left", name="Inference")
    viewer.window.add_dock_widget(save_panel, area="left", name="Save")
    return viewer
