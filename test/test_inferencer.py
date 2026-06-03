"""Tests for pure helpers in deeplogger.gui.inferencer.

The napari shell (launch_inferencer) requires a display; these cover only
the logic that backs it: height validation, predict shape, and Dice metric.
"""

import numpy as np
import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# _valid_height
# ---------------------------------------------------------------------------

from deeplogger.gui.inferencer import _valid_height


def test_valid_height_already_valid():
    # H % 16 == 8 must pass through unchanged
    for k in range(0, 5):
        H = 16 * k + 8
        if H == 0:
            continue
        assert _valid_height(H) == H


def test_valid_height_result_satisfies_constraint():
    for H in range(1, 300):
        H_pad = _valid_height(H)
        assert H_pad >= H
        assert H_pad % 16 == 8, f"_valid_height({H}) = {H_pad}, but {H_pad} % 16 != 8"


# ---------------------------------------------------------------------------
# run_predict — uses tiny synthetic models to avoid loading real .pt files
# ---------------------------------------------------------------------------

from deeplogger.gui.inferencer import run_predict


def _tiny_atv_model():
    """Minimal 1-channel model that returns a (H, W) sigmoid output."""

    class _TinyATV(nn.Module):
        def forward(self, x):
            # x: (1, H, W) — ATV forward unsqueezes then permutes
            x = x.unsqueeze(0).permute(1, 0, 2, 3)
            return torch.sigmoid(x.mean(dim=1, keepdim=True)).squeeze(0)

    return _TinyATV().eval()


def _tiny_otv_model():
    """Minimal 3-channel model that returns a (H, W) sigmoid output."""

    class _TinyOTV(nn.Module):
        def forward(self, x):
            # x: (1, H, W, 3) — OTV forward permutes to (1, 3, H, W)
            x = x.permute(0, 3, 1, 2)
            return torch.sigmoid(x.mean(dim=1, keepdim=True)).squeeze()

    return _TinyOTV().eval()


@pytest.mark.parametrize("H,W", [(100, 360), (200, 360), (17, 100), (89, 178), (33, 55)])
def test_run_predict_atv_output_shape(H, W):
    model = _tiny_atv_model()
    image = np.random.rand(H, W).astype(np.float32)
    pred = run_predict(model, image, torch.device("cpu"), in_channels=1)
    assert pred.shape == (H, W), f"Expected ({H},{W}), got {pred.shape}"


@pytest.mark.parametrize("H,W", [(100, 360), (200, 360), (17, 100), (89, 178), (33, 55)])
def test_run_predict_otv_output_shape(H, W):
    model = _tiny_otv_model()
    image = np.random.randint(0, 255, (H, W, 3), dtype=np.uint8)
    pred = run_predict(model, image, torch.device("cpu"), in_channels=3)
    assert pred.shape == (H, W), f"Expected ({H},{W}), got {pred.shape}"


def test_run_predict_atv_grayscale_from_rgb():
    """Single-channel model on an (H, W, 3) image: should use first channel."""
    model = _tiny_atv_model()
    H, W = 24, 60
    image = np.random.rand(H, W, 3).astype(np.float32)
    pred = run_predict(model, image, torch.device("cpu"), in_channels=1)
    assert pred.shape == (H, W)


def test_run_predict_otv_grayscale_stacked():
    """3-channel model on a (H, W) image: should stack into 3 channels."""
    model = _tiny_otv_model()
    H, W = 24, 60
    image = np.random.rand(H, W).astype(np.float32)
    pred = run_predict(model, image, torch.device("cpu"), in_channels=3)
    assert pred.shape == (H, W)


def test_run_predict_output_in_unit_interval():
    model = _tiny_atv_model()
    image = np.random.rand(40, 60).astype(np.float32)
    pred = run_predict(model, image, torch.device("cpu"), in_channels=1)
    assert pred.min() >= 0.0 and pred.max() <= 1.0


# ---------------------------------------------------------------------------
# compute_dice
# ---------------------------------------------------------------------------

from deeplogger.gui.inferencer import compute_dice


def test_dice_perfect():
    pred = np.ones((10, 10), dtype=np.float32)
    gt = np.ones((10, 10), dtype=np.uint8)
    assert compute_dice(pred, gt, threshold=0.5) == pytest.approx(1.0)


def test_dice_no_overlap():
    pred = np.zeros((10, 10), dtype=np.float32)
    pred[:5] = 1.0
    gt = np.zeros((10, 10), dtype=np.uint8)
    gt[5:] = 1
    assert compute_dice(pred, gt, threshold=0.5) == pytest.approx(0.0)


def test_dice_empty_both():
    pred = np.zeros((10, 10), dtype=np.float32)
    gt = np.zeros((10, 10), dtype=np.uint8)
    assert compute_dice(pred, gt, threshold=0.5) == pytest.approx(1.0)


def test_dice_threshold_applied():
    pred = np.full((10, 10), 0.4, dtype=np.float32)
    gt = np.ones((10, 10), dtype=np.uint8)
    # pred < threshold=0.5 → all zeros → no overlap
    assert compute_dice(pred, gt, threshold=0.5) == pytest.approx(0.0)
