"""Tests for the pure helpers in deeplogger.gui.labeler.

The napari/magicgui shell (``launch_labeler``) needs a display and is verified
by running the GUI; here we test the logic that backs it: mask rasterization
and the ``[image, mask]`` save bundle.
"""

from pathlib import Path

import numpy as np

from deeplogger.config import DataType, Fracture
from deeplogger.gui.labeler import MAX_LABEL_ROWS, build_label_mask, save_label_bundle
from deeplogger.labels import get_label


def _depth():
    return np.linspace(100.0, 101.0, 360)


def test_build_label_mask_matches_get_label():
    """A single fracture's mask should equal get_label's rasterization."""
    depth = _depth()
    fr = Fracture(azimuth=45.0, dip=30.0, depth=100.5, aperture=5.0)
    mask = build_label_mask([fr], depth, n_azimuth=360, diameter=0.1)
    expected = get_label(fr, depth, 0.1, 360)
    np.testing.assert_array_equal(mask, expected.astype(np.uint8))
    assert mask.dtype == np.uint8


def test_build_label_mask_unions_fractures():
    """Multiple fractures should OR together (union of bands)."""
    depth = _depth()
    f1 = Fracture(azimuth=0.0, dip=20.0, depth=100.3, aperture=4.0)
    f2 = Fracture(azimuth=180.0, dip=40.0, depth=100.7, aperture=4.0)
    union = build_label_mask([f1, f2], depth, 360, 0.1)
    m1 = build_label_mask([f1], depth, 360, 0.1)
    m2 = build_label_mask([f2], depth, 360, 0.1)
    np.testing.assert_array_equal(union, (m1 | m2))
    assert union.sum() >= max(m1.sum(), m2.sum())


def test_build_label_mask_empty():
    """No fractures yields an all-zero mask of the right shape."""
    depth = _depth()
    mask = build_label_mask([], depth, 360, 0.1)
    assert mask.shape == (360, 360)
    assert mask.sum() == 0


def test_save_label_bundle_atv_roundtrip(tmp_path: Path):
    """An ATV [image, mask] bundle should round-trip through torch.save/load."""
    import torch

    image = np.random.default_rng(0).random((128, 360), dtype=np.float32)
    mask = np.zeros((128, 360), dtype=np.uint8)
    mask[10:20, 30:40] = 1
    out = save_label_bundle(tmp_path / "lab.pt", image, mask)
    loaded = torch.load(out, weights_only=True)
    assert len(loaded) == 2
    np.testing.assert_array_equal(loaded[0].numpy(), image)
    np.testing.assert_array_equal(loaded[1].numpy(), mask)


def test_save_label_bundle_otv_shape(tmp_path: Path):
    """An OTV RGB image should save with its 3-channel shape preserved."""
    import torch

    image = np.random.default_rng(1).integers(0, 256, (64, 360, 3), dtype=np.uint8)
    mask = np.ones((64, 360), dtype=np.uint8)
    out = save_label_bundle(tmp_path / "otv.pt", image, mask)
    loaded = torch.load(out, weights_only=True)
    assert loaded[0].shape == (64, 360, 3)
    assert loaded[1].shape == (64, 360)


def test_max_label_rows_is_positive():
    """The labeling row cap is a sane positive bound."""
    assert MAX_LABEL_ROWS > 0
