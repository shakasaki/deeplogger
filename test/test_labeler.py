"""Tests for the pure helpers in deeplogger.gui.labeler.

The napari/magicgui shell (``launch_labeler``) needs a display and is verified
by running the GUI; here we test the logic that backs it: mask rasterization
and the ``[image, mask]`` save bundle.
"""

from pathlib import Path

import numpy as np

from deeplogger.config import DataType, Fracture
from deeplogger.gui.labeler import (
    MAX_LABEL_ROWS,
    PICKS_COLUMNS,
    append_picks_csv,
    build_label_mask,
    default_label_filename,
    gesture_to_fracture,
    save_label_bundle,
    sinusoid_curve,
)
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


# --- interactive pick gesture ----------------------------------------------


def test_gesture_depth_from_click_row():
    """The click row should set the fracture's centre depth."""
    depth = np.linspace(100.0, 110.0, 1000)
    fr = gesture_to_fracture(500, 0.0, 0.0, depth, 360, diameter=0.1, aperture=5.0)
    assert fr.depth == depth[500]
    assert fr.aperture == 5.0


def test_gesture_horizontal_drag_sets_azimuth():
    """A horizontal drag of a quarter image width is ~90° azimuth."""
    depth = np.linspace(100.0, 110.0, 1000)
    fr = gesture_to_fracture(500, 90.0, 0.0, depth, n_azimuth=360, diameter=0.1, aperture=5.0)
    assert abs(fr.azimuth - 90.0) < 1e-6


def test_gesture_vertical_drag_increases_dip():
    """A larger vertical drag should yield a steeper dip; zero drag ~0°."""
    depth = np.linspace(100.0, 110.0, 1000)  # 0.01 m/row
    flat = gesture_to_fracture(500, 0.0, 0.0, depth, 360, diameter=0.1, aperture=5.0)
    steep = gesture_to_fracture(500, 0.0, 40.0, depth, 360, diameter=0.1, aperture=5.0)
    assert flat.dip == 0.0
    assert steep.dip > flat.dip
    assert 0.0 <= steep.dip <= 89.0


def test_sinusoid_curve_shape_and_columns():
    """The traced curve should give one (row, col) per azimuth, cols 0..N-1."""
    depth = np.linspace(100.0, 101.0, 360)
    fr = Fracture(azimuth=30.0, dip=45.0, depth=100.5, aperture=5.0)
    pts = sinusoid_curve(fr, depth, n_azimuth=360, diameter=0.1)
    assert pts.shape == (360, 2)
    np.testing.assert_array_equal(pts[:, 1], np.arange(360))


def test_default_label_filename():
    """Filename should embed the borehole and rounded depth bounds."""
    depth = np.linspace(100.123, 101.789, 200)
    assert default_label_filename("MB7", depth) == "MB7_100.12m_101.79m.pt"


def test_append_picks_csv_creates_and_appends(tmp_path):
    """First write adds a header; later writes append without a new header."""
    path = tmp_path / "MB7_picks.csv"
    f1 = Fracture(azimuth=10.0, dip=20.0, depth=100.5, aperture=5.0)
    f2 = Fracture(azimuth=200.0, dip=55.0, depth=101.0, aperture=5.0)
    append_picks_csv(path, "MB7", [f1])
    append_picks_csv(path, "MB7", [f2])
    rows = path.read_text().strip().splitlines()
    assert rows[0] == ",".join(PICKS_COLUMNS)
    assert len(rows) == 3  # header + 2 picks
    assert rows[1].startswith("MB7,100.5000,20.00,10.00")
