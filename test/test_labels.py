"""Tests for deeplogger.labels."""

import numpy as np
import pandas as pd

from deeplogger.config import Fracture
from deeplogger.labels import get_label, apply_label, crop_depth, filter_labels_from_range


def test_get_label_returns_binary():
    """get_label should return an image with only 0s and 1s."""
    depth_vector = np.linspace(0, 1.0, 360)
    frac = Fracture(azimuth=90.0, dip=45.0, depth=0.5, aperture=10.0)
    label = get_label(frac, depth_vector, bh_diameter=0.076)
    assert set(np.unique(label)).issubset({0, 1})


def test_get_label_shape():
    """Output shape should match (len(depth_vector), azimuth_values)."""
    depth_vector = np.linspace(0, 1.0, 200)
    frac = Fracture(azimuth=0, dip=30.0, depth=0.5, aperture=5.0)
    label = get_label(frac, depth_vector, bh_diameter=0.076, azimuth_values=180)
    assert label.shape == (200, 180)


def test_get_label_has_nonzero_pixels():
    """A fracture within the depth range should produce non-zero pixels."""
    depth_vector = np.linspace(0, 2.0, 500)
    frac = Fracture(azimuth=45.0, dip=60.0, depth=1.0, aperture=20.0)
    label = get_label(frac, depth_vector, bh_diameter=0.076)
    assert np.sum(label) > 0


def test_apply_label_skips_out_of_range():
    """apply_label should return unchanged image if fracture is outside depth range."""
    depth_vector = np.linspace(0, 1.0, 100)
    image = np.zeros((100, 360))
    frac = Fracture(azimuth=0, dip=30.0, depth=5.0, aperture=10.0)  # depth=5 >> max(depth)=1
    result = apply_label(image, frac, depth_vector, bh_diameter=0.076)
    assert np.sum(result) == 0


def test_crop_depth_clips_values():
    """crop_depth should clip Z values to depth vector bounds."""
    depth = np.linspace(0, 10, 100)
    Z_up = np.array([11.0, 5.0, -1.0])
    Z_down = np.array([5.0, -2.0, 3.0])
    Z_up_c, Z_down_c = crop_depth(Z_up, Z_down, depth)
    assert Z_up_c[0] <= 10.0
    assert Z_down_c[1] >= 0.0


def test_filter_labels_from_range():
    """filter_labels_from_range should filter by column range."""
    df = pd.DataFrame({'depth': [1, 5, 10, 15], 'aperture': [2, 4, 6, 8]})
    result = filter_labels_from_range(df, 'depth', [4, 11])
    assert len(result) == 2
    assert list(result['depth']) == [5, 10]


def test_filter_labels_invalid_column():
    """filter_labels_from_range should return empty df for invalid column."""
    df = pd.DataFrame({'depth': [1, 5]})
    result = filter_labels_from_range(df, 'nonexistent', [0, 10])
    assert len(result) == 0
