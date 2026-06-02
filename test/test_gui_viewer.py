"""Headless smoke tests for the pyqtgraph browse viewer.

These run with the Qt 'offscreen' platform so no display is required. They check
that the widget constructs and refreshes; the interactive feel must be verified
by actually running the viewer.
"""

import os

# Must be set before any QApplication is created.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

pg = pytest.importorskip("pyqtgraph")

from deeplogger.config import DataType
from deeplogger.gui.viewer import LogViewer
from deeplogger.pyramid import (
    build_depth_pyramid,
    read_zarr_pyramid,
    write_zarr_pyramid,
)


@pytest.fixture(scope="module")
def qapp():
    return pg.mkQApp("deeplogger-test")


def _make_zarr(tmp_path, data_type):
    rng = np.random.default_rng(0)
    if data_type is DataType.ATV:
        img = rng.random((6000, 16), dtype=np.float32)
        n_az = 16
    else:
        img = rng.integers(0, 256, size=(6000, 8, 3), dtype=np.uint8)
        n_az = 8
    levels = build_depth_pyramid(img, min_rows=1000)
    depth = np.linspace(0.0, 60.0, 6000)
    return write_zarr_pyramid(
        tmp_path / "v.zarr", levels, depth,
        data_type=data_type, n_azimuth=n_az, diameter=None,
        start_depth=0.0, stop_depth=60.0,
    )


def test_viewer_constructs_atv(qapp, tmp_path):
    """An ATV viewer should construct, build a multi-level pyramid, show data."""
    store = _make_zarr(tmp_path, DataType.ATV)
    viewer = LogViewer.from_zarr(store)
    viewer.resize(400, 800)
    viewer._refresh()
    assert viewer.n_levels >= 2
    assert viewer.data_type is DataType.ATV
    assert viewer._img.image is not None


def test_viewer_constructs_otv(qapp, tmp_path):
    """An OTV viewer should construct and display an RGB window."""
    store = _make_zarr(tmp_path, DataType.OTV)
    viewer = LogViewer.from_zarr(store)
    assert viewer.data_type is DataType.OTV
    assert viewer._img.image is not None
    assert viewer._img.image.ndim == 3  # RGB


def test_viewer_x_axis_locked(qapp, tmp_path):
    """Azimuth (X) should be locked; only depth (Y) is interactive."""
    store = _make_zarr(tmp_path, DataType.ATV)
    viewer = LogViewer.from_zarr(store)
    x_enabled, y_enabled = viewer._plot.getViewBox().state["mouseEnabled"]
    assert x_enabled is False
    assert y_enabled is True


def test_viewer_svd_control_reduces_stripes(qapp, tmp_path):
    """Setting the SVD control should reprocess the log and cut stripe energy."""
    rng = np.random.default_rng(2)
    stripes = np.tile(rng.standard_normal(16) * 10.0, (4000, 1))
    img = (stripes + rng.standard_normal((4000, 16))).astype(np.float32)
    levels = build_depth_pyramid(img, min_rows=1000)
    depth = np.linspace(0.0, 40.0, 4000)
    store = write_zarr_pyramid(
        tmp_path / "striped.zarr", levels, depth,
        data_type=DataType.ATV, n_azimuth=16, diameter=None,
        start_depth=0.0, stop_depth=40.0,
    )
    viewer = LogViewer.from_zarr(store)
    before = float(np.var(np.asarray(viewer._levels[0][:]).mean(axis=0)))
    viewer._apply_svd(1)
    after = float(np.var(np.asarray(viewer._levels[0]).mean(axis=0)))
    assert after < before * 0.2


def test_save_processed_persists_destriped_data(qapp, tmp_path):
    """save_processed should write the destriped pyramid (not the raw log) plus
    the SVD provenance and carried-through diameter."""
    rng = np.random.default_rng(3)
    stripes = np.tile(rng.standard_normal(16) * 10.0, (4000, 1))
    img = (stripes + rng.standard_normal((4000, 16))).astype(np.float32)
    levels = build_depth_pyramid(img, min_rows=1000)
    depth = np.linspace(0.0, 40.0, 4000)
    store = write_zarr_pyramid(
        tmp_path / "striped.zarr", levels, depth,
        data_type=DataType.ATV, n_azimuth=16, diameter=0.076,
        start_depth=0.0, stop_depth=40.0,
    )
    viewer = LogViewer.from_zarr(store)
    viewer._apply_svd(2)
    out = viewer.save_processed(tmp_path / "processed.zarr")

    out_levels, out_depth, attrs = read_zarr_pyramid(out)
    assert attrs["svd_removed"] == 2  # provenance recorded
    assert attrs["diameter"] == 0.076  # metadata carried through
    saved0 = np.asarray(out_levels[0][:])
    # Saved level 0 matches the in-memory processed data, not the raw striped log.
    np.testing.assert_array_equal(saved0, np.asarray(viewer._levels[0]))
    assert not np.allclose(saved0, img)


def test_save_processed_no_svd_omits_provenance(qapp, tmp_path):
    """Saving without destriping should not write an svd_removed attr."""
    store = _make_zarr(tmp_path, DataType.ATV)
    viewer = LogViewer.from_zarr(store)
    out = viewer.save_processed(tmp_path / "copy.zarr")
    _, _, attrs = read_zarr_pyramid(out)
    assert "svd_removed" not in attrs


def test_viewer_zoom_y_loads_finer_level(qapp, tmp_path):
    """Zooming the depth axis to a small window should load a finer level than
    the full-extent overview (the integration path: range change -> refresh)."""
    store = _make_zarr(tmp_path, DataType.ATV)
    viewer = LogViewer.from_zarr(store)
    viewer.resize(400, 800)
    viewer._plot.setYRange(viewer._start, viewer._stop, padding=0)
    overview_level = viewer._current_level
    # Zoom to a thin depth slice; sigRangeChanged -> _refresh runs.
    viewer._plot.setYRange(20.0, 21.0, padding=0)
    assert viewer._current_level < overview_level
