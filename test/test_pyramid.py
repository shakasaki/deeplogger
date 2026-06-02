"""Tests for deeplogger.pyramid."""

from pathlib import Path

import numpy as np
import pytest
import zarr

from deeplogger.config import DataType
from deeplogger.pyramid import (
    build_depth_pyramid,
    write_zarr_pyramid,
    read_zarr_pyramid,
    select_pyramid_level,
    level_window,
)


def test_pyramid_atv_level_shapes():
    """ATV pyramid should halve depth rows per level, keeping azimuth full."""
    img = np.zeros((8000, 360), dtype=np.float32)
    levels = build_depth_pyramid(img)  # min_rows=2048
    assert [lvl.shape for lvl in levels] == [(8000, 360), (4000, 360), (2000, 360)]


def test_pyramid_single_level_when_small():
    """An image already at/under min_rows yields only the original level."""
    img = np.zeros((100, 360), dtype=np.float32)
    levels = build_depth_pyramid(img)
    assert len(levels) == 1
    assert levels[0] is img


def test_pyramid_otv_shapes_and_dtype():
    """OTV pyramid should keep the RGB channels and uint8 dtype."""
    img = np.zeros((8000, 720, 3), dtype=np.uint8)
    levels = build_depth_pyramid(img)
    assert levels[1].shape == (4000, 720, 3)
    assert all(lvl.dtype == np.uint8 for lvl in levels)


def test_pyramid_dtype_preserved_atv():
    """ATV float32 input should stay float32."""
    img = np.zeros((4000, 16), dtype=np.float32)
    levels = build_depth_pyramid(img, min_rows=1000)
    assert all(lvl.dtype == np.float32 for lvl in levels)


def test_pyramid_block_mean_values():
    """A downsample level should be the block mean of consecutive rows."""
    img = np.array([[0.0, 0.0], [2.0, 2.0], [4.0, 4.0], [6.0, 6.0]], dtype=np.float32)
    levels = build_depth_pyramid(img, min_rows=2)  # one downsample step
    assert len(levels) == 2
    np.testing.assert_allclose(levels[1], [[1.0, 1.0], [5.0, 5.0]])


def test_pyramid_nan_aware_mean():
    """NaN samples should be ignored within a block (not propagate)."""
    img = np.array([[1.0], [np.nan], [3.0], [5.0]], dtype=np.float32)
    levels = build_depth_pyramid(img, min_rows=2)
    np.testing.assert_allclose(levels[1], [[1.0], [4.0]])


def test_pyramid_trims_odd_rows():
    """Rows that do not fill a block are trimmed from the bottom."""
    img = np.ones((5, 2), dtype=np.float32)
    levels = build_depth_pyramid(img, min_rows=2)
    assert levels[1].shape == (2, 2)  # 5 -> 4 -> 2


@pytest.mark.parametrize("bad_factor", [1, 0, -2])
def test_pyramid_invalid_factor(bad_factor):
    """factor < 2 should raise ValueError."""
    with pytest.raises(ValueError, match="factor"):
        build_depth_pyramid(np.zeros((10, 4), dtype=np.float32), factor=bad_factor)


def test_pyramid_invalid_ndim():
    """A 1-D input is neither ATV nor OTV and should raise ValueError."""
    with pytest.raises(ValueError, match="2-D"):
        build_depth_pyramid(np.zeros((10,), dtype=np.float32))


# --- zarr persistence -------------------------------------------------------


def test_zarr_roundtrip_atv(tmp_path: Path):
    """An ATV pyramid should round-trip through zarr with values and attrs."""
    rng = np.random.default_rng(0)
    img = rng.random((4000, 16), dtype=np.float32)
    levels = build_depth_pyramid(img, min_rows=1000)
    depth = np.linspace(0.0, 40.0, 4000)
    store = write_zarr_pyramid(
        tmp_path / "atv.zarr", levels, depth,
        data_type=DataType.ATV, n_azimuth=16, diameter=0.076,
        start_depth=0.0, stop_depth=40.0,
    )
    out_levels, out_depth, attrs = read_zarr_pyramid(store)
    assert len(out_levels) == len(levels)
    for src, got in zip(levels, out_levels):
        np.testing.assert_array_equal(src, got[:])
    np.testing.assert_allclose(out_depth, depth)
    assert attrs["data_type"] is DataType.ATV
    assert attrs["n_azimuth"] == 16
    assert attrs["diameter"] == 0.076
    assert attrs["n_levels"] == len(levels)


def test_zarr_roundtrip_otv(tmp_path: Path):
    """An OTV pyramid should round-trip with uint8 RGB preserved."""
    rng = np.random.default_rng(1)
    img = rng.integers(0, 256, size=(2000, 8, 3), dtype=np.uint8)
    levels = build_depth_pyramid(img, min_rows=500)
    depth = np.linspace(0.0, 20.0, 2000)
    store = write_zarr_pyramid(
        tmp_path / "otv.zarr", levels, depth,
        data_type=DataType.OTV, n_azimuth=8, diameter=None,
        start_depth=0.0, stop_depth=20.0,
    )
    out_levels, _depth, attrs = read_zarr_pyramid(store)
    assert attrs["data_type"] is DataType.OTV
    assert out_levels[0].dtype == np.uint8
    np.testing.assert_array_equal(levels[1], out_levels[1][:])


def test_zarr_levels_are_lazy(tmp_path: Path):
    """Read levels should be lazy zarr arrays, not eager numpy arrays."""
    img = np.zeros((1000, 4), dtype=np.float32)
    levels = build_depth_pyramid(img, min_rows=200)
    store = write_zarr_pyramid(
        tmp_path / "lazy.zarr", levels, np.arange(1000.0),
        data_type=DataType.ATV, n_azimuth=4, diameter=None,
        start_depth=0.0, stop_depth=1.0,
    )
    out_levels, _d, _a = read_zarr_pyramid(store)
    assert all(isinstance(lvl, zarr.Array) for lvl in out_levels)


def test_zarr_chunk_shape(tmp_path: Path):
    """Level arrays should be chunked along depth by chunk_rows."""
    img = np.zeros((1000, 4), dtype=np.float32)
    levels = build_depth_pyramid(img, min_rows=1000)  # single level
    store = write_zarr_pyramid(
        tmp_path / "chunk.zarr", levels, np.arange(1000.0),
        data_type=DataType.ATV, n_azimuth=4, diameter=None,
        start_depth=0.0, stop_depth=1.0, chunk_rows=256,
    )
    out_levels, _d, _a = read_zarr_pyramid(store)
    assert out_levels[0].chunks == (256, 4)


def test_zarr_diameter_none_survives(tmp_path: Path):
    """A None diameter should round-trip as None (JSON null)."""
    img = np.zeros((100, 4), dtype=np.float32)
    levels = build_depth_pyramid(img)
    store = write_zarr_pyramid(
        tmp_path / "none.zarr", levels, np.arange(100.0),
        data_type=DataType.ATV, n_azimuth=4, diameter=None,
        start_depth=0.0, stop_depth=1.0,
    )
    _l, _d, attrs = read_zarr_pyramid(store)
    assert attrs["diameter"] is None


# --- pyramid navigation -----------------------------------------------------


@pytest.mark.parametrize(
    "visible_rows, expected",
    [
        (1000, 0),    # decimation 1 -> level 0
        (3000, 1),    # decimation 3 -> floor(log2 3) = 1
        (4000, 2),    # decimation 4 -> level 2
        (500, 0),     # zoomed in past full res -> level 0
    ],
)
def test_select_pyramid_level(visible_rows, expected):
    """Level selection should track the decimation between rows and pixels."""
    assert select_pyramid_level(visible_rows, 1000, n_levels=5) == expected


def test_select_pyramid_level_clamped_to_available():
    """A huge decimation should clamp to the coarsest available level."""
    assert select_pyramid_level(100_000, 1000, n_levels=3) == 2


def test_select_pyramid_level_oversample_favours_finer():
    """Higher oversample should pick a finer (lower) level."""
    assert select_pyramid_level(4000, 1000, n_levels=5, oversample=2.0) == 1


@pytest.mark.parametrize("bad", [dict(n_levels=0), dict(viewport_px=0), dict(oversample=0)])
def test_select_pyramid_level_invalid(bad):
    """Invalid arguments should raise ValueError."""
    kwargs = dict(visible_rows=1000, viewport_px=1000, n_levels=5)
    kwargs.update(bad)
    with pytest.raises(ValueError):
        select_pyramid_level(**kwargs)


def test_level_window_level_zero_identity():
    """Level 0 should return the input range unchanged."""
    assert level_window(10, 1000, level=0) == (10, 1000)


def test_level_window_floors_start_ceils_stop():
    """Start floors and stop ceils so the window fully covers the input."""
    assert level_window(10, 1000, level=1) == (5, 500)
    assert level_window(1, 1001, level=1) == (0, 501)
    assert level_window(10, 20, level=2) == (2, 5)


@pytest.mark.parametrize("args", [(5, 5, 0), (-1, 10, 0), (0, 10, -1)])
def test_level_window_invalid(args):
    """Invalid row ranges or levels should raise ValueError."""
    with pytest.raises(ValueError):
        level_window(*args)
