"""Tests for deeplogger.las_reader."""

import textwrap
from pathlib import Path

import numpy as np
import pytest

from deeplogger.config import DataType
from deeplogger.las_reader import (
    BoreholeLog,
    read_las_header,
    read_las_data,
    _parse_mnemonic_line,
)


# A minimal but format-faithful LAS 3.0 header followed by a few data rows.
# Built as fixtures so the suite stays fast and portable (the real logs are
# 90 MB-2.5 GB and are not committed to the repo).
_HEADER = textwrap.dedent("""\
    #------------------------------------------------------------
    ~VERSION INFORMATION
    VERS.               3.0  :CWLS LOG ASCII STANDARD - VERSION 3.0
    WRAP.                NO  :ONE LINE PER DEPTH STEP
    DLM.              COMMA  :DELIMITING CHARACTER
    #------------------------------------------------------------
    ~WELL INFORMATION
    STRT.M          1.57816  :FIRST INDEX VALUE
    STOP.M          100.382  :LAST INDEX VALUE
    STEP.M          0.00000  :STEP
    NULL.            -99999  :NULL VALUE
    ~LOG_DEFINITION
    DEPTH.M                  :Depth      {F}
    ~LOG_DATA | LOG_DEFINITION
""")

_ATV_ROWS = "   1.57816,   7502.00,   8124.00,   8439.00\n   1.58234,   7231.00,   7562.00,   7809.00\n"
_OTV_ROWS = "   1.93667,131.120.46,135.117.46,140.124.39\n   1.94100,130.119.45,134.116.45,139.123.38\n"


@pytest.fixture
def atv_las(tmp_path: Path) -> Path:
    """A small synthetic ATV LAS file (3 azimuth columns)."""
    p = tmp_path / "synthetic_atv.las"
    p.write_text(_HEADER + _ATV_ROWS, encoding="utf-8")
    return p


@pytest.fixture
def otv_las(tmp_path: Path) -> Path:
    """A small synthetic OTV LAS file (3 RGB azimuth columns)."""
    p = tmp_path / "synthetic_otv.las"
    p.write_text(_HEADER + _OTV_ROWS, encoding="utf-8")
    return p


def test_parse_mnemonic_line_with_unit():
    """A mnemonic with a unit token should drop the unit and keep the value."""
    assert _parse_mnemonic_line("STRT.M          1.57816  :FIRST INDEX VALUE") == ("STRT", "1.57816")


def test_parse_mnemonic_line_without_unit():
    """A mnemonic with no unit token (e.g. DLM) should return its bare value."""
    assert _parse_mnemonic_line("DLM.              COMMA  :DELIMITER") == ("DLM", "COMMA")


def test_parse_mnemonic_line_non_mnemonic():
    """A line without a dot before the colon is not a mnemonic definition."""
    assert _parse_mnemonic_line("~WELL INFORMATION") is None


def test_read_las_header_atv_fields(atv_las: Path):
    """ATV header should parse depth bounds, null, delimiter, and version."""
    h = read_las_header(atv_las)
    assert h.start_depth == pytest.approx(1.57816)
    assert h.stop_depth == pytest.approx(100.382)
    assert h.step == 0.0
    assert h.null_value == pytest.approx(-99999.0)
    assert h.delimiter == ","
    assert h.version == "3.0"


def test_read_las_header_atv_type_and_azimuth(atv_las: Path):
    """ATV data rows (single-dot floats) should be detected as ATV."""
    h = read_las_header(atv_las)
    assert h.data_type is DataType.ATV
    assert h.n_azimuth == 3


def test_read_las_header_otv_type_and_azimuth(otv_las: Path):
    """OTV data rows (R.G.B cells with two dots) should be detected as OTV."""
    h = read_las_header(otv_las)
    assert h.data_type is DataType.OTV
    assert h.n_azimuth == 3


def test_read_las_header_first_data_line(atv_las: Path):
    """first_data_line should equal the 1-based line number of the marker."""
    h = read_las_header(atv_las)
    # The ~LOG_DATA marker is the 14th line of the fixture content.
    assert h.first_data_line == 14


def test_read_las_header_missing_file(tmp_path: Path):
    """A nonexistent path should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        read_las_header(tmp_path / "does_not_exist.las")


def test_read_las_header_no_marker(tmp_path: Path):
    """A file without a ~LOG_DATA marker should raise ValueError."""
    p = tmp_path / "no_marker.las"
    p.write_text("~WELL INFORMATION\nSTRT.M  1.0  :x\nSTOP.M  2.0  :y\n", encoding="utf-8")
    with pytest.raises(ValueError, match="LOG_DATA"):
        read_las_header(p)


def test_read_las_header_no_data_after_marker(tmp_path: Path):
    """A marker with no following data row should raise ValueError."""
    p = tmp_path / "empty_data.las"
    p.write_text(_HEADER, encoding="utf-8")  # header ends right at the marker
    with pytest.raises(ValueError, match="No data rows"):
        read_las_header(p)


def test_read_las_header_missing_strt(tmp_path: Path):
    """A header missing STRT/STOP should raise ValueError."""
    p = tmp_path / "no_strt.las"
    p.write_text("~LOG_DATA | LOG_DEFINITION\n   1.0,   2.0,   3.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="STRT"):
        read_las_header(p)


# Optional integration check against a real log, skipped when data is absent.
_REAL_ATV = Path("data/Bedretto_Input_HS/MB7/MB7_ATV_unknown.las")


@pytest.mark.skipif(not _REAL_ATV.exists(), reason="real LAS data not present")
def test_read_las_header_real_atv():
    """A real ATV log should report 360 azimuth columns and type ATV."""
    h = read_las_header(_REAL_ATV)
    assert h.data_type is DataType.ATV
    assert h.n_azimuth == 360
    assert h.start_depth == pytest.approx(1.57816, abs=1e-3)


# --- read_las_data ----------------------------------------------------------


def test_read_las_data_atv_shape_and_values(atv_las: Path):
    """ATV data should be (N, n_az) float32 with depth as float64."""
    depth, data = read_las_data(atv_las)
    assert depth.dtype == np.float64
    assert depth == pytest.approx([1.57816, 1.58234])
    assert data.shape == (2, 3)
    assert data.dtype == np.float32
    assert data[0].tolist() == pytest.approx([7502.0, 8124.0, 8439.0])


def test_read_las_data_atv_null_to_nan(tmp_path: Path):
    """ATV cells equal to the null sentinel should become NaN."""
    p = tmp_path / "null_atv.las"
    rows = "   1.57816,   7502.00,   -99999,   8439.00\n"
    p.write_text(_HEADER + rows, encoding="utf-8")
    _depth, data = read_las_data(p)
    assert np.isnan(data[0, 1])
    assert data[0, 0] == pytest.approx(7502.0)


def test_read_las_data_otv_shape_and_values(otv_las: Path):
    """OTV data should be (N, n_az, 3) uint8 with correct RGB triplets."""
    depth, data = read_las_data(otv_las)
    assert depth == pytest.approx([1.93667, 1.94100])
    assert data.shape == (2, 3, 3)
    assert data.dtype == np.uint8
    assert data[0, 0].tolist() == [131, 120, 46]
    assert data[0, 2].tolist() == [140, 124, 39]


def test_read_las_data_row_slice(atv_las: Path):
    """row_slice should return only the requested contiguous rows."""
    depth, data = read_las_data(atv_las, row_slice=(1, 2))
    assert depth == pytest.approx([1.58234])
    assert data.shape == (1, 3)
    assert data[0].tolist() == pytest.approx([7231.0, 7562.0, 7809.0])


def test_read_las_data_slice_past_end_is_empty(atv_las: Path):
    """A slice beyond the available rows should yield empty arrays."""
    depth, data = read_las_data(atv_las, row_slice=(5, 6))
    assert depth.shape == (0,)
    assert data.shape == (0, 3)


@pytest.mark.parametrize("bad", [(2, 1), (-1, 3), (1, 1)])
def test_read_las_data_invalid_slice(atv_las: Path, bad):
    """Invalid row slices should raise ValueError."""
    with pytest.raises(ValueError, match="row_slice"):
        read_las_data(atv_las, row_slice=bad)


def test_read_las_data_reuses_header(atv_las: Path):
    """Passing a precomputed header should give the same result."""
    h = read_las_header(atv_las)
    depth_a, data_a = read_las_data(atv_las)
    depth_b, data_b = read_las_data(atv_las, header=h)
    assert np.array_equal(depth_a, depth_b)
    assert np.array_equal(data_a, data_b, equal_nan=True)


@pytest.mark.skipif(not _REAL_ATV.exists(), reason="real LAS data not present")
def test_read_las_data_real_atv_window():
    """A windowed read of a real ATV log should be (360, 360) float32."""
    h = read_las_header(_REAL_ATV)
    depth, data = read_las_data(_REAL_ATV, header=h, row_slice=(0, 360))
    assert depth.shape == (360,)
    assert data.shape == (360, 360)
    assert data.dtype == np.float32


# --- BoreholeLog ------------------------------------------------------------


def test_borehole_log_open_passthroughs(atv_las: Path):
    """open() should parse the header and expose its fields."""
    log = BoreholeLog.open(atv_las)
    assert log.data_type is DataType.ATV
    assert log.n_azimuth == 3
    assert log.start_depth == pytest.approx(1.57816)
    assert log.stop_depth == pytest.approx(100.382)
    assert log.diameter is None


def test_borehole_log_stores_diameter(atv_las: Path):
    """A diameter passed to open() should be stored."""
    log = BoreholeLog.open(atv_las, diameter=0.076)
    assert log.diameter == 0.076


def test_borehole_log_read_matches_function(atv_las: Path):
    """read() should match the standalone read_las_data."""
    log = BoreholeLog.open(atv_las)
    d_log, data_log = log.read()
    d_fn, data_fn = read_las_data(atv_las)
    assert np.array_equal(d_log, d_fn)
    assert np.array_equal(data_log, data_fn)


def test_borehole_log_read_slice(atv_las: Path):
    """read(row_slice) should forward the slice."""
    log = BoreholeLog.open(atv_las)
    depth, data = log.read(row_slice=(1, 2))
    assert data.shape == (1, 3)
    assert depth == pytest.approx([1.58234])


def test_borehole_log_depth_vector_cached(atv_las: Path):
    """depth_vector should match the data and be cached across calls."""
    log = BoreholeLog.open(atv_las)
    dv = log.depth_vector
    assert dv == pytest.approx([1.57816, 1.58234])
    assert log.depth_vector is dv  # same cached object


def test_borehole_log_n_rows(atv_las: Path):
    """n_rows should equal the number of data rows."""
    assert BoreholeLog.open(atv_las).n_rows == 2


def test_borehole_log_depth_to_row(atv_las: Path):
    """depth_to_row should return the nearest row, clamped to range."""
    log = BoreholeLog.open(atv_las)  # depths [1.57816, 1.58234]
    assert log.depth_to_row(1.5824) == 1
    assert log.depth_to_row(1.5782) == 0
    assert log.depth_to_row(0.0) == 0       # below range -> first
    assert log.depth_to_row(99.0) == 1      # above range -> last


def test_borehole_log_row_to_depth(atv_las: Path):
    """row_to_depth should return the depth at a row, and raise out of range."""
    log = BoreholeLog.open(atv_las)
    assert log.row_to_depth(0) == pytest.approx(1.57816)
    assert log.row_to_depth(1) == pytest.approx(1.58234)
    with pytest.raises(IndexError):
        log.row_to_depth(2)


@pytest.mark.skipif(not _REAL_ATV.exists(), reason="real LAS data not present")
def test_borehole_log_real_atv():
    """A real ATV log should expose many rows and round-trip a depth mapping."""
    log = BoreholeLog.open(_REAL_ATV, diameter=0.076)
    assert log.n_rows > 1000
    row = log.depth_to_row(50.0)
    assert abs(log.row_to_depth(row) - 50.0) < 1.0


def test_borehole_log_to_zarr_roundtrip(atv_las: Path, tmp_path: Path):
    """to_zarr should write a pyramid whose level 0 matches a full read."""
    from deeplogger.pyramid import read_zarr_pyramid

    log = BoreholeLog.open(atv_las, diameter=0.076)
    depth_full, data_full = log.read()
    store = log.to_zarr(tmp_path / "cache")
    assert store == tmp_path / "cache" / (atv_las.stem + ".zarr")
    levels, depth, attrs = read_zarr_pyramid(store)
    np.testing.assert_array_equal(levels[0][:], data_full)
    np.testing.assert_allclose(depth, depth_full)
    assert attrs["data_type"] is DataType.ATV
    assert attrs["n_azimuth"] == 3
    assert attrs["diameter"] == 0.076


def test_borehole_log_to_zarr_multilevel(atv_las: Path, tmp_path: Path):
    """to_zarr with a small min_rows should produce multiple pyramid levels."""
    from deeplogger.pyramid import read_zarr_pyramid

    log = BoreholeLog.open(atv_las)
    store = log.to_zarr(tmp_path / "cache", min_rows=1)
    levels, _depth, attrs = read_zarr_pyramid(store)
    assert attrs["n_levels"] == 2
    assert levels[0].shape == (2, 3)
    assert levels[1].shape == (1, 3)
