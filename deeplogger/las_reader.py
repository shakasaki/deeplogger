"""LAS 3.0 reader for borehole televiewer data (polars-based).

Parses Acoustic (ATV) and Optical (OTV) Televiewer logs in the CWLS LAS 3.0
ASCII format used by the Bedretto/VALTER datasets. This is the modern entry
point for the GUI; the legacy pandas reader in ``importLASv3.py`` is retained
for existing notebooks and preparation scripts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import polars as pl

from deeplogger.config import DataType

# DLM. mnemonic values mapped to the actual field separator used in the data
# section. Bedretto files use COMMA; the others are supported for robustness.
_DELIMITER_MAP = {"COMMA": ",", "SPACE": " ", "TAB": "\t"}


@dataclass
class LasHeader:
    """Structured metadata parsed from a LAS 3.0 header.

    Attributes:
        start_depth: First index (depth) value in meters (STRT).
        stop_depth: Last index (depth) value in meters (STOP).
        step: Depth step in meters (STEP). A value of 0.0 signals irregular
            sampling, in which case the true depth must be read from the data
            column rather than reconstructed from start/step.
        null_value: Sentinel value marking missing measurements (NULL).
        delimiter: Field separator for the data section (decoded from DLM).
        data_type: DataType.ATV for single-channel amplitude logs,
            DataType.OTV for three-channel optical (RGB) logs.
        n_azimuth: Number of azimuth columns (data fields excluding depth).
        first_data_line: Number of leading lines to skip to reach the first
            data row (i.e. the 1-based line number of the ~LOG_DATA marker).
        version: LAS format version string (VERS).
    """

    start_depth: float
    stop_depth: float
    step: float
    null_value: float
    delimiter: str
    data_type: DataType
    n_azimuth: int
    first_data_line: int
    version: str


def _parse_mnemonic_line(line: str) -> tuple[str, str] | None:
    """Extract the mnemonic and its value from a LAS header line.

    A LAS header line has the form ``MNEM.UNIT  VALUE : DESCRIPTION``. Only the
    first dot (separating the mnemonic from its unit) is consumed, so any
    decimal point inside VALUE is preserved.

    Args:
        line: A single raw header line.

    Returns:
        A (mnemonic, value) tuple with surrounding whitespace stripped, or None
        if the line is not a mnemonic definition (no dot before the colon).

    Examples:
        >>> _parse_mnemonic_line("STRT.M          1.57816  :FIRST INDEX VALUE")
        ('STRT', '1.57816')
        >>> _parse_mnemonic_line("DLM.              COMMA  :DELIMITER")
        ('DLM', 'COMMA')
    """
    left = line.split(":", 1)[0]
    if "." not in left:
        return None
    mnem, rest = left.split(".", 1)
    rest = rest.strip()
    parts = rest.split(None, 1)
    if len(parts) == 2:
        value = parts[1].strip()  # parts[0] is the unit token; keep the value
    elif len(parts) == 1:
        value = parts[0]  # no unit token (e.g. "DLM.  COMMA")
    else:
        value = ""
    return mnem.strip(), value


def read_las_header(path: str | Path) -> LasHeader:
    """Parse a LAS 3.0 televiewer header without loading the data body.

    Scans only the header section plus the single first data row, so it is
    cheap even for multi-gigabyte logs. The data type and azimuth count are
    inferred from the first data row (an RGB-packed cell such as "131.120.46"
    contains two dots and marks an OTV log), which is more robust than relying
    on channel names in the ~LOG_DEFINITION block.

    Args:
        path: Path to a ``.las`` file in CWLS LAS 3.0 format.

    Returns:
        A LasHeader with depth bounds, sampling step, null sentinel, delimiter,
        inferred data type, azimuth column count, data-section offset, and
        format version.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the ``~LOG_DATA`` marker is absent, no data row follows
            it, the required STRT/STOP keys are missing, or the first data row
            has no azimuth columns.

    Examples:
        >>> header = read_las_header("data/Bedretto_Input_HS/MB7/MB7_ATV_unknown.las")
        >>> header.data_type, header.n_azimuth
        (<DataType.ATV: 'atv'>, 360)

    See Also:
        read_las_data: Reads the data body these offsets point to.
        importLASv3.get_depth_only: Legacy pandas-based reader.
    """
    path = Path(path)
    keys: dict[str, str] = {}
    marker_line: int | None = None
    first_data_row: str | None = None

    with open(path, encoding="utf-8", errors="ignore") as f:
        for lineno, line in enumerate(f, start=1):
            stripped = line.strip()
            if stripped.startswith("~LOG_DATA"):
                marker_line = lineno
                first_data_row = next(f, None)
                break
            if stripped and not stripped.startswith(("~", "#")):
                parsed = _parse_mnemonic_line(line)
                if parsed is not None:
                    mnem, value = parsed
                    keys[mnem] = value

    if marker_line is None:
        raise ValueError(f"No '~LOG_DATA' marker found in {path}")
    if first_data_row is None or not first_data_row.strip():
        raise ValueError(f"No data rows found after the marker in {path}")
    for required in ("STRT", "STOP"):
        if required not in keys:
            raise ValueError(f"Missing required header key '{required}' in {path}")

    delimiter = _DELIMITER_MAP.get(keys.get("DLM", "COMMA").upper(), ",")
    fields = first_data_row.strip().split(delimiter)
    if len(fields) < 2:
        raise ValueError(f"First data row has no azimuth columns in {path}")
    n_azimuth = len(fields) - 1
    # An RGB-packed OTV cell ("R.G.B") has two dots; an ATV amplitude has <= 1.
    data_type = DataType.OTV if fields[1].strip().count(".") >= 2 else DataType.ATV

    return LasHeader(
        start_depth=float(keys["STRT"]),
        stop_depth=float(keys["STOP"]),
        step=float(keys.get("STEP", 0.0)),
        null_value=float(keys.get("NULL", "nan")),
        delimiter=delimiter,
        data_type=data_type,
        n_azimuth=n_azimuth,
        first_data_line=marker_line,
        version=keys.get("VERS", ""),
    )


def _read_atv_data(df: pl.DataFrame, header: LasHeader) -> np.ndarray:
    """Convert the string columns of an ATV data frame to a float32 image.

    Args:
        df: Data frame whose columns are the raw string fields (depth first).
        header: Parsed header, used for the null sentinel.

    Returns:
        Amplitude array of shape (N, n_azimuth), dtype float32, with cells equal
        to the null sentinel replaced by NaN.
    """
    azimuth_cols = df.columns[1:]
    data = (
        df.select(azimuth_cols)
        .with_columns(pl.all().str.strip_chars().cast(pl.Float32))
        .to_numpy()
        .astype(np.float32, copy=False)
    )
    if not np.isnan(header.null_value):
        data[data == np.float32(header.null_value)] = np.nan
    return data


def _read_otv_data(df: pl.DataFrame, header: LasHeader) -> np.ndarray:
    """Convert the "R.G.B" string columns of an OTV data frame to a uint8 image.

    Args:
        df: Data frame whose columns are the raw string fields (depth first);
            each azimuth cell is a dot-packed RGB triplet such as "131.120.46".
        header: Parsed header, used for the azimuth count.

    Returns:
        RGB array of shape (N, n_azimuth, 3), dtype uint8.
    """
    exprs = []
    for i, col in enumerate(df.columns[1:]):
        rgb = pl.col(col).str.split_exact(".", 2)  # -> struct(field_0, field_1, field_2)
        exprs.append(rgb.struct.field("field_0").cast(pl.UInt8).alias(f"c{i}_r"))
        exprs.append(rgb.struct.field("field_1").cast(pl.UInt8).alias(f"c{i}_g"))
        exprs.append(rgb.struct.field("field_2").cast(pl.UInt8).alias(f"c{i}_b"))
    flat = df.select(exprs).to_numpy().astype(np.uint8, copy=False)
    # Columns are laid out r,g,b per azimuth, in azimuth order -> (N, n_az, 3).
    return flat.reshape(flat.shape[0], header.n_azimuth, 3)


def read_las_data(
    path: str | Path,
    header: LasHeader | None = None,
    row_slice: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Read the data section of a LAS televiewer log into numpy arrays.

    Reads either the whole log (the conversion path) or a contiguous range of
    rows (the lazy, windowed path). The data body is read as strings via polars
    because the whitespace-padded ATV numerics defeat numeric type inference and
    OTV cells must be split on "." regardless.

    Args:
        path: Path to a ``.las`` file in CWLS LAS 3.0 format.
        header: A precomputed LasHeader to avoid re-parsing; read from ``path``
            if None.
        row_slice: Half-open ``(start, stop)`` row indices into the data section
            (0-based). None reads all rows.

    Returns:
        A ``(depth, data)`` tuple:
          - ``depth``: float64 array of shape (N,), the real depth column.
          - ``data``: for ATV, a float32 array of shape (N, n_azimuth) with the
            null sentinel mapped to NaN; for OTV, a uint8 array of shape
            (N, n_azimuth, 3).

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If ``row_slice`` is invalid (start < 0 or stop <= start), or
            a data row's field count does not match ``n_azimuth + 1``.

    Examples:
        >>> header = read_las_header("log.las")
        >>> depth, data = read_las_data("log.las", header, row_slice=(0, 360))
        >>> depth.shape, data.shape
        ((360,), (360, 360))

    See Also:
        read_las_header: Parses the metadata used to locate the data section.
    """
    path = Path(path)
    if header is None:
        header = read_las_header(path)

    if row_slice is not None:
        start, stop = row_slice
        if start < 0 or stop <= start:
            raise ValueError(f"Invalid row_slice {row_slice}: need 0 <= start < stop")
        skip_rows = header.first_data_line + start
        n_rows: int | None = stop - start
    else:
        skip_rows = header.first_data_line
        n_rows = None

    try:
        df = pl.read_csv(
            path,
            has_header=False,
            separator=header.delimiter,
            skip_rows=skip_rows,
            n_rows=n_rows,
            infer_schema_length=0,  # read every column as a string
        )
    except pl.exceptions.NoDataError:
        df = None  # the requested slice falls past the end of the data section
    except pl.exceptions.PolarsError as exc:  # ragged/corrupt data section
        raise ValueError(f"Failed to parse data section of {path}: {exc}") from exc

    if df is None or df.height == 0:
        depth = np.empty((0,), dtype=np.float64)
        if header.data_type is DataType.OTV:
            return depth, np.empty((0, header.n_azimuth, 3), dtype=np.uint8)
        return depth, np.empty((0, header.n_azimuth), dtype=np.float32)

    if df.width != header.n_azimuth + 1:
        raise ValueError(
            f"Expected {header.n_azimuth + 1} columns but found {df.width} in {path}"
        )

    depth = df[df.columns[0]].str.strip_chars().cast(pl.Float64).to_numpy()
    if header.data_type is DataType.OTV:
        return depth, _read_otv_data(df, header)
    return depth, _read_atv_data(df, header)


@dataclass
class BoreholeLog:
    """A single televiewer log, bundling its metadata with data access.

    This is the object the GUI holds per opened log. It is a thin façade over
    the module-level reader functions: it caches the header (and, on first use,
    the depth column) and offers windowed reads plus depth/row coordinate
    mapping, so callers need not thread loose arrays and parameters around.

    The depth vector is assumed to increase monotonically down the borehole,
    which holds for the Bedretto logs.

    Attributes:
        path: Path to the source ``.las`` file.
        header: Parsed LAS header.
        diameter: Borehole diameter in meters. Not stored in the LAS file; it is
            supplied separately and is required for sinusoidal fracture picking.
    """

    path: Path
    header: LasHeader
    diameter: float | None = None
    _depth_vector: np.ndarray | None = field(
        default=None, repr=False, compare=False
    )

    @classmethod
    def open(cls, path: str | Path, diameter: float | None = None) -> "BoreholeLog":
        """Open a log and parse its header.

        Args:
            path: Path to a ``.las`` file in CWLS LAS 3.0 format.
            diameter: Optional borehole diameter in meters (for picking).

        Returns:
            A BoreholeLog with its header parsed; data is read on demand.

        Examples:
            >>> log = BoreholeLog.open("log.las", diameter=0.076)
            >>> log.data_type
            <DataType.ATV: 'atv'>
        """
        path = Path(path)
        return cls(path=path, header=read_las_header(path), diameter=diameter)

    @property
    def data_type(self) -> DataType:
        """The log's data type (ATV or OTV)."""
        return self.header.data_type

    @property
    def n_azimuth(self) -> int:
        """Number of azimuth columns."""
        return self.header.n_azimuth

    @property
    def start_depth(self) -> float:
        """First index depth in meters (from the header)."""
        return self.header.start_depth

    @property
    def stop_depth(self) -> float:
        """Last index depth in meters (from the header)."""
        return self.header.stop_depth

    def read(
        self, row_slice: tuple[int, int] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Read the whole log or a contiguous range of rows.

        Args:
            row_slice: Half-open ``(start, stop)`` row indices, or None for all.

        Returns:
            A ``(depth, data)`` tuple, as returned by :func:`read_las_data`.
        """
        return read_las_data(self.path, header=self.header, row_slice=row_slice)

    @property
    def depth_vector(self) -> np.ndarray:
        """Full depth column in meters, read once and cached.

        Reads only the first column of the data section, so it touches the file
        once per log. Needed for coordinate mapping and ``n_rows``.
        """
        if self._depth_vector is None:
            df = pl.read_csv(
                self.path,
                has_header=False,
                separator=self.header.delimiter,
                skip_rows=self.header.first_data_line,
                columns=[0],
                infer_schema_length=0,
            )
            self._depth_vector = (
                df[df.columns[0]].str.strip_chars().cast(pl.Float64).to_numpy()
            )
        return self._depth_vector

    @property
    def n_rows(self) -> int:
        """Number of data rows (depth samples)."""
        return int(self.depth_vector.shape[0])

    def depth_to_row(self, depth_m: float) -> int:
        """Return the row index whose depth is closest to ``depth_m``.

        Args:
            depth_m: Target depth in meters.

        Returns:
            The nearest row index, clamped to ``[0, n_rows - 1]``.
        """
        depth = self.depth_vector
        idx = int(np.searchsorted(depth, depth_m))
        if idx <= 0:
            return 0
        if idx >= depth.shape[0]:
            return depth.shape[0] - 1
        # searchsorted gives the left insertion point; pick the closer neighbour.
        return idx if (depth[idx] - depth_m) < (depth_m - depth[idx - 1]) else idx - 1

    def row_to_depth(self, row: int) -> float:
        """Return the depth in meters at a given row index.

        Args:
            row: Row index in ``[0, n_rows - 1]``.

        Returns:
            The depth in meters at that row.

        Raises:
            IndexError: If ``row`` is out of range.
        """
        if row < 0 or row >= self.n_rows:
            raise IndexError(f"row {row} out of range [0, {self.n_rows})")
        return float(self.depth_vector[row])

    def to_zarr(
        self,
        cache_dir: str | Path,
        min_rows: int = 2048,
        factor: int = 2,
        chunk_rows: int = 4096,
    ) -> Path:
        """Convert this log to a cached multiscale zarr pyramid.

        Reads the whole log, builds a depth-axis pyramid, and writes it (with a
        depth vector and metadata) to ``<cache_dir>/<stem>.zarr``. This is the
        "Convert & cache" action; it loads the full log into memory once.

        Args:
            cache_dir: Directory for the zarr cache (created if needed).
            min_rows: Coarsest-level target row count for the pyramid.
            factor: Per-level depth downsampling factor.
            chunk_rows: Chunk size along the depth axis in the zarr store.

        Returns:
            Path to the written zarr group.

        See Also:
            deeplogger.pyramid.build_depth_pyramid, write_zarr_pyramid.
        """
        from deeplogger.pyramid import build_depth_pyramid, write_zarr_pyramid

        depth, data = self.read()
        levels = build_depth_pyramid(data, min_rows=min_rows, factor=factor)
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        store_path = cache_dir / f"{self.path.stem}.zarr"
        write_zarr_pyramid(
            store_path,
            levels,
            depth,
            data_type=self.data_type,
            n_azimuth=self.n_azimuth,
            diameter=self.diameter,
            start_depth=self.start_depth,
            stop_depth=self.stop_depth,
            factor=factor,
            chunk_rows=chunk_rows,
        )
        return store_path
