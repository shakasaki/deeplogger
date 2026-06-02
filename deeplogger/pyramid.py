"""Multiscale image pyramids for fast, lazy viewing of borehole logs.

A borehole televiewer log is very tall (millions of depth rows) but narrow
(360-720 azimuth columns). To browse it interactively, the viewer needs a
coarse overview that loads instantly and sharpens on zoom. This module builds
the image pyramid that backs that behaviour; the depth axis is downsampled while
the azimuth axis is kept at full resolution.
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path

import numpy as np

from deeplogger.config import DataType


def _downsample_depth(image: np.ndarray, factor: int) -> np.ndarray:
    """Downsample the depth axis (axis 0) by ``factor`` using a block mean.

    Rows that do not fill a complete block are trimmed from the bottom. ATV
    (floating) images use a NaN-aware mean so null-flagged samples do not
    poison a block; OTV (integer) images are averaged and rounded back.

    Args:
        image: ATV array (N, A) or OTV array (N, A, 3).
        factor: Block size along the depth axis (>= 2).

    Returns:
        Array with the depth axis reduced by ``factor``, same dtype and trailing
        shape as the input.
    """
    n_rows = image.shape[0]
    trimmed = n_rows - (n_rows % factor)
    blocks = image[:trimmed]
    # Group consecutive `factor` rows into a new axis to average over.
    grouped = blocks.reshape((trimmed // factor, factor) + image.shape[1:])

    if np.issubdtype(image.dtype, np.floating):
        with warnings.catch_warnings():
            # An all-NaN block legitimately averages to NaN; silence the notice.
            warnings.simplefilter("ignore", RuntimeWarning)
            reduced = np.nanmean(grouped, axis=1, dtype=np.float64)
        return reduced.astype(image.dtype)
    reduced = grouped.mean(axis=1, dtype=np.float64)
    return np.rint(reduced).astype(image.dtype)


def build_depth_pyramid(
    image: np.ndarray, min_rows: int = 2048, factor: int = 2
) -> list[np.ndarray]:
    """Build a multiscale pyramid by downsampling the depth axis.

    Successive levels halve (by ``factor``) the number of depth rows via a block
    mean, leaving the azimuth axis untouched. Level 0 is the original image;
    levels are added until the depth dimension is at most ``min_rows``. The
    returned list is in the order napari expects for a ``multiscale=True`` layer.

    Args:
        image: ATV array of shape (N, A) or OTV array of shape (N, A, 3).
        min_rows: Stop once a level has at most this many depth rows.
        factor: Downsampling factor per level along the depth axis (>= 2).

    Returns:
        A list ``[level0, level1, ...]`` of arrays, finest first, each with the
        same dtype and azimuth shape as the input.

    Raises:
        ValueError: If ``factor < 2``, ``min_rows < 1``, or ``image`` is not 2-D
            (ATV) or 3-D (OTV).

    Examples:
        >>> img = np.zeros((8000, 360), dtype=np.float32)
        >>> [lvl.shape[0] for lvl in build_depth_pyramid(img)]
        [8000, 4000, 2000]

    See Also:
        deeplogger.las_reader.read_las_data: Produces the full-resolution image.
    """
    if factor < 2:
        raise ValueError(f"factor must be >= 2, got {factor}")
    if min_rows < 1:
        raise ValueError(f"min_rows must be >= 1, got {min_rows}")
    if image.ndim not in (2, 3):
        raise ValueError(f"image must be 2-D (ATV) or 3-D (OTV), got {image.ndim}-D")

    levels = [image]
    current = image
    while current.shape[0] > min_rows and current.shape[0] >= factor:
        current = _downsample_depth(current, factor)
        levels.append(current)
    return levels


def write_zarr_pyramid(
    store_path: str | Path,
    levels: list[np.ndarray],
    depth: np.ndarray,
    *,
    data_type: DataType,
    n_azimuth: int,
    diameter: float | None,
    start_depth: float,
    stop_depth: float,
    factor: int = 2,
    chunk_rows: int = 4096,
    extra_attrs: dict | None = None,
) -> Path:
    """Write a multiscale pyramid and its depth vector to a zarr group.

    Uses a plain zarr layout: one array per pyramid level named "0", "1", ...
    (finest first), a "depth" array (full resolution), and group attributes
    holding the metadata needed to reconstruct a viewer.

    Args:
        store_path: Directory path for the zarr group (created/overwritten).
        levels: Pyramid levels as returned by :func:`build_depth_pyramid`.
            Lazy zarr arrays are accepted and read into memory on write.
        depth: Full-resolution depth vector of shape (N,).
        data_type: ATV or OTV.
        n_azimuth: Number of azimuth columns.
        diameter: Borehole diameter in meters, or None if unknown.
        start_depth: First depth value in meters.
        stop_depth: Last depth value in meters.
        factor: Downsampling factor used between levels (stored for reference).
        chunk_rows: Chunk size along the depth axis for the level arrays.
        extra_attrs: Optional provenance metadata merged into the group
            attributes after the standard keys (e.g. ``{"svd_removed": 3}`` to
            record that the saved data was destriped). Round-trips via
            :func:`read_zarr_pyramid`.

    Returns:
        The path to the written zarr group.

    Examples:
        >>> levels = build_depth_pyramid(np.zeros((4000, 16), np.float32))
        >>> depth = np.linspace(0, 40, 4000)
        >>> p = write_zarr_pyramid("log.zarr", levels, depth, data_type=DataType.ATV,
        ...     n_azimuth=16, diameter=0.076, start_depth=0.0, stop_depth=40.0)

    See Also:
        read_zarr_pyramid: Reads the group back as lazy arrays.
    """
    import zarr

    store_path = Path(store_path)
    group = zarr.open_group(store_path, mode="w")
    for i, level in enumerate(levels):
        level = np.asarray(level)  # materialize lazy zarr levels (e.g. re-saving)
        chunks = (min(chunk_rows, level.shape[0]),) + level.shape[1:]
        arr = group.create_array(str(i), shape=level.shape, chunks=chunks, dtype=level.dtype)
        arr[:] = level
    depth_chunks = (min(depth.shape[0], 1 << 20),)
    depth_arr = group.create_array(
        "depth", shape=depth.shape, chunks=depth_chunks, dtype=depth.dtype
    )
    depth_arr[:] = depth
    group.attrs.update(
        {
            "data_type": data_type.value,
            "n_azimuth": int(n_azimuth),
            "diameter": diameter,
            "start_depth": float(start_depth),
            "stop_depth": float(stop_depth),
            "factor": int(factor),
            "n_levels": len(levels),
        }
    )
    if extra_attrs:
        group.attrs.update(extra_attrs)
    return store_path


def read_zarr_pyramid(
    store_path: str | Path,
) -> tuple[list, np.ndarray, dict]:
    """Read a multiscale pyramid written by :func:`write_zarr_pyramid`.

    The level arrays are returned as lazy zarr arrays (not loaded into memory),
    which is what makes interactive viewing of multi-gigabyte logs possible. The
    depth vector is small and is loaded eagerly for coordinate mapping.

    Args:
        store_path: Path to the zarr group.

    Returns:
        A ``(levels, depth, attrs)`` tuple:
          - ``levels``: list of lazy zarr arrays, finest first.
          - ``depth``: depth vector (numpy array).
          - ``attrs``: metadata dict; ``data_type`` is returned as a DataType.

    See Also:
        write_zarr_pyramid: Writes the group this reads.
    """
    import zarr

    group = zarr.open_group(Path(store_path), mode="r")
    attrs = dict(group.attrs)
    levels = [group[str(i)] for i in range(int(attrs["n_levels"]))]
    depth = group["depth"][:]
    attrs["data_type"] = DataType(attrs["data_type"])
    return levels, depth, attrs


def select_pyramid_level(
    visible_rows: int,
    viewport_px: int,
    n_levels: int,
    factor: int = 2,
    oversample: float = 1.0,
) -> int:
    """Choose the pyramid level to display for the current view.

    Picks the coarsest level that still supplies at least
    ``viewport_px * oversample`` depth rows across the visible window, so the
    image is sharp without loading far more rows than there are pixels.

    Args:
        visible_rows: Number of full-resolution rows in the visible window.
        viewport_px: Height of the viewport in pixels.
        n_levels: Number of pyramid levels available.
        factor: Per-level downsampling factor (matches the pyramid).
        oversample: Target rows per viewport pixel (>1 favours finer levels).

    Returns:
        The level index in ``[0, n_levels - 1]`` (0 is full resolution).

    Raises:
        ValueError: If ``n_levels < 1``, ``viewport_px <= 0``, ``factor < 2``,
            or ``oversample <= 0``.

    Examples:
        >>> select_pyramid_level(4000, 1000, n_levels=5)
        2
        >>> select_pyramid_level(500, 1000, n_levels=5)
        0
    """
    if n_levels < 1:
        raise ValueError(f"n_levels must be >= 1, got {n_levels}")
    if viewport_px <= 0:
        raise ValueError(f"viewport_px must be > 0, got {viewport_px}")
    if factor < 2:
        raise ValueError(f"factor must be >= 2, got {factor}")
    if oversample <= 0:
        raise ValueError(f"oversample must be > 0, got {oversample}")
    if visible_rows <= 0:
        return 0

    decimation = visible_rows / (viewport_px * oversample)
    level = 0 if decimation < 1 else int(math.floor(math.log(decimation, factor)))
    return max(0, min(level, n_levels - 1))


def level_window(
    row_start: int, row_stop: int, level: int, factor: int = 2
) -> tuple[int, int]:
    """Map a full-resolution row range to a pyramid level's row range.

    Args:
        row_start: Inclusive start row at full resolution (>= 0).
        row_stop: Exclusive stop row at full resolution (> row_start).
        level: Pyramid level (>= 0; 0 returns the input range unchanged).
        factor: Per-level downsampling factor.

    Returns:
        Half-open ``(start, stop)`` row range at the given level. The start is
        floored and the stop is ceiled so the window fully covers the input.

    Raises:
        ValueError: If ``row_start < 0``, ``row_stop <= row_start``, ``level <
            0``, or ``factor < 2``.

    Examples:
        >>> level_window(10, 1000, level=1)
        (5, 500)
    """
    if row_start < 0 or row_stop <= row_start:
        raise ValueError(f"need 0 <= row_start < row_stop, got ({row_start}, {row_stop})")
    if level < 0:
        raise ValueError(f"level must be >= 0, got {level}")
    if factor < 2:
        raise ValueError(f"factor must be >= 2, got {factor}")
    step = factor**level
    start = row_start // step
    stop = -(-row_stop // step)  # ceiling division to cover the full window
    return int(start), int(stop)
