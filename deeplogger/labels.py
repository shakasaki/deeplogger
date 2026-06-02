"""Label generation for borehole fracture detection.

Provides functions to rasterize fracture picks (described by azimuth, dip,
depth, and aperture) into binary mask images that serve as training labels
for the U-Net segmentation model.
"""

import numpy as np
import pandas as pd

from deeplogger.config import Fracture


def filter_labels_from_range(label_df: pd.DataFrame,
                             filter_type: str,
                             filter_range: (list, tuple)):
    """Filter a label DataFrame by a numeric column range.

    Args:
        label_df: label DataFrame with numeric columns
        filter_type: column name to filter on (e.g. 'depth', 'aperture')
        filter_range: [min, max] range to keep

    Returns:
        Filtered copy of the DataFrame, or empty DataFrame on invalid input.
    """
    if filter_type not in label_df.columns:
        print('Filter type does not exist in dataframe columns')
        return pd.DataFrame()
    if not isinstance(filter_range, (list, tuple)) or len(filter_range) != 2:
        print('Filter range must be a list or tuple with two entries')
        return pd.DataFrame()
    return label_df[(label_df[filter_type] >= filter_range[0]) & (label_df[filter_type] <= filter_range[1])].copy()


def crop_depth(Z_up, Z_down, depth):
    """Crop fracture depth bounds to the borehole depth vector.

    Args:
        Z_up: upper depth boundary of the fracture (array)
        Z_down: lower depth boundary of the fracture (array)
        depth: depth vector of the borehole

    Returns:
        (Z_up, Z_down) clipped to the depth vector bounds.
    """
    max_depth = max(depth)
    min_depth = min(depth)
    Z_up[Z_up > max_depth] = max_depth
    Z_down[Z_down > max_depth] = max_depth
    Z_up[Z_up < min_depth] = min_depth
    Z_down[Z_down < min_depth] = min_depth
    Z_up[Z_up < depth[0]] = depth[0]
    Z_down[Z_down > depth[-1]] = depth[-1]
    return Z_up, Z_down


def get_label(fracture: Fracture,
              depth_vector: np.ndarray,
              bh_diameter: float,
              azimuth_values: int = 360) -> np.ndarray:
    """Rasterize a fracture into a binary mask image.

    Computes the sinusoidal trace of a fracture on an unwrapped borehole
    image and returns a binary image where 1 indicates fracture presence.

    Args:
        fracture: Fracture dataclass with azimuth, dip, depth, aperture, corrections
        depth_vector: depth values for each row of the output image
        bh_diameter: borehole diameter in meters
        azimuth_values: number of azimuth columns (typically 360)

    Returns:
        Binary numpy array of shape (len(depth_vector), azimuth_values).
    """
    aperture_m = fracture.aperture_m
    azimuth_radians = np.linspace(0, 2 * np.pi, azimuth_values)
    azimuth_range = range(azimuth_values)

    amplitude = fracture.depth + np.cos(
        azimuth_radians + np.deg2rad(fracture.azimuth + fracture.azimuth_correction)
    ) * (1 / np.cos(np.deg2rad(fracture.dip + fracture.dip_correction))) * (bh_diameter / 2)

    Z_up = amplitude + aperture_m / 2
    Z_down = amplitude - aperture_m / 2
    Z_up, Z_down = crop_depth(Z_up, Z_down, depth_vector)

    im = np.zeros((depth_vector.shape[0], azimuth_values))
    if np.array(Z_up == Z_down).all():
        im[np.digitize(Z_up, depth_vector), azimuth_range] = 1
    else:
        Z_up_M = np.tile(Z_up, (depth_vector.shape[0], 1))
        Z_down_M = np.tile(Z_down, (depth_vector.shape[0], 1))
        Z_M = np.tile(depth_vector, (azimuth_values, 1)).T
        im[np.logical_and(Z_M < Z_up_M, Z_M > Z_down_M)] = 1
    return im


def apply_label(image_in: np.ndarray,
                fracture: Fracture,
                depth_vector: np.ndarray,
                bh_diameter: float,
                azimuth_values: int = 360) -> np.ndarray:
    """Apply a fracture label in-place onto an existing mask image.

    Same as get_label but modifies image_in instead of creating a new array.
    Skips fractures whose mean depth is outside the depth vector range.

    Args:
        image_in: existing mask image to modify in-place
        fracture: Fracture dataclass
        depth_vector: depth values for each row
        bh_diameter: borehole diameter in meters
        azimuth_values: number of azimuth columns

    Returns:
        The modified image_in with the fracture label applied.
    """
    if fracture.depth > max(depth_vector) or fracture.depth < min(depth_vector):
        return image_in

    aperture_m = fracture.aperture_m
    azimuth_radians = np.linspace(0, 2 * np.pi, azimuth_values)
    azimuth_range = range(azimuth_values)

    amplitude = fracture.depth + np.cos(
        azimuth_radians + np.deg2rad(fracture.azimuth)
    ) * (1 / np.cos(np.deg2rad(fracture.dip))) * (bh_diameter / 2)

    Z_up = amplitude + aperture_m
    Z_down = amplitude - aperture_m
    Z_up, Z_down = crop_depth(Z_up, Z_down, depth_vector)

    if np.array(Z_up == Z_down).all():
        image_in[np.digitize(Z_up, depth_vector), azimuth_range] = 1
    else:
        Z_up_M = np.tile(Z_up, (depth_vector.shape[0], 1))
        Z_down_M = np.tile(Z_down, (depth_vector.shape[0], 1))
        Z_M = np.tile(depth_vector, (azimuth_values, 1)).T
        image_in[np.logical_and(Z_M < Z_up_M, Z_M > Z_down_M)] = 1
    return image_in


def invert_label_values(label_subset):
    """Invert a binary label image (0 <-> 1).

    Args:
        label_subset: binary numpy array

    Returns:
        Inverted binary array.
    """
    return (~label_subset.astype(bool)).astype(int)
