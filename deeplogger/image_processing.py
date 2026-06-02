import numpy as np


# you can add here a useful collection of functions and call them in your other scripts

def replace_empty_measurements(input_data,
                               replace_with: (int, float) = np.nan):
    """
    Replace empty cells with nan
    Args:
        input_data: 2D array of the data
    Returns:
        input_data: 2D array of the data with empty cells replaced by nan
    """
    input_data[input_data == -99999.0] = replace_with
    return input_data


def remove_svd(input_data, low_s=0, high_s=1):
    """
    Remove an svd-decomposed version of the data from the initial data.
    Args:
        input_data: 2D array of the data
        low_s: smaller singular value to keep
        high_s: largest singular value to keep
    Returns:
        svd_filterd: initial data with svd removed
        svd_decomp: decomposed part that is removed

    """
    from scipy import linalg
    # deal with nan values before svd by replacing them with 0
    input_data = replace_empty_measurements(input_data, replace_with=0)
    U, s, Vh = linalg.svd(input_data)
    sigma = np.zeros(input_data.shape)
    for i in range(low_s, high_s):
        sigma[i, i] = s[i]
    svd_decomp = np.dot(U, np.dot(sigma, Vh))
    return input_data - svd_decomp, svd_decomp


def remove_mean(image: np.ndarray,
                axis: int = 0) -> np.ndarray:
    """
    Mean removal along an axis of the image
    Args:
        data: Matrix or Trace to remove mean from
        axis: Axis along which to remove mean. 0 for rows, 1 for columns, 2 for both
    Returns:
        data with mean removed, mean matrix
    """
    samples_x, samples_y = image.shape
    if axis == 0:
        mean_matrix = np.tile(np.mean(image, axis=axis), [samples_x, 1])
        return image - mean_matrix, mean_matrix
    elif axis == 1:
        mean_matrix = np.tile(np.mean(image, axis=axis), [samples_y, 1]).T
        return image - mean_matrix, mean_matrix
    elif axis == 2:
        mean_matrix1 = np.tile(np.mean(image, axis=0), [samples_x, 1])
        mean_matrix2 = np.tile(np.mean(image, axis=1), [samples_y, 1]).T
        return image - mean_matrix1 - mean_matrix2, mean_matrix1 + mean_matrix2
    else:
        raise ValueError('Axis must be 0 or 1')


def remove_svd_components(image: np.ndarray, n_components: int = 1) -> np.ndarray:
    """Suppress coherent stripes by removing the first SVD components.

    Removes the ``n_components`` dominant singular components — the low-rank
    structure that appears as stripes running along the depth axis — and returns
    the residual. Uses an economy SVD (``full_matrices=False``), so the left
    factor is only ``(rows, min(rows, cols))`` rather than ``(rows, rows)``; this
    makes it practical on full-resolution logs with many depth rows, unlike
    :func:`remove_svd`.

    Args:
        image: 2-D amplitude image of shape (depth_rows, azimuth_cols).
        n_components: Number of leading singular components to remove (>= 0).
            0 returns the image unchanged.

    Returns:
        The filtered image, same shape and floating dtype as the input. NaNs are
        treated as 0 before decomposition (for ``n_components > 0``).

    Raises:
        ValueError: If ``image`` is not 2-D, ``n_components < 0``, or
            ``n_components`` exceeds ``min(image.shape)``.

    Examples:
        >>> stripes = np.tile(np.arange(8.0), (100, 1)).astype(np.float32)
        >>> bool(np.allclose(remove_svd_components(stripes, 1), 0, atol=1e-3))
        True

    See Also:
        remove_svd: Removes an arbitrary index range of components (builds a
            full U; fine only for small images).
    """
    if image.ndim != 2:
        raise ValueError(f"image must be 2-D, got {image.ndim}-D")
    if n_components < 0:
        raise ValueError(f"n_components must be >= 0, got {n_components}")
    rank = min(image.shape)
    if n_components > rank:
        raise ValueError(f"n_components ({n_components}) exceeds rank {rank}")

    dtype = image.dtype if np.issubdtype(image.dtype, np.floating) else np.float32
    if n_components == 0:
        return image.astype(dtype, copy=True)
    clean = np.nan_to_num(image, nan=0.0).astype(np.float64, copy=False)
    U, s, Vh = np.linalg.svd(clean, full_matrices=False)
    removed = (U[:, :n_components] * s[:n_components]) @ Vh[:n_components, :]
    return (clean - removed).astype(dtype, copy=False)


def high_pass_FFT_2D(image, cutoff_frequency):
    """High-pass filter via 2D FFT.

    Zeros out low-frequency components in the Fourier domain and reconstructs
    the image. This removes slowly-varying trends (e.g. background gradients)
    while preserving sharp features (e.g. fracture traces, edges).

    In scipy's fft2 convention (no fftshift), the DC component sits at index
    [0, 0], low positive frequencies occupy the first few indices, and their
    negative-frequency mirrors sit at the end of the array. We zero both ends.

    Args:
        image: 2D numpy array
        cutoff_frequency: fraction of frequencies to remove (0 to 0.5).
            E.g. 0.1 removes the lowest 10% of frequencies in each dimension.

    Returns:
        High-pass filtered image (real-valued).

    Note:
        Previously this function had a bug where the slice
        [int(r*keep):int(r*(1-keep))] was empty for cutoff < 0.5,
        meaning the filter silently did nothing. Fixed 2026-04-03.
    """
    from scipy import fftpack
    image_fft = fftpack.fft2(image)
    r, c = image_fft.shape

    # Number of low-frequency bins to zero in each dimension
    cutoff_r = max(1, int(r * cutoff_frequency))
    cutoff_c = max(1, int(c * cutoff_frequency))

    # Zero low positive frequencies (near DC at index 0)
    image_fft[:cutoff_r, :] = 0
    image_fft[:, :cutoff_c] = 0
    # Zero their negative-frequency mirrors (end of array)
    image_fft[-cutoff_r:, :] = 0
    image_fft[:, -cutoff_c:] = 0

    return fftpack.ifft2(image_fft).real


def high_pass_2D_kernel(image):
    '''A simple high pass filter for a 2D image that convolves the image with a kernel'''
    from scipy import ndimage
    kernel = np.array([[-1, -1, -1],
                       [-1, 8, -1],
                       [-1, -1, -1]])
    return ndimage.convolve(image, kernel)


def radon_transform(image_in: np.array,
                    theta: float = 180) -> np.array:
    '''A Radon filter for a 2D image,
    it applies a Radon transform to the image and returns the sinogram
    Args:
        image_in: 2D image
        theta: number of angles to use in the Radon transform
    Returns:
        radon_filtered: 2D image after applying the Radon transform'''
    from skimage.transform import radon, iradon
    theta = np.linspace(0., 180., max(image_in.shape), endpoint=False)
    return radon(image_in, theta=theta, circle=True)
