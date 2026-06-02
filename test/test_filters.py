import numpy as np
from deeplogger.filters import neighbor_filter, gaussian_blur, gaussian_kernel


def test_neighbor_filter_expands_pixels(sample_identity_image):
    """Dilation with kernel_size>1 should produce more non-zero pixels."""
    result = neighbor_filter(sample_identity_image, kernel_size=2)
    assert np.sum(result > 0) > np.sum(sample_identity_image > 0)


def test_neighbor_filter_kernel_size_1_is_identity(sample_identity_image):
    """Kernel size 1 should return the original image unchanged."""
    result = neighbor_filter(sample_identity_image, kernel_size=1)
    np.testing.assert_array_equal(sample_identity_image.astype(np.uint8), result)


def test_neighbor_filter_output_is_binary(sample_identity_image):
    """Dilation of a binary image should remain binary."""
    result = neighbor_filter(sample_identity_image, kernel_size=3)
    unique_values = np.unique(result)
    assert set(unique_values).issubset({0, 1})


def test_gaussian_kernel_is_symmetric():
    """Gaussian kernel should be symmetric."""
    kernel = gaussian_kernel(5, sigma=1.0)
    np.testing.assert_array_almost_equal(kernel, kernel.T)


def test_gaussian_kernel_peak_at_center():
    """Gaussian kernel should have its maximum at the center."""
    kernel = gaussian_kernel(5, sigma=1.0)
    center = kernel.shape[0] // 2
    assert kernel[center, center] == kernel.max()


def test_gaussian_blur_no_nans(sample_identity_image):
    """Gaussian blur should not introduce NaN values."""
    result = gaussian_blur(sample_identity_image, kernel_size=3)
    assert not np.any(np.isnan(result))
