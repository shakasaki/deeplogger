"""Tests for deeplogger.image_processing using synthetic borehole-like data.

Synthetic data strategy:
- Borehole images have strong horizontal banding (lithology layers) + noise.
- SVD removal targets this banding (first singular value captures it).
- FFT filtering removes low-frequency trends.
- We test mathematical properties rather than exact pixel values.

Visual tests save before/after images to test/output/ for manual inspection.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from deeplogger.image_processing import (
    replace_empty_measurements,
    remove_svd,
    remove_svd_components,
    remove_mean,
    high_pass_FFT_2D,
    high_pass_2D_kernel,
    radon_transform,
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def test_remove_svd_components_removes_pure_stripes():
    """A pure rank-1 vertical-stripe image should vanish when its top
    component is removed."""
    stripes = np.tile(np.arange(16.0), (200, 1)).astype(np.float32)
    out = remove_svd_components(stripes, 1)
    assert np.allclose(out, 0.0, atol=1e-3)


def test_remove_svd_components_reduces_stripe_energy():
    """Removing the first component should sharply reduce stripe energy while
    keeping the residual structure."""
    rng = np.random.default_rng(0)
    stripes = np.tile(rng.standard_normal(32) * 10.0, (300, 1))
    data = (stripes + rng.standard_normal((300, 32))).astype(np.float32)
    out = remove_svd_components(data, 1)
    # Stripe energy = variance of the per-azimuth (column) mean down depth.
    before = np.var(data.mean(axis=0))
    after = np.var(out.mean(axis=0))
    assert after < before * 0.1


def test_remove_svd_components_preserves_shape_and_dtype():
    """Output should match the input shape and floating dtype."""
    data = np.random.default_rng(1).standard_normal((100, 16)).astype(np.float32)
    out = remove_svd_components(data, 2)
    assert out.shape == (100, 16)
    assert out.dtype == np.float32


def test_remove_svd_components_zero_is_identity():
    """n_components=0 should return the image unchanged (NaNs preserved)."""
    data = np.array([[1.0, np.nan], [3.0, 4.0]], dtype=np.float32)
    out = remove_svd_components(data, 0)
    assert out[0, 0] == 1.0
    assert np.isnan(out[0, 1])


def test_remove_svd_components_handles_nan():
    """NaNs should be treated as 0, leaving no NaNs in the output."""
    data = np.tile(np.arange(8.0), (50, 1)).astype(np.float32)
    data[10, 3] = np.nan
    out = remove_svd_components(data, 1)
    assert not np.any(np.isnan(out))


@pytest.mark.parametrize("bad", [-1, 100])
def test_remove_svd_components_invalid_n(bad):
    """Negative n or n beyond the rank should raise ValueError."""
    data = np.zeros((20, 8), dtype=np.float32)  # rank <= 8
    with pytest.raises(ValueError):
        remove_svd_components(data, bad)


def test_remove_svd_components_requires_2d():
    """A 3-D (e.g. OTV RGB) image should raise ValueError."""
    with pytest.raises(ValueError, match="2-D"):
        remove_svd_components(np.zeros((10, 8, 3), dtype=np.float32), 1)


def _save_comparison(original, processed, title, filename, removed=None):
    """Save a side-by-side comparison image to test/output/."""
    n_cols = 3 if removed is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))
    axes[0].imshow(original, aspect="auto", cmap="hot")
    axes[0].set_title("Original")
    axes[0].set_xlabel("Azimuth")
    axes[0].set_ylabel("Depth index")
    axes[1].imshow(processed, aspect="auto", cmap="hot")
    axes[1].set_title("Processed")
    axes[1].set_xlabel("Azimuth")
    if removed is not None:
        axes[2].imshow(removed, aspect="auto", cmap="hot")
        axes[2].set_title("Removed")
        axes[2].set_xlabel("Azimuth")
    fig.suptitle(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, filename), dpi=100)
    plt.close(fig)


# --- Fixtures ---

@pytest.fixture
def synthetic_borehole_image():
    """A 100x360 image mimicking ATV amplitude data.

    Horizontal banding (rank-1 component) + random noise + a sinusoidal
    fracture trace.
    """
    np.random.seed(42)
    rows, cols = 100, 360
    banding = np.outer(np.linspace(0.5, 1.5, rows), np.ones(cols))
    noise = 0.1 * np.random.randn(rows, cols)
    azimuth = np.linspace(0, 2 * np.pi, cols)
    fracture_depth = int(rows / 2) + (5 * np.cos(azimuth)).astype(int)
    fracture = np.zeros((rows, cols))
    for col_idx, row_idx in enumerate(fracture_depth):
        if 0 <= row_idx < rows:
            fracture[row_idx, col_idx] = 0.8
    return banding + noise + fracture


@pytest.fixture
def constant_image():
    """A uniform image — useful for edge cases."""
    return np.ones((50, 50)) * 5.0


@pytest.fixture
def image_with_empty_measurements():
    """Image with sentinel -99999.0 values (empty measurements in LAS files)."""
    img = np.random.rand(20, 20)
    img[5, 10] = -99999.0
    img[15, 3] = -99999.0
    return img


# --- replace_empty_measurements ---

def test_replace_empty_with_nan(image_with_empty_measurements):
    """Sentinel values should be replaced with NaN."""
    result = replace_empty_measurements(image_with_empty_measurements.copy())
    assert np.isnan(result[5, 10])
    assert np.isnan(result[15, 3])


def test_replace_empty_with_zero(image_with_empty_measurements):
    """Sentinel values should be replaceable with 0."""
    result = replace_empty_measurements(image_with_empty_measurements.copy(), replace_with=0)
    assert result[5, 10] == 0
    assert result[15, 3] == 0


def test_replace_empty_preserves_normal_values(image_with_empty_measurements):
    """Non-sentinel values should be untouched."""
    original = image_with_empty_measurements.copy()
    result = replace_empty_measurements(image_with_empty_measurements.copy(), replace_with=0)
    assert result[0, 0] == original[0, 0]


# --- remove_svd ---

def test_remove_svd_output_shapes(synthetic_borehole_image):
    """Output shapes should match input."""
    filtered, decomp = remove_svd(synthetic_borehole_image.copy(), low_s=0, high_s=1)
    assert filtered.shape == synthetic_borehole_image.shape
    assert decomp.shape == synthetic_borehole_image.shape


def test_remove_svd_components_sum_to_input(synthetic_borehole_image):
    """filtered + decomp should reconstruct the (NaN-cleaned) input."""
    img = synthetic_borehole_image.copy()
    filtered, decomp = remove_svd(img, low_s=0, high_s=1)
    cleaned = replace_empty_measurements(synthetic_borehole_image.copy(), replace_with=0)
    np.testing.assert_array_almost_equal(filtered + decomp, cleaned)


def test_remove_svd_reduces_horizontal_banding():
    """Removing first singular value should reduce rank-1 horizontal banding."""
    np.random.seed(0)
    rows, cols = 50, 360
    banding = np.outer(np.linspace(1, 10, rows), np.ones(cols))
    noise = 0.01 * np.random.randn(rows, cols)
    img = banding + noise

    filtered, _ = remove_svd(img, low_s=0, high_s=1)
    row_var_before = np.var(np.mean(img, axis=1))
    row_var_after = np.var(np.mean(filtered, axis=1))
    assert row_var_after < row_var_before * 0.01


def test_remove_svd_no_nans_in_output(synthetic_borehole_image):
    """Output should be NaN-free even if input has sentinels."""
    img = synthetic_borehole_image.copy()
    img[10, 20] = -99999.0
    filtered, decomp = remove_svd(img, low_s=0, high_s=1)
    assert not np.any(np.isnan(filtered))
    assert not np.any(np.isnan(decomp))


def test_remove_svd_visual(synthetic_borehole_image):
    """Visual: SVD removal should strip horizontal banding, preserving fracture."""
    img = synthetic_borehole_image.copy()
    filtered, removed = remove_svd(img, low_s=0, high_s=1)
    _save_comparison(img, filtered, "SVD Removal (1st singular value)",
                     "svd_removal.png", removed=removed)


# --- remove_mean ---

def test_remove_mean_axis0_zero_column_mean(synthetic_borehole_image):
    """After removing mean along axis 0, column means should be ~0."""
    result, mean_matrix = remove_mean(synthetic_borehole_image.copy(), axis=0)
    col_means = np.mean(result, axis=0)
    np.testing.assert_array_almost_equal(col_means, 0, decimal=10)


def test_remove_mean_axis1_zero_row_mean(synthetic_borehole_image):
    """After removing mean along axis 1, row means should be ~0."""
    result, mean_matrix = remove_mean(synthetic_borehole_image.copy(), axis=1)
    row_means = np.mean(result, axis=1)
    np.testing.assert_array_almost_equal(row_means, 0, decimal=10)


def test_remove_mean_axis2_reconstructs(synthetic_borehole_image):
    """result + mean_matrix should equal the original for axis=2."""
    img = synthetic_borehole_image.copy()
    result, mean_matrix = remove_mean(img, axis=2)
    np.testing.assert_array_almost_equal(result + mean_matrix, img)


def test_remove_mean_reconstructs_input(synthetic_borehole_image):
    """result + mean_matrix should equal the original image."""
    img = synthetic_borehole_image.copy()
    result, mean_matrix = remove_mean(img, axis=0)
    np.testing.assert_array_almost_equal(result + mean_matrix, img)


def test_remove_mean_invalid_axis():
    """Axis > 2 should raise ValueError."""
    with pytest.raises(ValueError):
        remove_mean(np.ones((10, 10)), axis=5)


def test_remove_mean_constant_image(constant_image):
    """Constant image should yield all zeros after mean removal."""
    result, mean_matrix = remove_mean(constant_image, axis=0)
    np.testing.assert_array_almost_equal(result, 0)


def test_remove_mean_visual(synthetic_borehole_image):
    """Visual: mean removal along axis 0 (column-wise) and axis 1 (row-wise)."""
    img = synthetic_borehole_image.copy()
    result_ax0, removed_ax0 = remove_mean(img, axis=0)
    result_ax1, removed_ax1 = remove_mean(img, axis=1)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax_row, result, removed, label in [
        (axes[0], result_ax0, removed_ax0, "Axis 0 (column mean)"),
        (axes[1], result_ax1, removed_ax1, "Axis 1 (row mean)"),
    ]:
        ax_row[0].imshow(img, aspect="auto", cmap="hot")
        ax_row[0].set_title("Original")
        ax_row[1].imshow(result, aspect="auto", cmap="hot")
        ax_row[1].set_title(f"After {label} removal")
        ax_row[2].imshow(removed, aspect="auto", cmap="hot")
        ax_row[2].set_title(f"Removed ({label})")
    for ax in axes.flat:
        ax.set_xlabel("Azimuth")
        ax.set_ylabel("Depth index")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "mean_removal.png"), dpi=100)
    plt.close(fig)


# --- high_pass_FFT_2D ---

def test_fft_highpass_removes_dc():
    """After high-pass filtering, the DC component (mean) should be ~0."""
    np.random.seed(0)
    img = np.random.rand(64, 64) + 5.0  # large DC offset
    filtered = high_pass_FFT_2D(img, cutoff_frequency=0.05)
    assert abs(np.mean(filtered)) < 0.5  # mean was 5.5, should be near 0


def test_fft_highpass_preserves_shape(synthetic_borehole_image):
    """Output shape should match input."""
    filtered = high_pass_FFT_2D(synthetic_borehole_image.copy(), cutoff_frequency=0.1)
    assert filtered.shape == synthetic_borehole_image.shape


def test_fft_highpass_output_is_real(synthetic_borehole_image):
    """Output should be real-valued (no imaginary residuals)."""
    filtered = high_pass_FFT_2D(synthetic_borehole_image.copy(), cutoff_frequency=0.2)
    assert filtered.dtype in [np.float64, np.float32]


def test_fft_highpass_small_cutoff_still_works():
    """Cutoff < 0.5 should now actually filter (this was the bug)."""
    np.random.seed(0)
    img = np.random.rand(100, 100) + 10.0  # strong DC
    filtered = high_pass_FFT_2D(img, cutoff_frequency=0.1)
    # The mean should be dramatically reduced since DC is zeroed
    assert abs(np.mean(filtered)) < abs(np.mean(img)) * 0.1


def test_fft_highpass_larger_cutoff_removes_more():
    """Increasing cutoff should remove more low-frequency energy."""
    np.random.seed(0)
    img = np.random.rand(64, 64) + 5.0
    filtered_small = high_pass_FFT_2D(img.copy(), cutoff_frequency=0.05)
    filtered_large = high_pass_FFT_2D(img.copy(), cutoff_frequency=0.2)
    # Larger cutoff should leave less energy (lower variance)
    assert np.var(filtered_large) < np.var(filtered_small)


def test_fft_highpass_visual(synthetic_borehole_image):
    """Visual: FFT high-pass with multiple cutoff values."""
    img = synthetic_borehole_image.copy()
    cutoffs = [0.02, 0.05, 0.1, 0.2]

    fig, axes = plt.subplots(1, len(cutoffs) + 1, figsize=(4 * (len(cutoffs) + 1), 4))
    axes[0].imshow(img, aspect="auto", cmap="hot")
    axes[0].set_title("Original")
    axes[0].set_xlabel("Azimuth")
    axes[0].set_ylabel("Depth index")

    for i, cutoff in enumerate(cutoffs):
        filtered = high_pass_FFT_2D(img.copy(), cutoff_frequency=cutoff)
        axes[i + 1].imshow(filtered, aspect="auto", cmap="hot")
        axes[i + 1].set_title(f"Cutoff = {cutoff}")
        axes[i + 1].set_xlabel("Azimuth")

    fig.suptitle("FFT High-Pass Filter — increasing cutoff removes more low-freq content")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "fft_highpass.png"), dpi=100)
    plt.close(fig)


# --- high_pass_2D_kernel ---

def test_kernel_highpass_preserves_shape(synthetic_borehole_image):
    """Output shape should match input."""
    filtered = high_pass_2D_kernel(synthetic_borehole_image.copy())
    assert filtered.shape == synthetic_borehole_image.shape


def test_kernel_highpass_detects_edges():
    """A step edge should produce non-zero response at the boundary."""
    img = np.zeros((50, 50))
    img[:, 25:] = 1.0
    filtered = high_pass_2D_kernel(img)
    assert np.max(np.abs(filtered[:, 24:26])) > 0


def test_kernel_highpass_constant_is_zero(constant_image):
    """Laplacian of a constant image should be zero."""
    filtered = high_pass_2D_kernel(constant_image)
    np.testing.assert_array_almost_equal(filtered, 0)


def test_kernel_highpass_visual(synthetic_borehole_image):
    """Visual: Laplacian kernel high-pass filter."""
    img = synthetic_borehole_image.copy()
    filtered = high_pass_2D_kernel(img)
    _save_comparison(img, filtered,
                     "Laplacian Kernel High-Pass Filter",
                     "kernel_highpass.png")


# --- radon_transform ---

_skimage_available = True
try:
    from skimage.transform import radon
except (ImportError, ValueError):
    _skimage_available = False


@pytest.mark.skipif(not _skimage_available, reason="scikit-image not compatible with current numpy")
def test_radon_output_shape():
    """Radon transform output should have expected dimensions."""
    img = np.random.rand(64, 64)
    sinogram = radon_transform(img)
    assert sinogram.ndim == 2
    assert sinogram.shape[1] == 64  # number of angles = max(shape)


@pytest.mark.skipif(not _skimage_available, reason="scikit-image not compatible with current numpy")
def test_radon_zero_input():
    """Radon of a zero image should be zero."""
    img = np.zeros((32, 32))
    sinogram = radon_transform(img)
    np.testing.assert_array_almost_equal(sinogram, 0)


@pytest.mark.skipif(not _skimage_available, reason="scikit-image not compatible with current numpy")
def test_radon_nonzero_for_nonzero_input():
    """A non-trivial image should produce a non-zero sinogram."""
    np.random.seed(0)
    img = np.random.rand(32, 32)
    sinogram = radon_transform(img)
    assert np.sum(np.abs(sinogram)) > 0
