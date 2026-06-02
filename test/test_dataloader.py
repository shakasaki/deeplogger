"""Tests for deeplogger.dataloader using temporary .pt files."""

import os
import tempfile

import numpy as np
import torch
import pytest

from deeplogger.dataloader import BoreholeDataset


@pytest.fixture
def tmp_pt_dir():
    """Create a temporary directory with synthetic .pt files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for i in range(5):
            image = np.random.rand(100, 360).astype(np.float32)
            mask = (np.random.rand(100, 360) > 0.8).astype(np.float32)
            torch.save([image, mask], os.path.join(tmpdir, f"sample_{i:03d}.pt"))
        yield tmpdir


def test_from_directory_loads_all_files(tmp_pt_dir):
    """from_directory should find all .pt files."""
    ds = BoreholeDataset.from_directory(tmp_pt_dir, device=torch.device("cpu"))
    assert len(ds) == 5


def test_getitem_returns_image_mask_pair(tmp_pt_dir):
    """__getitem__ should return (image, mask) tensors."""
    ds = BoreholeDataset.from_directory(tmp_pt_dir, device=torch.device("cpu"))
    image, mask = ds[0]
    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)


def test_getitem_correct_dtype(tmp_pt_dir):
    """Returned tensors should be float32."""
    ds = BoreholeDataset.from_directory(tmp_pt_dir, device=torch.device("cpu"))
    image, mask = ds[0]
    assert image.dtype == torch.float32
    assert mask.dtype == torch.float32


def test_from_directory_ignores_non_pt_files(tmp_pt_dir):
    """Non-.pt files in the directory should be ignored."""
    # Add a non-.pt file
    with open(os.path.join(tmp_pt_dir, "readme.txt"), "w") as f:
        f.write("not a tensor")
    ds = BoreholeDataset.from_directory(tmp_pt_dir, device=torch.device("cpu"))
    assert len(ds) == 5


def test_from_directory_sorted(tmp_pt_dir):
    """File paths should be sorted for reproducibility."""
    ds = BoreholeDataset.from_directory(tmp_pt_dir, device=torch.device("cpu"))
    assert ds.file_paths == sorted(ds.file_paths)


def test_custom_transform(tmp_pt_dir):
    """A custom transform should be applied to (image, mask)."""
    def flip_both(image, mask):
        return torch.flip(image, [0]), torch.flip(mask, [0])

    ds = BoreholeDataset.from_directory(tmp_pt_dir, device=torch.device("cpu"),
                                        transform=flip_both)
    image, mask = ds[0]
    # Just verify it runs and returns tensors
    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)


def test_empty_directory():
    """from_directory on empty dir should return a dataset of length 0."""
    with tempfile.TemporaryDirectory() as tmpdir:
        ds = BoreholeDataset.from_directory(tmpdir, device=torch.device("cpu"))
        assert len(ds) == 0


def test_backward_compatible_aliases():
    """Dataset and Dataset_np should be aliases for BoreholeDataset."""
    from deeplogger.dataloader import Dataset, Dataset_np
    assert Dataset is BoreholeDataset
    assert Dataset_np is BoreholeDataset
