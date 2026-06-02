"""PyTorch Dataset classes for DeepLogger.

Provides Dataset implementations for loading borehole image/mask pairs
from pre-saved .pt files.
"""

import os
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch.utils import data


class BoreholeDataset(data.Dataset):
    """Dataset that loads pre-processed borehole image/mask pairs from .pt files.

    Each .pt file contains [image_tensor, mask_tensor]. The dataset handles
    device placement and optional transforms.

    Args:
        file_paths: list of paths to .pt files
        device: torch device to load tensors onto (default: auto-detect GPU/CPU)
        transform: optional callable applied to (image, mask) tuple
    """

    def __init__(self,
                 file_paths: List[str],
                 device: Optional[torch.device] = None,
                 transform=None):
        self.file_paths = file_paths
        self.device = device or torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sample = torch.load(self.file_paths[index], weights_only=False)
        if isinstance(sample[0], torch.Tensor):
            image = sample[0].to(self.device).float()
            mask = sample[1].to(self.device).float()
        else:
            image = torch.from_numpy(np.asarray(sample[0], dtype=np.float32)).to(self.device)
            mask = torch.from_numpy(np.asarray(sample[1], dtype=np.float32)).to(self.device)

        if self.transform:
            image, mask = self.transform(image, mask)

        return image, mask

    @classmethod
    def from_directory(cls, directory: str, **kwargs) -> "BoreholeDataset":
        """Create a dataset from all .pt files in a directory.

        Args:
            directory: path to directory containing .pt files
            **kwargs: additional arguments passed to __init__

        Returns:
            BoreholeDataset instance.
        """
        file_paths = sorted([
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.endswith('.pt') and os.path.isfile(os.path.join(directory, f))
        ])
        return cls(file_paths, **kwargs)


# Backward-compatible aliases
Dataset = BoreholeDataset
Dataset_np = BoreholeDataset
