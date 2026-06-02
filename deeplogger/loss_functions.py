#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
from typing import Callable, Optional, Sequence

class DiceLoss(nn.Module):

    def __init__(self):
        super(DiceLoss, self).__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size()
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (
            y_pred.sum() + y_true.sum() + self.smooth
        )
        return 1. - dsc


################Adapted versions to pytorch tensors ############################
# adapt the _compute_gaussian_distributions and _distribute_labels functions to work with pytorch tensors

# Define necessary functions
def _compute_gaussian_distributions(
        index_map: torch.Tensor,
        convergence: torch.Tensor,
        sigma: float,
        kernel_size: Sequence[int],
        epsilon: float,
        i: int,
        j: int
) -> torch.Tensor:
    """Compute a Gaussian distribution centered at the convergence coordinates of a given pixel location."""
    index_map = index_map[i:i + kernel_size[0], j:j + kernel_size[1]]
    gaussian = torch.exp(-torch.sum((index_map - convergence) ** 2, dim=-1) / (2 * sigma ** 2))
    gaussian = gaussian / (torch.sum(gaussian) + epsilon)
    return gaussian

def _distribute_labels(
        distributions: torch.Tensor,
        labels: torch.Tensor,
        kernel_size: Sequence[int],
        i: int,
        j: int
) -> torch.Tensor:
    """Distribute the label values of nearby pixels according to a given set of distributions."""
    distributions = distributions[i:i + kernel_size[0], j:j + kernel_size[1]]
    labels = labels[i:i + kernel_size[0], j:j + kernel_size[1]]
    k, m = torch.meshgrid(torch.arange(kernel_size[0]), torch.arange(kernel_size[1]), indexing='ij')
    weights = distributions[k, m, kernel_size[0] - 1 - k, kernel_size[1] - 1 - m]
    pooled_label = torch.sum(weights * labels)
    return pooled_label

def smooth_sum_pool(
        deltas: torch.Tensor,
        labels: torch.Tensor,
        sigma: float = 0.5,
        kernel_size: Sequence[int] = (3, 3),
        epsilon: float = 1e-7
) -> torch.Tensor:
    """Sum pool labels using deltas."""
    height, width = deltas.shape[0], deltas.shape[1]
    i, j = torch.arange(height), torch.arange(width)
    index_map = torch.stack(torch.meshgrid(i, j, indexing='ij'), dim=-1).to(deltas.device)

    # Ensure deltas and index_map have the same shape
    if deltas.shape[-1] != 2:
        deltas = deltas.unsqueeze(-1).expand(-1, -1, 2)

    convergence = deltas + index_map
    pad_width = ((kernel_size[0] - 1) // 2, (kernel_size[1] - 1) // 2)
    index_map = torch.nn.functional.pad(index_map, pad=(0, 0, *pad_width, *pad_width))
    labels = torch.nn.functional.pad(labels, pad=(*pad_width, *pad_width))

    gaussians = torch.zeros((height, width, kernel_size[0], kernel_size[1]), device=deltas.device)
    for i in range(height - kernel_size[0] + 1):
        for j in range(width - kernel_size[1] + 1):
            gaussians[i, j] = _compute_gaussian_distributions(index_map, convergence[i, j], sigma, kernel_size, epsilon, i, j)

    gaussians = torch.nn.functional.pad(gaussians, pad=(0, 0, 0, 0, *pad_width, *pad_width))

    pooled_labels = torch.zeros((height, width), device=deltas.device)
    for i in range(height):
        for j in range(width):
            pooled_labels[i, j] = _distribute_labels(gaussians, labels, kernel_size, i, j)
    return pooled_labels


def smoothf1_loss(labels_pred: torch.Tensor, labels: torch.Tensor, epsilon: float = 1e-7) -> torch.Tensor:
    # Assuming labels_pred and labels are of shape (batch_size, height, width)
    labels_pred = labels_pred.unsqueeze(-1)  # Expand to (batch_size, height, width, 1)
    labels = labels.unsqueeze(-1)  # Expand to (batch_size, height, width, 1)

    # Calculate True Positives (tp), False Positives (fp), and False Negatives (fn)
    tp = torch.sum(labels_pred * labels)
    fp = torch.sum(labels_pred) - tp
    fn = torch.sum(labels) - tp

    # Compute SmoothF1 score
    smoothf1 = - tp / (tp + 1/2*(fp + fn) + epsilon)

    return smoothf1


def wrap_loss_fn(loss_fn: Callable, axis: int = 0, reduction: Optional[str] = 'mean') -> Callable:
    """Wrap a loss function for vectorization and loss reduction."""
    def wrapped_loss_fn(*args):
        in_axes = tuple(axis if isinstance(arg, torch.Tensor) else None for arg in args)
        loss = torch.stack([loss_fn(*args) for _ in range(in_axes[0])])
        loss = reduce_loss(loss, reduction=reduction)
        return loss
    return wrapped_loss_fn

def reduce_loss(loss: torch.Tensor, reduction: Optional[str] = 'mean') -> torch.Tensor:
    """Reduce the loss."""
    if reduction == 'mean':
        loss = torch.mean(loss)
    elif reduction == 'sum':
        loss = torch.sum(loss)
    else:
        raise ValueError("Reduction method is not supported.")
    return loss
