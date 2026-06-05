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


class BCEDiceLoss(nn.Module):
    """Weighted combination of Binary Cross-Entropy and Dice losses.

    BCE alone penalises per-pixel errors equally regardless of class frequency,
    which is poor for highly imbalanced data (fractures are <1 % of pixels).
    Dice loss directly optimises the overlap coefficient, compensating for
    imbalance by focusing on the foreground class.  Combining both yields stable
    early training (BCE) and strong foreground sensitivity (Dice).

    The combined loss is::

        L = bce_weight * BCE(pred, target) + dice_weight * Dice(pred, target)

    Accepts predictions of shape ``(B, H, W)`` **or** ``(B, 1, H, W)``; targets
    must match.  Predictions must already be sigmoid-activated probabilities in
    ``[0, 1]``.

    Args:
        bce_weight: Scalar weight applied to the BCE term. Default ``0.5``.
        dice_weight: Scalar weight applied to the Dice term. Default ``0.5``.

    References:
        Sudre, C. H., Li, W., Vercauteren, T., Ourselin, S., & Cardoso, M. J.
        (2017). Generalised Dice overlap as a deep learning loss function for
        highly unbalanced segmentations. *MICCAI Workshop on Deep Learning in
        Medical Image Analysis*. arXiv:1707.03237.

        Alexakis, C., & Armenakis, C. (2020). DeepNadirSeg: Segmentation of
        flooded buildings from nadir drone imagery using an encoder-decoder
        architecture. *Remote Sensing*, 12(10), 1672.
        https://doi.org/10.3390/rs12101672

    Examples:
        >>> import torch
        >>> loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
        >>> pred   = torch.sigmoid(torch.randn(2, 4, 4))
        >>> target = (torch.rand(2, 4, 4) > 0.8).float()
        >>> loss   = loss_fn(pred, target)
        >>> 0.0 <= loss.item() <= 2.0
        True
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self._bce = nn.BCELoss()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Flatten both to 1-D so the function is shape-agnostic (3-D or 4-D).
        p = pred.contiguous().view(-1)
        t = target.contiguous().view(-1)

        smooth = 1.0
        intersection = (p * t).sum()
        dice = 1.0 - (2.0 * intersection + smooth) / (p.sum() + t.sum() + smooth)

        bce = self._bce(pred, target)
        return self.bce_weight * bce + self.dice_weight * dice


class FocalTverskyLoss(nn.Module):
    """Focal-Tversky loss for severely imbalanced segmentation.

    The Tversky index generalises Dice by decoupling the penalties on false
    positives and false negatives::

        TI = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)

    with ``TP = sum(p*t)``, ``FP = sum(p*(1-t))`` and ``FN = sum((1-p)*t)``.
    Setting ``alpha < beta`` penalises false negatives more, emphasising recall
    of the rare foreground class (fractures are <1 % of pixels). The Focal
    variant raises the Tversky loss to a focusing power ``gamma`` to concentrate
    learning on hard examples::

        L = (1 - TI) ** gamma

    With ``gamma == 1.0`` this is the plain Tversky loss; the literature-optimal
    focusing exponent is ``gamma = 4/3``. Accepts predictions of shape
    ``(B, H, W)`` **or** ``(B, 1, H, W)`` (targets must match); predictions must
    already be sigmoid-activated probabilities in ``[0, 1]``.

    Args:
        alpha: Weight on false positives. Default ``0.3``.
        beta: Weight on false negatives. Default ``0.7``.
        gamma: Focusing exponent. Default ``4/3``; use ``1.0`` for plain Tversky.
        smooth: Smoothing constant for numerical stability. Default ``1.0``.

    References:
        Salehi, S. S. M., Erdogmus, D., & Gholipour, A. (2017). Tversky loss
        function for image segmentation using 3D fully convolutional deep
        networks. *MICCAI Workshop on Machine Learning in Medical Imaging*.
        arXiv:1706.05721.

        Abraham, N., & Khan, N. M. (2019). A novel focal Tversky loss function
        with improved attention U-Net for lesion segmentation. *IEEE ISBI*,
        683-687. arXiv:1810.07842.

    Examples:
        >>> import torch
        >>> loss_fn = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=4/3)
        >>> pred   = torch.sigmoid(torch.randn(2, 4, 4))
        >>> target = (torch.rand(2, 4, 4) > 0.8).float()
        >>> 0.0 <= loss_fn(pred, target).item() <= 1.0
        True
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 4.0 / 3.0,
        smooth: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Flatten to 1-D so the function is shape-agnostic (3-D or 4-D).
        p = pred.contiguous().view(-1)
        t = target.contiguous().view(-1)

        tp = (p * t).sum()
        fp = (p * (1.0 - t)).sum()
        fn = ((1.0 - p) * t).sum()

        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return (1.0 - tversky).pow(self.gamma)


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
