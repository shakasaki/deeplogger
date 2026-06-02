"""Tests for deeplogger.loss_functions.

Uses synthetic prediction/target tensors to verify loss functions have
correct mathematical properties (boundary values, gradients, symmetry).
"""

import torch
import pytest

from deeplogger.loss_functions import DiceLoss, smoothf1_loss, reduce_loss


# --- DiceLoss ---

class TestDiceLoss:
    """Tests for the Dice loss (1 - Dice coefficient)."""

    def setup_method(self):
        self.loss_fn = DiceLoss()

    def test_perfect_prediction_loss_near_zero(self):
        """When prediction == target, Dice loss should be near 0."""
        target = torch.zeros(2, 1, 32, 32)
        target[:, :, 10:20, 10:20] = 1.0
        pred = target.clone()
        loss = self.loss_fn(pred, target)
        assert loss.item() < 0.01

    def test_completely_wrong_prediction_loss_near_one(self):
        """When prediction and target have no overlap, loss should be near 1."""
        pred = torch.zeros(2, 1, 32, 32)
        pred[:, :, 0:10, 0:10] = 1.0
        target = torch.zeros(2, 1, 32, 32)
        target[:, :, 20:30, 20:30] = 1.0
        loss = self.loss_fn(pred, target)
        assert loss.item() > 0.9

    def test_loss_is_symmetric(self):
        """Dice(a, b) should equal Dice(b, a)."""
        a = torch.rand(2, 1, 16, 16)
        b = torch.rand(2, 1, 16, 16)
        assert abs(self.loss_fn(a, b).item() - self.loss_fn(b, a).item()) < 1e-6

    def test_loss_bounded_zero_one(self):
        """Dice loss should always be between 0 and 1."""
        pred = torch.rand(4, 1, 16, 16)
        target = torch.rand(4, 1, 16, 16)
        loss = self.loss_fn(pred, target)
        assert 0.0 <= loss.item() <= 1.0

    def test_gradient_flows(self):
        """Loss should produce gradients for backpropagation."""
        pred = torch.rand(2, 1, 16, 16, requires_grad=True)
        target = torch.rand(2, 1, 16, 16)
        loss = self.loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert pred.grad.shape == pred.shape

    def test_all_zeros_with_smoothing(self):
        """All-zero pred and target should give loss = 0 (due to smoothing)."""
        pred = torch.zeros(2, 1, 16, 16)
        target = torch.zeros(2, 1, 16, 16)
        loss = self.loss_fn(pred, target)
        # smooth=1.0: dsc = (0 + 1) / (0 + 0 + 1) = 1, loss = 0
        assert loss.item() < 0.01


# --- smoothf1_loss ---

class TestSmoothF1Loss:
    """Tests for the smooth F1 loss."""

    def test_perfect_prediction_returns_negative_one(self):
        """Perfect overlap should give F1 = 1, so loss = -1."""
        target = torch.zeros(2, 32, 32)
        target[:, 10:20, 10:20] = 1.0
        pred = target.clone()
        loss = smoothf1_loss(pred, target)
        assert abs(loss.item() - (-1.0)) < 0.01

    def test_no_overlap_loss_near_zero(self):
        """No overlap means TP=0, so loss = -0 / (0 + ...) ≈ 0."""
        pred = torch.zeros(2, 32, 32)
        pred[:, 0:5, 0:5] = 1.0
        target = torch.zeros(2, 32, 32)
        target[:, 25:30, 25:30] = 1.0
        loss = smoothf1_loss(pred, target)
        assert loss.item() > -0.1  # near 0

    def test_loss_is_negative(self):
        """Smooth F1 loss should be <= 0 (negative F1 score)."""
        pred = torch.rand(2, 16, 16)
        target = torch.rand(2, 16, 16)
        loss = smoothf1_loss(pred, target)
        assert loss.item() <= 0.0

    def test_gradient_flows(self):
        """Loss should produce gradients."""
        pred = torch.rand(2, 16, 16, requires_grad=True)
        target = torch.rand(2, 16, 16)
        loss = smoothf1_loss(pred, target)
        loss.backward()
        assert pred.grad is not None


# --- reduce_loss ---

class TestReduceLoss:

    def test_mean_reduction(self):
        loss = torch.tensor([1.0, 2.0, 3.0])
        assert reduce_loss(loss, 'mean').item() == 2.0

    def test_sum_reduction(self):
        loss = torch.tensor([1.0, 2.0, 3.0])
        assert reduce_loss(loss, 'sum').item() == 6.0

    def test_invalid_reduction_raises(self):
        with pytest.raises(ValueError):
            reduce_loss(torch.tensor([1.0]), 'invalid')
