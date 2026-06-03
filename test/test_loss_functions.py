"""Tests for deeplogger.loss_functions.

Uses synthetic prediction/target tensors to verify loss functions have
correct mathematical properties (boundary values, gradients, symmetry).
"""

import torch
import pytest

from deeplogger.loss_functions import BCEDiceLoss, DiceLoss, smoothf1_loss, reduce_loss


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


# --- BCEDiceLoss ---

class TestBCEDiceLoss:
    """Tests for BCEDiceLoss (combined BCE + Dice)."""

    def setup_method(self):
        self.loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

    # -- output range --

    def test_output_is_scalar(self):
        pred   = torch.sigmoid(torch.randn(2, 4, 4))
        target = (torch.rand(2, 4, 4) > 0.8).float()
        loss = self.loss_fn(pred, target)
        assert loss.ndim == 0

    def test_output_non_negative(self):
        pred   = torch.sigmoid(torch.randn(2, 16, 16))
        target = (torch.rand(2, 16, 16) > 0.9).float()
        assert self.loss_fn(pred, target).item() >= 0.0

    # -- perfect prediction --

    def test_perfect_prediction_low_loss(self):
        target = torch.zeros(2, 32, 32)
        target[:, 8:16, 8:16] = 1.0
        pred = target.clone().clamp(1e-6, 1 - 1e-6)
        loss = self.loss_fn(pred, target)
        assert loss.item() < 0.05

    # -- shape flexibility: 3-D (B, H, W) and 4-D (B, 1, H, W) --

    def test_accepts_3d_input(self):
        pred   = torch.sigmoid(torch.randn(2, 16, 16))
        target = (torch.rand(2, 16, 16) > 0.8).float()
        loss = self.loss_fn(pred, target)
        assert loss.item() >= 0.0

    def test_accepts_4d_input(self):
        pred   = torch.sigmoid(torch.randn(2, 1, 16, 16))
        target = (torch.rand(2, 1, 16, 16) > 0.8).float()
        loss = self.loss_fn(pred, target)
        assert loss.item() >= 0.0

    # -- weights --

    def test_bce_only_weight(self):
        """With dice_weight=0, result should equal plain BCE."""
        fn_bce_only = BCEDiceLoss(bce_weight=1.0, dice_weight=0.0)
        import torch.nn as nn
        bce = nn.BCELoss()
        pred   = torch.sigmoid(torch.randn(2, 16, 16))
        target = (torch.rand(2, 16, 16) > 0.8).float()
        assert abs(fn_bce_only(pred, target).item() - bce(pred, target).item()) < 1e-5

    def test_custom_weights_interpolate(self):
        """Weighted combination lies between pure-BCE and pure-Dice values."""
        import torch.nn as nn
        pred   = torch.sigmoid(torch.randn(2, 16, 16))
        target = (torch.rand(2, 16, 16) > 0.8).float()
        from deeplogger.loss_functions import DiceLoss
        bce_val  = nn.BCELoss()(pred, target).item()
        dice_val = DiceLoss()(pred.unsqueeze(1), target.unsqueeze(1)).item()
        combined = BCEDiceLoss(0.5, 0.5)(pred, target).item()
        expected = 0.5 * bce_val + 0.5 * dice_val
        assert abs(combined - expected) < 1e-5

    # -- gradient --

    def test_gradient_flows(self):
        # Use a leaf tensor already in [0,1] so .grad is populated after backward.
        pred   = torch.rand(2, 16, 16, requires_grad=True)
        target = (torch.rand(2, 16, 16) > 0.8).float()
        loss = self.loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()


# --- ModelType in TrainingConfig ---

class TestModelType:
    """Verify ModelType enum round-trips through TrainingConfig serialisation."""

    def test_model_type_in_config(self):
        from deeplogger.config import ModelType, TrainingConfig
        cfg = TrainingConfig(data_dir="/tmp", model_dir="/tmp",
                             model_type=ModelType.ATTENTION_UNET_ATV)
        assert cfg.model_type == ModelType.ATTENTION_UNET_ATV

    def test_model_type_round_trips_dict(self):
        from deeplogger.config import ModelType, TrainingConfig
        cfg = TrainingConfig(data_dir="/tmp", model_dir="/tmp",
                             model_type=ModelType.UNET_ATV_V2)
        d   = cfg.to_dict()
        cfg2 = TrainingConfig.from_dict(d)
        assert cfg2.model_type == ModelType.UNET_ATV_V2

    def test_default_model_type_is_v2(self):
        from deeplogger.config import ModelType, TrainingConfig
        cfg = TrainingConfig(data_dir="/tmp", model_dir="/tmp")
        assert cfg.model_type == ModelType.UNET_ATV_V2

    def test_legacy_config_without_model_type(self):
        """from_dict with no model_type key should not raise."""
        from deeplogger.config import TrainingConfig
        d = {"data_dir": "/tmp", "model_dir": "/tmp"}
        cfg = TrainingConfig.from_dict(d)
        assert cfg.model_type is not None  # uses default
