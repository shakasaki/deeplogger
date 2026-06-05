"""Tests for deeplogger.loss_functions.

Uses synthetic prediction/target tensors to verify loss functions have
correct mathematical properties (boundary values, gradients, symmetry).
"""

import torch
import pytest

from deeplogger.loss_functions import (
    BCEDiceLoss,
    DiceLoss,
    FocalTverskyLoss,
    smoothf1_loss,
    reduce_loss,
)


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


# --- FocalTverskyLoss ---

class TestFocalTverskyLoss:
    """Tests for the Focal-Tversky loss (and its plain-Tversky gamma=1 case)."""

    def setup_method(self):
        self.loss_fn = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=4.0 / 3.0)

    # -- output range / shape --

    def test_output_is_scalar(self):
        pred   = torch.sigmoid(torch.randn(2, 4, 4))
        target = (torch.rand(2, 4, 4) > 0.8).float()
        assert self.loss_fn(pred, target).ndim == 0

    def test_loss_bounded_zero_one(self):
        """With smooth=1 and gamma>=1 the loss stays in [0, 1]."""
        pred   = torch.rand(4, 1, 16, 16)
        target = (torch.rand(4, 1, 16, 16) > 0.8).float()
        loss = self.loss_fn(pred, target)
        assert 0.0 <= loss.item() <= 1.0

    def test_accepts_3d_and_4d(self):
        for shape in [(2, 16, 16), (2, 1, 16, 16)]:
            pred   = torch.sigmoid(torch.randn(*shape))
            target = (torch.rand(*shape) > 0.8).float()
            assert self.loss_fn(pred, target).item() >= 0.0

    # -- boundary behaviour --

    def test_perfect_prediction_low_loss(self):
        target = torch.zeros(2, 32, 32)
        target[:, 8:16, 8:16] = 1.0
        pred = target.clone().clamp(1e-6, 1 - 1e-6)
        assert self.loss_fn(pred, target).item() < 0.05

    def test_no_overlap_high_loss(self):
        pred = torch.zeros(2, 32, 32)
        pred[:, 0:8, 0:8] = 1.0
        target = torch.zeros(2, 32, 32)
        target[:, 20:28, 20:28] = 1.0
        assert self.loss_fn(pred, target).item() > 0.9

    def test_all_zeros_with_smoothing(self):
        """Empty pred and target give TI = 1, loss = 0 (smooth term)."""
        pred = torch.zeros(2, 16, 16)
        target = torch.zeros(2, 16, 16)
        assert self.loss_fn(pred, target).item() < 0.01

    # -- gamma=1 reduces to Tversky; alpha=beta=0.5 reduces to Dice --

    def test_tversky_symmetric_alpha_beta_equals_dice(self):
        """Tversky with alpha=beta=0.5 (gamma=1) equals the Dice loss in the
        smooth->0 limit. The two place the smoothing constant differently, so
        they agree only as the foreground sums grow large relative to smooth;
        large tensors make the residual gap negligible."""
        pred   = torch.rand(2, 1, 128, 128)
        target = (torch.rand(2, 1, 128, 128) > 0.7).float()
        tversky = FocalTverskyLoss(alpha=0.5, beta=0.5, gamma=1.0)(pred, target)
        dice = DiceLoss()(pred, target)
        assert abs(tversky.item() - dice.item()) < 1e-3

    def test_recall_weighting_penalises_fn_more_than_fp(self):
        """alpha<beta: a false-negative-heavy error costs more than the mirror
        false-positive-heavy error of equal magnitude."""
        fn_fn = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.0)
        target = torch.zeros(1, 16, 16)
        target[:, 4:12, 4:12] = 1.0
        # Under-segment (misses half the foreground -> false negatives).
        under = torch.zeros(1, 16, 16)
        under[:, 4:8, 4:12] = 1.0
        # Over-segment by the same number of pixels (false positives).
        over = target.clone()
        over[:, 12:16, 4:12] = 1.0
        assert fn_fn(under, target).item() > fn_fn(over, target).item()

    def test_focusing_exponent_shrinks_small_loss(self):
        """For a loss in (0,1), gamma>1 yields a smaller value than gamma=1."""
        pred   = torch.sigmoid(torch.randn(2, 16, 16))
        target = (torch.rand(2, 16, 16) > 0.8).float()
        plain  = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=1.0)(pred, target)
        focal  = FocalTverskyLoss(alpha=0.3, beta=0.7, gamma=4.0 / 3.0)(pred, target)
        if 0.0 < plain.item() < 1.0:
            assert focal.item() < plain.item()

    # -- gradient --

    def test_gradient_flows(self):
        pred   = torch.rand(2, 16, 16, requires_grad=True)
        target = (torch.rand(2, 16, 16) > 0.8).float()
        loss = self.loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()


# --- _build_loss wiring ---

class TestBuildLossTversky:
    """The new LossType values must construct the right loss in train.py."""

    def test_focal_tversky_loss_type_builds(self):
        from deeplogger.config import LossType, TrainingConfig
        from deeplogger.train import _build_loss
        cfg = TrainingConfig(data_dir="/tmp", model_dir="/tmp",
                             loss_type=LossType.FOCAL_TVERSKY)
        loss = _build_loss(cfg)
        assert isinstance(loss, FocalTverskyLoss)
        assert abs(loss.gamma - 4.0 / 3.0) < 1e-9

    def test_tversky_loss_type_builds_with_gamma_one(self):
        from deeplogger.config import LossType, TrainingConfig
        from deeplogger.train import _build_loss
        cfg = TrainingConfig(data_dir="/tmp", model_dir="/tmp",
                             loss_type=LossType.TVERSKY)
        loss = _build_loss(cfg)
        assert isinstance(loss, FocalTverskyLoss)
        assert loss.gamma == 1.0


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
