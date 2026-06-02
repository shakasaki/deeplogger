"""Tests for deeplogger.config dataclasses."""

import numpy as np

from deeplogger.config import (
    Borehole, Fracture, TrainingConfig,
    DataType, LossType, OptimizerType,
)


def test_fracture_aperture_conversion():
    """Aperture in mm should convert to meters via property."""
    frac = Fracture(azimuth=45.0, dip=60.0, depth=100.0, aperture=5.0)
    assert frac.aperture_m == 0.005


def test_fracture_default_corrections():
    """Corrections should default to 0."""
    frac = Fracture(azimuth=0, dip=0, depth=0, aperture=0)
    assert frac.azimuth_correction == 0.0
    assert frac.dip_correction == 0.0


def test_fracture_to_array():
    """to_array should return [depth, azimuth, dip, aperture]."""
    frac = Fracture(azimuth=90.0, dip=45.0, depth=200.0, aperture=10.0)
    arr = frac.to_array()
    np.testing.assert_array_equal(arr, [200.0, 90.0, 45.0, 10.0])


def test_borehole_defaults():
    """Borehole should have sensible defaults."""
    bh = Borehole(name="CB1", diameter=0.076, data_path="/data/", data_type=DataType.ATV)
    assert bh.azimuth_values == 360


def test_training_config_round_trip():
    """TrainingConfig should survive to_dict -> from_dict round trip."""
    config = TrainingConfig(
        data_dir="/data/train",
        model_dir="/output/models",
        data_type=DataType.OTV,
        loss_type=LossType.BCE_DICE,
        optimizer_type=OptimizerType.SGD,
        max_epochs=100,
    )
    d = config.to_dict()
    restored = TrainingConfig.from_dict(d)
    assert restored.data_type == DataType.OTV
    assert restored.loss_type == LossType.BCE_DICE
    assert restored.optimizer_type == OptimizerType.SGD
    assert restored.max_epochs == 100


def test_training_config_from_dict_ignores_extra_keys():
    """from_dict should silently ignore keys not in the dataclass."""
    d = {
        "data_dir": "/data",
        "model_dir": "/models",
        "training_losses": [0.5, 0.3],  # not a field
        "test_ids": ["a", "b"],  # not a field
    }
    config = TrainingConfig.from_dict(d)
    assert config.data_dir == "/data"
