"""Dataclasses for DeepLogger configuration.

These bundle parameters that always travel together, reducing argument sprawl
and preventing bugs from wrong argument order or missing parameters.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class DataType(Enum):
    """Borehole data type."""
    ATV = "atv"
    OTV = "otv"


class LossType(Enum):
    """Available loss functions for training."""
    BCE = "bce"
    DICE = "dice"
    BCE_DICE = "bce_dice"
    BCE_LOGITS = "bce_logits"
    TVERSKY = "tversky"
    FOCAL_TVERSKY = "focal_tversky"


class OptimizerType(Enum):
    """Available optimizers."""
    ADAM = "adam"
    SGD = "sgd"


class ModelType(Enum):
    """Available U-Net model architectures."""
    UNET_ATV_V1 = "unet_atv_v1"          # original Perritaz architecture (model_architectures_ATV)
    UNET_ATV_V2 = "unet_atv_v2"          # Perritaz + per-sample normalisation (model_architectures_v2)
    ATTENTION_UNET_ATV = "attention_unet_atv"  # attention gates + dropout (model_architectures_v2)
    ATTENTION_ONLY_UNET_ATV = "attention_only_unet_atv"  # v2 + gates only, no dropout/pool change (clean ablation)
    UNET_OTV = "unet_otv"                 # original 3-channel OTV architecture (model_architectures_OTV)


@dataclass
class Borehole:
    """Physical and data parameters for a borehole.

    Attributes:
        name: borehole identifier (e.g. 'CB1', 'MB5')
        diameter: borehole diameter in meters
        data_path: path to the directory containing LAS files
        data_type: ATV (acoustic, single-channel) or OTV (optical, 3-channel)
        azimuth_values: number of azimuth samples per row (typically 360)
    """
    name: str
    diameter: float
    data_path: str
    data_type: DataType
    azimuth_values: int = 360


@dataclass
class Fracture:
    """Parameters describing a single fracture pick.

    All angles in degrees, depth in meters, aperture in millimeters.
    These are the values read from label/pick files.

    Attributes:
        azimuth: fracture azimuth in degrees
        dip: fracture dip angle in degrees
        depth: mean depth of the fracture in meters
        aperture: fracture aperture/width in millimeters
        azimuth_correction: additive correction to azimuth (degrees)
        dip_correction: additive correction to dip (degrees)
    """
    azimuth: float
    dip: float
    depth: float
    aperture: float
    azimuth_correction: float = 0.0
    dip_correction: float = 0.0

    @property
    def aperture_m(self) -> float:
        """Aperture converted to meters."""
        return self.aperture / 1000.0

    def to_array(self) -> np.ndarray:
        """Convert to numpy array [depth, azimuth, dip, aperture]."""
        return np.array([self.depth, self.azimuth, self.dip, self.aperture])


@dataclass
class TrainingConfig:
    """Configuration for a training run.

    Captures all hyperparameters so that training is reproducible from
    a single config object. Can be serialized to/from dict for saving
    alongside model checkpoints.

    Attributes:
        data_dir: path to training data (.pt files)
        model_dir: path to save trained models and configs
        data_type: ATV or OTV (determines model input channels)
        loss_type: which loss function to use
        optimizer_type: which optimizer to use
        max_epochs: number of training epochs
        batch_size: training batch size
        batch_size_val: validation batch size
        learning_rate: initial learning rate
        lr_step_size: epochs between LR decay steps
        lr_gamma: multiplicative LR decay factor
        momentum: SGD momentum (ignored for Adam)
        seed: random seed for reproducibility
        validate_every: run validation every N epochs
        test_fraction: fraction of data held out for testing
        val_fraction: fraction of remaining data used for validation
        augment: whether to apply data augmentation (random flips)
        init_features: initial feature count for U-Net
        model_name: optional name for saved model files
        test_borehole: hold out all snippets from this borehole as the test set
            (borehole-stratified split); overrides test_fraction when set
        metadata_csv: path to the id->borehole metadata CSV (required when
            test_borehole is set)
    """
    data_dir: str
    model_dir: str
    data_type: DataType = DataType.ATV
    model_type: ModelType = ModelType.UNET_ATV_V2
    loss_type: LossType = LossType.BCE
    optimizer_type: OptimizerType = OptimizerType.ADAM
    max_epochs: int = 600
    batch_size: int = 20
    batch_size_val: int = 16
    learning_rate: float = 0.001
    lr_step_size: int = 50
    lr_gamma: float = 0.5
    momentum: float = 0.9
    seed: int = 100
    validate_every: int = 15
    test_fraction: float = 0.1
    val_fraction: float = 0.2
    augment: bool = True
    init_features: int = 32
    model_name: Optional[str] = None
    test_borehole: Optional[str] = None
    metadata_csv: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to a serializable dictionary."""
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Enum):
                d[k] = v.value
            else:
                d[k] = v
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingConfig":
        """Create from a dictionary (e.g. loaded from pickle/json)."""
        d = d.copy()
        if "data_type" in d and isinstance(d["data_type"], str):
            d["data_type"] = DataType(d["data_type"])
        if "loss_type" in d and isinstance(d["loss_type"], str):
            d["loss_type"] = LossType(d["loss_type"])
        if "optimizer_type" in d and isinstance(d["optimizer_type"], str):
            d["optimizer_type"] = OptimizerType(d["optimizer_type"])
        if "model_type" in d and isinstance(d["model_type"], str):
            d["model_type"] = ModelType(d["model_type"])
        # Filter out keys that aren't TrainingConfig fields
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        d = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**d)
