"""Unified training module for DeepLogger.

Replaces the 6+ near-identical training scripts in model/ with a single
configurable training function driven by TrainingConfig.
"""

import os
import pickle
import random
import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision.transforms import Compose, RandomHorizontalFlip, RandomVerticalFlip

from deeplogger.config import TrainingConfig, DataType, LossType, OptimizerType
from deeplogger.model_architectures_ATV import UNetOTV as UNetATV
from deeplogger.model_architectures_OTV import UNetOTV as UNetOTV
from deeplogger.dataloader import Dataset_np as Dataset
from deeplogger.loss_functions import DiceLoss
from deeplogger.common_helpers import create_directory


def _build_model(config: TrainingConfig, device: torch.device) -> nn.Module:
    """Instantiate the appropriate U-Net model based on data type."""
    if config.data_type == DataType.ATV:
        model = UNetATV(in_channels=1, out_channels=1, init_features=config.init_features)
    else:
        model = UNetOTV(in_channels=3, out_channels=1, init_features=config.init_features)
    return model.to(device).float()


def _build_loss(config: TrainingConfig) -> nn.Module:
    """Instantiate the loss function."""
    if config.loss_type == LossType.BCE:
        return nn.BCELoss()
    elif config.loss_type == LossType.DICE:
        return DiceLoss()
    elif config.loss_type == LossType.BCE_DICE:
        bce = nn.BCELoss()
        dice = DiceLoss()
        class BCEDiceLoss(nn.Module):
            def forward(self, pred, target):
                return 0.5 * bce(pred, target) + dice(pred, target)
        return BCEDiceLoss()
    elif config.loss_type == LossType.BCE_LOGITS:
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss type: {config.loss_type}")


def _build_optimizer(config: TrainingConfig, model: nn.Module) -> optim.Optimizer:
    """Instantiate the optimizer."""
    if config.optimizer_type == OptimizerType.ADAM:
        return optim.Adam(model.parameters(), lr=config.learning_rate)
    elif config.optimizer_type == OptimizerType.SGD:
        return optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum)
    else:
        raise ValueError(f"Unknown optimizer type: {config.optimizer_type}")


def _collect_file_ids(data_dir: str) -> list:
    """Collect all .pt file paths from a directory."""
    file_ids = []
    for path in os.listdir(data_dir):
        full_path = os.path.join(data_dir, path)
        if os.path.isfile(full_path) and path.endswith('.pt'):
            file_ids.append(full_path)
    return sorted(file_ids)


def _binarize_mask(mask: torch.Tensor) -> torch.Tensor:
    """Convert multi-class mask labels to binary (0/1).

    Maps label values: 1->0, 2->1, 3->1.
    This handles the label encoding used in the Bedretto dataset.
    """
    mask = torch.where(mask == 1, 0, mask)
    mask = torch.where(mask == 2, 1, mask)
    mask = torch.where(mask == 3, 1, mask)
    return mask


def train(config: TrainingConfig) -> dict:
    """Run a full training loop.

    Args:
        config: TrainingConfig with all hyperparameters.

    Returns:
        Dictionary with training_losses, validation_losses, and config.
    """
    # Seed for reproducibility
    torch.manual_seed(config.seed)
    random.seed(config.seed)
    np.random.seed(config.seed)

    create_directory(config.model_dir)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model, loss, optimizer
    model = _build_model(config, device)
    loss_fn = _build_loss(config)
    optimizer = _build_optimizer(config, model)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config.lr_step_size, gamma=config.lr_gamma)

    # Data
    all_ids = _collect_file_ids(config.data_dir)
    n_test = int(len(all_ids) * config.test_fraction)
    test_ids = set(random.sample(all_ids, n_test))
    train_val_ids = [f for f in all_ids if f not in test_ids]

    n_val = int(len(train_val_ids) * config.val_fraction)
    n_train = len(train_val_ids) - n_val
    full_dataset = Dataset(train_val_ids)
    train_set, val_set = torch.utils.data.random_split(full_dataset, [n_train, n_val])

    train_loader = data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True, drop_last=True)
    val_loader = data.DataLoader(val_set, batch_size=config.batch_size_val, shuffle=False)

    # Augmentation
    transform = Compose([RandomHorizontalFlip(p=0.5)]) if config.augment else None

    # Model name
    date_str = datetime.date.today().strftime("%Y_%m_%d")
    model_name = config.model_name or f"deeplogger_{config.data_type.value}_{config.loss_type.value}_{date_str}"

    # Training loop
    training_losses = []
    validation_losses = []
    best_val_loss = float('inf')

    for epoch in range(config.max_epochs):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for images, masks in train_loader:
            if config.augment and transform and random.random() > 0.5:
                images = transform(images)
                masks = transform(masks)

            masks = _binarize_mask(masks)
            images = images.to(device).float()
            masks = masks.to(device).float().squeeze(1)

            optimizer.zero_grad()
            preds = model(images)
            loss = loss_fn(preds, masks)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_train_loss = epoch_loss / max(n_batches, 1)
        training_losses.append(avg_train_loss)

        # --- Validate ---
        if epoch % config.validate_every == 0 and epoch > 0:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for images, masks in val_loader:
                    masks = _binarize_mask(masks)
                    images = images.to(device).float()
                    masks = masks.to(device).float().squeeze(1)
                    preds = model(images)
                    loss = loss_fn(preds, masks)
                    val_loss += loss.item()
                    val_batches += 1

            avg_val_loss = val_loss / max(val_batches, 1)
            validation_losses.append(avg_val_loss)
            print(f"Epoch {epoch}: train_loss={avg_train_loss:.4f}, val_loss={avg_val_loss:.4f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                save_path = os.path.join(config.model_dir, f"{model_name}_best.pt")
                torch.save(model.state_dict(), save_path)
                print(f"  Saved best model (val_loss={avg_val_loss:.4f})")

        scheduler.step()

    # Save final config + losses
    results = config.to_dict()
    results['training_losses'] = training_losses
    results['validation_losses'] = validation_losses
    results['test_ids'] = list(test_ids)
    results['best_validation_loss'] = best_val_loss

    config_path = os.path.join(config.model_dir, f"{model_name}_config.p")
    with open(config_path, 'wb') as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")
    return results
