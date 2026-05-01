"""
Training loop for the lens parameter regressor.
Operates on lensed images only (positive class).
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from typing import Any

from gravlensai.train.callbacks import EarlyStopping, ModelCheckpoint

PARAM_NAMES = ['Einstein_R', 'e1', 'e2', 'gamma1', 'gamma2', 'M_sub']


def train_one_epoch(model, loader, optimiser, criterion, device):
    """
    Run one training epoch for the regressor.

    Args:
        model: LensParameterRegressor model.
        loader: Training DataLoader.
        optimiser: Optimizer.
        criterion: Loss function (HuberLoss).
        device: torch device.

    Returns:
        (loss, per_param_rmse): Average loss and (5,) RMSE tensor.
    """
    model.train()
    total_loss = 0
    all_pred, all_true = [], []

    for images, params in tqdm(loader, desc="Train", leave=False):
        images, params = images.to(device), params.to(device)
        optimiser.zero_grad()
        pred = model(images)
        loss = criterion(pred, params)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        total_loss += loss.item()
        all_pred.append(pred.detach().cpu())
        all_true.append(params.cpu())

    pred_cat = torch.cat(all_pred)
    true_cat = torch.cat(all_true)
    per_param_rmse = ((pred_cat - true_cat) ** 2).mean(0).sqrt()
    return total_loss / len(loader), per_param_rmse


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    """
    Run one validation epoch for the regressor.

    Returns:
        (loss, per_param_rmse): Average loss and (5,) RMSE tensor.
    """
    model.eval()
    total_loss = 0
    all_pred, all_true = [], []

    for images, params in tqdm(loader, desc="Val", leave=False):
        images, params = images.to(device), params.to(device)
        pred = model(images)
        loss = criterion(pred, params)
        total_loss += loss.item()
        all_pred.append(pred.cpu())
        all_true.append(params.cpu())

    pred_cat = torch.cat(all_pred)
    true_cat = torch.cat(all_true)
    per_param_rmse = ((pred_cat - true_cat) ** 2).mean(0).sqrt()
    return total_loss / len(loader), per_param_rmse


def train_regressor(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int = 100,
    lr: float = 1e-4,
    patience: int = 20,
    save_path: str = 'results/models/regressor_best.pt',
) -> dict:
    """
    Full training loop for the lens parameter regressor.

    Args:
        model: LensParameterRegressor instance.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        device: torch device.
        epochs: Number of training epochs.
        lr: Learning rate.
        patience: Early stopping patience.
        save_path: Path to save best model checkpoint.

    Returns:
        Dict with training history.
    """
    model.to(device)

    criterion = nn.HuberLoss(delta=0.1)
    optimiser = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimiser, patience=10, factor=0.5, min_lr=1e-7)

    checkpoint = ModelCheckpoint(save_path, mode='min')
    early_stop = EarlyStopping(patience=patience, mode='min')

    history: dict[str, Any] = {'train_loss': [], 'val_loss': [], 'val_rmse': []}

    for epoch in range(1, epochs + 1):
        train_loss, train_rmse = train_one_epoch(
            model, train_loader, optimiser, criterion, device
        )
        val_loss, val_rmse = validate_one_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append({n: v.item() for n, v in zip(PARAM_NAMES, val_rmse)})

        rmse_str = " ".join(f"{n}={v:.4f}" for n, v in zip(PARAM_NAMES, val_rmse))
        print(f"Epoch {epoch:04d} | Train {train_loss:.6f} | "
              f"Val {val_loss:.6f} | {rmse_str}")

        val_metrics = {
            'loss': val_loss,
            'rmse': {n: v.item() for n, v in zip(PARAM_NAMES, val_rmse)},
        }
        saved = checkpoint(val_loss, model, optimiser, epoch, val_metrics)
        if saved:
            print(f"  ↳ Saved best (val_loss={val_loss:.6f})")

        if early_stop(val_loss):
            print(f"\nEarly stopping at epoch {epoch}")
            break

    history['best_val_loss'] = checkpoint.best_value
    return history
