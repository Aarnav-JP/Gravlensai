"""
Train the lens parameter regressor.
Only runs on lensed images (positive class).

Usage:
    python scripts/04_train_regressor.py \\
        --data_dir data/simulated/ \\
        --output results/models/ \\
        --epochs 100

Expected results:
    - Einstein radius RMSE: ~0.05–0.10 arcsec
    - Ellipticity RMSE:     ~0.02–0.05
    Training time: ~3h on T4 for 30k lensed images / 100 epochs
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os

from gravlensai.data.dataset import SimulatedLensDataset
from gravlensai.models.regressor import LensParameterRegressor
from gravlensai.train.callbacks import EarlyStopping, ModelCheckpoint
from gravlensai.utils.reproducibility import set_global_seed
from gravlensai.utils.config import load_yaml_section

PARAM_NAMES = ['Einstein_R', 'e1', 'e2', 'gamma1', 'gamma2', 'M_sub']


def train_epoch(model, loader, optimiser, criterion, device):
    """Run one training epoch."""
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
def val_epoch(model, loader, criterion, device):
    """Run one validation epoch."""
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


def main(args):
    cfg = load_yaml_section(args.config, 'regressor')
    training_cfg = cfg.get('training', {})
    model_cfg = cfg.get('model', {})
    loss_cfg = cfg.get('loss', {})

    epochs = args.epochs if args.epochs is not None else int(training_cfg.get('epochs', 100))
    batch_size = args.batch_size if args.batch_size is not None else int(training_cfg.get('batch_size', 64))
    lr = args.lr if args.lr is not None else float(training_cfg.get('learning_rate', 1e-4))
    seed = args.seed if args.seed is not None else 42
    dropout = args.dropout if args.dropout is not None else float(model_cfg.get('dropout', 0.2))
    huber_delta = args.huber_delta if args.huber_delta is not None else float(loss_cfg.get('delta', 0.1))

    set_global_seed(seed)

    # Device
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )
    print(f"Training on: {device}")
    print(f"Global seed: {seed}")

    # Data (regression = lensed images only)
    train_ds = SimulatedLensDataset(
        args.data_dir, split='train', task='regress', split_seed=seed
    )
    val_ds = SimulatedLensDataset(
        args.data_dir, split='val', task='regress', augment=False, split_seed=seed
    )

    nw = 0 if device.type == 'mps' else min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=nw)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Model
    model = LensParameterRegressor(dropout=dropout).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    # Huber loss (robust to outliers vs MSE)
    criterion = nn.HuberLoss(delta=huber_delta)
    optimiser = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimiser, patience=10, factor=0.5, min_lr=1e-7)

    # Callbacks
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(out / "regressor_best.pt", mode='min')
    early_stop = EarlyStopping(patience=20, mode='min')

    # Optional TensorBoard
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(out / "tensorboard_regressor")
    except ImportError:
        pass

    # Training loop
    for epoch in range(1, epochs + 1):
        train_loss, train_rmse = train_epoch(model, train_loader, optimiser, criterion, device)
        val_loss, val_rmse = val_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        # Print progress
        rmse_str = " ".join(f"{n}={v:.4f}" for n, v in zip(PARAM_NAMES, val_rmse))
        print(f"Epoch {epoch:04d} | Train loss {train_loss:.6f} | "
              f"Val loss {val_loss:.6f} | RMSE: {rmse_str}")

        # TensorBoard logging
        if writer:
            writer.add_scalar("train/loss", train_loss, epoch)
            writer.add_scalar("val/loss", val_loss, epoch)
            for name, rmse_val in zip(PARAM_NAMES, val_rmse):
                writer.add_scalar(f"val/rmse_{name}", rmse_val.item(), epoch)

        # Checkpointing (save on best val loss)
        val_metrics = {'loss': val_loss, 'rmse': {n: v.item() for n, v in zip(PARAM_NAMES, val_rmse)}}
        saved = checkpoint(val_loss, model, optimiser, epoch, val_metrics)
        if saved:
            print(f"  ↳ Saved best checkpoint (val_loss={val_loss:.6f})")

        # Early stopping
        if early_stop(val_loss):
            print(f"\nEarly stopping at epoch {epoch}")
            break

    if writer:
        writer.close()

    print(f"\nBest val loss: {checkpoint.best_value:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train lens parameter regressor")
    parser.add_argument("--config", default="configs/regressor.yaml")
    parser.add_argument("--data_dir", default="data/simulated/")
    parser.add_argument("--output", default="results/models/")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--huber_delta", type=float, default=None)
    main(parser.parse_args())
