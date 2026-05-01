"""
Train the lens classifier.

Usage:
    python scripts/03_train_classifier.py \\
        --data_dir data/simulated/ \\
        --output results/models/ \\
        --epochs 50 \\
        --batch_size 64 \\
        --lr 1e-4

Expected results:
    - Val precision: ~92–95%
    - Val recall:    ~88–93%
    - Val AUC-ROC:   ~0.97–0.99
    Training time: ~2h on T4 GPU for 60k images / 50 epochs
"""

import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os

from gravlensai.data.dataset import SimulatedLensDataset
from gravlensai.models.classifier import LensClassifier
from gravlensai.train.callbacks import EarlyStopping, ModelCheckpoint
from gravlensai.utils.reproducibility import set_global_seed
from gravlensai.utils.config import load_yaml_section


def safe_auc(y_true, y_scores):
    """Compute AUC-ROC only if both classes are present."""
    from sklearn.metrics import roc_auc_score
    if len(set(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_scores)


def train_epoch(model, loader, optimiser, criterion, device):
    """Run one training epoch."""
    from sklearn.metrics import precision_score, recall_score
    model.train()
    total_loss, all_logits, all_labels = 0, [], []

    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimiser.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        total_loss += loss.item()
        all_logits.extend(logits.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)
    preds = (all_logits > 0).astype(int)

    return {
        'loss': total_loss / len(loader),
        'precision': precision_score(all_labels, preds, zero_division=0),
        'recall': recall_score(all_labels, preds, zero_division=0),
        'auc': safe_auc(all_labels.astype(int), all_logits),
    }


@torch.no_grad()
def val_epoch(model, loader, criterion, device):
    """Run one validation epoch."""
    from sklearn.metrics import precision_score, recall_score
    model.eval()
    total_loss, all_probs, all_labels = 0, [], []

    for images, labels in tqdm(loader, desc="Val", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        all_probs.extend(torch.sigmoid(logits).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds = (all_probs > 0.5).astype(int)

    return {
        'loss': total_loss / len(loader),
        'precision': precision_score(all_labels, preds, zero_division=0),
        'recall': recall_score(all_labels, preds, zero_division=0),
        'auc': safe_auc(all_labels.astype(int), all_probs),
    }


def main(args):
    cfg = load_yaml_section(args.config, 'classifier')
    training_cfg = cfg.get('training', {})
    model_cfg = cfg.get('model', {})

    epochs = args.epochs if args.epochs is not None else int(training_cfg.get('epochs', 50))
    batch_size = args.batch_size if args.batch_size is not None else int(training_cfg.get('batch_size', 64))
    lr = args.lr if args.lr is not None else float(training_cfg.get('learning_rate', 1e-4))
    seed = args.seed if args.seed is not None else 42
    pos_weight_value = (
        args.pos_weight if args.pos_weight is not None else float(training_cfg.get('pos_weight', 3.0))
    )
    dropout = args.dropout if args.dropout is not None else float(model_cfg.get('dropout', 0.3))

    set_global_seed(seed)

    # Device setup
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )
    print(f"Training on: {device}")
    print(f"Global seed: {seed}")

    # Data
    train_ds = SimulatedLensDataset(
        args.data_dir, split='train', task='classify', split_seed=seed
    )
    val_ds = SimulatedLensDataset(
        args.data_dir, split='val', task='classify', augment=False, split_seed=seed
    )

    # Adjust num_workers for MPS/CPU
    nw = 0 if device.type == 'mps' else min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=nw)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    # Model
    model = LensClassifier(dropout=dropout).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {param_count:,}")

    # Weighted BCE: lenses are rarer in real sky, simulate 1:3 imbalance
    pos_weight = torch.tensor([pos_weight_value], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimiser = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimiser, T_max=epochs, eta_min=1e-6)

    # Callbacks
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    checkpoint = ModelCheckpoint(out / "classifier_best.pt", mode='max')
    early_stop = EarlyStopping(patience=15, mode='max')

    # Optional TensorBoard
    writer = None
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(out / "tensorboard_classifier")
    except ImportError:
        pass

    # Training loop
    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(model, train_loader, optimiser, criterion, device)
        val_metrics = val_epoch(model, val_loader, criterion, device)
        scheduler.step()

        # Print progress
        print(f"Epoch {epoch:03d} | "
              f"Train loss {train_metrics['loss']:.4f} AUC {train_metrics['auc']:.4f} | "
              f"Val loss {val_metrics['loss']:.4f} AUC {val_metrics['auc']:.4f} "
              f"P {val_metrics['precision']:.4f} R {val_metrics['recall']:.4f}")

        # TensorBoard logging
        if writer:
            for k, v in train_metrics.items():
                writer.add_scalar(f"train/{k}", v, epoch)
            for k, v in val_metrics.items():
                writer.add_scalar(f"val/{k}", v, epoch)

        # Checkpointing
        saved = checkpoint(val_metrics['auc'], model, optimiser, epoch, val_metrics)
        if saved:
            print(f"  ↳ Saved best checkpoint (AUC={val_metrics['auc']:.4f})")

        # Early stopping
        if early_stop(val_metrics['auc']):
            print(f"\nEarly stopping at epoch {epoch}")
            break

    if writer:
        writer.close()

    print(f"\nBest val AUC: {checkpoint.best_value:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train lens classifier")
    parser.add_argument("--config", default="configs/classifier.yaml")
    parser.add_argument("--data_dir", default="data/simulated/")
    parser.add_argument("--output", default="results/models/")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--pos_weight", type=float, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    main(parser.parse_args())
