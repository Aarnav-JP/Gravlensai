"""
Training loop for the lens classifier.
Extracted as a reusable module for use by scripts and notebooks.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from typing import Any

from gravlensai.train.callbacks import EarlyStopping, ModelCheckpoint


def safe_auc(y_true, y_scores):
    """Compute AUC-ROC only if both classes are present."""
    from sklearn.metrics import roc_auc_score
    if len(set(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_scores)


def train_one_epoch(model, loader, optimiser, criterion, device):
    """
    Run one training epoch for the classifier.

    Args:
        model: LensClassifier model.
        loader: Training DataLoader.
        optimiser: Optimizer.
        criterion: Loss function (BCEWithLogitsLoss).
        device: torch device.

    Returns:
        Dict with 'loss', 'precision', 'recall', 'auc'.
    """
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
def validate_one_epoch(model, loader, criterion, device):
    """
    Run one validation epoch for the classifier.

    Args:
        model: LensClassifier model.
        loader: Validation DataLoader.
        criterion: Loss function.
        device: torch device.

    Returns:
        Dict with 'loss', 'precision', 'recall', 'auc'.
    """
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


def train_classifier(
    model,
    train_loader,
    val_loader,
    device,
    epochs: int = 50,
    lr: float = 1e-4,
    pos_weight: float = 3.0,
    patience: int = 15,
    save_path: str = 'results/models/classifier_best.pt',
) -> dict:
    """
    Full training loop for the lens classifier.

    Args:
        model: LensClassifier instance.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        device: torch device.
        epochs: Number of training epochs.
        lr: Learning rate.
        pos_weight: Positive class weight for BCE loss.
        patience: Early stopping patience.
        save_path: Path to save best model checkpoint.

    Returns:
        Dict with training history.
    """
    model.to(device)

    pw = torch.tensor([pos_weight], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimiser = AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimiser, T_max=epochs, eta_min=1e-6)

    checkpoint = ModelCheckpoint(save_path, mode='max')
    early_stop = EarlyStopping(patience=patience, mode='max')

    history: dict[str, Any] = {'train': [], 'val': []}

    for epoch in range(1, epochs + 1):
        train_m = train_one_epoch(model, train_loader, optimiser, criterion, device)
        val_m = validate_one_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history['train'].append(train_m)
        history['val'].append(val_m)

        print(f"Epoch {epoch:03d} | "
              f"Train loss {train_m['loss']:.4f} AUC {train_m['auc']:.4f} | "
              f"Val loss {val_m['loss']:.4f} AUC {val_m['auc']:.4f} "
              f"P {val_m['precision']:.4f} R {val_m['recall']:.4f}")

        saved = checkpoint(val_m['auc'], model, optimiser, epoch, val_m)
        if saved:
            print(f"  ↳ Saved best checkpoint (AUC={val_m['auc']:.4f})")

        if early_stop(val_m['auc']):
            print(f"\nEarly stopping at epoch {epoch}")
            break

    history['best_auc'] = checkpoint.best_value
    return history
