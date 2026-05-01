"""
Training callbacks: early stopping, model checkpointing, metric logging.
"""

import numpy as np
import torch
from pathlib import Path
from typing import Optional


class EarlyStopping:
    """
    Early stopping to terminate training when validation metric stops improving.

    Args:
        patience: Number of epochs to wait after last improvement.
        min_delta: Minimum change to qualify as an improvement.
        mode: 'min' (for loss) or 'max' (for AUC/accuracy).
    """

    def __init__(self, patience: int = 10, min_delta: float = 1e-4,
                 mode: str = 'max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = -np.inf if mode == 'max' else np.inf
        self.should_stop = False

    def __call__(self, value: float) -> bool:
        """
        Check if training should stop.

        Args:
            value: Current metric value.

        Returns:
            True if training should stop.
        """
        if self.mode == 'max':
            improved = value > self.best_value + self.min_delta
        else:
            improved = value < self.best_value - self.min_delta

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class ModelCheckpoint:
    """
    Save model checkpoint when validation metric improves.

    Args:
        save_path: Path to save checkpoint file.
        mode: 'min' or 'max'.
    """

    def __init__(self, save_path: str, mode: str = 'max'):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.best_value = -np.inf if mode == 'max' else np.inf

    def __call__(self, value: float, model: torch.nn.Module,
                 optimiser: torch.optim.Optimizer,
                 epoch: int, metrics: dict) -> bool:
        """
        Save checkpoint if metric improved.

        Returns:
            True if checkpoint was saved.
        """
        if self.mode == 'max':
            improved = value > self.best_value
        else:
            improved = value < self.best_value

        if improved:
            self.best_value = value
            torch.save({
                'epoch': epoch,
                'model_state': model.state_dict(),
                'optimiser_state': optimiser.state_dict(),
                'val_metrics': metrics,
                'best_value': self.best_value,
            }, self.save_path)
            return True

        return False
