"""
Custom loss functions for GravLensAI models.

Provides:
  - Weighted BCE for classification (handles class imbalance)
  - Huber loss for robust regression
  - Per-parameter weighted MSE for regression
  - Combined task + adaptation loss
"""

import torch
import torch.nn as nn
from typing import Optional, cast


class WeightedBCELoss(nn.Module):
    """
    Binary cross-entropy with positive class weighting.
    Wraps BCEWithLogitsLoss for convenience.

    Args:
        pos_weight: Weight for positive (lens) class. Higher = more recall.
    """

    def __init__(self, pos_weight: float = 3.0):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight])
        )

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Move pos_weight to same device as logits
        pos_weight = self.criterion.pos_weight
        if pos_weight is not None and pos_weight.device != logits.device:
            self.criterion.pos_weight = pos_weight.to(logits.device)
        return self.criterion(logits, targets)


class PerParameterLoss(nn.Module):
    """
    Weighted MSE/Huber loss with different weights per parameter.

    Useful when parameters have different scales or importance.
    E.g., Einstein radius errors matter more than shear errors.

    Args:
        weights: (5,) tensor of per-parameter weights.
        loss_type: 'mse' or 'huber'.
        huber_delta: Delta parameter for Huber loss.
    """

    def __init__(
        self,
        weights: Optional[torch.Tensor] = None,
        loss_type: str = 'huber',
        huber_delta: float = 0.1,
    ):
        super().__init__()
        if weights is None:
            # Default: weight Einstein radius higher
            weights = torch.tensor([2.0, 1.0, 1.0, 0.5, 0.5])
        self.register_buffer('weights', weights)
        self.loss_type = loss_type
        self.huber_delta = huber_delta

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, 5) predicted parameters.
            target: (B, 5) ground truth parameters.

        Returns:
            Scalar weighted loss.
        """
        if self.loss_type == 'mse':
            per_param = (pred - target).pow(2).mean(0)  # (5,)
        elif self.loss_type == 'huber':
            diff = (pred - target).abs()
            per_param = torch.where(
                diff < self.huber_delta,
                0.5 * diff.pow(2),
                self.huber_delta * (diff - 0.5 * self.huber_delta),
            ).mean(0)  # (5,)
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        weights = cast(torch.Tensor, self.weights)
        return (per_param * weights).mean()


class CombinedLoss(nn.Module):
    """
    Combined task loss + domain adaptation loss.

    total = task_loss + lambda_adapt * adapt_loss

    Args:
        task_loss: Primary task loss function.
        lambda_adapt: Weight for adaptation loss.
    """

    def __init__(self, task_loss: nn.Module, lambda_adapt: float = 0.1):
        super().__init__()
        self.task_loss = task_loss
        self.lambda_adapt = lambda_adapt

    def forward(
        self,
        task_pred: torch.Tensor,
        task_target: torch.Tensor,
        adapt_loss: torch.Tensor,
    ) -> torch.Tensor:
        return self.task_loss(task_pred, task_target) + self.lambda_adapt * adapt_loss
