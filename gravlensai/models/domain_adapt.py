"""
Domain Adaptation: closing the sim-to-real gap.

Implements:
1. Maximum Mean Discrepancy (MMD) loss with multi-kernel RBF
   - Aligns feature distributions between simulated and real HST images
   - Unsupervised: no labels needed for target (real) domain

2. DANN — Domain Adversarial Neural Network (Ganin et al. 2016)
   - Gradient reversal layer makes backbone features domain-invariant
   - Domain classifier tries to distinguish sim vs. real
   - Backbone learns to fool the domain classifier

Why this matters: Classifier trained on GalSim images encounters different
noise, PSF, and morphologies in real HST data. Without adaptation, precision
drops ~15–30%.

Reference: Ganin et al. 2016 — "Domain-Adversarial Training of Neural Networks" (JMLR)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional, Protocol


class FeatureExtractorClassifier(Protocol):
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        ...

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        ...


# ── MMD Loss ─────────────────────────────────────────────────────────────

def mmd_loss(
    source_features: torch.Tensor,
    target_features: torch.Tensor,
    kernel_bandwidths: List[float] = [1.0, 2.0, 4.0, 8.0],
) -> torch.Tensor:
    """
    Maximum Mean Discrepancy loss using multi-kernel RBF.

    Measures distance between feature distributions of source (sim) and
    target (real HST) domains. Minimising this during fine-tuning aligns
    the feature distributions.

    Args:
        source_features: (B_s, D) features from simulated images.
        target_features: (B_t, D) features from real HST images (no labels needed).
        kernel_bandwidths: RBF bandwidth values for multi-kernel MMD.

    Returns:
        Scalar MMD loss (non-negative).
    """
    def rbf_kernel(x, y, bandwidth):
        # (B_x, 1, D) - (1, B_y, D) -> (B_x, B_y)
        xx = (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(2)
        return torch.exp(-xx / (2 * bandwidth ** 2))

    loss = torch.tensor(0.0, device=source_features.device)
    for bw in kernel_bandwidths:
        K_ss = rbf_kernel(source_features, source_features, bw).mean()
        K_tt = rbf_kernel(target_features, target_features, bw).mean()
        K_st = rbf_kernel(source_features, target_features, bw).mean()
        loss = loss + K_ss + K_tt - 2 * K_st

    return loss


# ── Gradient Reversal (for DANN) ─────────────────────────────────────────

class GradientReversalFunction(torch.autograd.Function):
    """
    Gradient reversal layer for DANN.
    Forward: identity. Backward: negate and scale gradients by alpha.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(torch.tensor(alpha, dtype=x.dtype, device=x.device))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        alpha, = ctx.saved_tensors
        return -alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    """Wrapper module for gradient reversal."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)

    def set_alpha(self, alpha: float):
        """Update reversal strength (for annealing schedule)."""
        self.alpha = alpha


# ── Domain Classifier (for DANN) ─────────────────────────────────────────

class DomainClassifier(nn.Module):
    """
    Adversarial domain discriminator for DANN.
    Predicts whether input features come from sim (0) or real (1).
    """

    def __init__(self, in_features: int = 512, hidden: int = 256):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, features: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        Args:
            features: (B, D) feature tensor from backbone.
            alpha: Gradient reversal strength.

        Returns:
            (B,) domain logits.
        """
        self.grl.set_alpha(alpha)
        reversed_features = self.grl(features)
        return self.net(reversed_features).squeeze(1)


# ── Domain Adaptation Training Utilities ─────────────────────────────────

def compute_dann_alpha(epoch: int, max_epochs: int) -> float:
    """
    Gradual annealing schedule for DANN gradient reversal strength.
    Starts at 0, increases to 1 over training.

    From Ganin et al. 2016: α = 2/(1+exp(-10p)) - 1
    where p = epoch/max_epochs.
    """
    p = epoch / max_epochs
    return float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)


def domain_adaptation_step(
    classifier: FeatureExtractorClassifier,
    sim_images: torch.Tensor,
    real_images: torch.Tensor,
    sim_labels: torch.Tensor,
    task_criterion: nn.Module,
    method: str = 'mmd',
    lambda_adapt: float = 0.1,
    domain_classifier: Optional[DomainClassifier] = None,
    alpha: float = 1.0,
) -> dict:
    """
    One training step combining task loss and domain adaptation loss.

    Args:
        classifier: Lens classifier model (must have extract_features method).
        sim_images: (B, 1, 64, 64) simulated images with labels.
        real_images: (B, 1, 64, 64) real HST images (no labels needed).
        sim_labels: (B,) binary labels for simulated images.
        task_criterion: Loss function for classification (e.g., BCEWithLogitsLoss).
        method: 'mmd' or 'dann'.
        lambda_adapt: Weight of adaptation loss relative to task loss.
        domain_classifier: Required if method='dann'.
        alpha: DANN gradient reversal strength.

    Returns:
        Dict with 'total_loss', 'task_loss', 'adapt_loss' tensors.
    """
    # Task loss on labelled simulated data
    sim_logits = classifier(sim_images)
    task_loss = task_criterion(sim_logits, sim_labels)

    # Extract features for adaptation
    sim_features = classifier.extract_features(sim_images)
    real_features = classifier.extract_features(real_images)

    if method == 'mmd':
        adapt_loss = mmd_loss(sim_features, real_features)
    elif method == 'dann':
        assert domain_classifier is not None, "domain_classifier required for DANN"
        # Domain labels: sim=0, real=1
        domain_labels = torch.cat([
            torch.zeros(len(sim_features)),
            torch.ones(len(real_features)),
        ]).to(sim_features.device)

        all_features = torch.cat([sim_features, real_features], dim=0)
        domain_logits = domain_classifier(all_features, alpha=alpha)
        adapt_loss = nn.BCEWithLogitsLoss()(domain_logits, domain_labels)
    else:
        raise ValueError(f"Unknown method: {method}")

    total_loss = task_loss + lambda_adapt * adapt_loss

    return {
        'total_loss': total_loss,
        'task_loss': task_loss,
        'adapt_loss': adapt_loss,
    }
