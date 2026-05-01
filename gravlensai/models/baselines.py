"""Baseline models for research comparisons."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class LightweightLensCNN(nn.Module):
    """Small CNN baseline for fair architecture-size comparison."""

    def __init__(self, dropout: float = 0.2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


def train_logistic_regression_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
):
    """Train a logistic-regression baseline on flattened images."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    clf = Pipeline([
        ("scaler", StandardScaler()),
        (
            "model",
            LogisticRegression(
                max_iter=2000,
                random_state=seed,
                class_weight="balanced",
                n_jobs=None,
            ),
        ),
    ])
    clf.fit(x_train, y_train)
    return clf


def train_random_forest_baseline(
    x_train: np.ndarray,
    y_train: np.ndarray,
    seed: int = 42,
):
    """Train a random-forest baseline on flattened images."""
    from sklearn.ensemble import RandomForestClassifier

    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced_subsample",
    )
    rf.fit(x_train, y_train)
    return rf
