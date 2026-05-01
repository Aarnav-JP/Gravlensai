"""
XGBoost Meta-Ensemble combining Deep Learning and Classical ML features.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
# NOTE: XGBoost inference was previously run in a subprocess to work around an
# Apple Silicon OpenMP segfault (PyTorch + XGBoost libomp conflict).
# The ensemble is currently disabled in the main app; this module is retained
# for research reproducibility only.




class EnsembleClassifier:
    """
    Wraps multiple base models and uses an XGBoost meta-classifier
    to output the final ensembled probability.
    """

    def __init__(
        self,
        resnet_model: torch.nn.Module,
        lightweight_model: torch.nn.Module,
        rf_model: Any,
        lr_model: Any,
        meta_model: Any = None,
        device: torch.device | None = None,
    ):
        self.resnet = resnet_model
        self.lightweight = lightweight_model
        self.rf = rf_model
        self.lr = lr_model
        self.meta_model_path = None # Store path instead of model to run in subprocess
        
        self.device = device or torch.device("cpu")
        self.resnet.to(self.device).eval()
        self.lightweight.to(self.device).eval()

    def _extract_base_features(self, x: torch.Tensor) -> np.ndarray:
        """
        Pass image through all 4 models and return a (B, 4) feature matrix.
        Args:
            x: (B, 1, 64, 64) Tensor.
        Returns:
            (B, 4) numpy array of probabilities.
        """
        # DL Predictions
        with torch.no_grad():
            p_resnet = torch.sigmoid(self.resnet(x)).cpu().numpy()
            p_light = torch.sigmoid(self.lightweight(x)).cpu().numpy()

        # Classical ML Predictions (requires flattened numpy arrays)
        x_flat = x.cpu().numpy().reshape(x.shape[0], -1)
        p_rf = self.rf.predict_proba(x_flat)[:, 1]
        p_lr = self.lr.predict_proba(x_flat)[:, 1]

        # Stack into (B, 4) feature matrix
        features = np.column_stack((p_resnet, p_light, p_rf, p_lr))
        return features

    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """
        Get ensembled probabilities.
        """
        if self.meta_model_path is None:
            raise ValueError("XGBoost meta-classifier has not been loaded.")
            
        features = self._extract_base_features(x)
        
        try:
            import xgboost as xgb
            m = xgb.XGBClassifier()
            m.load_model(self.meta_model_path)
            prob = m.predict_proba(features)[0, 1]
            return np.array([prob])
        except Exception as e:
            print(f"! XGBoost inference failed: {e}")
            # Fallback to ResNet standalone prediction
            return np.array([features[0][0]])

    def save(self, path: str | Path):
        """Save the XGBoost model to disk."""
        pass # Training is done on Kaggle, we only do inference locally.

    def load(self, path: str | Path):
        """Load the XGBoost model from disk."""
        self.meta_model_path = str(path)
