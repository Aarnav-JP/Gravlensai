"""
Train the XGBoost Meta-Ensemble.

This script:
1. Trains the classical ML baselines (Random Forest, Logistic Regression) and Lightweight CNN on the training set.
2. Loads the fine-tuned ResNet-18 classifier.
3. Generates out-of-fold predictions on the validation set for all 4 models.
4. Trains an XGBoost meta-classifier to optimally combine these predictions.
5. Evaluates the final ensemble.
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

from gravlensai.data.dataset import SimulatedLensDataset
from gravlensai.evaluate.metrics import classifier_metrics
from gravlensai.models.baselines import (
    LightweightLensCNN,
    train_logistic_regression_baseline,
    train_random_forest_baseline,
)
from gravlensai.models.classifier import LensClassifier
from gravlensai.models.ensemble import EnsembleClassifier
from gravlensai.utils.reproducibility import set_global_seed
import importlib.util

def _load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

research_suite_path = Path(__file__).parent / "09_research_suite.py"
research_suite = _load_module("research_suite", str(research_suite_path))
_collect_classifier_outputs = research_suite._collect_classifier_outputs
_flatten_images = research_suite._flatten_images


def train_lightweight_cnn(train_ds, val_loader, device, epochs=2, lr=5e-4):
    """Train the lightweight CNN."""
    print("Training Lightweight CNN...")
    model = LightweightLensCNN().to(device)
    optimiser = Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    x_train = torch.from_numpy(train_ds.images).unsqueeze(1).float()
    y_train = torch.from_numpy(train_ds.targets).float()
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=64,
        shuffle=True,
    )

    for _ in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimiser.step()

    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser("Train XGBoost Ensemble")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resnet_ckpt", default="results/models/classifier_best.pt")
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    # 1. Load Data
    print("Loading datasets...")
    train_ds = SimulatedLensDataset("data/simulated/", split="train", task="classify", augment=False, split_seed=args.seed)
    val_ds = SimulatedLensDataset("data/simulated/", split="val", task="classify", augment=False, split_seed=args.seed)

    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)
    # Subsample training data to prevent OOM kills on macOS
    print("Subsampling data for baselines to prevent memory issues...")
    indices = np.random.choice(len(train_ds), size=10000, replace=False)
    sub_images = train_ds.images[indices]
    sub_targets = train_ds.targets[indices]

    x_train_flat = _flatten_images(sub_images)
    y_train = sub_targets.astype(np.int64)

    x_val_flat = _flatten_images(val_ds.images)
    y_val = val_ds.targets.astype(np.int64)

    # 2. Load ResNet-18
    print("Loading ResNet-18...")
    resnet = LensClassifier()
    ckpt = torch.load(args.resnet_ckpt, map_location=device, weights_only=False)
    resnet.load_state_dict(ckpt["model_state"])
    resnet.to(device).eval()

    # 3. Train Baselines
    print("Training Random Forest...")
    rf_model = train_random_forest_baseline(x_train_flat, y_train, seed=args.seed)

    print("Training Logistic Regression...")
    lr_model = train_logistic_regression_baseline(x_train_flat, y_train, seed=args.seed)

    lightweight_model = train_lightweight_cnn(train_ds, val_loader, device)

    # 4. Generate Validation Predictions (Level-1 Features)
    print("Generating out-of-fold predictions for meta-classifier...")
    _, p_resnet = _collect_classifier_outputs(resnet, val_loader, device)
    _, p_light = _collect_classifier_outputs(lightweight_model, val_loader, device)
    p_rf = rf_model.predict_proba(x_val_flat)[:, 1]
    p_lr = lr_model.predict_proba(x_val_flat)[:, 1]

    # Feature matrix: (N_val, 4)
    X_meta = np.column_stack((p_resnet, p_light, p_rf, p_lr))

    # 5. Train XGBoost Meta-Classifier
    print("Training Meta-Classifier...")
    meta_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.05,
        random_state=args.seed,
        eval_metric="auc",
        use_label_encoder=False
    )
    meta_model.fit(X_meta, y_val)

    # 6. Evaluate
    p_ensemble = meta_model.predict_proba(X_meta)[:, 1]
    metrics = classifier_metrics(y_val, p_ensemble)

    print("\n" + "="*50)
    print("Ensemble Evaluation (Validation Set)")
    print("="*50)
    print(f"  ResNet-18 standalone AUC:  {roc_auc_score(y_val, p_resnet):.5f}")
    print(f"  Ensemble AUC:              {metrics.get('auc_roc', 0.0):.5f}")
    print(f"  Ensemble F1:               {metrics.get('f1', 0.0):.5f}")
    print("="*50)

    # 7. Save Ensemble
    ensemble = EnsembleClassifier(
        resnet_model=resnet,
        lightweight_model=lightweight_model,
        rf_model=rf_model,
        lr_model=lr_model,
        meta_model=meta_model,
        device=device
    )

    out_path = "results/models/ensemble_xgb.json"
    ensemble.save(out_path)

    # Also save the classical models so they can be loaded later
    import joblib
    joblib.dump(rf_model, "results/models/rf_baseline.joblib")
    joblib.dump(lr_model, "results/models/lr_baseline.joblib")
    torch.save(lightweight_model.state_dict(), "results/models/lightweight_cnn.pt")

    print(f"\nSaved XGBoost model to {out_path}")
    print("Saved baseline components to results/models/")


if __name__ == "__main__":
    main()
