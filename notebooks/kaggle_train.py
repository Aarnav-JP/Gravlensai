"""
GravLensAI — Full Training on Kaggle T4
========================================
Self-contained notebook: classifier + regressor + HST download + domain adaptation.

Setup:
  1. Upload this as a Kaggle notebook
  2. Upload data/simulated/ as a Kaggle dataset
  3. Enable GPU (T4) in notebook settings
  4. Run all cells
"""

# ── Cell 1: Install dependencies ─────────────────────────────────────────
# !pip install -q galsim lenstronomy astropy astroquery scikit-learn scikit-image tqdm corner

# ── Cell 2: Upload check ─────────────────────────────────────────────────
import os, sys, torch
import numpy as np

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

# Check data — update this path based on your Kaggle dataset name
DATA_DIR = "/kaggle/input/gravlensai-simulated/"  # adjust to your dataset path
if not os.path.exists(DATA_DIR):
    # Try common Kaggle naming variations
    for alt in ["/kaggle/input/gravlensai-simulated-data/",
                "/kaggle/input/gravlensaisimulated/"]:
        if os.path.exists(alt):
            DATA_DIR = alt
            break
    else:
        DATA_DIR = "data/simulated/"  # fallback for local

for f in ['images_lens.npy', 'params_lens.npy', 'images_nonlens.npy']:
    path = os.path.join(DATA_DIR, f)
    if os.path.exists(path):
        arr = np.load(path)
        print(f"  ✓ {f}: shape={arr.shape}, dtype={arr.dtype}")
    else:
        print(f"  ✗ {f}: NOT FOUND at {path}")

# Output directory (Kaggle allows writing to /kaggle/working/)
OUTPUT_DIR = "/kaggle/working/results/" if os.path.exists("/kaggle") else "results/"
os.makedirs(f"{OUTPUT_DIR}/models", exist_ok=True)
os.makedirs(f"{OUTPUT_DIR}/figures", exist_ok=True)

# ── Cell 3: Copy source code (paste gravlensai package inline) ───────────
# On Kaggle, you need the gravlensai package available.
# Option A: Upload the whole repo as a dataset
# Option B: Paste the essential code inline (below)

# Source code uploaded as Kaggle dataset "gravlensai-repo"
REPO_DIR = "/kaggle/input/gravlensai-repo/"
if not os.path.exists(REPO_DIR):
    for alt in ["/kaggle/input/gravlensairepo/",
                "/kaggle/input/gravlensai-source/"]:
        if os.path.exists(alt):
            REPO_DIR = alt
            break
    else:
        REPO_DIR = os.path.dirname(os.path.abspath("."))
sys.path.insert(0, REPO_DIR)
print(f"Source code: {REPO_DIR}")

# ── Cell 4: Import modules ───────────────────────────────────────────────

from gravlensai.data.dataset import SimulatedLensDataset
from gravlensai.models.classifier import LensClassifier
from gravlensai.models.regressor import LensParameterRegressor
from gravlensai.train.callbacks import EarlyStopping, ModelCheckpoint
from gravlensai.evaluate.metrics import classifier_metrics, regressor_metrics, print_classifier_report, print_regressor_report
from gravlensai.evaluate.visualise import plot_detection_grid, plot_parameter_recovery, plot_confusion_matrix

print("All imports OK!")


# ══════════════════════════════════════════════════════════════════════════
# PART 1: CLASSIFIER TRAINING (50 epochs)
# ══════════════════════════════════════════════════════════════════════════

import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from tqdm import tqdm
import time


def safe_auc(y_true, y_scores):
    if len(set(y_true)) < 2:
        return 0.5
    return roc_auc_score(y_true, y_scores)


def train_epoch_clf(model, loader, optimiser, criterion, device):
    model.train()
    total_loss, all_logits, all_labels = 0, [], []
    for images, labels in tqdm(loader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimiser.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimiser.step()
        total_loss += loss.item()
        all_logits.extend(logits.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    all_logits, all_labels = np.array(all_logits), np.array(all_labels)
    preds = (all_logits > 0).astype(int)
    return {
        'loss': total_loss / len(loader),
        'precision': precision_score(all_labels, preds, zero_division=0),
        'recall': recall_score(all_labels, preds, zero_division=0),
        'auc': safe_auc(all_labels.astype(int), all_logits),
    }


@torch.no_grad()
def val_epoch_clf(model, loader, criterion, device):
    model.eval()
    total_loss, all_probs, all_labels = 0, [], []
    for images, labels in tqdm(loader, desc="Val", leave=False):
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        all_probs.extend(torch.sigmoid(logits).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    all_probs, all_labels = np.array(all_probs), np.array(all_labels)
    preds = (all_probs > 0.5).astype(int)
    return {
        'loss': total_loss / len(loader),
        'precision': precision_score(all_labels, preds, zero_division=0),
        'recall': recall_score(all_labels, preds, zero_division=0),
        'auc': safe_auc(all_labels.astype(int), all_probs),
    }


# ── Train Classifier ─────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TRAINING CLASSIFIER (ResNet-18)")
print("=" * 60)

train_ds = SimulatedLensDataset(DATA_DIR, split='train', task='classify')
val_ds = SimulatedLensDataset(DATA_DIR, split='val', task='classify', augment=False)

nw = 2 if device.type == 'cuda' else 0
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=nw, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=nw, pin_memory=True)
print(f"Train: {len(train_ds)} | Val: {len(val_ds)} | Device: {device}")

model_clf = LensClassifier(dropout=0.3).to(device)
print(f"Parameters: {sum(p.numel() for p in model_clf.parameters() if p.requires_grad):,}")

pos_weight = torch.tensor([3.0], device=device)
criterion_clf = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimiser_clf = AdamW(model_clf.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler_clf = CosineAnnealingLR(optimiser_clf, T_max=50, eta_min=1e-6)

checkpoint_clf = ModelCheckpoint(f"{OUTPUT_DIR}/models/classifier_best.pt", mode='max')
early_stop_clf = EarlyStopping(patience=15, mode='max')

t_start = time.time()
for epoch in range(1, 51):
    train_m = train_epoch_clf(model_clf, train_loader, optimiser_clf, criterion_clf, device)
    val_m = val_epoch_clf(model_clf, val_loader, criterion_clf, device)
    scheduler_clf.step()

    print(f"Epoch {epoch:03d} | "
          f"Train loss {train_m['loss']:.4f} AUC {train_m['auc']:.4f} | "
          f"Val loss {val_m['loss']:.4f} AUC {val_m['auc']:.4f} "
          f"P {val_m['precision']:.4f} R {val_m['recall']:.4f}")

    saved = checkpoint_clf(val_m['auc'], model_clf, optimiser_clf, epoch, val_m)
    if saved:
        print(f"  ↳ Saved best (AUC={val_m['auc']:.4f})")

    if early_stop_clf(val_m['auc']):
        print(f"\nEarly stopping at epoch {epoch}")
        break

print(f"\nClassifier training done in {(time.time()-t_start)/60:.1f} min")
print(f"Best val AUC: {checkpoint_clf.best_value:.4f}")


# ══════════════════════════════════════════════════════════════════════════
# PART 2: REGRESSOR TRAINING (100 epochs)
# ══════════════════════════════════════════════════════════════════════════

from torch.optim.lr_scheduler import ReduceLROnPlateau

PARAM_NAMES = ['Einstein_R', 'e1', 'e2', 'gamma1', 'gamma2', 'M_sub']


def train_epoch_reg(model, loader, optimiser, criterion, device):
    model.train()
    total_loss, all_pred, all_true = 0, [], []
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
    pred_cat, true_cat = torch.cat(all_pred), torch.cat(all_true)
    rmse = ((pred_cat - true_cat) ** 2).mean(0).sqrt()
    return total_loss / len(loader), rmse


@torch.no_grad()
def val_epoch_reg(model, loader, criterion, device):
    model.eval()
    total_loss, all_pred, all_true = 0, [], []
    for images, params in tqdm(loader, desc="Val", leave=False):
        images, params = images.to(device), params.to(device)
        pred = model(images)
        loss = criterion(pred, params)
        total_loss += loss.item()
        all_pred.append(pred.cpu())
        all_true.append(params.cpu())
    pred_cat, true_cat = torch.cat(all_pred), torch.cat(all_true)
    rmse = ((pred_cat - true_cat) ** 2).mean(0).sqrt()
    return total_loss / len(loader), rmse


# ── Train Regressor ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("TRAINING REGRESSOR (5-block CNN)")
print("=" * 60)

train_ds_reg = SimulatedLensDataset(DATA_DIR, split='train', task='regress')
val_ds_reg = SimulatedLensDataset(DATA_DIR, split='val', task='regress', augment=False)

train_loader_reg = DataLoader(train_ds_reg, batch_size=64, shuffle=True, num_workers=nw, pin_memory=True)
val_loader_reg = DataLoader(val_ds_reg, batch_size=64, shuffle=False, num_workers=nw, pin_memory=True)
print(f"Train: {len(train_ds_reg)} | Val: {len(val_ds_reg)}")

model_reg = LensParameterRegressor(dropout=0.2).to(device)
print(f"Parameters: {sum(p.numel() for p in model_reg.parameters() if p.requires_grad):,}")

criterion_reg = nn.HuberLoss(delta=0.1)
optimiser_reg = AdamW(model_reg.parameters(), lr=1e-4, weight_decay=1e-4)
scheduler_reg = ReduceLROnPlateau(optimiser_reg, patience=10, factor=0.5, min_lr=1e-7)

checkpoint_reg = ModelCheckpoint(f"{OUTPUT_DIR}/models/regressor_best.pt", mode='min')
early_stop_reg = EarlyStopping(patience=20, mode='min')

t_start = time.time()
for epoch in range(1, 101):
    train_loss, train_rmse = train_epoch_reg(model_reg, train_loader_reg, optimiser_reg, criterion_reg, device)
    val_loss, val_rmse = val_epoch_reg(model_reg, val_loader_reg, criterion_reg, device)
    scheduler_reg.step(val_loss)

    rmse_str = " ".join(f"{n}={v:.4f}" for n, v in zip(PARAM_NAMES, val_rmse))
    print(f"Epoch {epoch:04d} | Train {train_loss:.6f} | Val {val_loss:.6f} | {rmse_str}")

    val_metrics = {'loss': val_loss, 'rmse': {n: v.item() for n, v in zip(PARAM_NAMES, val_rmse)}}
    saved = checkpoint_reg(val_loss, model_reg, optimiser_reg, epoch, val_metrics)
    if saved:
        print(f"  ↳ Saved best (val_loss={val_loss:.6f})")

    if early_stop_reg(val_loss):
        print(f"\nEarly stopping at epoch {epoch}")
        break

print(f"\nRegressor training done in {(time.time()-t_start)/60:.1f} min")
print(f"Best val loss: {checkpoint_reg.best_value:.6f}")


# ══════════════════════════════════════════════════════════════════════════
# PART 3: EVALUATION & FIGURES
# ══════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("EVALUATION")
print("=" * 60)

# Reload best checkpoints
model_clf_best = LensClassifier()
ckpt = torch.load(f"{OUTPUT_DIR}/models/classifier_best.pt", map_location=device, weights_only=False)
model_clf_best.load_state_dict(ckpt['model_state'])
model_clf_best.to(device).eval()

model_reg_best = LensParameterRegressor()
ckpt = torch.load(f"{OUTPUT_DIR}/models/regressor_best.pt", map_location=device, weights_only=False)
model_reg_best.load_state_dict(ckpt['model_state'])
model_reg_best.to(device).eval()

# Classifier evaluation
test_ds = SimulatedLensDataset(DATA_DIR, split='test', task='classify', augment=False)
test_loader = DataLoader(test_ds, batch_size=64, num_workers=0)

all_probs, all_labels, all_images = [], [], []
t0 = time.time()
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        probs = model_clf_best.predict_proba(images)
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_images.extend(images.cpu().numpy()[:, 0])
t_clf = time.time() - t0

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)
all_images = np.array(all_images)

clf_metrics = classifier_metrics(all_labels, all_probs)
print_classifier_report(clf_metrics)
print(f"  Inference: {t_clf*1000/len(all_labels):.2f} ms/image")
print(f"  Speedup vs LENSTOOL: {120/(t_clf/len(all_labels)):,.0f}×")

# Figures
plot_detection_grid(all_images, all_probs, all_labels, save_path=f"{OUTPUT_DIR}/figures/detection_grid.png")
plot_confusion_matrix(all_labels.astype(int), (all_probs >= 0.5).astype(int), save_path=f"{OUTPUT_DIR}/figures/confusion_matrix.png")

# Regressor evaluation
test_ds_reg = SimulatedLensDataset(DATA_DIR, split='test', task='regress', augment=False)
test_loader_reg = DataLoader(test_ds_reg, batch_size=64, num_workers=0)

all_pred, all_true = [], []
with torch.no_grad():
    for images, params in test_loader_reg:
        images = images.to(device)
        pred = model_reg_best(images)
        all_pred.append(model_reg_best.denormalise(pred).cpu())
        all_true.append(model_reg_best.denormalise(params).cpu())

pred_phys = torch.cat(all_pred).numpy()
true_phys = torch.cat(all_true).numpy()

reg_m = regressor_metrics(true_phys, pred_phys)
print_regressor_report(reg_m)

plot_parameter_recovery(true_phys, pred_phys, save_path=f"{OUTPUT_DIR}/figures/parameter_recovery.png")

print("\n✓ All figures saved to results/figures/")
print("Download them from the 'Output' tab in Kaggle.")
