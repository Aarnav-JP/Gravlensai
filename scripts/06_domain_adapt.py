"""
Domain adaptation: close the sim-to-real gap between simulated and HST images.

Loads the trained classifier, aligns feature distributions between simulated
and real HST data using MMD or DANN, and saves the adapted model.

Usage:
    # MMD domain adaptation (recommended, simpler)
    python scripts/06_domain_adapt.py --method mmd --epochs 10

    # DANN domain adaptation (stronger, needs more tuning)
    python scripts/06_domain_adapt.py --method dann --epochs 20

    # Custom paths
    python scripts/06_domain_adapt.py --method mmd --epochs 10 \
        --classifier results/models/classifier_best.pt \
        --sim_dir data/simulated/ \
        --hst_dir data/raw/hst/Abell_2744/ \
        --output results/models/classifier_adapted.pt
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import time
import yaml  # type: ignore[import-untyped]
from pathlib import Path
from torch.utils.data import DataLoader
from torch.optim import Adam
from tqdm import tqdm
import os

from gravlensai.data.dataset import SimulatedLensDataset, HSTDataset
from gravlensai.models.classifier import LensClassifier
from gravlensai.models.domain_adapt import (
    domain_adaptation_step,
    compute_dann_alpha,
    DomainClassifier,
)
from gravlensai.utils.reproducibility import set_global_seed


def load_config(config_path: str = 'configs/domain_adapt.yaml') -> dict:
    """Load domain adaptation config."""
    if os.path.exists(config_path):
        with open(config_path) as f:
            return yaml.safe_load(f)['domain_adaptation']
    # Defaults
    return {
        'method': 'mmd',
        'mmd': {'kernel_bandwidths': [1.0, 2.0, 4.0, 8.0], 'lambda_mmd': 0.1},
        'dann': {'hidden_dim': 256, 'lambda_dann': 1.0},
        'fine_tuning': {'epochs': 10, 'learning_rate': 1e-5, 'batch_size': 32},
    }


def run_mmd_adaptation(
    model, sim_loader, hst_loader, device,
    epochs=10, lr=1e-5, lambda_mmd=0.1, pos_weight=3.0,
):
    """
    Run MMD domain adaptation.

    Aligns feature distributions between simulated and real HST images
    while maintaining classification performance on labelled sim data.
    """
    model.train()
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(device)
    )
    optimiser = Adam(model.parameters(), lr=lr)

    history = []

    for epoch in range(1, epochs + 1):
        total_task, total_adapt, n_batches = 0, 0, 0

        sim_iter = iter(sim_loader)
        hst_iter = iter(hst_loader)

        pbar = tqdm(desc=f"Epoch {epoch:02d}", leave=False)
        while True:
            try:
                sim_imgs, sim_labels = next(sim_iter)
                hst_imgs, _ = next(hst_iter)
            except StopIteration:
                break

            sim_imgs = sim_imgs.to(device)
            sim_labels = sim_labels.to(device)
            hst_imgs = hst_imgs.to(device).float()

            optimiser.zero_grad()
            result = domain_adaptation_step(
                model, sim_imgs, hst_imgs, sim_labels,
                criterion, method='mmd', lambda_adapt=lambda_mmd,
            )
            result['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            total_task += result['task_loss'].item()
            total_adapt += result['adapt_loss'].item()
            n_batches += 1
            pbar.update(1)

        pbar.close()

        if n_batches > 0:
            avg_task = total_task / n_batches
            avg_adapt = total_adapt / n_batches
        else:
            avg_task, avg_adapt = 0, 0

        history.append({'epoch': epoch, 'task_loss': avg_task, 'adapt_loss': avg_adapt})
        print(f"Epoch {epoch:02d} | task_loss={avg_task:.4f} | "
              f"mmd_loss={avg_adapt:.4f} | "
              f"total={avg_task + lambda_mmd * avg_adapt:.4f}")

    return history


def run_dann_adaptation(
    model, sim_loader, hst_loader, device,
    epochs=20, lr=1e-5, lambda_dann=1.0, pos_weight=3.0,
):
    """
    Run DANN domain adaptation with gradient reversal.

    Uses an adversarial domain classifier to make features domain-invariant.
    """
    model.train()
    domain_clf = DomainClassifier(in_features=512, hidden=256).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight]).to(device)
    )

    # Separate optimisers for model and domain classifier
    optimiser = Adam(
        list(model.parameters()) + list(domain_clf.parameters()),
        lr=lr,
    )

    history = []

    for epoch in range(1, epochs + 1):
        alpha = compute_dann_alpha(epoch, epochs)
        total_task, total_adapt, n_batches = 0, 0, 0

        sim_iter = iter(sim_loader)
        hst_iter = iter(hst_loader)

        pbar = tqdm(desc=f"Epoch {epoch:02d} (α={alpha:.3f})", leave=False)
        while True:
            try:
                sim_imgs, sim_labels = next(sim_iter)
                hst_imgs, _ = next(hst_iter)
            except StopIteration:
                break

            sim_imgs = sim_imgs.to(device)
            sim_labels = sim_labels.to(device)
            hst_imgs = hst_imgs.to(device).float()

            optimiser.zero_grad()
            result = domain_adaptation_step(
                model, sim_imgs, hst_imgs, sim_labels,
                criterion, method='dann', lambda_adapt=lambda_dann,
                domain_classifier=domain_clf, alpha=alpha,
            )
            result['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

            total_task += result['task_loss'].item()
            total_adapt += result['adapt_loss'].item()
            n_batches += 1
            pbar.update(1)

        pbar.close()

        if n_batches > 0:
            avg_task = total_task / n_batches
            avg_adapt = total_adapt / n_batches
        else:
            avg_task, avg_adapt = 0, 0

        history.append({
            'epoch': epoch, 'task_loss': avg_task,
            'adapt_loss': avg_adapt, 'alpha': alpha,
        })
        print(f"Epoch {epoch:02d} | task={avg_task:.4f} | "
              f"dann={avg_adapt:.4f} | α={alpha:.3f}")

    return history


def main(args):
    seed = args.seed

    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )
    print(f"Device: {device}")

    # Load config
    config = load_config(args.config)
    method = args.method or config.get('method', 'mmd')
    epochs = args.epochs or config['fine_tuning']['epochs']
    lr = args.lr or config['fine_tuning']['learning_rate']
    batch_size = args.batch_size or config['fine_tuning']['batch_size']
    if seed is None:
        seed = int(config.get('fine_tuning', {}).get('seed', 42))

    set_global_seed(seed)

    print(f"Method: {method.upper()} | Epochs: {epochs} | LR: {lr} | Seed: {seed}")

    # ── Load classifier ──────────────────────────────────────
    clf_path = args.classifier
    if not os.path.exists(clf_path):
        print(f"ERROR: Classifier not found at {clf_path}")
        print("Train it first: python scripts/03_train_classifier.py")
        return

    model = LensClassifier()
    ckpt = torch.load(clf_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    print(f"Loaded classifier from {clf_path} (epoch {ckpt.get('epoch', '?')})")

    # ── Load simulated data ──────────────────────────────────
    sim_ds = SimulatedLensDataset(args.sim_dir, split='train', task='classify', split_seed=seed)
    sim_loader = DataLoader(sim_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"Simulated data: {len(sim_ds)} images")

    # ── Load HST data ────────────────────────────────────────
    hst_dir = args.hst_dir
    if not os.path.exists(hst_dir) or len(list(Path(hst_dir).glob("*.npy"))) == 0:
        print(f"ERROR: No HST cutouts found at {hst_dir}")
        print("Run: python scripts/02_download_hst_data.py --extract ...")
        return

    hst_ds = HSTDataset(hst_dir)
    hst_loader = DataLoader(hst_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    print(f"HST data: {len(hst_ds)} cutouts from {hst_dir}")

    # ── Run domain adaptation ────────────────────────────────
    print(f"\n{'='*60}")
    print(f"DOMAIN ADAPTATION ({method.upper()})")
    print(f"{'='*60}")

    t0 = time.time()

    if method == 'mmd':
        lambda_val = config.get('mmd', {}).get('lambda_mmd', 0.1)
        history = run_mmd_adaptation(
            model, sim_loader, hst_loader, device,
            epochs=epochs, lr=lr, lambda_mmd=lambda_val,
        )
    elif method == 'dann':
        lambda_val = config.get('dann', {}).get('lambda_dann', 1.0)
        history = run_dann_adaptation(
            model, sim_loader, hst_loader, device,
            epochs=epochs, lr=lr, lambda_dann=lambda_val,
        )
    else:
        print(f"Unknown method: {method}. Use 'mmd' or 'dann'.")
        return

    elapsed = time.time() - t0

    # ── Save adapted model ───────────────────────────────────
    output_path = args.output
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_state': model.state_dict(),
        'method': method,
        'epochs': epochs,
        'history': history,
        'source_checkpoint': clf_path,
    }, output_path)

    print(f"\n{'='*60}")
    print(f"Domain adaptation complete!")
    print(f"  Method: {method.upper()}")
    print(f"  Epochs: {epochs}")
    print(f"  Time: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"  Saved: {output_path}")
    print(f"{'='*60}")

    # Print final summary
    if history:
        final = history[-1]
        print(f"\n  Final task loss:  {final['task_loss']:.4f}")
        print(f"  Final adapt loss: {final['adapt_loss']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GravLensAI Domain Adaptation")
    parser.add_argument("--method", choices=['mmd', 'dann'], default=None,
                        help="Adaptation method (default: from config)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--classifier", default="results/models/classifier_best.pt",
                        help="Path to trained classifier checkpoint")
    parser.add_argument("--sim_dir", default="data/simulated/",
                        help="Path to simulated data directory")
    parser.add_argument("--hst_dir", default="data/raw/hst/Abell_2744/",
                        help="Path to HST cutout directory")
    parser.add_argument("--output", default="results/models/classifier_adapted.pt",
                        help="Output path for adapted model")
    parser.add_argument("--config", default="configs/domain_adapt.yaml",
                        help="Config file path")
    parser.add_argument("--seed", type=int, default=None)
    main(parser.parse_args())
