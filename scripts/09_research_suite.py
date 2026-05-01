"""Research suite runner: baselines, multi-cluster ablations, taxonomy, report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from gravlensai.data.dataset import HSTDataset, SimulatedLensDataset
from gravlensai.evaluate.metrics import (
    bootstrap_classifier_ci,
    bootstrap_regressor_ci,
    classifier_metrics,
)
from gravlensai.evaluate.visualise import plot_confusion_matrix
from gravlensai.models.baselines import (
    LightweightLensCNN,
    train_logistic_regression_baseline,
    train_random_forest_baseline,
)
from gravlensai.models.classifier import LensClassifier
from gravlensai.models.domain_adapt import (
    DomainClassifier,
    compute_dann_alpha,
    domain_adaptation_step,
)
from gravlensai.models.regressor import LensParameterRegressor
from gravlensai.utils.config import load_yaml_section
from gravlensai.utils.reproducibility import set_global_seed


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _collect_classifier_outputs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_probs.append(probs)
            all_labels.append(y.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def _collect_regressor_outputs(
    model: LensParameterRegressor,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    pred_list: list[np.ndarray] = []
    true_list: list[np.ndarray] = []
    mins = model.PARAM_MINS.cpu().numpy().astype(np.float32)
    maxs = model.PARAM_MAXS.cpu().numpy().astype(np.float32)

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            pred_norm = model(x).cpu().numpy().astype(np.float32)
            true_norm = y.numpy().astype(np.float32)
            pred = (pred_norm + 1.0) / 2.0 * (maxs - mins) + mins
            true = (true_norm + 1.0) / 2.0 * (maxs - mins) + mins
            pred_list.append(pred)
            true_list.append(true)

    return np.concatenate(true_list), np.concatenate(pred_list)


def _collect_hst_probs(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    probs: list[np.ndarray] = []
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(device)
            logits = model(x)
            probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    if not probs:
        return np.array([], dtype=np.float32)
    return np.concatenate(probs).astype(np.float32)


def _flatten_images(images: np.ndarray) -> np.ndarray:
    return images.reshape(images.shape[0], -1)


def _train_lightweight_baseline(
    train_ds: SimulatedLensDataset,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    batch_size: int,
) -> dict[str, float]:
    model = LightweightLensCNN().to(device)
    optimiser = Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    x_train = torch.from_numpy(train_ds.images).unsqueeze(1).float()
    y_train = torch.from_numpy(train_ds.targets).float()
    train_loader = DataLoader(
        TensorDataset(x_train, y_train),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    for _ in tqdm(range(epochs), desc="Lightweight baseline", leave=False):
        model.train()
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimiser.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimiser.step()

    y_true, y_prob = _collect_classifier_outputs(model, val_loader, device)
    return classifier_metrics(y_true, y_prob)


def _run_baselines(
    cfg: dict[str, Any],
    output_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    train_ds = SimulatedLensDataset(
        cfg["data"]["simulated_dir"],
        split="train",
        task="classify",
        augment=False,
        split_seed=int(cfg["seed"]),
    )
    val_ds = SimulatedLensDataset(
        cfg["data"]["simulated_dir"],
        split="val",
        task="classify",
        augment=False,
        split_seed=int(cfg["seed"]),
    )

    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)
    out: dict[str, Any] = {}

    if cfg["baselines"].get("enable_classical", True):
        x_train_flat = _flatten_images(train_ds.images)
        x_val_flat = _flatten_images(val_ds.images)
        y_train = train_ds.targets.astype(np.int64)

        lr_model = train_logistic_regression_baseline(
            x_train_flat,
            y_train,
            seed=int(cfg["seed"]),
        )
        rf_model = train_random_forest_baseline(
            x_train_flat,
            y_train,
            seed=int(cfg["seed"]),
        )

        lr_prob = lr_model.predict_proba(x_val_flat)[:, 1]
        rf_prob = rf_model.predict_proba(x_val_flat)[:, 1]
        y_val = val_ds.targets.astype(np.int64)

        out["logistic_regression"] = classifier_metrics(y_val, lr_prob)
        out["random_forest"] = classifier_metrics(y_val, rf_prob)

    if cfg["baselines"].get("enable_lightweight_cnn", True):
        out["lightweight_cnn"] = _train_lightweight_baseline(
            train_ds=train_ds,
            val_loader=val_loader,
            device=device,
            epochs=int(cfg["baselines"].get("lightweight_epochs", 5)),
            lr=float(cfg["baselines"].get("lightweight_lr", 5e-4)),
            batch_size=int(cfg["baselines"].get("batch_size", 64)),
        )

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "baselines.json").write_text(json.dumps(out, indent=2))
    return out


def _evaluate_classifier_checkpoint(
    ckpt_path: Path,
    val_loader: DataLoader,
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    model = LensClassifier()
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    y_true, y_prob = _collect_classifier_outputs(model, val_loader, device)
    metrics = classifier_metrics(y_true, y_prob)
    return metrics, y_true, y_prob


def _clone_classifier_with_state(
    state_dict: dict[str, Any],
    device: torch.device,
) -> LensClassifier:
    model = LensClassifier()
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def _cluster_key(cluster_dir: str) -> str:
    return Path(cluster_dir).name or Path(cluster_dir).as_posix().replace("/", "_")


def _summarise_hst_predictions(probs: np.ndarray) -> dict[str, float]:
    if len(probs) == 0:
        return {
            "n": 0.0,
            "mean_probability": 0.0,
            "positive_rate_at_0_5": 0.0,
            "mean_entropy": 0.0,
        }

    p_raw = np.asarray(probs, dtype=np.float64)
    p_finite = p_raw[np.isfinite(p_raw)]
    if p_finite.size == 0:
        return {
            "n": 0.0,
            "mean_probability": 0.0,
            "positive_rate_at_0_5": 0.0,
            "mean_entropy": 0.0,
        }

    # Use wider clipping to avoid float16 underflow on accelerator outputs.
    p = np.clip(p_finite, 1e-6, 1.0 - 1e-6)
    entropy = -(p * np.log2(p) + (1.0 - p) * np.log2(1.0 - p))
    return {
        "n": float(len(p)),
        "mean_probability": float(p.mean()),
        "positive_rate_at_0_5": float((p >= 0.5).mean()),
        "mean_entropy": float(entropy.mean()),
    }


def _safe_mean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0
    return float(arr.mean())


def _resolve_hst_clusters(cfg: dict[str, Any]) -> list[str]:
    explicit = [str(p) for p in cfg["data"].get("hst_clusters", [])]
    if not cfg["data"].get("auto_discover_clusters", True):
        return sorted(set(explicit))

    root = Path(cfg["data"].get("hst_root_dir", "data/raw/hst"))
    discovered: list[str] = []
    if root.exists():
        discovered = [str(p) for p in sorted(root.iterdir()) if p.is_dir()]
    return sorted(set(explicit + discovered))


def _adapt_classifier_to_cluster(
    method: str,
    base_state: dict[str, Any],
    sim_loader: DataLoader,
    hst_loader: DataLoader,
    device: torch.device,
    adaptation_cfg: dict[str, Any],
) -> LensClassifier:
    model = _clone_classifier_with_state(base_state, device)
    if method == "baseline":
        model.eval()
        return model

    lr = float(adaptation_cfg.get("learning_rate", 1e-5))
    epochs = int(adaptation_cfg.get("epochs", 4))
    max_steps = int(adaptation_cfg.get("max_steps_per_epoch", 200))
    pos_weight = float(adaptation_cfg.get("pos_weight", 3.0))
    lambda_mmd = float(adaptation_cfg.get("lambda_mmd", 0.1))
    lambda_dann = float(adaptation_cfg.get("lambda_dann", 1.0))

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device)
    )

    domain_classifier: DomainClassifier | None = None
    parameters: list[nn.Parameter] = list(model.parameters())
    if method == "dann":
        domain_classifier = DomainClassifier(in_features=512, hidden=256).to(device)
        parameters += list(domain_classifier.parameters())

    optimiser = Adam(parameters, lr=lr)
    model.train()

    for epoch in range(1, epochs + 1):
        sim_iter = iter(sim_loader)
        hst_iter = iter(hst_loader)
        step = 0
        alpha = compute_dann_alpha(epoch, epochs) if method == "dann" else 1.0

        while step < max_steps:
            try:
                sim_imgs, sim_labels = next(sim_iter)
            except StopIteration:
                break
            try:
                hst_imgs, _ = next(hst_iter)
            except StopIteration:
                hst_iter = iter(hst_loader)
                hst_imgs, _ = next(hst_iter)

            sim_imgs = sim_imgs.to(device)
            sim_labels = sim_labels.to(device)
            hst_imgs = hst_imgs.to(device).float()

            optimiser.zero_grad()
            result = domain_adaptation_step(
                classifier=model,
                sim_images=sim_imgs,
                real_images=hst_imgs,
                sim_labels=sim_labels,
                task_criterion=criterion,
                method=method,
                lambda_adapt=lambda_mmd if method == "mmd" else lambda_dann,
                domain_classifier=domain_classifier,
                alpha=alpha,
            )
            result["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(parameters, max_norm=1.0)
            optimiser.step()
            step += 1

    model.eval()
    return model


def _run_domain_ablation(
    cfg: dict[str, Any],
    output_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    methods = cfg["domain_ablation"].get("methods", ["baseline", "mmd", "dann"])
    clusters = _resolve_hst_clusters(cfg)
    adaptation_cfg = cfg["domain_ablation"].get("adaptation", {})

    val_ds = SimulatedLensDataset(
        cfg["data"]["simulated_dir"],
        split="val",
        task="classify",
        augment=False,
        split_seed=int(cfg["seed"]),
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    baseline_ckpt = Path(cfg["checkpoints"]["classifier"])
    if not baseline_ckpt.exists():
        raise FileNotFoundError(f"Missing baseline classifier checkpoint: {baseline_ckpt}")

    base_ckpt = torch.load(baseline_ckpt, map_location=device, weights_only=False)
    base_state = base_ckpt["model_state"]

    sim_train_ds = SimulatedLensDataset(
        cfg["data"]["simulated_dir"],
        split="train",
        task="classify",
        augment=True,
        split_seed=int(cfg["seed"]),
    )
    sim_loader = DataLoader(
        sim_train_ds,
        batch_size=int(adaptation_cfg.get("batch_size", 32)),
        shuffle=True,
        num_workers=0,
    )

    ablation: dict[str, Any] = {"clusters": {}, "aggregate": {}}
    aggregate_buckets: dict[str, list[dict[str, float]]] = {m: [] for m in methods}

    for cluster_dir in clusters:
        p = Path(cluster_dir)
        cluster_name = _cluster_key(cluster_dir)
        entry: dict[str, Any] = {"cluster_dir": cluster_dir, "methods": {}}

        if not p.exists():
            entry["error"] = "missing_cluster_directory"
            ablation["clusters"][cluster_name] = entry
            continue

        hst_ds = HSTDataset(str(p))
        if len(hst_ds) == 0:
            entry["error"] = "no_hst_cutouts"
            entry["n_hst_cutouts"] = 0
            ablation["clusters"][cluster_name] = entry
            continue

        hst_loader = DataLoader(
            hst_ds,
            batch_size=int(adaptation_cfg.get("batch_size", 32)),
            shuffle=True,
            num_workers=0,
        )
        entry["n_hst_cutouts"] = len(hst_ds)

        for method in methods:
            adapted_model = _adapt_classifier_to_cluster(
                method=method,
                base_state=base_state,
                sim_loader=sim_loader,
                hst_loader=hst_loader,
                device=device,
                adaptation_cfg=adaptation_cfg,
            )

            y_true, y_prob = _collect_classifier_outputs(adapted_model, val_loader, device)
            metrics = classifier_metrics(y_true, y_prob)
            y_pred = (y_prob >= 0.5).astype(int)
            plot_confusion_matrix(
                y_true,
                y_pred,
                save_path=output_root / f"confusion_{cluster_name}_{method}.png",
            )

            hst_probs = _collect_hst_probs(adapted_model, hst_loader, device)
            hst_summary = _summarise_hst_predictions(hst_probs)
            entry["methods"][method] = {
                "sim_val": metrics,
                "hst_summary": hst_summary,
            }

            aggregate_buckets[method].append(
                {
                    "auc_roc": float(metrics.get("auc_roc", np.nan)),
                    "f1": float(metrics.get("f1", np.nan)),
                    "precision": float(metrics.get("precision", np.nan)),
                    "recall": float(metrics.get("recall", np.nan)),
                    "hst_positive_rate": float(hst_summary["positive_rate_at_0_5"]),
                    "hst_mean_entropy": float(hst_summary["mean_entropy"]),
                }
            )

        ablation["clusters"][cluster_name] = entry

    for method, rows in aggregate_buckets.items():
        if not rows:
            ablation["aggregate"][method] = {"n_clusters": 0}
            continue
        ablation["aggregate"][method] = {
            "n_clusters": len(rows),
            "auc_roc_mean": _safe_mean([r["auc_roc"] for r in rows]),
            "f1_mean": _safe_mean([r["f1"] for r in rows]),
            "precision_mean": _safe_mean([r["precision"] for r in rows]),
            "recall_mean": _safe_mean([r["recall"] for r in rows]),
            "hst_positive_rate_mean": _safe_mean([r["hst_positive_rate"] for r in rows]),
            "hst_mean_entropy_mean": _safe_mean([r["hst_mean_entropy"] for r in rows]),
        }

    (output_root / "domain_ablation.json").write_text(json.dumps(ablation, indent=2))
    return ablation


def _run_bootstrap_cis(
    cfg: dict[str, Any],
    output_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    n_bootstrap = int(cfg["bootstrap"].get("n_bootstrap", 300))
    threshold = float(cfg["bootstrap"].get("threshold", 0.5))

    cls_val_ds = SimulatedLensDataset(
        cfg["data"]["simulated_dir"],
        split="val",
        task="classify",
        augment=False,
        split_seed=int(cfg["seed"]),
    )
    cls_loader = DataLoader(cls_val_ds, batch_size=256, shuffle=False, num_workers=0)

    reg_val_ds = SimulatedLensDataset(
        cfg["data"]["simulated_dir"],
        split="val",
        task="regress",
        augment=False,
        split_seed=int(cfg["seed"]),
    )
    reg_loader = DataLoader(reg_val_ds, batch_size=256, shuffle=False, num_workers=0)

    cls_model = LensClassifier()
    cls_ckpt = torch.load(
        cfg["checkpoints"]["classifier"], map_location=device, weights_only=False
    )
    cls_model.load_state_dict(cls_ckpt["model_state"])
    cls_model.to(device)

    y_true_cls, y_prob_cls = _collect_classifier_outputs(cls_model, cls_loader, device)
    cls_ci = bootstrap_classifier_ci(
        y_true_cls,
        y_prob_cls,
        threshold=threshold,
        n_bootstrap=n_bootstrap,
        seed=int(cfg["seed"]),
    )

    reg_model = LensParameterRegressor()
    reg_ckpt = torch.load(
        cfg["checkpoints"]["regressor"], map_location=device, weights_only=False
    )
    reg_model.load_state_dict(reg_ckpt["model_state"])
    reg_model.to(device)

    y_true_reg, y_pred_reg = _collect_regressor_outputs(reg_model, reg_loader, device)
    reg_ci = bootstrap_regressor_ci(
        y_true_reg,
        y_pred_reg,
        n_bootstrap=n_bootstrap,
        seed=int(cfg["seed"]),
    )

    result = {
        "classifier_ci": cls_ci,
        "regressor_ci": reg_ci,
        "n_bootstrap": n_bootstrap,
    }
    (output_root / "bootstrap_ci.json").write_text(json.dumps(result, indent=2))
    return result


def _run_error_taxonomy(
    cfg: dict[str, Any],
    output_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    val_ds = SimulatedLensDataset(
        cfg["data"]["simulated_dir"],
        split="val",
        task="classify",
        augment=False,
        split_seed=int(cfg["seed"]),
    )
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=0)

    model = LensClassifier()
    ckpt = torch.load(
        cfg["checkpoints"]["classifier"], map_location=device, weights_only=False
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    y_true, y_prob = _collect_classifier_outputs(model, val_loader, device)
    y_pred = (y_prob >= 0.5).astype(int)

    images = val_ds.images.astype(np.float32)

    def concentration(img: np.ndarray) -> float:
        h, w = img.shape
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        r0 = min(h, w) * 0.2
        total = np.abs(img).sum() + 1e-8
        core = np.abs(img[rr <= r0]).sum()
        return float(core / total)

    def asymmetry(img: np.ndarray) -> float:
        rot = np.rot90(img, 2)
        denom = np.abs(img).mean() + 1e-8
        return float(np.abs(img - rot).mean() / denom)

    def snr_estimate(img: np.ndarray) -> float:
        p50 = np.percentile(img, 50)
        signal = float(np.percentile(img, 99) - p50)
        bg = img[img <= p50]
        noise = float(np.std(bg)) if bg.size > 0 else float(np.std(img))
        return signal / (noise + 1e-8)

    conc = np.array([concentration(img) for img in images], dtype=np.float32)
    asym = np.array([asymmetry(img) for img in images], dtype=np.float32)
    snr = np.array([snr_estimate(img) for img in images], dtype=np.float32)

    c1, c2 = np.quantile(conc, [0.33, 0.66])
    a1, a2 = np.quantile(asym, [0.33, 0.66])
    s1, s2 = np.quantile(snr, [0.33, 0.66])

    def morphology_label(c: float, a: float) -> str:
        if c >= c2 and a <= a1:
            return "compact_symmetric"
        if c <= c1 and a >= a2:
            return "diffuse_asymmetric"
        return "intermediate"

    def noise_label(s: float) -> str:
        if s <= s1:
            return "low_snr"
        if s >= s2:
            return "high_snr"
        return "mid_snr"

    morph_labels = np.array([morphology_label(c, a) for c, a in zip(conc, asym)])
    noise_labels = np.array([noise_label(s) for s in snr])

    fp = (y_true == 0) & (y_pred == 1)
    fn = (y_true == 1) & (y_pred == 0)
    high_fp = fp & (y_prob >= float(cfg["taxonomy"].get("high_confidence_fp_threshold", 0.8)))
    high_fn = fn & (y_prob <= float(cfg["taxonomy"].get("high_confidence_fn_threshold", 0.2)))

    summary: dict[str, Any] = {
        "false_positives": int(fp.sum()),
        "false_negatives": int(fn.sum()),
        "high_confidence_fp": int(high_fp.sum()),
        "high_confidence_fn": int(high_fn.sum()),
        "mean_prob_fp": float(y_prob[fp].mean()) if fp.any() else 0.0,
        "mean_prob_fn": float(y_prob[fn].mean()) if fn.any() else 0.0,
        "morphology_thresholds": {
            "concentration_q33": float(c1),
            "concentration_q66": float(c2),
            "asymmetry_q33": float(a1),
            "asymmetry_q66": float(a2),
        },
        "noise_thresholds": {
            "snr_q33": float(s1),
            "snr_q66": float(s2),
        },
    }

    morphology_names = ["compact_symmetric", "intermediate", "diffuse_asymmetric"]
    noise_names = ["low_snr", "mid_snr", "high_snr"]

    def count_grid(mask: np.ndarray) -> dict[str, dict[str, int]]:
        grid: dict[str, dict[str, int]] = {}
        for m in morphology_names:
            grid[m] = {}
            for n in noise_names:
                hit = mask & (morph_labels == m) & (noise_labels == n)
                grid[m][n] = int(hit.sum())
        return grid

    fp_grid = count_grid(fp)
    fn_grid = count_grid(fn)
    summary["fp_by_morphology_noise"] = fp_grid
    summary["fn_by_morphology_noise"] = fn_grid

    labels = [
        "FP",
        "FN",
        "HighConfFP",
        "HighConfFN",
    ]
    values = [
        float(summary["false_positives"]),
        float(summary["false_negatives"]),
        float(summary["high_confidence_fp"]),
        float(summary["high_confidence_fn"]),
    ]

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, values, color=["#EF5350", "#42A5F5", "#D32F2F", "#1565C0"])
    ax.set_ylabel("Count")
    ax.set_title("Error Taxonomy Summary")
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    fig.savefig(output_root / "error_taxonomy.png", dpi=160)
    plt.close(fig)

    fp_mat = np.array([[fp_grid[m][n] for n in noise_names] for m in morphology_names], dtype=np.float32)
    fn_mat = np.array([[fn_grid[m][n] for n in noise_names] for m in morphology_names], dtype=np.float32)

    fig2, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax_i, mat, title in zip(
        axes,
        [fp_mat, fn_mat],
        ["False Positives", "False Negatives"],
    ):
        im = ax_i.imshow(mat, cmap="magma")
        ax_i.set_xticks(range(len(noise_names)))
        ax_i.set_xticklabels(noise_names, rotation=25)
        ax_i.set_yticks(range(len(morphology_names)))
        ax_i.set_yticklabels(morphology_names)
        ax_i.set_title(title)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                ax_i.text(j, i, f"{int(mat[i, j])}", ha="center", va="center", color="white", fontsize=9)
        fig2.colorbar(im, ax=ax_i, fraction=0.046, pad=0.04)
    fig2.suptitle("Failure Modes by Morphology and Noise Regime")
    fig2.tight_layout()
    fig2.savefig(output_root / "failure_mode_heatmaps.png", dpi=170)
    plt.close(fig2)

    n_panels = int(cfg["taxonomy"].get("n_example_panels", 12))
    fp_idx = np.where(fp)[0]
    fn_idx = np.where(fn)[0]

    def save_error_panel(indices: np.ndarray, filename: str, title: str, reverse: bool) -> None:
        if len(indices) == 0:
            return
        order = np.argsort(y_prob[indices])
        if reverse:
            order = order[::-1]
        selected = indices[order[:n_panels]]
        cols = 4
        rows = int(np.ceil(len(selected) / cols))
        figp, axarr = plt.subplots(rows, cols, figsize=(10, 2.7 * rows))
        if rows == 1:
            axarr = np.array([axarr])
        for k, idx in enumerate(selected):
            ax = axarr[k // cols, k % cols]
            img = images[idx]
            vmin, vmax = np.percentile(img, [1, 99])
            ax.imshow(img, cmap="inferno", origin="lower", vmin=vmin, vmax=vmax)
            ax.set_title(
                f"p={y_prob[idx]:.3f}\n{morph_labels[idx]} | {noise_labels[idx]}",
                fontsize=8,
            )
            ax.axis("off")
        for k in range(len(selected), rows * cols):
            axarr[k // cols, k % cols].axis("off")
        figp.suptitle(title)
        figp.tight_layout()
        figp.savefig(output_root / filename, dpi=170)
        plt.close(figp)

    save_error_panel(fp_idx, "failure_examples_fp.png", "High-Confidence False Positives", reverse=True)
    save_error_panel(fn_idx, "failure_examples_fn.png", "Lowest-Probability False Negatives", reverse=False)

    (output_root / "error_taxonomy.json").write_text(json.dumps(summary, indent=2))
    return summary


def _to_table(rows: list[tuple[str, str, str]]) -> str:
    header = "| Section | Metric | Value |\n|---|---:|---:|\n"
    body = "\n".join([f"| {a} | {b} | {c} |" for a, b, c in rows])
    return header + body + "\n"


def _write_report(
    cfg: dict[str, Any],
    baselines: dict[str, Any],
    ablation: dict[str, Any],
    ci: dict[str, Any],
    taxonomy: dict[str, Any],
) -> None:
    report_path = Path(cfg["outputs"]["report_path"])
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, str, str]] = []

    for name, metrics in baselines.items():
        rows.append(("Baselines", f"{name} AUC", f"{metrics.get('auc_roc', 0.0):.4f}"))
        rows.append(("Baselines", f"{name} F1", f"{metrics.get('f1', 0.0):.4f}"))

    for method, agg in ablation.get("aggregate", {}).items():
        if int(agg.get("n_clusters", 0)) == 0:
            rows.append(("Ablation (aggregate)", method, "no successful clusters"))
            continue
        rows.append(("Ablation (aggregate)", f"{method} AUC", f"{agg.get('auc_roc_mean', 0.0):.4f}"))
        rows.append(("Ablation (aggregate)", f"{method} F1", f"{agg.get('f1_mean', 0.0):.4f}"))
        rows.append((
            "Ablation (aggregate)",
            f"{method} HST positive rate",
            f"{agg.get('hst_positive_rate_mean', 0.0):.4f}",
        ))

    for metric_name, metric_vals in ci["classifier_ci"].items():
        rows.append(
            (
                "Classifier CI",
                metric_name,
                f"{metric_vals['mean']:.4f} [{metric_vals['ci95_low']:.4f}, {metric_vals['ci95_high']:.4f}]",
            )
        )

    for param_name, metric_vals in ci["regressor_ci"].items():
        rows.append(
            (
                "Regressor CI",
                f"RMSE {param_name}",
                f"{metric_vals['mean']:.4f} [{metric_vals['ci95_low']:.4f}, {metric_vals['ci95_high']:.4f}]",
            )
        )

    rows.append(("Taxonomy", "False Positives", str(taxonomy["false_positives"])))
    rows.append(("Taxonomy", "False Negatives", str(taxonomy["false_negatives"])))
    rows.append(("Taxonomy", "High-Confidence FP", str(taxonomy["high_confidence_fp"])))
    rows.append(("Taxonomy", "High-Confidence FN", str(taxonomy["high_confidence_fn"])))

    report = (
        "# GravLensAI Research Report\n\n"
        "## Methods\n"
        "- Data: simulated train/val splits with deterministic split seed from configs/research.yaml.\n"
        "- Baselines: logistic regression, random forest, and lightweight CNN.\n"
        "- Domain adaptation ablations: baseline vs MMD vs DANN, run independently per HST cluster and aggregated.\n"
        "- Confidence analysis: non-parametric bootstrap CIs for precision/recall/F1/AUC and per-parameter RMSE.\n"
        "- Error taxonomy: false positives/negatives stratified by morphology proxy and SNR regime.\n\n"
        "## Ablations\n"
        + _to_table(rows)
        + "\n## Limitations\n"
        "- Most HST clusters are unlabeled, so in-domain performance is summarized via confidence and entropy rather than true precision/recall.\n"
        "- Morphology/noise bins use proxy heuristics (concentration, asymmetry, SNR) and are not substitute for expert morphological catalogs.\n"
        "- Domain adaptation sweeps optimize for practicality; deeper hyperparameter sweeps may change cluster-level rankings.\n\n"
        "## Reproducibility Checklist\n"
        "- [x] Fixed global seed configured and applied\n"
        "- [x] Config-first execution through configs/research.yaml\n"
        "- [x] Deterministic train/val splits\n"
        "- [x] Script-generated JSON and figure artifacts\n"
        "- [x] Bootstrap confidence intervals reported\n"
        "- [x] Cluster-wise ablation outputs persisted\n\n"
        "## Artifacts\n"
        "- Baseline metrics: results/research/baselines.json\n"
        "- Domain ablation: results/research/domain_ablation.json\n"
        "- Bootstrap CIs: results/research/bootstrap_ci.json\n"
        "- Error taxonomy summary: results/research/error_taxonomy.json\n"
        "- Failure heatmaps: results/research/failure_mode_heatmaps.png\n"
        "- Failure examples: results/research/failure_examples_fp.png and results/research/failure_examples_fn.png\n"
    )
    report_path.write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GravLensAI research suite")
    parser.add_argument(
        "--config",
        default="configs/research.yaml",
        help="Path to research config YAML",
    )
    args = parser.parse_args()

    cfg = load_yaml_section(args.config, "research")
    set_global_seed(int(cfg["seed"]))

    output_root = Path(cfg["outputs"]["root_dir"])
    output_root.mkdir(parents=True, exist_ok=True)
    device = _device()

    print(f"[research] Device: {device}")
    print("[research] Running baselines...")
    baselines = _run_baselines(cfg, output_root, device)

    print("[research] Running domain ablation...")
    ablation = _run_domain_ablation(cfg, output_root, device)

    print("[research] Running bootstrap confidence intervals...")
    ci = _run_bootstrap_cis(cfg, output_root, device)

    print("[research] Building error taxonomy...")
    taxonomy = _run_error_taxonomy(cfg, output_root, device)

    print("[research] Writing report...")
    _write_report(cfg, baselines, ablation, ci, taxonomy)

    print("[research] Complete")


if __name__ == "__main__":
    main()
