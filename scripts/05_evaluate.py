"""
Full evaluation pipeline.

Runs classifier + regressor on the test set, computes all metrics,
generates publication-quality figures, and estimates LENSTOOL speedup.

Usage:
    python scripts/05_evaluate.py --data_dir data/simulated/ --models_dir results/models/
"""

import argparse
import torch
import numpy as np
import time
from pathlib import Path
from torch.utils.data import DataLoader

from gravlensai.data.dataset import SimulatedLensDataset
from gravlensai.models.classifier import LensClassifier
from gravlensai.models.regressor import LensParameterRegressor
from gravlensai.evaluate.metrics import (
    classifier_metrics, optimal_threshold, regressor_metrics,
    speedup_vs_lenstool, print_classifier_report, print_regressor_report,
    expected_calibration_error, negative_log_likelihood,
)
from gravlensai.evaluate.visualise import (
    plot_detection_grid, plot_parameter_recovery, plot_confusion_matrix,
    plot_calibration_curve, plot_roc_pr_curves, plot_uncertainty_distribution,
    plot_residual_histograms,
)
from gravlensai.evaluate.grad_cam import (
    generate_cam_for_batch, plot_cam_grid, get_target_layer,
)
from gravlensai.utils.config import load_yaml_section
from gravlensai.utils.reproducibility import set_global_seed


def load_model(model_class, checkpoint_path, device):
    """Load a model from checkpoint."""
    model = model_class()
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.to(device)
    model.eval()
    print(f"  Loaded {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return model


def evaluate_classifier(
    model,
    dataset,
    device,
    batch_size=64,
    lenstool_seconds_per_image=120.0,
):
    """Run classifier evaluation on a dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_probs, all_labels = [], []
    all_images = []

    t0 = time.time()
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            probs = model.predict_proba(images)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_images.extend(images.cpu().numpy()[:, 0])  # remove channel dim

    elapsed = time.time() - t0

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_images = np.array(all_images)

    # Metrics at default threshold
    metrics = classifier_metrics(all_labels, all_probs, threshold=0.5)

    # Optimal threshold
    if len(set(all_labels.astype(int))) >= 2:
        opt_thresh = optimal_threshold(all_labels, all_probs)
        metrics_opt = classifier_metrics(all_labels, all_probs, threshold=opt_thresh)
        metrics['optimal_threshold'] = opt_thresh
        metrics['optimal_f1'] = metrics_opt['f1']

    # Speedup
    speed = speedup_vs_lenstool(
        len(all_labels), elapsed, lenstool_seconds_per_image=lenstool_seconds_per_image
    )
    metrics['speedup'] = speed

    return metrics, all_probs, all_labels, all_images


def evaluate_regressor(model, dataset, device, batch_size=64):
    """Run regressor evaluation on a dataset."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_pred_norm, all_true_norm = [], []

    t0 = time.time()
    with torch.no_grad():
        for images, params in loader:
            images = images.to(device)
            pred = model(images)
            all_pred_norm.append(pred.cpu())
            all_true_norm.append(params)

    elapsed = time.time() - t0

    pred_norm = torch.cat(all_pred_norm)
    true_norm = torch.cat(all_true_norm)

    # Denormalise to physical units
    pred_phys = model.denormalise(pred_norm).numpy()
    true_phys = model.denormalise(true_norm).numpy()

    metrics = regressor_metrics(true_phys, pred_phys)
    metrics['inference_time'] = elapsed
    metrics['n_images'] = len(pred_phys)

    return metrics, true_phys, pred_phys


def main(args):
    cfg = load_yaml_section(args.config, 'evaluate')
    batch_size = args.batch_size if args.batch_size is not None else int(cfg.get('batch_size', 64))
    n_uncertainty = (
        args.uncertainty_samples
        if args.uncertainty_samples is not None
        else int(cfg.get('uncertainty_samples', 20))
    )
    calibration_bins = args.calibration_bins if args.calibration_bins is not None else int(cfg.get('calibration_bins', 10))
    lenstool_seconds = (
        args.lenstool_seconds_per_image
        if args.lenstool_seconds_per_image is not None
        else float(cfg.get('lenstool_seconds_per_image', 120.0))
    )
    seed = args.seed if args.seed is not None else int(cfg.get('seed', 42))

    set_global_seed(seed)

    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )
    print(f"Evaluating on: {device}")
    print(f"Benchmark assumption: LENSTOOL {lenstool_seconds:.1f}s/image")

    models_dir = Path(args.models_dir)
    output_dir = Path(args.output)
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ── Classifier ──────────────────────────────────────────
    clf_path = models_dir / "classifier_best.pt"
    if clf_path.exists():
        print("\n── Classifier Evaluation ──")
        classifier = load_model(LensClassifier, clf_path, device)
        test_ds = SimulatedLensDataset(args.data_dir, split='test',
                                        task='classify', augment=False)
        print(f"  Test set: {len(test_ds)} images")

        clf_metrics, probs, labels, images = evaluate_classifier(
            classifier,
            test_ds,
            device,
            batch_size=batch_size,
            lenstool_seconds_per_image=lenstool_seconds,
        )

        print_classifier_report(clf_metrics)
        print(f"\n  Inference: {clf_metrics['speedup']['cnn_ms_per_image']:.2f} ms/image")
        print(f"  Speedup vs LENSTOOL: {clf_metrics['speedup']['speedup_factor']:,.0f}×")

        # Detection grid figure
        plot_detection_grid(
            images, probs, labels,
            n=min(16, len(images)),
            save_path=fig_dir / "detection_grid.png",
        )
        print(f"  Saved: {fig_dir / 'detection_grid.png'}")

        # Confusion matrix
        preds = (probs >= 0.5).astype(int)
        plot_confusion_matrix(
            labels.astype(int), preds,
            save_path=fig_dir / "confusion_matrix.png",
        )
        print(f"  Saved: {fig_dir / 'confusion_matrix.png'}")

        # Calibration analysis (ECE + NLL)
        ece, bin_data = expected_calibration_error(labels, probs, n_bins=calibration_bins)
        nll = negative_log_likelihood(labels, probs)
        clf_metrics['ece'] = ece
        clf_metrics['nll'] = nll
        print(f"\n  ECE:        {ece:.4f}")
        print(f"  NLL:        {nll:.4f}")

        # Calibration curve
        plot_calibration_curve(
            labels, probs, n_bins=calibration_bins,
            save_path=fig_dir / "calibration_curve.png",
        )
        print(f"  Saved: {fig_dir / 'calibration_curve.png'}")

        # ROC + PR curves
        plot_roc_pr_curves(
            labels, probs,
            save_path=fig_dir / "roc_pr_curves.png",
        )
        print(f"  Saved: {fig_dir / 'roc_pr_curves.png'}")

        # MC Dropout uncertainty
        print("\n── Classifier Uncertainty (MC Dropout) ──")
        test_subset_imgs = []
        test_subset_labels = []
        subset_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
        all_means, all_stds = [], []
        for batch_images, batch_labels in subset_loader:
            batch_images = batch_images.to(device)
            unc = classifier.predict_with_uncertainty(batch_images, n_forward=n_uncertainty)
            all_means.extend(unc['mean'].detach().cpu().numpy())
            all_stds.extend(unc['std'].detach().cpu().numpy())
            test_subset_imgs.extend(batch_images.cpu().numpy()[:, 0])
            test_subset_labels.extend(batch_labels.numpy())

        all_stds = np.array(all_stds)
        test_subset_labels = np.array(test_subset_labels)
        print(f"  Mean uncertainty: {all_stds.mean():.4f}")
        print(f"  Max uncertainty:  {all_stds.max():.4f}")

        plot_uncertainty_distribution(
            all_stds, test_subset_labels,
            save_path=fig_dir / "uncertainty_distribution.png",
        )
        print(f"  Saved: {fig_dir / 'uncertainty_distribution.png'}")

        # Grad-CAM
        print("\n── Grad-CAM Interpretability ──")
        cam_images = np.array(test_subset_imgs[:50])
        cam_tensor = torch.from_numpy(cam_images).unsqueeze(1).float().to(device)
        target_layer = get_target_layer(classifier)
        heatmaps = generate_cam_for_batch(classifier, cam_tensor, target_layer)
        plot_cam_grid(
            cam_images[:8], heatmaps[:8], probs[:8],
            n=8, save_path=fig_dir / "grad_cam_grid.png",
        )
        print(f"  Saved: {fig_dir / 'grad_cam_grid.png'}")
    else:
        print(f"  Classifier checkpoint not found: {clf_path}")

    # ── Regressor ───────────────────────────────────────────
    reg_path = models_dir / "regressor_best.pt"
    if reg_path.exists():
        print("\n── Regressor Evaluation ──")
        regressor = load_model(LensParameterRegressor, reg_path, device)
        test_ds_reg = SimulatedLensDataset(args.data_dir, split='test',
                                            task='regress', augment=False)
        print(f"  Test set: {len(test_ds_reg)} lensed images")

        reg_metrics, true_phys, pred_phys = evaluate_regressor(
            regressor, test_ds_reg, device, batch_size=batch_size
        )

        print_regressor_report(reg_metrics)

        # Parameter recovery plot
        plot_parameter_recovery(
            true_phys, pred_phys,
            save_path=fig_dir / "parameter_recovery.png",
        )
        print(f"  Saved: {fig_dir / 'parameter_recovery.png'}")

        # Residual histograms
        plot_residual_histograms(
            true_phys, pred_phys,
            save_path=fig_dir / "residual_histograms.png",
        )
        print(f"  Saved: {fig_dir / 'residual_histograms.png'}")

        # MC Dropout uncertainty for regressor
        print("\n── Regressor Uncertainty (MC Dropout) ──")
        reg_loader = DataLoader(test_ds_reg, batch_size=batch_size, shuffle=False, num_workers=0)
        all_reg_means, all_reg_stds, all_reg_true = [], [], []
        for batch_images, batch_params in reg_loader:
            batch_images = batch_images.to(device)
            unc = regressor.predict_with_uncertainty(batch_images, n_forward=n_uncertainty)
            all_reg_means.append(unc['mean'].detach().cpu().numpy())
            all_reg_stds.append(unc['std'].detach().cpu().numpy())
            all_reg_true.append(regressor.denormalise(batch_params).numpy())

        all_reg_means = np.concatenate(all_reg_means)
        all_reg_stds = np.concatenate(all_reg_stds)
        all_reg_true = np.concatenate(all_reg_true)

        param_names = ['theta_E', 'e1', 'e2', 'gamma1', 'gamma2']
        print("  Per-parameter mean uncertainty:")
        for i, name in enumerate(param_names):
            print(f"    {name}: {all_reg_stds[:, i].mean():.4f}")

        # Coverage probability
        try:
            from gravlensai.evaluate.metrics import coverage_probability
            cov = coverage_probability(all_reg_true, all_reg_means,
                                       all_reg_stds, alpha=0.68)
            print("\n  68% Coverage Probability:")
            for name, vals in cov.items():
                status = '✓' if abs(vals['gap']) < 0.15 else '!'
                print(f"    {status} {name}: {vals['coverage']:.2%} "
                      f"(expected 68%, gap {vals['gap']:+.2%})")
        except ImportError:
            print("  (scipy not available — skipping coverage analysis)")

        # Save predictions
        pred_dir = output_dir / "predictions"
        pred_dir.mkdir(parents=True, exist_ok=True)
        np.save(pred_dir / "test_true_params.npy", true_phys)
        np.save(pred_dir / "test_pred_params.npy", pred_phys)
        np.save(pred_dir / "test_uncertainty_std.npy", all_reg_stds)
        print(f"  Saved predictions to {pred_dir}/")
    else:
        print(f"  Regressor checkpoint not found: {reg_path}")

    print("\n" + "=" * 50)
    print("Evaluation complete!")
    print(f"Figures saved to: {fig_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GravLensAI models")
    parser.add_argument("--config", default="configs/evaluate.yaml")
    parser.add_argument("--data_dir", default="data/simulated/")
    parser.add_argument("--models_dir", default="results/models/")
    parser.add_argument("--output", default="results/")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--uncertainty_samples", type=int, default=None)
    parser.add_argument("--calibration_bins", type=int, default=None)
    parser.add_argument("--lenstool_seconds_per_image", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    main(parser.parse_args())
