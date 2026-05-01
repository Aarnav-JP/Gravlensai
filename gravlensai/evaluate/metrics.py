"""
Evaluation metrics for both the classifier and regressor.

Primary metrics reported in results:
Classifier:
  - Precision @ threshold=0.5 (and optimised threshold)
  - Recall @ threshold=0.5
  - AUC-ROC
  - F1 score

Regressor (per parameter, in physical units after denormalisation):
  - RMSE (root mean squared error)
  - Median absolute error
  - 68th percentile error (1-sigma equivalent)
  - Speedup vs. LENSTOOL (compute time ratio)
"""

import numpy as np
from sklearn.metrics import (precision_score, recall_score, f1_score,
                              roc_auc_score, precision_recall_curve,
                              confusion_matrix)
import time


PARAM_NAMES = ['Einstein_radius_arcsec', 'e1', 'e2', 'gamma1', 'gamma2', 'subhalo_mass']
PARAM_UNITS = ['arcsec', '', '', '', '', 'log10(M_sun)']


def classifier_metrics(y_true, y_prob, threshold=0.5):
    """
    Compute classification metrics.

    Args:
        y_true: (N,) ground truth binary labels.
        y_prob: (N,) predicted probabilities (after sigmoid).
        threshold: Decision threshold.

    Returns:
        Dict of metrics.
    """
    y_pred = (np.asarray(y_prob) >= threshold).astype(int)
    y_true = np.asarray(y_true).astype(int)

    metrics = {
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'threshold': threshold,
    }

    # AUC only if both classes present
    if len(set(y_true)) >= 2:
        metrics['auc_roc'] = roc_auc_score(y_true, y_prob)
    else:
        metrics['auc_roc'] = float('nan')

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm.tolist()

    return metrics


def optimal_threshold(y_true, y_prob):
    """
    Find the F1-maximising decision threshold.

    Args:
        y_true: (N,) ground truth binary labels.
        y_prob: (N,) predicted probabilities.

    Returns:
        Optimal threshold value.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    return float(thresholds[f1s.argmax()])


def regressor_metrics(y_true, y_pred):
    """
    Per-parameter regression metrics in physical units.

    Args:
        y_true: (N, 6) ground truth in physical units (after denormalisation).
        y_pred: (N, 6) predictions in physical units.

    Returns:
        Dict of per-parameter metrics.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    errors = y_pred - y_true

    results = {}
    for i, (name, unit) in enumerate(zip(PARAM_NAMES, PARAM_UNITS)):
        abs_err = np.abs(errors[:, i])
        results[name] = {
            'rmse': float(np.sqrt(np.mean(errors[:, i] ** 2))),
            'median': float(np.median(abs_err)),
            'p68': float(np.percentile(abs_err, 68)),
            'mean_bias': float(np.mean(errors[:, i])),
            'unit': unit,
        }
    return results


def speedup_vs_lenstool(n_images, cnn_seconds, lenstool_seconds_per_image=120):
    """
    Compute inference speedup versus LENSTOOL.

    Args:
        n_images: Number of images processed.
        cnn_seconds: Total CNN inference time in seconds.
        lenstool_seconds_per_image: LENSTOOL time per image (default 120s = 2min).

    Returns:
        Dict with speedup metrics.
    """
    cnn_per_image = cnn_seconds / max(n_images, 1)
    speedup = lenstool_seconds_per_image / max(cnn_per_image, 1e-9)
    return {
        'cnn_ms_per_image': cnn_per_image * 1000,
        'lenstool_s_per_image': lenstool_seconds_per_image,
        'speedup_factor': speedup,
        'n_images': n_images,
        'total_cnn_seconds': cnn_seconds,
    }


def print_classifier_report(metrics: dict):
    """Pretty-print classifier evaluation results."""
    print("\n" + "=" * 50)
    print("CLASSIFIER EVALUATION")
    print("=" * 50)
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  F1:         {metrics['f1']:.4f}")
    print(f"  AUC-ROC:    {metrics.get('auc_roc', 'N/A')}")
    print(f"  Threshold:  {metrics['threshold']:.4f}")
    if 'ece' in metrics:
        print(f"  ECE:        {metrics['ece']:.4f}")
    if 'nll' in metrics:
        print(f"  NLL:        {metrics['nll']:.4f}")


def print_regressor_report(metrics: dict):
    """Pretty-print regressor evaluation results."""
    print("\n" + "=" * 50)
    print("REGRESSOR EVALUATION")
    print("=" * 50)
    for name, vals in metrics.items():
        if not isinstance(vals, dict):
            continue
        unit = vals.get('unit', '')
        print(f"  {name}:")
        print(f"    RMSE:   {vals['rmse']:.4f} {unit}")
        print(f"    Median: {vals['median']:.4f} {unit}")
        print(f"    P68:    {vals['p68']:.4f} {unit}")


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """
    Expected Calibration Error (ECE).

    Measures how well predicted probabilities match observed frequencies.
    A perfectly calibrated model has ECE = 0.

    For a model predicting p=0.8, approximately 80% of those predictions
    should actually be positive.

    Args:
        y_true: (N,) ground truth binary labels.
        y_prob: (N,) predicted probabilities.
        n_bins: Number of calibration bins.

    Returns:
        ECE value (float). Lower is better.
    """
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.asarray(y_prob).astype(float)

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    bin_data = []

    for i in range(n_bins):
        mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if i == n_bins - 1:  # Include right edge
            mask = mask | (y_prob == bin_boundaries[i + 1])

        n_in_bin = mask.sum()
        if n_in_bin == 0:
            bin_data.append({'confidence': 0, 'accuracy': 0, 'count': 0})
            continue

        avg_confidence = y_prob[mask].mean()
        avg_accuracy = y_true[mask].mean()
        ece += (n_in_bin / len(y_true)) * abs(avg_accuracy - avg_confidence)

        bin_data.append({
            'confidence': float(avg_confidence),
            'accuracy': float(avg_accuracy),
            'count': int(n_in_bin),
        })

    return float(ece), bin_data


def negative_log_likelihood(y_true, y_prob):
    """
    Negative log-likelihood (proper scoring rule).

    Heavily penalises confident wrong predictions. Unlike accuracy,
    this metric captures calibration quality.

    Args:
        y_true: (N,) ground truth binary labels.
        y_prob: (N,) predicted probabilities.

    Returns:
        NLL value (float). Lower is better.
    """
    y_true = np.asarray(y_true).astype(float)
    y_prob = np.clip(np.asarray(y_prob), 1e-7, 1 - 1e-7)
    nll = -(y_true * np.log(y_prob) + (1 - y_true) * np.log(1 - y_prob))
    return float(nll.mean())


def coverage_probability(y_true, y_pred_mean, y_pred_std, alpha=0.68):
    """
    Coverage probability for uncertainty estimates.

    Checks whether the fraction of true values falling within the
    predicted confidence interval matches the expected coverage.

    For alpha=0.68 (1-sigma), a well-calibrated model should have
    ~68% of true values within [mean - std, mean + std].

    Args:
        y_true: (N, P) ground truth parameters.
        y_pred_mean: (N, P) predicted mean.
        y_pred_std: (N, P) predicted standard deviation.
        alpha: Expected coverage (default 0.68 = 1-sigma).

    Returns:
        Dict with per-parameter coverage rates.
    """
    from scipy.stats import norm

    y_true = np.asarray(y_true)
    y_pred_mean = np.asarray(y_pred_mean)
    y_pred_std = np.asarray(y_pred_std)

    z = norm.ppf(0.5 + alpha / 2)  # z-score for the interval

    lower = y_pred_mean - z * y_pred_std
    upper = y_pred_mean + z * y_pred_std

    in_interval = (y_true >= lower) & (y_true <= upper)

    if y_true.ndim == 1:
        return {'coverage': float(in_interval.mean()), 'expected': alpha}

    coverages = {}
    for i, name in enumerate(PARAM_NAMES):
        coverages[name] = {
            'coverage': float(in_interval[:, i].mean()),
            'expected': alpha,
            'gap': float(in_interval[:, i].mean() - alpha),
        }
    return coverages


def bootstrap_classifier_ci(
    y_true,
    y_prob,
    threshold=0.5,
    n_bootstrap=1000,
    seed=42,
):
    """Bootstrap confidence intervals for classifier metrics."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rng = np.random.default_rng(seed)

    records = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        m = classifier_metrics(y_true[idx], y_prob[idx], threshold=threshold)
        records.append([
            m['precision'],
            m['recall'],
            m['f1'],
            m.get('auc_roc', float('nan')),
        ])

    arr = np.asarray(records, dtype=float)
    names = ['precision', 'recall', 'f1', 'auc_roc']
    out = {}
    for i, name in enumerate(names):
        values = arr[:, i]
        out[name] = {
            'mean': float(np.nanmean(values)),
            'ci95_low': float(np.nanpercentile(values, 2.5)),
            'ci95_high': float(np.nanpercentile(values, 97.5)),
        }
    return out


def bootstrap_regressor_ci(
    y_true,
    y_pred,
    n_bootstrap=1000,
    seed=42,
):
    """Bootstrap confidence intervals for regressor RMSE per parameter."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rng = np.random.default_rng(seed)
    n = len(y_true)

    rmse_samples = {name: [] for name in PARAM_NAMES}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        m = regressor_metrics(y_true[idx], y_pred[idx])
        for name in PARAM_NAMES:
            rmse_samples[name].append(m[name]['rmse'])

    out = {}
    for name in PARAM_NAMES:
        values = np.asarray(rmse_samples[name], dtype=float)
        out[name] = {
            'mean': float(values.mean()),
            'ci95_low': float(np.percentile(values, 2.5)),
            'ci95_high': float(np.percentile(values, 97.5)),
        }
    return out
