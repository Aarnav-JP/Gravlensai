"""
Generate the main results figures for the README and paper.

Figures:
  1. Detection grid — 4x4 top lens candidates with probabilities
  2. Parameter recovery — true vs. predicted scatter for each parameter
  3. Confusion matrix — classification performance heatmap
  4. Training curves — loss and AUC over epochs
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Optional

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'figure.dpi': 150,
})


def plot_detection_grid(images, probs, labels=None, n=16, save_path=None):
    """
    4×4 grid of lens candidates with predicted probability overlaid.
    Sorted by descending confidence. This is the key README figure.

    Args:
        images: (N, 64, 64) numpy array.
        probs: (N,) predicted lens probabilities.
        labels: (N,) ground truth labels (optional).
        n: Number of images to show.
        save_path: Path to save the figure.

    Returns:
        matplotlib Figure.
    """
    idx = np.argsort(-probs)[:n]
    rows = int(np.ceil(n / 4))

    fig, axes = plt.subplots(rows, 4, figsize=(10, 2.5 * rows))
    fig.patch.set_facecolor('black')

    if rows == 1:
        axes = [axes]

    for plot_i, data_i in enumerate(idx):
        ax = axes[plot_i // 4][plot_i % 4]

        img = images[data_i]
        vmin, vmax = np.percentile(img, [1, 99])
        ax.imshow(img, cmap='inferno', origin='lower', vmin=vmin, vmax=vmax)

        p = probs[data_i]
        color = 'lime' if p > 0.8 else 'orange' if p > 0.5 else 'red'
        title = f"p={p:.3f}"
        if labels is not None:
            gt = "✓" if labels[data_i] == 1 else "✗"
            title += f" {gt}"
        ax.set_title(title, fontsize=9, color=color, pad=2)
        ax.axis('off')

    # Hide empty subplots
    for i in range(n, rows * 4):
        axes[i // 4][i % 4].set_visible(False)

    fig.suptitle("Top Lens Candidates — GravLensAI",
                 color='white', fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='black')
    return fig


def plot_parameter_recovery(y_true, y_pred, save_path=None):
    """
    Scatter plot per parameter: true vs. predicted.
    Headline figure for the regressor evaluation.

    Args:
        y_true: (N, 5) ground truth in physical units.
        y_pred: (N, 5) predictions in physical units.
        save_path: Path to save the figure.

    Returns:
        matplotlib Figure.
    """
    names = ['θ_E (arcsec)', 'e1', 'e2', 'γ1', 'γ2']

    fig, axes = plt.subplots(1, 5, figsize=(18, 4))

    for i, (ax, name) in enumerate(zip(axes, names)):
        x, y = y_true[:, i], y_pred[:, i]
        rmse = np.sqrt(np.mean((x - y) ** 2))

        ax.scatter(x, y, alpha=0.3, s=5, c='steelblue', rasterized=True)
        lim = [min(x.min(), y.min()), max(x.max(), y.max())]
        margin = (lim[1] - lim[0]) * 0.05
        lim = [lim[0] - margin, lim[1] + margin]
        ax.plot(lim, lim, 'r--', linewidth=1, label='1:1')
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Predicted {name}")
        ax.set_title(f"RMSE = {rmse:.4f}")
        ax.legend(fontsize=9)
        ax.set_xlim(lim)
        ax.set_ylim(lim)

    plt.suptitle("Lens Parameter Recovery — GravLensAI Regressor", fontsize=13)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Confusion matrix heatmap.

    Args:
        y_true: (N,) ground truth labels.
        y_pred: (N,) predicted labels.
        save_path: Path to save.

    Returns:
        matplotlib Figure.
    """
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    classes = ['Non-lens', 'Lens']
    ax.set(xticks=[0, 1], yticks=[0, 1],
           xticklabels=classes, yticklabels=classes,
           ylabel='True label', xlabel='Predicted label',
           title='Confusion Matrix — GravLensAI')

    # Text annotations
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_calibration_curve(y_true, y_prob, n_bins=10, save_path=None):
    """
    Reliability diagram showing model calibration.

    A perfectly calibrated model falls on the diagonal. Points above
    the diagonal indicate underconfidence; below indicates overconfidence.
    This is the standard calibration plot expected in ML papers.

    Args:
        y_true: (N,) ground truth binary labels.
        y_prob: (N,) predicted probabilities.
        n_bins: Number of calibration bins.
        save_path: Path to save figure.

    Returns:
        matplotlib Figure.
    """
    from sklearn.calibration import calibration_curve

    fraction_pos, mean_predicted = calibration_curve(
        y_true, y_prob, n_bins=n_bins, strategy='uniform'
    )

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7, 8), gridspec_kw={'height_ratios': [3, 1]}
    )

    # Reliability diagram
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect calibration')
    ax1.plot(mean_predicted, fraction_pos, 's-', color='#2196F3',
             markersize=8, linewidth=2, label='GravLensAI')
    ax1.fill_between(mean_predicted, fraction_pos, mean_predicted,
                     alpha=0.15, color='#2196F3')
    ax1.set_xlabel('Mean Predicted Probability', fontsize=12)
    ax1.set_ylabel('Fraction of Positives', fontsize=12)
    ax1.set_title('Calibration Curve (Reliability Diagram)', fontsize=13)
    ax1.legend(fontsize=11)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.grid(alpha=0.3)

    # Histogram of predictions
    ax2.hist(y_prob[y_true == 0], bins=50, alpha=0.6, color='#FF5722',
             label='Non-lens', density=True)
    ax2.hist(y_prob[y_true == 1], bins=50, alpha=0.6, color='#2196F3',
             label='Lens', density=True)
    ax2.set_xlabel('Predicted Probability', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.set_xlim([0, 1])

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_roc_pr_curves(y_true, y_prob, save_path=None):
    """
    Side-by-side ROC and Precision-Recall curves.

    Standard evaluation figures for binary classifiers in ML papers.
    Includes AUC values in the legend.

    Args:
        y_true: (N,) ground truth binary labels.
        y_prob: (N,) predicted probabilities.
        save_path: Path to save figure.

    Returns:
        matplotlib Figure.
    """
    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    # ROC curve
    ax1.plot(fpr, tpr, color='#2196F3', linewidth=2.5,
             label=f'GravLensAI (AUC = {roc_auc:.6f})')
    ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax1.fill_between(fpr, tpr, alpha=0.1, color='#2196F3')
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('ROC Curve', fontsize=13)
    ax1.legend(fontsize=11, loc='lower right')
    ax1.grid(alpha=0.3)

    # PR curve
    ax2.plot(recall, precision, color='#4CAF50', linewidth=2.5,
             label=f'GravLensAI (AUC = {pr_auc:.6f})')
    ax2.axhline(y=np.mean(y_true), color='k', linestyle='--',
                linewidth=1, label='Baseline')
    ax2.fill_between(recall, precision, alpha=0.1, color='#4CAF50')
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curve', fontsize=13)
    ax2.legend(fontsize=11, loc='lower left')
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_uncertainty_distribution(
    uncertainties, labels=None, save_path=None
):
    """
    Distribution of MC Dropout uncertainty values.

    Shows that high-uncertainty predictions cluster around the decision
    boundary, while confident predictions are either strongly lens or
    strongly non-lens. This validates the uncertainty estimates.

    Args:
        uncertainties: (N,) standard deviations from MC Dropout.
        labels: (N,) ground truth binary labels (optional).
        save_path: Path to save figure.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    if labels is not None:
        ax.hist(uncertainties[labels == 0], bins=40, alpha=0.6,
                color='#FF5722', label='Non-lens', density=True)
        ax.hist(uncertainties[labels == 1], bins=40, alpha=0.6,
                color='#2196F3', label='Lens', density=True)
        ax.legend(fontsize=11)
    else:
        ax.hist(uncertainties, bins=40, alpha=0.7, color='#9C27B0',
                density=True)

    ax.set_xlabel('Epistemic Uncertainty (σ)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.set_title('MC Dropout Uncertainty Distribution', fontsize=13)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig


def plot_residual_histograms(y_true, y_pred, save_path=None):
    """
    Per-parameter residual distribution histograms.

    Shows bias (mean offset from zero) and spread of prediction errors.
    Ideally, residuals should be centred at zero with small spread.

    Args:
        y_true: (N, 5) ground truth in physical units.
        y_pred: (N, 5) predictions in physical units.
        save_path: Path to save figure.

    Returns:
        matplotlib Figure.
    """
    names = ['θ_E (arcsec)', 'e1', 'e2', 'γ1', 'γ2']
    residuals = y_pred - y_true

    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    for i, (ax, name) in enumerate(zip(axes, names)):
        r = residuals[:, i]
        color = '#2196F3'
        ax.hist(r, bins=50, alpha=0.7, color=color, density=True,
                edgecolor='white', linewidth=0.5)
        ax.axvline(0, color='red', linestyle='--', linewidth=1.5,
                   label='Zero bias')
        ax.axvline(r.mean(), color='orange', linestyle='-', linewidth=1.5,
                   label=f'Mean={r.mean():.4f}')
        ax.set_xlabel(f'Residual ({name})', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title(f'{name}\nσ={r.std():.4f}', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle('Parameter Estimation Residuals', fontsize=13, y=1.02)
    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    return fig
