"""
Compare CNN predictions to LENSTOOL ray-tracing results.
Computes parameter agreement and inference speedup factor.

Since LENSTOOL is not available in all environments, this module
provides utilities to:
  1. Compare pre-computed LENSTOOL results (from CSV) with CNN predictions
  2. Estimate speedup based on typical LENSTOOL runtimes
"""

import numpy as np
from typing import Optional


# Typical LENSTOOL runtimes per lens system (from literature)
LENSTOOL_TIMES = {
    'simple_sie': 60,    # seconds — single SIE model
    'complex_multi': 300, # seconds — multi-component model
    'mcmc_full': 3600,    # seconds — full MCMC parameter estimation
    'default': 120,       # seconds — conservative estimate
}


def compare_parameters(
    cnn_params: np.ndarray,
    lenstool_params: np.ndarray,
    param_names: Optional[list[str]] = None,
) -> dict:
    """
    Compare CNN predictions to LENSTOOL results.

    Args:
        cnn_params: (N, 5) CNN predicted parameters in physical units.
        lenstool_params: (N, 5) LENSTOOL parameters in physical units.
        param_names: Names for each parameter.

    Returns:
        Dict of comparison metrics per parameter.
    """
    if param_names is None:
        param_names = ['Einstein_radius', 'e1', 'e2', 'gamma1', 'gamma2']

    cnn_params = np.asarray(cnn_params)
    lenstool_params = np.asarray(lenstool_params)
    diff = cnn_params - lenstool_params

    results = {}
    for i, name in enumerate(param_names):
        abs_diff = np.abs(diff[:, i])
        results[name] = {
            'mean_diff': float(np.mean(diff[:, i])),
            'std_diff': float(np.std(diff[:, i])),
            'rmse': float(np.sqrt(np.mean(diff[:, i] ** 2))),
            'median_abs_diff': float(np.median(abs_diff)),
            'max_abs_diff': float(np.max(abs_diff)),
            'correlation': float(np.corrcoef(cnn_params[:, i],
                                              lenstool_params[:, i])[0, 1])
                          if len(cnn_params) > 1 else float('nan'),
        }

    return results


def compute_speedup(
    cnn_total_seconds: float,
    n_images: int,
    lenstool_mode: str = 'default',
) -> dict:
    """
    Compute CNN vs. LENSTOOL speedup factor.

    Args:
        cnn_total_seconds: Total CNN inference time for all images.
        n_images: Number of images processed.
        lenstool_mode: LENSTOOL complexity mode.

    Returns:
        Dict with speedup metrics.
    """
    lenstool_per_image = LENSTOOL_TIMES.get(lenstool_mode, 120)
    cnn_per_image = cnn_total_seconds / max(n_images, 1)
    speedup = lenstool_per_image / max(cnn_per_image, 1e-9)

    return {
        'cnn_ms_per_image': cnn_per_image * 1000,
        'lenstool_s_per_image': lenstool_per_image,
        'lenstool_mode': lenstool_mode,
        'speedup_factor': speedup,
        'speedup_order_magnitude': np.log10(max(speedup, 1)),
    }


def load_lenstool_results(csv_path: str) -> np.ndarray:
    """
    Load pre-computed LENSTOOL results from CSV.

    Expected CSV format:
        id, theta_E, e1, e2, gamma1, gamma2
        0, 1.23, 0.05, -0.02, 0.01, -0.01
        ...

    Args:
        csv_path: Path to CSV file.

    Returns:
        (N, 5) numpy array of parameters.
    """
    import pandas as pd
    df = pd.read_csv(csv_path)

    param_cols = ['theta_E', 'e1', 'e2', 'gamma1', 'gamma2']
    available = [c for c in param_cols if c in df.columns]

    if not available:
        raise ValueError(f"No parameter columns found. Expected: {param_cols}")

    return df[available].values.astype(np.float32)


def print_comparison_report(comparison: dict, speedup: dict):
    """Pretty-print LENSTOOL comparison results."""
    print("\n" + "=" * 50)
    print("CNN vs. LENSTOOL COMPARISON")
    print("=" * 50)

    print("\nParameter Agreement:")
    for name, vals in comparison.items():
        print(f"  {name}:")
        print(f"    RMSE: {vals['rmse']:.4f}")
        print(f"    Correlation: {vals['correlation']:.4f}")

    print("\nSpeedup:")
    print(f"  CNN:      {speedup['cnn_ms_per_image']:.2f} ms/image")
    print(f"  LENSTOOL: {speedup['lenstool_s_per_image']} s/image ({speedup['lenstool_mode']})")
    print(f"  Speedup:  {speedup['speedup_factor']:,.0f}×")
    print(f"  (~10^{speedup['speedup_order_magnitude']:.1f})")
