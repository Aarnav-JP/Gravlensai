"""
Interactive demo: run the full GravLensAI pipeline on a single image.

Modes:
  --simulate: Generate a new simulated lens, classify and estimate parameters
  --image:    Run on a user-provided FITS or .npy image

Usage:
    python scripts/demo.py --simulate
    python scripts/demo.py --image path/to/image.npy
"""

import argparse
import os
import torch
import numpy as np
import time
from pathlib import Path

from gravlensai.simulate.lens_generator import LensImageGenerator
from gravlensai.models.classifier import LensClassifier
from gravlensai.models.regressor import LensParameterRegressor
from gravlensai.data.normalisation import normalise_single
from gravlensai.utils.config import load_yaml_section
from gravlensai.utils.reproducibility import set_global_seed


def _infer_regressor_output_dim(state_dict):
    candidate_dims = []
    for key, value in state_dict.items():
        if key.startswith('regressor.') and key.endswith('.weight') and getattr(value, 'ndim', 0) == 2:
            candidate_dims.append((value.shape[0], key))
    if not candidate_dims:
        return 6
    return min(candidate_dims, key=lambda item: item[0])[0]


def load_models(models_dir, device):
    """Load classifier and regressor from checkpoints."""
    models_dir = Path(models_dir)
    models = {}

    clf_path = models_dir / "classifier_best.pt"
    if clf_path.exists():
        clf = LensClassifier()
        ckpt = torch.load(clf_path, map_location=device, weights_only=False)
        clf.load_state_dict(ckpt['model_state'], strict=False)
        clf.to(device).eval()
        models['classifier'] = clf
        print(f"  ✓ Classifier loaded (epoch {ckpt.get('epoch', '?')})")

    reg_path = models_dir / "regressor_best.pt"
    if reg_path.exists():
        ckpt = torch.load(reg_path, map_location=device, weights_only=False)
        output_dim = _infer_regressor_output_dim(ckpt['model_state'])
        reg = LensParameterRegressor(output_dim=output_dim)
        reg.load_state_dict(ckpt['model_state'])
        reg.to(device).eval()
        models['regressor'] = reg
        print(f"  ✓ Regressor loaded (epoch {ckpt.get('epoch', '?')}, outputs={output_dim})")

    return models


def run_inference(image_raw, models, device):
    """Run classification + regression on a single image."""
    # Normalise
    img_norm = normalise_single(image_raw)
    img_tensor = torch.from_numpy(img_norm).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,64,64)

    results = {}

    # Classification
    if 'classifier' in models:
        t0 = time.time()
        with torch.no_grad():
            prob = models['classifier'].predict_proba(img_tensor).item()
        t_clf = (time.time() - t0) * 1000
        results['is_lens'] = prob > 0.5
        results['lens_probability'] = prob
        results['classifier_ms'] = t_clf

    # Regression (only if classified as lens or forced)
    if 'regressor' in models:
        t0 = time.time()
        with torch.no_grad():
            pred_norm = models['regressor'](img_tensor)
            pred_phys = models['regressor'].denormalise(pred_norm)
        t_reg = (time.time() - t0) * 1000
        params = pred_phys.cpu().squeeze().numpy()
        parameter_names = ['Einstein_radius', 'e1', 'e2', 'gamma1', 'gamma2']
        if params.shape[0] > 5:
            parameter_names.append('subhalo_mass')
        results['parameters'] = {
            name: float(value) for name, value in zip(parameter_names, params)
        }
        results['regressor_ms'] = t_reg

    return results


def demo_simulate(models, device, seed=None, lenstool_seconds_per_image: float = 120.0):
    """Generate and analyse a simulated lens."""
    if seed is None:
        seed = int(time.time()) % 100000

    gen = LensImageGenerator(seed=seed)
    image, true_params = gen.generate_lens()

    print(f"\n{'='*50}")
    print("SIMULATED LENS DEMO")
    print(f"{'='*50}")
    print(f"Generated a 64×64 simulated lens image (seed={seed})")

    if true_params:
        print("\nGround Truth:")
        tgt = true_params.regression_targets(include_subhalo='regressor' in models and models['regressor'].output_dim > 5)
        names = ['θ_E', 'e1', 'e2', 'γ1', 'γ2']
        if len(tgt) > 5:
            names.append('M_sub')
        for name, val in zip(names, tgt):
            print(f"  {name:4s}: {val:+.4f}")

    results = run_inference(image, models, device)

    if 'lens_probability' in results:
        prob = results['lens_probability']
        verdict = "LENS ✓" if results['is_lens'] else "NON-LENS"
        print(f"\nClassifier: {verdict} (p={prob:.4f}, {results['classifier_ms']:.1f} ms)")

    if 'parameters' in results:
        print(f"\nPredicted Parameters ({results['regressor_ms']:.1f} ms):")
        for name, val in results['parameters'].items():
            print(f"  {name:16s}: {val:+.4f}")

        # Compare if ground truth available
        if true_params:
            print("\nErrors:")
            pred_vals = list(results['parameters'].values())
            for name, pred, true in zip(names, pred_vals, tgt):
                error = abs(pred - true)
                print(f"  {name:4s}: |Δ| = {error:.4f}")

    total_ms = results.get('classifier_ms', 0) + results.get('regressor_ms', 0)
    print(f"\nTotal inference: {total_ms:.1f} ms")
    print(
        f"LENSTOOL equivalent: ~{lenstool_seconds_per_image:.1f} s "
        f"→ speedup ≈ {(lenstool_seconds_per_image * 1000)/max(total_ms,1):,.0f}×"
    )


def demo_image(image_path, models, device):
    """Analyse a user-provided image."""
    path = Path(image_path)

    if path.suffix == '.npy':
        image = np.load(path).astype(np.float32)
    elif path.suffix in ('.fits', '.fit'):
        from astropy.io import fits
        with fits.open(path) as hdul:
            image = hdul[0].data.astype(np.float32)
    else:
        raise ValueError(f"Unsupported format: {path.suffix}. Use .npy or .fits")

    # Crop/resize to 64×64 if needed
    if image.shape != (64, 64):
        h, w = image.shape
        cy, cx = h // 2, w // 2
        image = image[cy-32:cy+32, cx-32:cx+32]

    print(f"\n{'='*50}")
    print(f"IMAGE ANALYSIS: {path.name}")
    print(f"{'='*50}")
    print(f"Shape: {image.shape}, range: [{image.min():.2f}, {image.max():.2f}]")

    results = run_inference(image, models, device)

    if 'lens_probability' in results:
        prob = results['lens_probability']
        verdict = "GRAVITATIONAL LENS DETECTED ✓" if results['is_lens'] else "NOT A LENS"
        print(f"\n{verdict}")
        print(f"  Confidence: {prob:.4f} ({results['classifier_ms']:.1f} ms)")

    if results.get('is_lens', True) and 'parameters' in results:
        print("\nEstimated Lens Parameters:")
        for name, val in results['parameters'].items():
            print(f"  {name:16s}: {val:+.4f}")


def main(args):
    cfg = load_yaml_section(args.config, 'demo')
    seed = args.seed if args.seed is not None else cfg.get('seed', None)
    lenstool_seconds = (
        args.lenstool_seconds_per_image
        if args.lenstool_seconds_per_image is not None
        else float(cfg.get('lenstool_seconds_per_image', 120.0))
    )
    models_dir = (
        args.models_dir
        or os.getenv('GRAVLENS_MODELS_DIR')
        or cfg.get('models_dir', 'results/models/')
    )

    if seed is not None:
        set_global_seed(int(seed))

    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps' if torch.backends.mps.is_available() else 'cpu'
    )
    print(f"Device: {device}")
    print("Loading models...")
    models = load_models(models_dir, device)

    if not models:
        print("\n⚠ No trained models found! Run training first:")
        print("  python scripts/03_train_classifier.py")
        print("  python scripts/04_train_regressor.py")
        return

    if args.image:
        demo_image(args.image, models, device)
    else:
        demo_simulate(
            models,
            device,
            seed=int(seed) if seed is not None else None,
            lenstool_seconds_per_image=lenstool_seconds,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GravLensAI Demo")
    parser.add_argument("--config", default="configs/demo.yaml")
    parser.add_argument("--simulate", action="store_true", default=True)
    parser.add_argument("--image", type=str, help="Path to .npy or .fits image")
    parser.add_argument("--models_dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--lenstool_seconds_per_image", type=float, default=None)
    main(parser.parse_args())
