"""
Generate synthetic HST-analogue cutouts for domain adaptation.

Creates cutouts that resemble real HST Frontier Fields observations
with different noise profiles, PSF characteristics, and background
structure compared to the simulated training data — producing a genuine
domain gap for MMD/DANN adaptation to close.

Differences from simulation pipeline:
  - Broader PSF (HST focus varies across field)
  - Correlated background structure (ICL, cluster galaxy light)
  - Different readout noise characteristics
  - Cosmic ray artefacts (random hot pixels)
  - Slight pixel-scale variation

Usage:
    python scripts/07_generate_hst_analogues.py --n_cutouts 500 \
        --output data/raw/hst/Abell_2744/
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from gravlensai.simulate.lens_generator import LensImageGenerator
from gravlensai.utils.config import load_yaml_section
from gravlensai.utils.reproducibility import set_global_seed


def add_correlated_background(image, rng, scale=0.02):
    """
    Add correlated background structure simulating intracluster light (ICL).

    Real HST cluster images have spatially correlated low-surface-brightness
    emission from the ICL and unresolved sources.
    """
    h, w = image.shape
    # Low-frequency spatial structure
    n_modes = rng.integers(3, 8)
    bg = np.zeros_like(image)

    for _ in range(n_modes):
        freq_x = rng.uniform(0.02, 0.15)
        freq_y = rng.uniform(0.02, 0.15)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(0.5, 2.0) * scale

        y_grid, x_grid = np.mgrid[0:h, 0:w]
        bg += amp * np.sin(2 * np.pi * freq_x * x_grid + phase)
        bg += amp * np.cos(2 * np.pi * freq_y * y_grid + phase)

    return image + bg.astype(np.float32)


def add_cosmic_rays(image, rng, n_cr=3, intensity_range=(5, 50)):
    """
    Add random cosmic ray hits (hot pixels).

    Real HST images have cosmic ray artefacts that survive
    the drizzle combination process.
    """
    out = image.copy()
    h, w = out.shape
    n = rng.integers(0, n_cr + 1)

    for _ in range(n):
        y, x = rng.integers(0, h), rng.integers(0, w)
        intensity = rng.uniform(*intensity_range)
        out[y, x] += intensity
        # Sometimes bleeds to neighbours
        if rng.random() > 0.5 and y + 1 < h:
            out[y + 1, x] += intensity * 0.3
        if rng.random() > 0.5 and x + 1 < w:
            out[y, x + 1] += intensity * 0.3

    return out


def apply_hst_psf_variation(image, rng, extra_sigma=0.5):
    """
    Apply additional PSF broadening to simulate HST focus breathing.

    The HST PSF varies across the field due to thermal cycling
    ('focus breathing') and off-axis aberrations.
    """
    from scipy.ndimage import gaussian_filter
    sigma = rng.uniform(0.2, extra_sigma)
    return gaussian_filter(image, sigma=sigma)


def adjust_noise_profile(image, rng, sky_factor=1.5, read_factor=1.3):
    """
    Add extra noise to simulate different observation depth.

    Real HST images have different effective exposure times
    and background levels compared to our idealised simulations.
    """
    # Extra sky background (Poisson-like)
    sky_extra = rng.poisson(lam=0.01 * sky_factor, size=image.shape).astype(np.float32)
    # Extra readout noise
    read_extra = rng.normal(0, 0.005 * read_factor, size=image.shape).astype(np.float32)

    return image + sky_extra + read_extra


def generate_hst_analogues(
    n_cutouts: int = 500,
    output_dir: str = 'data/raw/hst/Abell_2744/',
    seed: int = 42,
):
    """
    Generate synthetic HST-analogue cutouts with realistic domain gap.

    These cutouts use the same physics engine but with modified
    noise, PSF, and background characteristics to create a genuine
    distribution shift that domain adaptation can address.
    """
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    set_global_seed(seed)
    rng = np.random.default_rng(seed)
    gen = LensImageGenerator(seed=seed + 1000)  # Different seed from training data

    print(f"Generating {n_cutouts} HST-analogue cutouts...")
    print(f"Output: {output}")
    print(f"Domain gap features: ICL background, cosmic rays, PSF variation, extra noise")

    for i in tqdm(range(n_cutouts), desc="Generating"):
        # Mix of lensed and non-lensed (real survey has mostly non-lenses)
        if rng.random() < 0.15:  # ~15% lens fraction (realistic for cluster core)
            img, _ = gen.generate_lens()
        else:
            img, _ = gen.generate_nonlens()

        # Apply domain-shifting transforms
        img = add_correlated_background(img, rng, scale=rng.uniform(0.01, 0.05))
        img = apply_hst_psf_variation(img, rng, extra_sigma=rng.uniform(0.3, 0.8))
        img = adjust_noise_profile(img, rng,
                                    sky_factor=rng.uniform(1.0, 2.0),
                                    read_factor=rng.uniform(1.0, 2.0))
        img = add_cosmic_rays(img, rng, n_cr=rng.integers(0, 5))

        np.save(output / f"cutout_{i:06d}.npy", img.astype(np.float32))

    print(f"\n✓ Generated {n_cutouts} HST-analogue cutouts")
    print(f"  Saved to: {output}")

    # Verify
    sample = np.load(output / "cutout_000000.npy")
    print(f"  Sample shape: {sample.shape}, dtype: {sample.dtype}")
    print(f"  Range: [{sample.min():.4f}, {sample.max():.4f}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate HST-analogue cutouts for domain adaptation"
    )
    parser.add_argument("--config", default="configs/hst_analogues.yaml")
    parser.add_argument("--n_cutouts", type=int, default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_yaml_section(args.config, 'hst_analogues')
    n_cutouts = args.n_cutouts if args.n_cutouts is not None else int(cfg.get('n_cutouts', 500))
    output_dir = args.output or cfg.get('output', 'data/raw/hst/Abell_2744/')
    seed = args.seed if args.seed is not None else int(cfg.get('seed', 42))

    generate_hst_analogues(
        n_cutouts=n_cutouts,
        output_dir=output_dir,
        seed=seed,
    )
