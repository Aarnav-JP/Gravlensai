"""
Generate the full simulated dataset.

Usage:
    python scripts/01_generate_simulations.py --n_lens 30000 --n_nonlens 30000 --output data/simulated/

For quick testing:
    python scripts/01_generate_simulations.py --n_lens 50 --n_nonlens 50 --output data/simulated/
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from gravlensai.simulate.lens_generator import LensImageGenerator
from gravlensai.utils.config import load_yaml_section
from gravlensai.utils.reproducibility import set_global_seed


def main(args):
    cfg = load_yaml_section(args.config, 'simulation')

    n_lens = args.n_lens if args.n_lens is not None else int(cfg.get('n_lens', 30000))
    n_nonlens = args.n_nonlens if args.n_nonlens is not None else int(cfg.get('n_nonlens', 30000))
    seed = args.seed if args.seed is not None else int(cfg.get('seed', 42))

    set_global_seed(seed)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    gen = LensImageGenerator(seed=seed)

    # Generate lensed images
    images_lens, params_lens = [], []
    for i in tqdm(range(n_lens), desc="Lensed"):
        img, p = gen.generate_lens()
        images_lens.append(img)
        params_lens.append(p.regression_targets())

    # Generate non-lensed images
    images_nonlens = []
    for i in tqdm(range(n_nonlens), desc="Non-lensed"):
        img, _ = gen.generate_nonlens()
        images_nonlens.append(img)

    # Stack and save
    images_lens = np.stack(images_lens)       # (N, 64, 64)
    params_lens = np.array(params_lens)       # (N, 6)
    images_nonlens = np.stack(images_nonlens) # (N, 64, 64)

    np.save(out / "images_lens.npy", images_lens)
    np.save(out / "params_lens.npy", params_lens)
    np.save(out / "images_nonlens.npy", images_nonlens)

    # Print summary
    print(f"\n{'='*50}")
    print(f"Saved {n_lens} lens + {n_nonlens} non-lens images to {out}")
    print(f"  images_lens.npy:    shape={images_lens.shape}, dtype={images_lens.dtype}")
    print(f"  params_lens.npy:    shape={params_lens.shape}, dtype={params_lens.dtype}")
    print(f"  images_nonlens.npy: shape={images_nonlens.shape}, dtype={images_nonlens.dtype}")
    print(f"\nParam ranges:")
    names = ['θ_E', 'e1', 'e2', 'γ1', 'γ2', 'M_sub']
    for i, name in enumerate(names):
        col = params_lens[:, i]
        print(f"  {name:6s}: min={col.min():.4f}, max={col.max():.4f}, mean={col.mean():.4f}")

    # Quick sanity checks
    assert images_lens.shape == (n_lens, 64, 64), f"Unexpected shape: {images_lens.shape}"
    assert not np.isnan(images_lens).any(), "NaN in lens images!"
    assert not np.isinf(images_lens).any(), "Inf in lens images!"
    print("\n✓ All sanity checks passed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate simulated lens dataset")
    parser.add_argument("--config", default="configs/simulation.yaml")
    parser.add_argument("--n_lens", type=int, default=None)
    parser.add_argument("--n_nonlens", type=int, default=None)
    parser.add_argument("--output", default="data/simulated/")
    parser.add_argument("--seed", type=int, default=None)
    main(parser.parse_args())
