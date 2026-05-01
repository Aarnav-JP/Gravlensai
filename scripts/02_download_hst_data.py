"""
Download HST Frontier Fields cutouts via astroquery (MAST archive).
Free, no API key required.

Downloads:
  - 6 cluster fields (Abell 2744, MACSJ0416, MACSJ0717, MACSJ1149, AbellS1063, Abell 370)
  - F814W band (I-band, best for lens arcs)
  - Full FITS mosaics -> cut 64×64 postage stamps around detected sources

Usage:
    # Step 1: Download a Frontier Fields mosaic (manual or via script)
    python scripts/02_download_hst_data.py --download --cluster "Abell 2744" --output data/raw/hst/

    # Step 2: Extract cutouts from a downloaded FITS mosaic
    python scripts/02_download_hst_data.py --extract --fits_path data/raw/hst/abell2744_mosaic.fits \\
        --output data/raw/hst/Abell_2744/ --n_cutouts 500

    # Step 3: Verify the HSTDataset loads them
    python scripts/02_download_hst_data.py --verify --cutout_dir data/raw/hst/Abell_2744/

Note: Full Frontier Fields mosaics are ~4GB each. Start with Abell 2744 only.
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

from gravlensai.utils.config import load_yaml_section
from gravlensai.utils.reproducibility import set_global_seed


# ── Frontier Fields cluster coordinates ──────────────────────────────────

FRONTIER_FIELDS = {
    'Abell 2744':  {'ra': 3.5858,   'dec': -30.3933},
    'MACSJ0416':   {'ra': 64.0336,  'dec': -24.0725},
    'MACSJ0717':   {'ra': 109.3797, 'dec': 37.7452},
    'MACSJ1149':   {'ra': 177.3981, 'dec': 22.3972},
    'AbellS1063':  {'ra': 342.1833, 'dec': -44.5306},
    'Abell 370':   {'ra': 39.9700,  'dec': -1.5783},
}

CUTOUT_SIZE = 64  # pixels (matches simulation size)


# ── Download via astroquery ──────────────────────────────────────────────

def download_frontier_field(cluster_name: str, output_dir: str):
    """
    Query MAST for HST ACS F814W observations of a Frontier Field cluster.
    Downloads the drizzled mosaic FITS file.
    """
    from astroquery.mast import Observations
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    coords = FRONTIER_FIELDS[cluster_name]
    center = SkyCoord(ra=coords['ra'], dec=coords['dec'], unit='deg')

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Querying MAST for {cluster_name} (F814W)...")
    obs_table = Observations.query_criteria(
        coordinates=center,
        radius=3 * u.arcmin,
        obs_collection="HST",
        filters="F814W",
        dataproduct_type="image",
    )

    if len(obs_table) == 0:
        print(f"  No observations found for {cluster_name}")
        return None

    print(f"  Found {len(obs_table)} observations")

    # Get products for the first (deepest) observation
    products = Observations.get_product_list(obs_table[:1])
    drizzled = products[products['productSubGroupDescription'] == 'DRZ']

    if len(drizzled) == 0:
        print(f"  No drizzled product found for {cluster_name}")
        return None

    print(f"  Downloading {cluster_name} F814W mosaic (~4GB)...")
    manifest = Observations.download_products(
        drizzled[:1],
        download_dir=str(output_path),
    )

    print(f"  Download complete: {manifest['Local Path'][0]}")
    return manifest['Local Path'][0]


# ── Cutout extraction ────────────────────────────────────────────────────

def extract_cutouts_from_fits(
    fits_path: str,
    output_dir: str,
    n_cutouts: int = 500,
    cutout_size: int = 64,
    seed: int = 42,
):
    """
    Given a downloaded Frontier Fields FITS mosaic, extract random 64×64 cutouts
    covering the cluster region (where lenses are most likely).

    Skips:
      - Cutouts with NaN pixels (edge/gap regions)
      - Flat/empty cutouts (sky-only, std < threshold)

    Args:
        fits_path: Path to the FITS mosaic file.
        output_dir: Directory to save .npy cutout files.
        n_cutouts: Number of cutouts to extract.
        cutout_size: Size of each cutout in pixels.
        seed: Random seed for reproducibility.
    """
    from astropy.io import fits

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    np.random.seed(seed)

    print(f"Opening {fits_path}...")
    with fits.open(fits_path) as hdul:
        # Try SCI extension, fall back to primary
        if 'SCI' in hdul:
            sci = hdul['SCI'].data.astype(np.float32)
        else:
            sci = hdul[0].data.astype(np.float32)

    h, w = sci.shape
    margin = cutout_size // 2
    print(f"  Mosaic size: {w}×{h} pixels")
    print(f"  Extracting {n_cutouts} cutouts of {cutout_size}×{cutout_size}...")

    saved = 0
    attempts = 0
    max_attempts = n_cutouts * 20  # oversample to account for bad regions

    pbar = tqdm(total=n_cutouts, desc="Extracting")
    while saved < n_cutouts and attempts < max_attempts:
        attempts += 1

        y = np.random.randint(margin, h - margin)
        x = np.random.randint(margin, w - margin)
        cutout = sci[y - margin:y + margin, x - margin:x + margin]

        # Validation checks
        if cutout.shape != (cutout_size, cutout_size):
            continue
        if np.isnan(cutout).any():
            continue
        if cutout.std() < 1e-6:  # skip flat/empty regions
            continue

        np.save(output_path / f"cutout_{saved:06d}.npy", cutout)
        saved += 1
        pbar.update(1)

    pbar.close()
    print(f"  Extracted {saved} valid cutouts from {Path(fits_path).name}")
    print(f"  ({attempts} total attempts, {attempts - saved} rejected)")
    return saved


# ── Verification ─────────────────────────────────────────────────────────

def verify_cutouts(cutout_dir: str):
    """Verify that cutouts load correctly in HSTDataset."""
    from gravlensai.data.dataset import HSTDataset

    ds = HSTDataset(cutout_dir)
    print(f"HSTDataset loaded: {len(ds)} cutouts from {cutout_dir}")

    if len(ds) == 0:
        print("  ⚠ No cutouts found!")
        return

    # Check a batch
    img, label = ds[0]
    print(f"  Shape: {img.shape} (expected: [1, 64, 64])")
    print(f"  Dtype: {img.dtype}")
    print(f"  Range: [{img.min():.3f}, {img.max():.3f}]")
    print(f"  NaN: {img.isnan().any()}, Inf: {img.isinf().any()}")

    # Check multiple
    all_ok = True
    for i in range(min(10, len(ds))):
        img, _ = ds[i]
        if img.isnan().any() or img.isinf().any():
            print(f"  ✗ cutout {i} has NaN/Inf!")
            all_ok = False

    if all_ok:
        print("  ✓ All checked cutouts are valid!")


# ── Main ─────────────────────────────────────────────────────────────────

def main(args):
    cfg = load_yaml_section(args.config, 'hst_data')

    seed = args.seed if args.seed is not None else int(cfg.get('seed', 42))
    output_dir = args.output or cfg.get('output', 'data/raw/hst/')
    n_cutouts = args.n_cutouts if args.n_cutouts is not None else int(cfg.get('n_cutouts', 500))
    cluster = args.cluster or cfg.get('cluster', 'Abell 2744')

    set_global_seed(seed)

    if args.download:
        if cluster not in FRONTIER_FIELDS:
            print(f"Unknown cluster: {cluster}")
            print(f"Available: {list(FRONTIER_FIELDS.keys())}")
            return
        download_frontier_field(cluster, output_dir)

    elif args.extract:
        if not args.fits_path:
            print("Error: --fits_path required for --extract mode")
            return
        extract_cutouts_from_fits(
            args.fits_path,
            output_dir,
            n_cutouts=n_cutouts,
            seed=seed,
        )

    elif args.verify:
        cutout_dir = args.cutout_dir or output_dir
        verify_cutouts(cutout_dir)

    else:
        # Default: print instructions
        print("HST Frontier Fields Data Download")
        print("=" * 50)
        print()
        print("Option 1: Download via astroquery (slow, ~4GB per cluster):")
        print("  python scripts/02_download_hst_data.py --download --cluster 'Abell 2744'")
        print()
        print("Option 2: Manual download from frontierfields.org (recommended):")
        print("  Visit: https://frontierfields.org/data-access/")
        print("  Download the F814W drizzled mosaic for your cluster")
        print("  Then extract cutouts:")
        print("  python scripts/02_download_hst_data.py --extract \\")
        print("      --fits_path /path/to/mosaic.fits \\")
        print("      --output data/raw/hst/Abell_2744/ --n_cutouts 500")
        print()
        print("Available clusters:")
        for name, coords in FRONTIER_FIELDS.items():
            print(f"  {name:15s}  RA={coords['ra']:.4f}, Dec={coords['dec']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download HST Frontier Fields data")
    parser.add_argument("--config", default="configs/hst_data.yaml")
    parser.add_argument("--download", action="store_true", help="Download via MAST")
    parser.add_argument("--extract", action="store_true", help="Extract cutouts from FITS")
    parser.add_argument("--verify", action="store_true", help="Verify HSTDataset loading")
    parser.add_argument("--cluster", type=str, help="Cluster name (for download)")
    parser.add_argument("--fits_path", type=str, help="Path to FITS mosaic (for extract)")
    parser.add_argument("--cutout_dir", type=str, help="Path to cutouts (for verify)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--n_cutouts", type=int, default=None, help="Cutouts to extract")
    parser.add_argument("--seed", type=int, default=None)
    main(parser.parse_args())
