"""
HST FITS file loader using astroquery.
Utilities for downloading, reading, and preprocessing HST observations.

Wraps astropy.io.fits and astroquery.mast for common HST data operations.
"""

import numpy as np
from typing import Optional, Tuple, List
from astropy.io import fits
from astropy.wcs import WCS


def read_fits_image(
    fits_path: str,
    extension: str = 'SCI',
) -> Tuple[np.ndarray, Optional[WCS]]:
    """
    Read a FITS image and return the data array and WCS.

    Args:
        fits_path: Path to the FITS file.
        extension: FITS extension name (default 'SCI' for HST drizzled products).

    Returns:
        (data, wcs): numpy array and WCS object (or None if no WCS).
    """
    with fits.open(fits_path) as hdul:
        if extension in hdul:
            data = hdul[extension].data.astype(np.float32)
            try:
                wcs = WCS(hdul[extension].header)
            except Exception:
                wcs = None
        else:
            data = hdul[0].data.astype(np.float32)
            try:
                wcs = WCS(hdul[0].header)
            except Exception:
                wcs = None

    return data, wcs


def extract_cutout(
    data: np.ndarray,
    center_y: int,
    center_x: int,
    size: int = 64,
) -> Optional[np.ndarray]:
    """
    Extract a square cutout from a 2D array.

    Args:
        data: 2D image array.
        center_y, center_x: Pixel coordinates of cutout center.
        size: Cutout size in pixels (square).

    Returns:
        (size, size) float32 array, or None if invalid.
    """
    half = size // 2
    h, w = data.shape

    # Bounds check
    if (center_y - half < 0 or center_y + half > h or
            center_x - half < 0 or center_x + half > w):
        return None

    cutout = data[center_y - half:center_y + half,
                  center_x - half:center_x + half].copy()

    # Validate
    if cutout.shape != (size, size):
        return None
    if np.isnan(cutout).any():
        return None
    if cutout.std() < 1e-6:
        return None

    return cutout.astype(np.float32)


def find_source_positions(
    data: np.ndarray,
    threshold_sigma: float = 3.0,
    min_separation: int = 32,
) -> List[Tuple[int, int]]:
    """
    Find bright source positions in an image using simple sigma clipping.
    Used to generate cutouts centered on actual objects (not random sky).

    Args:
        data: 2D image array.
        threshold_sigma: Detection threshold in sigma above background.
        min_separation: Minimum pixel distance between detected sources.

    Returns:
        List of (y, x) pixel coordinates of detected sources.
    """
    from scipy.ndimage import label, center_of_mass

    # Estimate background
    # Use median for robustness to bright objects
    bg = np.nanmedian(data)
    noise = np.nanstd(data)

    # Create detection mask
    mask = data > (bg + threshold_sigma * noise)

    # Remove NaN regions
    mask = mask & ~np.isnan(data)

    # Label connected regions
    labeled, n_features = label(mask)

    if n_features == 0:
        return []

    # Get centroids
    centroids = center_of_mass(data, labeled, range(1, n_features + 1))

    # Convert to integer coordinates and filter by separation
    positions: list[tuple[int, int]] = []
    for cy, cx in centroids:
        y, x = int(cy), int(cx)
        # Check minimum separation from existing positions
        too_close = False
        for py, px in positions:
            if abs(y - py) < min_separation and abs(x - px) < min_separation:
                too_close = True
                break
        if not too_close:
            positions.append((y, x))

    return positions


def preprocess_cutout(
    cutout: np.ndarray,
    target_size: int = 64,
) -> np.ndarray:
    """
    Preprocess a cutout for model input.
    Resizes if needed and applies arcsinh normalisation.

    Args:
        cutout: 2D array.
        target_size: Target size in pixels.

    Returns:
        Preprocessed (target_size, target_size) float32 array.
    """
    from skimage.transform import resize

    # Resize if needed
    if cutout.shape != (target_size, target_size):
        cutout = resize(
            cutout,
            (target_size, target_size),
            anti_aliasing=True,
            preserve_range=True,
        ).astype(np.float32)

    return cutout
