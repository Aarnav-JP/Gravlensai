"""
World Coordinate System (WCS) helpers.
Converts between pixel coordinates and sky coordinates (RA, Dec).
"""

import numpy as np
from typing import Tuple, Optional
from astropy.wcs import WCS
from astropy.io import fits


def pixel_to_sky(
    wcs: WCS,
    x: float,
    y: float,
) -> Tuple[float, float]:
    """
    Convert pixel coordinates to sky coordinates (RA, Dec).

    Args:
        wcs: Astropy WCS object.
        x: Pixel x coordinate (0-indexed).
        y: Pixel y coordinate (0-indexed).

    Returns:
        (ra, dec) in degrees.
    """
    ra, dec = wcs.pixel_to_world_values(x, y)
    return float(ra), float(dec)


def sky_to_pixel(
    wcs: WCS,
    ra: float,
    dec: float,
) -> Tuple[float, float]:
    """
    Convert sky coordinates (RA, Dec) to pixel coordinates.

    Args:
        wcs: Astropy WCS object.
        ra: Right ascension in degrees.
        dec: Declination in degrees.

    Returns:
        (x, y) pixel coordinates (0-indexed).
    """
    x, y = wcs.world_to_pixel_values(ra, dec)
    return float(x), float(y)


def load_wcs(fits_path: str, extension: str = 'SCI') -> Optional[WCS]:
    """
    Load WCS from a FITS file.

    Args:
        fits_path: Path to FITS file.
        extension: FITS extension to read WCS from.

    Returns:
        WCS object, or None if WCS cannot be parsed.
    """
    try:
        with fits.open(fits_path) as hdul:
            if extension in hdul:
                return WCS(hdul[extension].header)
            else:
                return WCS(hdul[0].header)
    except Exception:
        return None


def cutout_wcs(
    parent_wcs: WCS,
    center_x: int,
    center_y: int,
    size: int = 64,
) -> WCS:
    """
    Create a WCS for a cutout extracted from a larger image.

    Adjusts the CRPIX values to account for the cutout offset
    while preserving the pixel scale and orientation.

    Args:
        parent_wcs: WCS of the parent (full) image.
        center_x, center_y: Pixel coordinates of the cutout center in the parent.
        size: Cutout size in pixels.

    Returns:
        New WCS object for the cutout.
    """
    half = size // 2
    new_wcs = parent_wcs.deepcopy()

    # Shift the reference pixel to account for the cutout origin
    if hasattr(new_wcs.wcs, 'crpix'):
        new_wcs.wcs.crpix[0] -= (center_x - half)
        new_wcs.wcs.crpix[1] -= (center_y - half)

    return new_wcs


def pixel_scale_arcsec(wcs: WCS) -> float:
    """
    Estimate the pixel scale in arcseconds from a WCS.

    Args:
        wcs: Astropy WCS object.

    Returns:
        Pixel scale in arcseconds/pixel.
    """
    # proj_plane_pixel_scales returns degrees per pixel
    from astropy.wcs.utils import proj_plane_pixel_scales
    scales = proj_plane_pixel_scales(wcs)
    # Average of both axes, convert to arcsec
    return float(np.mean(scales) * 3600)
