"""
FITS I/O helpers for reading/writing astronomical image files.
Wraps astropy.io.fits for common operations.
"""

import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from astropy.io import fits


def read_fits(
    fits_path: str,
    extension: str = 'SCI',
) -> Tuple[np.ndarray, dict]:
    """
    Read a FITS image and return the data array and header as dict.

    Args:
        fits_path: Path to the FITS file.
        extension: FITS extension name (default 'SCI' for HST drizzled products).

    Returns:
        (data, header_dict): numpy array and header dictionary.
    """
    with fits.open(fits_path) as hdul:
        if extension in hdul:
            data = hdul[extension].data.astype(np.float32)
            header = dict(hdul[extension].header)
        elif len(hdul) > 1:
            data = hdul[1].data.astype(np.float32)
            header = dict(hdul[1].header)
        else:
            data = hdul[0].data.astype(np.float32)
            header = dict(hdul[0].header)

    return data, header


def write_fits(
    data: np.ndarray,
    fits_path: str,
    header: Optional[Dict[str, Any]] = None,
    overwrite: bool = True,
) -> None:
    """
    Write a numpy array to a FITS file.

    Args:
        data: 2D numpy array.
        fits_path: Output path.
        header: Optional header dictionary.
        overwrite: Whether to overwrite existing files.
    """
    path = Path(fits_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    hdu = fits.PrimaryHDU(data)
    if header:
        for key, val in header.items():
            # Skip reserved FITS keywords
            if key.upper() in ('SIMPLE', 'BITPIX', 'NAXIS', 'NAXIS1', 'NAXIS2',
                               'EXTEND', 'COMMENT', 'HISTORY', ''):
                continue
            try:
                hdu.header[key] = val
            except (ValueError, TypeError):
                pass

    hdu.writeto(str(path), overwrite=overwrite)


def fits_info(fits_path: str) -> Dict[str, Any]:
    """
    Get summary information about a FITS file.

    Args:
        fits_path: Path to the FITS file.

    Returns:
        Dict with file summary info.
    """
    info: Dict[str, Any] = {'path': str(fits_path)}

    with fits.open(fits_path) as hdul:
        info['n_extensions'] = len(hdul)
        info['extensions'] = []
        for i, hdu in enumerate(hdul):
            ext_info = {
                'index': i,
                'name': hdu.name,
                'type': type(hdu).__name__,
            }
            if hdu.data is not None:
                ext_info['shape'] = hdu.data.shape
                ext_info['dtype'] = str(hdu.data.dtype)
            info['extensions'].append(ext_info)

    return info


def extract_subimage(
    data: np.ndarray,
    center_y: int,
    center_x: int,
    size: int = 64,
) -> Optional[np.ndarray]:
    """
    Extract a square subimage from a larger 2D array.

    Args:
        data: 2D image array.
        center_y, center_x: Pixel coordinates of subimage center.
        size: Subimage size in pixels (square).

    Returns:
        (size, size) float32 array, or None if invalid.
    """
    half = size // 2
    h, w = data.shape

    if (center_y - half < 0 or center_y + half > h or
            center_x - half < 0 or center_x + half > w):
        return None

    sub = data[center_y - half:center_y + half,
               center_x - half:center_x + half].copy()

    if sub.shape != (size, size):
        return None
    if np.isnan(sub).any():
        return None

    return sub.astype(np.float32)
