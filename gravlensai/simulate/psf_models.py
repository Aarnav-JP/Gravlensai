"""
PSF simulation models.
Provides PSF kernels for both GalSim and lenstronomy pipelines.

Supports:
  - Gaussian PSF (simple, fast)
  - Kolmogorov PSF (atmospheric turbulence — space-based HST approximation)

HST ACS WFC FWHM ~ 0.1 arcsec in F814W.
"""

import numpy as np
import galsim


def make_psf_kernel(
    fwhm: float = 0.1,
    pixel_scale: float = 0.05,
    size: int = 21,
    psf_type: str = "kolmogorov",
) -> np.ndarray:
    """
    Generate a PSF kernel as a 2D numpy array.
    Used by both GalSim rendering and lenstronomy ImageModel.

    Args:
        fwhm: Full width at half maximum in arcsec.
        pixel_scale: Pixel scale in arcsec/pixel.
        size: Kernel size in pixels (should be odd).
        psf_type: 'kolmogorov' or 'gaussian'.

    Returns:
        (size, size) float64 array, normalised to sum to 1.
    """
    if psf_type == "kolmogorov":
        psf = galsim.Kolmogorov(fwhm=fwhm)
    elif psf_type == "gaussian":
        psf = galsim.Gaussian(fwhm=fwhm)
    else:
        raise ValueError(f"Unknown PSF type: {psf_type}")

    # Render into a small image
    image = galsim.Image(size, size, scale=pixel_scale)
    psf.drawImage(image=image)

    kernel = image.array.copy()
    # Normalise to unit sum
    kernel /= kernel.sum()

    return kernel


def make_galsim_psf(fwhm: float = 0.1, psf_type: str = "kolmogorov") -> galsim.GSObject:
    """
    Return a GalSim PSF object for convolution.

    Args:
        fwhm: Full width at half maximum in arcsec.
        psf_type: 'kolmogorov' or 'gaussian'.

    Returns:
        GalSim GSObject representing the PSF.
    """
    if psf_type == "kolmogorov":
        return galsim.Kolmogorov(fwhm=fwhm)
    elif psf_type == "gaussian":
        return galsim.Gaussian(fwhm=fwhm)
    else:
        raise ValueError(f"Unknown PSF type: {psf_type}")
