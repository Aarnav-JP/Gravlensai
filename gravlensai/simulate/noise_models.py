"""
Noise models: sky background (Poisson) and CCD readout (Gaussian).
Calibrated to HST ACS WFC F814W band characteristics.

HST ACS WFC F814W typical values:
  - Sky background: ~26.3 ADU/pixel
  - Read noise: ~4.0 electrons rms/pixel
  - Gain: ~2.0 electrons/ADU
"""

import galsim
import numpy as np


def add_noise(
    image: galsim.Image,
    rng: galsim.BaseDeviate,
    sky_level: float = 26.3,
    read_noise: float = 4.0,
    gain: float = 2.0,
) -> galsim.Image:
    """
    Add Poisson sky noise + Gaussian read noise to a GalSim image,
    matching HST ACS WFC F814W characteristics.

    Args:
        image: GalSim Image to add noise to (modified in-place).
        rng: GalSim random number generator.
        sky_level: Sky background level in ADU/pixel.
        read_noise: Read noise in electrons rms/pixel.
        gain: Detector gain in electrons/ADU.

    Returns:
        The modified image (same object, for chaining).
    """
    # Poisson noise from sky background
    sky_noise = galsim.PoissonNoise(rng, sky_level=sky_level)
    image.addNoise(sky_noise)

    # Gaussian read noise (converted from electrons to ADU)
    read_noise_adu = read_noise / gain
    gauss_noise = galsim.GaussianNoise(rng, sigma=read_noise_adu)
    image.addNoise(gauss_noise)

    return image


def estimate_noise_level(
    sky_level: float = 26.3,
    read_noise: float = 4.0,
    gain: float = 2.0,
) -> float:
    """
    Estimate the total noise standard deviation per pixel in ADU.
    Useful for setting detection thresholds.

    σ_total = sqrt(sky_level + (read_noise/gain)^2)
    """
    return np.sqrt(sky_level + (read_noise / gain) ** 2)
