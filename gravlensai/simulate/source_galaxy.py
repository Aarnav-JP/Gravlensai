"""
Source galaxy models using Sersic profiles via GalSim.
Provides configurable bulge + disk galaxy components for simulation.

Typical ranges (SLACS-calibrated):
  - Lens galaxies: Sersic n=3–5, R_eff=0.3–1.5 arcsec (ellipticals)
  - Source galaxies: Sersic n=0.5–4, R_eff=0.1–0.5 arcsec (spirals/irregulars)
"""

import galsim
import numpy as np


def make_sersic_galaxy(
    sersic_n: float,
    half_light_radius: float,
    flux: float,
    e1: float = 0.0,
    e2: float = 0.0,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> galsim.GSObject:
    """
    Create a Sersic profile galaxy using GalSim.

    Args:
        sersic_n: Sersic index (0.5=Gaussian, 1=exponential, 4=de Vaucouleurs).
        half_light_radius: Half-light radius in arcsec.
        flux: Total flux in ADU.
        e1, e2: Ellipticity components.
        offset_x, offset_y: Position offset in arcsec.

    Returns:
        GalSim GSObject representing the galaxy.
    """
    gal = galsim.Sersic(
        n=sersic_n,
        half_light_radius=half_light_radius,
        flux=flux,
    )

    # Apply ellipticity
    if e1 != 0.0 or e2 != 0.0:
        # Clip to avoid GalSim errors for |e| >= 1
        e_mag = np.sqrt(e1**2 + e2**2)
        if e_mag > 0.7:
            scale = 0.7 / e_mag
            e1, e2 = e1 * scale, e2 * scale
        gal = gal.shear(e1=e1, e2=e2)

    # Apply offset
    if offset_x != 0.0 or offset_y != 0.0:
        gal = gal.shift(offset_x, offset_y)

    return gal


def make_lens_galaxy(
    sersic_n: float = 4.0,
    half_light_radius: float = 0.8,
    flux: float = 1e5,
    e1: float = 0.0,
    e2: float = 0.0,
) -> galsim.GSObject:
    """
    Create a typical lens (foreground elliptical) galaxy.
    Default parameters match a typical SLACS lens.
    """
    return make_sersic_galaxy(
        sersic_n=sersic_n,
        half_light_radius=half_light_radius,
        flux=flux,
        e1=e1,
        e2=e2,
    )


def make_source_galaxy(
    sersic_n: float = 1.0,
    half_light_radius: float = 0.2,
    flux: float = 1e3,
    offset_x: float = 0.0,
    offset_y: float = 0.0,
) -> galsim.GSObject:
    """
    Create a typical source (background lensed) galaxy.
    Default parameters match a typical lensed source.
    """
    return make_sersic_galaxy(
        sersic_n=sersic_n,
        half_light_radius=half_light_radius,
        flux=flux,
        offset_x=offset_x,
        offset_y=offset_y,
    )
