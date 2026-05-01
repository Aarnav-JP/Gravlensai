"""
Lens image generator using GalSim + lenstronomy.
Simulates strong gravitational lensing images matching HST ACS WFC characteristics.

Physics:
  - SIE lens model (Singular Isothermal Ellipsoid) via lenstronomy ray-tracing
  - Sersic source galaxy profiles via GalSim
  - Kolmogorov PSF matching HST ACS average
  - Poisson + readout noise matching F814W band

Reference: Hezaveh et al. 2017 (Nature), Metcalf et al. 2019 (A&A)
"""

import galsim
import numpy as np
from dataclasses import dataclass
from typing import Tuple

from lenstronomy.LensModel.lens_model import LensModel
from lenstronomy.LightModel.light_model import LightModel
from lenstronomy.ImSim.image_model import ImageModel
from lenstronomy.Data.imaging_data import ImageData
from lenstronomy.Data.psf import PSF as LenstronomyPSF
from lenstronomy.Util.simulation_util import data_configure_simple

from gravlensai.simulate.psf_models import make_psf_kernel
from gravlensai.simulate.noise_models import add_noise


@dataclass
class LensParameters:
    """
    All physical parameters for one simulated lens system.
    Units: angles in arcseconds, mass in solar masses.
    """
    einstein_radius: float      # θ_E in arcsec, range [0.5, 2.5]
    ellipticity_e1: float       # e1 component, range [-0.3, 0.3]
    ellipticity_e2: float       # e2 component, range [-0.3, 0.3]
    shear_g1: float             # External shear g1, range [-0.05, 0.05]
    shear_g2: float             # External shear g2, range [-0.05, 0.05]
    lens_sersic_n: float        # Lens galaxy Sersic index, range [3, 5]
    lens_half_light: float      # Lens half-light radius arcsec, range [0.3, 1.5]
    source_sersic_n: float      # Source galaxy Sersic index, range [0.5, 4]
    source_half_light: float    # Source half-light radius arcsec, range [0.1, 0.5]
    source_offset_x: float      # Source position offset x arcsec, range [-0.3, 0.3]
    source_offset_y: float      # Source position offset y arcsec, range [-0.3, 0.3]
    flux_lens: float            # Lens galaxy total flux (ADU)
    flux_source: float          # Source galaxy total flux (ADU)
    
    # Dark Matter Subhalo Properties
    subhalo_mass: float         # Log10(M_sun), range [8.0, 10.0]
    subhalo_x: float            # Subhalo x position arcsec
    subhalo_y: float            # Subhalo y position arcsec

    def regression_targets(self, include_subhalo: bool = True) -> np.ndarray:
        """Return the regression targets used by the training pipeline.

        The default contract is the expanded 6-parameter target vector. Pass
        ``include_subhalo=False`` only when you need the legacy 5-parameter
        lens-only targets.
        """
        targets = [
            self.einstein_radius,
            self.ellipticity_e1,
            self.ellipticity_e2,
            self.shear_g1,
            self.shear_g2,
        ]
        if include_subhalo:
            targets.append(self.subhalo_mass)
        return np.array(targets, dtype=np.float32)


class LensImageGenerator:
    """
    Generates simulated strong gravitational lens images using
    lenstronomy (ray-tracing) + GalSim (noise/PSF).

    Each call to generate_lens() or generate_nonlens() returns a 64×64 numpy
    array and the associated LensParameters (None for non-lens images).

    Pixel scale: 0.05 arcsec/pixel (HST ACS WFC F814W default)
    Image size: 64×64 pixels = 3.2 × 3.2 arcsec field of view
    """

    PIXEL_SCALE = 0.05      # arcsec/pixel — HST ACS WFC
    IMAGE_SIZE = 64          # pixels
    SKY_LEVEL = 26.3         # ADU/pixel (HST ACS WFC F814W typical background)
    READ_NOISE = 4.0         # electrons rms per pixel
    GAIN = 2.0               # electrons per ADU

    def __init__(self, seed: int = 42):
        self.rng = galsim.BaseDeviate(seed)
        self.np_rng = np.random.default_rng(seed)

    def _sample_lens_params(self) -> LensParameters:
        """
        Sample lens parameters from physically motivated prior distributions.
        Ranges calibrated to match the SLACS lens population.
        """
        rng = self.np_rng
        return LensParameters(
            einstein_radius=rng.uniform(0.5, 2.5),
            ellipticity_e1=float(np.clip(rng.normal(0, 0.1), -0.3, 0.3)),
            ellipticity_e2=float(np.clip(rng.normal(0, 0.1), -0.3, 0.3)),
            shear_g1=float(np.clip(rng.normal(0, 0.02), -0.05, 0.05)),
            shear_g2=float(np.clip(rng.normal(0, 0.02), -0.05, 0.05)),
            lens_sersic_n=rng.uniform(3.0, 5.0),
            lens_half_light=rng.uniform(0.3, 1.5),
            source_sersic_n=rng.uniform(0.5, 4.0),
            source_half_light=rng.uniform(0.1, 0.5),
            source_offset_x=rng.uniform(-0.3, 0.3),
            source_offset_y=rng.uniform(-0.3, 0.3),
            flux_lens=rng.uniform(1e4, 1e6),
            flux_source=rng.uniform(5e2, 1e4),
            subhalo_mass=rng.uniform(8.0, 10.0),  # Log10(M_sun) proxy
            subhalo_x=rng.uniform(-1.5, 1.5),
            subhalo_y=rng.uniform(-1.5, 1.5),
        )

    def _render_lensed_source(self, params: LensParameters) -> np.ndarray:
        """
        Use lenstronomy to ray-trace the source through an SIE lens.
        Returns the lensed source image as a (64, 64) array.
        """
        # Configure the data/coordinate grid
        kwargs_data = data_configure_simple(
            numPix=self.IMAGE_SIZE,
            deltaPix=self.PIXEL_SCALE,
            exposure_time=1.0,
            background_rms=1.0,
        )
        data_class = ImageData(**kwargs_data)

        # PSF for lenstronomy (Gaussian approximation for speed)
        psf_kernel = make_psf_kernel(
            fwhm=0.1,
            pixel_scale=self.PIXEL_SCALE,
            size=21,
        )
        kwargs_psf = {'psf_type': 'PIXEL', 'kernel_point_source': psf_kernel}
        psf_class = LenstronomyPSF(**kwargs_psf)

        # --- Lens model: SIE + external shear + Dark Matter NFW Subhalo ---
        lens_model = LensModel(lens_model_list=['SIE', 'SHEAR', 'NFW'])
        
        # Convert subhalo mass proxy to NFW deflection angle
        # 10^8 M_sun ~ 0.005 arcsec, 10^10 M_sun ~ 0.05 arcsec (roughly linear in log)
        alpha_Rs = 0.005 + (params.subhalo_mass - 8.0) * (0.045 / 2.0)
        Rs = 0.1  # Fixed scale radius for simplicity

        kwargs_lens = [
            {
                'theta_E': params.einstein_radius,
                'e1': params.ellipticity_e1,
                'e2': params.ellipticity_e2,
                'center_x': 0.0,
                'center_y': 0.0,
            },
            {
                'gamma1': params.shear_g1,
                'gamma2': params.shear_g2,
                'ra_0': 0.0,
                'dec_0': 0.0,
            },
            {
                'Rs': Rs,
                'alpha_Rs': alpha_Rs,
                'center_x': params.subhalo_x,
                'center_y': params.subhalo_y,
            }
        ]

        # --- Source light: Sersic profile ---
        source_light_model = LightModel(light_model_list=['SERSIC_ELLIPSE'])
        # Convert GalSim-style flux to lenstronomy amplitude (approx)
        kwargs_source = [
            {
                'amp': params.flux_source,
                'R_sersic': params.source_half_light,
                'n_sersic': params.source_sersic_n,
                'e1': 0.0,
                'e2': 0.0,
                'center_x': params.source_offset_x,
                'center_y': params.source_offset_y,
            },
        ]

        # --- Lens light: Sersic profile (foreground galaxy) ---
        lens_light_model = LightModel(light_model_list=['SERSIC_ELLIPSE'])
        kwargs_lens_light = [
            {
                'amp': params.flux_lens,
                'R_sersic': params.lens_half_light,
                'n_sersic': params.lens_sersic_n,
                'e1': params.ellipticity_e1,
                'e2': params.ellipticity_e2,
                'center_x': 0.0,
                'center_y': 0.0,
            },
        ]

        # Build image model
        image_model = ImageModel(
            data_class=data_class,
            psf_class=psf_class,
            lens_model_class=lens_model,
            source_model_class=source_light_model,
            lens_light_model_class=lens_light_model,
        )

        # Render noiseless image
        image = image_model.image(
            kwargs_lens=kwargs_lens,
            kwargs_source=kwargs_source,
            kwargs_lens_light=kwargs_lens_light,
        )

        return image.astype(np.float32)

    def _render_galaxy(self, params: LensParameters) -> np.ndarray:
        """
        Render a non-lensed galaxy image using Lenstronomy to match the lensed dataset's 
        simulation rendering engine and avoid simulation artifact bias.
        """
        kwargs_data = data_configure_simple(numPix=self.IMAGE_SIZE, deltaPix=self.PIXEL_SCALE, exposure_time=1.0, background_rms=1.0)
        data_class = ImageData(**kwargs_data)
        psf_kernel = make_psf_kernel(fwhm=0.1, pixel_scale=self.PIXEL_SCALE, size=21)
        psf_class = LenstronomyPSF(psf_type='PIXEL', kernel_point_source=psf_kernel)
        
        lens_model = LensModel(lens_model_list=[])
        source_light_model = LightModel(light_model_list=['SERSIC_ELLIPSE'])
        kwargs_source = [{
            'amp': params.flux_lens, 
            'R_sersic': params.lens_half_light, 
            'n_sersic': params.lens_sersic_n, 
            'e1': params.ellipticity_e1, 
            'e2': params.ellipticity_e2, 
            'center_x': 0.0, 
            'center_y': 0.0
        }]
        
        image_model = ImageModel(data_class=data_class, psf_class=psf_class, lens_model_class=lens_model, source_model_class=source_light_model)
        return image_model.image(kwargs_source=kwargs_source).astype(np.float32)

    def generate_lens(self) -> Tuple[np.ndarray, LensParameters]:
        """
        Generate one lensed image (positive class).

        Returns:
            image: (64, 64) float32 array, pixel values in ADU
            params: LensParameters with ground truth values
        """
        params = self._sample_lens_params()

        # Ray-traced lensed image via lenstronomy
        image = self._render_lensed_source(params)

        # Add realistic noise via GalSim
        galsim_img = galsim.Image(image, scale=self.PIXEL_SCALE)
        add_noise(
            galsim_img,
            rng=self.rng,
            sky_level=self.SKY_LEVEL,
            read_noise=self.READ_NOISE,
            gain=self.GAIN,
        )

        return galsim_img.array.astype(np.float32), params

    def generate_nonlens(self) -> Tuple[np.ndarray, None]:
        """
        Generate one non-lensed image (negative class).
        Just a galaxy + noise, no lensing.
        """
        params = self._sample_lens_params()

        # Non-lensed galaxy via GalSim
        image = self._render_galaxy(params)

        # Add noise
        galsim_img = galsim.Image(image, scale=self.PIXEL_SCALE)
        add_noise(
            galsim_img,
            rng=self.rng,
            sky_level=self.SKY_LEVEL,
            read_noise=self.READ_NOISE,
            gain=self.GAIN,
        )

        return galsim_img.array.astype(np.float32), None
