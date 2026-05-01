"""Tests for simulation pipeline."""

import pytest
import numpy as np


class TestLensImageGenerator:
    """Tests for LensImageGenerator."""

    @pytest.fixture(scope='class')
    def generator(self):
        from gravlensai.simulate.lens_generator import LensImageGenerator
        return LensImageGenerator(seed=42)

    def test_generate_lens_shape(self, generator):
        """Generated lens image should be (64, 64) float32."""
        img, params = generator.generate_lens()
        assert img.shape == (64, 64), f"Expected (64, 64), got {img.shape}"
        assert img.dtype == np.float32

    def test_generate_nonlens_shape(self, generator):
        """Generated non-lens image should be (64, 64) float32."""
        img, params = generator.generate_nonlens()
        assert img.shape == (64, 64), f"Expected (64, 64), got {img.shape}"
        assert img.dtype == np.float32
        assert params is None

    def test_lens_has_params(self, generator):
        """Generated lens should return LensParameters."""
        _, params = generator.generate_lens()
        assert params is not None
        assert hasattr(params, 'einstein_radius')
        assert hasattr(params, 'ellipticity_e1')
        assert hasattr(params, 'shear_g1')

    def test_lens_params_ranges(self, generator):
        """Sampled parameters should fall within documented ranges."""
        _, params = generator.generate_lens()
        assert 0.5 <= params.einstein_radius <= 2.5
        assert -0.3 <= params.ellipticity_e1 <= 0.3
        assert -0.3 <= params.ellipticity_e2 <= 0.3
        assert -0.05 <= params.shear_g1 <= 0.05
        assert -0.05 <= params.shear_g2 <= 0.05
        assert 3.0 <= params.lens_sersic_n <= 5.0
        assert 0.3 <= params.lens_half_light <= 1.5
        assert 0.5 <= params.source_sersic_n <= 4.0

    def test_regression_targets(self, generator):
        """regression_targets() should return length-6 array."""
        _, params = generator.generate_lens()
        targets = params.regression_targets()
        assert targets.shape == (6,)
        assert targets.dtype == np.float32

    def test_reproducibility(self):
        """Same seed should produce identical images."""
        from gravlensai.simulate.lens_generator import LensImageGenerator
        gen1 = LensImageGenerator(seed=123)
        gen2 = LensImageGenerator(seed=123)
        img1, _ = gen1.generate_lens()
        img2, _ = gen2.generate_lens()
        np.testing.assert_array_equal(img1, img2)

    def test_no_nan_inf(self, generator):
        """Generated images should not contain NaN or Inf values."""
        img, _ = generator.generate_lens()
        assert not np.isnan(img).any(), "Lens image contains NaN"
        assert not np.isinf(img).any(), "Lens image contains Inf"

        img2, _ = generator.generate_nonlens()
        assert not np.isnan(img2).any(), "Non-lens image contains NaN"
        assert not np.isinf(img2).any(), "Non-lens image contains Inf"


class TestPSFModels:
    """Tests for PSF kernel generation."""

    def test_kolmogorov_kernel_shape(self):
        from gravlensai.simulate.psf_models import make_psf_kernel
        kernel = make_psf_kernel(fwhm=0.1, pixel_scale=0.05, size=21)
        assert kernel.shape == (21, 21)

    def test_kernel_normalised(self):
        from gravlensai.simulate.psf_models import make_psf_kernel
        kernel = make_psf_kernel(fwhm=0.1, pixel_scale=0.05, size=21)
        np.testing.assert_almost_equal(kernel.sum(), 1.0, decimal=5)

    def test_gaussian_kernel(self):
        from gravlensai.simulate.psf_models import make_psf_kernel
        kernel = make_psf_kernel(psf_type='gaussian')
        assert kernel.shape[0] == kernel.shape[1]
        np.testing.assert_almost_equal(kernel.sum(), 1.0, decimal=5)


class TestNoiseModels:
    """Tests for noise application."""

    def test_noise_increases_variance(self):
        import galsim
        from gravlensai.simulate.noise_models import add_noise
        rng = galsim.BaseDeviate(42)
        img = galsim.Image(64, 64, scale=0.05, init_value=100.0)
        std_before = np.std(img.array)
        add_noise(img, rng)
        std_after = np.std(img.array)
        assert std_after > std_before, "Noise should increase pixel variance"

    def test_estimate_noise_level(self):
        from gravlensai.simulate.noise_models import estimate_noise_level
        sigma = estimate_noise_level()
        assert sigma > 0, "Noise level should be positive"
        assert sigma < 10, "Noise level should be reasonable"
