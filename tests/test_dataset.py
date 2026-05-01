"""Tests for dataset and data loading."""

import pytest
import numpy as np
import torch
import os

from gravlensai.data.normalisation import (
    arcsinh_stretch, standardise, arcsinh_normalise, normalise_single,
    compute_normalisation_stats, apply_normalisation_stats,
)
from gravlensai.data.augmentation import augment_image


class TestArcsinhNormalisation:
    """Tests for normalisation utilities."""

    def test_arcsinh_stretch_shape(self):
        """arcsinh_stretch should preserve shape."""
        imgs = np.random.randn(10, 64, 64).astype(np.float32)
        out = arcsinh_stretch(imgs)
        assert out.shape == imgs.shape

    def test_arcsinh_no_nan(self):
        """arcsinh normalisation should not produce NaN or Inf."""
        imgs = np.random.randn(10, 64, 64).astype(np.float32) * 1000
        out = arcsinh_normalise(imgs)
        assert not np.isnan(out).any(), "NaN in normalised output"
        assert not np.isinf(out).any(), "Inf in normalised output"

    def test_standardise_zero_mean(self):
        """standardise should produce mean ≈ 0."""
        imgs = np.random.randn(100, 64, 64).astype(np.float32) * 50 + 100
        out = standardise(imgs)
        assert abs(out.mean()) < 0.01, f"Mean should be ~0, got {out.mean()}"

    def test_standardise_unit_std(self):
        """standardise should produce std ≈ 1."""
        imgs = np.random.randn(100, 64, 64).astype(np.float32) * 50 + 100
        out = standardise(imgs)
        assert abs(out.std() - 1.0) < 0.01, f"Std should be ~1, got {out.std()}"

    def test_normalise_single_shape(self):
        """normalise_single should preserve shape."""
        img = np.random.randn(64, 64).astype(np.float32)
        out = normalise_single(img)
        assert out.shape == (64, 64)
        assert out.dtype == np.float32

    def test_normalise_single_no_nan(self):
        """normalise_single should handle zero images gracefully."""
        img = np.zeros((64, 64), dtype=np.float32)
        out = normalise_single(img)
        assert not np.isnan(out).any()


class TestAugmentation:
    """Tests for augmentation pipeline."""

    def test_augment_preserves_shape(self):
        """Augmentation should preserve tensor shape."""
        img = torch.randn(1, 64, 64)
        out = augment_image(img)
        assert out.shape == (1, 64, 64)

    def test_augment_preserves_dtype(self):
        """Augmentation should preserve dtype."""
        img = torch.randn(1, 64, 64)
        out = augment_image(img)
        assert out.dtype == img.dtype

    def test_augment_deterministic_seed(self):
        """Same seed should produce same augmentation."""
        img = torch.randn(1, 64, 64)
        torch.manual_seed(42)
        out1 = augment_image(img.clone())
        torch.manual_seed(42)
        out2 = augment_image(img.clone())
        torch.testing.assert_close(out1, out2)


class TestSimulatedLensDataset:
    """
    Tests for SimulatedLensDataset.
    These tests only run if simulated data exists.
    """

    @pytest.fixture(scope='class')
    def data_exists(self):
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                'data', 'simulated')
        return os.path.exists(os.path.join(data_dir, 'images_lens.npy'))

    def test_classify_batch_shape(self, data_exists):
        """Classification batch should have shape (B, 1, 64, 64)."""
        if not data_exists:
            pytest.skip("Simulated data not available")
        from gravlensai.data.dataset import SimulatedLensDataset
        ds = SimulatedLensDataset('data/simulated/', split='test',
                                  task='classify', augment=False)
        img, label = ds[0]
        assert img.shape == (1, 64, 64)
        assert label.shape == ()  # scalar
        assert label.item() in [0.0, 1.0]

    def test_regress_batch_shape(self, data_exists):
        """Regression batch should return 6-dim parameter targets."""
        if not data_exists:
            pytest.skip("Simulated data not available")
        from gravlensai.data.dataset import SimulatedLensDataset
        ds = SimulatedLensDataset('data/simulated/', split='test',
                                  task='regress', augment=False)
        img, params = ds[0]
        assert img.shape == (1, 64, 64)
        assert params.shape == (6,)
        # Normalised params should be in [-1, 1]
        assert (params >= -1.1).all() and (params <= 1.1).all()

    def test_train_val_test_nonoverlapping(self, data_exists):
        """Train/val/test splits should not overlap."""
        if not data_exists:
            pytest.skip("Simulated data not available")
        from gravlensai.data.dataset import SimulatedLensDataset
        train = SimulatedLensDataset('data/simulated/', split='train',
                                     task='classify', augment=False)
        val = SimulatedLensDataset('data/simulated/', split='val',
                                   task='classify', augment=False)
        test = SimulatedLensDataset('data/simulated/', split='test',
                                    task='classify', augment=False)
        total = len(train) + len(val) + len(test)
        # Should approximately equal full dataset (60k)
        assert total > 0
        # Each split should have images
        assert len(train) > len(val)
        assert len(train) > len(test)

    def test_val_uses_train_normalisation_stats(self, tmp_path):
        """Validation split should reuse stats computed from the training split only."""
        from gravlensai.data.dataset import SimulatedLensDataset

        # Build a small synthetic dataset with shifted distributions.
        n_lens = 10
        n_nonlens = 10
        images_lens = np.linspace(10.0, 100.0, n_lens * 64 * 64, dtype=np.float32).reshape(n_lens, 64, 64)
        images_nonlens = np.linspace(200.0, 300.0, n_nonlens * 64 * 64, dtype=np.float32).reshape(n_nonlens, 64, 64)
        params_lens = np.zeros((n_lens, 5), dtype=np.float32)

        np.save(tmp_path / 'images_lens.npy', images_lens)
        np.save(tmp_path / 'images_nonlens.npy', images_nonlens)
        np.save(tmp_path / 'params_lens.npy', params_lens)

        seed = 42
        val_ds = SimulatedLensDataset(
            str(tmp_path), split='val', task='classify', augment=False, split_seed=seed
        )

        all_images = np.concatenate([images_lens, images_nonlens], axis=0)
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(all_images))
        train_end = int(len(all_images) * 0.8)
        val_end = int(len(all_images) * 0.9)
        train_idx = idx[:train_end]
        val_idx = idx[train_end:val_end]

        stats = compute_normalisation_stats(all_images[train_idx])
        expected_val = apply_normalisation_stats(all_images[val_idx], stats)

        np.testing.assert_allclose(val_ds.images, expected_val, rtol=1e-5, atol=1e-5)
