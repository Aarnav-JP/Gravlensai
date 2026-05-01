"""
PyTorch Dataset classes for both simulated and real HST data.
Handles normalisation, augmentation, and train/val/test splits.

Classes:
  - SimulatedLensDataset: for GalSim/lenstronomy-generated lens/non-lens images
  - HSTDataset: for real HST Frontier Fields cutouts (used in domain adaptation)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Optional, Tuple, Dict

from gravlensai.data.normalisation import (
    arcsinh_normalise,
    normalise_single,
    compute_normalisation_stats,
)
from gravlensai.data.augmentation import augment_image


class SimulatedLensDataset(Dataset):
    """
    Dataset for simulated lens/non-lens images.

    Args:
        data_dir: Path to directory containing images_lens.npy, params_lens.npy,
                  images_nonlens.npy
        split: 'train', 'val', or 'test' (80/10/10 default split)
        task: 'classify' (returns binary label) or 'regress' (returns 5 params,
               lensed images only)
        augment: Whether to apply data augmentation
    """

    SPLIT_RATIOS = {'train': 0.8, 'val': 0.1, 'test': 0.1}

    def __init__(
        self,
        data_dir: str,
        split: str = 'train',
        task: str = 'classify',
        augment: bool = True,
        split_seed: int = 42,
        regression_targets: Optional[int] = 6,
    ):
        super().__init__()
        data_path = Path(data_dir)

        images_lens = np.load(data_path / "images_lens.npy")     # (N, 64, 64)
        params_lens = np.load(data_path / "params_lens.npy")     # (N, 5)
        images_nonlens = np.load(data_path / "images_nonlens.npy")

        self.norm_stats: Dict[str, float]

        if task == 'classify':
            # Combine lens + non-lens
            all_images = np.concatenate([images_lens, images_nonlens], axis=0)
            all_labels = np.concatenate([
                np.ones(len(images_lens)),
                np.zeros(len(images_nonlens))
            ])
            idx = self._split_indices(len(all_images), split, split_seed)
            train_idx = self._split_indices(len(all_images), 'train', split_seed)
            self.images = all_images[idx]
            self.targets = all_labels[idx].astype(np.float32)
            self.norm_stats = compute_normalisation_stats(all_images[train_idx])
            self.task = 'classify'

        elif task == 'regress':
            # Lensed images only with parameter labels
            idx = self._split_indices(len(images_lens), split, split_seed)
            train_idx = self._split_indices(len(images_lens), 'train', split_seed)
            self.images = images_lens[idx]
            self.targets = params_lens[idx].astype(np.float32)
            self.norm_stats = compute_normalisation_stats(images_lens[train_idx])
            target_dim = self.targets.shape[1]
            if regression_targets is not None:
                if regression_targets > target_dim:
                    raise ValueError(
                        f"Requested {regression_targets} regression targets, but dataset only has {target_dim}."
                    )
                self.targets = self.targets[:, :regression_targets]
                target_dim = regression_targets

            # Normalise parameter targets to [-1, 1] for stable training
            self.param_mins = np.array([0.5, -0.3, -0.3, -0.05, -0.05, 8.0], dtype=np.float32)[:target_dim]
            self.param_maxs = np.array([2.5,  0.3,  0.3,  0.05,  0.05, 10.0], dtype=np.float32)[:target_dim]
            self.targets = (2 * (self.targets - self.param_mins) / \
                           (self.param_maxs - self.param_mins) - 1).astype(np.float32)
            self.task = 'regress'

        else:
            raise ValueError(f"Unknown task: {task}. Use 'classify' or 'regress'.")

        # Normalise with train-split statistics for consistent val/test scaling.
        self.images = arcsinh_normalise(self.images, stats=self.norm_stats)

        # Augmentations (only training, only flip/rotate — no photometric changes)
        self.augment = augment and (split == 'train')

    def _split_indices(self, n: int, split: str, split_seed: int = 42) -> np.ndarray:
        """Reproducible train/val/test split (80/10/10)."""
        rng = np.random.default_rng(split_seed)
        idx = rng.permutation(n)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)
        if split == 'train':
            return idx[:train_end]
        elif split == 'val':
            return idx[train_end:val_end]
        else:
            return idx[val_end:]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        img = torch.from_numpy(self.images[idx]).unsqueeze(0)  # (1, 64, 64)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)

        if self.augment:
            img = augment_image(img)

        return img, target


class HSTDataset(Dataset):
    """
    Dataset for real HST Frontier Fields images.
    Used for domain adaptation and final evaluation.

    Images are loaded from preprocessed .npy cutout files.
    Each cutout is normalised independently (no global stats for real data).
    """

    def __init__(self, fits_dir: str, catalog_path: Optional[str] = None):
        """
        Args:
            fits_dir: Directory of preprocessed .npy cutout files
            catalog_path: Optional CSV with known lens labels (for evaluation)
        """
        fits_path = Path(fits_dir)
        self.image_paths = sorted(fits_path.glob("*.npy"))

        self.labels = None
        if catalog_path:
            import pandas as pd
            df = pd.read_csv(catalog_path)
            self.labels = dict(zip(df['filename'], df['is_lens']))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = np.load(self.image_paths[idx]).astype(np.float32)
        # Normalise each cutout independently
        img = normalise_single(img)
        img = torch.from_numpy(img).unsqueeze(0)  # (1, 64, 64)

        label = -1  # Unknown by default
        if self.labels:
            fname = self.image_paths[idx].name
            label = float(self.labels.get(fname, -1))

        return img, label
