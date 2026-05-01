"""Data sub-package: PyTorch datasets, augmentation, normalisation."""

from gravlensai.data.normalisation import (
    arcsinh_stretch,
    standardise,
    arcsinh_normalise,
    normalise_single,
    compute_normalisation_stats,
    apply_normalisation_stats,
)
from gravlensai.data.augmentation import augment_image
from gravlensai.data.dataset import SimulatedLensDataset, HSTDataset

__all__ = [
    "arcsinh_stretch",
    "standardise",
    "arcsinh_normalise",
    "normalise_single",
    "compute_normalisation_stats",
    "apply_normalisation_stats",
    "augment_image",
    "SimulatedLensDataset",
    "HSTDataset",
]
