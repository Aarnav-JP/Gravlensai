"""
Image augmentation pipeline for astronomical images.

Only geometric transforms (flip, rotate) — no photometric changes
to preserve physical flux information. This is standard practice
in astronomical image analysis.

Augmentations:
  - Random horizontal flip
  - Random vertical flip
  - Random 90° rotations (0°, 90°, 180°, 270°)

These preserve the rotational symmetry of gravitational lensing.
"""

import torch


def augment_image(img: torch.Tensor) -> torch.Tensor:
    """
    Apply random geometric augmentations to a single image tensor.

    Args:
        img: (C, H, W) tensor.

    Returns:
        Augmented (C, H, W) tensor.
    """
    # Random horizontal flip
    if torch.rand(1).item() > 0.5:
        img = torch.flip(img, dims=[2])

    # Random vertical flip
    if torch.rand(1).item() > 0.5:
        img = torch.flip(img, dims=[1])

    # Random 90° rotation (0, 1, 2, or 3 quarter-turns)
    k = int(torch.randint(0, 4, (1,), dtype=torch.int64).item())
    if k > 0:
        img = torch.rot90(img, k=k, dims=[1, 2])

    return img
