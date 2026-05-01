"""
Per-band normalisation utilities for astronomical images.
Implements arcsinh stretch (standard for wide dynamic range astronomical images)
followed by per-dataset standardisation.

The arcsinh function compresses bright pixels while preserving faint features
like gravitational arcs — critical for lens detection.
"""

import numpy as np
from typing import Optional


def arcsinh_stretch(images: np.ndarray, softening: Optional[float] = None) -> np.ndarray:
    """
    Apply arcsinh stretch to images.

    The arcsinh function is approximately linear for small values and
    logarithmic for large values — ideal for astronomical dynamic range.

    Args:
        images: (N, H, W) or (H, W) array of pixel values.
        softening: Scale parameter. If None, uses median(|images|).

    Returns:
        Stretched images with same shape.
    """
    if softening is None:
        softening = np.median(np.abs(images)) + 1e-5
    return np.arcsinh(images / softening)


def compute_normalisation_stats(images: np.ndarray) -> dict:
    """
    Compute arcsinh+standardisation statistics from a reference set.

    Args:
        images: (N, H, W) array used as the reference distribution
            (typically the training split only).

    Returns:
        Dict with softening, mean, std to be reused on other splits.
    """
    softening = float(np.median(np.abs(images)) + 1e-5)
    stretched = np.arcsinh(images / softening)
    mean = float(stretched.mean())
    std = float(stretched.std() + 1e-8)
    return {'softening': softening, 'mean': mean, 'std': std}


def standardise(images: np.ndarray) -> np.ndarray:
    """
    Zero-mean, unit-variance standardisation.

    Args:
        images: Array of any shape.

    Returns:
        Standardised array with mean≈0, std≈1.
    """
    mean = images.mean()
    std = images.std() + 1e-8
    return ((images - mean) / std).astype(np.float32)


def apply_normalisation_stats(images: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    """
    Apply precomputed arcsinh+standardisation stats to images.

    Args:
        images: (N, H, W) float array of raw pixel values.
        stats: Dict containing softening, mean, std.

    Returns:
        (N, H, W) float32 array, normalised.
    """
    stretched = np.arcsinh(images / stats['softening'])
    normalised = ((stretched - stats['mean']) / stats['std']).astype(np.float32)

    assert not np.isnan(normalised).any(), "NaN in normalised images!"
    assert not np.isinf(normalised).any(), "Inf in normalised images!"

    return normalised


def arcsinh_normalise(images: np.ndarray, stats: Optional[dict[str, float]] = None) -> np.ndarray:
    """
    Full normalisation pipeline: arcsinh stretch → standardise.
    Standard preprocessing for astronomical CNN inputs.

    Args:
        images: (N, H, W) float array of raw pixel values.

    Returns:
        (N, H, W) float32 array, normalised.
    """
    if stats is None:
        stretched = arcsinh_stretch(images)
        normalised = standardise(stretched)
    else:
        normalised = apply_normalisation_stats(images, stats)

    # Safety check
    assert not np.isnan(normalised).any(), "NaN in normalised images!"
    assert not np.isinf(normalised).any(), "Inf in normalised images!"

    return normalised


def normalise_single(image: np.ndarray) -> np.ndarray:
    """
    Normalise a single image independently (for real HST data where
    there are no global dataset statistics).

    Args:
        image: (H, W) float array.

    Returns:
        (H, W) float32 array, normalised.
    """
    softening = np.median(np.abs(image)) + 1e-5
    img = np.arcsinh(image / softening)
    img = (img - img.mean()) / (img.std() + 1e-8)
    return img.astype(np.float32)
