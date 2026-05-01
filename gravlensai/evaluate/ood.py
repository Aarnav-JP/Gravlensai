"""
Out-of-Distribution (OOD) Detection.

Identifies whether an input image is likely an artifact (cosmic ray, sensor noise,
or unrecognised telescope anomaly) rather than a clean galaxy image, using
epistemic uncertainty (MC Dropout entropy) and morphological heuristics.
"""

from __future__ import annotations

import numpy as np


class OODDetector:
    """
    Detects anomalous out-of-distribution artifacts.
    """

    def __init__(self, entropy_threshold: float = 0.8, snr_threshold: float = 1.0):
        """
        Args:
            entropy_threshold: Max predictive entropy (in bits). Above this, the
                               model is confused (OOD). Max entropy for binary
                               classification is 1.0.
            snr_threshold: Minimum Signal-to-Noise ratio.
        """
        self.entropy_threshold = entropy_threshold
        self.snr_threshold = snr_threshold

    def _estimate_snr(self, img: np.ndarray) -> float:
        """Estimate pseudo-SNR of an image."""
        p50 = np.percentile(img, 50)
        signal = float(np.percentile(img, 99) - p50)
        bg = img[img <= p50]
        noise = float(np.std(bg)) if bg.size > 0 else float(np.std(img))
        return signal / (noise + 1e-8)

    def detect(self, image: np.ndarray, predictive_entropy: float) -> tuple[bool, str]:
        """
        Evaluate if an image is OOD.

        Args:
            image: (64, 64) 2D numpy array.
            predictive_entropy: Epistemic entropy derived from MC Dropout variance.

        Returns:
            (is_ood, reason_string)
        """
        snr = self._estimate_snr(image)

        if predictive_entropy > self.entropy_threshold:
            return True, f"High epistemic uncertainty (Entropy: {predictive_entropy:.2f} > {self.entropy_threshold}). Model has never seen features like this."

        if snr < self.snr_threshold:
            return True, f"Extremely low Signal-to-Noise Ratio (SNR: {snr:.2f} < {self.snr_threshold}). Image is likely pure noise."

        return False, "Image is in-distribution."
