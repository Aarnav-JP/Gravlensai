"""
Reproducibility helpers for deterministic ML experiments.
"""

import random

import numpy as np
import torch


def set_global_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Seed Python, NumPy, and PyTorch RNGs.

    Args:
        seed: Random seed value.
        deterministic: If True, request deterministic PyTorch behavior where possible.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            # Some ops/platforms may not support strict determinism.
            pass
