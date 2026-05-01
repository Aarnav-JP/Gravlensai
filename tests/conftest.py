"""
Pytest configuration file for GravLensAI tests.
Configures test environment, fixtures, and warning filters.
"""

import warnings


def pytest_configure(config):
    """Configure pytest and suppress non-blocking external dependency warnings."""
    # Suppress NumPy 2.0 deprecation warnings from lenstronomy's convolution module
    # (lenstronomy hasn't updated np.fft.rfftn calls yet)
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module="lenstronomy.*",
        message=".*axes.*",
    )
    # Catch-all filter for any lenstronomy NumPy 2.0 related warnings
    warnings.filterwarnings(
        "ignore",
        category=DeprecationWarning,
        module="lenstronomy.ImSim.Numerics.convolution",
    )
