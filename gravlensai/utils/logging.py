"""
Structured logging setup for GravLensAI.
Provides consistent log formatting across all modules.
"""

import logging


def setup_logger(name: str = "gravlensai", level: int = logging.INFO) -> logging.Logger:
    """Create a structured logger with consistent formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
