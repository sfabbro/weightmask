"""
WeightMask: A modular tool for generating mask, weight, and confidence maps for astronomical images.

This package includes modules for detecting various effects in astronomical images:
- Saturation
- Bad pixels
- Cosmic rays
- Streaks
- Object detection
- Background estimation
- Variance calculation
"""

from ._version import __version__
from .contract import QUALITY_BITS

MASK_DTYPE = "uint32"  # Data type for the bitmask

__all__ = ["MASK_DTYPE", "QUALITY_BITS", "__version__"]
