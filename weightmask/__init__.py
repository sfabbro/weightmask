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

# Define version
__version__ = '1.0.0'

# Import mask bit definitions for easy access
from weightmask import MASK_BITS, MASK_DTYPE
