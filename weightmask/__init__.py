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
__version__ = "1.0.0"

# Define mask bit definitions for easy access
MASK_BITS = {
    "BAD": 1 << 0,  # 1    - Bad pixels from flat field
    "SAT": 1 << 1,  # 2    - Saturated pixels
    "CR": 1 << 2,  # 4    - Cosmic ray hits
    "DETECTED": 1 << 3,  # 8    - Detected astronomical objects
    "STREAK": 1 << 4,  # 16   - Satellite/artifact streaks
}

MASK_DTYPE = "uint32"  # Data type for the bitmask
