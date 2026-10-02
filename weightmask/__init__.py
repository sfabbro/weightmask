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

try:
    from ._version import __version__
except ImportError:  # pragma: no cover - source tree without the module
    try:
        from importlib.metadata import version as _distribution_version

        __version__ = _distribution_version("weightmask")
    except Exception:
        __version__ = "0.0.0.dev0"

# Legacy public dictionary kept stable; the contract adds INVALID_VARIANCE.
from .contract import QUALITY_BITS

MASK_BITS = QUALITY_BITS

MASK_DTYPE = "uint32"  # Data type for the bitmask

__all__ = ["MASK_BITS", "MASK_DTYPE", "QUALITY_BITS", "__version__"]
