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
    from importlib.metadata import version as _distribution_version

    __version__ = _distribution_version("weightmask")
except Exception:
    __version__ = "0.1.0"

# Legacy public dictionary kept stable; the contract adds INVALID_VARIANCE.
from .contract import QUALITY_BITS

MASK_BITS = QUALITY_BITS

MASK_DTYPE = "uint32"  # Data type for the bitmask

__all__ = ["MASK_BITS", "MASK_DTYPE", "QUALITY_BITS", "WeightMapGenerator", "__version__"]


def __getattr__(name: str):
    if name == "WeightMapGenerator":
        from .pipeline import WeightMapGenerator

        return WeightMapGenerator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
