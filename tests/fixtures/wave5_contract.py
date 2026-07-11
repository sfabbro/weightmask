"""Labeled, deterministic Wave 5 classical contract fixture."""

import numpy as np

from weightmask.contract import QualityBit

FIXTURE_LABEL = "wave5-classical-3x3-v1"


def classical_contract_fixture():
    """Return inverse variance and flags covering valid and invalid cases."""
    inverse_variance = np.array([[4.0, np.nan, -2.0], [0.0, 2.0, np.inf], [1.0, 8.0, 16.0]], dtype=np.float32)
    quality_mask = np.zeros((3, 3), dtype=np.uint32)
    quality_mask[1, 1] = QualityBit.SATURATED
    quality_mask[2, 0] = QualityBit.DETECTED
    quality_mask[2, 1] = QualityBit.COSMIC_RAY
    return inverse_variance, quality_mask
