"""Photometric behavior through the per-image production pipeline."""

import contextlib
import io
import unittest
from pathlib import Path

import numpy as np
import yaml

from weightmask.contract import QualityBit
from weightmask.process import process_image

ROOT = Path(__file__).resolve().parents[1]
SHAPE = (64, 64)
HEADER = {
    "GAIN": 1.5,
    "RDNOISE": 5.0,
    "DATASEC": "[1:64,1:64]",
    "CCDSIZE": "[1:64,1:64]",
}


def _aperture(shape, cy, cx, radius):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2


def _config():
    config = yaml.safe_load((ROOT / "weightmask.yml").read_text())
    config["streak_masking"]["enable"] = False
    config["sep_background"].update({"method": "median_filter", "iterations": 1})
    config["sep_objects"]["extract_thresh"] = 100.0
    config["cosmic_ray"]["faint_cr"]["enable"] = False
    config["saturation"]["mask_bleed_trails"] = False
    config["variance"]["rescale_variance"] = False
    return config


def _run(data, flat):
    with contextlib.redirect_stdout(io.StringIO()):
        return process_image(data, HEADER, flat, _config(), tile_size=32)


def _weighted_aperture_flux(image, weight, aperture):
    values = np.asarray(image, dtype=np.float64)[aperture]
    weights = np.asarray(weight, dtype=np.float64)[aperture]
    valid = weights > 0.0
    return float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]) * np.count_nonzero(aperture))


class TestPhotometryBias(unittest.TestCase):
    def test_process_image_masks_cosmic_ray_before_aperture_flux(self):
        rng = np.random.default_rng(0)
        yy, xx = np.mgrid[:64, :64]
        clean = (1000.0 + rng.normal(0.0, 5.0, SHAPE)).astype(np.float32)
        clean += 200.0 * np.exp(-0.5 * ((yy - 32) ** 2 + (xx - 32) ** 2) / 4.0)
        spiked = clean.copy()
        spiked[32, 40] += 500.0
        aperture = _aperture(SHAPE, 32, 32, 8)

        clean_mask, _clean_ivar, clean_weight, *_ = _run(clean, np.ones(SHAPE, dtype=np.float32))
        spiked_mask, _spiked_ivar, spiked_weight, *_ = _run(spiked, np.ones(SHAPE, dtype=np.float32))

        bit = int(QualityBit.COSMIC_RAY)
        self.assertNotEqual(int(spiked_mask[32, 40] & bit), 0)
        self.assertEqual(float(spiked_weight[32, 40]), 0.0)
        clean_flux = _weighted_aperture_flux(clean, clean_weight, aperture)
        spiked_flux = _weighted_aperture_flux(spiked, spiked_weight, aperture)
        self.assertLess(abs(spiked_flux - clean_flux), 0.001 * abs(clean_flux))

    def test_process_image_uses_flat_fielded_poisson_variance(self):
        rng = np.random.default_rng(1)
        data = (1000.0 + rng.normal(0.0, 5.0, SHAPE)).astype(np.float32)
        flat = np.ones(SHAPE, dtype=np.float32)
        flat[:, :32] = 0.7

        _mask, inverse_variance, _weight, _confidence, sky, _header = _run(data, flat)
        expected = 1.5**2 * flat**2 / (sky * 1.5 * flat + 5.0**2)
        np.testing.assert_allclose(inverse_variance, expected, rtol=0.03, atol=1.0e-6)
        self.assertGreater(float(np.mean(inverse_variance[:, 32:])), float(np.mean(inverse_variance[:, :32])))


if __name__ == "__main__":
    unittest.main()
