"""Aperture flux on original pixels, with the weight plane used as a mask.

The flat-fielded Poisson formula must be used for a spatially varying flat;
omitting F from its sky term moves weighted flux beyond the read-noise floor.
"""

import unittest

import numpy as np

from weightmask.variance import _calculate_inverse_variance_theoretical


def _aperture(shape, cy, cx, radius):
    yy, xx = np.ogrid[: shape[0], : shape[1]]
    return (yy - cy) ** 2 + (xx - cx) ** 2 <= radius**2


def _flux(image, keep):
    return float(np.sum(image[keep]))


def _weighted_flux(image, weight, aperture):
    w = np.asarray(weight, dtype=np.float64)[aperture]
    total = float(np.sum(w))
    return float(np.sum(np.asarray(image, dtype=np.float64)[aperture] * w) / total * np.count_nonzero(aperture))


class TestPhotometryBias(unittest.TestCase):
    def test_cr_in_the_wing_moves_flux_until_it_is_masked(self):
        shape = (32, 32)
        yy, xx = np.ogrid[: shape[0], : shape[1]]
        star = 200.0 * np.exp(-0.5 * ((yy - 16) ** 2 + (xx - 16) ** 2) / 4.0)
        aperture = _aperture(shape, 16, 16, 8)
        clean = _flux(star, aperture)
        spiked = star.copy()
        spiked[16, 22] += 500.0
        self.assertGreater(_flux(spiked, aperture), clean + 100.0)
        keep = aperture.copy()
        keep[16, 22] = False
        self.assertAlmostEqual(_flux(spiked, keep), _flux(star, keep), places=5)

    def test_a_wider_trail_pad_moves_more_flux(self):
        shape = (32, 32)
        yy, xx = np.ogrid[: shape[0], : shape[1]]
        star = 200.0 * np.exp(-0.5 * ((yy - 16) ** 2 + (xx - 16) ** 2) / 4.0)
        aperture = _aperture(shape, 16, 16, 8)
        trail = np.zeros(shape, dtype=bool)
        trail[10:24, 18] = True
        tight = aperture & ~trail
        pad = trail.copy()
        pad[:, 16:21] = True
        wide = aperture & ~pad
        clean = _flux(star, aperture)
        self.assertGreater(clean - _flux(star, wide), clean - _flux(star, tight))

    def test_vignette_weight_ratio_and_flux_against_the_read_noise_floor(self):
        gain = 1.5
        read_noise_e = 5.0
        sky_adu = 1000.0
        flat_value = 0.7
        sky = np.full((32, 32), sky_adu, dtype=np.float32)
        uniform = np.full((32, 32), flat_value, dtype=np.float32)
        photon = _calculate_inverse_variance_theoretical(sky, uniform, gain, 0.0, 1e-9)
        np.testing.assert_allclose(photon, gain * flat_value / sky_adu, rtol=1e-6)

        varying = np.ones((32, 32), dtype=np.float32)
        varying[:, :16] = flat_value
        yy, xx = np.ogrid[:32, :32]
        star = 800.0 * np.exp(-0.5 * ((yy - 16) ** 2 + (xx - 10) ** 2) / 4.0)
        aperture = _aperture((32, 32), 16, 12, 8)
        w_photon = _calculate_inverse_variance_theoretical(sky, varying, gain, read_noise_e, 1e-9)
        expected = gain**2 * varying**2 / (sky * gain * varying + read_noise_e**2)
        np.testing.assert_allclose(w_photon, expected, rtol=1e-6)
        wrong_sky_term = gain**2 * varying**2 / (sky * gain + read_noise_e**2)
        delta = abs(_weighted_flux(star, wrong_sky_term, aperture) - _weighted_flux(star, w_photon, aperture))
        floor = (read_noise_e / gain) * np.sqrt(float(np.count_nonzero(aperture)))
        self.assertGreater(delta, floor)


if __name__ == "__main__":
    unittest.main()
