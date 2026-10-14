"""PSF protection subtracts the preliminary sky, not a global median."""

import unittest

import numpy as np
import yaml

from weightmask.cosmics import _apply_psf_protection, _fwhm_from_header


class TestPsfSky(unittest.TestCase):
    def test_header_seeing_sets_fwhm_in_pixels(self):
        fwhm = _fwhm_from_header({"SEEING": 0.74, "PIXSCAL1": 0.185}, 3.0)
        self.assertAlmostEqual(fwhm, 0.74 / 0.185, places=5)

    def test_star_on_a_tilted_sky_is_protected_only_with_the_sky_map(self):
        shape = (40, 40)
        sky = np.full(shape, 500.0, dtype=np.float32)
        sky[:8, :8] = 20.0
        yy, xx = np.ogrid[: shape[0], : shape[1]]
        star = 80.0 * np.exp(-0.5 * ((yy - 3) ** 2 + (xx - 3) ** 2) / (1.2**2))
        sci = sky + star.astype(np.float32)
        crmask = np.zeros(shape, dtype=bool)
        crmask[2:5, 2:5] = True
        rms = np.ones(shape, dtype=np.float32)
        cfg = {"psf_aware": True, "psf_fwhm_guess": 3.0}
        with_sky = _apply_psf_protection(crmask.copy(), sci, cfg, 1.5, 5.0, rms, sky_map=sky)
        without = _apply_psf_protection(crmask.copy(), sci, cfg, 1.5, 5.0, rms, sky_map=None)
        self.assertFalse(bool(with_sky[3, 3]))
        self.assertTrue(bool(without[3, 3]))

    def test_single_pass_stays_off(self):
        cosmic = yaml.safe_load(open("weightmask.yml"))["cosmic_ray"]
        self.assertFalse(bool(cosmic.get("single_pass", False)))


if __name__ == "__main__":
    unittest.main()
