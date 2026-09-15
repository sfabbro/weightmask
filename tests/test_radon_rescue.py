"""Regression tests for the MRT Radon rescue.

Two defects are pinned here, both found by measurement on real MegaCam data:

1. ``radon`` and ``_candidate_from_rho_theta`` use different line conventions
   (radon's theta is the reflection of the Hesse angle the builder assumes), so
   the rescue handed the strip refiner the *mirror* of each candidate at every
   oblique angle. Near theta = 0 the two conventions agree, which is why the
   rescue kept proposing the chip's near-vertical column artefacts and never a
   real trail. The calibration below reconstructs injected lines at eight
   (angle, offset) combinations.
2. The per-angle peak significance was normalised by a ``mad_std`` taken over a
   column that is mostly zero padding, so its MAD was 0 and a hard-coded ``1.0``
   fallback turned every value into a million-sigma "detection". Statistics are
   now taken over the geometrically valid rho range only.
"""

import unittest

import numpy as np
import yaml

from weightmask.streaks import (
    _candidate_from_rho_theta,
    _detect_streaks_mrt_like,
    _radon_projections,
    _valid_rho_range,
)

# A small frame for the geometry/statistics tests: the calibration is
# scale-invariant and the transform cost is quadratic in the padded side.
SH = 240
SW = 360
SY, SX = np.mgrid[0:SH, 0:SW].astype(np.float32)

# A larger frame for the end-to-end detection tests, closer to a real cutout.
H, W = 600, 900
YY, XX = np.mgrid[0:H, 0:W].astype(np.float32)


def project_config():
    return yaml.safe_load(open("weightmask.yml"))["streak_masking"]


def rescue_config(**overrides):
    """The project's streak config with only ``mrt_rescue_params`` overridden.

    Built from the real config on purpose: a minimal dict silently falls back to
    much stricter internal defaults (profile_sigma_threshold 3.0 instead of 1.5,
    confidence_threshold 0.35 instead of 0.25) and tests a different pipeline.
    """
    config = dict(project_config())
    config["mrt_rescue_params"] = {**config["mrt_rescue_params"], **overrides}
    return config


def line_image(grid, height, width, phi_deg, offset, length, width_px=1.2, amplitude=1.0, half_width=3.0):
    """A line at image angle ``phi_deg`` with signed perpendicular offset ``offset``.

    Returns ``(flux, geometry, body)`` where ``body`` is the trail's footprint:
    cut in *both* along-track and across-track coordinates, since the along-track
    cut alone selects a band that spans the whole frame.
    """
    yy, xx = grid
    cy, cx = (height - 1) / 2.0, (width - 1) / 2.0
    phi = np.radians(phi_deg)
    nx, ny = -np.sin(phi), np.cos(phi)
    dx, dy = np.cos(phi), np.sin(phi)
    px, py = cx + offset * nx, cy + offset * ny
    along = (xx - px) * dx + (yy - py) * dy
    across = (xx - px) * nx + (yy - py) * ny
    on_trail = (np.abs(along) <= length / 2) & (np.abs(across) <= half_width)
    image = np.where(on_trail, amplitude * np.exp(-0.5 * (across / width_px) ** 2), 0.0)
    return image.astype(np.float32), (px, py, dx, dy), on_trail


def radon_peak(image, theta_step=1.0):
    """The strongest (theta, rho) of ``image`` over the geometrically valid rho."""
    thetas = np.arange(0.0, 180.0, theta_step, dtype=np.float32)
    sinogram = _radon_projections(image, thetas)
    side = sinogram.shape[0]
    rho = np.arange(side, dtype=np.float64) - 0.5 * (side - 1)
    low, high = _valid_rho_range(image.shape, thetas)
    valid = (rho[:, None] >= low[None, :]) & (rho[:, None] <= high[None, :])
    masked = np.where(valid, sinogram, -np.inf)
    row, col = np.unravel_index(int(np.argmax(masked)), masked.shape)
    return float(thetas[col]), float(rho[row])


def line_matches(candidate, geometry, angle_tol=3.0, distance_tol=3.0):
    """Does the candidate reproduce the injected line (angle *and* offset)?"""
    px, py, dx, dy = geometry
    (x0, y0), (x1, y1) = candidate["clipped_endpoints"]
    length = np.hypot(x1 - x0, y1 - y0)
    if length < 1:
        return False
    lx, ly = (x1 - x0) / length, (y1 - y0) / length
    angle_error = float(np.degrees(np.arccos(np.clip(abs(lx * dx + ly * dy), 0.0, 1.0))))
    distance = abs((px - x0) * (-ly) + (py - y0) * lx)
    return angle_error <= angle_tol and distance <= distance_tol


class TestRadonCandidateConversion(unittest.TestCase):
    """The (radon theta, rho) -> image line conversion, calibrated on injections."""

    CASES = (
        (90.0, 100.0),
        (90.0, -100.0),
        (0.0, 75.0),
        (0.0, -75.0),
        (45.0, 60.0),
        (22.9, 0.0),
        (22.9, 40.0),
        (140.0, -30.0),
    )

    def test_every_injected_line_is_reconstructed(self):
        for phi_deg, offset in self.CASES:
            with self.subTest(phi=phi_deg, offset=offset):
                image, geometry, _ = line_image((SY, SX), SH, SW, phi_deg, offset, length=150.0)
                theta, rho = radon_peak(image)
                candidate = _candidate_from_rho_theta(-rho, 180.0 - theta, image.shape)
                self.assertIsNotNone(candidate)
                self.assertTrue(
                    line_matches(candidate, geometry),
                    f"phi={phi_deg} offset={offset}: radon peak (theta={theta}, rho={rho}) "
                    f"did not reconstruct the injected line",
                )

    def test_the_unconverted_coordinates_describe_a_mirror(self):
        """Guard the bug itself: the raw radon pair must not reproduce an oblique line."""
        image, geometry, _ = line_image((SY, SX), SH, SW, 22.9, 0.0, length=150.0)
        theta, rho = radon_peak(image)
        mirrored = _candidate_from_rho_theta(rho, theta, image.shape)
        self.assertFalse(line_matches(mirrored, geometry))


class TestValidRhoRange(unittest.TestCase):
    def test_bounds_follow_the_projected_extent(self):
        for theta, half in ((0.0, SH / 2.0), (90.0, SW / 2.0)):
            low, high = _valid_rho_range((SH, SW), np.array([theta]))
            self.assertAlmostEqual(float(-low[0]), half, places=6)
            self.assertAlmostEqual(float(high[0]), half, places=6)

    def test_most_of_a_theta_zero_column_is_padding(self):
        """The measurement behind the significance fix: a theta=0 column is mostly zeros."""
        image, _, _ = line_image((SY, SX), SH, SW, 90.0, 0.0, length=150.0)
        sinogram = _radon_projections(image, np.array([0.0], dtype=np.float32))
        column = sinogram[:, 0]
        self.assertGreater(np.count_nonzero(column == 0) / column.size, 0.3)


class TestSignificanceIsNotCollapsed(unittest.TestCase):
    def test_quiet_frame_shows_no_multi_million_sigma_peak(self):
        """A pure-noise frame must not clear the gate by orders of magnitude."""
        rng = np.random.default_rng(1)
        image = rng.normal(0.0, 1.0, (SH, SW)).astype(np.float32)
        rms = np.full((SH, SW), 1.0, dtype=np.float32)
        _mask, accepted, _debug = _detect_streaks_mrt_like(image, rms, None, rescue_config(bin=1))
        for item in accepted:
            self.assertLess(item["peak_snr"], 1.0e4, "significance normalised against a collapsed sigma")


class TestRescueFindsItsOwnCase(unittest.TestCase):
    """The rescue must recover the kind of trail it exists for."""

    def _trail(self, sigma, amplitude=None, length=400.0):
        rng = np.random.default_rng(0)
        image = rng.normal(0.0, 1.0, (H, W)).astype(np.float32)
        trail, _geometry, body = line_image((YY, XX), H, W, 22.9, 0.0, length=length, amplitude=sigma)
        return image + trail, body, np.full((H, W), 1.0, dtype=np.float32)

    def test_a_faint_oblique_trail_is_found_and_confirmed(self):
        image, truth, rms = self._trail(6.0)
        mask, _accepted, debug = _detect_streaks_mrt_like(image, rms, None, rescue_config(bin=1))
        self.assertGreaterEqual(debug["accepted_count"], 1, "a 6-sigma oblique trail was not accepted")
        recall = np.count_nonzero(mask & truth) / max(1, np.count_nonzero(truth))
        self.assertGreater(recall, 0.5, f"confirmed a line but masked only {recall:.2f} of the trail")

    def test_binning_still_finds_the_trail(self):
        image, truth, rms = self._trail(12.0)
        mask, _accepted, debug = _detect_streaks_mrt_like(image, rms, None, rescue_config(bin=4))
        self.assertGreaterEqual(debug["accepted_count"], 1)
        recall = np.count_nonzero(mask & truth) / max(1, np.count_nonzero(truth))
        self.assertGreater(recall, 0.3)

    def test_it_does_not_mask_a_bright_column(self):
        """A full-height bright column must not be claimed as a trail."""
        image = np.random.default_rng(2).normal(0.0, 1.0, (H, W)).astype(np.float32)
        image[:, W - 100] += 40.0
        rms = np.full((H, W), 1.0, dtype=np.float32)
        mask, _accepted, _debug = _detect_streaks_mrt_like(image, rms, None, rescue_config(bin=1))
        self.assertEqual(int(np.count_nonzero(mask[:, W - 110 : W - 90])), 0)


if __name__ == "__main__":
    unittest.main()
