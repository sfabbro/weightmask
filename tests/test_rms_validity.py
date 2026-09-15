"""The background-RMS sentinel contract, and each consumer's declared intent.

``estimate_background`` marks pixels whose local RMS it could not measure with
``+inf``. That sentinel covers a large area of a real chip (7% of pixels, and
150 of 2112 whole columns on a MegaPrime HDU), so every consumer has to state
which reading it takes:

* detection thresholds treat unknown-RMS pixels as inert, because significance
  cannot be asserted without an RMS;
* rejection/quality gates substitute a robust value, because they still have to
  judge what they can.

These tests pin both halves, and in particular the data-dependent flip that
``np.nanmedian`` used to introduce: ``nanmedian`` ignores NaN but *not* ``inf``,
so once more than half a map carried the sentinel every "substitute the typical
RMS" call site silently became "never detect".
"""

import re
import unittest
from pathlib import Path

import numpy as np

from weightmask.streaks import (
    _detect_streaks_contours,
    _detect_trails_sparse_ransac,
    _refine_trail_mask,
)
from weightmask.utils import rms_or_robust, rms_valid_mask, robust_rms

PACKAGE_ROOT = Path(__file__).resolve().parents[1] / "weightmask"


def _trail_candidate(endpoints, segments=3):
    return {
        "clipped_endpoints": endpoints,
        "endpoints": endpoints,
        "segments": [endpoints] * segments,
        "span": float(np.hypot(endpoints[1][0] - endpoints[0][0], endpoints[1][1] - endpoints[0][1])),
        "raw_span": float(np.hypot(endpoints[1][0] - endpoints[0][0], endpoints[1][1] - endpoints[0][1])),
        "edge_touches": 2,
        "corridor_overlap": 0.0,
    }


def _trail_image(shape=(320, 320), noise=0.0, seed=0):
    """A bright horizontal trail across the middle of an otherwise flat image."""
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, noise, shape).astype(np.float32) if noise else np.zeros(shape, dtype=np.float32)
    center = shape[0] // 2
    data[center - 1 : center + 2, 10:-10] = 200.0
    return data


class TestRmsHelpers(unittest.TestCase):
    def test_valid_mask_excludes_every_unknown_form(self):
        rms = np.array([1.0, 5.0, np.inf, -np.inf, np.nan, 0.0, -2.0])
        np.testing.assert_array_equal(
            rms_valid_mask(rms),
            np.array([True, True, False, False, False, False, False]),
        )

    def test_valid_mask_passes_none_through(self):
        self.assertIsNone(rms_valid_mask(None))
        self.assertIsNone(rms_or_robust(None))

    def test_robust_rms_uses_only_measured_pixels(self):
        rms = np.array([5.0, 5.0, 5.0, 5.0, np.inf])
        self.assertAlmostEqual(robust_rms(rms), 5.0)

    def test_robust_rms_survives_a_mostly_sentinel_map(self):
        """The regression: >50% sentinel must not collapse the estimate to inf.

        ``np.nanmedian`` returns ``inf`` here, which is what made four call
        sites silently switch from "substitute the typical RMS" to "never
        detect" purely as a function of how much of the chip was unmeasured.
        """
        rms = np.concatenate([np.full(70, np.inf), np.full(30, 12.0)])
        self.assertTrue(np.isinf(np.nanmedian(rms)))
        self.assertAlmostEqual(robust_rms(rms), 12.0)

    def test_robust_rms_defaults_when_nothing_is_measured(self):
        self.assertAlmostEqual(robust_rms(np.full(10, np.inf), default=7.5), 7.5)
        self.assertAlmostEqual(robust_rms(np.zeros(10), default=2.5), 2.5)

    def test_rms_or_robust_fills_only_the_sentinel(self):
        rms = np.array([4.0, np.inf, 6.0, np.inf])
        np.testing.assert_allclose(rms_or_robust(rms), np.array([4.0, 5.0, 6.0, 5.0]))
        np.testing.assert_allclose(rms_or_robust(rms, fallback=9.0), np.array([4.0, 9.0, 6.0, 9.0]))

    def test_no_consumer_takes_a_median_over_a_sentinel_bearing_map(self):
        """Guard: ``np.nanmedian`` must never be handed an RMS map again."""
        offenders = []
        pattern = re.compile(r"nanmedian\([^)]*\brms")
        for path in sorted(PACKAGE_ROOT.glob("*.py")):
            for lineno, line in enumerate(path.read_text().splitlines(), start=1):
                if pattern.search(line):
                    offenders.append(f"{path.name}:{lineno}: {line.strip()}")
        self.assertEqual(offenders, [], "np.nanmedian ignores NaN but not the inf sentinel: " + "; ".join(offenders))


class TestDetectionThresholdsAreInertOnUnknownRms(unittest.TestCase):
    """Unknown-RMS pixels must not be able to manufacture a detection."""

    def setUp(self):
        self.data = _trail_image()
        self.candidate = _trail_candidate(((10.0, 160.0), (310.0, 160.0)))
        self.mask_cfg = {
            "strip_length": 256,
            "strip_width": 96,
            "profile_sigma_threshold": 1.5,
            "min_mask_pixels": 24,
            "min_row_hits": 4,
            "min_row_hit_fraction": 0.2,
            "min_col_hit_fraction": 0.1,
            "max_support_width": 12,
        }

    def test_trail_is_confirmed_when_the_rms_is_measured(self):
        rms = np.full(self.data.shape, 5.0, dtype=np.float32)
        mask, info = _refine_trail_mask(self.data, rms, self.candidate, self.mask_cfg)
        self.assertGreater(int(np.count_nonzero(mask)), 24)
        self.assertNotIn("reject_reason", info)

    def test_sentinel_rms_confirms_nothing(self):
        rms = np.full(self.data.shape, np.inf, dtype=np.float32)
        mask, info = _refine_trail_mask(self.data, rms, self.candidate, self.mask_cfg)
        self.assertEqual(int(np.count_nonzero(mask)), 0)
        self.assertEqual(info["reject_reason"], "bg_starved")

    def test_a_sentinel_region_cannot_add_support_elsewhere(self):
        """Only the trail's own band is sentinel: the rest must still not detect it."""
        rms = np.full(self.data.shape, 5.0, dtype=np.float32)
        rms[158:163, :] = np.inf
        mask, _info = _refine_trail_mask(self.data, rms, self.candidate, self.mask_cfg)
        self.assertEqual(int(np.count_nonzero(mask)), 0)

    def test_sparse_ransac_skips_unknown_rms_pixels(self):
        config = {
            "enable_sparse_ransac": True,
            "sparse_ransac_params": {
                "detect_thresh_sig": 5.0,
                "residual_threshold": 3.0,
                "min_inliers": 25,
                "min_length": 50.0,
                "min_line_density": 0.2,
                "max_trials": 50,
                "max_trails": 2,
            },
        }
        measured = _detect_trails_sparse_ransac(
            self.data, np.full(self.data.shape, 5.0, dtype=np.float32), None, config
        )
        self.assertGreater(int(np.count_nonzero(measured)), 0)

        sentinel = _detect_trails_sparse_ransac(
            self.data, np.full(self.data.shape, np.inf, dtype=np.float32), None, config
        )
        self.assertEqual(int(np.count_nonzero(sentinel)), 0)

    def test_contour_scale_ignores_sentinel_pixels(self):
        """The contour level comes from the measured pixels, sentinel excluded.

        Halving the map between a uniform value and the sentinel must therefore
        leave the stage's output untouched: the sentinel is excluded from the
        scale estimate rather than counted into it (which is what
        ``np.nanmedian`` would fail to do, since it does not ignore ``inf``).
        """
        config = {"contour_params": {"thresh_sig": 2.5}, "mask_params": {}}
        data = _trail_image() + 60.0
        measured = _detect_streaks_contours(data, np.full(data.shape, 5.0, dtype=np.float32), None, config)
        half_sentinel = np.full(data.shape, np.inf, dtype=np.float32)
        half_sentinel[:, :160] = 5.0
        split = _detect_streaks_contours(data, half_sentinel, None, config)

        self.assertIsInstance(measured, tuple)
        self.assertEqual(len(measured), 3)
        self.assertTrue(np.array_equal(measured[0], split[0]))
        self.assertEqual(measured[1], split[1])

        all_sentinel = _detect_streaks_contours(data, np.full(data.shape, np.inf, dtype=np.float32), None, config)
        self.assertEqual(int(np.count_nonzero(all_sentinel[0])), 0)
        self.assertEqual(all_sentinel[1], [])


class TestRejectionGatesSubstitute(unittest.TestCase):
    """Gates that must still judge a component borrow a robust value instead."""

    def test_post_filter_keeps_a_bright_component_in_an_unmeasured_region(self):
        from weightmask.cosmics import _post_filter_components

        sci = np.zeros((60, 60), dtype=np.float32)
        sci[30, 30:33] = 500.0
        flag = np.zeros(sci.shape, dtype=bool)
        flag[30, 30:33] = True
        rms = np.full(sci.shape, np.inf, dtype=np.float32)
        kept = _post_filter_components(flag, sci, rms, {"max_component_area": 12, "min_component_contrast_sigma": 4.0})
        self.assertTrue(np.array_equal(kept, flag))

    def test_psf_protection_is_not_claimed_without_an_rms(self):
        from weightmask.cosmics import _apply_psf_protection

        yy, xx = np.mgrid[0:80, 0:80]
        blob = 200.0 * np.exp(-(((yy - 40.0) ** 2 + (xx - 40.0) ** 2) / (2 * 2.0**2)))
        sci = (100.0 + blob).astype(np.float32)
        flag = np.zeros(sci.shape, dtype=bool)
        flag[38:43, 38:43] = True

        rms = np.full(sci.shape, 5.0, dtype=np.float32)
        protected = _apply_psf_protection(flag.copy(), sci, {"psf_aware": True, "psf_fwhm_guess": 3.0}, 1.5, 5.0, rms)
        self.assertLess(int(np.count_nonzero(protected)), int(np.count_nonzero(flag)))

        rms[:] = np.inf
        unprotected = _apply_psf_protection(flag.copy(), sci, {"psf_aware": True, "psf_fwhm_guess": 3.0}, 1.5, 5.0, rms)
        self.assertTrue(np.array_equal(unprotected, flag))

    def test_dynamic_sigclip_uses_measured_pixels_only(self):
        """The >50% flip: a mostly-sentinel chip must still tune from what was seen."""
        from weightmask.cosmics import _adjust_dynamic_sigclip

        rms = np.full((40, 40), np.inf, dtype=np.float32)
        rms[:10, :] = 10.0
        sigclip = _adjust_dynamic_sigclip({"dynamic_sigclip": True}, rms, default_sigclip=8.5)
        self.assertLess(sigclip, 8.5)
        self.assertAlmostEqual(sigclip, 4.5 * (10.0 / 11.0), places=3)

        all_sentinel = _adjust_dynamic_sigclip(
            {"dynamic_sigclip": True}, np.full((40, 40), np.inf, dtype=np.float32), default_sigclip=8.5
        )
        self.assertAlmostEqual(all_sentinel, 8.5)

    def test_sep_is_given_no_objects_in_an_unmeasured_region(self):
        """``err=inf`` is honoured by SEP, so ``detect_objects`` needs no substitution."""
        from weightmask.objects import detect_objects

        rng = np.random.default_rng(3)
        data = rng.normal(0.0, 1.0, (200, 200)).astype(np.float32)
        for cy, cx in ((50, 50), (150, 150)):
            data[cy - 2 : cy + 3, cx - 2 : cx + 3] += 400.0
        rms = np.full(data.shape, 1.0, dtype=np.float32)
        rms[:, :100] = np.inf

        mask = detect_objects(data, rms, None, {"extract_thresh": 3.0, "min_area": 5})
        self.assertTrue(mask[145:156, 145:156].any(), "measured half should still detect its source")
        self.assertFalse(mask[:, :100].any(), "no object may be claimed where the RMS is unknown")


class TestSentinelWeightAndGrowth(unittest.TestCase):
    def test_variance_sentinel_yields_zero_weight(self):
        from weightmask.variance import _calculate_inverse_variance_rms

        rms = np.array([[5.0, np.inf, 0.0]], dtype=np.float32)
        inv_var = _calculate_inverse_variance_rms(rms, epsilon=1e-6)
        self.assertAlmostEqual(float(inv_var[0, 0]), 1.0 / 25.0, places=6)
        self.assertEqual(float(inv_var[0, 1]), 0.0)
        self.assertEqual(float(inv_var[0, 2]), 0.0)


if __name__ == "__main__":
    unittest.main()
