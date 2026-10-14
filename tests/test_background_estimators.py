"""Direct unit tests for the SEP background estimators in background.py.

These helpers decide the sky level every downstream weight depends on, but were
previously reachable only through ``estimate_background`` on a full image.
"""

import unittest
import warnings

import numpy as np

from weightmask.background import (
    _check_and_fix_edge_artifacts,
    _estimate_global_sep,
    _estimate_robust_median,
    _estimate_sep_tiered,
    _estimate_smooth_surface,
)


def _quiet(call, *args, **kwargs):
    """Run a helper that is documented to warn on a failure path."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return call(*args, **kwargs)


class TestEstimateGlobalSep(unittest.TestCase):
    def test_recovers_a_known_flat_offset(self):
        data = np.full((64, 64), 100.0, dtype=np.float32)
        mask = np.zeros(data.shape, dtype=bool)
        bkg_map, rms_map = _estimate_global_sep(data, mask)
        self.assertIsNotNone(bkg_map)
        self.assertEqual(bkg_map.shape, data.shape)
        self.assertEqual(rms_map.shape, data.shape)
        self.assertAlmostEqual(float(np.median(bkg_map)), 100.0, delta=1.0)
        self.assertTrue(np.all(np.isfinite(rms_map)))

    def test_a_zero_background_over_valid_data_is_refused(self):
        # With all pixels valid and a global background of exactly 0, the
        # helper reports failure rather than returning a zero sky as a fit.
        data = np.zeros((32, 32), dtype=np.float32)
        mask = np.zeros(data.shape, dtype=bool)
        bkg_map, rms_map = _quiet(_estimate_global_sep, data, mask)
        self.assertIsNone(bkg_map)
        self.assertIsNone(rms_map)

    def test_an_all_invalid_image_never_yields_a_finite_background(self):
        data = np.full((32, 32), np.nan, dtype=np.float32)
        mask = np.zeros(data.shape, dtype=bool)
        bkg_map, rms_map = _quiet(_estimate_global_sep, data, mask)
        # Either the helper refuses, or (if SEP still returns a level) that
        # level must not be a finite number fabricated out of no valid data.
        if bkg_map is not None:
            self.assertFalse(np.any(np.isfinite(bkg_map)))
        self.assertTrue(bkg_map is None or rms_map is not None)


class TestCheckAndFixEdgeArtifacts(unittest.TestCase):
    def test_a_small_image_is_returned_untouched(self):
        # h or w <= 2 * edge_width (100) means there is no interior to compare.
        bkg_map = np.full((50, 50), 10.0, dtype=np.float32)
        self.assertIs(_check_and_fix_edge_artifacts(bkg_map, (50, 50)), bkg_map)

    def test_a_clean_image_is_not_smoothed(self):
        bkg_map = np.full((200, 200), 10.0, dtype=np.float32)
        self.assertIs(_check_and_fix_edge_artifacts(bkg_map, (200, 200)), bkg_map)

    def test_a_bright_edge_is_smoothed(self):
        bkg_map = np.full((200, 200), 10.0, dtype=np.float32)
        bkg_map[:, -50:] = 500.0  # a full-height artefact in the right edge band
        fixed = _check_and_fix_edge_artifacts(bkg_map, (200, 200))
        self.assertIsNot(fixed, bkg_map, "the artefact should trigger smoothing")
        self.assertEqual(fixed.shape, bkg_map.shape)
        # The hard step at the band boundary is blurred across it.
        self.assertLess(float(fixed[100, 150]), 500.0)
        self.assertGreater(float(fixed[100, 150]), 10.0)
        self.assertTrue(np.all(np.isfinite(fixed)))

    def test_a_threshold_above_the_edge_difference_leaves_the_map_alone(self):
        bkg_map = np.full((200, 200), 10.0, dtype=np.float32)
        bkg_map[:, -50:] = 500.0
        fixed = _check_and_fix_edge_artifacts(bkg_map, (200, 200), config={"edge_artifact_thresh": 1.0e9})
        self.assertIs(fixed, bkg_map)


class TestEstimateSepTiered(unittest.TestCase):
    @staticmethod
    def _gradient(shape=(256, 256), slope_x=0.5, slope_y=0.2):
        yy, xx = np.indices(shape, dtype=np.float32)
        return (500.0 + slope_x * xx + slope_y * yy).astype(np.float32)

    def test_it_tracks_a_smooth_gradient_better_than_a_constant(self):
        data = self._gradient()
        mask = np.zeros(data.shape, dtype=bool)
        bkg_map, rms_map, box_used = _estimate_sep_tiered(data, mask, 64, 3, 512)
        self.assertIsNotNone(bkg_map)
        self.assertEqual(box_used, 64, "the first box should succeed")
        self.assertEqual(bkg_map.shape, data.shape)
        self.assertEqual(rms_map.shape, data.shape)
        self.assertTrue(np.all(np.isfinite(rms_map)))
        residual = np.nanmedian(np.abs(bkg_map - data))
        spread = np.nanmedian(np.abs(data - np.nanmedian(data)))
        self.assertLess(residual, spread / 5.0)

    def test_a_non_contiguous_input_is_accepted(self):
        data = np.asfortranarray(self._gradient((256, 256)))
        self.assertFalse(data.flags["C_CONTIGUOUS"])
        bkg_map, _rms_map, box_used = _estimate_sep_tiered(data, None, 64, 3, 512)
        self.assertIsNotNone(bkg_map)
        self.assertEqual(box_used, 64)

    def test_a_none_mask_is_accepted(self):
        data = self._gradient((128, 128))
        bkg_map, rms_map, box_used = _estimate_sep_tiered(data, None, 64, 3, 512)
        self.assertIsNotNone(bkg_map)
        self.assertIsNotNone(rms_map)
        self.assertEqual(box_used, 64)

    def test_a_box_above_the_maximum_is_refused_without_trying(self):
        data = self._gradient((128, 128))
        mask = np.zeros(data.shape, dtype=bool)
        # The retry loop is `while current_box <= max_box_size`; a first box
        # already above the ceiling never executes.
        self.assertEqual(_estimate_sep_tiered(data, mask, 256, 3, 128), (None, None, None))


class TestEstimateRobustMedian(unittest.TestCase):
    def test_the_global_median_fallback_recovers_a_flat_level(self):
        rng = np.random.default_rng(0)
        data = (100.0 + rng.normal(0.0, 2.0, (128, 128))).astype(np.float32)
        mask = np.zeros(data.shape, dtype=bool)
        bkg_map, rms_map = _estimate_robust_median(data, mask, "robust_median_fallback", {})
        self.assertIsNotNone(bkg_map)
        self.assertEqual(bkg_map.shape, data.shape)
        self.assertAlmostEqual(float(np.median(bkg_map)), 100.0, delta=1.0)
        # A single flat level everywhere.
        self.assertAlmostEqual(float(np.ptp(bkg_map)), 0.0, places=5)

    def test_the_global_median_fallback_resists_bright_outliers(self):
        rng = np.random.default_rng(1)
        data = (100.0 + rng.normal(0.0, 1.0, (200, 200))).astype(np.float32)
        data[50:60, 50:60] = 5000.0
        mask = np.zeros(data.shape, dtype=bool)
        bkg_map, _rms_map = _estimate_robust_median(data, mask, "robust_median_fallback", {})
        level = float(np.median(bkg_map))
        self.assertAlmostEqual(level, 100.0, delta=1.0)
        self.assertGreater(abs(level - float(np.mean(data))), 5.0, "the mean is dragged by the bright block")

    def test_it_reports_a_positive_rms_for_noisy_data(self):
        rng = np.random.default_rng(2)
        data = (50.0 + rng.normal(0.0, 3.0, (128, 128))).astype(np.float32)
        mask = np.zeros(data.shape, dtype=bool)
        _bkg_map, rms_map = _estimate_robust_median(data, mask, "robust_median_fallback", {})
        self.assertTrue(np.all(np.isfinite(rms_map)))
        self.assertGreater(float(np.median(rms_map)), 0.0)
        self.assertLess(float(np.median(rms_map)), 12.0)

    def test_the_median_filter_path_uses_the_configured_kernel(self):
        rng = np.random.default_rng(3)
        data = (200.0 + rng.normal(0.0, 1.0, (128, 128))).astype(np.float32)
        data[60:68, 60:68] += 3000.0  # a compact bright blob
        mask = np.zeros(data.shape, dtype=bool)
        bkg_map, rms_map = _estimate_robust_median(data, mask, "median_filter", {"median_kernel_size": 15})
        self.assertIsNotNone(bkg_map)
        self.assertEqual(bkg_map.shape, data.shape)
        # The median filter does not follow the blob.
        self.assertLess(float(np.median(np.abs(bkg_map - 200.0))), 5.0)
        self.assertTrue(np.all(np.isfinite(rms_map)))

    def test_a_fully_masked_median_filter_returns_no_background(self):
        data = np.full((64, 64), 10.0, dtype=np.float32)
        mask = np.ones(data.shape, dtype=bool)
        self.assertEqual(_estimate_robust_median(data, mask, "median_filter", {"median_kernel_size": 15}), (None, None))

    def test_non_finite_pixels_do_not_enter_the_median_filter(self):
        rng = np.random.default_rng(4)
        data = (75.0 + rng.normal(0.0, 1.0, (128, 128))).astype(np.float32)
        data[0:10, 0:10] = np.nan
        data[10:20, 0:10] = np.inf
        mask = np.zeros(data.shape, dtype=bool)
        bkg_map, _rms_map = _estimate_robust_median(data, mask, "median_filter", {"median_kernel_size": 15})
        self.assertIsNotNone(bkg_map)
        self.assertTrue(np.all(np.isfinite(bkg_map)))
        self.assertAlmostEqual(float(np.median(bkg_map)), 75.0, delta=2.0)


class TestEstimateSmoothSurface(unittest.TestCase):
    def test_too_few_valid_pixels_returns_no_background(self):
        small = np.full((9, 9), 10.0, dtype=np.float32)
        mask = np.zeros(small.shape, dtype=bool)
        self.assertEqual(_estimate_smooth_surface(small, mask, {}), (None, None))

    def test_an_all_invalid_image_returns_no_background(self):
        data = np.full((64, 64), 10.0, dtype=np.float32)
        mask = np.ones(data.shape, dtype=bool)
        self.assertEqual(_estimate_smooth_surface(data, mask, {}), (None, None))

    def test_it_fits_a_linear_surface(self):
        rng = np.random.default_rng(5)
        yy, xx = np.indices((64, 64), dtype=np.float64)
        plane = 1000.0 + 3.0 * xx - 2.0 * yy
        data = (plane + rng.normal(0.0, 1.5, plane.shape)).astype(np.float32)
        mask = np.zeros(data.shape, dtype=bool)
        bkg_map, rms_map = _estimate_smooth_surface(data, mask, {})
        self.assertIsNotNone(bkg_map)
        self.assertEqual(bkg_map.shape, data.shape)
        # The quadratic design contains the plane exactly.
        self.assertLess(float(np.median(np.abs(bkg_map - plane))), 0.5)
        self.assertTrue(np.all(np.isfinite(rms_map)))
        self.assertGreater(float(np.median(rms_map)), 0.0)
        self.assertLess(float(np.median(rms_map)), 3.0)

    def test_masked_and_non_finite_pixels_are_excluded_from_the_fit(self):
        rng = np.random.default_rng(6)
        yy, xx = np.indices((64, 64), dtype=np.float64)
        plane = 500.0 + 1.0 * xx + 0.5 * yy
        data = (plane + rng.normal(0.0, 1.0, plane.shape)).astype(np.float32)
        mask = np.zeros(data.shape, dtype=bool)
        mask[:32, :] = True  # excluded by the mask
        # A bright blob and a non-finite patch are both excluded; the fit is a
        # plain least squares, so leaving either in would dominate the surface.
        data[40:50, 40:50] = 1.0e6
        mask[40:50, 40:50] = True
        data[50:58, 0:8] = np.nan
        bkg_map, _rms_map = _estimate_smooth_surface(data, mask, {})
        self.assertIsNotNone(bkg_map)
        self.assertTrue(np.all(np.isfinite(bkg_map)))
        self.assertLess(float(np.median(np.abs(bkg_map - plane))), 1.5)

    def test_the_fit_ignores_non_finite_pixels(self):
        rng = np.random.default_rng(7)
        yy, xx = np.indices((64, 64), dtype=np.float64)
        plane = 300.0 + 2.0 * xx
        data = (plane + rng.normal(0.0, 1.0, plane.shape)).astype(np.float32)
        data[:10, :10] = np.nan
        mask = np.zeros(data.shape, dtype=bool)
        bkg_map, _rms_map = _estimate_smooth_surface(data, mask, {})
        self.assertIsNotNone(bkg_map)
        self.assertTrue(np.all(np.isfinite(bkg_map)))
        self.assertLess(float(np.median(np.abs(bkg_map - plane))), 1.5)


if __name__ == "__main__":
    unittest.main()
