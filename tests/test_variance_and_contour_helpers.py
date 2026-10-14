"""Direct coverage for the variance, crowding-threshold and contour helpers.

These three surfaces were only reachable through a full image run, so a
regression reached the release as a benchmark swing: the unmeasured-RMS
sentinel (``+inf``) is a contract, not an accident, and it decides both the
inverse-variance weight and the detection level of the contour stage.
"""

import unittest
import warnings

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal

from weightmask.objects import _adaptive_extract_threshold
from weightmask.streaks import _contour_candidates
from weightmask.variance import _handle_empirical_fit, _handle_theoretical, _mask_unmeasured_rms

SENTINEL = np.inf


class TestMaskUnmeasuredRms(unittest.TestCase):
    def test_missing_inputs_stay_missing(self):
        self.assertIsNone(_mask_unmeasured_rms(None, None))
        rms = np.ones((4, 4), dtype=np.float32)
        self.assertIsNone(_mask_unmeasured_rms(None, rms))

    def test_no_rms_map_leaves_the_weights_alone(self):
        inv_var = np.ones((4, 4), dtype=np.float32)
        self.assertIs(_mask_unmeasured_rms(inv_var, None), inv_var)

    def test_an_all_measured_map_leaves_the_weights_alone(self):
        inv_var = np.ones((4, 4), dtype=np.float32)
        rms = np.full((4, 4), 3.0, dtype=np.float32)
        self.assertIs(_mask_unmeasured_rms(inv_var, rms), inv_var)

    def test_a_mismatched_rms_shape_leaves_the_weights_alone(self):
        inv_var = np.ones((4, 4), dtype=np.float32)
        rms = np.full((2, 2), 3.0, dtype=np.float32)
        self.assertIs(_mask_unmeasured_rms(inv_var, rms), inv_var)

    def test_unmeasured_pixels_lose_their_weight_and_the_input_is_not_mutated(self):
        inv_var = np.full((4, 4), 5.0, dtype=np.float32)
        original = inv_var.copy()
        rms = np.full((4, 4), 3.0, dtype=np.float32)
        rms[:2, :] = SENTINEL  # exactly the sentinel estimate_background writes
        masked = _mask_unmeasured_rms(inv_var, rms)
        self.assertIsNot(masked, inv_var)
        assert_array_equal(masked[:2, :], 0.0)
        assert_array_equal(masked[2:, :], 5.0)
        assert_array_equal(inv_var, original)

    def test_a_fully_unmeasured_map_zeroes_every_weight(self):
        inv_var = np.full((4, 4), 5.0, dtype=np.float32)
        masked = _mask_unmeasured_rms(inv_var, np.full((4, 4), SENTINEL, dtype=np.float32))
        assert_array_equal(masked, np.zeros((4, 4), dtype=np.float32))

    def test_a_zero_or_negative_rms_counts_as_unmeasured(self):
        inv_var = np.full((2, 2), 5.0, dtype=np.float32)
        rms = np.array([[0.0, -1.0], [np.nan, 2.0]], dtype=np.float32)
        masked = _mask_unmeasured_rms(inv_var, rms)
        assert_array_equal(masked, np.array([[0.0, 0.0], [0.0, 5.0]], dtype=np.float32))


class TestHandleTheoretical(unittest.TestCase):
    def _inputs(self):
        sky = np.full((8, 8), 100.0, dtype=np.float32)
        flat = np.ones((8, 8), dtype=np.float32)
        return sky, flat

    def test_missing_flat_or_sky_is_refused_with_a_warning(self):
        sky, flat = self._inputs()
        with self.assertWarns(RuntimeWarning):
            self.assertIsNone(_handle_theoretical({}, sky, None, 1.0, 5.0, 1e-9))
        with self.assertWarns(RuntimeWarning):
            self.assertIsNone(_handle_theoretical({}, None, flat, 1.0, 5.0, 1e-9))

    def test_a_measured_flat_and_sky_produce_finite_positive_weights(self):
        sky, flat = self._inputs()
        inv_var = _handle_theoretical({}, sky, flat, 1.0, 0.0, 1e-9)
        self.assertEqual(inv_var.shape, sky.shape)
        self.assertTrue(np.all(np.isfinite(inv_var)))
        self.assertTrue(np.all(inv_var > 0))

    def test_the_unmeasured_rms_sentinel_is_applied_here_too(self):
        sky, flat = self._inputs()
        rms = np.full((8, 8), 4.0, dtype=np.float32)
        rms[0, :] = SENTINEL
        inv_var = _handle_theoretical({}, sky, flat, 1.0, 0.0, 1e-9, rms)
        assert_array_equal(inv_var[0, :], 0.0)
        self.assertTrue(np.all(inv_var[1:, :] > 0))


class TestHandleEmpiricalFit(unittest.TestCase):
    def _inputs(self, shape=(64, 64)):
        sky = np.full(shape, 100.0, dtype=np.float32)
        flat = np.ones(shape, dtype=np.float32)
        sci = np.full(shape, 100.0, dtype=np.float32)
        mask = np.zeros(shape, dtype=bool)
        return sci, mask, sky, flat

    def test_missing_science_data_is_refused_with_a_warning(self):
        _sci, mask, sky, flat = self._inputs()
        with self.assertWarns(RuntimeWarning):
            self.assertIsNone(_handle_empirical_fit({}, sky, flat, None, mask, 1.0, 5.0, 1e-9))

    def test_a_missing_object_mask_is_refused_with_a_warning(self):
        sci, _mask, sky, flat = self._inputs()
        with self.assertWarns(RuntimeWarning):
            self.assertIsNone(_handle_empirical_fit({}, sky, flat, sci, None, 1.0, 5.0, 1e-9))

    def test_too_few_patches_falls_back_to_the_header_gain_and_read_noise(self):
        # patch_size 128 on a 64x64 frame leaves one patch, below the fit's
        # ten-patch floor, so the empirical route must decline and the supplied
        # gain/read noise must be used unchanged.
        sci, mask, sky, flat = self._inputs()
        cfg = {"empirical_patch_size": 128}
        fallback = _handle_empirical_fit(cfg, sky, flat, sci, mask, 1.0, 5.0, 1e-9)
        expected = _handle_theoretical(cfg, sky, flat, 1.0, 5.0, 1e-9)
        self.assertIsNotNone(fallback)
        assert_allclose(fallback, expected)

    def test_a_fitted_map_is_usable_even_when_the_fit_declines(self):
        rng = np.random.default_rng(11)
        shape = (128, 128)
        sci = (100.0 + rng.normal(0.0, 4.0, shape)).astype(np.float32)
        mask = np.zeros(shape, dtype=bool)
        sky = np.full(shape, 100.0, dtype=np.float32)
        flat = np.ones(shape, dtype=np.float32)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            inv_var = _handle_empirical_fit({"empirical_patch_size": 16}, sky, flat, sci, mask, 1.0, 5.0, 1e-9)
        self.assertIsNotNone(inv_var)
        self.assertEqual(inv_var.shape, shape)
        self.assertTrue(np.all(np.isfinite(inv_var)))
        self.assertTrue(np.all(inv_var >= 0))


class TestAdaptiveExtractThreshold(unittest.TestCase):
    def test_a_small_frame_is_left_alone(self):
        data = np.zeros((20, 20), dtype=np.float32)
        self.assertEqual(_adaptive_extract_threshold(data, None, None, 1.5), 1.5)

    def test_gaussian_noise_is_not_crowded_and_keeps_the_base_threshold(self):
        rng = np.random.default_rng(3)
        data = rng.normal(0.0, 1.0, (100, 100)).astype(np.float32)
        self.assertEqual(_adaptive_extract_threshold(data, None, None, 1.5), 1.5)

    def test_a_heavy_tail_raises_the_threshold_but_only_to_the_cap(self):
        rng = np.random.default_rng(4)
        data = rng.normal(0.0, 1.0, (100, 100)).astype(np.float32)
        data.ravel()[::20] = 1.0e6  # 5% of pixels at a physically absurd level
        raised = _adaptive_extract_threshold(data, None, None, 1.5)
        self.assertGreater(raised, 1.5)
        self.assertLessEqual(raised, 3.0)
        self.assertAlmostEqual(raised, 3.0)  # tail_ratio is saturating, so the cap binds

    def test_masking_the_crowding_restores_the_base_threshold(self):
        rng = np.random.default_rng(5)
        data = rng.normal(0.0, 1.0, (100, 100)).astype(np.float32)
        mask = np.zeros(data.shape, dtype=bool)
        mask.ravel()[::20] = True
        data.ravel()[::20] = 1.0e6
        self.assertEqual(_adaptive_extract_threshold(data, None, mask, 1.5), 1.5)

    def test_pixels_without_a_measured_rms_do_not_set_the_threshold(self):
        rng = np.random.default_rng(6)
        data = rng.normal(0.0, 1.0, (100, 100)).astype(np.float32)
        rms = np.full(data.shape, 2.0, dtype=np.float32)
        # The crowding lives entirely where the RMS was never measured.
        rms.ravel()[::20] = SENTINEL
        data.ravel()[::20] = 1.0e6
        self.assertEqual(_adaptive_extract_threshold(data, rms, None, 1.5), 1.5)

    def test_an_entirely_unmeasured_frame_keeps_the_base_threshold(self):
        data = np.full((100, 100), 1.0e6, dtype=np.float32)
        rms = np.full(data.shape, SENTINEL, dtype=np.float32)
        self.assertEqual(_adaptive_extract_threshold(data, rms, None, 1.5), 1.5)


class TestContourCandidates(unittest.TestCase):
    """Geometry of the ASTRiDE-style contour proposer."""

    def _frame(self):
        return np.zeros((400, 400), dtype=np.float32)

    def _horizontal_trail(self, amplitude=10.0):
        data = self._frame()
        data[200:203, 50:350] = amplitude
        return data

    def _vertical_trail(self, amplitude=10.0):
        data = self._frame()
        data[50:350, 200:203] = amplitude
        return data

    def test_a_blank_frame_proposes_nothing(self):
        self.assertEqual(_contour_candidates(self._frame(), None, None, {}), [])

    def test_a_long_thin_trail_is_proposed_once_with_a_boundary_to_boundary_span(self):
        candidates = _contour_candidates(self._horizontal_trail(), None, None, {})
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate["source"], "contours")
        # The PCA span is the blob; the reported span is the clipped line.
        self.assertAlmostEqual(candidate["raw_span"], 300.0, delta=5.0)
        self.assertGreater(candidate["span"], 350.0)
        self.assertEqual(candidate["edge_touches"], 2)
        # A horizontal trail has an axis angle of 0 degrees. Normalised into
        # [0, 180), so accept either end of the axis.
        angle = candidate["angle_deg"]
        self.assertLess(min(abs(angle), abs(180.0 - angle)), 1.0)

    def test_a_vertical_trail_is_proposed_on_the_other_axis(self):
        candidates = _contour_candidates(self._vertical_trail(), None, None, {})
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertAlmostEqual(candidate["angle_deg"], 90.0, delta=1.0)
        self.assertAlmostEqual(candidate["raw_span"], 300.0, delta=5.0)
        self.assertEqual(candidate["edge_touches"], 2)
        (x0, y0), (x1, y1) = candidate["clipped_endpoints"]
        self.assertEqual(candidate["endpoints"], candidate["clipped_endpoints"])
        self.assertEqual(candidate["segments"], [candidate["clipped_endpoints"]])
        # Clipped to the frame: a vertical trail spans top edge to bottom edge.
        self.assertLess(min(y0, y1), 1.0)
        self.assertGreater(max(y0, y1), 398.0)

    def test_a_compact_blob_is_rejected_by_the_shape_gate(self):
        data = self._frame()
        yy, xx = np.indices(data.shape)
        data[np.hypot(yy - 200, xx - 200) < 25.0] = 10.0
        self.assertEqual(_contour_candidates(data, None, None, {}), [])

    def test_a_bright_existing_mask_suppresses_the_trail(self):
        data = self._horizontal_trail()
        mask = data > 0
        self.assertEqual(_contour_candidates(data, mask, None, {}), [])

    def test_the_detection_level_is_set_by_the_measured_rms(self):
        data = self._horizontal_trail(amplitude=10.0)
        # A scale of 8 puts the 2.5-sigma level at 20, above the 10 ADU trail.
        rms = np.full(data.shape, 8.0, dtype=np.float32)
        self.assertEqual(_contour_candidates(data, None, rms, {}), [])
        rms_low = np.full(data.shape, 1.0, dtype=np.float32)
        self.assertEqual(len(_contour_candidates(data, None, rms_low, {})), 1)

    def test_an_unmeasured_rms_map_falls_back_to_unit_scale_rather_than_the_sentinel(self):
        # If the sentinel were taken at face value the level would be infinite
        # and the stage would never propose anything.
        data = self._horizontal_trail()
        rms = np.full(data.shape, SENTINEL, dtype=np.float32)
        self.assertEqual(len(_contour_candidates(data, None, rms, {})), 1)

    def test_the_span_gate_is_configurable(self):
        data = self._horizontal_trail()
        self.assertEqual(len(_contour_candidates(data, None, None, {"contour_params": {"min_span": 400.0}})), 0)

    def test_the_shape_gate_is_configurable(self):
        data = self._frame()
        yy, xx = np.indices(data.shape)
        data[np.hypot(yy - 200, xx - 200) < 25.0] = 10.0
        permissive = {"contour_params": {"shape_cut": 2.0, "radius_dev_cut": 0.0, "min_span": 1.0}}
        self.assertEqual(len(_contour_candidates(data, None, None, permissive)), 1)


if __name__ == "__main__":
    unittest.main()
