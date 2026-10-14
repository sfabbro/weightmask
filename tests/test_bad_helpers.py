"""Direct unit tests for the flat-masking helpers in ``weightmask/bad.py``.

``detect_bad_pixels`` decides which flat pixels become ``BAD``, but its two
stages (local median deviation and horizontal-derivative columns) were only
reachable through it. Pinning each stage on its own makes a threshold or
partitioning change fail on the line that caused it.
"""

import json
import unittest

import numpy as np

from weightmask.bad import (
    _detect_bad_columns_derivative,
    _detect_bad_pixels_local,
    _flat_bad_settings,
    _get_global_median,
    _parse_section,
)


class TestGetGlobalMedian(unittest.TestCase):
    def test_a_uniform_flat_returns_its_own_value(self):
        self.assertAlmostEqual(_get_global_median(np.full((4, 4), 5.0)), 5.0)

    def test_only_positive_pixels_are_used(self):
        flat = np.array([[1.0, 2.0], [-5.0, 4.0]])
        self.assertAlmostEqual(_get_global_median(flat), 2.0)

    def test_non_finite_pixels_do_not_enter_the_median(self):
        flat = np.array([[1.0, 2.0], [3.0, np.nan]])
        self.assertAlmostEqual(_get_global_median(flat), 2.0)

    def test_an_all_invalid_flat_falls_back_to_one(self):
        for flat in (
            np.zeros((3, 3)),
            np.full((2, 2), -1.0),
            np.full((2, 2), np.nan),
            np.full((2, 2), np.inf),
        ):
            with self.subTest(flat=flat.tolist()):
                self.assertEqual(_get_global_median(flat), 1.0)

    def test_the_chunked_sample_stays_close_to_the_exact_median(self):
        # 200k positive pixels make the sampler step by 2, so its answer may
        # differ from the exact median by at most the step.
        flat = np.arange(1.0, 200001.0)
        median = _get_global_median(flat)
        self.assertAlmostEqual(median, float(np.median(flat)), delta=1.0)

    def test_a_multi_dimensional_flat_is_flattened(self):
        flat = np.arange(6.0).reshape(2, 3)  # includes a zero, which is excluded
        self.assertAlmostEqual(_get_global_median(flat), np.median([1.0, 2.0, 3.0, 4.0, 5.0]))


class TestDetectBadPixelsLocal(unittest.TestCase):
    @staticmethod
    def _uniform(size=20, value=1.0):
        return np.full((size, size), value, dtype=np.float64)

    def test_a_clean_uniform_flat_has_no_bad_pixels(self):
        flat = self._uniform()
        mask = _detect_bad_pixels_local(flat, {}, 1.0)
        self.assertEqual(mask.shape, flat.shape)
        self.assertFalse(mask.any())

    def test_a_pixel_below_the_default_low_ratio_is_flagged(self):
        flat = self._uniform()
        flat[10, 10] = 0.1
        mask = _detect_bad_pixels_local(flat, {}, 1.0)
        self.assertTrue(mask[10, 10])
        self.assertEqual(int(mask.sum()), 1)

    def test_a_pixel_above_the_default_high_ratio_is_flagged(self):
        flat = self._uniform()
        flat[4, 15] = 5.0
        mask = _detect_bad_pixels_local(flat, {}, 1.0)
        self.assertTrue(mask[4, 15])
        self.assertEqual(int(mask.sum()), 1)

    def test_a_pixel_inside_the_band_is_left_alone(self):
        flat = self._uniform()
        flat[10, 10] = 0.85  # above the 0.5 floor, below the 2.0 ceiling
        self.assertFalse(_detect_bad_pixels_local(flat, {}, 1.0).any())

    def test_the_low_ratio_threshold_is_taken_from_the_config(self):
        flat = self._uniform()
        flat[10, 10] = 0.85
        mask = _detect_bad_pixels_local(flat, {"local_low_thresh": 0.9}, 1.0)
        self.assertTrue(mask[10, 10])
        self.assertEqual(int(mask.sum()), 1)

    def test_the_high_ratio_threshold_is_taken_from_the_config(self):
        flat = self._uniform()
        flat[10, 10] = 1.5
        self.assertFalse(_detect_bad_pixels_local(flat, {}, 1.0).any())
        self.assertTrue(_detect_bad_pixels_local(flat, {"local_high_thresh": 1.2}, 1.0)[10, 10])

    def test_zero_and_negative_flat_pixels_are_bad(self):
        flat = self._uniform()
        flat[3, 3] = 0.0
        flat[6, 6] = -1.0
        mask = _detect_bad_pixels_local(flat, {}, 1.0)
        self.assertTrue(mask[3, 3])
        self.assertTrue(mask[6, 6])
        self.assertEqual(int(mask.sum()), 2)

    def test_an_all_zero_flat_is_entirely_bad(self):
        mask = _detect_bad_pixels_local(np.zeros((8, 8)), {}, 1.0)
        self.assertTrue(mask.all())

    def test_non_finite_pixels_are_flagged_without_poisoning_neighbours(self):
        flat = self._uniform()
        flat[10, 10] = np.nan
        flat[2, 2] = np.inf
        mask = _detect_bad_pixels_local(flat, {}, 1.0)
        self.assertTrue(mask[10, 10])
        self.assertTrue(mask[2, 2])
        self.assertEqual(int(mask.sum()), 2)
        # The non-finite samples were replaced by the global median inside the
        # smoothed model, so no neighbour inherits their ratio: exactly the
        # centre of the 3x3 window around the NaN is flagged.
        self.assertEqual(int(mask[9:12, 9:12].sum()), 1)

    def test_the_caller_flat_is_not_modified(self):
        flat = self._uniform()
        flat[10, 10] = np.nan
        _detect_bad_pixels_local(flat, {}, 1.0)
        # The non-finite sample is replaced in a copy, never in the caller's array.
        self.assertTrue(np.isnan(flat[10, 10]))
        self.assertEqual(int(np.count_nonzero(~np.isfinite(flat))), 1)
        self.assertEqual(int(np.count_nonzero(flat == 1.0)), flat.size - 1)


class TestDetectBadColumnsDerivative(unittest.TestCase):
    @staticmethod
    def _uniform(rows=30, cols=40, value=1.0):
        return np.full((rows, cols), value, dtype=np.float64)

    def test_detection_can_be_disabled_by_config(self):
        bad = _detect_bad_columns_derivative(self._uniform(), {"col_enable": False}, 1.0)
        self.assertEqual(len(bad), 0)

    def test_a_clean_uniform_flat_has_no_bad_columns(self):
        bad = _detect_bad_columns_derivative(self._uniform(), {}, 1.0)
        self.assertEqual(list(bad), [])

    def test_a_smooth_gradient_is_not_a_bad_column(self):
        # Exactly integer-valued, so neighbouring differences are all equal and
        # the derivative statistic sees no jump at any column.
        flat = np.tile(np.arange(10.0, 50.0), (30, 1))
        bad = _detect_bad_columns_derivative(flat, {}, 29.5)
        self.assertEqual(list(bad), [])

    def test_an_isolated_dead_column_is_flagged_and_nothing_else(self):
        flat = self._uniform()
        flat[:, 20] = 0.05
        bad = _detect_bad_columns_derivative(flat, {}, 1.0)
        self.assertEqual(list(bad), [20])

    def test_a_mostly_non_finite_column_is_flagged(self):
        flat = self._uniform()
        flat[:, 7] = np.nan
        bad = _detect_bad_columns_derivative(flat, {}, 1.0)
        self.assertEqual(list(bad), [7])

    def test_the_result_is_invariant_to_row_duplication(self):
        flat = self._uniform()
        flat[:, 20] = 0.05
        once = _detect_bad_columns_derivative(flat, {}, 1.0)
        twice = _detect_bad_columns_derivative(np.repeat(flat, 2, axis=0), {}, 1.0)
        self.assertEqual(list(once), list(twice))

    def test_the_dead_column_threshold_is_taken_from_the_config(self):
        flat = self._uniform()
        # 1.5 * global_med exceeds every column median, so every column is dead.
        bad = _detect_bad_columns_derivative(flat, {"col_dead_thresh": 1.5}, 1.0)
        self.assertEqual(list(bad), list(range(40)))

    def test_the_derivative_sigma_governs_whether_a_step_is_a_jump(self):
        # Five columns at 1.0 beside thirty-five at 5.0: the step is 4.0 and the
        # robust fallback scatter is ~0.64, so sigma 1.0 flags the leading
        # column of the step and the default 10.0 does not.
        flat = self._uniform()
        flat[:, :5] = 1.0
        flat[:, 5:] = 5.0
        self.assertEqual(list(_detect_bad_columns_derivative(flat, {}, 5.0)), [])
        self.assertEqual(list(_detect_bad_columns_derivative(flat, {"col_deriv_sigma": 1.0}, 5.0)), [4])


class TestFlatBadSettings(unittest.TestCase):
    def test_a_missing_or_wrong_shaped_section_yields_an_empty_digest(self):
        for section in (None, {}, 5, "flat_masking", ["flat_masking"], ()):
            with self.subTest(section=section):
                self.assertEqual(_flat_bad_settings(section), "{}")

    def test_cache_controls_do_not_enter_the_digest(self):
        self.assertEqual(_flat_bad_settings({"bad_mask_cache": False}), "{}")
        self.assertEqual(_flat_bad_settings({"bad_mask_cache_dir": "/tmp/x"}), "{}")
        self.assertEqual(
            _flat_bad_settings({"bad_mask_cache": True, "bad_mask_cache_dir": "/tmp/x", "local_filter_size": 15}),
            json.dumps({"local_filter_size": 15}, sort_keys=True),
        )

    def test_the_digest_is_key_order_independent(self):
        self.assertEqual(_flat_bad_settings({"a": 1, "b": 2}), _flat_bad_settings({"b": 2, "a": 1}))

    def test_a_retuned_setting_changes_the_digest(self):
        self.assertNotEqual(
            _flat_bad_settings({"local_filter_size": 15}),
            _flat_bad_settings({"local_filter_size": 21}),
        )

    def test_a_value_type_is_not_collapsed_into_a_string(self):
        self.assertNotEqual(_flat_bad_settings({"x": 1}), _flat_bad_settings({"x": "1"}))

    def test_an_unserialisable_value_still_returns_a_string(self):
        result = _flat_bad_settings({"x": {1, 2}})
        self.assertIsInstance(result, str)
        self.assertIn("x", result)


class TestParseSection(unittest.TestCase):
    def test_a_standard_datasection_maps_to_half_open_bounds(self):
        # FITS is 1-based inclusive; the helper returns 0-based exclusive.
        self.assertEqual(_parse_section("[33:2080,1:4612]"), (0, 4612, 32, 2080))

    def test_reversed_bounds_are_normalised(self):
        self.assertEqual(_parse_section("[2080:33,4612:1]"), (0, 4612, 32, 2080))
        self.assertEqual(_parse_section("[10:1,20:2]"), (1, 20, 0, 10))

    def test_surrounding_space_is_tolerated(self):
        self.assertEqual(_parse_section("[ 1 : 10 , 2 : 20 ]"), (1, 20, 0, 10))

    def test_negative_bounds_are_parsed_and_left_unclamped(self):
        # Clamping belongs to the caller; the parser reports what the header said.
        self.assertEqual(_parse_section("[-5:10,1:10]"), (0, 10, -6, 10))

    def test_a_non_string_is_not_a_section(self):
        for section in (None, 5, 1.5, b"[1:2,1:2]", ["[1:2,1:2]"], {"DATASEC": "[1:2,1:2]"}):
            with self.subTest(section=section):
                self.assertIsNone(_parse_section(section))

    def test_a_zero_bound_is_rejected(self):
        for section in ("[0:10,1:10]", "[1:10,0:10]", "[1:0,1:10]", "[1:10,1:0]"):
            with self.subTest(section=section):
                self.assertIsNone(_parse_section(section))

    def test_a_malformed_section_is_rejected(self):
        for section in (
            "",
            "[]",
            "1:10,1:10",
            "[1:10]",
            "[1:10,2:20",
            "[a:b,c:d]",
            "[1.5:10,1:10]",
            "garbage",
        ):
            with self.subTest(section=section):
                self.assertIsNone(_parse_section(section))


if __name__ == "__main__":
    unittest.main()
