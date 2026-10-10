"""Direct unit tests for the guarded saturation cascade in satur.py.

``detect_saturated_pixels`` composes four helpers that decide the saturation
level. They were previously only exercised end-to-end, so a change in the
precedence between the histogram, the plateau tail and the header advisory
surfaced as a shifted threshold rather than a failing assertion. These tests
pin each decision point on its own line.
"""

import unittest

import numpy as np

from weightmask.satur import (
    _MIN_CLUMP_PIXELS,
    _choose_saturation_level,
    _estimate_effective_full_scale,
    _estimate_plateau_tail,
    _get_saturation_from_header,
    _grow_bleed_down,
    _grow_bleed_up,
)


def _hist_config(**overrides):
    params = {"guard_fraction": 0.75, "max_upper_factor": 1.05}
    params.update(overrides)
    return {"histogram_params": params}


class TestSaturationFromHeader(unittest.TestCase):
    def test_a_single_string_keyword_is_read(self):
        header = {"SATURATE": 63123.0}
        self.assertEqual(_get_saturation_from_header(header, "SATURATE"), 63123.0)

    def test_the_first_parseable_keyword_wins(self):
        header = {"FIRST": 1234.0, "SECOND": 9999.0}
        self.assertEqual(_get_saturation_from_header(header, ("FIRST", "SECOND")), 1234.0)

    def test_a_missing_earlier_keyword_falls_through_to_a_later_one(self):
        header = {"SECOND": 9999.0}
        self.assertEqual(_get_saturation_from_header(header, ["FIRST", "SECOND"]), 9999.0)

    def test_an_unparseable_value_falls_through_to_the_next_keyword(self):
        header = {"BAD": "not-a-number", "GOOD": 5555.0}
        self.assertEqual(_get_saturation_from_header(header, ["BAD", "GOOD"]), 5555.0)

    def test_a_missing_keyword_returns_none(self):
        self.assertIsNone(_get_saturation_from_header({"OTHER": 1.0}, "SATURATE"))

    def test_an_unparseable_only_value_returns_none(self):
        self.assertIsNone(_get_saturation_from_header({"SATURATE": "abc"}, "SATURATE"))

    def test_a_missing_header_returns_none(self):
        self.assertIsNone(_get_saturation_from_header(None, "SATURATE"))

    def test_non_string_keywords_are_discarded(self):
        header = {"SATURATE": 1234.0}
        self.assertIsNone(_get_saturation_from_header(header, [None, "", 7]))


class TestEstimateEffectiveFullScale(unittest.TestCase):
    def test_explicit_config_scale_is_used(self):
        data = np.array([0.0, 100.0], dtype=np.float32)
        effective, advisory = _estimate_effective_full_scale(data, None, {"effective_full_scale": 65535.0}, "SATURATE")
        self.assertEqual(effective, 65535.0)
        self.assertIsNone(advisory)

    def test_a_small_integer_ceiling_beats_the_float_fallback(self):
        data = np.array([0, 100], dtype=np.uint8)
        effective, _ = _estimate_effective_full_scale(data, None, {}, "SATURATE")
        self.assertEqual(effective, 255.0)

    def test_a_header_advisory_becomes_a_candidate_and_is_returned(self):
        data = np.array([0.0, 100.0], dtype=np.float32)
        header = {"SATURATE": 3000.0}
        effective, advisory = _estimate_effective_full_scale(data, header, {}, "SATURATE")
        self.assertEqual(effective, 3000.0)
        self.assertEqual(advisory, 3000.0)

    def test_scales_below_the_data_range_are_rejected(self):
        # data_max 1e5 makes every candidate implausible (< 0.75 * data_max), so
        # the largest candidate is the only safe bound left.
        data = np.array([0.0, 1.0e5], dtype=np.float32)
        config = {"effective_full_scale": 1000.0, "fallback_level": 2000.0}
        effective, _ = _estimate_effective_full_scale(data, None, config, "SATURATE")
        self.assertEqual(effective, 2000.0)

    def test_non_positive_candidates_leave_the_hard_floor(self):
        data = np.array([0.0, 10.0], dtype=np.float32)
        config = {"effective_full_scale": -5.0, "fallback_level": 0.0}
        effective, _ = _estimate_effective_full_scale(data, None, config, "SATURATE")
        self.assertEqual(effective, 65535.0)

    def test_histogram_max_adu_is_honoured_as_a_candidate(self):
        data = np.array([0.0, 100.0], dtype=np.float32)
        config = {"histogram_params": {"hist_max_adu": 4096.0}}
        effective, _ = _estimate_effective_full_scale(data, None, config, "SATURATE")
        self.assertEqual(effective, 4096.0)


class TestEstimatePlateauTail(unittest.TestCase):
    def test_min_tail_pixels_must_be_a_positive_int(self):
        data = np.array([900.0, 900.0])
        for bad in (True, False, 2.5, 0, -3):
            with self.subTest(value=bad):
                config = {"histogram_params": {"min_tail_pixels": bad}}
                with self.assertRaises(ValueError):
                    _estimate_plateau_tail(data, 1000.0, config)

    def test_data_below_the_guard_fraction_has_no_plateau(self):
        data = np.zeros(50, dtype=np.float64) + 100.0
        self.assertEqual(_estimate_plateau_tail(data, 1000.0, {}), (None, 0))

    def test_a_tail_smaller_than_min_tail_pixels_has_no_plateau(self):
        data = np.array([900.0, 900.0, 100.0])
        config = {"histogram_params": {"min_tail_pixels": 8}}
        self.assertEqual(_estimate_plateau_tail(data, 1000.0, config), (None, 0))

    def test_repeated_levels_that_never_reach_the_count_have_no_plateau(self):
        data = np.array([760.0, 770.0, 780.0, 790.0, 800.0, 810.0, 820.0, 830.0])
        config = {"histogram_params": {"min_tail_pixels": 8}}
        self.assertEqual(_estimate_plateau_tail(data, 1000.0, config), (None, 0))

    def test_the_highest_qualifying_repeated_level_wins(self):
        data = np.concatenate(
            [
                np.full(8, 750.0),
                np.full(3, 800.0),
                np.full(8, 900.0),
            ]
        )
        config = {"histogram_params": {"min_tail_pixels": 8}}
        self.assertEqual(_estimate_plateau_tail(data, 1000.0, config), (900.0, 8))

    def test_empty_finite_data_has_no_plateau(self):
        self.assertEqual(_estimate_plateau_tail(np.array([]), 1000.0, {}), (None, 0))

    def test_non_finite_values_are_excluded_from_the_tail(self):
        raw = np.array([np.nan, np.nan, 900.0, 900.0, 900.0, 900.0, 900.0, 900.0, 900.0, 900.0])
        config = {"histogram_params": {"min_tail_pixels": 8}}
        levels, support = _estimate_plateau_tail(raw, 1000.0, config)
        self.assertEqual((levels, support), (900.0, 8))


class TestChooseSaturationLevel(unittest.TestCase):
    def test_a_guarded_histogram_level_wins(self):
        level, label = _choose_saturation_level(900.0, None, 0, 1000.0, None, _hist_config())
        self.assertEqual((level, label), (900.0, "histogram (guarded)"))

    def test_the_guard_window_is_inclusive_at_both_bounds(self):
        for hist in (750.0, 1050.0):
            with self.subTest(hist=hist):
                level, label = _choose_saturation_level(hist, None, 0, 1000.0, None, _hist_config())
                self.assertEqual((level, label), (hist, "histogram (guarded)"))

    def test_an_unguarded_histogram_level_is_ignored(self):
        level, label = _choose_saturation_level(500.0, None, 0, 1000.0, None, _hist_config())
        self.assertEqual((level, label), (1000.0, "default guarded fallback"))

    def test_a_supported_plateau_beats_the_header_advisory(self):
        level, label = _choose_saturation_level(None, 950.0, _MIN_CLUMP_PIXELS, 1000.0, 800.0, _hist_config())
        self.assertEqual((level, label), (950.0, "plateau-tail fallback"))

    def test_an_unsupported_plateau_yields_to_the_header_advisory(self):
        level, label = _choose_saturation_level(None, 950.0, _MIN_CLUMP_PIXELS - 1, 1000.0, 800.0, _hist_config())
        self.assertEqual((level, label), (800.0, "header advisory fallback"))

    def test_an_unsupported_plateau_is_still_used_without_an_advisory(self):
        level, label = _choose_saturation_level(None, 950.0, 1, 1000.0, None, _hist_config())
        self.assertEqual((level, label), (950.0, "plateau-tail fallback"))

    def test_an_unguarded_advisory_is_ignored(self):
        level, label = _choose_saturation_level(None, None, 0, 1000.0, 100.0, _hist_config())
        self.assertEqual((level, label), (1000.0, "default guarded fallback"))

    def test_everything_out_of_window_falls_back_to_the_full_scale(self):
        level, label = _choose_saturation_level(200.0, 300.0, 100, 1000.0, 400.0, _hist_config())
        self.assertEqual((level, label), (1000.0, "default guarded fallback"))


class TestGrowBleedUp(unittest.TestCase):
    """``y_min`` is the first core row; growth walks upward from ``y_min - 1``."""

    def _scene(self, rows, h=10):
        data = np.zeros((h, 1), dtype=np.float64)
        data[:, 0] = rows
        return data, np.full(h, 100.0)

    def test_grows_until_the_first_sample_below_the_threshold(self):
        # rows 5 and 4 are above threshold walking up from y_min - 1; row 3 is not.
        data, stop = self._scene([0, 0, 0, 50, 200, 200, 200, 0, 0, 0])
        new_mask = np.zeros_like(data, dtype=bool)
        _grow_bleed_up(data, stop, 0, 6, 10, new_mask)
        self.assertTrue(new_mask[5, 0])
        self.assertTrue(new_mask[4, 0])
        self.assertFalse(new_mask[3, 0])
        self.assertFalse(new_mask[2, 0])

    def test_the_core_row_itself_is_left_to_the_caller(self):
        data, stop = self._scene([0, 0, 0, 50, 200, 200, 200, 0, 0, 0])
        new_mask = np.zeros_like(data, dtype=bool)
        _grow_bleed_up(data, stop, 0, 6, 10, new_mask)
        self.assertFalse(new_mask[6, 0])

    def test_a_column_entirely_above_the_threshold_grows_to_the_top(self):
        data, stop = self._scene([200] * 10)
        new_mask = np.zeros_like(data, dtype=bool)
        _grow_bleed_up(data, stop, 0, 4, 10, new_mask)
        self.assertTrue(new_mask[0:4, 0].all())
        self.assertFalse(new_mask[4:, 0].any())

    def test_max_grow_zero_grows_nothing(self):
        data, stop = self._scene([200] * 10)
        new_mask = np.zeros_like(data, dtype=bool)
        _grow_bleed_up(data, stop, 0, 4, 0, new_mask)
        self.assertFalse(new_mask.any())

    def test_a_first_sample_already_below_threshold_grows_nothing(self):
        data, stop = self._scene([0, 0, 0, 50, 200, 200, 200, 0, 0, 0])
        new_mask = np.zeros_like(data, dtype=bool)
        _grow_bleed_up(data, stop, 0, 4, 10, new_mask)
        self.assertFalse(new_mask.any())

    def test_y_min_at_the_top_edge_is_a_no_op(self):
        data, stop = self._scene([200] * 10)
        new_mask = np.zeros_like(data, dtype=bool)
        _grow_bleed_up(data, stop, 0, 0, 10, new_mask)
        self.assertFalse(new_mask.any())

    def test_the_infinite_rms_sentinel_does_not_grow(self):
        data, stop = self._scene([200] * 10)
        stop = np.full(10, np.inf)
        new_mask = np.zeros_like(data, dtype=bool)
        _grow_bleed_up(data, stop, 0, 6, 10, new_mask)
        self.assertFalse(new_mask.any())


class TestGrowBleedDown(unittest.TestCase):
    def _scene(self, rows, h=10):
        data = np.zeros((h, 1), dtype=np.float64)
        data[:, 0] = rows
        return data, np.full(h, 100.0)

    def test_grows_until_the_first_sample_below_the_threshold(self):
        # y_max is 4, so growth walks down from row 5: rows 5 and 6 pass, row 7 fails.
        data, stop = self._scene([0, 0, 0, 500, 500, 200, 200, 50, 0, 0])
        new_mask = np.zeros_like(data, dtype=bool)
        _grow_bleed_down(data, stop, 10, 0, 4, 10, new_mask)
        self.assertTrue(new_mask[5, 0])
        self.assertTrue(new_mask[6, 0])
        self.assertFalse(new_mask[7, 0])
        self.assertFalse(new_mask[4, 0])

    def test_growth_stops_at_the_bottom_edge(self):
        data, stop = self._scene([500] * 10)
        new_mask = np.zeros_like(data, dtype=bool)
        _grow_bleed_down(data, stop, 10, 0, 8, 10, new_mask)
        self.assertTrue(new_mask[9, 0])
        self.assertFalse(new_mask[8, 0])

    def test_max_grow_zero_grows_nothing(self):
        data, stop = self._scene([500] * 10)
        new_mask = np.zeros_like(data, dtype=bool)
        _grow_bleed_down(data, stop, 10, 0, 2, 0, new_mask)
        self.assertFalse(new_mask.any())

    def test_a_first_sample_already_below_threshold_grows_nothing(self):
        data, stop = self._scene([0, 0, 0, 500, 500, 50, 200, 200, 0, 0])
        new_mask = np.zeros_like(data, dtype=bool)
        _grow_bleed_down(data, stop, 10, 0, 4, 10, new_mask)
        self.assertFalse(new_mask.any())

    def test_y_max_at_the_bottom_edge_is_a_no_op(self):
        data, stop = self._scene([500] * 10)
        new_mask = np.zeros_like(data, dtype=bool)
        _grow_bleed_down(data, stop, 10, 0, 9, 10, new_mask)
        self.assertFalse(new_mask.any())

    def test_the_infinite_rms_sentinel_does_not_grow(self):
        data, stop = self._scene([500] * 10)
        stop = np.full(10, np.inf)
        new_mask = np.zeros_like(data, dtype=bool)
        _grow_bleed_down(data, stop, 10, 0, 2, 10, new_mask)
        self.assertFalse(new_mask.any())


if __name__ == "__main__":
    unittest.main()
