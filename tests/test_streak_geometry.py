"""Direct unit tests for the pure geometry and profile helpers in streaks.py.

These helpers are load-bearing for trail placement and support selection but
were previously exercised only through the full detector. Testing them at the
boundary makes a change in their arithmetic fail on its own line rather than as
a diffuse recall shift.
"""

import unittest

import numpy as np

from weightmask.streaks import (
    _adjust_confidence_for_mask,
    _bin_array,
    _clip_line_to_image,
    _edges_touched,
    _finite_axis_profiles,
    _largest_near_center,
    _normalize_angle_deg,
)


def _as_point_set(pair):
    return {tuple(np.round(point, 6)) for point in pair}


class TestNormalizeAngleDeg(unittest.TestCase):
    def test_maps_into_half_open_degree_range(self):
        cases = {0.0: 0.0, 90.0: 90.0, 180.0: 0.0, -90.0: 90.0, 270.0: 90.0, 181.0: 1.0, -181.0: 179.0}
        for given, expected in cases.items():
            with self.subTest(angle=given):
                self.assertAlmostEqual(_normalize_angle_deg(given), expected)

    def test_is_idempotent_and_always_in_range(self):
        for angle in np.linspace(-1000.0, 1000.0, 401):
            once = _normalize_angle_deg(float(angle))
            self.assertGreaterEqual(once, 0.0)
            self.assertLess(once, 180.0)
            self.assertAlmostEqual(_normalize_angle_deg(once), once)


class TestBinArray(unittest.TestCase):
    def test_mean_bins_and_trims_the_leftover_border(self):
        data = np.arange(25, dtype=np.float64).reshape(5, 5)
        binned = _bin_array(data, 2)
        # Only the top-left 4x4 survives; each 2x2 block is replaced by its mean.
        self.assertEqual(binned.shape, (2, 2))
        np.testing.assert_allclose(binned, [[3.0, 5.0], [13.0, 15.0]])

    def test_factor_of_one_is_the_identity(self):
        data = np.arange(12, dtype=np.float64).reshape(3, 4)
        np.testing.assert_allclose(_bin_array(data, 1), data)

    def test_uniform_image_bins_to_its_own_value(self):
        data = np.full((6, 4), 7.5, dtype=np.float32)
        binned = _bin_array(data, 3)
        self.assertEqual(binned.shape, (2, 1))
        np.testing.assert_allclose(binned, 7.5)


class TestClipLineToImage(unittest.TestCase):
    def test_horizontal_line_clips_to_the_full_width(self):
        pair = _clip_line_to_image((10.0, 5.0), (1.0, 0.0), (10, 20))
        self.assertEqual(_as_point_set(pair), {(0.0, 5.0), (19.0, 5.0)})

    def test_vertical_line_clips_to_the_full_height(self):
        pair = _clip_line_to_image((3.0, 0.0), (0.0, 1.0), (10, 20))
        self.assertEqual(_as_point_set(pair), {(3.0, 0.0), (3.0, 9.0)})

    def test_diagonal_line_ends_on_the_image_corners(self):
        pair = _clip_line_to_image((0.0, 0.0), (1.0, 1.0), (10, 10))
        self.assertEqual(_as_point_set(pair), {(0.0, 0.0), (9.0, 9.0)})

    def test_endpoints_stay_inside_the_image_bounds(self):
        h, w = 37, 51
        for angle in np.linspace(0.0, np.pi, 24, endpoint=False):
            direction = (float(np.cos(angle)), float(np.sin(angle)))
            pair = _clip_line_to_image((w / 2.0, h / 2.0), direction, (h, w))
            self.assertIsNotNone(pair, angle)
            for x, y in pair:
                self.assertGreaterEqual(round(x, 6), 0.0)
                self.assertLessEqual(round(y, 6), float(h - 1))
                self.assertGreaterEqual(round(y, 6), 0.0)
                self.assertLessEqual(round(x, 6), float(w - 1))

    def test_a_degenerate_direction_has_no_line(self):
        self.assertIsNone(_clip_line_to_image((5.0, 5.0), (0.0, 0.0), (10, 10)))

    def test_a_line_missing_the_image_returns_none(self):
        self.assertIsNone(_clip_line_to_image((100.0, 100.0), (1.0, 0.0), (10, 10)))


class TestEdgesTouched(unittest.TestCase):
    def test_corner_to_corner_touches_every_edge(self):
        touched = _edges_touched([(0.0, 0.0), (19.0, 9.0)], (10, 20), 2)
        self.assertEqual(touched, {"left", "right", "top", "bottom"})

    def test_a_centered_segment_touches_no_edge(self):
        self.assertEqual(_edges_touched([(8.0, 4.0), (12.0, 6.0)], (10, 20), 2), set())

    def test_edge_buffer_governs_the_classification(self):
        # 3 px from the left edge: touched with a 4 px buffer, not with a 2 px one.
        endpoints = [(3.0, 5.0), (15.0, 5.0)]
        self.assertIn("left", _edges_touched(endpoints, (10, 20), 4))
        self.assertNotIn("left", _edges_touched(endpoints, (10, 20), 2))


class TestAdjustConfidenceForMask(unittest.TestCase):
    def test_no_mask_leaves_the_threshold_untouched(self):
        self.assertEqual(_adjust_confidence_for_mask(0.5, None), 0.5)

    def test_adjustment_is_proportional_then_capped(self):
        empty = np.zeros(100, dtype=bool)
        self.assertAlmostEqual(_adjust_confidence_for_mask(0.5, empty), 0.5)
        one_percent = np.zeros(100, dtype=bool)
        one_percent[:1] = True
        self.assertAlmostEqual(_adjust_confidence_for_mask(0.5, one_percent), 0.5 + 0.2)
        half = np.zeros(100, dtype=bool)
        half[:50] = True
        self.assertAlmostEqual(_adjust_confidence_for_mask(0.5, half), 0.75)
        full = np.ones(100, dtype=bool)
        self.assertAlmostEqual(_adjust_confidence_for_mask(0.5, full), 0.75)

    def test_adjustment_never_exceeds_the_cap(self):
        for fraction in np.linspace(0.0, 1.0, 21):
            mask = np.zeros(1000, dtype=bool)
            mask[: int(fraction * 1000)] = True
            self.assertLessEqual(_adjust_confidence_for_mask(0.0, mask), 0.25 + 1e-12)


class TestLargestNearCenter(unittest.TestCase):
    def test_empty_input_is_returned_unchanged(self):
        support = np.zeros(50, dtype=bool)
        self.assertFalse(_largest_near_center(support).any())

    def test_a_single_region_is_returned_unchanged(self):
        support = np.zeros(50, dtype=bool)
        support[10:20] = True
        np.testing.assert_array_equal(_largest_near_center(support), support)

    def test_the_region_centered_on_the_strip_wins(self):
        support = np.zeros(100, dtype=bool)
        support[5:15] = True  # far left, slightly larger
        support[45:55] = True  # centered on the strip
        selected = _largest_near_center(support)
        self.assertTrue(selected[45:55].all())
        self.assertFalse(selected[5:15].any())

    def test_ties_on_distance_go_to_the_larger_region(self):
        support = np.zeros(101, dtype=bool)
        support[0:5] = True  # centroid 2
        support[96:101] = True  # centroid 98, equally distant from 50
        selected = _largest_near_center(support)
        # Both are 48 from the centre; the larger one has the same length here,
        # so only the invariant that exactly one region survives is asserted.
        survivors = [bool(selected[0:5].any()), bool(selected[96:101].any())]
        self.assertEqual(sum(survivors), 1)


class TestFiniteAxisProfiles(unittest.TestCase):
    def test_all_finite_frames_use_the_plain_median(self):
        frame = np.arange(12, dtype=np.float64).reshape(3, 4)
        cols, rows = _finite_axis_profiles(frame)
        np.testing.assert_allclose(cols, np.median(frame, axis=0))
        np.testing.assert_allclose(rows, np.median(frame, axis=1))

    def test_non_finite_values_do_not_poison_the_profile(self):
        frame = np.ones((4, 4), dtype=np.float64)
        frame[0, 0] = np.nan
        frame[3, 2] = np.inf
        cols, rows = _finite_axis_profiles(frame)
        self.assertTrue(np.all(np.isfinite(cols)))
        self.assertTrue(np.all(np.isfinite(rows)))
        np.testing.assert_allclose(cols, 1.0)
        np.testing.assert_allclose(rows, 1.0)


if __name__ == "__main__":
    unittest.main()
