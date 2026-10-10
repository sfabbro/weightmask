"""Direct unit tests for the Hough candidate/peak helpers in streaks.py.

Three of the six helpers named in the brief (`_image_at`, `_support_count`,
`_better`) are nested closures inside `_refit_endpoints_from_strip` and
`_refine_trail_mask`, so they cannot be imported or asserted on directly. This
module covers the three that are module-level: `_candidate_from_rho_theta`,
`_refine_hough_peak` and `_score_candidate`. The closures keep their coverage
through the existing `_refine_trail_mask` tests (`test_streak_gap.py`,
`test_profile_accept.py`).

Geometry is the point below: the candidate tests assert the recovered line
satisfies the Hesse identity about the image centre, and the peak tests assert
the analytic parabola offset, rather than pinning opaque scalars.
"""

import unittest

import numpy as np

from weightmask.streaks import _candidate_from_rho_theta, _refine_hough_peak, _score_candidate


def _centre(shape):
    h, w = shape
    return 0.5 * (w - 1), 0.5 * (h - 1)


class TestCandidateFromRhoTheta(unittest.TestCase):
    def _assert_hesse_identity(self, candidate, rho, theta_deg, shape):
        """Every endpoint lies on ``(p - centre) . normal == rho``."""
        cx, cy = _centre(shape)
        theta = np.radians(theta_deg)
        for x, y in candidate["clipped_endpoints"]:
            offset = (x - cx) * np.cos(theta) + (y - cy) * np.sin(theta)
            self.assertAlmostEqual(offset, rho, places=4)

    def test_theta_zero_is_a_vertical_line_through_the_centre(self):
        candidate = _candidate_from_rho_theta(0.0, 0.0, (10, 20))
        self.assertEqual(candidate["angle_deg"], 0.0)
        # theta = 0 has normal (1, 0), so both endpoints share the centre column.
        xs = sorted({round(x, 6) for x, _y in candidate["clipped_endpoints"]})
        self.assertEqual(xs, [9.5])
        self._assert_hesse_identity(candidate, 0.0, 0.0, (10, 20))

    def test_theta_ninety_is_a_horizontal_line_through_the_centre(self):
        candidate = _candidate_from_rho_theta(0.0, 90.0, (10, 20))
        self.assertAlmostEqual(candidate["angle_deg"], 90.0)
        ys = sorted({round(y, 6) for _x, y in candidate["clipped_endpoints"]})
        self.assertEqual(ys, [4.5])
        self._assert_hesse_identity(candidate, 0.0, 90.0, (10, 20))

    def test_positive_and_negative_rho_shift_the_line_both_ways(self):
        for rho in (3.0, -3.0, 7.5):
            with self.subTest(rho=rho):
                candidate = _candidate_from_rho_theta(rho, 0.0, (10, 20))
                expected_x = _centre((10, 20))[0] + rho
                xs = {round(x, 6) for x, _y in candidate["clipped_endpoints"]}
                self.assertEqual(xs, {round(expected_x, 6)})
                self._assert_hesse_identity(candidate, rho, 0.0, (10, 20))

    def test_oblique_theta_satisfies_the_hesse_identity(self):
        for theta_deg in (15.0, 30.0, 63.5, 120.0, 179.0):
            for rho in (-4.0, 0.0, 2.5):
                with self.subTest(theta=theta_deg, rho=rho):
                    candidate = _candidate_from_rho_theta(rho, theta_deg, (80, 120))
                    self.assertIsNotNone(candidate)
                    self._assert_hesse_identity(candidate, rho, theta_deg, (80, 120))

    def test_angle_is_normalized_into_half_open_degrees(self):
        cases = {0.0: 0.0, 90.0: 90.0, 179.0: 179.0, 180.0: 0.0, 181.0: 1.0, -30.0: 150.0, 270.0: 90.0}
        for given, expected in cases.items():
            with self.subTest(theta=given):
                candidate = _candidate_from_rho_theta(0.0, given, (60, 60))
                self.assertAlmostEqual(candidate["angle_deg"], expected)

    def test_span_is_the_endpoint_distance_and_matches_raw_span(self):
        candidate = _candidate_from_rho_theta(0.0, 0.0, (100, 100))
        (x0, y0), (x1, y1) = candidate["clipped_endpoints"]
        self.assertAlmostEqual(candidate["span"], float(np.hypot(x1 - x0, y1 - y0)))
        self.assertEqual(candidate["raw_span"], candidate["span"])
        # A vertical line through the centre of a 100x100 image spans 99 px.
        self.assertAlmostEqual(candidate["span"], 99.0)

    def test_segments_endpoints_and_clipped_endpoints_agree(self):
        candidate = _candidate_from_rho_theta(2.0, 45.0, (80, 80))
        ((sx0, sy0), (sx1, sy1)) = candidate["segments"][0]
        (x0, y0), (x1, y1) = candidate["endpoints"]
        self.assertEqual((sx0, sy0, sx1, sy1), (x0, y0, x1, y1))
        self.assertEqual(candidate["endpoints"], candidate["clipped_endpoints"])

    def test_a_centered_vertical_line_touches_top_and_bottom_only(self):
        candidate = _candidate_from_rho_theta(0.0, 0.0, (100, 100))
        self.assertEqual(candidate["edge_touches"], 2)

    def test_a_line_that_misses_the_image_is_none(self):
        # theta = 0 makes a vertical line; a rho beyond the half-width cannot
        # cross any column of the image.
        self.assertIsNone(_candidate_from_rho_theta(100.0, 0.0, (10, 20)))
        self.assertIsNone(_candidate_from_rho_theta(-100.0, 0.0, (10, 20)))

    def test_endpoints_are_reported_as_plain_floats(self):
        candidate = _candidate_from_rho_theta(1.5, 33.0, (40, 50))
        for point in candidate["segments"][0]:
            for coordinate in point:
                self.assertIsInstance(coordinate, float)


class TestRefineHoughPeak(unittest.TestCase):
    def _parabola_accumulator(self, rho_peak, theta_peak, thetas, rhos):
        rr = rhos[:, None] - rho_peak
        tt = np.degrees(thetas)[None, :] - theta_peak
        return -(rr**2 + tt**2)

    def test_a_synthetic_parabola_recovers_the_sub_bin_peak(self):
        thetas = np.radians(np.array([0.0, 1.0, 2.0, 3.0]))
        rhos = np.array([0.0, 1.0, 2.0])
        H = self._parabola_accumulator(1.2, 1.5, thetas, rhos)
        theta_deg, rho = _refine_hough_peak(H, thetas, rhos, 1, 1)
        # True peak is theta_deg 1.5, rho 1.2; the coarse bin is (1, 1).
        self.assertAlmostEqual(theta_deg, 1.5, places=6)
        self.assertAlmostEqual(rho, 1.2, places=6)

    def test_the_recovered_peak_is_sub_bin_not_a_grid_snap(self):
        thetas = np.radians(np.array([0.0, 1.0, 2.0]))
        rhos = np.array([0.0, 1.0, 2.0])
        H = self._parabola_accumulator(1.3, 1.7, thetas, rhos)
        theta_deg, rho = _refine_hough_peak(H, thetas, rhos, 1, 1)
        self.assertNotAlmostEqual(theta_deg, 1.0, places=3)
        self.assertNotAlmostEqual(rho, 1.0, places=3)
        self.assertAlmostEqual(theta_deg, 1.7, places=5)
        self.assertAlmostEqual(rho, 1.3, places=5)

    def test_a_flat_accumulator_keeps_the_coarse_grid_values(self):
        thetas = np.radians(np.array([0.0, 1.0, 2.0, 3.0]))
        rhos = np.array([5.0, 6.0, 7.0])
        H = np.ones((3, 4))
        theta_deg, rho = _refine_hough_peak(H, thetas, rhos, 2, 1)
        self.assertAlmostEqual(theta_deg, 2.0, places=6)
        self.assertAlmostEqual(rho, 6.0, places=6)

    def test_a_transposed_accumulator_gives_the_same_answer(self):
        thetas = np.radians(np.array([0.0, 1.0, 2.0, 3.0]))
        rhos = np.array([0.0, 1.0, 2.0])
        H = self._parabola_accumulator(1.25, 1.4, thetas, rhos)
        oriented = _refine_hough_peak(H, thetas, rhos, 1, 1)
        transposed = _refine_hough_peak(H.T, thetas, rhos, 1, 1)
        self.assertAlmostEqual(oriented[0], transposed[0], places=9)
        self.assertAlmostEqual(oriented[1], transposed[1], places=9)

    def test_the_accumulator_edge_is_clamped_rather_than_indexing_out(self):
        thetas = np.radians(np.array([0.0, 1.0, 2.0]))
        rhos = np.array([0.0, 1.0, 2.0])
        H = self._parabola_accumulator(0.0, 0.0, thetas, rhos)
        # Corner indices would read (-1, -1) and (3, 3) without the clamp.
        for ti, ri in ((0, 0), (0, 2), (2, 0), (2, 2)):
            with self.subTest(ti=ti, ri=ri):
                theta_deg, rho = _refine_hough_peak(H, thetas, rhos, ti, ri)
                self.assertTrue(np.isfinite(theta_deg))
                self.assertTrue(np.isfinite(rho))

    def test_single_bin_axes_fall_back_to_unit_spacing(self):
        thetas = np.radians(np.array([7.0]))
        rhos = np.array([3.0, 4.0, 5.0])
        H = np.array([[1.0, 5.0, 2.0]]).T  # shape (3, 1): rhos x thetas
        theta_deg, rho = _refine_hough_peak(H, thetas, rhos, 0, 1)
        self.assertAlmostEqual(theta_deg, 7.0, places=6)
        self.assertTrue(np.isfinite(rho))
        # A single theta bin has no neighbours to fit, so theta is unmoved.
        self.assertAlmostEqual(theta_deg, float(np.degrees(thetas[0])), places=9)

    def test_an_out_of_range_peak_index_is_clamped_rather_than_raised(self):
        thetas = np.radians(np.array([7.0]))
        rhos = np.array([3.0, 4.0, 5.0])
        H = np.array([[1.0, 5.0, 2.0]]).T
        # ti=1 does not exist on a one-bin theta axis; the neighbour probes are
        # clamped, so the returned grid value must be too.
        theta_deg, rho = _refine_hough_peak(H, thetas, rhos, 1, 0)
        self.assertAlmostEqual(theta_deg, 7.0, places=6)
        self.assertTrue(np.isfinite(rho))
        self.assertAlmostEqual(theta_deg, float(np.degrees(thetas[0])), places=9)


class TestScoreCandidate(unittest.TestCase):
    def _mask(self, count, size=100):
        mask = np.zeros(size, dtype=bool)
        mask[:count] = True
        return mask

    def _candidate(self, **overrides):
        candidate = {
            "span": 100.0,
            "segments": [((0.0, 0.0), (100.0, 0.0))],
            "edge_touches": 0,
            "corridor_overlap": 0.0,
        }
        candidate.update(overrides)
        return candidate

    def _info(self, **overrides):
        info = {"support_width": 8, "row_hit_fraction": 0.6}
        info.update(overrides)
        return info

    def test_an_absent_or_empty_mask_scores_zero(self):
        self.assertEqual(_score_candidate(self._candidate(), None, self._info(), None), 0.0)
        self.assertEqual(_score_candidate(self._candidate(), np.zeros(100, dtype=bool), self._info(), None), 0.0)

    def test_the_documented_weighting_is_pinned(self):
        score = _score_candidate(self._candidate(), self._mask(30), self._info(), None)
        # 0.45 * min(1, (30/100)/6) + 0.35 * min(1, 1/6) + 0.25 * min(1, 0.6/0.6)
        expected = 0.45 * 0.05 + 0.35 * (1.0 / 6.0) + 0.25 * 1.0
        self.assertAlmostEqual(score, expected, places=9)
        self.assertAlmostEqual(score, 0.3308333333333333, places=9)

    def test_more_support_raises_the_score_until_it_saturates(self):
        low = _score_candidate(self._candidate(), self._mask(10), self._info(), None)
        high = _score_candidate(self._candidate(), self._mask(300), self._info(), None)
        self.assertGreater(high, low)
        saturated = _score_candidate(self._candidate(), self._mask(600), self._info(), None)
        beyond = _score_candidate(self._candidate(), self._mask(950), self._info(), None)
        self.assertAlmostEqual(saturated, beyond, places=9)

    def test_wide_support_is_penalised(self):
        narrow = _score_candidate(self._candidate(), self._mask(300), self._info(support_width=8), None)
        wide = _score_candidate(self._candidate(), self._mask(300), self._info(support_width=20), None)
        self.assertAlmostEqual(narrow - wide, 0.45 * (20.0 - 8.0) / 12.0, places=9)

    def test_corridor_overlap_is_penalised(self):
        clean = _score_candidate(self._candidate(), self._mask(300), self._info(), None)
        dirty = _score_candidate(self._candidate(corridor_overlap=0.5), self._mask(300), self._info(), None)
        self.assertAlmostEqual(clean - dirty, 0.80 * 0.5, places=9)

    def test_existing_mask_overlap_is_penalised(self):
        refined = self._mask(300)
        clean = _score_candidate(self._candidate(), refined, self._info(), None)
        fully_overlapping = _score_candidate(self._candidate(), refined, self._info(), np.ones_like(refined))
        self.assertAlmostEqual(clean - fully_overlapping, 0.35, places=9)
        empty_existing = _score_candidate(self._candidate(), refined, self._info(), np.zeros_like(refined))
        self.assertAlmostEqual(clean, empty_existing, places=9)

    def test_more_segments_and_edge_touches_raise_the_score(self):
        base = _score_candidate(self._candidate(), self._mask(300), self._info(), None)
        segmented = _score_candidate(
            self._candidate(segments=[((0.0, 0.0), (100.0, 0.0))] * 3), self._mask(300), self._info(), None
        )
        self.assertGreater(segmented, base)
        edged = _score_candidate(self._candidate(edge_touches=2), self._mask(300), self._info(), None)
        self.assertGreater(edged, base)
        self.assertAlmostEqual(edged - base, 0.20, places=9)

    def test_the_score_is_a_plain_float(self):
        score = _score_candidate(self._candidate(), self._mask(300), self._info(), None)
        self.assertIsInstance(score, float)


if __name__ == "__main__":
    unittest.main()
