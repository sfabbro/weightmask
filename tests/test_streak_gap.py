"""Regression tests for gap-tolerant streak-mask support selection."""

import unittest
from unittest.mock import patch

import numpy as np

from weightmask.streaks import _largest_contiguous_run, _refine_trail_mask


class TestGapTolerantStreakSupport(unittest.TestCase):
    def test_narrow_refit_cannot_replace_a_coherent_trail_with_a_short_fragment(self):
        rng = np.random.default_rng(0)
        yy, xx = np.indices((200, 40))
        strips = []
        for rows, columns in ((slice(5, 195), slice(18, 23)), (slice(60, 100), slice(20, 21))):
            sampled = rng.normal(0, 0.1, yy.shape).astype(np.float32)
            sampled[rows, columns] += 20
            strips.append({"sampled": sampled, "inside": np.ones(yy.shape, bool), "y_coords": yy, "x_coords": xx})
        candidate = {"source": "houghpeaks", "clipped_endpoints": ((20, 0), (20, 199))}
        config = {
            "padding": 0,
            "profile_percentile": 85,
            "min_col_hit_fraction": 0.1,
            "min_row_hit_fraction": 0.5,
            "max_support_width": 8,
        }
        data = np.zeros(yy.shape, np.float32)
        with (
            patch("weightmask.streaks._sample_trail_strip", side_effect=strips),
            patch("weightmask.streaks._refit_endpoints_from_strip", return_value=candidate["clipped_endpoints"]),
        ):
            mask, info = _refine_trail_mask(data, np.ones_like(data), candidate, config)
        self.assertGreater(info["row_hit_fraction"], 0.9)
        self.assertGreater(mask.sum(), 900)

    def test_short_gaps_do_not_truncate_one_validated_trail(self):
        support = np.zeros(2200, dtype=bool)
        support[100:1100] = True
        support[1120:1380] = True  # 19 missing samples
        support[1400:1660] = True  # 19 missing samples
        support[2100:2140] = True  # a distant, unrelated run

        selected = _largest_contiguous_run(support, max_gap=24)

        self.assertTrue(selected[100:1100].all())
        self.assertTrue(selected[1120:1380].all())
        self.assertTrue(selected[1400:1660].all())
        self.assertFalse(selected[1100:1120].any())
        self.assertFalse(selected[2100:2140].any())

    def test_short_isolated_runs_are_not_used_as_gap_bridges(self):
        support = np.zeros(160, dtype=bool)
        support[10:50] = True
        support[58:62] = True
        support[70:110] = True

        selected = _largest_contiguous_run(support, max_gap=24, min_run_length=8)

        self.assertTrue(selected[10:50].all())
        self.assertTrue(selected[70:110].all())
        self.assertFalse(selected[58:62].any())

    def test_minimum_run_length_also_rejects_a_single_short_run(self):
        support = np.zeros(100, dtype=bool)
        support[10:14] = True
        self.assertFalse(_largest_contiguous_run(support, max_gap=24, min_run_length=8).any())

    def test_masked_samples_do_not_split_support_or_become_detection(self):
        support = np.zeros(160, dtype=bool)
        support[10:50] = support[90:130] = True
        valid = np.ones_like(support)
        valid[50:90] = False
        selected = _largest_contiguous_run(support, min_run_length=8, valid=valid)
        self.assertTrue(selected[10:50].all())
        self.assertTrue(selected[90:130].all())
        self.assertFalse(selected[50:90].any())

    def test_zero_gap_retains_the_longest_run_only(self):
        support = np.zeros(100, dtype=bool)
        support[10:30] = True
        support[40:55] = True

        selected = _largest_contiguous_run(support, max_gap=0)

        self.assertTrue(selected[10:30].all())
        self.assertFalse(selected[40:55].any())

    def test_only_full_span_hough_candidates_may_bridge_gaps(self):
        data = np.zeros((120, 260), dtype=np.float32)
        for start, stop in ((20, 70), (90, 140), (160, 220)):
            data[59:62, start:stop] = 20.0
        candidate = {
            "segments": [((10.0, 60.0), (240.0, 60.0))],
            "angle_deg": 0.0,
            "endpoints": ((10.0, 60.0), (240.0, 60.0)),
            "clipped_endpoints": ((10.0, 60.0), (240.0, 60.0)),
            "span": 230.0,
            "raw_span": 230.0,
            "edge_touches": 2,
        }
        mask_cfg = {
            "strip_length": 256,
            "strip_width": 80,
            "profile_sigma_threshold": 1.0,
            "profile_percentile": 50.0,
            "padding": 1,
            "min_mask_pixels": 10,
            "min_row_hits": 3,
            "min_row_hit_fraction": 0.1,
            "min_col_hit_fraction": 0.1,
            "max_support_width": 30,
            "max_row_gap": 20,
        }
        rms = np.ones_like(data)

        contour_candidate = dict(candidate, source="contours")
        contour_mask, _ = _refine_trail_mask(data, rms, contour_candidate, mask_cfg)
        self.assertEqual(int(np.count_nonzero(contour_mask[59:62, 20:70])), 0)
        self.assertGreater(int(np.count_nonzero(contour_mask[59:62, 160:220])), 0)

        hough_candidate = dict(candidate, source="houghpeaks")
        hough_mask, _ = _refine_trail_mask(data, rms, hough_candidate, mask_cfg)
        for start, stop in ((20, 70), (90, 140), (160, 220)):
            self.assertGreater(int(np.count_nonzero(hough_mask[59:62, start:stop])), 0)


if __name__ == "__main__":
    unittest.main()
