import unittest

import numpy as np

from weightmask.streaks import detect_streaks


class TestStreaks(unittest.TestCase):
    def setUp(self):
        self.shape = (256, 256)
        self.rms = np.full(self.shape, 5.0, dtype=np.float32)
        self.empty_mask = np.zeros(self.shape, dtype=bool)

    def _satdet_config(self, **overrides):
        config = {
            "enable": True,
            "mode": "auto_ground",
            "dilation_radius": 2,
            "satdet_params": {
                "rescale_percentiles": [1.0, 99.5],
                "gaussian_sigma": 1.0,
                "gaussian_sigmas": [1.0, 1.5],
                "canny_low_threshold": 0.05,
                "canny_high_threshold": 0.2,
                "small_edge_perimeter": 5,
                "hough_threshold": 5,
                "hough_min_line_length": 40,
                "hough_max_line_gap": 6,
                "cluster_angle_tol_deg": 4.0,
                "cluster_rho_tol_px": 16.0,
                "min_cluster_segments": 3,
                "edge_buffer": 16,
                "min_edge_touches": 0,
                "min_interior_span": 120.0,
                "min_segment_density": 0.015,
                "candidate_corridor_radius": 8,
                "max_existing_mask_fraction": 0.8,
                "confidence_threshold": 0.2,
            },
            "mrt_rescue_params": {
                "theta_step_deg": 1.0,
                "peak_threshold_sig": 3.0,
                "max_candidates": 4,
                "confidence_threshold": 0.15,
            },
            "mask_params": {
                "strip_length": 180,
                "strip_width": 48,
                "profile_sigma_threshold": 1.0,
                "profile_percentile": 75.0,
                "rotation_interpolation_order": 1,
                "padding": 2,
                "min_mask_pixels": 10,
                "min_row_hits": 4,
                "min_row_hit_fraction": 0.2,
                "max_support_width": 12,
            },
            "enable_sparse_ransac": False,
        }
        config.update(overrides)
        return config

    def test_detect_single_continuous_streak_satdet(self):
        data_sub = np.zeros(self.shape, dtype=np.float32)
        for x in range(20, 236):
            y = 80 + int(0.35 * (x - 20))
            data_sub[max(0, y - 1) : min(self.shape[0], y + 2), max(0, x - 1) : min(self.shape[1], x + 2)] = 80.0

        mask = detect_streaks(data_sub, self.rms, self.empty_mask, self._satdet_config())

        self.assertTrue(np.sum(mask) > 0)
        self.assertTrue(mask[120, 135] or mask[121, 135])

    def test_detect_multiple_streaks_satdet(self):
        data_sub = np.zeros(self.shape, dtype=np.float32)
        for x in range(10, 246):
            y1 = 40 + int(0.2 * x)
            y2 = 220 - int(0.4 * x)
            data_sub[max(0, y1 - 1) : min(self.shape[0], y1 + 2), x] = 60.0
            data_sub[max(0, y2 - 1) : min(self.shape[0], y2 + 2), x] = 60.0

        mask = detect_streaks(data_sub, self.rms, self.empty_mask, self._satdet_config())

        self.assertTrue(np.sum(mask) > 0)
        self.assertTrue(mask[80, 200])
        self.assertTrue(mask[140, 200] or mask[139, 200] or mask[141, 200])

    def test_reject_star_field_clutter_without_true_trail(self):
        rng = np.random.default_rng(7)
        data_sub = rng.normal(0.0, 1.0, self.shape).astype(np.float32)
        ys = rng.integers(10, self.shape[0] - 10, size=80)
        xs = rng.integers(10, self.shape[1] - 10, size=80)
        data_sub[ys, xs] += 30.0
        existing_mask = np.zeros(self.shape, dtype=bool)
        for y, x in zip(ys, xs):
            existing_mask[max(0, y - 1) : min(self.shape[0], y + 2), max(0, x - 1) : min(self.shape[1], x + 2)] = True

        config = self._satdet_config()
        config["satdet_params"].update(
            {
                "min_cluster_segments": 4,
                "min_interior_span": 140.0,
                "max_existing_mask_fraction": 0.3,
                "confidence_threshold": 0.55,
            }
        )
        config["mask_params"].update({"max_support_width": 6, "min_row_hit_fraction": 0.5})

        masked = detect_streaks(data_sub, self.rms, existing_mask, config)

        self.assertLess(np.sum(masked), 0.35 * masked.size)

    def test_preserve_sparse_trail_detection_with_ransac(self):
        data_sub = np.zeros(self.shape, dtype=np.float32)
        for x in range(20, 220, 18):
            y = 20 + x
            if y < self.shape[0]:
                data_sub[max(0, y - 1) : min(self.shape[0], y + 2), max(0, x - 1) : min(self.shape[1], x + 2)] = 120.0

        config = self._satdet_config(
            enable_sparse_ransac=True,
            sparse_ransac_params={
                "detect_thresh_sig": 3.0,
                "residual_threshold": 2.0,
                "min_inliers": 5,
                "min_length": 50,
                "min_line_density": 0.03,
                "max_trials": 500,
                "max_trails": 2,
            },
        )

        mask = detect_streaks(data_sub, self.rms, self.empty_mask, config)

        self.assertTrue(np.sum(mask) > 0)
        self.assertTrue(mask[120, 100] or mask[121, 100] or mask[119, 100])

    def test_legacy_frangi_path_still_runs(self):
        data_sub = np.zeros(self.shape, dtype=np.float32)
        data_sub[120:123, 30:220] = 50.0

        config = {
            "enable": True,
            "mode": "legacy_compare",
            "dilation_radius": 1,
            "enable_sparse_ransac": False,
            "frangi_legacy_params": {
                "tophat_radius": 4,
                "sigmas": [1, 2],
                "min_area": 10,
                "min_elongation": 2.0,
                "block_size": 512,
                "block_pad": 16,
            },
        }

        mask = detect_streaks(data_sub, self.rms, self.empty_mask, config)

        self.assertEqual(mask.dtype, bool)
        self.assertTrue(np.sum(mask) > 0)


if __name__ == "__main__":
    unittest.main()
