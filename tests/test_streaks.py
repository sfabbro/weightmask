import contextlib
import io
import unittest

import numpy as np
from scipy.ndimage import binary_dilation, label

from weightmask.streaks import _drop_bright_components, _drop_premasked_components, detect_streaks


class TestStreaks(unittest.TestCase):
    def setUp(self):
        self.shape = (256, 256)
        self.rms = np.full(self.shape, 5.0, dtype=np.float32)
        self.empty_mask = np.zeros(self.shape, dtype=bool)

    def _config(self, **overrides):
        config = {
            "enable": True,
            "mode": "auto_ground",
            "houghpeak_params": {"enable": True, "bin": 2, "thresh_sig": 2.5, "min_votes": 40, "max_candidates": 6},
            "contour_params": {"enable": False},
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

    def _clutter_scene(self, seed, trail=False):
        rng = np.random.default_rng(seed)
        data_sub = np.zeros(self.shape, dtype=np.float32)
        source = np.zeros(self.shape, dtype=bool)
        source_signal = np.zeros(self.shape, dtype=np.float32)
        ys = rng.integers(10, self.shape[0] - 10, size=80)
        xs = rng.integers(10, self.shape[1] - 10, size=80)
        for y, x in zip(ys, xs):
            data_sub[y - 1 : y + 2, x - 1 : x + 2] += 30.0
            source[y - 1 : y + 2, x - 1 : x + 2] = True
            source_signal[y - 1 : y + 2, x - 1 : x + 2] += 30.0
        truth = np.zeros(self.shape, dtype=bool)
        if trail:
            for x in range(20, 236):
                y = 80 + int(0.35 * (x - 20))
                truth[y - 1 : y + 2, x - 1 : x + 2] = True
                data_sub[y - 1 : y + 2, x - 1 : x + 2] = 60.0
        return data_sub, source, source_signal, truth

    def _detect_quietly(self, data_sub, existing_mask, config):
        with contextlib.redirect_stdout(io.StringIO()):
            return detect_streaks(data_sub, self.rms, existing_mask, config)

    def test_detect_single_continuous_streak(self):
        data_sub = np.zeros(self.shape, dtype=np.float32)
        for x in range(20, 236):
            y = 80 + int(0.35 * (x - 20))
            data_sub[max(0, y - 1) : min(self.shape[0], y + 2), max(0, x - 1) : min(self.shape[1], x + 2)] = 80.0

        mask = detect_streaks(data_sub, self.rms, self.empty_mask, self._config())

        self.assertTrue(np.sum(mask) > 0)
        self.assertTrue(mask[120, 135] or mask[121, 135])

    def test_detect_multiple_streaks(self):
        data_sub = np.zeros(self.shape, dtype=np.float32)
        for x in range(10, 246):
            y1 = 40 + int(0.2 * x)
            y2 = 220 - int(0.4 * x)
            data_sub[max(0, y1 - 1) : min(self.shape[0], y1 + 2), x] = 60.0
            data_sub[max(0, y2 - 1) : min(self.shape[0], y2 + 2), x] = 60.0

        mask = detect_streaks(data_sub, self.rms, self.empty_mask, self._config())

        self.assertTrue(np.sum(mask) > 0)
        self.assertTrue(mask[80, 200])
        self.assertTrue(mask[140, 200] or mask[139, 200] or mask[141, 200])

    def test_reject_star_field_clutter_without_true_trail(self):
        config = self._config()
        config["mask_params"].update({"max_support_width": 6, "min_row_hit_fraction": 0.5})
        for seed in range(30):
            data_sub, source, source_signal, _truth = self._clutter_scene(seed)
            masked = self._detect_quietly(data_sub, source, config)
            components, n_components = label(masked, structure=np.ones((3, 3), dtype=bool))
            largest = max((int(np.count_nonzero(components == i)) for i in range(1, n_components + 1)), default=0)
            false_pixels_per_mpix = float(np.count_nonzero(masked)) / masked.size * 1.0e6
            source_flux_loss = float(source_signal[masked].sum() / source_signal[source].sum())
            self.assertLessEqual(false_pixels_per_mpix, 100.0)
            self.assertLessEqual(largest, 100)
            self.assertLessEqual(source_flux_loss, 0.01)

    def test_trail_precision_and_clean_source_loss_are_bounded(self):
        metrics = []
        for seed in range(30):
            data_sub, source, source_signal, truth = self._clutter_scene(seed, trail=True)
            masked = self._detect_quietly(data_sub, source, self._config())
            eval_truth = binary_dilation(truth, iterations=5)
            true_positive = int(np.count_nonzero(masked & eval_truth))
            predicted = int(np.count_nonzero(masked))
            precision = true_positive / predicted if predicted else 0.0
            recall = float(np.count_nonzero(masked & truth)) / np.count_nonzero(truth)
            false_pixels_per_mpix = float(np.count_nonzero(masked & ~eval_truth)) / masked.size * 1.0e6
            components, n_components = label(masked & ~eval_truth, structure=np.ones((3, 3), dtype=bool))
            largest = max((int(np.count_nonzero(components == i)) for i in range(1, n_components + 1)), default=0)
            clean_source = source & ~eval_truth
            source_flux_loss = float(source_signal[masked & clean_source].sum() / source_signal[clean_source].sum())
            metrics.append((precision, recall, false_pixels_per_mpix, largest, source_flux_loss))
        self.assertGreaterEqual(min(metric[0] for metric in metrics), 0.95)
        self.assertGreaterEqual(min(metric[1] for metric in metrics), 0.85)
        self.assertLessEqual(max(metric[2] for metric in metrics), 250.0)
        self.assertLessEqual(max(metric[3] for metric in metrics), 32)
        self.assertLessEqual(max(metric[4] for metric in metrics), 0.005)

    def test_preserve_sparse_trail_detection_with_ransac(self):
        data_sub = np.zeros(self.shape, dtype=np.float32)
        for x in range(20, 220, 18):
            y = 20 + x
            if y < self.shape[0]:
                data_sub[max(0, y - 1) : min(self.shape[0], y + 2), max(0, x - 1) : min(self.shape[1], x + 2)] = 120.0

        config = self._config(
            enable_sparse_ransac=True,
            sparse_ransac_params={
                "dilation_radius": 2,
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


class TestPremaskedVeto(unittest.TestCase):
    """A component the pipeline already flagged is not a new finding.

    On real MegaCam amps the dominant false positive is a saturated star's bleed,
    45% of whose pixels are already masked, against 2% for an unconfirmed linear
    feature. This is the cheapest of the four ways the two classes separate: both
    masks are already in hand at that point in the pipeline.
    """

    def _band(self, shape, y0, y1, x0, x1):
        m = np.zeros(shape, dtype=bool)
        m[y0:y1, x0:x1] = True
        return m

    def test_component_mostly_inside_the_upstream_mask_is_dropped(self):
        shape = (64, 64)
        # 40x10 band, 50% of it already masked.
        mask = self._band(shape, 20, 30, 10, 50)
        existing = self._band(shape, 20, 25, 10, 50)
        out = _drop_premasked_components(mask, existing, 0.25, min_pixels=10)
        self.assertEqual(int(out.sum()), 0)

    def test_component_mostly_outside_the_upstream_mask_is_kept(self):
        shape = (64, 64)
        mask = self._band(shape, 20, 30, 10, 50)
        existing = self._band(shape, 20, 21, 10, 15)  # ~5% overlap
        out = _drop_premasked_components(mask, existing, 0.25, min_pixels=10)
        np.testing.assert_array_equal(out, mask)

    def test_a_bleed_and_a_trail_in_one_mask_keeps_the_trail(self):
        """The veto is per component, so one stage can return both."""
        shape = (64, 64)
        bleed = self._band(shape, 5, 12, 5, 60)  # 7 x 55 = 385 px
        trail = self._band(shape, 40, 44, 5, 60)  # 4 x 55 = 220 px
        mask = bleed | trail
        existing = self._band(shape, 5, 9, 5, 60)  # 4 x 55 = 220 px, 57% of the bleed
        out = _drop_premasked_components(mask, existing, 0.25, min_pixels=10)
        np.testing.assert_array_equal(out, trail, "the bleed must go, the trail must survive")
        self.assertEqual(int(out.sum()), 220)

    def test_threshold_boundary_is_inclusive(self):
        shape = (64, 64)
        mask = self._band(shape, 20, 24, 10, 60)  # 4 x 50 = 200 px
        existing = self._band(shape, 20, 24, 10, 35)  # 4 x 25 = 100 px = exactly 0.5
        kept = _drop_premasked_components(mask, existing, 0.50, min_pixels=10)
        dropped = _drop_premasked_components(mask, existing, 0.49, min_pixels=10)
        self.assertEqual(int(kept.sum()), 200, "at the threshold the component is kept")
        self.assertEqual(int(dropped.sum()), 0, "above the threshold it is dropped")

    def test_disabled_threshold_is_a_no_op(self):
        shape = (64, 64)
        mask = self._band(shape, 20, 30, 10, 50)
        existing = self._band(shape, 20, 25, 10, 50)
        for disabled in (None, 1.0, 2.0):
            np.testing.assert_array_equal(_drop_premasked_components(mask, existing, disabled, 10), mask)

    def test_no_upstream_mask_is_a_no_op(self):
        shape = (64, 64)
        mask = self._band(shape, 20, 30, 10, 50)
        np.testing.assert_array_equal(_drop_premasked_components(mask, None, 0.25, 10), mask)

    def test_small_components_are_left_to_the_profile_gate(self):
        """Too small to judge on overlap; ``_gate_mask_by_profile`` owns those."""
        shape = (64, 64)
        mask = np.zeros(shape, dtype=bool)
        mask[30, 30] = True  # 1 px, 100% "overlapped"
        existing = mask.copy()
        out = _drop_premasked_components(mask, existing, 0.25, min_pixels=24)
        self.assertEqual(int(out.sum()), 1)

    def test_shipped_threshold_sits_between_the_two_measured_classes(self):
        from pathlib import Path

        import yaml

        repo = Path(__file__).resolve().parents[1]
        frac = yaml.safe_load(open(repo / "weightmask.yml"))["streak_masking"]["mask_params"]["max_premasked_fraction"]
        self.assertGreater(frac, 0.02, "must not touch unconfirmed linear features (measured 0.02)")
        self.assertLess(frac, 0.45, "must catch bleed (measured 0.45)")


class TestBrightnessVeto(unittest.TestCase):
    """A component orders of magnitude above the sky is not a trail.

    On real MegaCam amps, as a multiple of the local background RMS at p90:
    trails measure ~2 sigma, saturated-star bleed 54-66, and a group of
    near-saturated columns (97-99.5% of the SATURATE level, missed by the
    saturation stage by a hair and by ``bad.py`` because the flat shows those
    columns as ordinary) about 2000.
    """

    def _scene(self, noise=1.0):
        shape = (64, 64)
        return np.zeros(shape, dtype=np.float32), np.full(shape, noise, dtype=np.float32)

    def _band(self, shape, y0, y1, x0, x1):
        m = np.zeros(shape, dtype=bool)
        m[y0:y1, x0:x1] = True
        return m

    def test_a_trail_brightness_is_kept(self):
        shape = (64, 64)
        data, rms = self._scene()
        data[30:34, 5:60] = 2.0  # p90 = 2 sigma
        mask = self._band(shape, 30, 34, 5, 60)
        out = _drop_bright_components(mask, data, rms, 20.0, min_pixels=10)
        self.assertEqual(int(out.sum()), int(mask.sum()))

    def test_a_brightness_like_bleed_is_dropped(self):
        shape = (64, 64)
        data, rms = self._scene()
        data[30:34, 5:60] = 60.0  # p90 = 60 sigma
        mask = self._band(shape, 30, 34, 5, 60)
        out = _drop_bright_components(mask, data, rms, 20.0, min_pixels=10)
        self.assertEqual(int(out.sum()), 0)

    def test_a_near_saturated_column_group_is_dropped(self):
        shape = (64, 64)
        data, rms = self._scene()
        data[10:60, 20:24] = 2000.0  # ~2000 sigma
        mask = self._band(shape, 10, 60, 20, 24)
        out = _drop_bright_components(mask, data, rms, 20.0, min_pixels=10)
        self.assertEqual(int(out.sum()), 0)

    def test_threshold_boundary_is_inclusive(self):
        shape = (64, 64)
        data, rms = self._scene()
        data[30:34, 5:60] = 20.0
        mask = self._band(shape, 30, 34, 5, 60)
        self.assertEqual(int(_drop_bright_components(mask, data, rms, 20.0, 10).sum()), int(mask.sum()))
        self.assertEqual(int(_drop_bright_components(mask, data, rms, 19.0, 10).sum()), 0)

    def test_disabled_threshold_is_a_no_op(self):
        shape = (64, 64)
        data, rms = self._scene()
        data[10:60, 20:24] = 2000.0
        mask = self._band(shape, 10, 60, 20, 24)
        for disabled in (None, 1e9):
            np.testing.assert_array_equal(_drop_bright_components(mask, data, rms, disabled, 10), mask)

    def test_missing_rms_map_is_a_no_op(self):
        shape = (64, 64)
        data, _ = self._scene()
        mask = self._band(shape, 10, 60, 20, 24)
        np.testing.assert_array_equal(_drop_bright_components(mask, data, None, 20.0, 10), mask)

    def test_shipped_threshold_separates_the_two_measured_classes(self):
        from pathlib import Path

        import yaml

        repo = Path(__file__).resolve().parents[1]
        sigma = yaml.safe_load(open(repo / "weightmask.yml"))["streak_masking"]["mask_params"]["max_component_sigma"]
        self.assertGreater(sigma, 3.2, "must not touch unconfirmed linear features (measured 3.0/3.2 sigma p90)")
        self.assertLess(sigma, 155.0, "must catch bright star arm (155 sigma) and bleed (1138/1345 sigma)")


if __name__ == "__main__":
    unittest.main()
