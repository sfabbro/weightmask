"""Streak behavior regression tests (kept with the streak overhaul commit).

Unknown modes must fail fast instead of silently returning empty masks;
production accepts only auto_ground through the mode key.
"""

import unittest
from pathlib import Path

import numpy as np
import yaml


class TestStreakUnknownMode(unittest.TestCase):
    def test_unknown_mode_raises(self):
        from weightmask.streaks import detect_streaks

        data = np.zeros((64, 64), dtype=np.float32)
        with self.assertRaises(ValueError):
            detect_streaks(data, None, None, {"enable": True, "mode": "bogus"})

    def test_only_auto_ground_accepted(self):
        from weightmask.streaks import _resolve_streak_mode

        self.assertEqual(_resolve_streak_mode({"mode": "auto_ground"}), "auto_ground")
        self.assertEqual(_resolve_streak_mode({}), "auto_ground")
        with self.assertRaises(ValueError):
            _resolve_streak_mode({"mode": "satdet"})

    def test_obsolete_parameters_fail_even_when_disabled(self):
        from weightmask.streaks import detect_streaks

        data = np.zeros((8, 8), dtype=np.float32)
        for key in (
            "method",
            "dilation_radius",
            "enable_ransac_trails",
            "ransac_params",
            "frangi_params",
            "frangi_legacy_params",
            "mrt_rescue_params",
            "satdet_params",
            "retry_without_existing_mask",
            "retry_if_area_fraction_exceeds",
            "retry_if_support_width_exceeds",
            "sparse_on_primary_weak_only",
        ):
            for enabled in (False, True):
                with self.subTest(key=key, enabled=enabled), self.assertRaisesRegex(ValueError, "Unsupported streak"):
                    detect_streaks(data, None, None, {"enable": enabled, key: None})

    def test_explicit_invalid_modes_are_not_defaulted_or_coerced(self):
        from weightmask.streaks import _resolve_streak_mode

        for mode in (None, "", "AUTO_GROUND", 4, [], True):
            with self.subTest(mode=mode), self.assertRaisesRegex(ValueError, "Unknown streak detection mode"):
                _resolve_streak_mode({"mode": mode})


class TestResidualRansac(unittest.TestCase):
    def test_rejected_clutter_does_not_consume_the_trail_budget(self):
        from weightmask.streaks import _detect_trails_sparse_ransac

        image = np.zeros((500, 500), dtype=np.float32)
        image[20:100, 80:85] = 8.0
        image[320, 220:400] = 8.0
        config = {
            "sparse_ransac_params": {"min_length": 100, "min_line_density": 0.6, "max_trials": 300, "max_trails": 1}
        }
        mask = _detect_trails_sparse_ransac(image, np.ones_like(image), None, config)
        self.assertGreaterEqual(float(mask[320, 220:400].mean()), 0.95)
        self.assertFalse(np.any(mask[20:100, 80:85]))

    def test_a_primary_trail_does_not_hide_a_separate_dashed_trail(self):
        from benchmarks.streak_inject import trail_flux, trail_recall
        from weightmask.streaks import detect_streaks

        config = yaml.safe_load((Path(__file__).resolve().parents[1] / "weightmask.yml").read_text())["streak_masking"]
        shape = (700, 700)
        image = np.random.default_rng(15).normal(0, 1, shape).astype(np.float32)
        primary_flux, primary_truth = trail_flux(shape, 500, 8, False, np.radians(8), 260, 150)
        residual_flux, residual_truth = trail_flux(shape, 240, 14, True, np.radians(65), 530, 500, dash_on=15)
        image += primary_flux + residual_flux
        mask = detect_streaks(image, np.ones(shape, np.float32), None, config)
        self.assertGreaterEqual(trail_recall(mask, primary_truth)[1], 0.95)
        self.assertGreaterEqual(trail_recall(mask, residual_truth)[1], 0.95)

    def test_residual_detection_is_invariant_to_adu_scale(self):
        from weightmask.streaks import _detect_trails_sparse_ransac

        config = {"sparse_ransac_params": {"detect_thresh_sig": 6.0}}
        image = np.zeros((256, 256), dtype=np.float32)
        image[128, 50:200] = 6.8 * 5.0
        rms = np.full(image.shape, 5.0, dtype=np.float32)
        original = _detect_trails_sparse_ransac(image, rms, None, config)
        scaled = _detect_trails_sparse_ransac(image * 6, rms * 6, None, config)
        self.assertTrue(np.any(original))
        np.testing.assert_array_equal(scaled, original)


class TestOccludedTrail(unittest.TestCase):
    def test_unrelated_trails_cannot_revive_a_vetoed_prediction(self):
        from weightmask.streaks import detect_streaks

        config = yaml.safe_load((Path(__file__).resolve().parents[1] / "weightmask.yml").read_text())["streak_masking"]
        shape = (700, 700)
        yy, xx = np.mgrid[: shape[0], : shape[1]]
        image = np.random.default_rng(9).normal(0, 1, shape).astype(np.float32)
        image[346:353, 60:640] += 100
        image[60:640, 196:203] += 8
        image[60:640, 496:503] += 8
        existing = (xx - 350) ** 2 + (yy - 350) ** 2 < 50**2
        image += (100 * np.exp(-((xx - 350) ** 2 + (yy - 350) ** 2) / (2 * 15**2))).astype(np.float32)
        mask = detect_streaks(image, np.ones(shape, np.float32), existing, config)
        self.assertFalse(mask[350, 350])
        self.assertFalse(np.any(mask[346:353, 80:180]))
        self.assertGreaterEqual(float(mask[100:600, 196:203].mean()), 0.9)
        self.assertGreaterEqual(float(mask[100:600, 496:503].mean()), 0.9)

    def test_confirmed_trail_crosses_a_known_star_but_not_unknown_noise(self):
        from benchmarks.streak_inject import trail_flux
        from weightmask.streaks import detect_streaks

        config = yaml.safe_load((Path(__file__).resolve().parents[1] / "weightmask.yml").read_text())["streak_masking"]
        shape = (400, 700)
        yy, xx = np.mgrid[: shape[0], : shape[1]]
        star = 100 * np.exp(-((xx - 350) ** 2 + (yy - 200) ** 2) / (2 * 15**2))
        image = np.random.default_rng(8).normal(0, 1, shape).astype(np.float32) + star.astype(np.float32)
        flux, _truth = trail_flux(shape, 500, 8, False, np.radians(8), 350, 200)
        image += flux
        existing = (xx - 350) ** 2 + (yy - 200) ** 2 < 50**2
        for measured in (True, False):
            rms = np.ones(shape, dtype=np.float32)
            if not measured:
                rms[existing] = np.inf
            with self.subTest(measured=measured):
                mask = detect_streaks(image, rms, existing, config)
                self.assertTrue(mask[193, 300])
                self.assertTrue(mask[207, 400])
                self.assertEqual(bool(mask[200, 350]), measured)


if __name__ == "__main__":
    unittest.main()
