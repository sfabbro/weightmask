"""Streak behavior regression tests (kept with the streak overhaul commit)."""

import unittest
import warnings

import numpy as np
from skimage.measure import LineModelND

import weightmask.streaks as streaks
from benchmarks.streak_inject import trail_flux, trail_recall
from weightmask.config import default_config, section_config
from weightmask.streaks import _detect_trails_sparse_ransac, detect_streaks


def _ransac_cfg(**params):
    return section_config("streak_masking", {"sparse_ransac_params": params})


class TestStreakRetiredKeys(unittest.TestCase):
    def test_obsolete_parameters_fail_even_when_disabled(self):
        data = np.zeros((8, 8), dtype=np.float32)
        for key in (
            "mode",
            "debug",
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
                with (
                    self.subTest(key=key, enabled=enabled),
                    self.assertRaisesRegex(ValueError, "Unsupported configuration key"),
                ):
                    detect_streaks(data, None, None, {"enable": enabled, key: None}).mask


class TestResidualRansac(unittest.TestCase):
    def test_rejected_clutter_does_not_consume_the_trail_budget(self):
        image = np.zeros((500, 500), dtype=np.float32)
        image[20:100, 80:85] = 8.0
        image[320, 220:400] = 8.0
        config = _ransac_cfg(min_length=100, min_line_density=0.6, max_trials=300, max_trails=1)
        mask = _detect_trails_sparse_ransac(image, np.ones_like(image), None, config)
        self.assertGreaterEqual(float(mask[320, 220:400].mean()), 0.95)
        self.assertFalse(np.any(mask[20:100, 80:85]))

    def test_a_primary_trail_does_not_hide_a_separate_dashed_trail(self):
        config = default_config()["streak_masking"]
        shape = (700, 700)
        image = np.random.default_rng(15).normal(0, 1, shape).astype(np.float32)
        primary_flux, primary_truth = trail_flux(shape, 500, 8, False, np.radians(8), 260, 150)
        residual_flux, residual_truth = trail_flux(shape, 240, 14, True, np.radians(65), 530, 500, dash_on=15)
        image += primary_flux + residual_flux
        mask = detect_streaks(image, np.ones(shape, np.float32), None, config).mask
        self.assertGreaterEqual(trail_recall(mask, primary_truth)[1], 0.95)
        self.assertGreaterEqual(trail_recall(mask, residual_truth)[1], 0.95)

    def test_crowded_field_rejection_cascade_is_bounded(self):
        """A field with no line to find must not spend unbounded time proposing models.

        Measured on non-sidereal frames: 154-236k residual candidates per HDU, and
        the rejection cascade ran until the candidate set drained -- 230+ passes in
        one pod against 24 for the same HDU locally, with ~6,800 passes at ~3.5 s
        each as the worst case. The bound is on the model budget, not on the pixels
        a model may explain; the re-scoring that keeps the cap honest is pinned by
        the next test.
        """

        rng = np.random.default_rng(3)
        image = np.zeros((600, 600), dtype=np.float32)
        # Sparse clutter: dense enough that the cap binds, thin enough that no
        # corridor reaches the density gate (clutter fraction ~3% against 0.6).
        image[rng.integers(0, 600, 12000), rng.integers(0, 600, 12000)] = 8.0

        calls = {"n": 0, "sizes": []}
        real = streaks.ransac

        def counting(coords, model, **kwargs):
            calls["n"] += 1
            calls["sizes"].append(len(coords))
            return real(coords, model, **kwargs)

        streaks.ransac = counting
        try:
            config = _ransac_cfg(max_trials=50, max_candidates=5000, max_rejected=12)
            mask = streaks._detect_trails_sparse_ransac(image, np.ones_like(image), None, config)
        finally:
            streaks.ransac = real
        self.assertFalse(np.any(mask))
        self.assertGreaterEqual(calls["n"], 1)
        self.assertLessEqual(calls["n"], 13)
        self.assertTrue(all(size <= 5000 for size in calls["sizes"]), calls["sizes"][:3])

    def test_candidate_cap_does_not_change_what_a_model_explains(self):
        """The fit is capped; the model is not.

        A stride sample of a crowded field holds only a fraction of a trail's
        pixels, so an accepted model is re-scored against every candidate. With the
        same model handed to both runs, the capped and uncapped outputs must be
        identical -- the cap may bound the search, never the evidence.
        """

        image = np.zeros((400, 400), dtype=np.float32)
        image[200, 50:300] = 8.0  # the trail the injected model will claim
        rng = np.random.default_rng(5)
        image[rng.integers(0, 400, 5000), rng.integers(0, 400, 5000)] = 8.0  # the cap binds here

        points = np.array([[200.0, 50.0], [200.0, 300.0]])
        if hasattr(LineModelND, "from_estimate"):
            model = LineModelND.from_estimate(points)  # skimage >= 0.26
        else:  # the spelling older supported versions have
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", FutureWarning)
                model = LineModelND().estimate(points)
        real = streaks.ransac

        def fixed(coords, _model_cls, **kwargs):
            return model, model.residuals(coords) <= kwargs["residual_threshold"]

        def run(params):
            streaks.ransac = fixed
            try:
                return streaks._detect_trails_sparse_ransac(image, np.ones_like(image), None, _ransac_cfg(**params))
            finally:
                streaks.ransac = real

        base = {"min_inliers": 25, "min_length": 100, "min_line_density": 0.6, "max_trails": 1}
        uncapped = run(dict(base))
        capped = run({**base, "max_candidates": 500})
        self.assertTrue(np.any(uncapped))
        np.testing.assert_array_equal(capped, uncapped)

    def test_residual_detection_is_invariant_to_adu_scale(self):
        config = _ransac_cfg(detect_thresh_sig=6.0)
        image = np.zeros((256, 256), dtype=np.float32)
        image[128, 50:200] = 6.8 * 5.0
        rms = np.full(image.shape, 5.0, dtype=np.float32)
        original = _detect_trails_sparse_ransac(image, rms, None, config)
        scaled = _detect_trails_sparse_ransac(image * 6, rms * 6, None, config)
        self.assertTrue(np.any(original))
        np.testing.assert_array_equal(scaled, original)


class TestOccludedTrail(unittest.TestCase):
    def test_unrelated_trails_cannot_revive_a_vetoed_prediction(self):
        config = default_config()["streak_masking"]
        shape = (700, 700)
        yy, xx = np.mgrid[: shape[0], : shape[1]]
        image = np.random.default_rng(9).normal(0, 1, shape).astype(np.float32)
        image[346:353, 60:640] += 100
        image[60:640, 196:203] += 8
        image[60:640, 496:503] += 8
        existing = (xx - 350) ** 2 + (yy - 350) ** 2 < 50**2
        image += (100 * np.exp(-((xx - 350) ** 2 + (yy - 350) ** 2) / (2 * 15**2))).astype(np.float32)
        mask = detect_streaks(image, np.ones(shape, np.float32), existing, config).mask
        self.assertFalse(mask[350, 350])
        self.assertFalse(np.any(mask[346:353, 80:180]))
        self.assertGreaterEqual(float(mask[100:600, 196:203].mean()), 0.9)
        self.assertGreaterEqual(float(mask[100:600, 496:503].mean()), 0.9)

    def test_confirmed_trail_crosses_a_known_star_but_not_unknown_noise(self):
        config = default_config()["streak_masking"]
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
                mask = detect_streaks(image, rms, existing, config).mask
                self.assertTrue(mask[193, 300])
                self.assertTrue(mask[207, 400])
                self.assertEqual(bool(mask[200, 350]), measured)


if __name__ == "__main__":
    unittest.main()
