"""The Radon rescue is gone, and the surviving stages are qualified in CI."""

import contextlib
import copy
import io
import unittest
from pathlib import Path

import numpy as np
import yaml

from weightmask import streaks
from weightmask.streaks import detect_streaks

REPO = Path(__file__).resolve().parents[1]


def _shipped_streak_config():
    return yaml.safe_load(open(REPO / "weightmask.yml"))["streak_masking"]


def _quiet_scene(shape=(256, 256)):
    rng = np.random.default_rng(5)
    data = rng.normal(0.0, 1.0, shape).astype(np.float32)
    return data, np.full(shape, 5.0, dtype=np.float32), np.zeros(shape, dtype=bool)


class TestRescueStageIsGone(unittest.TestCase):
    def test_streaks_module_has_no_radon_stage(self):
        for gone in (
            "_detect_streaks_mrt_like",
            "_radon_projections",
            "_radon_pad_offsets",
            "_valid_rho_range",
            "_subtract_hot_axis_bands",
            "_PERIMETER_4CONN_CORNER_WEIGHT",
            "_PERIMETER_HV_STEP_WEIGHT",
            "_PERIMETER_DIAG_STEP_WEIGHT",
        ):
            self.assertFalse(hasattr(streaks, gone), f"{gone} should be removed")

    def test_debug_flag_populates_last_run_without_mrt(self):
        data, rms, existing_mask = _quiet_scene((64, 64))
        config = dict(_shipped_streak_config())
        config["debug"] = True
        with contextlib.redirect_stdout(io.StringIO()):
            detect_streaks(data, rms, existing_mask, config)
        self.assertIn("_last_run", config)
        self.assertNotIn("mrt", config["_last_run"])
        self.assertEqual(set(config["_last_run"].keys()), {"mode", "houghpeaks", "contours", "sparse_ransac"})

    def test_shipped_config_carries_no_rescue_or_satdet_keys(self):
        streak_masking = _shipped_streak_config()
        for gone in (
            "mrt_rescue_params",
            "satdet_params",
            "retry_without_existing_mask",
            "retry_if_area_fraction_exceeds",
            "retry_if_support_width_exceeds",
        ):
            self.assertNotIn(gone, streak_masking, f"{gone} is dead config")

    def test_a_stale_rescue_key_fails_before_detection(self):
        data, rms, existing_mask = _quiet_scene()
        config = dict(_shipped_streak_config())
        config["mrt_rescue_params"] = {"enable": True, "max_candidates": 4}
        with self.assertRaisesRegex(ValueError, "mrt_rescue_params"):
            detect_streaks(data, rms, existing_mask, config)


class TestDetectorStillWorksWithoutTheRescue(unittest.TestCase):
    def test_a_long_bright_line_is_still_detected(self):
        data, rms, existing_mask = _quiet_scene((400, 400))
        data[195:199, 20:380] = 40.0
        with contextlib.redirect_stdout(io.StringIO()):
            mask = detect_streaks(data, rms, existing_mask, dict(_shipped_streak_config()))
        self.assertGreater(int(mask[195:199, 20:380].sum()), 0)

    def test_a_quiet_frame_still_produces_nothing(self):
        data, rms, existing_mask = _quiet_scene()
        with contextlib.redirect_stdout(io.StringIO()):
            mask = detect_streaks(data, rms, existing_mask, dict(_shipped_streak_config()))
        self.assertEqual(int(np.count_nonzero(mask)), 0)


class TestSurvivingStagesEarnTheirCost(unittest.TestCase):
    def _config(self, *, contours=False, ransac=False):
        config = copy.deepcopy(_shipped_streak_config())
        config["houghpeak_params"]["enable"] = False
        config["contour_params"]["enable"] = contours
        config["enable_sparse_ransac"] = ransac
        return config

    @staticmethod
    def _run(data, config):
        rms = np.full(data.shape, 5.0, dtype=np.float32)
        existing = np.zeros(data.shape, dtype=bool)
        with contextlib.redirect_stdout(io.StringIO()):
            return detect_streaks(data, rms, existing, config)

    def test_contour_stage_is_the_sole_finder_in_a_current_fixture(self):
        data = np.zeros((256, 256), dtype=np.float32)
        data[126:130, 20:236] = 40.0
        truth = np.zeros(data.shape, dtype=bool)
        truth[126:130, 20:236] = True
        on = self._run(data, self._config(contours=True))
        off = self._run(data, self._config())
        self.assertGreater(float(np.count_nonzero(on & truth)) / np.count_nonzero(truth), 0.5)
        self.assertEqual(int(np.count_nonzero(off)), 0)

    def test_ransac_stage_is_the_sole_finder_in_a_current_fixture(self):
        data = np.zeros((256, 256), dtype=np.float32)
        truth = np.zeros(data.shape, dtype=bool)
        for x in range(20, 236):
            y = 100 + int(0.35 * (x - 20))
            if x % 20 < 14:
                data[y - 1 : y + 2, x] = 40.0
                truth[y - 1 : y + 2, x] = True
        on = self._run(data, self._config(ransac=True))
        off = self._run(data, self._config())
        self.assertGreater(float(np.count_nonzero(on & truth)) / np.count_nonzero(truth), 0.5)
        self.assertEqual(int(np.count_nonzero(off)), 0)


if __name__ == "__main__":
    unittest.main()
