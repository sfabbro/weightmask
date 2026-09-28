"""The Radon rescue gate: when the sensitive stage runs.

The rescue is the stage that exists to find what the cheap prescreen misses, so
it must run whenever the prescreen has not already masked enough to have handled
the frame -- and it must not run once it has. It used to be gated on a "low
confidence" flag owned by the deleted Canny/Hough stage, which accepted nothing
on any real amp, so the gate was a cost heuristic rather than a decision about
the image.
"""

import unittest
from unittest import mock

import numpy as np

from weightmask import streaks
from weightmask.streaks import detect_streaks


def _streak_config():
    return {
        "enable": True,
        "mode": "auto_ground",
        "houghpeak_params": {"enable": False},
        "contour_params": {"enable": False},
        "mrt_rescue_params": {"theta_step_deg": 4.0, "peak_threshold_sig": 3.0, "max_candidates": 2},
        "mask_params": {
            "strip_length": 120,
            "strip_width": 48,
            "profile_sigma_threshold": 1.0,
            "profile_percentile": 75.0,
            "padding": 2,
            "min_mask_pixels": 10,
            "min_row_hits": 4,
            "min_row_hit_fraction": 0.2,
            "max_support_width": 12,
        },
        "enable_sparse_ransac": False,
    }


def _quiet_scene(shape=(256, 256)):
    rng = np.random.default_rng(5)
    data = rng.normal(0.0, 1.0, shape).astype(np.float32)
    return data, np.full(shape, 5.0, dtype=np.float32), np.zeros(shape, dtype=bool)


class TestMrtRescueGate(unittest.TestCase):
    def test_rescue_runs_when_the_prescreen_masked_nothing(self):
        data, rms, existing_mask = _quiet_scene()
        config = _streak_config()
        original = streaks._detect_streaks_mrt_like
        calls = []

        def counting(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        with mock.patch.object(streaks, "_detect_streaks_mrt_like", side_effect=counting):
            detect_streaks(data, rms, existing_mask, config)
        self.assertEqual(len(calls), 1, "an unmasked frame must reach the rescue pass")

    def test_rescue_is_skipped_once_the_prescreen_has_masked_enough(self):
        """The gate is about the image, not about cost."""
        data, rms, existing_mask = _quiet_scene()
        config = _streak_config()
        min_px = int(config["mask_params"]["min_mask_pixels"])

        def prescreen(*args, **kwargs):
            mask = np.zeros(data.shape, dtype=bool)
            mask[100:140, 40:60] = True  # comfortably over min_mask_pixels
            return mask, [{"endpoints": ((40, 100), (40, 140))}], {}

        with (
            mock.patch.object(streaks, "_detect_streaks_houghpeaks", side_effect=prescreen),
            mock.patch.object(streaks, "_detect_streaks_mrt_like") as rescue,
        ):
            detect_streaks(data, rms, existing_mask, config)
        self.assertGreater(min_px, 0)
        rescue.assert_not_called()

    def test_rescue_can_be_disabled(self):
        data, rms, existing_mask = _quiet_scene()
        config = _streak_config()
        config["mrt_rescue_params"] = {**config["mrt_rescue_params"], "enable": False}
        with mock.patch.object(streaks, "_detect_streaks_mrt_like") as rescue:
            detect_streaks(data, rms, existing_mask, config)
        rescue.assert_not_called()


class TestNoDeadStagesRemain(unittest.TestCase):
    def test_streaks_module_has_no_canny_hough_stage(self):
        """The stage is deleted, not merely disabled: no dead code left behind."""
        for gone in (
            "_detect_streaks_satdet",
            "_extract_multiscale_segments",
            "_build_satdet_candidates",
            "_cluster_segments",
            "_prune_small_edges",
            "_representative_line",
            "_line_corridor_mask",
            "_StreakImageCache",
        ):
            self.assertFalse(hasattr(streaks, gone), f"{gone} should be removed")

    def test_shipped_config_carries_no_satdet_or_retry_keys(self):
        from pathlib import Path

        import yaml

        repo = Path(__file__).resolve().parents[1]
        streak_masking = yaml.safe_load(open(repo / "weightmask.yml"))["streak_masking"]
        for gone in (
            "satdet_params",
            "retry_without_existing_mask",
            "retry_if_area_fraction_exceeds",
            "retry_if_support_width_exceeds",
        ):
            self.assertNotIn(gone, streak_masking, f"{gone} is dead config")


if __name__ == "__main__":
    unittest.main()
