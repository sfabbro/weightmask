"""The Radon rescue is gone, and the deletions are pinned.

The rescue was the *sensitive* stage -- the one meant to find what the cheap
prescreen misses. Before removal it was measured, not assumed:

* over 83 real amps it accepted on 3, and all three were false positives, while
  houghpeaks explained all 23,035 px of the real detections;
* on injected trails at 4/6/8/12 sigma, two lengths, two seeds, it moved
  ``recall_line`` by +0.000 in 8 of 8 cells;
* it cost 121.7 s/amp of a 125.7 s/amp stage.

So this file no longer tests a gate. It asserts the stage is *absent* -- deleted
functions, deleted config, and a detector that still runs and still finds real
trails without it. The recall curve that justified the removal lives in
``benchmarks/streak_recall_floor.py``; the cost attribution in
``benchmarks/streak_stage_sweep.py``. A future proposal to restore a faint-trail
stage must beat the recall curve it can now measure, not argue from this file.
"""

import contextlib
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
        """Deleted, not merely disabled: no dead code left behind."""
        for gone in (
            "_detect_streaks_mrt_like",
            "_radon_projections",
            "_radon_pad_offsets",
            "_valid_rho_range",
            "_subtract_hot_axis_bands",
        ):
            self.assertFalse(hasattr(streaks, gone), f"{gone} should be removed")

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

    def test_a_stale_rescue_key_in_a_user_config_is_simply_ignored(self):
        """A config carried over from an older release must not reach the stage.

        Nothing reads ``mrt_rescue_params`` any more, so the key is inert rather
        than an error. That is the only sensible reading for a tuning block whose
        stage is gone: refuse to invent behaviour for a knob nothing owns.
        """
        data, rms, existing_mask = _quiet_scene()
        config = dict(_shipped_streak_config())
        config["mrt_rescue_params"] = {"enable": True, "max_candidates": 4}
        with contextlib.redirect_stdout(io.StringIO()) as captured:
            mask = detect_streaks(data, rms, existing_mask, dict(config))
        self.assertEqual(mask.shape, data.shape)
        self.assertNotIn("MRT rescue", captured.getvalue())


class TestDetectorStillWorksWithoutTheRescue(unittest.TestCase):
    """Removing the rescue must not have removed the detector."""

    def test_a_long_bright_line_is_still_detected(self):
        data, rms, existing_mask = _quiet_scene((400, 400))
        data[195:199, 20:380] = 40.0
        with contextlib.redirect_stdout(io.StringIO()):
            mask = detect_streaks(data, rms, existing_mask, dict(_shipped_streak_config()))
        self.assertGreater(int(mask[195:199, 20:380].sum()), 0, "houghpeaks must still find a real trail")

    def test_a_quiet_frame_still_produces_nothing(self):
        data, rms, existing_mask = _quiet_scene()
        with contextlib.redirect_stdout(io.StringIO()):
            mask = detect_streaks(data, rms, existing_mask, dict(_shipped_streak_config()))
        self.assertEqual(int(np.count_nonzero(mask)), 0)


if __name__ == "__main__":
    unittest.main()
