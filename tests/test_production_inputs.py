"""The benchmark input capture must equal what ``process_image`` really passes.

The curated false-positive score and the injected-trail recall grid both used to
build their own detector inputs: one background pass over an unmasked frame, with
an empty exclusion. That is not a code path production runs, and it changed the
answer in both directions -- it manufactured false positives out of chip-fixed
columns, and it flipped ``prescreen_confirmed`` so satdet and the Radon rescue
were skipped. These tests pin the replacement to production rather than to a
reimplementation of it.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "benchmarks") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmarks"))

from production_inputs import capture_detector_inputs, streak_config  # noqa: E402

from weightmask.process import process_image  # noqa: E402

SHAPE = (120, 120)


def _frame(seed=1):
    """A quiet frame with a bright bar, so the upstream stages produce a real mask."""
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 1.0, SHAPE).astype(np.float32)
    data[58:62, 20:100] = 40.0
    return data


def _config(**streak_overrides):
    cfg = yaml.safe_load(open(ROOT / "weightmask.yml"))
    cfg["sep_objects"]["extract_thresh"] = 2.0
    cfg["sep_objects"]["min_area"] = 5
    cfg["sep_objects"]["max_elongation"] = 2.0
    cfg["sep_objects"]["spike_enable"] = False
    cfg["streak_masking"] = dict(cfg["streak_masking"])
    cfg["streak_masking"].update(streak_overrides)
    return cfg


def _run_pipeline(data, cfg):
    """Run ``process_image`` and return what it handed to ``detect_streaks``."""
    seen = {}

    def record(data_sub, bkg_rms_map, existing_mask, streak_cfg):
        seen["data_sub"] = np.array(data_sub, copy=True)
        seen["rms"] = None if bkg_rms_map is None else np.array(bkg_rms_map, copy=True)
        seen["existing"] = np.array(existing_mask, copy=True)
        return np.zeros(SHAPE, dtype=bool)

    with (
        patch("weightmask.process.detect_cosmic_rays", return_value=np.zeros(SHAPE, dtype=bool)),
        patch("weightmask.process.detect_streaks", side_effect=record),
        patch("weightmask.streaks.detect_streaks", side_effect=record),
    ):
        process_image(data, {"GAIN": 1.5, "RDNOISE": 5.0}, np.ones(SHAPE, dtype=np.float32), cfg, tile_size=32)
    return seen


def _enabled_config(**overrides):
    """A config with ``streak_masking`` forced on, ready for the capture."""
    cfg = _config(**overrides)
    streak_config(cfg)
    return cfg


class TestCaptureMatchesProduction(unittest.TestCase):
    def test_capture_reproduces_the_pipelines_own_detector_inputs(self):
        data = _frame()
        production = _run_pipeline(data, _config())
        self.assertIn("data_sub", production, "pipeline never reached the streak stage")

        data_sub, rms, existing = capture_detector_inputs(
            data,
            {"GAIN": 1.5, "RDNOISE": 5.0},
            _enabled_config(),
            flat=np.ones(SHAPE, dtype=np.float32),
        )
        np.testing.assert_array_equal(data_sub, production["data_sub"])
        np.testing.assert_array_equal(rms, production["rms"])
        np.testing.assert_array_equal(existing, production["existing"])

    def test_captured_exclusion_is_a_real_mask_not_an_empty_one(self):
        """An all-False exclusion is the defect this module exists to prevent.

        With one, the binned prescreen accepts chip-fixed clutter, sets
        ``prescreen_confirmed`` and suppresses satdet and the Radon rescue.
        """
        _data_sub, _rms, existing = capture_detector_inputs(
            _frame(),
            {"GAIN": 1.5, "RDNOISE": 5.0},
            _enabled_config(),
            flat=np.ones(SHAPE, dtype=np.float32),
        )
        self.assertGreater(int(np.count_nonzero(existing)), 0)
        self.assertEqual(existing.dtype, np.bool_)
        self.assertEqual(existing.shape, SHAPE)

    def test_inputs_are_copies_not_views_into_the_pipeline(self):
        """The capture must survive the pipeline's own reuse of its buffers."""
        data = _frame()
        flat = np.ones(SHAPE, dtype=np.float32)
        header = {"GAIN": 1.5, "RDNOISE": 5.0}
        data_sub, _rms, existing = capture_detector_inputs(data, header, _enabled_config(), flat=flat)
        data_sub[:] = 0.0
        existing[:] = False
        again = capture_detector_inputs(data, header, _enabled_config(), flat=flat)
        self.assertGreater(int(np.count_nonzero(again[2])), 0)
        self.assertFalse(np.all(again[0] == 0.0))


class TestCaptureFailsLoudly(unittest.TestCase):
    def test_disabled_streak_stage_raises_instead_of_returning_wrong_inputs(self):
        """Silently returning the non-production inputs is the failure mode.

        A harness that could not reach the streak stage must stop, not fall back
        to a single background pass over an unmasked frame.
        """
        cfg = _config(enable=False)
        with self.assertRaises(RuntimeError):
            capture_detector_inputs(
                _frame(),
                {"GAIN": 1.5, "RDNOISE": 5.0},
                cfg,
                flat=np.ones(SHAPE, dtype=np.float32),
            )

    def test_the_pipeline_call_is_restored_after_a_failed_capture(self):
        import weightmask.process as proc

        original = proc.detect_streaks
        with self.assertRaises(RuntimeError):
            capture_detector_inputs(
                _frame(),
                {"GAIN": 1.5, "RDNOISE": 5.0},
                _config(enable=False),
                flat=np.ones(SHAPE, dtype=np.float32),
            )
        self.assertIs(proc.detect_streaks, original)


if __name__ == "__main__":
    unittest.main()
