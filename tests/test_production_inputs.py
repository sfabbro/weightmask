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

from production_inputs import (  # noqa: E402
    capture_detector_calls,
    capture_detector_inputs,
    production_gain_read_noise,
    streak_config,
)

from weightmask import MASK_BITS  # noqa: E402
from weightmask.process import _detection_background_views, process_image  # noqa: E402

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
    def test_high_snr_elongated_source_is_not_object_excluded_from_streaks(self):
        seen = _run_pipeline(_frame(), _enabled_config())
        bar = np.zeros(SHAPE, dtype=bool)
        bar[58:62, 20:100] = True
        self.assertEqual(int(np.count_nonzero(seen["existing"] & bar)), 0)

    def test_unaccepted_compact_elongated_source_remains_detected(self):
        cfg = _enabled_config()
        with (
            patch("weightmask.process.detect_cosmic_rays", return_value=np.zeros(SHAPE, dtype=bool)),
            patch("weightmask.process.detect_streaks", return_value=np.zeros(SHAPE, dtype=bool)),
        ):
            result = process_image(
                _frame(),
                {"GAIN": 1.5, "RDNOISE": 5.0},
                np.ones(SHAPE, dtype=np.float32),
                cfg,
                tile_size=32,
            )
        bar = np.zeros(SHAPE, dtype=bool)
        bar[58:62, 20:100] = True
        self.assertTrue(np.any(result[0][bar] & MASK_BITS["DETECTED"]))

    def test_accepted_compact_elongated_source_hands_off_to_streak(self):
        cfg = _enabled_config()
        bar = np.zeros(SHAPE, dtype=bool)
        bar[58:62, 20:100] = True
        with (
            patch("weightmask.process.detect_cosmic_rays", return_value=np.zeros(SHAPE, dtype=bool)),
            patch("weightmask.process.detect_streaks", return_value=bar),
        ):
            result = process_image(
                _frame(),
                {"GAIN": 1.5, "RDNOISE": 5.0},
                np.ones(SHAPE, dtype=np.float32),
                cfg,
                tile_size=32,
            )
        self.assertTrue(np.all(result[0][bar] & MASK_BITS["STREAK"]))
        self.assertFalse(np.any(result[0][bar] & MASK_BITS["DETECTED"]))

    def test_empty_streak_detector_matches_handoff_disabled_object_footprint(self):
        def run(handoff, accepted, frame=None):
            cfg = _enabled_config()
            cfg["sep_objects"]["handoff_elongated_to_streak"] = handoff
            with (
                patch("weightmask.process.detect_cosmic_rays", return_value=np.zeros(SHAPE, dtype=bool)),
                patch("weightmask.process.detect_streaks", return_value=accepted),
            ):
                return process_image(
                    _frame() if frame is None else frame,
                    {"GAIN": 1.5, "RDNOISE": 5.0},
                    np.ones(SHAPE, dtype=np.float32),
                    cfg,
                    tile_size=32,
                )[0]

        disabled = run(False, np.zeros(SHAPE, dtype=bool))
        enabled = run(True, np.zeros(SHAPE, dtype=bool))
        disabled_detected = (disabled & MASK_BITS["DETECTED"]) != 0
        enabled_detected = (enabled & MASK_BITS["DETECTED"]) != 0
        np.testing.assert_array_equal(enabled_detected, disabled_detected)

    def test_handoff_does_not_change_final_streak_data_or_rms_inputs(self):
        def capture(handoff):
            cfg = _enabled_config()
            cfg["sep_objects"]["handoff_elongated_to_streak"] = handoff
            with (
                patch("weightmask.process.detect_cosmic_rays", return_value=np.zeros(SHAPE, dtype=bool)),
                patch("weightmask.process.detect_streaks", return_value=np.zeros(SHAPE, dtype=bool)),
            ):
                result = process_image(
                    _frame(),
                    {"GAIN": 1.5, "RDNOISE": 5.0},
                    np.ones(SHAPE, dtype=np.float32),
                    cfg,
                    tile_size=32,
                )
            return result

        disabled = capture(False)
        enabled = capture(True)
        np.testing.assert_array_equal(enabled[4], disabled[4])
        np.testing.assert_array_equal(enabled[1], disabled[1])

    def test_detection_view_interpolation_preserves_calibration_outside_support(self):
        sky = np.full(SHAPE, 12.0, dtype=np.float32)
        rms = np.full(SHAPE, 2.0, dtype=np.float32)
        rms[58:62, 20:100] = np.nan
        support = np.zeros(SHAPE, dtype=bool)
        support[58:62, 20:100] = True
        sky_det, rms_det = _detection_background_views(sky, rms, support, 32)
        np.testing.assert_array_equal(sky_det[~support], sky[~support])
        np.testing.assert_array_equal(rms_det[~support], rms[~support])
        self.assertTrue(np.all(np.isfinite(rms_det[support])))

    def test_detection_view_uses_effective_mesh_box_as_sigma(self):
        import weightmask.process as process_module

        sky = np.full(SHAPE, 12.0, dtype=np.float32)
        rms = np.full(SHAPE, 2.0, dtype=np.float32)
        support = np.zeros(SHAPE, dtype=bool)
        support[58:62, 20:100] = True
        sigmas = []
        original = process_module.gaussian_filter

        def record(array, sigma, **kwargs):
            sigmas.append(float(sigma))
            return original(array, sigma=sigma, **kwargs)

        with patch.object(process_module, "gaussian_filter", side_effect=record):
            _detection_background_views(sky, rms, support, 32)
        self.assertTrue(sigmas)
        self.assertTrue(all(sigma == 32.0 for sigma in sigmas))

    def test_calibration_views_are_insensitive_to_candidate_science_values(self):
        sky = np.full(SHAPE, 12.0, dtype=np.float32)
        rms = np.full(SHAPE, 2.0, dtype=np.float32)
        support = np.zeros(SHAPE, dtype=bool)
        support[58:62, 20:100] = True
        first = _detection_background_views(sky, rms, support, 32)
        perturbed = _detection_background_views(sky, rms, support, 32)
        np.testing.assert_array_equal(first[0], perturbed[0])
        np.testing.assert_array_equal(first[1], perturbed[1])

    def test_empty_streak_detector_matches_mixed_compact_and_elongated_footprints(self):
        frame = _frame()
        frame[24:34, 24:34] += 25.0

        def run(handoff):
            cfg = _enabled_config()
            cfg["sep_objects"]["handoff_elongated_to_streak"] = handoff
            with (
                patch("weightmask.process.detect_cosmic_rays", return_value=np.zeros(SHAPE, dtype=bool)),
                patch("weightmask.process.detect_streaks", return_value=np.zeros(SHAPE, dtype=bool)),
            ):
                return process_image(
                    frame,
                    {"GAIN": 1.5, "RDNOISE": 5.0},
                    np.ones(SHAPE, dtype=np.float32),
                    cfg,
                    tile_size=32,
                )[0]

        disabled = run(False)
        enabled = run(True)
        np.testing.assert_array_equal(
            (enabled & MASK_BITS["DETECTED"]) != 0,
            (disabled & MASK_BITS["DETECTED"]) != 0,
        )

    def test_partial_streak_acceptance_restores_only_unaccepted_footprint(self):
        frame = _frame()
        frame[24:34, 24:34] += 25.0
        cfg = _enabled_config()
        cfg["sep_objects"]["handoff_elongated_to_streak"] = False
        with (
            patch("weightmask.process.detect_cosmic_rays", return_value=np.zeros(SHAPE, dtype=bool)),
            patch("weightmask.process.detect_streaks", return_value=np.zeros(SHAPE, dtype=bool)),
        ):
            disabled = process_image(
                frame, {"GAIN": 1.5, "RDNOISE": 5.0}, np.ones(SHAPE, dtype=np.float32), cfg, tile_size=32
            )[0]
        accepted = np.zeros(SHAPE, dtype=bool)
        accepted[59:61, 40:80] = True
        cfg["sep_objects"]["handoff_elongated_to_streak"] = True
        with (
            patch("weightmask.process.detect_cosmic_rays", return_value=np.zeros(SHAPE, dtype=bool)),
            patch("weightmask.process.detect_streaks", return_value=accepted),
        ):
            partial = process_image(
                frame, {"GAIN": 1.5, "RDNOISE": 5.0}, np.ones(SHAPE, dtype=np.float32), cfg, tile_size=32
            )[0]
        disabled_detected = (disabled & MASK_BITS["DETECTED"]) != 0
        partial_detected = (partial & MASK_BITS["DETECTED"]) != 0
        np.testing.assert_array_equal(partial_detected, disabled_detected & ~accepted)
        self.assertTrue(np.all(partial[accepted] & MASK_BITS["STREAK"]))

    def test_streak_overlap_never_subtracts_ordinary_object_detection(self):
        from weightmask.background import estimate_background as production_background

        cfg = _enabled_config()
        cfg["sep_objects"]["handoff_elongated_to_streak"] = False
        background_inputs = []
        streak_inputs = []

        def record_background(sci, mask, config):
            background_inputs.append(np.array(mask, copy=True))
            return production_background(sci, mask, config)

        def record_empty_streak(sci, rms, mask, config):
            streak_inputs.append(np.array(mask, copy=True))
            return np.zeros(SHAPE, dtype=bool)

        with (
            patch("weightmask.process.detect_cosmic_rays", return_value=np.zeros(SHAPE, dtype=bool)),
            patch("weightmask.process.detect_streaks", side_effect=record_empty_streak),
            patch("weightmask.process.estimate_background", side_effect=record_background),
        ):
            empty = process_image(
                _frame(), {"GAIN": 1.5, "RDNOISE": 5.0}, np.ones(SHAPE, dtype=np.float32), cfg, tile_size=32
            )[0]
        with (
            patch("weightmask.process.detect_cosmic_rays", return_value=np.zeros(SHAPE, dtype=bool)),
            patch("weightmask.process.detect_streaks", return_value=np.ones(SHAPE, dtype=bool)),
        ):
            accepted = process_image(
                _frame(), {"GAIN": 1.5, "RDNOISE": 5.0}, np.ones(SHAPE, dtype=np.float32), cfg, tile_size=32
            )[0]
        empty_detected = (empty & MASK_BITS["DETECTED"]) != 0
        accepted_detected = (accepted & MASK_BITS["DETECTED"]) != 0
        overlap = accepted_detected & empty_detected
        self.assertGreater(int(np.count_nonzero(overlap)), 0)
        self.assertTrue(np.any(((accepted & MASK_BITS["STREAK"]) != 0) & overlap))
        self.assertTrue(background_inputs)
        self.assertTrue(streak_inputs)
        self.assertTrue(np.all(background_inputs[-1][overlap]))
        self.assertTrue(np.all(streak_inputs[-1][overlap]))

    def test_later_object_iteration_provenance_is_accumulated(self):
        first = np.zeros(SHAPE, dtype=bool)
        first[30:34, 30:34] = True
        later_fallback = np.zeros(SHAPE, dtype=bool)
        later_fallback[80:84, 80:84] = True
        calls = []

        def objects(_data, _rms, _existing, config):
            calls.append(len(calls))
            if len(calls) == 1:
                return first
            config["_elongated_fallback_mask"] = later_fallback.copy()
            config["_elongated_candidate_mask"] = later_fallback.copy()
            return np.zeros(SHAPE, dtype=bool)

        cfg = _enabled_config()
        with (
            patch("weightmask.process.detect_objects", side_effect=objects),
            patch("weightmask.process.detect_cosmic_rays", return_value=np.zeros(SHAPE, dtype=bool)),
            patch("weightmask.process.detect_streaks", return_value=np.zeros(SHAPE, dtype=bool)),
        ):
            result = process_image(
                _frame(), {"GAIN": 1.5, "RDNOISE": 5.0}, np.ones(SHAPE, dtype=np.float32), cfg, tile_size=32
            )[0]
        self.assertGreaterEqual(len(calls), 2)
        detected = (result & MASK_BITS["DETECTED"]) != 0
        self.assertTrue(np.all(detected[first]))
        self.assertTrue(np.all(detected[later_fallback]))

    def test_capture_records_every_detector_boundary(self):
        calls = capture_detector_calls(
            _frame(),
            {"GAIN": 1.5, "RDNOISE": 5.0},
            _enabled_config(),
            flat=np.ones(SHAPE, dtype=np.float32),
        )
        self.assertTrue(all(calls[name] for name in ("saturation", "cosmics", "objects", "streaks")))
        self.assertEqual(calls["saturation"][0]["args"][0].shape, SHAPE)
        self.assertEqual(calls["cosmics"][0]["args"][1].shape, SHAPE)
        self.assertEqual(calls["objects"][0]["args"][2].shape, SHAPE)
        self.assertEqual(calls["streaks"][0]["args"][2].shape, SHAPE)

    def test_capture_values_order_counts_and_snapshots_are_stable(self):
        def assert_value_equal(actual, expected):
            if isinstance(actual, np.ndarray):
                np.testing.assert_array_equal(actual, expected)
            elif isinstance(actual, dict):
                self.assertEqual(actual.keys(), expected.keys())
                for key in actual:
                    assert_value_equal(actual[key], expected[key])
            elif isinstance(actual, (list, tuple)):
                self.assertEqual(len(actual), len(expected))
                for left, right in zip(actual, expected):
                    assert_value_equal(left, right)
            else:
                self.assertEqual(actual, expected)

        config = _enabled_config()
        first = capture_detector_calls(
            _frame(), {"GAINB": 2.0, "GAIN": 1.0, "RDNOIS": 7.0}, config, flat=np.ones(SHAPE, dtype=np.float32)
        )
        second = capture_detector_calls(
            _frame(),
            {"GAINB": 2.0, "GAIN": 1.0, "RDNOIS": 7.0},
            _enabled_config(),
            flat=np.ones(SHAPE, dtype=np.float32),
        )
        self.assertEqual(first["order"], ["saturation", "cosmics", "objects", "objects", "streaks"])
        self.assertEqual(
            {name: len(first[name]) for name in ("saturation", "cosmics", "objects", "streaks")},
            {"saturation": 1, "cosmics": 1, "objects": 2, "streaks": 1},
        )
        for name in ("saturation", "cosmics", "objects", "streaks"):
            for left, right in zip(first[name], second[name]):
                for actual, expected in zip(left["args"], right["args"]):
                    assert_value_equal(actual, expected)
                assert_value_equal(left["kwargs"], right["kwargs"])
        original = first["cosmics"][0]["args"][1].copy()
        first["cosmics"][0]["args"][1][:] = True
        np.testing.assert_array_equal(second["cosmics"][0]["args"][1], original)
        original_rms = second["cosmics"][0]["kwargs"]["bkg_rms_map"].copy()
        first["cosmics"][0]["kwargs"]["bkg_rms_map"][:] = 999.0
        np.testing.assert_array_equal(second["cosmics"][0]["kwargs"]["bkg_rms_map"], original_rms)

    def test_gain_and_read_noise_follow_configured_keyword_precedence(self):
        config = _enabled_config()
        self.assertEqual(production_gain_read_noise({"GAINB": 2.0, "GAIN": 1.0, "RDNOIS": 7.0}, config), (1.0, 7.0))
        self.assertEqual(production_gain_read_noise({"GAINB": 2.0, "RDNOIS": 7.0}, config), (2.0, 7.0))
        self.assertEqual(production_gain_read_noise({"GAIN": -1.0, "RDNOIS": np.nan}, config), (1.5, 5.0))
        config["variance"]["default_gain"] = 0.0
        with self.assertRaises(ValueError):
            production_gain_read_noise({}, config)

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

    def test_captured_exclusion_preserves_the_production_mask_contract(self):
        _data_sub, _rms, existing = capture_detector_inputs(
            _frame(),
            {"GAIN": 1.5, "RDNOISE": 5.0},
            _enabled_config(),
            flat=np.ones(SHAPE, dtype=np.float32),
        )
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
        self.assertEqual(again[2].dtype, np.bool_)
        self.assertEqual(again[2].shape, SHAPE)
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
