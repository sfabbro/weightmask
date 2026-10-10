import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import yaml

from weightmask.cosmics import detect_cosmic_rays
from weightmask.errors import StageFailure

ASTROSCRAPPY_AVAILABLE = find_spec("astroscrappy") is not None


class TestCosmics(unittest.TestCase):
    def test_shipped_config_uses_only_fixed_dimensionless_sigclip(self):
        config = yaml.safe_load((Path(__file__).resolve().parents[1] / "weightmask.yml").read_text())

        self.assertNotIn("dynamic_sigclip", config["cosmic_ray"])

    def test_retired_adu_rms_sigclip_config_is_rejected(self):
        from weightmask.process import validate_config

        self.assertFalse(validate_config({"cosmic_ray": {"dynamic_sigclip": False}}))

    def test_sigclip_does_not_change_when_adu_scale_changes(self):
        normalized = np.zeros((16, 16), dtype=np.float32)
        normalized[8, 8] = 9.0
        empty = np.zeros_like(normalized, dtype=bool)
        config = {"sigclip": 8.5, "niter": 1, "dilate_cr": False, "psf_aware": False}

        with patch("weightmask.cosmics.detect_cosmics", return_value=(empty, normalized)) as detector:
            for rms_adu in (1.0, 100.0):
                detect_cosmic_rays(
                    normalized * rms_adu,
                    empty,
                    65000.0 * rms_adu,
                    1.0,
                    0.0,
                    config,
                    bkg_rms_map=np.full_like(normalized, rms_adu),
                )

        self.assertEqual([call.kwargs["sigclip"] for call in detector.call_args_list], [8.5, 8.5])

    def test_get_psf_peakiness(self):
        """Test the calculation of PSF peakiness."""
        from weightmask.cosmics import _get_psf_peakiness

        # Test a standard FWHM value (e.g., FWHM=3.0)
        peakiness = _get_psf_peakiness(3.0)
        self.assertAlmostEqual(peakiness, 0.1639546, places=5)

        # Test a very small FWHM (approaches 1.0)
        peakiness_small = _get_psf_peakiness(0.1)
        self.assertAlmostEqual(peakiness_small, 1.0, places=5)

        # Test a very large FWHM (approaches 1/9 for 3x3 kernel)
        peakiness_large = _get_psf_peakiness(1000.0)
        self.assertAlmostEqual(peakiness_large, 1.0 / 9.0, places=5)

    def test_detect_cosmic_rays(self):
        """Test cosmic ray detection."""
        # Create test science data
        rng = np.random.default_rng(0)
        sci_data = rng.poisson(100, (100, 100)).astype(np.float32)

        # Add a cosmic ray-like feature
        sci_data[50, 50] = 1000.0
        sci_data[50, 51] = 800.0
        sci_data[51, 50] = 700.0

        # Create an existing mask (no masked pixels)
        existing_mask = np.zeros((100, 100), dtype=bool)

        # Parameters
        saturation_level = 65000.0
        gain = 1.5
        read_noise = 5.0

        config = {"sigclip": 4.5, "objlim": 5.0}

        if not ASTROSCRAPPY_AVAILABLE:
            with self.assertRaises(StageFailure):
                detect_cosmic_rays(sci_data, existing_mask, saturation_level, gain, read_noise, config)
            return

        # If astroscrappy is available, run the full test
        mask = detect_cosmic_rays(sci_data, existing_mask, saturation_level, gain, read_noise, config)

        # Check that we got a result
        self.assertIsNotNone(mask)

        # Check that mask is boolean
        self.assertEqual(mask.dtype, bool)
        self.assertTrue(mask[50, 50])

    def test_detect_cosmic_rays_with_existing_mask(self):
        """Test cosmic ray detection with existing masked pixels."""
        # Create test science data
        rng = np.random.default_rng(1)
        sci_data = rng.poisson(100, (100, 100)).astype(np.float32)

        # Create an existing mask with some pixels masked
        existing_mask = np.zeros((100, 100), dtype=bool)
        existing_mask[10, 10] = True

        # Parameters
        saturation_level = 65000.0
        gain = 1.5
        read_noise = 5.0

        config = {"sigclip": 4.5, "objlim": 5.0}

        if not ASTROSCRAPPY_AVAILABLE:
            with self.assertRaises(StageFailure):
                detect_cosmic_rays(sci_data, existing_mask, saturation_level, gain, read_noise, config)
            return

        # If astroscrappy is available, run the full test
        mask = detect_cosmic_rays(sci_data, existing_mask, saturation_level, gain, read_noise, config)

        # Check that existing masked pixels are not in the result
        self.assertFalse(mask[10, 10])

    def test_detect_cosmic_rays_astroscrappy_failure(self):
        rng = np.random.default_rng(2)
        sci_data = rng.poisson(100, (100, 100)).astype(np.float32)
        existing_mask = np.zeros((100, 100), dtype=bool)
        saturation_level = 65000.0
        gain = 1.5
        read_noise = 5.0
        config = {"sigclip": 4.5, "objlim": 5.0}

        with patch("weightmask.cosmics.detect_cosmics", side_effect=OSError("injected failure")) as detector:
            with self.assertRaisesRegex(StageFailure, "cosmic_ray.primary") as raised:
                detect_cosmic_rays(sci_data, existing_mask, saturation_level, gain, read_noise, config)
        detector.assert_called_once()
        self.assertEqual(raised.exception.stage, "cosmic_ray.primary")
        self.assertIsInstance(raised.exception.__cause__, OSError)

    def test_missing_astroscrappy_fails_when_primary_pass_is_required(self):
        data = np.ones((16, 16), dtype=np.float32)

        with patch("weightmask.cosmics.detect_cosmics", None):
            with self.assertRaisesRegex(StageFailure, "cosmic_ray.primary") as raised:
                detect_cosmic_rays(data, np.zeros_like(data, bool), 65000.0, 1.0, 0.0, {"niter": 1})

        self.assertEqual(raised.exception.stage, "cosmic_ray.primary")

    def test_zero_primary_and_faint_iterations_are_successfully_disabled(self):
        data = np.ones((16, 16), dtype=np.float32)
        config = {"niter": 0, "faint_cr": {"enable": True, "niter": 0}}
        detector = MagicMock()

        with patch("weightmask.cosmics.detect_cosmics", detector):
            mask = detect_cosmic_rays(data, np.zeros_like(data, bool), 65000.0, 1.0, 0.0, config)

        self.assertFalse(mask.any())
        detector.assert_not_called()

    def test_genuine_zero_cosmic_detections_are_successful(self):
        data = np.ones((16, 16), dtype=np.float32)
        empty = np.zeros_like(data, dtype=bool)

        with patch("weightmask.cosmics.detect_cosmics", return_value=(empty, data)):
            mask = detect_cosmic_rays(
                data,
                empty,
                65000.0,
                1.0,
                0.0,
                {"niter": 1, "dilate_cr": False, "psf_aware": False},
            )

        self.assertFalse(mask.any())

    def test_enabled_faint_cosmic_failure_is_not_an_empty_detection(self):
        data = np.ones((16, 16), dtype=np.float32)
        empty = np.zeros_like(data, dtype=bool)
        config = {
            "niter": 1,
            "dilate_cr": False,
            "psf_aware": False,
            "faint_cr": {"enable": True, "enhancement": "lacosmic", "niter": 1},
        }

        with patch("weightmask.cosmics.detect_cosmics", side_effect=[(empty, data), OSError("faint failure")]):
            with self.assertRaisesRegex(StageFailure, "cosmic_ray.faint") as raised:
                detect_cosmic_rays(data, empty, 65000.0, 1.0, 0.0, config)

        self.assertEqual(raised.exception.stage, "cosmic_ray.faint")
        self.assertIsInstance(raised.exception.__cause__, OSError)


if __name__ == "__main__":
    unittest.main()
