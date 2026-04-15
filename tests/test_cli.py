import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import fitsio
import numpy as np

from weightmask.cli import run_pipeline, validate_config, validate_fits_file


class TestValidateFitsFile(unittest.TestCase):
    @patch("weightmask.cli.fitsio.FITS")
    def test_validate_fits_file_valid(self, mock_fits):
        mock_fits_instance = MagicMock()
        mock_fits_instance.__len__.return_value = 2
        mock_fits.return_value.__enter__.return_value = mock_fits_instance

        result = validate_fits_file("dummy.fits")
        self.assertTrue(result)
        mock_fits.assert_called_once_with("dummy.fits", "r")

    @patch("weightmask.cli.fitsio.FITS")
    def test_validate_fits_file_empty(self, mock_fits):
        mock_fits_instance = MagicMock()
        mock_fits_instance.__len__.return_value = 0
        mock_fits.return_value.__enter__.return_value = mock_fits_instance

        result = validate_fits_file("empty.fits")
        self.assertFalse(result)
        mock_fits.assert_called_once_with("empty.fits", "r")

    @patch("weightmask.cli.fitsio.FITS")
    def test_validate_fits_file_oserror(self, mock_fits):
        mock_fits.side_effect = OSError("Invalid file format")

        result = validate_fits_file("invalid.fits")
        self.assertFalse(result)
        mock_fits.assert_called_once_with("invalid.fits", "r")


class TestValidateConfig(unittest.TestCase):
    def test_validate_config_valid(self):
        """Test with a fully valid configuration."""
        valid_config = {
            "flat_masking": {},
            "saturation": {},
            "sep_background": {},
            "cosmic_ray": {},
            "sep_objects": {},
            "streak_masking": {},
            "variance": {"method": "theoretical"},
            "confidence_params": {},
            "output_params": {},
        }
        self.assertTrue(validate_config(valid_config))

    def test_validate_config_missing_sections(self):
        """Test with missing sections. It should print warnings but return True."""
        empty_config = {}
        self.assertTrue(validate_config(empty_config))

    def test_validate_config_invalid_variance_not_dict(self):
        """Test with an invalid variance section (not a dictionary)."""
        invalid_config = {"variance": "not_a_dict"}
        self.assertFalse(validate_config(invalid_config))

    def test_validate_config_invalid_variance_method(self):
        """Test with an invalid variance method."""
        invalid_config = {"variance": {"method": "invalid_method"}}
        self.assertFalse(validate_config(invalid_config))

    def test_validate_config_valid_variance_methods(self):
        """Test with all valid variance methods."""
        for method in ["theoretical", "rms_map", "empirical_fit"]:
            valid_config = {"variance": {"method": method}}
            self.assertTrue(validate_config(valid_config))


class TestRunPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.workspace = self.test_dir.name

        self.input_file = os.path.join(self.workspace, "test_input.fits")
        data = np.random.normal(100, 10, (100, 100)).astype(np.float32)
        fitsio.write(self.input_file, data, clobber=True)

        self.config_file = os.path.join(self.workspace, "test_config.yml")
        with open(self.config_file, "w") as f:
            f.write(
                """
flat_masking: {}
saturation: {}
sep_background: {}
cosmic_ray: {}
sep_objects: {}
streak_masking: {}
variance:
  method: theoretical
confidence_params: {}
output_params:
  output_map_format: weight
"""
            )

    def tearDown(self):
        self.test_dir.cleanup()

    @patch("sys.argv", ["weightmask"])
    def test_run_pipeline_missing_args(self):
        with self.assertRaises(SystemExit) as cm:
            run_pipeline()
        self.assertEqual(cm.exception.code, 2)

    def test_run_pipeline_missing_input_file(self):
        with patch(
            "sys.argv",
            ["weightmask", "nonexistent.fits", "--config", self.config_file],
        ):
            result = run_pipeline()
            self.assertEqual(result, 1)

    def test_run_pipeline_success(self):
        output_file = os.path.join(self.workspace, "output.weight.fits")
        with patch(
            "sys.argv",
            [
                "weightmask",
                self.input_file,
                "--config",
                self.config_file,
                "-o",
                output_file,
            ],
        ):
            result = run_pipeline()
            self.assertEqual(result, 0)
            self.assertTrue(os.path.exists(output_file))

    def test_run_pipeline_individual_masks(self):
        output_file = os.path.join(self.workspace, "output2.weight.fits")
        with patch(
            "sys.argv",
            [
                "weightmask",
                self.input_file,
                "--config",
                self.config_file,
                "-o",
                output_file,
                "--individual_masks",
            ],
        ):
            result = run_pipeline()
            self.assertEqual(result, 0)
            self.assertTrue(os.path.exists(output_file))
            self.assertTrue(
                os.path.exists(os.path.join(self.workspace, "output2.weight.bad.fits"))
            )
            self.assertTrue(
                os.path.exists(os.path.join(self.workspace, "output2.weight.sat.fits"))
            )

    def test_run_pipeline_missing_config(self):
        with patch(
            "sys.argv",
            ["weightmask", self.input_file, "--config", "nonexistent_config.yml"],
        ):
            result = run_pipeline()
            self.assertEqual(result, 1)

    def test_run_pipeline_bad_config(self):
        bad_config_file = os.path.join(self.workspace, "bad_config.yml")
        with open(bad_config_file, "w") as f:
            f.write("this is not a valid yaml file: [")
        with patch(
            "sys.argv",
            ["weightmask", self.input_file, "--config", bad_config_file],
        ):
            result = run_pipeline()
            self.assertEqual(result, 1)

    def test_run_pipeline_missing_flat(self):
        with patch(
            "sys.argv",
            [
                "weightmask",
                self.input_file,
                "--config",
                self.config_file,
                "--flat_image",
                "nonexistent_flat.fits",
            ],
        ):
            result = run_pipeline()
            self.assertEqual(result, 1)


class TestCLIConfigFallback(unittest.TestCase):
    @patch("weightmask.cli.validate_fits_file")
    @patch("os.path.exists")
    @patch.object(sys, "argv", ["weightmask", "dummy.fits"])
    def test_missing_default_config(self, mock_exists, mock_validate_fits):
        def exists_side_effect(path):
            if path == "dummy.fits":
                return True
            return False

        mock_exists.side_effect = exists_side_effect
        mock_validate_fits.return_value = True

        with patch("builtins.print") as mock_print:
            result = run_pipeline()
            self.assertEqual(result, 1)
            mock_print.assert_any_call(
                "ERROR: Config file not specified and no default found."
            )

    @patch("weightmask.cli.validate_fits_file")
    @patch("os.path.exists")
    @patch("builtins.open")
    @patch.object(sys, "argv", ["weightmask", "dummy.fits"])
    def test_first_fallback_config_found(self, mock_open, mock_exists, mock_validate_fits):
        def exists_side_effect(path):
            if path == "dummy.fits":
                return True
            if path == "weightmask.yml":
                return True
            return False

        mock_exists.side_effect = exists_side_effect
        mock_validate_fits.return_value = True
        mock_open.side_effect = OSError("Mocked error to stop pipeline")

        with patch("builtins.print") as mock_print:
            result = run_pipeline()
            self.assertEqual(result, 1)
            mock_print.assert_any_call(
                "Using default config file found at: weightmask.yml"
            )

    @patch("weightmask.cli.validate_fits_file")
    @patch("os.path.exists")
    @patch("builtins.open")
    @patch.object(sys, "argv", ["weightmask", "dummy.fits"])
    def test_second_fallback_config_found(self, mock_open, mock_exists, mock_validate_fits):
        def exists_side_effect(path):
            if path == "dummy.fits":
                return True
            if path == "config.yml":
                return True
            return False

        mock_exists.side_effect = exists_side_effect
        mock_validate_fits.return_value = True
        mock_open.side_effect = OSError("Mocked error to stop pipeline")

        with patch("builtins.print") as mock_print:
            result = run_pipeline()
            self.assertEqual(result, 1)
            mock_print.assert_any_call("Using default config file found at: config.yml")

    @patch("weightmask.cli.validate_fits_file")
    @patch("os.path.exists")
    @patch("builtins.open")
    @patch.object(sys, "argv", ["weightmask", "dummy.fits"])
    def test_third_fallback_config_found(self, mock_open, mock_exists, mock_validate_fits):
        def exists_side_effect(path):
            if path == "dummy.fits":
                return True
            if path == ".weightmask.yml":
                return True
            return False

        mock_exists.side_effect = exists_side_effect
        mock_validate_fits.return_value = True
        mock_open.side_effect = OSError("Mocked error to stop pipeline")

        with patch("builtins.print") as mock_print:
            result = run_pipeline()
            self.assertEqual(result, 1)
            mock_print.assert_any_call(
                "Using default config file found at: .weightmask.yml"
            )


if __name__ == "__main__":
    unittest.main()
