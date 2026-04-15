import unittest
import os
from unittest.mock import patch
import sys
from weightmask.cli import run_pipeline

class TestCLIConfigFallback(unittest.TestCase):

    @patch('weightmask.cli.validate_fits_file')
    @patch('os.path.exists')
    @patch.object(sys, 'argv', ['weightmask', 'dummy.fits'])
    def test_missing_default_config(self, mock_exists, mock_validate_fits):
        def exists_side_effect(path):
            if path == 'dummy.fits':
                return True
            return False
        mock_exists.side_effect = exists_side_effect
        mock_validate_fits.return_value = True

        with patch('builtins.print') as mock_print:
            result = run_pipeline()
            self.assertEqual(result, 1)
            mock_print.assert_any_call("ERROR: Config file not specified and no default found.")

    @patch('weightmask.cli.validate_fits_file')
    @patch('os.path.exists')
    @patch('builtins.open')
    @patch.object(sys, 'argv', ['weightmask', 'dummy.fits'])
    def test_first_fallback_config_found(self, mock_open, mock_exists, mock_validate_fits):
        def exists_side_effect(path):
            if path == 'dummy.fits':
                return True
            if path == 'weightmask.yml':
                return True
            return False
        mock_exists.side_effect = exists_side_effect
        mock_validate_fits.return_value = True
        mock_open.side_effect = OSError("Mocked error to stop pipeline")

        with patch('builtins.print') as mock_print:
            result = run_pipeline()
            self.assertEqual(result, 1)
            mock_print.assert_any_call("Using default config file found at: weightmask.yml")

    @patch('weightmask.cli.validate_fits_file')
    @patch('os.path.exists')
    @patch('builtins.open')
    @patch.object(sys, 'argv', ['weightmask', 'dummy.fits'])
    def test_second_fallback_config_found(self, mock_open, mock_exists, mock_validate_fits):
        def exists_side_effect(path):
            if path == 'dummy.fits':
                return True
            if path == 'config.yml':
                return True
            return False
        mock_exists.side_effect = exists_side_effect
        mock_validate_fits.return_value = True
        mock_open.side_effect = OSError("Mocked error to stop pipeline")

        with patch('builtins.print') as mock_print:
            result = run_pipeline()
            self.assertEqual(result, 1)
            mock_print.assert_any_call("Using default config file found at: config.yml")

    @patch('weightmask.cli.validate_fits_file')
    @patch('os.path.exists')
    @patch('builtins.open')
    @patch.object(sys, 'argv', ['weightmask', 'dummy.fits'])
    def test_third_fallback_config_found(self, mock_open, mock_exists, mock_validate_fits):
        def exists_side_effect(path):
            if path == 'dummy.fits':
                return True
            if path == '.weightmask.yml':
                return True
            return False
        mock_exists.side_effect = exists_side_effect
        mock_validate_fits.return_value = True
        mock_open.side_effect = OSError("Mocked error to stop pipeline")

        with patch('builtins.print') as mock_print:
            result = run_pipeline()
            self.assertEqual(result, 1)
            mock_print.assert_any_call("Using default config file found at: .weightmask.yml")

if __name__ == '__main__':
    unittest.main()
