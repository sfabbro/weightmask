import unittest
from unittest.mock import MagicMock, patch

from weightmask.cli import validate_fits_file


class TestCLI(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
