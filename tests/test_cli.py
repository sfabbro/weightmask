import unittest
from unittest.mock import patch, MagicMock
from weightmask.cli import validate_fits_file

class TestCLI(unittest.TestCase):
    @patch('weightmask.cli.fitsio.FITS')
    def test_validate_fits_file_valid(self, mock_fits):
        # Setup mock to return a non-empty list-like object
        mock_f = MagicMock()
        mock_f.__len__.return_value = 1
        mock_fits.return_value.__enter__.return_value = mock_f

        result = validate_fits_file("dummy.fits")
        self.assertTrue(result)

    @patch('weightmask.cli.fitsio.FITS')
    def test_validate_fits_file_empty(self, mock_fits):
        # Setup mock to return an empty list-like object
        mock_f = MagicMock()
        mock_f.__len__.return_value = 0
        mock_fits.return_value.__enter__.return_value = mock_f

        result = validate_fits_file("empty.fits")
        self.assertFalse(result)

    @patch('weightmask.cli.fitsio.FITS')
    def test_validate_fits_file_oserror(self, mock_fits):
        # Setup mock to raise OSError
        mock_fits.side_effect = OSError("Mocked OSError")

        result = validate_fits_file("corrupt.fits")
        self.assertFalse(result)

if __name__ == '__main__':
    unittest.main()
