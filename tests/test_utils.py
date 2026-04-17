import numpy as np
import unittest
from weightmask.utils import extract_hdu_spec, create_binary_mask


class TestUtils(unittest.TestCase):
    def test_extract_hdu_spec_with_hdu(self):
        """Test extracting HDU specifier from path with HDU."""
        path = "file.fits[1]"
        clean_path, hdu_index = extract_hdu_spec(path)

        self.assertEqual(clean_path, "file.fits")
        self.assertEqual(hdu_index, 1)

    def test_extract_hdu_spec_without_hdu(self):
        """Test extracting HDU specifier from path without HDU."""
        path = "file.fits"
        clean_path, hdu_index = extract_hdu_spec(path)

        self.assertEqual(clean_path, "file.fits")
        self.assertIsNone(hdu_index)

    def test_extract_hdu_spec_none_input(self):
        """Test extracting HDU specifier with None input."""
        clean_path, hdu_index = extract_hdu_spec(None)

        self.assertIsNone(clean_path)
        self.assertIsNone(hdu_index)

    def test_create_binary_mask(self):
        """Test creating binary mask from bitmask."""
        # Create a test bitmask
        mask_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.uint32)

        # Test with bit flag 1 (first bit)
        binary_mask = create_binary_mask(mask_data, 1)
        expected = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
        np.testing.assert_array_equal(binary_mask, expected)

        # Test with bit flag 2 (second bit)
        binary_mask = create_binary_mask(mask_data, 2)
        expected = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
        np.testing.assert_array_equal(binary_mask, expected)

        # Test with bit flag 4 (third bit)
        binary_mask = create_binary_mask(mask_data, 4)
        expected = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0], dtype=np.uint8)
        np.testing.assert_array_equal(binary_mask, expected)


if __name__ == "__main__":
    unittest.main()
