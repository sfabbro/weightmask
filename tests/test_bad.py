import numpy as np
import unittest
from weightmask.bad import detect_bad_pixels


class TestBadPixels(unittest.TestCase):
    def test_detect_bad_pixels_with_flat_data(self):
        """Test bad pixel detection with realistic flat field data."""
        # Create test flat field data
        flat_data = np.ones((100, 100), dtype=np.float32)

        # Add some bad pixels
        flat_data[10, 10] = 0.3  # Below threshold
        flat_data[20, 20] = 2.5  # Above threshold
        flat_data[30, 30] = np.nan  # NaN value
        flat_data[40, 40] = np.inf  # Inf value

        config = {"low_thresh": 0.5, "high_thresh": 2.0, "col_enable": False}

        mask = detect_bad_pixels(flat_data, config, using_unit_flat=False)

        # Check that bad pixels are correctly identified
        self.assertTrue(mask[10, 10])  # Low value
        self.assertTrue(mask[20, 20])  # High value
        self.assertTrue(mask[30, 30])  # NaN
        self.assertTrue(mask[40, 40])  # Inf
        self.assertFalse(mask[50, 50])  # Normal pixel

    def test_detect_bad_pixels_with_unit_flat(self):
        """Test bad pixel detection with unit flat (should skip pixel thresholding)."""
        flat_data = np.ones((100, 100), dtype=np.float32)
        flat_data[10, 10] = 0.3  # This should be ignored with unit flat

        config = {"low_thresh": 0.5, "high_thresh": 2.0, "col_enable": False}

        mask = detect_bad_pixels(flat_data, config, using_unit_flat=True)

        # With unit flat, no pixel thresholding should occur
        self.assertEqual(np.sum(mask), 0)

    def test_detect_bad_pixels_column_detection(self):
        """Test bad column detection."""
        flat_data = np.ones((100, 100), dtype=np.float32)

        # Create a bad column (low variance)
        flat_data[:, 50] = 0.1

        config = {
            "low_thresh": 0.5,
            "high_thresh": 2.0,
            "col_enable": True,
            "col_low_var_factor": 0.05,
            "col_median_dev_factor": 0.1,
        }

        mask = detect_bad_pixels(flat_data, config, using_unit_flat=False)

        # Check that the bad column is identified
        self.assertTrue(np.all(mask[:, 50]))

    def test_detect_bad_pixels_no_finite_data(self):
        """Test bad pixel detection with no finite data."""
        flat_data = np.full((100, 100), np.nan, dtype=np.float32)

        config = {"low_thresh": 0.5, "high_thresh": 2.0, "col_enable": False}

        mask = detect_bad_pixels(flat_data, config, using_unit_flat=False)

        # Should return empty mask
        self.assertEqual(np.sum(mask), 0)

    def test_detect_bad_pixels_empty_config(self):
        """Test bad pixel detection with empty configuration."""
        flat_data = np.ones((100, 100), dtype=np.float32)
        flat_data[10, 10] = 0.3

        config = {}

        mask = detect_bad_pixels(flat_data, config, using_unit_flat=False)

        # Should use default thresholds
        self.assertTrue(mask[10, 10])


if __name__ == "__main__":
    unittest.main()
