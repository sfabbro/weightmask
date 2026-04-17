import numpy as np
import unittest
from weightmask.streaks import detect_streaks


class TestStreaks(unittest.TestCase):
    def test_detect_streaks_ransac_method(self):
        """Test streak detection using RANSAC method."""
        # Create test background-subtracted data
        data_sub = np.random.poisson(10, (100, 100)).astype(np.float32)

        # Add a streak-like feature
        data_sub[50, 30:70] = 100.0

        # Create background RMS map
        bkg_rms_map = np.full((100, 100), 5.0, dtype=np.float32)

        # Create an existing mask (no masked pixels)
        existing_mask = np.zeros((100, 100), dtype=bool)

        config = {
            "enable": True,
            "method": "ransac",
            "dilation_radius": 2,
            "ransac_params": {
                "input_threshold_sigma": 3.0,
                "min_elongation": 5.0,
                "min_pixels": 5,
                "max_pixels": 1000,
                "ransac_min_samples": 2,
                "ransac_residual_threshold": 1.0,
                "ransac_max_trials": 10,
                "min_inliers": 5,
            },
        }

        mask = detect_streaks(data_sub, bkg_rms_map, existing_mask, config)

        # Check that we got a result
        self.assertIsNotNone(mask)

        # Check that mask is boolean
        self.assertEqual(mask.dtype, bool)

    def test_detect_streaks_hough_method(self):
        """Test streak detection using Hough method."""
        # Create test background-subtracted data
        data_sub = np.random.poisson(10, (100, 100)).astype(np.float32)

        # Add a streak-like feature
        data_sub[50, 30:70] = 100.0

        # Create background RMS map
        bkg_rms_map = np.full((100, 100), 5.0, dtype=np.float32)

        # Create an existing mask (no masked pixels)
        existing_mask = np.zeros((100, 100), dtype=bool)

        config = {
            "enable": True,
            "method": "hough",
            "dilation_radius": 2,
            "hough_params": {
                "input_threshold_sigma": 3.0,
                "prob_hough_threshold": 5,
                "prob_hough_line_length": 10,
                "prob_hough_line_gap": 5,
            },
        }

        mask = detect_streaks(data_sub, bkg_rms_map, existing_mask, config)

        # Check that we got a result
        self.assertIsNotNone(mask)

        # Check that mask is boolean
        self.assertEqual(mask.dtype, bool)

    def test_detect_streaks_disabled(self):
        """Test streak detection when disabled."""
        # Create test background-subtracted data
        data_sub = np.random.poisson(10, (100, 100)).astype(np.float32)

        # Create background RMS map
        bkg_rms_map = np.full((100, 100), 5.0, dtype=np.float32)

        # Create an existing mask (no masked pixels)
        existing_mask = np.zeros((100, 100), dtype=bool)

        config = {"enable": False, "method": "ransac"}

        mask = detect_streaks(data_sub, bkg_rms_map, existing_mask, config)

        # Should return empty mask when disabled
        self.assertEqual(np.sum(mask), 0)

    def test_detect_streaks_with_existing_mask(self):
        """Test streak detection with existing masked pixels."""
        # Create test background-subtracted data
        data_sub = np.random.poisson(10, (100, 100)).astype(np.float32)

        # Add a streak-like feature
        data_sub[50, 30:70] = 100.0

        # Create background RMS map
        bkg_rms_map = np.full((100, 100), 5.0, dtype=np.float32)

        # Create an existing mask with some pixels masked
        existing_mask = np.zeros((100, 100), dtype=bool)
        existing_mask[10, 10] = True

        config = {
            "enable": True,
            "method": "ransac",
            "dilation_radius": 2,
            "ransac_params": {
                "input_threshold_sigma": 3.0,
                "min_elongation": 5.0,
                "min_pixels": 5,
                "max_pixels": 1000,
                "ransac_min_samples": 2,
                "ransac_residual_threshold": 1.0,
                "ransac_max_trials": 10,
                "min_inliers": 5,
            },
        }

        mask = detect_streaks(data_sub, bkg_rms_map, existing_mask, config)

        # Check that existing masked pixels are not in the result
        self.assertFalse(mask[10, 10])


if __name__ == "__main__":
    unittest.main()
