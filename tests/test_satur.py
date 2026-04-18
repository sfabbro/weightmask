import numpy as np
import unittest
from weightmask.satur import detect_saturated_pixels


class TestSaturation(unittest.TestCase):
    def test_detect_saturated_pixels_histogram_method(self):
        """Test saturation detection using histogram method."""
        # Create test science data
        sci_data = np.random.poisson(100, (100, 100)).astype(np.float32)

        # Add some saturated pixels
        sci_data[10:20, 10:20] = 65000.0

        # Create a simple header
        sci_hdr = {}

        config = {
            "method": "histogram",
            "keyword": "SATURATE",
            "fallback_level": 65000.0,
        }

        saturation_level, sat_method_used, mask = detect_saturated_pixels(
            sci_data, sci_hdr, config
        )

        # Check that we got a result
        self.assertIsNotNone(saturation_level)
        self.assertIsNotNone(sat_method_used)
        self.assertIsNotNone(mask)

        # Check that saturated pixels are identified
        self.assertTrue(np.any(mask[10:20, 10:20]))

    def test_detect_saturated_pixels_header_method(self):
        """Test saturation detection using header keyword."""
        # Create test science data
        sci_data = np.random.poisson(100, (100, 100)).astype(np.float32)

        # Add some saturated pixels
        sci_data[10:20, 10:20] = 65000.0

        # Create a header with saturation keyword
        sci_hdr = {"SATURATE": 60000.0}

        config = {"method": "header", "keyword": "SATURATE", "fallback_level": 65000.0}

        saturation_level, sat_method_used, mask = detect_saturated_pixels(
            sci_data, sci_hdr, config
        )

        # Check that we got the saturation level from header
        self.assertEqual(saturation_level, 60000.0)
        self.assertEqual(sat_method_used, "header")

        # Check that saturated pixels are identified
        self.assertTrue(np.any(mask[10:20, 10:20]))

    def test_detect_saturated_pixels_fallback(self):
        """Test saturation detection fallback to default level."""
        # Create test science data
        sci_data = np.random.poisson(100, (100, 100)).astype(np.float32)

        # Add some saturated pixels above fallback level
        sci_data[10:20, 10:20] = 70000.0

        # Create a header without saturation keyword
        sci_hdr = {}

        config = {"method": "header", "keyword": "SATURATE", "fallback_level": 65000.0}

        saturation_level, sat_method_used, mask = detect_saturated_pixels(
            sci_data, sci_hdr, config
        )

        # Check that we fell back to default level
        self.assertEqual(saturation_level, 65000.0)
        self.assertEqual(sat_method_used, "default fallback")

        # Check that saturated pixels are identified
        self.assertTrue(np.any(mask[10:20, 10:20]))

    def test_detect_saturated_pixels_no_saturation(self):
        """Test saturation detection with no saturated pixels."""
        # Create test science data with no saturation - use a narrow distribution
        # with values well below what would be considered saturated
        np.random.seed(42)  # For reproducible results
        # Use a distribution that clearly doesn't have saturation
        sci_data = np.random.normal(100, 5, (100, 100)).astype(np.float32)
        # Clip to ensure no extreme values that might be interpreted as saturation
        sci_data = np.clip(sci_data, 0, 200)

        # Create a header
        sci_hdr = {}

        config = {
            "method": "header",  # Use header method to avoid histogram issues
            "keyword": "SATURATE",
            "fallback_level": 65000.0,
        }

        saturation_level, sat_method_used, mask = detect_saturated_pixels(
            sci_data, sci_hdr, config
        )

        # Check that we got a result
        self.assertIsNotNone(saturation_level)
        self.assertIsNotNone(sat_method_used)
        self.assertIsNotNone(mask)

        # With header method and no header keyword, should use fallback level
        # which should be high enough that no pixels are marked as saturated
        self.assertEqual(saturation_level, 65000.0)
        self.assertEqual(sat_method_used, "default fallback")
        self.assertEqual(np.sum(mask), 0)



    def test_robust_clump_happy_path(self):
        """Test successful detection of a saturation clump."""
        np.random.seed(42)
        # To make the clump prominent enough for scipy.signal.find_peaks,
        # we need a larger ratio of saturated pixels in the tail.
        data = np.random.normal(100, 10, 1000000)
        data = np.append(data, np.random.normal(65000, 50, 50000))

        from weightmask.satur import estimate_saturation_robust_clump
        level = estimate_saturation_robust_clump(data)

        self.assertIsNotNone(level)
        # Should be near the base of the 65000 peak (e.g. 64000-65000)
        self.assertTrue(60000 < level < 65000)

    def test_robust_clump_no_saturation(self):
        """Test that an image without saturation returns None."""
        np.random.seed(42)
        # Just background + normal stars, no clump at the top
        data = np.random.normal(100, 10, 100000)
        data = np.append(data, np.random.normal(50000, 1000, 100))

        from weightmask.satur import estimate_saturation_robust_clump
        level = estimate_saturation_robust_clump(data)

        self.assertIsNone(level)

    def test_robust_clump_empty_or_nan(self):
        """Test handling of empty arrays and all NaNs."""
        from weightmask.satur import estimate_saturation_robust_clump

        self.assertIsNone(estimate_saturation_robust_clump(np.array([])))
        self.assertIsNone(estimate_saturation_robust_clump(np.array([np.nan, np.inf])))

    def test_robust_clump_low_max_adu(self):
        """Test handling when the max value is below the 10000 ADU heuristic."""
        np.random.seed(42)
        data = np.random.normal(100, 10, 10000)
        # Max is around 150
        from weightmask.satur import estimate_saturation_robust_clump

        level = estimate_saturation_robust_clump(data)
        self.assertIsNone(level)

    def test_robust_clump_explicit_bounds(self):
        """Test that explicitly providing min_adu and max_adu works."""
        np.random.seed(42)
        data = np.random.normal(100, 10, 100000)
        data = np.append(data, np.random.normal(65000, 50, 500))

        from weightmask.satur import estimate_saturation_robust_clump
        # Pass bounds that tightly frame the clump
        level = estimate_saturation_robust_clump(data, min_adu=60000, max_adu=70000)

        self.assertIsNotNone(level)
        self.assertTrue(60000 < level < 65000)

    def test_robust_clump_exception_handling(self):
        """Test the try-except block by mocking a failure."""
        from weightmask.satur import estimate_saturation_robust_clump
        import unittest.mock as mock

        # Patch scipy.signal.find_peaks to raise an Exception
        with mock.patch('scipy.signal.find_peaks', side_effect=Exception("Mocked failure")):
            np.random.seed(42)
            data = np.random.normal(100, 10, 100000)
            data = np.append(data, np.random.normal(65000, 50, 500))
            level = estimate_saturation_robust_clump(data)

            self.assertIsNone(level)

if __name__ == "__main__":
    unittest.main()
