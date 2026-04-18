import unittest

import numpy as np

from weightmask.satur import detect_saturated_pixels, grow_bleed_trails


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

        saturation_level, sat_method_used, mask = detect_saturated_pixels(sci_data, sci_hdr, config)

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

        saturation_level, sat_method_used, mask = detect_saturated_pixels(sci_data, sci_hdr, config)

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

        saturation_level, sat_method_used, mask = detect_saturated_pixels(sci_data, sci_hdr, config)

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

        saturation_level, sat_method_used, mask = detect_saturated_pixels(sci_data, sci_hdr, config)

        # Check that we got a result
        self.assertIsNotNone(saturation_level)
        self.assertIsNotNone(sat_method_used)
        self.assertIsNotNone(mask)

        # With header method and no header keyword, should use fallback level
        # which should be high enough that no pixels are marked as saturated
        self.assertEqual(saturation_level, 65000.0)
        self.assertEqual(sat_method_used, "default fallback")
        self.assertEqual(np.sum(mask), 0)


class TestGrowBleedTrails(unittest.TestCase):
    def setUp(self):
        self.shape = (100, 100)
        self.sci_data = np.zeros(self.shape, dtype=np.float32)
        self.sat_mask = np.zeros(self.shape, dtype=bool)
        self.sky_map = np.zeros(self.shape, dtype=np.float32)
        self.bkg_rms_map = np.full(self.shape, 10.0, dtype=np.float32)
        self.config = {
            "mask_bleed_trails": True,
            "bleed_thresh_sigma": 5.0,
            "bleed_grow_vertical": 10,
            "bleed_grow_horizontal": 0,
        }

    def test_grow_bleed_trails_disabled(self):
        """Test that no growth occurs when mask_bleed_trails is False."""
        self.config["mask_bleed_trails"] = False
        self.sat_mask[50, 50] = True
        self.sci_data[50, 50] = 60000.0
        # Set values that would normally trigger growth
        self.sci_data[40:60, 50] = 1000.0

        mask = grow_bleed_trails(self.sci_data, self.sat_mask, self.sky_map, self.bkg_rms_map, self.config)
        np.testing.assert_array_equal(mask, self.sat_mask)

    def test_grow_bleed_trails_basic_vertical(self):
        """Test basic vertical growth of bleed trails."""
        self.sat_mask[50, 50] = True
        self.sci_data[50, 50] = 60000.0

        # Create a trail: pixels 45 to 55 in column 50 are above threshold
        # Threshold is 0 + 5 * 10 = 50
        self.sci_data[45:56, 50] = 100.0

        mask = grow_bleed_trails(self.sci_data, self.sat_mask, self.sky_map, self.bkg_rms_map, self.config)

        # Should have grown to cover 45:56
        expected_indices = np.arange(45, 56)
        self.assertTrue(np.all(mask[expected_indices, 50]))
        # Check that it didn't grow further (max_grow is 10, but trail ends at 45/55)
        self.assertFalse(mask[44, 50])
        self.assertFalse(mask[56, 50])

    def test_grow_bleed_trails_threshold_stop(self):
        """Test that growth stops when pixel values hit the background threshold."""
        self.sat_mask[50, 50] = True
        self.sci_data[50, 50] = 60000.0

        # Threshold = 0 + 5 * 10 = 50
        # Set pixels above threshold until 48 and 52
        self.sci_data[48:53, 50] = 100.0
        # Pixels 47 and 53 are at background level (0)
        self.sci_data[47, 50] = 0.0
        self.sci_data[53, 50] = 0.0

        mask = grow_bleed_trails(self.sci_data, self.sat_mask, self.sky_map, self.bkg_rms_map, self.config)

        self.assertTrue(np.all(mask[48:53, 50]))
        self.assertFalse(mask[47, 50])
        self.assertFalse(mask[53, 50])

    def test_grow_bleed_trails_max_grow_limit(self):
        """Test that growth is limited by bleed_grow_vertical."""
        self.config["bleed_grow_vertical"] = 5
        self.sat_mask[50, 50] = True
        self.sci_data[50, 50] = 60000.0

        # Trail extends very far
        self.sci_data[:, 50] = 100.0

        mask = grow_bleed_trails(self.sci_data, self.sat_mask, self.sky_map, self.bkg_rms_map, self.config)

        # Saturated core is at 50. Max grow 5 up and 5 down.
        # Should cover 45 to 55 inclusive (50-5=45, 50+5=55)
        self.assertTrue(np.all(mask[45:56, 50]))
        self.assertFalse(mask[44, 50])
        self.assertFalse(mask[56, 50])

    def test_grow_bleed_trails_horizontal_dilation(self):
        """Test that the mask is expanded horizontally."""
        self.config["bleed_grow_horizontal"] = 2
        self.sat_mask[50, 50] = True
        self.sci_data[50, 50] = 60000.0

        mask = grow_bleed_trails(self.sci_data, self.sat_mask, self.sky_map, self.bkg_rms_map, self.config)

        # Saturated pixel at (50, 50).
        # Horizontal dilation of 2 should mark (50, 48) to (50, 52)
        self.assertTrue(np.all(mask[50, 48:53]))
        self.assertFalse(mask[50, 47])
        self.assertFalse(mask[50, 53])

    def test_grow_bleed_trails_no_saturation(self):
        """Test behavior when there are no saturated pixels."""
        mask = grow_bleed_trails(self.sci_data, self.sat_mask, self.sky_map, self.bkg_rms_map, self.config)
        self.assertFalse(np.any(mask))

    def test_grow_bleed_trails_image_boundaries(self):
        """Test that growth handles image boundaries correctly."""
        # Top boundary
        self.sat_mask[0, 50] = True
        self.sci_data[0, 50] = 60000.0
        self.sci_data[1:10, 50] = 100.0

        # Bottom boundary
        self.sat_mask[99, 20] = True
        self.sci_data[99, 20] = 60000.0
        self.sci_data[90:99, 20] = 100.0

        mask = grow_bleed_trails(self.sci_data, self.sat_mask, self.sky_map, self.bkg_rms_map, self.config)

        # Check top growth (downwards)
        self.assertTrue(np.all(mask[0:10, 50]))
        # Check bottom growth (upwards)
        self.assertTrue(np.all(mask[90:100, 20]))


if __name__ == "__main__":
    unittest.main()
