import numpy as np
import unittest

from weightmask.cosmics import detect_cosmic_rays

# Try to find astroscrappy
from importlib.util import find_spec
ASTROSCRAPPY_AVAILABLE = find_spec("astroscrappy") is not None


class TestCosmics(unittest.TestCase):
    def test_detect_cosmic_rays(self):
        """Test cosmic ray detection."""
        # Create test science data
        sci_data = np.random.poisson(100, (100, 100)).astype(np.float32)

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

        # Skip test if astroscrappy is not available
        if not ASTROSCRAPPY_AVAILABLE:
            # Test that function handles missing astroscrappy gracefully
            mask = detect_cosmic_rays(
                sci_data, existing_mask, saturation_level, gain, read_noise, config
            )
            # Should return empty mask when astroscrappy is not available
            self.assertEqual(np.sum(mask), 0)
            return

        # If astroscrappy is available, run the full test
        mask = detect_cosmic_rays(
            sci_data, existing_mask, saturation_level, gain, read_noise, config
        )

        # Check that we got a result
        self.assertIsNotNone(mask)

        # Check that mask is boolean
        self.assertEqual(mask.dtype, bool)

    def test_detect_cosmic_rays_with_existing_mask(self):
        """Test cosmic ray detection with existing masked pixels."""
        # Create test science data
        sci_data = np.random.poisson(100, (100, 100)).astype(np.float32)

        # Create an existing mask with some pixels masked
        existing_mask = np.zeros((100, 100), dtype=bool)
        existing_mask[10, 10] = True

        # Parameters
        saturation_level = 65000.0
        gain = 1.5
        read_noise = 5.0

        config = {"sigclip": 4.5, "objlim": 5.0}

        # Skip test if astroscrappy is not available
        if not ASTROSCRAPPY_AVAILABLE:
            # Test that function handles missing astroscrappy gracefully
            mask = detect_cosmic_rays(
                sci_data, existing_mask, saturation_level, gain, read_noise, config
            )
            # Should return empty mask when astroscrappy is not available
            self.assertEqual(np.sum(mask), 0)
            return

        # If astroscrappy is available, run the full test
        mask = detect_cosmic_rays(
            sci_data, existing_mask, saturation_level, gain, read_noise, config
        )

        # Check that existing masked pixels are not in the result
        self.assertFalse(mask[10, 10])

    def test_detect_cosmic_rays_astroscrappy_failure(self):
        """Test cosmic ray detection when astroscrappy fails."""
        # Create test science data
        sci_data = np.random.poisson(100, (100, 100)).astype(np.float32)

        # Create an existing mask (no masked pixels)
        existing_mask = np.zeros((100, 100), dtype=bool)

        # Parameters
        saturation_level = 65000.0
        gain = 1.5
        read_noise = 5.0

        config = {"sigclip": 4.5, "objlim": 5.0}

        # Test behavior when astroscrappy is not available
        if not ASTROSCRAPPY_AVAILABLE:
            mask = detect_cosmic_rays(
                sci_data, existing_mask, saturation_level, gain, read_noise, config
            )
            # Should return empty mask when astroscrappy is not available
            self.assertEqual(np.sum(mask), 0)


if __name__ == "__main__":
    unittest.main()
