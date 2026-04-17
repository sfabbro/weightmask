import numpy as np
import unittest
from weightmask.objects import detect_objects


class TestObjects(unittest.TestCase):
    def setUp(self):
        # Create a small synthetic image
        self.shape = (100, 100)
        # Background is random noise around 0
        self.data_sub = np.random.normal(0, 1.0, self.shape).astype(np.float32)
        self.bkg_rms_map = np.ones(self.shape, dtype=np.float32)
        self.existing_mask = np.zeros(self.shape, dtype=bool)

    def add_star(self, data, y, x, flux, radius):
        """Add a simple 2D Gaussian star."""
        Y, X = np.ogrid[: data.shape[0], : data.shape[1]]
        dist_sq = (X - x) ** 2 + (Y - y) ** 2
        # Use simple gaussian profile
        star = flux * np.exp(-dist_sq / (2.0 * radius**2))
        data += star

    def test_detect_objects_basic(self):
        """Test basic object detection without diffraction spikes or dynamic halos."""
        # Add a star at the center
        self.add_star(self.data_sub, 50, 50, flux=100.0, radius=2.0)

        config = {
            "extract_thresh": 3.0,
            "min_area": 5,
            "dynamic_halo_scaling": False,
            "spike_enable": False,
        }

        obj_mask = detect_objects(
            self.data_sub, self.bkg_rms_map, self.existing_mask, config
        )

        # Check that objects were detected
        self.assertTrue(np.sum(obj_mask) > 0)
        # The star is at (50, 50), so this pixel should be masked
        self.assertTrue(obj_mask[50, 50])
        # Background should not be masked
        self.assertFalse(obj_mask[10, 10])

    def test_detect_objects_with_spikes(self):
        """Test detection with diffraction spike masking."""
        # Add a very bright star to trigger spike masking
        self.add_star(self.data_sub, 50, 50, flux=200000.0, radius=3.0)

        config = {
            "extract_thresh": 3.0,
            "min_area": 5,
            "dynamic_halo_scaling": False,
            "spike_enable": True,
            "spike_flux_thresh": 1e4,  # Lower threshold to ensure our star triggers it
            "spike_length_base": 20,
            "spike_width": 3,
        }

        obj_mask = detect_objects(
            self.data_sub, self.bkg_rms_map, self.existing_mask, config
        )

        self.assertTrue(np.sum(obj_mask) > 0)
        self.assertTrue(obj_mask[50, 50])

        # Check horizontal spike
        self.assertTrue(obj_mask[50, 50 + 15])
        self.assertTrue(obj_mask[50, 50 - 15])

        # Check vertical spike
        self.assertTrue(obj_mask[50 + 15, 50])
        self.assertTrue(obj_mask[50 - 15, 50])

    def test_detect_objects_dynamic_halo(self):
        """Test detection with dynamic halo scaling enabled."""
        # Add two stars, one bright, one faint
        self.add_star(self.data_sub, 25, 25, flux=1000.0, radius=2.0)
        self.add_star(self.data_sub, 75, 75, flux=20.0, radius=2.0)

        config = {
            "extract_thresh": 3.0,
            "min_area": 5,
            "dynamic_halo_scaling": True,
            "halo_scale_factor": 0.5,
            "spike_enable": False,
        }

        obj_mask = detect_objects(
            self.data_sub, self.bkg_rms_map, self.existing_mask, config
        )

        self.assertTrue(np.sum(obj_mask) > 0)
        self.assertTrue(obj_mask[25, 25])
        self.assertTrue(obj_mask[75, 75])

    def test_detect_objects_clutter_scaling(self):
        """Test adaptive thresholding in highly cluttered images."""
        # Create a highly cluttered image to trigger the mad_approx logic
        # We need tail_ratio > 3.0 and enough valid data
        cluttered_data = np.random.normal(0, 1.0, (150, 150)).astype(np.float32)
        # Add many bright "clutter" pixels to create a fat tail
        clutter_indices_y = np.random.randint(0, 150, 500)
        clutter_indices_x = np.random.randint(0, 150, 500)
        cluttered_data[clutter_indices_y, clutter_indices_x] = 20.0

        bkg_rms = np.ones((150, 150), dtype=np.float32)

        config = {
            "extract_thresh": 3.0,
            "min_area": 5,
        }

        # This should trigger the "Clutter penalty" print statement
        obj_mask = detect_objects(cluttered_data, bkg_rms, None, config)

        # As long as it doesn't crash and returns a boolean array, we are good
        self.assertIsInstance(obj_mask, np.ndarray)
        self.assertEqual(obj_mask.dtype, bool)
        self.assertEqual(obj_mask.shape, (150, 150))

    def test_detect_objects_empty_input(self):
        """Test behavior with no objects detected."""
        # Very high threshold, no stars
        config = {
            "extract_thresh": 100.0,
            "min_area": 5,
        }

        obj_mask = detect_objects(
            self.data_sub, self.bkg_rms_map, self.existing_mask, config
        )

        self.assertIsInstance(obj_mask, np.ndarray)
        self.assertEqual(obj_mask.dtype, bool)
        self.assertEqual(np.sum(obj_mask), 0)

    def test_detect_objects_with_existing_mask(self):
        """Test that already masked pixels are not included in the returned new mask."""
        self.add_star(self.data_sub, 50, 50, flux=100.0, radius=2.0)

        config = {
            "extract_thresh": 3.0,
            "min_area": 5,
        }

        # Mask the center pixel manually beforehand
        existing_mask = np.zeros(self.shape, dtype=bool)
        existing_mask[50, 50] = True

        obj_mask = detect_objects(
            self.data_sub, self.bkg_rms_map, existing_mask, config
        )

        # The star is detected, but the center pixel shouldn't be in the *new* mask
        self.assertFalse(obj_mask[50, 50])
        # But pixels around it should be in the new mask
        self.assertTrue(obj_mask[50, 51] or obj_mask[51, 50])

    def test_detect_objects_exception_handling(self):
        """Return empty mask when SEP raises (e.g. 1D arrays)."""
        data_sub = np.ones(10, dtype=np.float32)
        bkg_rms_map = np.ones(10, dtype=np.float32)

        mask = detect_objects(data_sub, bkg_rms_map, None, {})

        self.assertEqual(mask.shape, data_sub.shape)
        self.assertFalse(np.any(mask))
        self.assertEqual(mask.dtype, bool)

    def test_detect_objects_happy_path_smoke(self):
        """Minimal happy-path smoke alongside PR #11 suite."""
        data_sub = np.zeros((100, 100), dtype=np.float32)
        bkg_rms_map = np.ones((100, 100), dtype=np.float32)
        data_sub[45:55, 45:55] = 100.0

        mask = detect_objects(data_sub, bkg_rms_map, None, {"extract_thresh": 3.0})

        self.assertEqual(mask.shape, data_sub.shape)
        self.assertTrue(np.any(mask))
        self.assertEqual(mask.dtype, bool)


if __name__ == "__main__":
    unittest.main()
