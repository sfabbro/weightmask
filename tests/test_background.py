import numpy as np
import unittest
from weightmask.background import estimate_background


class TestBackground(unittest.TestCase):
    
    def test_estimate_background(self):
        """Test background estimation."""
        # Create test science data
        sci_data = np.random.poisson(100, (100, 100)).astype(np.float32)
        
        # Add some sources
        sci_data[30:40, 30:40] = 1000.0
        
        # Create a mask for the sources
        mask = np.zeros((100, 100), dtype=bool)
        mask[30:40, 30:40] = True
        
        config = {
            'box_size': 32,
            'filter_size': 3
        }
        
        bkg_map, bkg_rms_map = estimate_background(sci_data, mask, config)
        
        # Check that we got results
        self.assertIsNotNone(bkg_map)
        self.assertIsNotNone(bkg_rms_map)
        
        # Check that results have correct shape
        self.assertEqual(bkg_map.shape, sci_data.shape)
        self.assertEqual(bkg_rms_map.shape, sci_data.shape)
        
        # Check that results are finite
        self.assertTrue(np.isfinite(bkg_map).all())
        self.assertTrue(np.isfinite(bkg_rms_map).all())
        
        # Check that RMS values are positive
        self.assertTrue(np.all(bkg_rms_map > 0))
    
    def test_estimate_background_no_mask(self):
        """Test background estimation with no masked pixels."""
        # Create test science data
        sci_data = np.random.poisson(100, (100, 100)).astype(np.float32)
        
        # Create an empty mask
        mask = np.zeros((100, 100), dtype=bool)
        
        config = {
            'box_size': 32,
            'filter_size': 3
        }
        
        bkg_map, bkg_rms_map = estimate_background(sci_data, mask, config)
        
        # Check that we got results
        self.assertIsNotNone(bkg_map)
        self.assertIsNotNone(bkg_rms_map)
    
    def test_estimate_background_all_masked(self):
        """Test background estimation with all pixels masked."""
        # Create test science data
        sci_data = np.random.poisson(100, (100, 100)).astype(np.float32)
        
        # Create a mask with all pixels masked
        mask = np.ones((100, 100), dtype=bool)
        
        config = {
            'box_size': 32,
            'filter_size': 3
        }
        
        bkg_map, bkg_rms_map = estimate_background(sci_data, mask, config)
        
        # Check that we got results (fallback to global)
        self.assertIsNotNone(bkg_map)
        self.assertIsNotNone(bkg_rms_map)


if __name__ == "__main__":
    unittest.main()