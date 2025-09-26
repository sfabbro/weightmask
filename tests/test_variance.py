import numpy as np
import unittest
from weightmask.variance import calculate_inverse_variance


class TestVariance(unittest.TestCase):
    
    def test_calculate_inverse_variance_theoretical(self):
        """Test theoretical inverse variance calculation."""
        # Create test data
        sky_map = np.full((100, 100), 100.0, dtype=np.float32)
        flat_map = np.ones((100, 100), dtype=np.float32)
        gain = 1.5
        read_noise_e = 5.0
        bkg_rms_map = None  # Not used for theoretical method
        epsilon = 1e-9
        
        inv_variance = calculate_inverse_variance(
            variance_cfg={'method': 'theoretical', 'epsilon': epsilon, 'gain': gain, 'read_noise': read_noise_e},
            sky_map=sky_map,
            flat_map=flat_map,
            bkg_rms_map=bkg_rms_map,
            sci_data=None,
            obj_mask=None
        )
        
        # Check that we got a result
        self.assertIsNotNone(inv_variance)
        
        # If we got a result, check its properties
        if inv_variance is not None:
            # Check that result has correct shape
            self.assertEqual(inv_variance.shape, sky_map.shape)
            
            # Check that result is finite
            self.assertTrue(np.isfinite(inv_variance).all())
            
            # Check that values are positive
            self.assertTrue(np.all(inv_variance >= 0))
    
    def test_calculate_inverse_variance_rms_map(self):
        """Test RMS map inverse variance calculation."""
        # Create test data
        sky_map = None  # Not used for rms_map method
        flat_map = None  # Not used for rms_map method
        gain = 1.5  # Not used for rms_map method
        read_noise_e = 5.0  # Not used for rms_map method
        bkg_rms_map = np.full((100, 100), 10.0, dtype=np.float32)
        epsilon = 1e-9
        
        inv_variance = calculate_inverse_variance(
            variance_cfg={'method': 'rms_map', 'epsilon': epsilon, 'gain': gain, 'read_noise': read_noise_e},
            sky_map=sky_map,
            flat_map=flat_map,
            bkg_rms_map=bkg_rms_map,
            sci_data=None,
            obj_mask=None
        )
        
        # Check that we got a result
        self.assertIsNotNone(inv_variance)
        
        # If we got a result, check its properties
        if inv_variance is not None:
            # Check that result has correct shape
            self.assertEqual(inv_variance.shape, bkg_rms_map.shape)
            
            # Check that result is finite
            self.assertTrue(np.isfinite(inv_variance).all())
            
            # Check that values are positive
            self.assertTrue(np.all(inv_variance >= 0))
    
    def test_calculate_inverse_variance_invalid_method(self):
        """Test inverse variance calculation with invalid method."""
        inv_variance = calculate_inverse_variance(
            variance_cfg={'method': 'invalid', 'epsilon': 1e-9, 'gain': 1.5, 'read_noise': 5.0},
            sky_map=None,
            flat_map=None,
            bkg_rms_map=None,
            sci_data=None,
            obj_mask=None
        )
        
        # Should return None for invalid method
        self.assertIsNone(inv_variance)


if __name__ == "__main__":
    unittest.main()