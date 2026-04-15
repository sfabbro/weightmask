import numpy as np
import unittest
from unittest.mock import patch
from weightmask.variance import calculate_inverse_variance, _unbias_variance


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

    @patch('weightmask.variance._calculate_empirical_noise_params')
    def test_calculate_inverse_variance_empirical_fit_fallback(self, mock_emp_noise):
        """Test empirical fit fallback to theoretical method."""
        # Setup mock to fail
        mock_emp_noise.return_value = (None, None)

        # Create test data
        sky_map = np.full((10, 10), 100.0, dtype=np.float32)
        flat_map = np.ones((10, 10), dtype=np.float32)
        sci_data = np.full((10, 10), 110.0, dtype=np.float32)
        obj_mask = np.zeros((10, 10), dtype=bool)

        gain = 1.5
        read_noise_e = 5.0
        epsilon = 1e-9

        variance_cfg = {
            'method': 'empirical_fit',
            'epsilon': epsilon,
            'gain': gain,
            'read_noise': read_noise_e,
            'empirical_patch_size': 128,
            'empirical_clip_sigma': 3.0
        }

        # Calculate inverse variance
        inv_variance = calculate_inverse_variance(
            variance_cfg=variance_cfg,
            sky_map=sky_map,
            flat_map=flat_map,
            bkg_rms_map=None,
            sci_data=sci_data,
            obj_mask=obj_mask
        )

        # Check that we got a result
        self.assertIsNotNone(inv_variance)

        # Expected theoretical calculation:
        # variance_e = (sky / flat) * gain + read_noise_e**2
        # variance_e = (100.0 / 1.0) * 1.5 + 5.0**2 = 150 + 25 = 175
        # inv_variance = gain**2 / variance_e = 1.5**2 / 175 = 2.25 / 175
        expected_val = (gain**2) / ((100.0 / 1.0) * gain + read_noise_e**2)

        np.testing.assert_allclose(inv_variance, expected_val, rtol=1e-5)

        # Verify that the mock was called
        mock_emp_noise.assert_called_once_with(sci_data, obj_mask, 128, 3.0)

    def test_unbias_variance_none_inv_variance(self):
        """Test _unbias_variance with inv_variance=None."""
        sci_data = np.ones((10, 10))
        sky_map = np.ones((10, 10))
        result = _unbias_variance(None, sci_data, sky_map, gain=1.0, epsilon=1e-9)
        self.assertIsNone(result)

    def test_unbias_variance_zero_gain(self):
        """Test _unbias_variance with gain=0."""
        inv_variance = np.ones((10, 10))
        sci_data = np.ones((10, 10))
        sky_map = np.ones((10, 10))
        result = _unbias_variance(inv_variance, sci_data, sky_map, gain=0.0, epsilon=1e-9)
        np.testing.assert_array_equal(result, inv_variance)

    def test_unbias_variance_negative_gain(self):
        """Test _unbias_variance with negative gain."""
        inv_variance = np.ones((10, 10))
        sci_data = np.ones((10, 10))
        sky_map = np.ones((10, 10))
        result = _unbias_variance(inv_variance, sci_data, sky_map, gain=-1.0, epsilon=1e-9)
        np.testing.assert_array_equal(result, inv_variance)


if __name__ == "__main__":
    unittest.main()