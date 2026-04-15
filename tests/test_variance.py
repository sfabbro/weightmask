import numpy as np
import unittest
from unittest.mock import patch
from weightmask.variance import calculate_inverse_variance, _calculate_inverse_variance_theoretical


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
    
    def test__calculate_inverse_variance_theoretical_happy_path(self):
        """Test internal theoretical inverse variance logic with normal values."""
        sky_map = np.array([[100.0, 100.0], [100.0, 100.0]], dtype=np.float32)
        flat_map = np.array([[1.0, 0.5], [1.0, 1.0]], dtype=np.float32)
        gain = 2.0
        read_noise_e = 4.0
        epsilon = 1e-9

        inv_var = _calculate_inverse_variance_theoretical(
            sky_map=sky_map,
            flat_map=flat_map,
            gain=gain,
            read_noise_e=read_noise_e,
            epsilon=epsilon
        )

        # Calculate expected values manually
        # Top-left: sky=100, flat=1. variance_e = (100/1)*2 + 4^2 = 200 + 16 = 216. inv_var = 2^2 / 216 = 4/216 = 1/54
        # Top-right: sky=100, flat=0.5. variance_e = (100/0.5)*2 + 4^2 = 200*2 + 16 = 416. inv_var = 4 / 416 = 1/104
        # Bottom-left/right: same as Top-left
        expected_inv_var = np.array([
            [4.0 / 216.0, 4.0 / 416.0],
            [4.0 / 216.0, 4.0 / 216.0]
        ], dtype=np.float32)

        np.testing.assert_allclose(inv_var, expected_inv_var, rtol=1e-5)

    def test__calculate_inverse_variance_theoretical_edge_cases(self):
        """Test internal theoretical inverse variance logic with edge cases like negative sky and zero flat."""
        sky_map = np.array([[100.0, -50.0], [100.0, 100.0]], dtype=np.float32) # Negative sky
        flat_map = np.array([[1.0, 1.0], [1e-10, -1.0]], dtype=np.float32) # Flat <= epsilon and negative flat
        gain = 2.0
        read_noise_e = 4.0
        epsilon = 1e-9

        inv_var = _calculate_inverse_variance_theoretical(
            sky_map=sky_map,
            flat_map=flat_map,
            gain=gain,
            read_noise_e=read_noise_e,
            epsilon=epsilon
        )

        # Expected calculations
        # Top-left: sky=100, flat=1 -> 4 / ((100/1)*2 + 16) = 4 / 216
        # Top-right: sky=-50 (clamped to 0), flat=1 -> variance_e = (0/1)*2 + 16 = 16. inv_var = 4 / 16 = 0.25
        # Bottom-left: flat=1e-10 (<= epsilon). Should be masked to 0.0.
        # Bottom-right: flat=-1.0 (<= epsilon). Should be masked to 0.0.

        expected_inv_var = np.array([
            [4.0 / 216.0, 0.25],
            [0.0, 0.0]
        ], dtype=np.float32)

        np.testing.assert_allclose(inv_var, expected_inv_var, rtol=1e-5)
        self.assertEqual(inv_var.dtype, np.float32)

    def test__calculate_inverse_variance_theoretical_zero_read_noise(self):
        """Test internal theoretical inverse variance logic with zero read noise."""
        sky_map = np.array([[100.0]], dtype=np.float32)
        flat_map = np.array([[1.0]], dtype=np.float32)
        gain = 2.0
        read_noise_e = 0.0
        epsilon = 1e-9

        inv_var = _calculate_inverse_variance_theoretical(
            sky_map=sky_map,
            flat_map=flat_map,
            gain=gain,
            read_noise_e=read_noise_e,
            epsilon=epsilon
        )

        # variance_e = (100/1)*2 + 0 = 200. inv_var = 4 / 200 = 0.02
        expected_inv_var = np.array([[0.02]], dtype=np.float32)
        np.testing.assert_allclose(inv_var, expected_inv_var, rtol=1e-5)

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


if __name__ == "__main__":
    unittest.main()