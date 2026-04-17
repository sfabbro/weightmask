import numpy as np
import unittest
from unittest.mock import patch

from weightmask.variance import (
    _calculate_inverse_variance_theoretical,
    _unbias_variance,
    calculate_inverse_variance,
)


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
            variance_cfg={
                "method": "theoretical",
                "epsilon": epsilon,
                "gain": gain,
                "read_noise": read_noise_e,
            },
            sky_map=sky_map,
            flat_map=flat_map,
            bkg_rms_map=bkg_rms_map,
            sci_data=None,
            obj_mask=None,
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
            epsilon=epsilon,
        )

        # Calculate expected values manually
        # Top-left: sky=100, flat=1. variance_e = (100/1)*2 + 4^2 = 200 + 16 = 216. inv_var = 2^2 / 216 = 4/216 = 1/54
        # Top-right: sky=100, flat=0.5. variance_e = (100/0.5)*2 + 4^2 = 200*2 + 16 = 416. inv_var = 4 / 416 = 1/104
        # Bottom-left/right: same as Top-left
        expected_inv_var = np.array(
            [[4.0 / 216.0, 4.0 / 416.0], [4.0 / 216.0, 4.0 / 216.0]], dtype=np.float32
        )

        np.testing.assert_allclose(inv_var, expected_inv_var, rtol=1e-5)

    def test__calculate_inverse_variance_theoretical_edge_cases(self):
        """Test internal theoretical inverse variance logic with edge cases like negative sky and zero flat."""
        sky_map = np.array(
            [[100.0, -50.0], [100.0, 100.0]], dtype=np.float32
        )  # Negative sky
        flat_map = np.array(
            [[1.0, 1.0], [1e-10, -1.0]], dtype=np.float32
        )  # Flat <= epsilon and negative flat
        gain = 2.0
        read_noise_e = 4.0
        epsilon = 1e-9

        inv_var = _calculate_inverse_variance_theoretical(
            sky_map=sky_map,
            flat_map=flat_map,
            gain=gain,
            read_noise_e=read_noise_e,
            epsilon=epsilon,
        )

        # Expected calculations
        # Top-left: sky=100, flat=1 -> 4 / ((100/1)*2 + 16) = 4 / 216
        # Top-right: sky=-50 (clamped to 0), flat=1 -> variance_e = (0/1)*2 + 16 = 16. inv_var = 4 / 16 = 0.25
        # Bottom-left: flat=1e-10 (<= epsilon). Should be masked to 0.0.
        # Bottom-right: flat=-1.0 (<= epsilon). Should be masked to 0.0.

        expected_inv_var = np.array([[4.0 / 216.0, 0.25], [0.0, 0.0]], dtype=np.float32)

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
            epsilon=epsilon,
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
            variance_cfg={
                "method": "rms_map",
                "epsilon": epsilon,
                "gain": gain,
                "read_noise": read_noise_e,
            },
            sky_map=sky_map,
            flat_map=flat_map,
            bkg_rms_map=bkg_rms_map,
            sci_data=None,
            obj_mask=None,
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
            variance_cfg={
                "method": "invalid",
                "epsilon": 1e-9,
                "gain": 1.5,
                "read_noise": 5.0,
            },
            sky_map=None,
            flat_map=None,
            bkg_rms_map=None,
            sci_data=None,
            obj_mask=None,
        )

        # Should return None for invalid method
        self.assertIsNone(inv_variance)

    @patch("weightmask.variance._calculate_empirical_noise_params")
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
            "method": "empirical_fit",
            "epsilon": epsilon,
            "gain": gain,
            "read_noise": read_noise_e,
            "empirical_patch_size": 128,
            "empirical_clip_sigma": 3.0,
        }

        # Calculate inverse variance
        inv_variance = calculate_inverse_variance(
            variance_cfg=variance_cfg,
            sky_map=sky_map,
            flat_map=flat_map,
            bkg_rms_map=None,
            sci_data=sci_data,
            obj_mask=obj_mask,
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
        result = _unbias_variance(
            inv_variance, sci_data, sky_map, gain=0.0, epsilon=1e-9
        )
        np.testing.assert_array_equal(result, inv_variance)

    def test_unbias_variance_negative_gain(self):
        """Test _unbias_variance with negative gain."""
        inv_variance = np.ones((10, 10))
        sci_data = np.ones((10, 10))
        sky_map = np.ones((10, 10))
        result = _unbias_variance(
            inv_variance, sci_data, sky_map, gain=-1.0, epsilon=1e-9
        )
        np.testing.assert_array_equal(result, inv_variance)

    def test_calculate_empirical_noise_params_success(self):
        """Test successful calculation of empirical noise parameters."""
        from weightmask.variance import _calculate_empirical_noise_params

        # Set up a random seed for reproducibility
        np.random.seed(42)

        # True parameters
        true_gain = 2.0
        true_rn_e = 5.0
        true_rn_adu = true_rn_e / true_gain

        shape = (1024, 1024)
        patch_size = 64
        sci_data = np.zeros(shape, dtype=np.float32)
        obj_mask = np.zeros(shape, dtype=bool)

        # Generate synthetic data
        for y in range(0, shape[0], patch_size):
            for x in range(0, shape[1], patch_size):
                median_signal = np.random.uniform(50, 1000)
                var_adu = true_rn_adu**2 + median_signal / true_gain
                std_adu = np.sqrt(var_adu)

                noise = np.random.normal(
                    loc=0, scale=std_adu, size=(patch_size, patch_size)
                )
                patch_data = median_signal + noise
                sci_data[y : y + patch_size, x : x + patch_size] = patch_data

        emp_gain, emp_rn_e = _calculate_empirical_noise_params(
            sci_data, obj_mask, patch_size, robust_sigma_clip=3.0
        )

        # Verify that we got a result
        self.assertIsNotNone(emp_gain)
        self.assertIsNotNone(emp_rn_e)

        # Check if the calculated values are close to the true values
        # Tolerance is relatively large because it's an empirical fit on random data
        self.assertAlmostEqual(emp_gain, true_gain, delta=0.2)
        self.assertAlmostEqual(emp_rn_e, true_rn_e, delta=0.5)

    def test_calculate_empirical_noise_params_not_enough_pixels(self):
        """Test empirical calculation when mostly masked."""
        from weightmask.variance import _calculate_empirical_noise_params

        shape = (256, 256)
        patch_size = 64
        sci_data = np.random.normal(100, 10, shape).astype(np.float32)

        # Mask almost everything so valid_pixels.size < 100
        obj_mask = np.ones(shape, dtype=bool)

        emp_gain, emp_rn_e = _calculate_empirical_noise_params(
            sci_data, obj_mask, patch_size, robust_sigma_clip=3.0
        )

        self.assertIsNone(emp_gain)
        self.assertIsNone(emp_rn_e)

    def test_calculate_empirical_noise_params_not_enough_patches(self):
        """Test empirical calculation with too few patches."""
        from weightmask.variance import _calculate_empirical_noise_params

        # Shape 128x128 with patch size 64 -> only 4 patches (needs 10)
        shape = (128, 128)
        patch_size = 64
        sci_data = np.random.normal(100, 10, shape).astype(np.float32)
        obj_mask = np.zeros(shape, dtype=bool)

        emp_gain, emp_rn_e = _calculate_empirical_noise_params(
            sci_data, obj_mask, patch_size, robust_sigma_clip=3.0
        )

        self.assertIsNone(emp_gain)
        self.assertIsNone(emp_rn_e)

    def test_calculate_empirical_noise_params_zero_variance(self):
        """Test empirical calculation when patches have zero variance."""
        from weightmask.variance import _calculate_empirical_noise_params

        shape = (512, 512)
        patch_size = 64
        # Constant data, variance is 0
        sci_data = np.ones(shape, dtype=np.float32) * 100.0
        obj_mask = np.zeros(shape, dtype=bool)

        emp_gain, emp_rn_e = _calculate_empirical_noise_params(
            sci_data, obj_mask, patch_size, robust_sigma_clip=3.0
        )

        self.assertIsNone(emp_gain)
        self.assertIsNone(emp_rn_e)

    def test_calculate_empirical_noise_params_linear_regression_failure(self):
        """Test empirical calculation when linear regression fails (invalid slope/intercept)."""
        from weightmask.variance import _calculate_empirical_noise_params

        # Set up a random seed
        np.random.seed(42)

        shape = (1024, 1024)
        patch_size = 64
        sci_data = np.zeros(shape, dtype=np.float32)
        obj_mask = np.zeros(shape, dtype=bool)

        # Create patches such that variance decreases as median increases
        # This will result in a negative slope, triggering the invalid slope error.

        for y in range(0, shape[0], patch_size):
            for x in range(0, shape[1], patch_size):
                median_signal = np.random.uniform(50, 1000)
                # Intentionally make variance negatively correlated with signal
                var_adu = 1000.0 / median_signal
                std_adu = np.sqrt(var_adu)

                noise = np.random.normal(
                    loc=0, scale=std_adu, size=(patch_size, patch_size)
                )
                patch_data = median_signal + noise
                sci_data[y : y + patch_size, x : x + patch_size] = patch_data

        emp_gain, emp_rn_e = _calculate_empirical_noise_params(
            sci_data, obj_mask, patch_size, robust_sigma_clip=3.0
        )

        self.assertIsNone(emp_gain)
        self.assertIsNone(emp_rn_e)


if __name__ == "__main__":
    unittest.main()
