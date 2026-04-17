import numpy as np
import unittest
import warnings
from weightmask.background import estimate_background
from weightmask.variance import _rescale_variance_robust, _unbias_variance
from weightmask.cosmics import detect_cosmic_rays
from weightmask.streaks import detect_streaks
from weightmask.satur import grow_bleed_trails


class TestRobustFeatures(unittest.TestCase):
    def setUp(self):
        # Common test data
        self.size = 256
        self.shape = (self.size, self.size)
        self.noise = 10.0
        self.data = np.random.normal(100, self.noise, self.shape).astype(np.float32)

    def test_background_tiered_retry(self):
        """Test that background estimation retries with larger box if it fails."""
        # Create data with some NaN values or extremely sparse unmasked pixels to force SEP failure
        # in a small box but success in a larger one.
        data = np.full(self.shape, 100.0, dtype=np.float32)
        mask = np.ones(self.shape, dtype=bool)
        # unmask only 2 pixels in a 32x32 corner
        mask[0, 0] = False
        mask[1, 1] = False

        cfg = {
            "box_size": 32,  # This will fail for the first tile
            "filter_size": 3,
            "padding": 10,
        }

        # This should trigger tiered retries
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            bkg, rms = estimate_background(data, mask, cfg)

        self.assertIsNotNone(bkg)
        self.assertIsNotNone(rms)
        self.assertTrue(np.all(rms > 0))

    def test_background_mask_fallback(self):
        """Test that high mask coverage triggers global fallback."""
        data = np.random.normal(100, 10, self.shape).astype(np.float32)
        mask = np.ones(self.shape, dtype=bool)
        # Only 5% unmasked
        mask[50:60, 50:60] = False

        cfg = {"box_size": 64, "filter_size": 3}
        bkg, rms = estimate_background(data, mask, cfg)

        # Global robust median of data is ~100
        self.assertAlmostEqual(np.median(bkg), 100.0, delta=5.0)

    def test_variance_rescaling(self):
        """Test robust variance rescaling (SNR=1 check)."""
        sky = np.full(self.shape, 100.0, dtype=np.float32)
        # True variance is 100 (noise 10), but we provide a wrong inv_var (e.g. 0.04 -> var 25)
        # So SNR stdev will be noise/sqrt(var) = 10/5 = 2.0
        # Scaling should reduce inv_var by factor of 4 to 0.01
        inv_var = np.full(self.shape, 0.04, dtype=np.float32)
        obj_mask = np.zeros(self.shape, dtype=bool)

        # Need sci_data with the noise
        data = np.random.normal(100, 10, self.shape).astype(np.float32)

        scaled_inv_var = _rescale_variance_robust(inv_var, data, sky, obj_mask, 1e-9)

        # Expected scale factor: 1.0 / (2.0^2) = 0.25
        # 0.04 * 0.25 = 0.01
        self.assertAlmostEqual(np.median(scaled_inv_var), 0.01, delta=0.005)

    def test_variance_unbiasing(self):
        """Test removal of signal-dependent Poisson noise."""
        sky = np.full(self.shape, 0.0, dtype=np.float32)
        gain = 1.0
        # Variance = BG_var + Signal/Gain
        # If Signal=100 and BG_var=25, Total_var=125, inv_var=0.008
        # Unbiasing should return inv_var=1/25 = 0.04
        data = np.full(self.shape, 100.0, dtype=np.float32)
        inv_var = np.full(self.shape, 1.0 / 125.0, dtype=np.float32)

        unbiased = _unbias_variance(inv_var, data, sky, gain, 1e-9)
        self.assertAlmostEqual(np.median(unbiased), 0.04, delta=0.001)

    def test_psf_cr_protection(self):
        """Test that star cores are protected from CR detection."""
        # Create a star with FWHM ~ 3.5 (sigma = 1.5)
        x, y = np.mgrid[0 : self.size, 0 : self.size]
        cx, cy = 128, 128
        sigma = 1.5
        star = 5000.0 * np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma**2))
        data = star.astype(np.float32)

        mask = np.zeros(self.shape, dtype=bool)
        config = {
            "sigclip": 4.0,
            "objlim": 1.0,
            "psf_aware": True,
            "psf_fwhm_guess": 3.0,
            "dilate_cr": False,
        }

        # CR detection
        cr_mask = detect_cosmic_rays(data, mask, 65000, 1.5, 5.0, config)

        # The star core should NOT be masked as a CR
        self.assertFalse(cr_mask[128, 128])

    def test_ransac_trails(self):
        """Test detection of sparse trails using RANSAC."""
        data = np.zeros(self.shape, dtype=np.float32)
        # Dotted line
        for i in range(10, 200, 20):
            data[i, i] = 1000.0

        mask = np.zeros(self.shape, dtype=bool)
        config = {
            "enable": True,
            "method": "frangi",  # Run frangi first
            "enable_ransac_trails": True,
            "ransac_params": {
                "min_inliers": 5,
                "min_length": 50,
                "detect_thresh_sig": 5.0,
                # Sparse dotted trails: inlier count / endpoint span is ~0.04 for this fixture
                "min_line_density": 0.03,
            },
            "dilation_radius": 1,
        }

        # Need background RMS for thresholding
        rms = np.full(self.shape, 10.0, dtype=np.float32)

        streak_mask = detect_streaks(data, rms, mask, config)
        self.assertTrue(np.sum(streak_mask) > 0)
        # Check if the line pixels are included
        self.assertTrue(streak_mask[100, 100])

    def test_bleed_trail_growth(self):
        """Test vertical region-growing for bleed trails."""
        data = np.zeros(self.shape, dtype=np.float32)
        sky = np.zeros(self.shape, dtype=np.float32)
        rms = np.full(self.shape, 10.0, dtype=np.float32)

        # Saturated core
        data[100, 100] = 65535.0
        sat_mask = np.zeros(self.shape, dtype=bool)
        sat_mask[100, 100] = True

        # Add bleed trail above and below
        data[90:110, 100] = 1000.0

        config = {
            "mask_bleed_trails": True,
            "bleed_thresh_sigma": 2.0,
            "bleed_grow_horizontal": 0,
        }

        full_mask = grow_bleed_trails(data, sat_mask, sky, rms, config)

        # Should have grown vertically
        self.assertTrue(full_mask[95, 100])
        self.assertTrue(full_mask[105, 100])
        self.assertFalse(full_mask[85, 100])


if __name__ == "__main__":
    unittest.main()
