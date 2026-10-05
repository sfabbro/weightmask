import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import fitsio
import numpy as np
import yaml

from weightmask.bad import compute_flat_bad_mask, detect_bad_pixels, detect_dark_hot_pixels
from weightmask.process import process_image


class TestBadPixels(unittest.TestCase):
    def test_detect_bad_pixels_with_flat_data(self):
        """Test bad pixel detection with realistic flat field data."""
        # Create test flat field data
        flat_data = np.ones((100, 100), dtype=np.float32)

        # Add some bad pixels
        flat_data[10, 10] = 0.3  # Below threshold
        flat_data[20, 20] = 2.5  # Above threshold
        flat_data[30, 30] = np.nan  # NaN value
        flat_data[40, 40] = np.inf  # Inf value

        config = {"low_thresh": 0.5, "high_thresh": 2.0, "col_enable": False}

        mask = detect_bad_pixels(flat_data, config, using_unit_flat=False)

        # Check that bad pixels are correctly identified
        self.assertTrue(mask[10, 10])  # Low value
        self.assertTrue(mask[20, 20])  # High value
        self.assertTrue(mask[30, 30])  # NaN
        self.assertTrue(mask[40, 40])  # Inf
        self.assertFalse(mask[50, 50])  # Normal pixel

    def test_detect_bad_pixels_with_unit_flat(self):
        """Test bad pixel detection with unit flat (should skip pixel thresholding)."""
        flat_data = np.ones((100, 100), dtype=np.float32)
        flat_data[10, 10] = 0.3  # This should be ignored with unit flat

        config = {"low_thresh": 0.5, "high_thresh": 2.0, "col_enable": False}

        mask = detect_bad_pixels(flat_data, config, using_unit_flat=True)

        # With unit flat, no pixel thresholding should occur
        self.assertEqual(np.sum(mask), 0)

    def test_detect_bad_pixels_column_detection(self):
        """Test bad column detection."""
        flat_data = np.ones((100, 100), dtype=np.float32)

        # Create a bad column (low variance)
        flat_data[:, 50] = 0.1

        config = {
            "local_low_thresh": 0.5,
            "local_high_thresh": 2.0,
            "col_enable": True,
            "col_deriv_sigma": 5.0,
            "col_dead_thresh": 0.1,
        }

        mask = detect_bad_pixels(flat_data, config, using_unit_flat=False)

        # Check that the bad column is identified
        self.assertTrue(np.all(mask[:, 50]))

    def test_detect_bad_pixels_no_finite_data(self):
        """Test bad pixel detection with no finite data."""
        flat_data = np.full((100, 100), np.nan, dtype=np.float32)

        config = {"low_thresh": 0.5, "high_thresh": 2.0, "col_enable": False}

        mask = detect_bad_pixels(flat_data, config, using_unit_flat=False)

        # No usable flat response means every pixel is bad.
        self.assertTrue(mask.all())

    def test_nonfinite_flat_tiles_remain_bad(self):
        flat = np.ones((8, 8), dtype=np.float32)
        flat[:4, :4] = np.inf
        mask = compute_flat_bad_mask(flat, {"col_enable": False}, tile_size=4)
        self.assertTrue(mask[:4, :4].all())
        self.assertFalse(mask[4:, 4:].any())

    def test_detect_bad_pixels_empty_config(self):
        """Test bad pixel detection with empty configuration."""
        flat_data = np.ones((100, 100), dtype=np.float32)
        flat_data[10, 10] = 0.3

        config = {}

        mask = detect_bad_pixels(flat_data, config, using_unit_flat=False)

        # Should use default thresholds
        self.assertTrue(mask[10, 10])

    def test_nonfinite_science_has_no_data_and_zero_weight(self):
        from weightmask import MASK_BITS

        data = np.random.default_rng(0).normal(100, 2, (32, 32)).astype(np.float32)
        data[0, :3] = [np.nan, np.inf, -np.inf]
        config = {
            "cosmic_ray": {"niter": 0, "faint_cr": {"enable": False}},
            "sep_background": {"iterations": 1},
            "sep_objects": {"extract_thresh": 1e6},
            "saturation": {"mask_bleed_trails": False},
        }
        mask, ivar, weight, _confidence, _sky, info = process_image(data, {}, None, config)
        self.assertTrue(np.all(mask[0, :3] & MASK_BITS["NO_DATA"]))
        self.assertTrue(info["individual_masks"]["nodata"][0, :3].all())
        self.assertTrue(np.all(weight[0, :3] == 0))
        self.assertTrue(np.isfinite(ivar).all())
        self.assertGreater(weight[16, 16], 0)

    def test_science_requires_a_nonempty_image(self):
        for data in (np.array(1), np.zeros(3), np.zeros((0, 3)), np.zeros((2, 2, 2))):
            with self.subTest(shape=data.shape), self.assertRaisesRegex(ValueError, "nonempty 2-D"):
                process_image(data, {}, None, {})

    def test_integer_science_preserves_saturation_full_scale(self):
        from weightmask import MASK_BITS
        from weightmask.satur import detect_saturated_pixels

        with open("weightmask.yml") as stream:
            config = yaml.safe_load(stream)
        config["saturation"]["mask_bleed_trails"] = False
        config["cosmic_ray"] = {"niter": 0, "faint_cr": {"enable": False}}
        config["sep_background"]["iterations"] = 0
        config["streak_masking"]["enable"] = False
        for dtype in (np.uint8, np.int16, np.uint16):
            with self.subTest(dtype=dtype):
                data = np.full((32, 32), 100, dtype=dtype)
                data[10:20, 10:20] = np.iinfo(dtype).max
                expected_level, expected_method, expected_mask = detect_saturated_pixels(data, {}, config["saturation"])
                self.assertEqual(np.count_nonzero(expected_mask), 100)
                mask, _ivar, weight, _confidence, _sky, info = process_image(data, {}, None, config)
                np.testing.assert_array_equal((mask & MASK_BITS["SAT"]) != 0, expected_mask)
                self.assertEqual(info["SAT_LVL"], expected_level)
                self.assertEqual(info["SAT_METH"], expected_method)
                self.assertTrue(np.all(weight[expected_mask] == 0))

    def test_float32_pipeline_saturation_matches_direct_detection(self):
        from weightmask import MASK_BITS
        from weightmask.satur import detect_saturated_pixels

        with open("weightmask.yml") as stream:
            config = yaml.safe_load(stream)
        config["saturation"]["mask_bleed_trails"] = False
        config["cosmic_ray"] = {"niter": 0, "faint_cr": {"enable": False}}
        config["sep_background"]["iterations"] = 0
        config["streak_masking"]["enable"] = False
        data = np.random.default_rng(0).normal(100.0, 2.0, (32, 32)).astype(np.float32)
        data[10:20, 10:20] = 65535.0
        expected_level, expected_method, expected_mask = detect_saturated_pixels(data, {}, config["saturation"])
        mask, _ivar, _weight, _confidence, _sky, info = process_image(data, {}, None, config)
        np.testing.assert_array_equal((mask & MASK_BITS["SAT"]) != 0, expected_mask)
        self.assertEqual(info["SAT_LVL"], expected_level)
        self.assertEqual(info["SAT_METH"], expected_method)

    def test_integer_bad_masks_are_pixel_masks(self):
        data = np.random.default_rng(0).normal(100, 2, (32, 32)).astype(np.float32)
        supplied = np.zeros(data.shape, dtype=np.uint8)
        supplied[5, 5] = 1
        original = supplied.copy()
        config = {
            "cosmic_ray": {"niter": 0, "faint_cr": {"enable": False}},
            "sep_background": {"iterations": 0},
            "saturation": {"mask_bleed_trails": False},
        }
        for name in ("bad_mask", "badpix_mask"):
            with self.subTest(name=name):
                mask, _ivar, weight, _confidence, _sky, info = process_image(data, {}, None, config, **{name: supplied})
                np.testing.assert_array_equal(info["individual_masks"]["bad"], supplied.astype(bool))
                self.assertEqual(weight[5, 5], 0)
                self.assertGreater(weight[16, 16], 0)
                np.testing.assert_array_equal(supplied, original)
                self.assertEqual(mask.shape, supplied.shape)

    def test_bad_mask_shape_must_match_science(self):
        for name in ("bad_mask", "badpix_mask"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "mask shape"):
                process_image(np.ones((32, 32)), {}, None, {}, **{name: np.zeros(32, dtype=bool)})

    def test_zero_dark_only_masks_positive_hot_pixels(self):
        from weightmask.mef import process_all_hdus

        with TemporaryDirectory() as directory:
            science_path, dark_path = (str(Path(directory) / name) for name in ("science.fits", "dark.fits"))
            image = np.ones((32, 32), dtype=np.float32)
            fitsio.write(science_path, image)
            for hot in (False, True):
                dark = np.zeros_like(image)
                if hot:
                    dark[5, 5] = 100
                fitsio.write(dark_path, dark, clobber=True)
                with fitsio.FITS(science_path) as science, fitsio.FITS(dark_path) as dark_hdul:
                    with patch("weightmask.mef.process_hdu", return_value=(None,) * 6) as process:
                        process_all_hdus(
                            [0],
                            science,
                            None,
                            {},
                            {},
                            Namespace(max_workers=1),
                            hdul_dark=dark_hdul,
                        )
                actual = process.call_args.kwargs["precomputed_bad_mask"]
                expected = np.zeros(image.shape, dtype=bool)
                expected[5, 5] = hot
                np.testing.assert_array_equal(actual, expected)

    def test_dark_hot_mask_ignores_pedestal_and_flags_nonfinite_values(self):
        data = np.random.default_rng(0).normal(0, 1, (32, 32)).astype(np.float32)
        data[5, 5] = 100
        data[3, 3] = np.nan
        expected = np.zeros(data.shape, dtype=bool)
        expected[5, 5] = expected[3, 3] = True
        for pedestal in (0, 100, -100):
            np.testing.assert_array_equal(detect_dark_hot_pixels(data + pedestal, {}), expected)

    def test_dark_configuration_rejects_unknown_keys_and_invalid_sigma(self):
        for config in (
            {"col_enable": True},
            {"hot_sigma": 0},
            {"hot_sigma": np.nan},
            {"hot_sigma": np.inf},
            {"hot_sigma": True},
            {"hot_sigma": False},
        ):
            with self.subTest(config=config), self.assertRaises(ValueError):
                detect_dark_hot_pixels(np.zeros((8, 8)), config)


if __name__ == "__main__":
    unittest.main()
