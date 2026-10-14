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

    def test_tiled_local_mask_matches_full_mask_near_tile_boundaries(self):
        y, x = np.mgrid[:48, :48]
        flat = (1.0 + 0.001 * x + 0.002 * y).astype(np.float32)
        defects = np.zeros(flat.shape, dtype=bool)
        for rows, columns in (
            (slice(5, 8), slice(13, 16)),
            (slice(20, 23), slice(16, 19)),
            (slice(13, 16), slice(5, 8)),
            (slice(16, 19), slice(24, 27)),
        ):
            flat[rows, columns] = 0.2
            defects[rows, columns] = True
        config = {
            "local_filter_size": 5,
            "local_low_thresh": 0.5,
            "local_high_thresh": 2.0,
            "col_enable": False,
        }

        full = detect_bad_pixels(flat, config, using_unit_flat=False)

        self.assertTrue(full[defects].all())
        for tile_size in (8, 16, 24):
            with self.subTest(tile_size=tile_size):
                tiled = compute_flat_bad_mask(flat, config, tile_size=tile_size)
                np.testing.assert_array_equal(tiled, full)

    def test_tiled_column_statistics_cover_the_full_hdu(self):
        flat = np.ones((64, 64), dtype=np.float32)
        flat[:40, 31] = 0.2
        config = {
            "local_low_thresh": 0.0,
            "local_high_thresh": 10.0,
            "col_enable": True,
            "col_deriv_sigma": 3.0,
            "col_dead_thresh": 0.5,
        }

        full = detect_bad_pixels(flat, config, using_unit_flat=False)

        self.assertTrue(full[:, 31].all())
        for tile_size in (8, 16, 32):
            with self.subTest(tile_size=tile_size):
                tiled = compute_flat_bad_mask(flat, config, tile_size=tile_size)
                np.testing.assert_array_equal(tiled, full)

    def test_bad_column_derivative_is_attributed_to_the_deviating_column(self):
        flat = np.broadcast_to(np.linspace(0.9, 1.1, 96, dtype=np.float32), (64, 96)).copy()
        flat[:, 47] = 1.8
        config = {
            "local_low_thresh": 0.0,
            "local_high_thresh": 10.0,
            "col_enable": True,
            "col_deriv_sigma": 5.0,
            "col_dead_thresh": 0.0,
        }

        mask = detect_bad_pixels(flat, config, using_unit_flat=False)

        self.assertTrue(mask[:, 47].all())
        self.assertFalse(mask[:, :47].any())
        self.assertFalse(mask[:, 48:].any())

    def test_bad_column_attribution_handles_adjacent_and_edge_defects(self):
        config = {
            "local_low_thresh": 0.0,
            "local_high_thresh": 10.0,
            "col_enable": True,
            "col_deriv_sigma": 5.0,
            "col_dead_thresh": 0.0,
        }
        for columns, value in (((0,), 1.8), ((95,), 0.2), ((47, 48), 1.8)):
            with self.subTest(columns=columns):
                flat = np.broadcast_to(np.linspace(0.9, 1.1, 96, dtype=np.float32), (64, 96)).copy()
                flat[:, columns] = value
                expected = np.zeros(flat.shape, dtype=bool)
                expected[:, columns] = True

                mask = detect_bad_pixels(flat, config, using_unit_flat=False)

                np.testing.assert_array_equal(mask, expected)

    def test_tiled_local_filter_working_set_is_limited_to_core_plus_halo(self):
        from scipy.ndimage import median_filter

        shapes = []

        def recording_filter(values, *args, **kwargs):
            shapes.append(values.shape)
            return median_filter(values, *args, **kwargs)

        flat = np.ones((48, 48), dtype=np.float32)
        config = {"local_filter_size": 5, "col_enable": False}
        with patch("scipy.ndimage.median_filter", side_effect=recording_filter):
            compute_flat_bad_mask(flat, config, tile_size=16)

        self.assertIn((20, 20), shapes)
        self.assertTrue(all(height <= 20 and width <= 20 for height, width in shapes))

    def test_tiled_mask_allocates_only_one_full_hdu_boolean_array(self):
        flat = np.ones((48, 64), dtype=np.float32)
        original_zeros = np.zeros
        full_hdu_boolean_allocations = []

        def recording_zeros(shape, *args, **kwargs):
            dtype = kwargs.get("dtype", args[0] if args else float)
            allocation_shape = (shape,) if np.isscalar(shape) else tuple(shape)
            if allocation_shape == flat.shape and np.dtype(dtype) == np.dtype(bool):
                full_hdu_boolean_allocations.append(allocation_shape)
            return original_zeros(shape, *args, **kwargs)

        with patch("weightmask.bad.np.zeros", side_effect=recording_zeros):
            compute_flat_bad_mask(flat, {"local_filter_size": 5}, tile_size=16)

        self.assertEqual(full_hdu_boolean_allocations, [flat.shape])

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
