import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from weightmask.cli import process_hdu


class TestProcessHDU(unittest.TestCase):
    def setUp(self):
        # Create a basic configuration
        self.config = {
            "flat_masking": {},
            "saturation": {"mask_bleed_trails": False},
            "sep_background": {"iterations": 1},
            "sep_objects": {},
            "cosmic_ray": {},
            "variance": {},
            "streaks": {"enable": False},
            "output_params": {"output_map_format": "weight"},
        }

        # Mock Science HDU
        self.mock_hdu_sci = MagicMock()
        self.mock_hdu_sci.name = "SCI"
        self.mock_sci_data = np.ones((100, 100), dtype=np.float32)
        self.mock_hdu_sci.read.return_value = self.mock_sci_data
        self.mock_hdu_sci.read_header.return_value = {"GAIN": 1.0, "RDNOISE": 0.0}

        # Mock Flat HDU
        self.mock_hdu_flat = MagicMock()
        self.mock_flat_data = np.ones((100, 100), dtype=np.float32)
        self.mock_hdu_flat.read.return_value = self.mock_flat_data

    @patch("weightmask.cli.detect_bad_pixels")
    @patch("weightmask.cli.detect_saturated_pixels")
    @patch("weightmask.cli.estimate_background")
    @patch("weightmask.cli.detect_cosmic_rays")
    @patch("weightmask.cli.detect_objects")
    @patch("weightmask.cli.calculate_inverse_variance")
    @patch("weightmask.cli.generate_weight_and_confidence")
    def test_process_hdu_happy_path(
        self,
        mock_generate_weight,
        mock_calc_inv_var,
        mock_detect_objs,
        mock_detect_crs,
        mock_est_bkg,
        mock_detect_sat,
        mock_detect_bad,
    ):
        # Setup mocks to return minimal expected values
        mock_detect_bad.return_value = np.zeros((100, 100), dtype=bool)
        mock_detect_sat.return_value = (
            65000.0,
            "mock",
            np.zeros((100, 100), dtype=bool),
        )
        mock_detect_crs.return_value = np.zeros((100, 100), dtype=bool)

        # estimate_background returns (bkg_map, bkg_rms_map)
        mock_est_bkg.return_value = (
            np.zeros((100, 100), dtype=np.float32),
            np.ones((100, 100), dtype=np.float32),
        )

        mock_detect_objs.return_value = np.zeros((100, 100), dtype=bool)

        # calculate_inverse_variance returns (inv_var_map, final_method)
        mock_calc_inv_var.return_value = np.ones((100, 100), dtype=np.float32)

        # generate_weight_and_confidence returns (weight_map, conf_map)
        mock_generate_weight.return_value = (
            np.ones((100, 100), dtype=np.float32),
            np.ones((100, 100), dtype=np.float32),
        )

        # Run process_hdu
        mask_data, inv_var_data, weight_map, confidence_map, sky_map, header_info = process_hdu(
            self.mock_hdu_sci,
            self.mock_hdu_flat,
            self.config,
            hdu_index=1,
            tile_size=100,
        )

        # Assert returns are correct
        self.assertIsNotNone(weight_map)
        self.assertIsNotNone(inv_var_data)
        self.assertIsNotNone(mask_data)
        self.assertIsNotNone(confidence_map)
        self.assertIsNotNone(sky_map)
        self.assertIsNotNone(header_info)

        self.assertEqual(header_info.get("SAT_LVL"), 65000.0)
        self.assertEqual(header_info.get("SAT_METH"), "mock")

    def test_process_hdu_sci_read_error(self):
        # Simulate OSError when reading science data
        self.mock_hdu_sci.read.side_effect = OSError("Mock error")

        mask_data, inv_var_data, weight_map, confidence_map, sky_map, header_info = process_hdu(
            self.mock_hdu_sci, self.mock_hdu_flat, self.config, hdu_index=1
        )

        self.assertIsNone(weight_map)
        self.assertIsNone(inv_var_data)
        self.assertIsNone(mask_data)

    def test_process_hdu_flat_read_error(self):
        # Simulate OSError when reading flat data
        self.mock_hdu_flat.read.side_effect = OSError("Mock flat error")

        mask_data, inv_var_data, weight_map, confidence_map, sky_map, header_info = process_hdu(
            self.mock_hdu_sci, self.mock_hdu_flat, self.config, hdu_index=1
        )

        self.assertIsNone(weight_map)
        self.assertIsNone(inv_var_data)
        self.assertIsNone(mask_data)

    def test_process_hdu_flat_shape_mismatch(self):
        # Simulate flat data with wrong shape
        self.mock_hdu_flat.read.return_value = np.ones((50, 50), dtype=np.float32)

        mask_data, inv_var_data, weight_map, confidence_map, sky_map, header_info = process_hdu(
            self.mock_hdu_sci, self.mock_hdu_flat, self.config, hdu_index=1
        )

        self.assertIsNone(weight_map)
        self.assertIsNone(inv_var_data)
        self.assertIsNone(mask_data)

    @patch("weightmask.cli.detect_bad_pixels")
    @patch("weightmask.cli.detect_saturated_pixels")
    @patch("weightmask.cli.estimate_background")
    @patch("weightmask.cli.detect_cosmic_rays")
    @patch("weightmask.cli.detect_objects")
    @patch("weightmask.cli.calculate_inverse_variance")
    @patch("weightmask.cli.generate_weight_and_confidence")
    def test_process_hdu_no_flat(
        self,
        mock_generate_weight,
        mock_calc_inv_var,
        mock_detect_objs,
        mock_detect_crs,
        mock_est_bkg,
        mock_detect_sat,
        mock_detect_bad,
    ):
        # Setup mocks
        mock_detect_bad.return_value = np.zeros((100, 100), dtype=bool)
        mock_detect_sat.return_value = (
            65000.0,
            "mock",
            np.zeros((100, 100), dtype=bool),
        )
        mock_detect_crs.return_value = np.zeros((100, 100), dtype=bool)
        mock_est_bkg.return_value = (
            np.zeros((100, 100), dtype=np.float32),
            np.ones((100, 100), dtype=np.float32),
        )
        mock_detect_objs.return_value = np.zeros((100, 100), dtype=bool)
        mock_calc_inv_var.return_value = np.ones((100, 100), dtype=np.float32)
        mock_generate_weight.return_value = (
            np.ones((100, 100), dtype=np.float32),
            np.ones((100, 100), dtype=np.float32),
        )

        # Run process_hdu with hdu_flat = None
        mask_data, inv_var_data, weight_map, confidence_map, sky_map, header_info = process_hdu(
            self.mock_hdu_sci, None, self.config, hdu_index=1, tile_size=100
        )

        # Assert returns are correct
        self.assertIsNotNone(weight_map)
        self.assertIsNotNone(mask_data)

        # Verify using_unit_flat flag logic
        # When flat is None, using_unit_flat is True
        mock_detect_bad.assert_called()
        self.assertTrue(mock_detect_bad.call_args[0][2])  # The third argument to detect_bad_pixels is using_unit_flat

    @patch("weightmask.cli.detect_bad_pixels")
    @patch("weightmask.cli.detect_saturated_pixels")
    @patch("weightmask.cli.estimate_background")
    @patch("weightmask.cli.detect_cosmic_rays")
    @patch("weightmask.cli.detect_objects")
    @patch("weightmask.cli.calculate_inverse_variance")
    @patch("weightmask.cli.generate_weight_and_confidence")
    @patch("weightmask.cli.detect_streaks")
    def test_process_hdu_with_streaks(
        self,
        mock_detect_streaks,
        mock_generate_weight,
        mock_calc_inv_var,
        mock_detect_objs,
        mock_detect_crs,
        mock_est_bkg,
        mock_detect_sat,
        mock_detect_bad,
    ):
        # Enable streak masking
        self.config["streak_masking"] = {"enable": True}

        # Setup mocks
        mock_detect_bad.return_value = np.zeros((100, 100), dtype=bool)
        mock_detect_sat.return_value = (
            65000.0,
            "mock",
            np.zeros((100, 100), dtype=bool),
        )
        mock_detect_crs.return_value = np.zeros((100, 100), dtype=bool)
        mock_est_bkg.return_value = (
            np.zeros((100, 100), dtype=np.float32),
            np.ones((100, 100), dtype=np.float32),
        )
        mock_detect_objs.return_value = np.zeros((100, 100), dtype=bool)
        mock_calc_inv_var.return_value = np.ones((100, 100), dtype=np.float32)
        mock_detect_streaks.return_value = np.zeros((100, 100), dtype=bool)
        mock_generate_weight.return_value = (
            np.ones((100, 100), dtype=np.float32),
            np.ones((100, 100), dtype=np.float32),
        )

        # Run process_hdu
        mask_data, inv_var_data, weight_map, confidence_map, sky_map, header_info = process_hdu(
            self.mock_hdu_sci,
            self.mock_hdu_flat,
            self.config,
            hdu_index=1,
            tile_size=100,
        )

        # Assert streaks were detected
        mock_detect_streaks.assert_called_once()
        self.assertIsNotNone(weight_map)

    @patch("weightmask.cli.detect_bad_pixels")
    @patch("weightmask.cli.detect_saturated_pixels")
    @patch("weightmask.cli.estimate_background")
    @patch("weightmask.cli.detect_cosmic_rays")
    @patch("weightmask.cli.detect_objects")
    @patch("weightmask.cli.calculate_inverse_variance")
    @patch("weightmask.cli.generate_weight_and_confidence")
    @patch("weightmask.cli.grow_bleed_trails")
    def test_process_hdu_with_bleed_trails(
        self,
        mock_grow_bleed,
        mock_generate_weight,
        mock_calc_inv_var,
        mock_detect_objs,
        mock_detect_crs,
        mock_est_bkg,
        mock_detect_sat,
        mock_detect_bad,
    ):
        # Enable bleed trail masking
        self.config["saturation"]["mask_bleed_trails"] = True

        # Setup mocks
        mock_detect_bad.return_value = np.zeros((100, 100), dtype=bool)
        mock_detect_sat.return_value = (
            65000.0,
            "mock",
            np.zeros((100, 100), dtype=bool),
        )
        mock_detect_crs.return_value = np.zeros((100, 100), dtype=bool)
        mock_est_bkg.return_value = (
            np.zeros((100, 100), dtype=np.float32),
            np.ones((100, 100), dtype=np.float32),
        )
        mock_detect_objs.return_value = np.zeros((100, 100), dtype=bool)
        mock_calc_inv_var.return_value = np.ones((100, 100), dtype=np.float32)
        mock_grow_bleed.return_value = np.zeros((100, 100), dtype=bool)
        mock_generate_weight.return_value = (
            np.ones((100, 100), dtype=np.float32),
            np.ones((100, 100), dtype=np.float32),
        )

        # Run process_hdu
        mask_data, inv_var_data, weight_map, confidence_map, sky_map, header_info = process_hdu(
            self.mock_hdu_sci,
            self.mock_hdu_flat,
            self.config,
            hdu_index=1,
            tile_size=100,
        )

        # Assert bleed trails were grown
        mock_grow_bleed.assert_called_once()
        self.assertIsNotNone(weight_map)

    @patch("weightmask.cli.detect_bad_pixels")
    @patch("weightmask.cli.detect_saturated_pixels")
    @patch("weightmask.cli.estimate_background")
    @patch("weightmask.cli.detect_cosmic_rays")
    @patch("weightmask.cli.detect_objects")
    @patch("weightmask.cli.calculate_inverse_variance")
    @patch("weightmask.cli.generate_weight_and_confidence")
    def test_process_hdu_none_returns(
        self,
        mock_generate_weight,
        mock_calc_inv_var,
        mock_detect_objs,
        mock_detect_crs,
        mock_est_bkg,
        mock_detect_sat,
        mock_detect_bad,
    ):
        # Setup mocks to return minimal expected values
        mock_detect_bad.return_value = np.zeros((100, 100), dtype=bool)
        mock_detect_sat.return_value = (
            65000.0,
            "mock",
            np.zeros((100, 100), dtype=bool),
        )
        mock_detect_crs.return_value = np.zeros((100, 100), dtype=bool)

        # estimate_background returns None to trigger early return
        mock_est_bkg.return_value = (None, None)

        # Run process_hdu
        mask_data, inv_var_data, weight_map, confidence_map, sky_map, header_info = process_hdu(
            self.mock_hdu_sci,
            self.mock_hdu_flat,
            self.config,
            hdu_index=1,
            tile_size=100,
        )

        self.assertIsNone(mask_data)
        self.assertIsNone(weight_map)


if __name__ == "__main__":
    unittest.main()
