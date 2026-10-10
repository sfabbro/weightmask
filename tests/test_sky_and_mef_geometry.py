"""Direct unit tests for sky-HDU selection and the MEF geometry/dtype helpers.

`_image_hdus` decides which HDUs of a SKYMESH product are rebuilt, and
`_angle_sep_deg`, `_line_to_common_geometry`, `_wire_dtype` and
`_ensure_hdu_extname` are load-bearing for chip-replica matching and for the
dtype of every written map. They were previously exercised only through the
full pipeline, so a change in their arithmetic showed up as a diffuse product
difference rather than a local failure.
"""

import os
import tempfile
import unittest

import fitsio
import numpy as np

from weightmask.mef import (
    _angle_sep_deg,
    _ensure_hdu_extname,
    _line_to_common_geometry,
    _wire_dtype,
)
from weightmask.reconstruct_sky import _image_hdus


def _wcs(cd=((1.0, 0.0), (0.0, 1.0)), crpix=(0.0, 0.0), scale=0.0002, crval=(10.0, 20.0)):
    return {
        "cd": np.asarray(cd, dtype=np.float64),
        "crpix": np.asarray(crpix, dtype=np.float64),
        "scale_deg_per_px": scale,
        "crval": crval,
    }


class TestAngleSepDeg(unittest.TestCase):
    def test_parallel_and_identical_angles_are_zero(self):
        self.assertEqual(_angle_sep_deg(10.0, 10.0), 0.0)
        self.assertEqual(_angle_sep_deg(10.0, 190.0), 0.0)
        self.assertEqual(_angle_sep_deg(0.0, 180.0), 0.0)

    def test_perpendicular_and_wrapping_cases(self):
        self.assertEqual(_angle_sep_deg(10.0, 100.0), 90.0)
        self.assertEqual(_angle_sep_deg(0.0, 90.0), 90.0)
        self.assertAlmostEqual(_angle_sep_deg(179.0, 0.0), 1.0)
        self.assertAlmostEqual(_angle_sep_deg(10.0, 15.0), 5.0)

    def test_is_symmetric(self):
        for left in (0.0, 17.5, 90.0, 179.0, 359.0):
            for right in (0.0, 12.0, 91.0, 200.0):
                with self.subTest(left=left, right=right):
                    self.assertEqual(_angle_sep_deg(left, right), _angle_sep_deg(right, left))

    def test_is_bounded_to_ninety_degrees(self):
        for left in np.linspace(0.0, 720.0, 145):
            for right in np.linspace(-180.0, 180.0, 37):
                value = _angle_sep_deg(float(left), float(right))
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 90.0)

    def test_accepts_non_float_numbers(self):
        self.assertEqual(_angle_sep_deg(10, 10), 0.0)
        self.assertAlmostEqual(_angle_sep_deg(np.float64(10.0), 14.0), 4.0)


class TestLineToCommonGeometry(unittest.TestCase):
    def test_missing_line_or_wcs_returns_none(self):
        self.assertIsNone(_line_to_common_geometry(None, _wcs()))
        self.assertIsNone(_line_to_common_geometry({"point": [0.0, 0.0], "direction": [1.0, 0.0]}, None))

    def test_horizontal_line_maps_to_a_ninety_degree_normal(self):
        common = _line_to_common_geometry({"point": [10.0, 0.0], "direction": [1.0, 0.0]}, _wcs())
        self.assertAlmostEqual(common["angle_deg"], 90.0)
        np.testing.assert_allclose(common["normal"], [0.0, 1.0])
        self.assertAlmostEqual(common["offset_deg"], 0.0)

    def test_vertical_line_is_canonicalised_to_a_positive_x_normal(self):
        common = _line_to_common_geometry({"point": [3.0, 5.0], "direction": [0.0, 1.0]}, _wcs())
        # The raw normal would be [-1, 0]; canonicalisation flips it and the offset.
        np.testing.assert_allclose(common["normal"], [1.0, 0.0])
        self.assertAlmostEqual(common["angle_deg"], 0.0)
        self.assertAlmostEqual(common["offset_deg"], 3.0)

    def test_offset_is_measured_from_crpix(self):
        line = {"point": [3.0, 5.0], "direction": [0.0, 1.0]}
        self.assertAlmostEqual(_line_to_common_geometry(line, _wcs(crpix=(0.0, 0.0)))["offset_deg"], 3.0)
        self.assertAlmostEqual(_line_to_common_geometry(line, _wcs(crpix=(2.0, 0.0)))["offset_deg"], 1.0)

    def test_reversing_the_direction_leaves_geometry_unchanged(self):
        for direction in ([1.0, 0.0], [0.0, 1.0], [3.0, 4.0], [-2.0, 5.0]):
            point = [7.0, 11.0]
            forward = _line_to_common_geometry({"point": point, "direction": direction}, _wcs())
            backward = _line_to_common_geometry({"point": point, "direction": [-v for v in direction]}, _wcs())
            with self.subTest(direction=direction):
                self.assertAlmostEqual(forward["angle_deg"], backward["angle_deg"])
                self.assertAlmostEqual(forward["offset_deg"], backward["offset_deg"])
                np.testing.assert_allclose(forward["normal"], backward["normal"])

    def test_direction_is_normalised(self):
        common = _line_to_common_geometry({"point": [0.0, 0.0], "direction": [3.0, 4.0]}, _wcs())
        self.assertAlmostEqual(float(np.hypot(*common["normal"])), 1.0)

    def test_angle_stays_in_the_half_open_degree_range(self):
        for direction in ([1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 2.0], [0.0, -1.0]):
            common = _line_to_common_geometry({"point": [1.0, 1.0], "direction": direction}, _wcs())
            self.assertGreaterEqual(common["angle_deg"], 0.0)
            self.assertLess(common["angle_deg"], 180.0)

    def test_degenerate_and_non_finite_transforms_return_none(self):
        zero_cd = _wcs(cd=((0.0, 0.0), (0.0, 0.0)))
        self.assertIsNone(_line_to_common_geometry({"point": [1.0, 1.0], "direction": [1.0, 0.0]}, zero_cd))
        nan_cd = _wcs(cd=((np.nan, 0.0), (0.0, 1.0)))
        self.assertIsNone(_line_to_common_geometry({"point": [1.0, 1.0], "direction": [1.0, 0.0]}, nan_cd))

    def test_wcs_scalars_are_carried_through(self):
        common = _line_to_common_geometry(
            {"point": [1.0, 1.0], "direction": [1.0, 0.0]}, _wcs(scale=0.00031, crval=(1.5, -2.5))
        )
        self.assertEqual(common["scale_deg_per_px"], 0.00031)
        self.assertEqual(common["crval"], (1.5, -2.5))


class TestWireDtype(unittest.TestCase):
    def test_mask_dtypes_follow_bitpix(self):
        self.assertEqual(_wire_dtype(8, is_mask=True), np.uint8)
        self.assertEqual(_wire_dtype(16, is_mask=True), np.uint16)
        self.assertEqual(_wire_dtype(32, is_mask=True), np.uint32)
        self.assertEqual(_wire_dtype(64, is_mask=True), np.uint64)

    def test_float_dtypes_follow_bitpix(self):
        self.assertEqual(_wire_dtype(32, is_mask=False), np.float32)
        self.assertEqual(_wire_dtype(64, is_mask=False), np.float64)
        self.assertEqual(_wire_dtype(-32, is_mask=False), np.float32)
        self.assertEqual(_wire_dtype(-64, is_mask=False), np.float64)

    def test_unmapped_bitpix_falls_back_to_the_documented_default(self):
        self.assertEqual(_wire_dtype(12, is_mask=True), np.uint16)
        self.assertEqual(_wire_dtype(-16, is_mask=False), np.float32)
        self.assertEqual(_wire_dtype(16, is_mask=False), np.float32)
        self.assertEqual(_wire_dtype(None, is_mask=True), np.uint16)
        self.assertEqual(_wire_dtype(None, is_mask=False), np.float32)
        self.assertEqual(_wire_dtype("not a bitpix", is_mask=True), np.uint16)
        self.assertEqual(_wire_dtype("not a bitpix", is_mask=False), np.float32)

    def test_numeric_text_and_float_bitpix_are_coerced(self):
        self.assertEqual(_wire_dtype("32", is_mask=True), np.uint32)
        self.assertEqual(_wire_dtype(16.0, is_mask=True), np.uint16)
        self.assertEqual(_wire_dtype(64.0, is_mask=False), np.float64)


class TestEnsureHduExtname(unittest.TestCase):
    def test_a_missing_extname_is_written(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ext.fits")
            fitsio.write(path, np.zeros((3, 3), dtype=np.float32), clobber=True)
            with fitsio.FITS(path, "rw") as hdul:
                self.assertNotIn("EXTNAME", hdul[0].read_header())
                _ensure_hdu_extname(hdul, 0, "PRIMARY_MAP")
            with fitsio.FITS(path, "r") as hdul:
                self.assertEqual(hdul[0].read_header()["EXTNAME"], "PRIMARY_MAP")

    def test_a_matching_extname_is_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ext.fits")
            fitsio.write(path, np.zeros((2, 2), dtype=np.float32), clobber=True)
            fitsio.write(path, np.ones((2, 2), dtype=np.float32), extname="MAP_A")
            with fitsio.FITS(path, "rw") as hdul:
                _ensure_hdu_extname(hdul, 1, "MAP_A")
            with fitsio.FITS(path, "r") as hdul:
                self.assertEqual(hdul[1].read_header()["EXTNAME"], "MAP_A")

    def test_a_different_extname_is_replaced(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ext.fits")
            fitsio.write(path, np.zeros((2, 2), dtype=np.float32), clobber=True)
            fitsio.write(path, np.ones((2, 2), dtype=np.float32), extname="MAP_A")
            with fitsio.FITS(path, "rw") as hdul:
                _ensure_hdu_extname(hdul, 1, "MAP_B")
            with fitsio.FITS(path, "r") as hdul:
                self.assertEqual(hdul[1].read_header()["EXTNAME"], "MAP_B")

    def test_the_match_is_case_sensitive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ext.fits")
            fitsio.write(path, np.zeros((2, 2), dtype=np.float32), clobber=True)
            fitsio.write(path, np.ones((2, 2), dtype=np.float32), extname="map_a")
            with fitsio.FITS(path, "rw") as hdul:
                _ensure_hdu_extname(hdul, 1, "MAP_A")
            with fitsio.FITS(path, "r") as hdul:
                self.assertEqual(hdul[1].read_header()["EXTNAME"], "MAP_A")


class TestImageHdus(unittest.TestCase):
    def _product(self, tmp):
        path = os.path.join(tmp, "mesh.fits")
        fitsio.write(path, np.zeros((4, 4), dtype=np.float32), clobber=True)
        fitsio.write(path, np.array([(1, 2.0)], dtype=[("a", "i4"), ("b", "f4")]), extname="CAT")
        return path

    def test_only_image_hdus_are_selected_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._product(tmp)
            with fitsio.FITS(path, "r") as hdul:
                self.assertEqual(_image_hdus(hdul, None), [0])

    def test_every_image_extension_is_selected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "two.fits")
            fitsio.write(path, np.zeros((4, 4), dtype=np.float32), clobber=True)
            fitsio.write(path, np.ones((4, 4), dtype=np.float32), extname="MAP_A")
            fitsio.write(path, np.full((4, 4), 2.0, dtype=np.float32), extname="MAP_B")
            with fitsio.FITS(path, "r") as hdul:
                self.assertEqual(_image_hdus(hdul, None), [0, 1, 2])

    def test_an_explicit_image_index_is_returned(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._product(tmp)
            with fitsio.FITS(path, "r") as hdul:
                self.assertEqual(_image_hdus(hdul, 0), [0])

    def test_an_explicit_table_index_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._product(tmp)
            with fitsio.FITS(path, "r") as hdul:
                self.assertEqual(_image_hdus(hdul, 1), [])

    def test_an_explicit_one_dimensional_image_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "oned.fits")
            fitsio.write(path, np.zeros((4, 4), dtype=np.float32), clobber=True)
            fitsio.write(path, np.arange(5, dtype=np.float32), extname="SPEC")
            with fitsio.FITS(path, "r") as hdul:
                self.assertEqual(_image_hdus(hdul, 1), [])

    def test_an_out_of_range_index_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._product(tmp)
            with fitsio.FITS(path, "r") as hdul:
                self.assertEqual(_image_hdus(hdul, 9), [])
                self.assertEqual(_image_hdus(hdul, -1), [])


if __name__ == "__main__":
    unittest.main()
