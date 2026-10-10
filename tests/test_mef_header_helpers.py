"""Direct unit tests for the header, mapping and WCS helpers in mef.py.

These decide what provenance an output product carries and which pixels reach
the output map set, so they are pinned at the boundary rather than through a
full multi-HDU run.
"""

import json
import pathlib
import unittest

import numpy as np

from weightmask._version import __version__
from weightmask.contract import _json_from_header_cards
from weightmask.mef import (
    _INDIVIDUAL_MASK_KEYS,
    _PRODUCT_PATH_KEYS,
    _assign_map_if_valid,
    _common_wcs_geometry,
    _header_with_contract_metadata,
    _ordered_product_paths,
    _strip_compression_keywords,
)

_COMPRESSION_CARDS = {
    "BZERO": 32668.0,
    "BSCALE": 1.0,
    "BITPIX": 16,
    "ZQUANTIZ": "SUBTRACTIVE_DITHER_1",
    "ZBLANK": -2147483648,
    "ZSIMPLE": True,
    "ZCMPTYPE": "RICE_1",
    "ZNAXIS": 2,
    "ZBITPIX": 16,
    "ZDITHER0": 7,
    "ZTILE1": 100,
    "ZNAME1": "BLOCKSIZE",
    "ZVAL1": 32,
}


def _tan_header(cdelt=True):
    header = {
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRVAL1": 10.0,
        "CRVAL2": 20.0,
        "CRPIX1": 100.0,
        "CRPIX2": 200.0,
    }
    if cdelt:
        header.update({"CDELT1": -0.0002, "CDELT2": 0.0002})
    return header


class TestStripCompressionKeywords(unittest.TestCase):
    def test_none_passes_through(self):
        self.assertIsNone(_strip_compression_keywords(None))

    def test_every_fpack_card_is_removed(self):
        stripped = _strip_compression_keywords({**_COMPRESSION_CARDS, "OBJECT": "M31", "NAXIS": 2})
        for name in _COMPRESSION_CARDS:
            self.assertNotIn(name, stripped, name)
        self.assertEqual(stripped, {"OBJECT": "M31", "NAXIS": 2})

    def test_the_input_header_is_not_mutated(self):
        original = {**_COMPRESSION_CARDS, "OBJECT": "M31"}
        before = dict(original)
        _strip_compression_keywords(original)
        self.assertEqual(original, before)

    def test_an_uncompressed_header_is_returned_unchanged(self):
        clean = {"NAXIS": 2, "OBJECT": "M31", "EXPTIME": 120.0}
        self.assertEqual(_strip_compression_keywords(clean), clean)

    def test_only_z_prefixed_cards_are_swept(self):
        # A real science keyword that merely starts with Z must survive; only
        # the tile-compression families (ZTILE/ZNAME/ZVAL) are removed.
        stripped = _strip_compression_keywords({"ZODIAC": "yes", "ZTILE1": 100, "EXPTIME": 1.0})
        self.assertEqual(stripped, {"ZODIAC": "yes", "EXPTIME": 1.0})


class TestOrderedProductPaths(unittest.TestCase):
    def test_full_set_uses_the_declared_product_order(self):
        paths = {key: f"/tmp/{key}.fits" for key in _PRODUCT_PATH_KEYS}
        paths["individual_mask_paths"] = {key: f"/tmp/mask_{key}.fits" for key in _INDIVIDUAL_MASK_KEYS}
        ordered = _ordered_product_paths(paths)
        expected = [(key, f"/tmp/{key}.fits") for key in _PRODUCT_PATH_KEYS]
        expected += [(f"individual_mask_paths.{key}", f"/tmp/mask_{key}.fits") for key in _INDIVIDUAL_MASK_KEYS]
        self.assertEqual(ordered, expected)

    def test_missing_products_are_skipped_with_order_kept(self):
        ordered = _ordered_product_paths({"out_mask_path": "/tmp/mask.fits", "out_sky_path": "/tmp/sky.fits"})
        self.assertEqual(ordered, [("out_mask_path", "/tmp/mask.fits"), ("out_sky_path", "/tmp/sky.fits")])

    def test_falsy_entries_are_dropped(self):
        ordered = _ordered_product_paths({"out_map_path": "", "out_mask_path": None, "out_sky_path": "/tmp/sky.fits"})
        self.assertEqual(ordered, [("out_sky_path", "/tmp/sky.fits")])

    def test_individual_masks_are_dropped_when_the_mapping_is_absent(self):
        ordered = _ordered_product_paths({"out_map_path": "/tmp/map.fits"})
        self.assertEqual(ordered, [("out_map_path", "/tmp/map.fits")])

    def test_empty_input_yields_no_products(self):
        self.assertEqual(_ordered_product_paths({}), [])

    def test_path_objects_are_stringified(self):
        ordered = _ordered_product_paths({"out_map_path": pathlib.Path("/tmp/map.fits")})
        self.assertEqual(ordered, [("out_map_path", "/tmp/map.fits")])


class TestAssignMapIfValid(unittest.TestCase):
    def test_none_is_skipped_without_creating_an_entry(self):
        output_data = {}
        _assign_map_if_valid(output_data, 0, "mask", None, {"A": 1}, "HDU1")
        self.assertEqual(output_data, {})

    def test_a_present_map_is_stored_with_its_header_and_name(self):
        output_data = {}
        data = np.ones((4, 4), dtype=np.float32)
        header = {"A": 1}
        _assign_map_if_valid(output_data, 2, "mask", data, header, "8341-7-5")
        self.assertEqual(set(output_data), {2})
        self.assertEqual(set(output_data[2]), {"mask"})
        entry = output_data[2]["mask"]
        self.assertIs(entry["data"], data)
        self.assertIs(entry["header"], header)
        self.assertEqual(entry["name"], "MASK_8341-7-5")

    def test_names_are_upper_cased_for_every_product_key(self):
        output_data = {}
        for key in ("weight_raw", "invvar", "sky"):
            _assign_map_if_valid(output_data, 0, key, np.zeros(2), None, "HDU3")
        self.assertEqual(output_data[0]["weight_raw"]["name"], "WEIGHT_RAW_HDU3")
        self.assertEqual(output_data[0]["invvar"]["name"], "INVVAR_HDU3")
        self.assertEqual(output_data[0]["sky"]["name"], "SKY_HDU3")

    def test_existing_entries_for_an_hdu_are_preserved(self):
        output_data = {1: {"mask": {"data": "kept", "header": None, "name": "MASK_H1"}}}
        _assign_map_if_valid(output_data, 1, "sky", np.zeros(2), None, "H1")
        self.assertEqual(set(output_data[1]), {"mask", "sky"})
        self.assertEqual(output_data[1]["mask"]["data"], "kept")

    def test_shapes_and_finiteness_are_not_validated_here(self):
        # The name says "if valid", but this helper only gates on presence;
        # shape/finiteness are enforced later by WeightMaskProduct. A wrong
        # shape or a non-finite map is therefore stored unchanged, and that
        # behaviour is pinned so a later move of the checks is deliberate.
        output_data = {}
        wrong_shape = np.zeros((2, 2), dtype=np.float32)
        non_finite = np.full((3, 3), np.nan, dtype=np.float32)
        _assign_map_if_valid(output_data, 0, "mask", wrong_shape, None, "HDU1")
        _assign_map_if_valid(output_data, 0, "sky", non_finite, None, "HDU1")
        self.assertEqual(output_data[0]["mask"]["data"].shape, (2, 2))
        self.assertTrue(np.isnan(output_data[0]["sky"]["data"]).all())


class TestHeaderWithContractMetadata(unittest.TestCase):
    def test_required_contract_cards_are_written(self):
        header = _header_with_contract_metadata({"OBJECT": "M31"}, "quality_mask")
        self.assertEqual(header["WMART"], "quality_mask")
        self.assertEqual(header["WMVERS"], "1.0")
        self.assertEqual(header["OBJECT"], "M31")

    def test_mask_polarity_and_semantics_are_optional(self):
        plain = _header_with_contract_metadata({}, "weight")
        self.assertNotIn("WMMASK", plain)
        self.assertNotIn("WMSEM", plain)

        flagged = _header_with_contract_metadata({}, "quality_mask", mask=True, semantics="boolean_mask")
        self.assertEqual(flagged["WMMASK"], "set_means_flagged")
        self.assertEqual(flagged["WMSEM"], "boolean_mask")

    def test_provenance_carries_the_package_version_and_stage(self):
        header = _header_with_contract_metadata({}, "weight")
        payload = json.loads(_json_from_header_cards(header, "WMPROV", "WMPV"))
        self.assertEqual(payload["producer"]["version"], __version__)
        self.assertEqual(payload["provenance"]["producer_stage"], "weightmask.cli")

    def test_input_header_values_survive(self):
        header = _header_with_contract_metadata({"EXPTIME": 120.0, "NAXIS": 2}, "weight")
        self.assertEqual(header["EXPTIME"], 120.0)
        self.assertEqual(header["NAXIS"], 2)

    def test_none_and_unconvertible_headers_still_produce_contract_cards(self):
        for bad in (None, 5, "ab"):
            with self.subTest(header=bad):
                header = _header_with_contract_metadata(bad, "weight")
                self.assertEqual(header["WMART"], "weight")


class TestCommonWcsGeometry(unittest.TestCase):
    def test_none_header_has_no_geometry(self):
        self.assertIsNone(_common_wcs_geometry(None))

    def test_a_cdelt_tan_wcs_is_measured(self):
        geometry = _common_wcs_geometry(_tan_header())
        self.assertIsNotNone(geometry)
        np.testing.assert_allclose(geometry["crval"], [10.0, 20.0])
        # crpix is converted from FITS 1-based to 0-based pixels.
        np.testing.assert_allclose(geometry["crpix"], [99.0, 199.0])
        np.testing.assert_allclose(geometry["cd"], [[-0.0002, 0.0], [0.0, 0.0002]])
        self.assertAlmostEqual(geometry["scale_deg_per_px"], 0.0002)

    def test_a_cd_matrix_tan_wcs_is_measured(self):
        header = _tan_header(cdelt=False)
        header.update({"CD1_1": 1e-4, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": 1e-4})
        geometry = _common_wcs_geometry(header)
        self.assertIsNotNone(geometry)
        np.testing.assert_allclose(geometry["cd"], [[1e-4, 0.0], [0.0, 1e-4]])
        self.assertAlmostEqual(geometry["scale_deg_per_px"], 1e-4)

    def test_a_header_without_cd_or_cdelt_has_no_geometry(self):
        header = _tan_header(cdelt=False)
        self.assertIsNone(_common_wcs_geometry(header))

    def test_a_non_celestial_wcs_has_no_geometry(self):
        header = {"CTYPE1": "X", "CTYPE2": "Y", "CRVAL1": 0.0, "CRVAL2": 0.0, "CRPIX1": 1.0, "CRPIX2": 1.0}
        header.update({"CDELT1": 1.0, "CDELT2": 1.0})
        self.assertIsNone(_common_wcs_geometry(header))

    def test_a_non_tan_projection_is_refused(self):
        header = _tan_header()
        header["CTYPE1"] = "RA---SIN"
        self.assertIsNone(_common_wcs_geometry(header))

    def test_a_degenerate_matrix_is_refused(self):
        header = _tan_header(cdelt=False)
        header.update({"CD1_1": 0.0, "CD1_2": 0.0, "CD2_1": 0.0, "CD2_2": 0.0})
        self.assertIsNone(_common_wcs_geometry(header))

    def test_a_non_numeric_card_is_refused_rather_than_raised(self):
        header = _tan_header()
        header["CDELT1"] = "not a number"
        self.assertIsNone(_common_wcs_geometry(header))


if __name__ == "__main__":
    unittest.main()
