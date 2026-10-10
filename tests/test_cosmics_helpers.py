"""Direct unit tests for the cosmic-ray helpers in ``cosmics.py``.

These helpers gate the CR mask but were previously reached only through
``detect_cosmic_rays``. Pinning them directly means a change in the crowding
rule, the dilation footprint or the axis-length spelling fails on its own line
instead of as a diffuse recall shift.
"""

import unittest

import numpy as np
from skimage.measure import label, regionprops

from weightmask.cosmics import (
    _adjust_dynamic_objlim,
    _apply_morphological_dilation,
    _axis_lengths,
    _header_value,
)


class _RaisingHeader:
    """A header whose ``get`` exists but raises, as a corrupt FITS header can."""

    def get(self, key, default=None):
        raise RuntimeError("header is corrupt")


class TestHeaderValue(unittest.TestCase):
    def test_a_present_keyword_is_returned(self):
        self.assertEqual(_header_value({"SEEING": 1.2}, "SEEING"), 1.2)

    def test_an_absent_keyword_returns_none(self):
        self.assertIsNone(_header_value({"SEEING": 1.2}, "PIXSCALE"))

    def test_a_none_header_returns_none(self):
        self.assertIsNone(_header_value(None, "SEEING"))

    def test_a_header_without_a_callable_get_returns_none(self):
        for header in (5, "SEEING=1.2", object(), ["SEEING", 1.2]):
            with self.subTest(header=repr(header)):
                self.assertIsNone(_header_value(header, "SEEING"))

    def test_a_get_that_raises_is_swallowed(self):
        self.assertIsNone(_header_value(_RaisingHeader(), "SEEING"))


class TestAdjustDynamicObjlim(unittest.TestCase):
    def test_disabling_the_rule_keeps_the_default(self):
        mask = np.ones((10, 10), dtype=bool)
        self.assertEqual(_adjust_dynamic_objlim({"dynamic_objlim": False}, mask, 5.0), 5.0)

    def test_a_missing_mask_keeps_the_default(self):
        for config in ({}, {"dynamic_objlim": True}):
            with self.subTest(config=config):
                self.assertEqual(_adjust_dynamic_objlim(config, None, 7.5), 7.5)

    def test_an_empty_mask_keeps_the_default(self):
        self.assertEqual(_adjust_dynamic_objlim({}, np.zeros((10, 10), dtype=bool), 5.0), 5.0)

    def test_crowding_scales_the_limit_four_times_the_coverage(self):
        mask = np.zeros(100, dtype=bool)
        mask[:25] = True  # coverage 0.25 -> factor 1 + 1.0
        self.assertAlmostEqual(_adjust_dynamic_objlim({}, mask, 4.0), 8.0)

    def test_the_boost_is_capped_at_two_and_a_half(self):
        half = np.zeros(100, dtype=bool)
        half[:50] = True  # coverage 0.5 -> 4 * 0.5 would be 2.0, capped at 1.5
        self.assertAlmostEqual(_adjust_dynamic_objlim({}, half, 4.0), 10.0)
        self.assertAlmostEqual(_adjust_dynamic_objlim({}, np.ones((10, 10), dtype=bool), 4.0), 10.0)

    def test_a_tiny_coverage_still_moves_the_limit(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[0, 0] = True  # coverage 1e-4 -> factor 1 + 4e-4
        self.assertAlmostEqual(_adjust_dynamic_objlim({}, mask, 5.0), 5.0 * (1.0 + 0.0004))

    def test_the_result_is_a_plain_float(self):
        self.assertIsInstance(_adjust_dynamic_objlim({}, np.zeros((4, 4), dtype=bool), 5), float)


class TestApplyMorphologicalDilation(unittest.TestCase):
    def test_disabling_dilation_returns_the_same_array(self):
        mask = np.zeros((20, 20), dtype=bool)
        mask[10, 10] = True
        result = _apply_morphological_dilation(mask, {"dilate_cr": False})
        self.assertIs(result, mask)
        self.assertEqual(int(result.sum()), 1)

    def test_an_empty_mask_stays_empty(self):
        result = _apply_morphological_dilation(np.zeros((20, 20), dtype=bool), {})
        self.assertFalse(result.any())

    def test_dilation_grows_the_footprint_without_mutating_the_input(self):
        mask = np.zeros((21, 21), dtype=bool)
        mask[10, 10] = True
        result = _apply_morphological_dilation(mask, {"dilation_radius": 1})
        self.assertGreater(int(result.sum()), int(mask.sum()))
        self.assertTrue(result[10, 10])
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            self.assertTrue(result[10 + dy, 10 + dx], f"orthogonal neighbour {(dy, dx)} not grown")
        # The caller's array must not be written in place.
        self.assertEqual(int(mask.sum()), 1)
        self.assertTrue(mask[10, 10])

    def test_dilation_is_a_superset_of_its_input(self):
        mask = np.zeros((30, 30), dtype=bool)
        mask[3, 4] = True
        mask[10:12, 20:24] = True
        mask[28, 0] = True
        for radius in (1, 2, 3):
            with self.subTest(radius=radius):
                result = _apply_morphological_dilation(mask, {"dilation_radius": radius})
                self.assertTrue(np.all(result[mask]), "dilation dropped an input pixel")
                self.assertGreaterEqual(int(result.sum()), int(mask.sum()))

    def test_a_zero_radius_leaves_the_mask_unchanged(self):
        mask = np.zeros((11, 11), dtype=bool)
        mask[5, 5] = True
        result = _apply_morphological_dilation(mask, {"dilation_radius": 0})
        self.assertEqual(int(result.sum()), int(mask.sum()))
        self.assertTrue(result[5, 5])


def _single_region(mask):
    regions = regionprops(label(np.ascontiguousarray(mask).astype(np.uint8), connectivity=2))
    if len(regions) != 1:
        raise AssertionError(f"fixture should label exactly one region, got {len(regions)}")
    return regions[0]


class TestAxisLengths(unittest.TestCase):
    def test_returns_the_skimage_attributes_as_python_floats(self):
        region = _single_region(np.ones((1, 1), dtype=bool))
        major, minor = _axis_lengths(region)
        self.assertIsInstance(major, float)
        self.assertIsInstance(minor, float)
        self.assertEqual(major, float(region.axis_major_length))
        self.assertEqual(minor, float(region.axis_minor_length))

    def test_symmetric_components_report_equal_axes(self):
        plus = np.zeros((3, 3), dtype=bool)
        plus[1, 1] = True
        plus[0, 1] = plus[2, 1] = plus[1, 0] = plus[1, 2] = True
        for name, mask in (("2x2", np.ones((2, 2), dtype=bool)), ("plus", plus)):
            with self.subTest(component=name):
                major, minor = _axis_lengths(_single_region(mask))
                self.assertEqual(major, minor, f"{name} is isotropic")

    def test_a_compact_component_has_finite_lengths_ordered_major_first(self):
        region = _single_region(np.ones((3, 3), dtype=bool))
        major, minor = _axis_lengths(region)
        self.assertTrue(np.isfinite(major) and np.isfinite(minor))
        self.assertGreaterEqual(major, minor)

    def test_elongated_components_report_major_above_minor(self):
        row = np.ones((1, 9), dtype=bool)
        column = np.ones((9, 1), dtype=bool)
        diagonal = np.zeros((9, 9), dtype=bool)
        np.fill_diagonal(diagonal, True)
        for name, mask in (("row", row), ("column", column), ("diagonal", diagonal)):
            with self.subTest(component=name):
                major, minor = _axis_lengths(_single_region(mask))
                self.assertGreater(major, minor, f"{name} should be elongated")
                self.assertTrue(np.isfinite(major) and np.isfinite(minor))

    def test_a_component_touching_the_array_edge_is_still_measured(self):
        mask = np.zeros((12, 12), dtype=bool)
        mask[0:2, 0:6] = True  # clamped into the top-left corner
        major, minor = _axis_lengths(_single_region(mask))
        self.assertGreaterEqual(major, minor)
        self.assertTrue(np.isfinite(major) and np.isfinite(minor))

    def test_an_empty_mask_produces_no_region_to_measure(self):
        # ``_axis_lengths`` takes a region, so the empty case is the labelling
        # path: an all-false mask must yield no region at all rather than a
        # degenerate one the caller would then measure.
        self.assertEqual(list(regionprops(label(np.zeros((8, 8), dtype=np.uint8)))), [])


if __name__ == "__main__":
    unittest.main()
