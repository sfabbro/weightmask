"""Direct unit tests for the cosmic-ray helpers in ``cosmics.py``.

These helpers gate the CR mask but were previously reached only through
``detect_cosmic_rays``. Pinning them directly means a change in the crowding
rule or the axis-length spelling fails on its own line
instead of as a diffuse recall shift.
"""

import unittest

import numpy as np
from skimage.measure import label, regionprops

from weightmask.cosmics import _adjust_dynamic_objlim, _axis_lengths


class TestAdjustDynamicObjlim(unittest.TestCase):
    def test_disabling_the_rule_keeps_the_default(self):
        mask = np.ones((10, 10), dtype=bool)
        self.assertEqual(_adjust_dynamic_objlim(False, mask, 5.0), 5.0)

    def test_a_missing_mask_keeps_the_default(self):
        self.assertEqual(_adjust_dynamic_objlim(True, None, 7.5), 7.5)

    def test_an_empty_mask_keeps_the_default(self):
        self.assertEqual(_adjust_dynamic_objlim(True, np.zeros((10, 10), dtype=bool), 5.0), 5.0)

    def test_crowding_scales_the_limit_four_times_the_coverage(self):
        mask = np.zeros(100, dtype=bool)
        mask[:25] = True  # coverage 0.25 -> factor 1 + 1.0
        self.assertAlmostEqual(_adjust_dynamic_objlim(True, mask, 4.0), 8.0)

    def test_the_boost_is_capped_at_two_and_a_half(self):
        half = np.zeros(100, dtype=bool)
        half[:50] = True  # coverage 0.5 -> 4 * 0.5 would be 2.0, capped at 1.5
        self.assertAlmostEqual(_adjust_dynamic_objlim(True, half, 4.0), 10.0)
        self.assertAlmostEqual(_adjust_dynamic_objlim(True, np.ones((10, 10), dtype=bool), 4.0), 10.0)

    def test_a_tiny_coverage_still_moves_the_limit(self):
        mask = np.zeros((100, 100), dtype=bool)
        mask[0, 0] = True  # coverage 1e-4 -> factor 1 + 4e-4
        self.assertAlmostEqual(_adjust_dynamic_objlim(True, mask, 5.0), 5.0 * (1.0 + 0.0004))

    def test_the_result_is_a_plain_float(self):
        self.assertIsInstance(_adjust_dynamic_objlim(True, np.zeros((4, 4), dtype=bool), 5), float)


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
