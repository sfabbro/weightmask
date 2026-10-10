"""Direct unit tests for the per-pixel amplifier gain map in variance.py.

``amplifier_gain_map`` decides the gain for every pixel of a dual-amplifier
MegaCam HDU. It only accepts a section pair that is inside the image and
disjoint, and it tries the ``DETSEC*`` mosaic coordinates first, then the local
``DATASEC*`` and ``AMPSEC*`` cards. A wrong answer here scales the whole
inverse-variance plane, so each branch is pinned explicitly rather than only
through ``calculate_inverse_variance``.
"""

import unittest

import numpy as np

from weightmask.variance import amplifier_gain_map


class _RaisingHeader:
    """A header whose reads blow up, as a malformed FITS handle might."""

    def get(self, key, default=None):
        raise RuntimeError("header read failed")


class _NonCallableGet:
    get = "not callable"


DUAL = {
    "GAINA": 1.0,
    "GAINB": 2.0,
    # Columns 1-4 -> amplifier A, columns 5-8 -> amplifier B, on a 4x8 HDU.
    "DETSECA": "[1:4,1:4]",
    "DETSECB": "[5:8,1:4]",
}


class TestRejections(unittest.TestCase):
    def test_a_missing_header_returns_none(self):
        self.assertIsNone(amplifier_gain_map(None, (4, 8), fallback=1.0))

    def test_a_missing_shape_returns_none(self):
        self.assertIsNone(amplifier_gain_map(dict(DUAL), None, fallback=1.0))

    def test_shapes_that_are_not_two_dimensional_are_refused(self):
        for shape in ((8,), (2, 3, 4), ()):
            with self.subTest(shape=shape):
                self.assertIsNone(amplifier_gain_map(dict(DUAL), shape, fallback=1.0))

    def test_a_header_without_a_callable_get_is_refused(self):
        for header in (5, "GAINA", ["GAINA"], object(), _NonCallableGet()):
            with self.subTest(header=type(header).__name__):
                self.assertIsNone(amplifier_gain_map(header, (4, 8), fallback=1.0))

    def test_a_get_that_raises_is_treated_as_absent(self):
        self.assertIsNone(amplifier_gain_map(_RaisingHeader(), (4, 8), fallback=1.0))

    def test_both_gains_are_required(self):
        for missing in ("GAINA", "GAINB"):
            header = {k: v for k, v in DUAL.items() if k != missing}
            with self.subTest(missing=missing):
                self.assertIsNone(amplifier_gain_map(header, (4, 8), fallback=1.0))

    def test_a_non_numeric_gain_is_refused(self):
        header = dict(DUAL, GAINB="bright")
        self.assertIsNone(amplifier_gain_map(header, (4, 8), fallback=1.0))

    def test_non_finite_or_non_positive_gains_are_refused(self):
        for value in (0.0, -1.0, float("nan"), float("inf"), float("-inf")):
            for key in ("GAINA", "GAINB"):
                with self.subTest(key=key, value=value):
                    self.assertIsNone(amplifier_gain_map(dict(DUAL, **{key: value}), (4, 8), fallback=1.0))

    def test_no_section_pair_at_all_returns_none(self):
        header = {"GAINA": 1.0, "GAINB": 2.0}
        self.assertIsNone(amplifier_gain_map(header, (4, 8), fallback=1.0))

    def test_malformed_or_zero_based_sections_are_refused(self):
        malformed = (
            "[1:4,1:4",  # unterminated
            "1:4,1:4",  # no brackets
            "[1:4]",  # one axis only
            "[0:4,1:4]",  # FITS sections are 1-based; a 0 is invalid
            "[1:4,0:4]",
            "",
            None,
            4,
        )
        for value in malformed:
            with self.subTest(value=value):
                self.assertIsNone(amplifier_gain_map(dict(DUAL, DETSECA=value), (4, 8), fallback=1.0))

    def test_identical_sections_overlap_and_are_refused(self):
        header = dict(DUAL, DETSECB="[1:4,1:4]")
        self.assertIsNone(amplifier_gain_map(header, (4, 8), fallback=1.0))

    def test_a_full_frame_amplifier_with_a_partial_second_amplifier_is_refused(self):
        # A covers the whole HDU; no disjoint partner can exist, and the caller
        # must keep the single first-present gain keyword instead.
        header = {"GAINA": 1.0, "GAINB": 2.0, "DETSECA": "[1:8,1:4]", "DETSECB": "[1:4,1:4]"}
        self.assertIsNone(amplifier_gain_map(header, (4, 8), fallback=1.0))


class TestSectionSelection(unittest.TestCase):
    def test_dual_amplifier_split_matches_the_two_gains(self):
        gain = amplifier_gain_map(dict(DUAL), (4, 8), fallback=1.0)
        self.assertIsNotNone(gain)
        self.assertTrue(np.all(gain[:, :4] == 1.0))
        self.assertTrue(np.all(gain[:, 4:] == 2.0))

    def test_rows_and_columns_come_from_the_fits_one_based_section(self):
        # Columns 1-4 of this 8x4 HDU are split into a top and a bottom half.
        header = {"GAINA": 1.0, "GAINB": 2.0, "DETSECA": "[1:4,1:4]", "DETSECB": "[1:4,5:8]"}
        gain = amplifier_gain_map(header, (8, 4), fallback=9.0)
        self.assertTrue(np.all(gain[:4, :] == 1.0))
        self.assertTrue(np.all(gain[4:, :] == 2.0))
        self.assertNotIn(9.0, set(gain.ravel().tolist()))

    def test_out_of_image_detector_rows_fall_back_to_local_sections(self):
        # DETSEC describes the mosaic, not this HDU: its rows run past the frame.
        header = {
            "GAINA": 1.0,
            "GAINB": 2.0,
            "DETSECA": "[1:4,101:104]",
            "DETSECB": "[5:8,101:104]",
            "DATASECA": "[1:4,1:4]",
            "DATASECB": "[5:8,1:4]",
        }
        gain = amplifier_gain_map(header, (4, 8), fallback=1.0)
        self.assertTrue(np.all(gain[:, :4] == 1.0))
        self.assertTrue(np.all(gain[:, 4:] == 2.0))

    def test_ampsec_is_used_when_the_other_cards_are_absent(self):
        header = {"GAINA": 1.0, "GAINB": 2.0, "AMPSECA": "[1:4,1:4]", "AMPSECB": "[5:8,1:4]"}
        gain = amplifier_gain_map(header, (4, 8), fallback=1.0)
        self.assertTrue(np.all(gain[:, :4] == 1.0))
        self.assertTrue(np.all(gain[:, 4:] == 2.0))

    def test_overlapping_detector_sections_fall_back_to_a_valid_data_section(self):
        header = {
            "GAINA": 1.0,
            "GAINB": 2.0,
            "DETSECA": "[1:4,1:4]",
            "DETSECB": "[3:6,1:4]",  # rows 2-5 overlap A's rows 0-3
            "DATASECA": "[1:4,1:4]",
            "DATASECB": "[5:8,1:4]",
        }
        gain = amplifier_gain_map(header, (4, 8), fallback=1.0)
        self.assertTrue(np.all(gain[:, :4] == 1.0))
        self.assertTrue(np.all(gain[:, 4:] == 2.0))

    def test_a_list_shape_is_accepted(self):
        gain = amplifier_gain_map(dict(DUAL), [4, 8], fallback=1.0)
        self.assertIsNotNone(gain)
        self.assertEqual(gain.shape, (4, 8))


class TestFallbackAndOutput(unittest.TestCase):
    def test_pixels_outside_both_sections_keep_the_fallback(self):
        # A 12-column HDU whose two sections only cover columns 0-8.
        header = {"GAINA": 1.0, "GAINB": 2.0, "DETSECA": "[1:4,1:4]", "DETSECB": "[5:8,1:4]"}
        gain = amplifier_gain_map(header, (4, 12), fallback=1.5)
        self.assertTrue(np.all(gain[:, :4] == 1.0))
        self.assertTrue(np.all(gain[:, 4:8] == 2.0))
        self.assertTrue(np.all(gain[:, 8:] == 1.5))

    def test_an_invalid_fallback_falls_back_to_gain_a(self):
        header = {"GAINA": 1.0, "GAINB": 2.0, "DETSECA": "[1:4,1:4]", "DETSECB": "[5:8,1:4]"}
        for fallback in (0.0, -3.0, float("nan"), float("inf"), "x", None):
            with self.subTest(fallback=fallback):
                gain = amplifier_gain_map(header, (4, 12), fallback=fallback)
                self.assertTrue(np.all(gain[:, 8:] == 1.0))

    def test_equal_gains_produce_a_uniform_map(self):
        header = {"GAINA": 7.0, "GAINB": 7.0, "DETSECA": "[1:4,1:4]", "DETSECB": "[5:8,1:4]"}
        gain = amplifier_gain_map(header, (4, 8), fallback=7.0)
        np.testing.assert_allclose(gain, 7.0)

    def test_the_output_is_float32_and_shaped_like_the_image(self):
        gain = amplifier_gain_map(dict(DUAL), (37, 11), fallback=1.0)
        self.assertEqual(gain.shape, (37, 11))
        self.assertEqual(gain.dtype, np.float32)

    def test_an_astropy_header_is_read_the_same_way(self):
        from astropy.io import fits

        header = fits.Header([("GAINA", 1.0), ("GAINB", 2.0), ("DETSECA", "[1:4,1:4]"), ("DETSECB", "[5:8,1:4]")])
        gain = amplifier_gain_map(header, (4, 8), fallback=1.0)
        self.assertTrue(np.all(gain[:, :4] == 1.0))
        self.assertTrue(np.all(gain[:, 4:] == 2.0))


if __name__ == "__main__":
    unittest.main()
