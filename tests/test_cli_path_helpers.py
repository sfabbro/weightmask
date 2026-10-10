"""Direct unit tests for the CLI path helpers and the process header lookups.

These sit on the boundary between a user-supplied path and the filesystem:
``_check_aux_input`` decides whether an auxiliary input is usable,
``_validate_output_paths`` decides whether writing would destroy an input, and
``_first_present_keyword``/``_header_lookup`` decide which FITS keyword a
str-or-list configuration entry resolves to. Each was previously reached only
through the whole CLI or an image run.
"""

import os
import tempfile
import unittest
from pathlib import Path

import fitsio
import numpy as np

from weightmask.cli import (
    _check_aux_input,
    _find_default_config,
    _image_shape,
    _read_and_clean_config,
    _validate_output_paths,
)
from weightmask.process import _first_present_keyword, _header_lookup


def _write_valid_fits(path):
    fitsio.write(str(path), np.zeros((4, 4), dtype=np.float32), clobber=True)
    return str(path)


class TestCheckAuxInput(unittest.TestCase):
    def test_an_explicit_hdu_spec_is_rejected(self):
        # Flat/dark/keep-maps are matched to the science HDU by index, so an
        # [N] would be accepted and then silently ignored.
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_valid_fits(Path(tmp) / "flat.fits")
            self.assertFalse(_check_aux_input(f"{path}[1]", "Flat field"))

    def test_a_missing_file_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(_check_aux_input(str(Path(tmp) / "absent.fits"), "Dark frame"))

    def test_a_valid_fits_file_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = _write_valid_fits(Path(tmp) / "badpix.fits")
            self.assertTrue(_check_aux_input(path, "Bad pixel mask"))

    def test_a_non_fits_path_is_rejected(self):
        # A directory exists but is not a FITS file; fitsio raises OSError,
        # which the validator turns into a refusal rather than a traceback.
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(_check_aux_input(tmp, "Flat field"))


class TestImageShape(unittest.TestCase):
    def test_shape_is_returned_row_major(self):
        header = {"NAXIS1": 10, "NAXIS2": 5, "NAXIS": 2}
        self.assertEqual(_image_shape(header), (5, 10))

    def test_naxis_defaults_to_two_when_absent(self):
        self.assertEqual(_image_shape({"NAXIS1": 10, "NAXIS2": 5}), (5, 10))

    def test_numeric_strings_are_coerced(self):
        self.assertEqual(_image_shape({"NAXIS1": "10", "NAXIS2": "5", "NAXIS": 2}), (5, 10))

    def test_non_image_rank_is_not_a_shape(self):
        for naxis in (1, 3, 0):
            with self.subTest(naxis=naxis):
                self.assertIsNone(_image_shape({"NAXIS1": 10, "NAXIS2": 5, "NAXIS": naxis}))

    def test_non_positive_dimensions_are_not_a_shape(self):
        for nx, ny in ((0, 5), (10, 0), (-4, 5)):
            with self.subTest(nx=nx, ny=ny):
                self.assertIsNone(_image_shape({"NAXIS1": nx, "NAXIS2": ny, "NAXIS": 2}))

    def test_missing_or_unparseable_axis_cards_return_none(self):
        self.assertIsNone(_image_shape({"NAXIS2": 5, "NAXIS": 2}))
        self.assertIsNone(_image_shape({"NAXIS1": "wide", "NAXIS2": 5, "NAXIS": 2}))
        self.assertIsNone(_image_shape(None))


class TestFindDefaultConfig(unittest.TestCase):
    def test_none_when_no_default_is_present(self):
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                self.assertIsNone(_find_default_config())
            finally:
                os.chdir(previous)

    def test_the_working_directory_copy_is_preferred(self):
        previous = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                Path("weightmask.yml").write_text("output_params: {}\n")
                self.assertEqual(_find_default_config(), "weightmask.yml")
            finally:
                os.chdir(previous)


class TestReadAndCleanConfig(unittest.TestCase):
    def test_a_plain_document_is_parsed_and_cleaned(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weightmask.yml"
            path.write_text(
                "# a comment\n"
                "output_params:\n"
                "  mask_bitpix: 32\n"
                "  compress: false\n"
                "streak_masking:\n"
                "  mode: auto_ground\n"
            )
            config = _read_and_clean_config(str(path))
        self.assertEqual(
            config,
            {"output_params": {"mask_bitpix": 32, "compress": False}, "streak_masking": {"mode": "auto_ground"}},
        )

    def test_a_non_dictionary_document_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weightmask.yml"
            for text in ("- 1\n- 2\n", "just a string\n"):
                with self.subTest(document=text.splitlines()[0]):
                    path.write_text(text)
                    self.assertIsNone(_read_and_clean_config(str(path)))

    def test_a_missing_file_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertIsNone(_read_and_clean_config(str(Path(tmp) / "absent.yml")))

    def test_a_malformed_document_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weightmask.yml"
            path.write_text("output_params: [1, 2\n")
            self.assertIsNone(_read_and_clean_config(str(path)))


class TestValidateOutputPaths(unittest.TestCase):
    def test_distinct_paths_are_accepted(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = {
                "map_path": str(Path(tmp) / "a.fits"),
                "mask_path": str(Path(tmp) / "b.fits"),
                "individual_mask_paths": {"STREAK": str(Path(tmp) / "c.fits")},
            }
            self.assertTrue(_validate_output_paths(paths, [str(Path(tmp) / "input.fits")]))

    def test_an_output_aliasing_an_input_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            science = _write_valid_fits(Path(tmp) / "science.fits")
            self.assertFalse(_validate_output_paths({"map_path": science}, [science]))

    def test_two_outputs_aliasing_each_other_are_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            shared = str(Path(tmp) / "shared.fits")
            paths = {"map_path": shared, "mask_path": shared}
            self.assertFalse(_validate_output_paths(paths, [str(Path(tmp) / "input.fits")]))

    def test_an_individual_mask_path_aliasing_an_input_is_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            science = _write_valid_fits(Path(tmp) / "science.fits")
            paths = {"map_path": str(Path(tmp) / "out.fits"), "individual_mask_paths": {"CR": science}}
            self.assertFalse(_validate_output_paths(paths, [science]))

    def test_a_symlinked_output_is_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            science = _write_valid_fits(Path(tmp) / "science.fits")
            link = Path(tmp) / "link.fits"
            os.symlink(science, link)
            self.assertFalse(_validate_output_paths({"map_path": str(link)}, [science]))

    def test_empty_top_level_entries_are_skipped(self):
        # ``determine_output_paths`` leaves optional outputs (weight_raw) and
        # every individual mask as None/absent when not requested, so those
        # must not be treated as paths to compare.
        with tempfile.TemporaryDirectory() as tmp:
            paths = {
                "map_path": str(Path(tmp) / "a.fits"),
                "mask_path": None,
                "out_weight_raw_path": None,
                "individual_mask_paths": {},
            }
            self.assertTrue(_validate_output_paths(paths, [None, ""]))


class TestHeaderKeywordHelpers(unittest.TestCase):
    def test_first_present_keyword_finds_the_configured_name(self):
        header = {"GAIN": 2.0}
        self.assertEqual(_first_present_keyword(header, "GAIN"), "GAIN")
        self.assertEqual(_first_present_keyword(header, ["EGAIN", "GAIN"]), "GAIN")
        self.assertIsNone(_first_present_keyword(header, ["EGAIN", "BUNIT"]))
        self.assertIsNone(_first_present_keyword(None, "GAIN"))

    def test_non_string_and_empty_keywords_are_skipped(self):
        header = {"GAIN": 2.0}
        self.assertEqual(_first_present_keyword(header, [None, "", 3, "GAIN"]), "GAIN")

    def test_header_lookup_returns_the_first_present_value(self):
        header = {"EGAIN": 3.5, "GAIN": 2.0}
        self.assertEqual(_header_lookup(header, "GAIN", 1.0), 2.0)
        self.assertEqual(_header_lookup(header, ["EGAIN", "GAIN"], 1.0), 3.5)

    def test_header_lookup_falls_back_to_the_default(self):
        self.assertEqual(_header_lookup({"GAIN": 2.0}, ["BUNIT", "RDNOISE"], 1.0), 1.0)
        self.assertEqual(_header_lookup(None, "GAIN", 1.0), 1.0)
        self.assertEqual(_header_lookup({}, "GAIN", None), None)


if __name__ == "__main__":
    unittest.main()
