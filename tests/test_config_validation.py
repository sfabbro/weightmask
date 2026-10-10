"""Direct unit tests for the declarative configuration validation engine.

``weightmask/config.py`` is the fail-closed surface: every consumed key is
checked before any FITS handle or output writer opens, so a wrong type or an
out-of-range value fails deterministically instead of surfacing mid-run. These
tests pin the rule arithmetic, the dotted-path error text, and the cross-field
constraints directly, rather than through a full ``process_image`` call.
"""

import os
import stat
import tempfile
import unittest
from pathlib import Path

import numpy as np

from weightmask.config import (
    _KEYWORDS,
    CONFIG_SCHEMA,
    _cross_field_errors,
    _destination_path,
    _enum,
    _integer,
    _nested,
    _number,
    _parse_config_value,
    _Rule,
    _type_error,
    _validate_mapping,
    _validate_rule,
    configuration_errors,
)


class TestParseConfigValue(unittest.TestCase):
    def test_non_strings_pass_through_unchanged(self):
        for value in (5, 2.5, None, True, [1, 2], {"a": 1}):
            with self.subTest(value=value):
                self.assertIs(_parse_config_value(value), value)

    def test_boolean_words_are_case_insensitive(self):
        for text in ("true", "TRUE", "True", " yes ", "on"):
            with self.subTest(text=text):
                self.assertIs(_parse_config_value(text), True)
        for text in ("false", "FALSE", "False", " no ", "off"):
            with self.subTest(text=text):
                self.assertIs(_parse_config_value(text), False)

    def test_signed_and_unsigned_decimals_become_integers(self):
        cases = {"42": 42, " 42 ": 42, "-7": -7, "+5": 5, "-0": 0}
        for text, expected in cases.items():
            with self.subTest(text=text):
                parsed = _parse_config_value(text)
                self.assertEqual(parsed, expected)
                self.assertIsInstance(parsed, int)
                self.assertNotIsInstance(parsed, bool)

    def test_floats_require_a_decimal_point_or_an_exponent(self):
        self.assertEqual(_parse_config_value("2.5"), 2.5)
        self.assertEqual(_parse_config_value("1e3"), 1000.0)
        self.assertEqual(_parse_config_value("-1.5e-2"), -0.015)

    def test_non_numeric_text_is_left_verbatim_including_whitespace(self):
        for text in ("hello", "  hello  ", "1_000", "0x10", "nan", "inf", "not"):
            with self.subTest(text=text):
                self.assertEqual(_parse_config_value(text), text)

    def test_nan_and_infinity_text_is_not_silently_coerced(self):
        self.assertIsInstance(_parse_config_value("nan"), str)
        self.assertIsInstance(_parse_config_value("inf"), str)

    def test_bytes_are_decoded_then_parsed(self):
        self.assertIs(_parse_config_value(b"true"), True)
        self.assertEqual(_parse_config_value(b"42"), 42)
        self.assertIs(_parse_config_value(bytearray(b"off")), False)

    def test_invalid_utf8_bytes_are_returned_unchanged(self):
        raw = b"\xff\xfe"
        self.assertEqual(_parse_config_value(raw), raw)


class TestNumberRule(unittest.TestCase):
    def test_integers_and_floats_are_accepted(self):
        self.assertEqual(_validate_rule(1, _number(0), "x"), [])
        self.assertEqual(_validate_rule(1.5, _number(0), "x"), [])
        self.assertEqual(_validate_rule(np.float64(1.5), _number(0), "x"), [])

    def test_non_finite_numbers_are_rejected_as_not_finite(self):
        self.assertEqual(_validate_rule(float("nan"), _number(0), "x"), ["'x' must be finite."])
        self.assertEqual(_validate_rule(float("inf"), _number(0), "x"), ["'x' must be finite."])
        self.assertEqual(_validate_rule(float("-inf"), _number(0), "x"), ["'x' must be finite."])

    def test_text_and_booleans_are_not_numbers(self):
        self.assertEqual(_validate_rule("3", _number(0), "x"), ["'x' must be a number."])
        self.assertEqual(_validate_rule("2.5", _number(0), "x"), ["'x' must be a number."])
        self.assertEqual(_validate_rule(True, _number(0), "x"), ["'x' must be a number."])
        self.assertEqual(_validate_rule(np.bool_(False), _number(0), "x"), ["'x' must be a number."])

    def test_minimum_is_inclusive_and_exclusive_minimum_is_not(self):
        self.assertEqual(_validate_rule(0, _number(0), "x"), [])
        self.assertEqual(_validate_rule(-1, _number(0), "x"), ["'x' must be at least 0."])
        self.assertEqual(_validate_rule(0, _number(0, exclusive_minimum=True), "x"), ["'x' must be greater than 0."])
        self.assertEqual(_validate_rule(0.5, _number(0, exclusive_minimum=True), "x"), [])

    def test_maximum_is_inclusive(self):
        self.assertEqual(_validate_rule(1, _number(0, 1), "x"), [])
        self.assertEqual(_validate_rule(1.0001, _number(0, 1), "x"), ["'x' must be at most 1."])

    def test_nullable_rules_accept_none_and_others_reject_it(self):
        self.assertEqual(_validate_rule(None, _number(0, nullable=True), "x"), [])
        self.assertEqual(_validate_rule(None, _number(0), "x"), ["'x' must be a number."])


class TestIntegerRule(unittest.TestCase):
    def test_python_and_numpy_integers_are_accepted(self):
        self.assertEqual(_validate_rule(3, _integer(0), "x"), [])
        self.assertEqual(_validate_rule(np.int64(3), _integer(0), "x"), [])

    def test_booleans_floats_and_text_are_rejected(self):
        self.assertEqual(_validate_rule(True, _integer(0), "x"), ["'x' must be an integer."])
        self.assertEqual(_validate_rule(np.bool_(True), _integer(0), "x"), ["'x' must be an integer."])
        self.assertEqual(_validate_rule(3.0, _integer(0), "x"), ["'x' must be an integer."])
        self.assertEqual(_validate_rule(3.5, _integer(0), "x"), ["'x' must be an integer."])
        self.assertEqual(_validate_rule("3", _integer(0), "x"), ["'x' must be an integer."])

    def test_range_and_oddness_are_enforced(self):
        self.assertEqual(_validate_rule(-1, _integer(0), "x"), ["'x' must be at least 0."])
        self.assertEqual(_validate_rule(6, _integer(0, 5), "x"), ["'x' must be at most 5."])
        self.assertEqual(_validate_rule(4, _integer(0, exclusive_minimum=True, odd=True), "x"), ["'x' must be odd."])
        self.assertEqual(_validate_rule(3, _integer(0, exclusive_minimum=True, odd=True), "x"), [])
        self.assertEqual(
            _validate_rule(0, _integer(0, exclusive_minimum=True, odd=True), "x"), ["'x' must be greater than 0."]
        )


class TestEnumRule(unittest.TestCase):
    def test_members_are_accepted_and_non_members_rejected(self):
        rule = _enum("sep", "median_filter", "robust_median_fallback")
        self.assertEqual(_validate_rule("sep", rule, "x"), [])
        self.assertEqual(
            _validate_rule("filter", rule, "x"),
            ["'x' must be one of 'median_filter', 'robust_median_fallback', 'sep'."],
        )

    def test_casefold_rules_accept_other_cases_but_still_reject_non_members(self):
        rule = _enum("full", "mesh", casefold=True)
        self.assertEqual(_validate_rule("FULL", rule, "x"), [])
        self.assertEqual(_validate_rule("Full", rule, "x"), [])
        self.assertEqual(_validate_rule("partial", rule, "x"), ["'x' must be one of 'full', 'mesh'."])

    def test_empty_and_non_string_values_are_not_strings(self):
        rule = _enum("a", "b")
        self.assertEqual(_validate_rule("", rule, "x"), ["'x' must be a non-empty string."])
        self.assertEqual(_validate_rule(5, rule, "x"), ["'x' must be a non-empty string."])


class TestKeywordRule(unittest.TestCase):
    def test_a_single_keyword_or_a_non_empty_list_is_accepted(self):
        self.assertEqual(_validate_rule("SATURATE", _KEYWORDS, "x"), [])
        self.assertEqual(_validate_rule(["GAIN", "RDNOISE"], _KEYWORDS, "x"), [])
        self.assertEqual(_validate_rule(("GAIN",), _KEYWORDS, "x"), [])

    def test_empty_lists_and_non_string_entries_are_rejected(self):
        expected = ["'x' must be a header keyword string or non-empty list of strings."]
        self.assertEqual(_validate_rule([], _KEYWORDS, "x"), expected)
        self.assertEqual(_validate_rule([""], _KEYWORDS, "x"), expected)
        self.assertEqual(_validate_rule([1], _KEYWORDS, "x"), expected)
        self.assertEqual(_validate_rule(5, _KEYWORDS, "x"), expected)


class TestValidateRuleEdgeCases(unittest.TestCase):
    def test_type_error_text_is_the_dotted_path_then_the_expectation(self):
        self.assertEqual(_type_error("a.b", "an integer"), "'a.b' must be an integer.")

    def test_an_unknown_rule_kind_is_an_internal_error_not_a_pass(self):
        self.assertEqual(_validate_rule(1, _Rule("mystery"), "x"), ["Internal validation error for 'x'."])

    def test_nullable_and_non_nullable_boolean_rules(self):
        self.assertEqual(_validate_rule(None, _Rule("boolean", nullable=True), "x"), [])
        self.assertEqual(_validate_rule(None, _Rule("boolean"), "x"), ["'x' must be a boolean."])


class TestValidateMapping(unittest.TestCase):
    def test_a_non_dictionary_root_is_rejected(self):
        self.assertEqual(_validate_mapping(5, CONFIG_SCHEMA), ["'Configuration' must be a dictionary."])

    def test_a_non_dictionary_section_names_its_dotted_path(self):
        self.assertEqual(
            _validate_mapping({"flat_masking": 5}, CONFIG_SCHEMA), ["'flat_masking' must be a dictionary."]
        )

    def test_unknown_keys_are_rejected_with_their_dotted_path(self):
        self.assertEqual(
            _validate_mapping({"bogus": 1}, CONFIG_SCHEMA),
            ["Unsupported configuration key 'bogus'."],
        )
        self.assertEqual(
            _validate_mapping({"flat_masking": {"bogus": 1}}, CONFIG_SCHEMA),
            ["Unsupported configuration key 'flat_masking.bogus'."],
        )

    def test_absent_keys_are_optional_by_design(self):
        # The schema is a shape for keys that are present, not a required-key list.
        self.assertEqual(_validate_mapping({}, CONFIG_SCHEMA), [])

    def test_a_valid_subset_produces_no_errors(self):
        self.assertEqual(_validate_mapping({"sep_background": {"method": "sep", "box_size": 64}}, CONFIG_SCHEMA), [])

    def test_errors_accumulate_across_sections(self):
        errors = _validate_mapping(
            {"flat_masking": {"bogus": 1}, "sep_background": {"method": "nope"}},
            CONFIG_SCHEMA,
        )
        self.assertIn("Unsupported configuration key 'flat_masking.bogus'.", errors)
        self.assertEqual(len(errors), 2)


class TestSchemaEndToEnd(unittest.TestCase):
    def test_output_bitpix_choices_are_reported(self):
        self.assertEqual(
            configuration_errors({"output_params": {"mask_bitpix": 0}}),
            ["'output_params.mask_bitpix' must be one of 8, 16, 32, 64."],
        )
        self.assertEqual(configuration_errors({"output_params": {"mask_bitpix": 32}}), [])

    def test_negative_signed_ivar_bitpix_is_a_member(self):
        self.assertEqual(configuration_errors({"output_params": {"ivar_bitpix": -32}}), [])
        self.assertEqual(
            configuration_errors({"output_params": {"ivar_bitpix": -16}}),
            ["'output_params.ivar_bitpix' must be one of -64, -32, 32, 64."],
        )

    def test_casefolded_schema_values_are_accepted(self):
        self.assertEqual(configuration_errors({"output_params": {"sky_format": "MESH"}}), [])

    def test_streak_mode_is_closed_to_the_single_production_value(self):
        self.assertEqual(configuration_errors({"streak_masking": {"mode": "auto_ground"}}), [])
        self.assertEqual(
            configuration_errors({"streak_masking": {"mode": "radon"}}),
            ["'streak_masking.mode' must be one of 'auto_ground'."],
        )

    def test_a_boolean_is_not_accepted_where_an_integer_is_required(self):
        self.assertEqual(
            configuration_errors({"sep_background": {"box_size": True}}),
            ["'sep_background.box_size' must be an integer."],
        )

    def test_a_boolean_is_not_accepted_where_a_number_is_required(self):
        self.assertEqual(
            configuration_errors({"saturation": {"bleed_thresh_sigma": True}}),
            ["'saturation.bleed_thresh_sigma' must be a number."],
        )


class TestNested(unittest.TestCase):
    def test_present_values_are_returned_and_missing_ones_use_the_default(self):
        config = {"a": {"b": {"c": 7}}}
        self.assertEqual(_nested(config, ("a", "b", "c")), 7)
        self.assertEqual(_nested(config, ("a", "b", "missing"), "fallback"), "fallback")
        self.assertEqual(_nested(config, ("a", "z"), None), None)

    def test_a_non_mapping_intermediate_uses_the_default(self):
        self.assertEqual(_nested({"a": 5}, ("a", "b"), "fallback"), "fallback")

    def test_lookups_do_not_mutate_the_config(self):
        config = {"a": {"b": {"c": 7}}}
        _nested(config, ("a", "b", "c"))
        self.assertEqual(config, {"a": {"b": {"c": 7}}})


class TestCrossFieldErrors(unittest.TestCase):
    def test_the_default_configuration_has_no_cross_field_conflicts(self):
        self.assertEqual(_cross_field_errors({}), [])

    def test_inverted_thresholds_are_reported_with_both_paths(self):
        errors = _cross_field_errors({"flat_masking": {"local_low_thresh": 2.0, "local_high_thresh": 1.0}})
        self.assertEqual(
            errors,
            ["'flat_masking.local_low_thresh' must be < 'flat_masking.local_high_thresh'."],
        )

    def test_strict_comparisons_reject_equality_and_inclusive_ones_allow_it(self):
        # local_low_thresh < local_high_thresh is strict.
        self.assertEqual(
            _cross_field_errors({"flat_masking": {"local_low_thresh": 1.0, "local_high_thresh": 1.0}}),
            ["'flat_masking.local_low_thresh' must be < 'flat_masking.local_high_thresh'."],
        )
        # bleed_cap_min <= bleed_cap_max is inclusive, so equality is legal.
        self.assertEqual(_cross_field_errors({"saturation": {"bleed_cap_min": 200, "bleed_cap_max": 200}}), [])
        self.assertEqual(
            _cross_field_errors({"saturation": {"bleed_cap_min": 201, "bleed_cap_max": 200}}),
            ["'saturation.bleed_cap_min' must be <= 'saturation.bleed_cap_max'."],
        )

    def test_box_size_may_not_exceed_max_box_size(self):
        errors = _cross_field_errors({"sep_background": {"box_size": 2048, "max_box_size": 512}})
        self.assertEqual(errors, ["'sep_background.box_size' must be <= 'sep_background.max_box_size'."])

    def test_histogram_bounds_must_be_ordered_when_both_are_given(self):
        errors = _cross_field_errors({"saturation": {"histogram_params": {"hist_min_adu": 500, "hist_max_adu": 500}}})
        self.assertEqual(
            errors,
            ["'saturation.histogram_params.hist_min_adu' must be < 'saturation.histogram_params.hist_max_adu'."],
        )
        self.assertEqual(
            _cross_field_errors({"saturation": {"histogram_params": {"hist_min_adu": 100, "hist_max_adu": 200}}}),
            [],
        )

    def test_a_single_histogram_bound_is_not_a_conflict(self):
        self.assertEqual(_cross_field_errors({"saturation": {"histogram_params": {"hist_min_adu": 500}}}), [])
        self.assertEqual(_cross_field_errors({"saturation": {"histogram_params": {"hist_max_adu": 500}}}), [])

    def test_streak_support_may_not_exceed_the_strip(self):
        errors = _cross_field_errors({"streak_masking": {"mask_params": {"min_row_hits": 300}}})
        self.assertEqual(
            errors,
            ["'streak_masking.mask_params.min_row_hits' must be <= 'streak_masking.mask_params.strip_length'."],
        )

    def test_cross_field_checks_are_skipped_while_the_mapping_is_invalid(self):
        errors = configuration_errors(
            {
                "flat_masking": {"local_low_thresh": 2.0, "local_high_thresh": 1.0, "bogus": 1},
            }
        )
        self.assertEqual(errors, ["Unsupported configuration key 'flat_masking.bogus'."])
        self.assertNotIn("local_low_thresh", "".join(errors))


class TestDestinationPath(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def test_a_new_path_is_returned_as_is(self):
        target = self.root / "config.yml"
        self.assertEqual(_destination_path(target), target)

    def test_a_directory_receives_the_canonical_file_name(self):
        self.assertEqual(_destination_path(self.root), self.root / "weightmask.yml")

    def test_an_existing_regular_file_is_returned_as_is(self):
        target = self.root / "config.yml"
        target.write_text("existing")
        self.assertEqual(_destination_path(target), target)

    def test_a_symlink_destination_is_refused(self):
        real = self.root / "real.yml"
        real.write_text("existing")
        link = self.root / "link.yml"
        os.symlink(real, link)
        with self.assertRaises(FileExistsError) as raised:
            _destination_path(link)
        self.assertEqual(Path(raised.exception.args[0]), link)

    def test_a_symlinked_child_inside_a_directory_is_refused(self):
        real = self.root / "real.yml"
        real.write_text("existing")
        directory = self.root / "out"
        directory.mkdir()
        os.symlink(real, directory / "weightmask.yml")
        with self.assertRaises(FileExistsError):
            _destination_path(directory)

    @unittest.skipUnless(hasattr(os, "mkfifo"), "FIFOs need a POSIX platform")
    def test_an_existing_non_regular_file_is_refused(self):
        fifo = self.root / "pipe"
        os.mkfifo(fifo)
        self.assertTrue(stat.S_ISFIFO(fifo.lstat().st_mode))
        with self.assertRaises(FileExistsError):
            _destination_path(fifo)


if __name__ == "__main__":
    unittest.main()
