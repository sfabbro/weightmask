"""Direct coverage for the fail-closed metadata, naming and path helpers.

``ProducerMetadata`` and ``ArtifactMetadata`` reject a malformed contract at
construction, ``_ccd_name``/``_hdu_identifier`` decide what a published product
is called, and ``_fits_path_of``/``_hdu_at`` decide whether an auxiliary handle
is used at all. None of them had direct coverage: a regression reached the
release only as a wrong EXTNAME or a missing auxiliary input.
"""

import unittest

import numpy as np

from weightmask.cli import _ccd_name
from weightmask.contract import (
    CONTRACT_VERSION,
    MASK_POLARITY,
    ArtifactMetadata,
    ProducerMetadata,
    _json_from_header_cards,
)
from weightmask.mef import _fits_path_of, _hdu_at, _hdu_identifier
from weightmask.torchfits_adapter import _as_numpy


class _RaisingGet:
    """A header whose every ``get`` raises, as a corrupt handle would."""

    def get(self, *args, **kwargs):
        raise RuntimeError("unreadable header")

    def __contains__(self, key):
        raise RuntimeError("unreadable header")


class _RaisingIndex:
    """A container whose indexing raises, as a truncated file would."""

    def __len__(self):
        return 2

    def __getitem__(self, index):
        raise OSError("truncated extension")


class TestProducerMetadata(unittest.TestCase):
    def test_the_defaults_are_valid(self):
        producer = ProducerMetadata()
        self.assertEqual(producer.kind, "classical")
        self.assertEqual(producer.name, "weightmask")
        self.assertIsNone(producer.model_id)

    def test_an_ml_producer_is_allowed(self):
        producer = ProducerMetadata(kind="ml", name="streaknet", model_id="v3")
        self.assertEqual(producer.kind, "ml")
        self.assertEqual(producer.to_dict()["model_id"], "v3")

    def test_an_unknown_kind_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            ProducerMetadata(kind="hybrid")
        self.assertIn("classical", str(ctx.exception))
        self.assertIn("ml", str(ctx.exception))

    def test_an_empty_name_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            ProducerMetadata(name="")
        self.assertIn("name", str(ctx.exception))

    def test_from_dict_fills_in_every_reserved_field(self):
        producer = ProducerMetadata.from_dict({})
        self.assertEqual(producer.kind, "classical")
        self.assertEqual(producer.version, "unknown")
        self.assertEqual(sorted(producer.to_dict()), sorted(ProducerMetadata().to_dict()))

    def test_to_dict_is_json_serialisable_and_round_trips_at_every_field(self):
        import json

        payload = json.loads(json.dumps(ProducerMetadata(kind="ml", model_id="m").to_dict()))
        restored = ProducerMetadata.from_dict(payload)
        self.assertEqual(restored, ProducerMetadata(kind="ml", model_id="m", version=payload["version"]))


class TestArtifactMetadata(unittest.TestCase):
    def _metadata(self, **overrides):
        values = {"artifact_type": "quality_mask", "producer": ProducerMetadata()}
        values.update(overrides)
        return ArtifactMetadata(**values)

    def test_an_empty_artifact_type_is_refused(self):
        with self.assertRaises(ValueError) as ctx:
            self._metadata(artifact_type="")
        self.assertIn("artifact_type", str(ctx.exception))

    def test_a_declared_polarity_must_be_the_canonical_one(self):
        self._metadata(mask_polarity=MASK_POLARITY)  # accepted
        self._metadata(mask_polarity=None)  # accepted
        with self.assertRaises(ValueError) as ctx:
            self._metadata(mask_polarity="clear_means_flagged")
        self.assertIn(MASK_POLARITY, str(ctx.exception))

    def test_the_contract_version_defaults_to_the_module_constant(self):
        self.assertEqual(self._metadata().contract_version, CONTRACT_VERSION)

    def test_the_provenance_reaches_the_header(self):
        header = self._metadata(provenance={"note": "first"}).to_header()
        self.assertEqual(header["WMVERS"], CONTRACT_VERSION)
        self.assertEqual(header["WMART"], "quality_mask")
        # The producer block overflows the single-card limit, so this also pins
        # the chunked round trip rather than only the short-payload path.
        encoded = _json_from_header_cards(header, "WMPROV", "WMPV")
        self.assertIsNotNone(encoded)
        self.assertIn("first", encoded)
        self.assertIn("weightmask", encoded)


class TestCcdName(unittest.TestCase):
    def test_no_header_has_no_name(self):
        self.assertIsNone(_ccd_name(None))

    def test_the_primary_keyword_wins(self):
        self.assertEqual(_ccd_name({"CCDNAME": "8341-7-5", "CCDNAM": "older"}), "8341-7-5")

    def test_the_older_spelling_is_used_as_a_fallback(self):
        self.assertEqual(_ccd_name({"CCDNAM": "8341-7-6"}), "8341-7-6")

    def test_surrounding_whitespace_is_trimmed(self):
        self.assertEqual(_ccd_name({"CCDNAME": "  8341-7-5  "}), "8341-7-5")

    def test_a_blank_name_falls_through_rather_than_being_returned(self):
        self.assertEqual(_ccd_name({"CCDNAME": "   ", "CCDNAM": "8341-7-6"}), "8341-7-6")

    def test_a_numeric_name_is_stringified(self):
        self.assertEqual(_ccd_name({"CCDNAME": 8341}), "8341")

    def test_an_unreadable_header_has_no_name(self):
        self.assertIsNone(_ccd_name(_RaisingGet()))

    def test_a_bytes_name_is_decoded_rather_than_repr_ified(self):
        # fitsio can hand back bytes; the token must be the CCD id itself, or the
        # prior keys computed from two files' headers would not match.
        self.assertEqual(_ccd_name({"CCDNAME": b"8341-7-5"}), "8341-7-5")


class TestFitsPathOf(unittest.TestCase):
    def test_an_explicit_path_wins(self):
        self.assertEqual(_fits_path_of(object(), "a/flat.fits"), "a/flat.fits")

    def test_an_empty_explicit_path_falls_back_to_the_handle(self):
        hdul = type("H", (), {"_filename": "opened.fits"})()
        self.assertEqual(_fits_path_of(hdul, ""), "opened.fits")

    def test_a_non_string_explicit_path_falls_back_to_the_handle(self):
        hdul = type("H", (), {"_filename": "opened.fits"})()
        self.assertEqual(_fits_path_of(hdul, 5), "opened.fits")

    def test_a_bytes_filename_is_decoded(self):
        hdul = type("H", (), {"_filename": b"opened.fits"})()
        self.assertEqual(_fits_path_of(hdul, None), "opened.fits")

    def test_a_handle_without_a_filename_has_no_path(self):
        self.assertIsNone(_fits_path_of(object(), None))
        self.assertIsNone(_fits_path_of(None, None))

    def test_an_empty_filename_is_not_a_path(self):
        hdul = type("H", (), {"_filename": ""})()
        self.assertIsNone(_fits_path_of(hdul, None))

    def test_a_raising_attribute_yields_no_path(self):
        class Boom:
            @property
            def _filename(self):
                raise OSError("closed")

        self.assertIsNone(_fits_path_of(Boom(), None))


class TestHduAt(unittest.TestCase):
    def test_a_missing_container_has_no_hdu(self):
        self.assertIsNone(_hdu_at(None, 0))

    def test_an_in_range_index_is_returned(self):
        first, second = object(), object()
        self.assertIs(_hdu_at([first, second], 1), second)

    def test_an_out_of_range_index_is_none_rather_than_an_error(self):
        self.assertIsNone(_hdu_at([object()], 7))

    def test_a_raising_index_is_none(self):
        self.assertIsNone(_hdu_at(_RaisingIndex(), 0))


class TestHduIdentifier(unittest.TestCase):
    def test_the_ccd_id_is_preferred_over_any_other_name(self):
        hdu = type("H", (), {"name": "MAP_HDU1"})()
        self.assertEqual(_hdu_identifier(hdu, {"CCDNAME": "8341-7-5"}, 0), "8341-7-5")

    def test_the_older_spelling_is_used_when_the_new_one_is_absent(self):
        self.assertEqual(_hdu_identifier(None, {"CCDNAM": "8341-7-6"}, 0), "8341-7-6")

    def test_a_blank_ccd_id_falls_through_to_the_next_key(self):
        self.assertEqual(_hdu_identifier(None, {"CCDNAME": "  ", "CCDNAM": "8341-7-6"}, 0), "8341-7-6")

    def test_a_slash_is_replaced_so_the_token_is_header_safe(self):
        self.assertEqual(_hdu_identifier(None, {"CCDNAME": "8341-7/5"}, 0), "8341-7-5")

    def test_internal_and_surrounding_whitespace_is_collapsed(self):
        self.assertEqual(_hdu_identifier(None, {"CCDNAME": "8341 7 5"}, 0), "8341-7-5")
        self.assertEqual(_hdu_identifier(None, {"CCDNAME": "  8341-7-5  "}, 0), "8341-7-5")

    def test_a_bytes_ccd_id_is_decoded(self):
        self.assertEqual(_hdu_identifier(None, {"CCDNAME": b"8341-7-5"}, 0), "8341-7-5")

    def test_an_unreadable_header_falls_through_to_the_hdu_name(self):
        hdu = type("H", (), {"name": "MAP_A"})()
        self.assertEqual(_hdu_identifier(hdu, _RaisingGet(), 0), "MAP_A")

    def test_the_hdu_name_is_the_second_choice(self):
        hdu = type("H", (), {"name": "MAP_A"})()
        self.assertEqual(_hdu_identifier(hdu, {}, 3), "MAP_A")

    def test_the_index_is_the_last_resort(self):
        self.assertEqual(_hdu_identifier(None, {}, 3), "HDU3")
        hdu = type("H", (), {"name": "   "})()
        self.assertEqual(_hdu_identifier(hdu, {}, 4), "HDU4")

    def test_a_numeric_ccd_id_is_stringified(self):
        self.assertEqual(_hdu_identifier(None, {"CCDNAME": 8341}, 0), "8341")


class _FakeTensor:
    """A minimal torch-like array: detach -> cpu -> numpy, recording the order."""

    def __init__(self, array):
        self._array = np.asarray(array)
        self.calls = []

    def detach(self):
        self.calls.append("detach")
        return self

    def cpu(self):
        self.calls.append("cpu")
        return self

    def numpy(self):
        self.calls.append("numpy")
        return self._array


class TestAsNumpy(unittest.TestCase):
    def test_a_numpy_array_passes_through_with_the_same_values(self):
        source = np.arange(6, dtype=np.float32).reshape(2, 3)
        result = _as_numpy(source)
        np.testing.assert_array_equal(result, source)

    def test_a_tensor_is_moved_off_the_device_before_conversion(self):
        tensor = _FakeTensor([[1.0, 2.0]])
        result = _as_numpy(tensor)
        self.assertEqual(tensor.calls, ["detach", "cpu", "numpy"])
        np.testing.assert_array_equal(result, np.array([[1.0, 2.0]]))

    def test_a_plain_sequence_becomes_an_array(self):
        np.testing.assert_array_equal(_as_numpy([[1, 2], [3, 4]]), np.array([[1, 2], [3, 4]]))


if __name__ == "__main__":
    unittest.main()
