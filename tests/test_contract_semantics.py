"""Direct unit tests for the product-contract semantics in ``contract.py``.

The contract is what a consumer relies on: the mask dtype and polarity, the
producer identity that is stamped into every header, and the JSON cards that
carry metadata through FITS. These tests exercise those semantics at the
boundary instead of only through a full pipeline run.
"""

import json
import unittest

import numpy as np

from weightmask.contract import (
    CONTRACT_VERSION,
    MASK_POLARITY,
    ArtifactMetadata,
    MaskPolarity,
    ProducerMetadata,
    WeightMaskProduct,
    _json_from_header_cards,
    _json_header_cards,
    _validate_confidence_percentile,
    canonical_quality_mask,
)


class TestMaskPolarity(unittest.TestCase):
    def test_value_matches_the_documented_polarity(self):
        self.assertEqual(MASK_POLARITY, "set_means_flagged")
        self.assertEqual(MaskPolarity.SET_MEANS_FLAGGED.value, MASK_POLARITY)

    def test_membership_and_lookup_round_trip(self):
        self.assertIn(MASK_POLARITY, {member.value for member in MaskPolarity})
        self.assertIs(MaskPolarity(MASK_POLARITY), MaskPolarity.SET_MEANS_FLAGGED)

    def test_an_unknown_polarity_is_rejected(self):
        with self.assertRaises(ValueError):
            MaskPolarity("clear_means_flagged")

    def test_the_polarity_survives_a_header_round_trip(self):
        metadata = ArtifactMetadata(
            "quality_mask",
            ProducerMetadata(),
            {"config_sha": "abc"},
            mask_polarity=MASK_POLARITY,
            semantics="named_quality_bits",
        )
        header = metadata.to_header()
        self.assertEqual(header["WMMASK"], MASK_POLARITY)
        recovered = ArtifactMetadata.from_header(header)
        self.assertEqual(recovered.mask_polarity, MASK_POLARITY)

    def test_a_mismatched_polarity_cannot_be_constructed(self):
        with self.assertRaises(ValueError):
            ArtifactMetadata("quality_mask", ProducerMetadata(), mask_polarity="clear_means_flagged")


class TestCanonicalQualityMask(unittest.TestCase):
    def test_non_negative_signed_input_becomes_uint32(self):
        signed = np.array([[0, 1], [2, 3]], dtype=np.int64)
        result = canonical_quality_mask(signed, (2, 2))
        self.assertEqual(result.dtype, np.dtype(np.uint32))
        np.testing.assert_array_equal(result, signed.astype(np.uint32))

    def test_negative_signed_values_are_rejected_not_wrapped(self):
        for dtype in (np.int8, np.int16, np.int32, np.int64):
            with self.subTest(dtype=dtype):
                with self.assertRaises(ValueError):
                    canonical_quality_mask(np.array([-1], dtype=dtype), (1,))

    def test_a_future_bit_inside_uint32_is_preserved(self):
        top_bit = 1 << 31
        result = canonical_quality_mask(np.array([top_bit], dtype=np.uint64), (1,))
        self.assertEqual(int(result[0]), top_bit)

    def test_values_wider_than_uint32_are_rejected(self):
        with self.assertRaises(ValueError):
            canonical_quality_mask(np.array([1 << 32], dtype=np.uint64), (1,))
        with self.assertRaises(ValueError):
            canonical_quality_mask(np.array([np.iinfo(np.uint64).max], dtype=np.uint64), (1,))

    def test_byte_order_is_accepted_and_values_are_preserved(self):
        big_endian = np.array([1, 2, 1 << 20], dtype=">u4")
        result = canonical_quality_mask(big_endian, (3,))
        self.assertEqual(result.dtype, np.dtype(np.uint32))
        np.testing.assert_array_equal(result, np.array([1, 2, 1 << 20], dtype=np.uint32))

    def test_non_contiguous_input_is_accepted_and_copied(self):
        base = np.arange(12, dtype=np.uint32).reshape(3, 4)
        view = base[:, ::2]
        self.assertFalse(view.flags["C_CONTIGUOUS"])
        result = canonical_quality_mask(view, (3, 2))
        np.testing.assert_array_equal(result, view)
        self.assertFalse(np.shares_memory(result, view))

    def test_the_callers_array_is_never_mutated(self):
        mask = np.zeros((2, 2), dtype=np.uint32)
        result = canonical_quality_mask(mask, (2, 2))
        result[0, 0] = 7
        self.assertEqual(int(mask[0, 0]), 0)
        self.assertFalse(np.shares_memory(result, mask))

        signed = np.array([[1, 2], [3, 4]], dtype=np.int64)
        signed_result = canonical_quality_mask(signed, (2, 2))
        signed_result[0, 0] |= np.uint32(1 << 5)
        self.assertEqual(int(signed[0, 0]), 1)

    def test_a_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            canonical_quality_mask(np.zeros((2, 2), dtype=np.uint32), (3, 2))

    def test_non_integer_dtypes_are_rejected(self):
        for dtype in (np.float32, np.float64, np.bool_):
            with self.subTest(dtype=dtype):
                with self.assertRaises(TypeError):
                    canonical_quality_mask(np.zeros((2, 2), dtype=dtype), (2, 2))


class TestProducerMetadataRoundTrip(unittest.TestCase):
    def test_to_dict_and_from_dict_recover_the_same_producer(self):
        producer = ProducerMetadata(
            name="weightmask",
            version="0.2.1",
            kind="classical",
            algorithm="classical_mask_and_variance",
            model_id="none",
            model_version="1",
            inference_backend="numpy",
        )
        recovered = ProducerMetadata.from_dict(producer.to_dict())
        self.assertEqual(recovered, producer)

    def test_the_default_producer_round_trips(self):
        self.assertEqual(ProducerMetadata.from_dict(ProducerMetadata().to_dict()), ProducerMetadata())

    def test_an_empty_name_or_unknown_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            ProducerMetadata(name="")
        with self.assertRaises(ValueError):
            ProducerMetadata(kind="neural")


class TestArtifactMetadataHeaderRoundTrip(unittest.TestCase):
    def test_metadata_survives_a_header_round_trip(self):
        producer = ProducerMetadata(version="0.2.1")
        provenance = {"config_sha": "deadbeef", "exposure": "1013719p"}
        metadata = ArtifactMetadata(
            "inverse_variance",
            producer,
            provenance,
            semantics="inverse_variance_adu^-2",
        )
        header = metadata.to_header()
        self.assertEqual(header["WMVERS"], CONTRACT_VERSION)
        self.assertEqual(header["WMART"], "inverse_variance")
        recovered = ArtifactMetadata.from_header(header)
        self.assertEqual(recovered.artifact_type, "inverse_variance")
        self.assertEqual(recovered.contract_version, CONTRACT_VERSION)
        self.assertEqual(recovered.producer, producer)
        self.assertEqual(dict(recovered.provenance), provenance)
        self.assertEqual(recovered.semantics, "inverse_variance_adu^-2")

    def test_a_long_provenance_survives_the_chunked_cards(self):
        producer = ProducerMetadata()
        provenance = {"note": "x" * 500}
        recovered = ArtifactMetadata.from_header(ArtifactMetadata("weight", producer, provenance).to_header())
        self.assertEqual(dict(recovered.provenance), provenance)
        self.assertEqual(recovered.producer, producer)

    def test_an_empty_artifact_type_is_rejected(self):
        with self.assertRaises(ValueError):
            ArtifactMetadata("", ProducerMetadata())


class TestJsonHeaderCards(unittest.TestCase):
    def test_a_short_payload_uses_the_single_legacy_card(self):
        payload = {"a": 1}
        cards = _json_header_cards("WMLEG", "WMX", payload)
        self.assertEqual(set(cards), {"WMLEG"})
        self.assertEqual(json.loads(_json_from_header_cards(cards, "WMLEG", "WMX")), payload)

    def test_the_single_card_boundary_is_sixty_characters(self):
        # json.dumps({"a": "b" * k}, ...) is 8 + k characters long.
        at_limit = {"a": "b" * 52}
        self.assertEqual(len(json.dumps(at_limit, sort_keys=True, separators=(",", ":"))), 60)
        self.assertEqual(set(_json_header_cards("WMLEG", "WMX", at_limit)), {"WMLEG"})

        over_limit = {"a": "b" * 53}
        cards = _json_header_cards("WMLEG", "WMX", over_limit)
        self.assertNotIn("WMLEG", cards)
        self.assertIn("WMXCNT", cards)

    def test_a_long_payload_chunks_into_bounded_cards_and_round_trips(self):
        payload = {"blob": "y" * 200, "count": 7}
        cards = _json_header_cards("WMLEG", "WMX", payload)
        self.assertNotIn("WMLEG", cards)
        for key, value in cards.items():
            with self.subTest(card=key):
                self.assertLessEqual(len(value), 60)
        self.assertEqual(json.loads(_json_from_header_cards(cards, "WMLEG", "WMX")), payload)

    def test_a_missing_payload_reads_as_none(self):
        self.assertIsNone(_json_from_header_cards({}, "WMLEG", "WMX"))

    def test_a_zero_or_unreadable_chunk_count_reads_as_empty(self):
        self.assertEqual(_json_from_header_cards({"WMXCNT": "0"}, "WMLEG", "WMX"), "")
        self.assertEqual(_json_from_header_cards({"WMXCNT": "-1"}, "WMLEG", "WMX"), "")
        self.assertEqual(_json_from_header_cards({"WMXCNT": "not-a-count"}, "WMLEG", "WMX"), "")
        self.assertEqual(_json_from_header_cards({"WMXCNT": "2", "WMX00": "a"}, "WMLEG", "WMX"), "")


class TestValidateConfidencePercentile(unittest.TestCase):
    def test_valid_percentiles_are_accepted(self):
        for percentile in (0.001, 1, 50.0, 99, 100, 100.0):
            with self.subTest(percentile=percentile):
                self.assertIsNone(_validate_confidence_percentile(percentile))

    def test_out_of_range_and_non_finite_values_are_rejected(self):
        for percentile in (0, 0.0, -1, 100.0001, 101, float("nan"), float("inf"), float("-inf")):
            with self.subTest(percentile=percentile):
                with self.assertRaises(ValueError):
                    _validate_confidence_percentile(percentile)

    def test_booleans_are_rejected_even_though_one_is_in_range(self):
        for percentile in (True, False, np.bool_(True)):
            with self.subTest(percentile=percentile):
                with self.assertRaises(ValueError):
                    _validate_confidence_percentile(percentile)

    def test_a_non_numeric_percentile_cannot_be_compared(self):
        # The guard compares before it type-checks, so a string surfaces as a
        # TypeError rather than the ValueError the message describes.
        with self.assertRaises((TypeError, ValueError)):
            _validate_confidence_percentile("99")


class TestWeightMaskProductMaskDtype(unittest.TestCase):
    def test_a_canonical_mask_is_accepted_by_the_product(self):
        mask = canonical_quality_mask(np.zeros((2, 2), dtype=np.uint32), (2, 2))
        zeros = np.zeros((2, 2), dtype=np.float32)
        product = WeightMaskProduct(mask, zeros, zeros, zeros, {})
        self.assertEqual(product.quality_mask.dtype, np.dtype(np.uint32))

    def test_a_non_uint32_mask_is_rejected_by_the_product(self):
        zeros = np.zeros((2, 2), dtype=np.float32)
        with self.assertRaises(TypeError):
            WeightMaskProduct(np.zeros((2, 2), dtype=np.int64), zeros, zeros, zeros, {})


if __name__ == "__main__":
    unittest.main()
