import sys
from types import ModuleType

import numpy as np
import pytest

import weightmask
from tests.fixtures.wave5_contract import FIXTURE_LABEL, classical_contract_fixture
from weightmask.contract import (
    CONFIDENCE_SEMANTICS,
    INVERSE_VARIANCE_SEMANTICS,
    MASK_POLARITY,
    MAX_CONFIDENCE_SAMPLES,
    ArtifactMetadata,
    ProducerMetadata,
    QualityBit,
    build_weight_product,
    quality_bit_names,
    quality_bits_from_names,
)
from weightmask.torchfits_adapter import TorchfitsArrayHeaderIO, TorchfitsUnavailableError, torchfits_available


def test_wave5_fixture_has_a_stable_label_and_explicit_polarity():
    inverse_variance, quality_mask = classical_contract_fixture()
    product = build_weight_product(inverse_variance, quality_mask)

    assert FIXTURE_LABEL == "wave5-classical-3x3-v1"
    assert product.metadata["quality_mask"].mask_polarity == MASK_POLARITY
    assert product.weight[2, 0] == 1.0  # DETECTED is informative by default.
    assert product.weight[1, 1] == 0.0  # SATURATED is excluded.


def test_invalid_variance_is_zero_weight_and_marked():
    inverse_variance, quality_mask = classical_contract_fixture()
    product = build_weight_product(inverse_variance, quality_mask)

    invalid = ~np.isfinite(inverse_variance) | (inverse_variance <= 0)
    assert np.all(product.inverse_variance[invalid] == 0)
    assert np.all(product.weight[invalid] == 0)
    assert np.all((product.quality_mask[invalid] & QualityBit.INVALID_VARIANCE) != 0)
    assert product.confidence.dtype == np.float32
    assert 0.0 <= product.confidence.min() <= product.confidence.max() <= 1.0


def test_confidence_percentile_is_deterministically_bounded(monkeypatch):
    inverse_variance = np.linspace(1.0, 2.0, 2 * MAX_CONFIDENCE_SAMPLES, dtype=np.float32)
    percentile_inputs = []
    original_percentile = np.percentile

    def bounded_percentile(values, percentile):
        percentile_inputs.append(values.size)
        return original_percentile(values, percentile)

    monkeypatch.setattr("weightmask.contract.np.percentile", bounded_percentile)
    first = build_weight_product(inverse_variance).confidence
    second = build_weight_product(inverse_variance).confidence

    assert percentile_inputs == [MAX_CONFIDENCE_SAMPLES, MAX_CONFIDENCE_SAMPLES]
    np.testing.assert_array_equal(first, second)


def test_quality_bits_round_trip_through_integer_mask():
    bits = QualityBit.BAD_PIXEL | QualityBit.COSMIC_RAY | QualityBit.STREAK
    restored = QualityBit(np.uint32(bits))

    assert restored & QualityBit.BAD_PIXEL
    assert restored & QualityBit.COSMIC_RAY
    assert restored & QualityBit.STREAK
    assert not restored & QualityBit.SATURATED
    assert quality_bit_names(restored) == {"BAD_PIXEL", "COSMIC_RAY", "STREAK"}
    assert quality_bits_from_names(quality_bit_names(restored)) == restored


def test_producer_and_artifact_metadata_round_trip_with_ml_fields_reserved():
    producer = ProducerMetadata(
        name="weightmask",
        version="1.2.3",
        kind="classical",
        algorithm="robust_classical",
        model_id=None,
        model_version=None,
        inference_backend=None,
    )
    artifact = ArtifactMetadata(
        "quality_mask",
        producer,
        provenance={"input_id": "fixture-42", "config_digest": "abc"},
        mask_polarity=MASK_POLARITY,
        semantics="named_quality_bits",
    )

    restored = ArtifactMetadata.from_header(artifact.to_header())
    assert restored.producer == producer
    assert restored.provenance == artifact.provenance
    assert restored.mask_polarity == MASK_POLARITY
    assert "WMQDCNT" in artifact.to_header()
    assert CONFIDENCE_SEMANTICS == "normalized_weight_0_to_1"
    assert INVERSE_VARIANCE_SEMANTICS == "elixir_style_flat2_coadd_weight"
    assert weightmask.__version__ == "0.1.0"


def test_torchfits_adapter_uses_only_present_public_root_api():
    calls = []
    module = ModuleType("torchfits")

    def read(path, *, hdu, return_header):
        calls.append(("read", path, hdu, return_header))
        return np.array([[3]], dtype=np.uint32), {"EXTNAME": "MASK"}

    module.read = read
    adapter = TorchfitsArrayHeaderIO(module)
    array, header = adapter.read_array("fixture.fits", hdu=2)

    np.testing.assert_array_equal(array, [[3]])
    assert header == {"EXTNAME": "MASK"}
    assert calls == [("read", "fixture.fits", 2, True)]


def test_torchfits_adapter_writes_through_public_root_api(monkeypatch):
    calls = []
    module = ModuleType("torchfits")
    torch = ModuleType("torch")
    torch.as_tensor = lambda value: ("tensor", np.asarray(value))

    def write(path, data, *, header, overwrite):
        calls.append((path, data, header, overwrite))

    module.write = write
    monkeypatch.setitem(sys.modules, "torch", torch)
    TorchfitsArrayHeaderIO(module).write_array("out.fits", np.array([5]), {"WMVERS": "1.0"}, overwrite=True)

    assert calls[0][0] == "out.fits"
    np.testing.assert_array_equal(calls[0][1][1], [5])
    assert calls[0][2] == {"WMVERS": "1.0"}
    assert calls[0][3] is True


def test_torchfits_adapter_reports_absence_without_importing_private_modules(monkeypatch):
    def missing_torchfits(name, *args, **kwargs):
        if name == "torchfits":
            raise ImportError("not installed")
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr("weightmask.torchfits_adapter.importlib.import_module", missing_torchfits)
    assert not torchfits_available()
    with pytest.raises(TorchfitsUnavailableError):
        TorchfitsArrayHeaderIO()


@pytest.mark.skipif(not torchfits_available(), reason="torchfits is optional")
def test_metadata_round_trips_through_real_torchfits_header(tmp_path):
    artifact = ArtifactMetadata(
        "quality_mask",
        ProducerMetadata(version="1.2.3"),
        provenance={"input_id": "fixture-42", "config_digest": "abc"},
        mask_polarity=MASK_POLARITY,
        semantics="named_quality_bits",
    )
    adapter = TorchfitsArrayHeaderIO()
    path = tmp_path / "contract.fits"
    adapter.write_array(str(path), np.zeros((2, 2), dtype=np.uint32), artifact.to_header(), overwrite=True)

    _, header = adapter.read_array(str(path))
    restored = ArtifactMetadata.from_header(header)
    assert restored == artifact
