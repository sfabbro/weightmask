"""Public interoperability contract for WeightMask array products.

The pipeline remains NumPy-first.  This module only describes the arrays and
their metadata, so callers may use any FITS (or non-FITS) transport that
implements :class:`ArrayHeaderIO`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum, IntFlag
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

CONTRACT_VERSION = "1.0"
MASK_POLARITY = "set_means_flagged"
# Frozen 0.1 plane: g²F²/(Sg+RN²). Exact Poisson+RN at F=1; Elixir-style F² coadd weight otherwise.
INVERSE_VARIANCE_SEMANTICS = "elixir_style_flat2_coadd_weight"
CONFIDENCE_SEMANTICS = "normalized_weight_0_to_1"
MAX_CONFIDENCE_SAMPLES = 100_000
_FITS_HEADER_VALUE_CHARS = 60


class MaskPolarity(str, Enum):
    """The boolean meaning of a set bit in a quality mask."""

    SET_MEANS_FLAGGED = MASK_POLARITY


class QualityBit(IntFlag):
    """Named quality conditions encoded in the unsigned integer quality mask.

    A set bit means that the named condition is present.  It does not always
    mean that a pixel has zero weight: ``DETECTED`` is informational unless a
    caller elects to exclude detected sources.  ``INVALID_VARIANCE`` is set
    when inverse variance is non-finite or non-positive.
    """

    BAD_PIXEL = 1 << 0
    SATURATED = 1 << 1
    COSMIC_RAY = 1 << 2
    DETECTED = 1 << 3
    STREAK = 1 << 4
    INVALID_VARIANCE = 1 << 5

    # Compact aliases retain the original WeightMask vocabulary.
    BAD = BAD_PIXEL
    SAT = SATURATED
    CR = COSMIC_RAY


QUALITY_BITS = {
    "BAD": int(QualityBit.BAD_PIXEL),
    "SAT": int(QualityBit.SATURATED),
    "CR": int(QualityBit.COSMIC_RAY),
    "DETECTED": int(QualityBit.DETECTED),
    "STREAK": int(QualityBit.STREAK),
    "INVALID_VARIANCE": int(QualityBit.INVALID_VARIANCE),
}
QUALITY_BIT_NAMES = {
    int(bit): bit.name
    for bit in (
        QualityBit.BAD_PIXEL,
        QualityBit.SATURATED,
        QualityBit.COSMIC_RAY,
        QualityBit.DETECTED,
        QualityBit.STREAK,
        QualityBit.INVALID_VARIANCE,
    )
}
DEFAULT_ZERO_WEIGHT_BITS = (
    QualityBit.BAD_PIXEL
    | QualityBit.SATURATED
    | QualityBit.COSMIC_RAY
    | QualityBit.STREAK
    | QualityBit.INVALID_VARIANCE
)


@runtime_checkable
class ArrayHeaderIO(Protocol):
    """Minimal public transport protocol for a named array and its header."""

    def read_array(self, path: str, *, hdu: int = 0) -> tuple[np.ndarray, Mapping[str, Any]]:
        """Return an array and a plain mapping of its header values."""

    def write_array(
        self,
        path: str,
        array: np.ndarray,
        header: Mapping[str, Any],
        *,
        overwrite: bool = False,
    ) -> None:
        """Write an array and header mapping."""


@dataclass(frozen=True)
class ProducerMetadata:
    """Versioned producer identity, with reserved fields for future ML models."""

    name: str = "weightmask"
    version: str = "0.1.0"
    kind: str = "classical"
    algorithm: str = "classical_mask_and_variance"
    model_id: str | None = None
    model_version: str | None = None
    inference_backend: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"classical", "ml"}:
            raise ValueError("producer kind must be 'classical' or 'ml'")
        if not self.name:
            raise ValueError("producer name must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "kind": self.kind,
            "algorithm": self.algorithm,
            "model_id": self.model_id,
            "model_version": self.model_version,
            "inference_backend": self.inference_backend,
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "ProducerMetadata":
        return cls(
            name=str(values.get("name", "weightmask")),
            version=str(values.get("version", "unknown")),
            kind=str(values.get("kind", "classical")),
            algorithm=str(values.get("algorithm", "classical_mask_and_variance")),
            model_id=values.get("model_id"),
            model_version=values.get("model_version"),
            inference_backend=values.get("inference_backend"),
        )


@dataclass(frozen=True)
class ArtifactMetadata:
    """Metadata shared by a contract artifact and its serialised header."""

    artifact_type: str
    producer: ProducerMetadata
    provenance: Mapping[str, Any] = field(default_factory=dict)
    contract_version: str = CONTRACT_VERSION
    mask_polarity: str | None = None
    semantics: str | None = None

    def __post_init__(self) -> None:
        if not self.artifact_type:
            raise ValueError("artifact_type must not be empty")
        if self.mask_polarity not in {None, MASK_POLARITY}:
            raise ValueError(f"mask_polarity must be {MASK_POLARITY!r}")

    def to_header(self) -> dict[str, str]:
        """Return portable FITS-style header values using short keyword names."""
        payload = {
            "producer": self.producer.to_dict(),
            "provenance": dict(self.provenance),
        }
        header = {
            "WMVERS": self.contract_version,
            "WMART": self.artifact_type,
        }
        header.update(_json_header_cards("WMPROV", "WMPV", payload))
        if self.artifact_type == "quality_mask":
            header.update(_json_header_cards("WMQDEF", "WMQD", QUALITY_BIT_NAMES))
        if self.mask_polarity is not None:
            header["WMMASK"] = self.mask_polarity
        if self.semantics is not None:
            header["WMSEM"] = self.semantics
        return header

    @classmethod
    def from_header(cls, header: Mapping[str, Any]) -> "ArtifactMetadata":
        payload_raw = _json_from_header_cards(header, "WMPROV", "WMPV")
        try:
            payload = json.loads(payload_raw)
        except (TypeError, json.JSONDecodeError):
            payload = {}
        return cls(
            artifact_type=str(header.get("WMART", "unknown")),
            contract_version=str(header.get("WMVERS", CONTRACT_VERSION)),
            producer=ProducerMetadata.from_dict(payload.get("producer", {})),
            provenance=payload.get("provenance", {}),
            mask_polarity=header.get("WMMASK"),
            semantics=header.get("WMSEM"),
        )


def _json_header_cards(legacy_key: str, prefix: str, value: Mapping[str, Any]) -> dict[str, str]:
    """Encode JSON without relying on unsupported FITS long-string cards."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    if len(encoded) <= _FITS_HEADER_VALUE_CHARS:
        return {legacy_key: encoded}
    # ponytail: fixed-size cards avoid a dependency on FITS CONTINUE support.
    chunks = [
        encoded[index : index + _FITS_HEADER_VALUE_CHARS] for index in range(0, len(encoded), _FITS_HEADER_VALUE_CHARS)
    ]
    return {f"{prefix}CNT": str(len(chunks))} | {f"{prefix}{index:02d}": chunk for index, chunk in enumerate(chunks)}


def _json_from_header_cards(header: Mapping[str, Any], legacy_key: str, prefix: str) -> str:
    """Read JSON stored in single legacy or chunked FITS header cards."""
    count = header.get(f"{prefix}CNT")
    if count is None:
        return str(header.get(legacy_key, "{}"))
    try:
        return "".join(str(header[f"{prefix}{index:02d}"]) for index in range(int(count)))
    except (KeyError, TypeError, ValueError):
        return "{}"


@dataclass(frozen=True)
class WeightMaskProduct:
    """Contract product arrays plus per-artifact metadata.

    ``quality_mask`` is uint32 and follows ``set_means_flagged`` polarity.
    ``inverse_variance`` and ``weight`` are non-negative float32 arrays.
    ``confidence`` is float32 and normalized to the closed interval [0, 1].
    """

    quality_mask: np.ndarray
    inverse_variance: np.ndarray
    weight: np.ndarray
    confidence: np.ndarray
    metadata: Mapping[str, ArtifactMetadata]


def canonical_quality_mask(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Validate and copy a quality mask into the contract uint32 representation."""
    array = np.asarray(mask)
    if array.shape != shape:
        raise ValueError(f"quality mask shape {array.shape} does not match data shape {shape}")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("quality mask must have an integer dtype")
    return array.astype(np.uint32, copy=True)


def valid_inverse_variance(inverse_variance: np.ndarray) -> np.ndarray:
    """Return where inverse variance has the contract's usable meaning (> 0)."""
    array = np.asarray(inverse_variance)
    return np.isfinite(array) & (array > 0)


def quality_bit_names(value: int | QualityBit) -> frozenset[str]:
    """Decode an integer quality value into its stable public bit names."""
    value = int(value)
    return frozenset(name for bit, name in QUALITY_BIT_NAMES.items() if value & bit)


def quality_bits_from_names(names: list[str] | tuple[str, ...] | set[str]) -> QualityBit:
    """Encode public quality-bit names into one integer flag value."""
    value = QualityBit(0)
    for name in names:
        try:
            value |= QualityBit[name]
        except KeyError as exc:
            raise ValueError(f"unknown quality-bit name: {name}") from exc
    return value


def _bounded_percentile(values: np.ndarray, percentile: float) -> float:
    """Calculate a deterministic percentile without survey-scale global work."""
    # ponytail: regular striding caps percentile memory/work at 100k values;
    # upgrade to a streaming quantile estimator only if this ceiling is insufficient.
    step = max(1, (values.size + MAX_CONFIDENCE_SAMPLES - 1) // MAX_CONFIDENCE_SAMPLES)
    return float(np.percentile(values[::step], percentile))


def build_weight_product(
    inverse_variance: np.ndarray,
    quality_mask: np.ndarray | None = None,
    *,
    exclude_detected: bool = False,
    confidence_percentile: float = 99.0,
    producer: ProducerMetadata | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> WeightMaskProduct:
    """Build contract arrays from a classical inverse-variance estimate.

    No learned mask or weight generation occurs here.  Invalid inverse variance
    (NaN, infinity, zero, or negative) is represented as zero weight and marked
    with :class:`QualityBit.INVALID_VARIANCE`.
    """
    inverse_variance = np.asarray(inverse_variance)
    if inverse_variance.ndim == 0:
        raise ValueError("inverse variance must be an array")
    if not 0 < confidence_percentile <= 100:
        raise ValueError("confidence_percentile must be in (0, 100]")

    raw_mask = np.zeros(inverse_variance.shape, dtype=np.uint32) if quality_mask is None else quality_mask
    mask = canonical_quality_mask(raw_mask, inverse_variance.shape)
    valid = valid_inverse_variance(inverse_variance)
    mask[~valid] |= np.uint32(QualityBit.INVALID_VARIANCE)

    exclusion = DEFAULT_ZERO_WEIGHT_BITS
    if exclude_detected:
        exclusion |= QualityBit.DETECTED
    usable = valid & ((mask & np.uint32(exclusion)) == 0)

    ivar = np.zeros(inverse_variance.shape, dtype=np.float32)
    ivar[valid] = inverse_variance[valid].astype(np.float32, copy=False)
    weight = np.zeros_like(ivar)
    weight[usable] = ivar[usable]

    confidence = np.zeros_like(weight)
    positive = weight[weight > 0]
    if positive.size:
        normalization = _bounded_percentile(positive, confidence_percentile)
        if np.isfinite(normalization) and normalization > 0:
            confidence = np.clip(weight / normalization, 0.0, 1.0).astype(np.float32, copy=False)

    producer = producer or ProducerMetadata()
    provenance = dict(provenance or {})
    metadata = {
        "quality_mask": ArtifactMetadata(
            "quality_mask", producer, provenance, mask_polarity=MASK_POLARITY, semantics="named_quality_bits"
        ),
        "inverse_variance": ArtifactMetadata(
            "inverse_variance", producer, provenance, semantics=INVERSE_VARIANCE_SEMANTICS
        ),
        "weight": ArtifactMetadata("weight", producer, provenance, semantics="masked_inverse_variance"),
        "confidence": ArtifactMetadata("confidence", producer, provenance, semantics=CONFIDENCE_SEMANTICS),
    }
    return WeightMaskProduct(mask, ivar, weight, confidence, metadata)
