"""Public interoperability contract for WeightMask array products.

The pipeline remains NumPy-first.  This module only describes the arrays and
their metadata, so callers may use any FITS (or non-FITS) transport that
implements :class:`ArrayHeaderIO`.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import dataclass, field
from enum import Enum, IntFlag
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from ._version import __version__

CONTRACT_VERSION = "1.0"
MASK_POLARITY = "set_means_flagged"
# The selected variance method and flat convention belong in provenance.
INVERSE_VARIANCE_SEMANTICS = "inverse_variance_adu^-2"
CONFIDENCE_SEMANTICS = "normalized_weight_0_to_1"
# `confidence_params.scale_to_100` multiplies the map by 100 before it is
# written, so the card must describe the range the file actually holds.
CONFIDENCE_SEMANTICS_SCALED = "normalized_weight_0_to_100"
MAX_CONFIDENCE_SAMPLES = 100_000
_CONFIDENCE_SAMPLE_CHUNK_SIZE = 65_536
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
    NO_DATA = 1 << 6


QUALITY_BITS = {
    "BAD": int(QualityBit.BAD_PIXEL),
    "SAT": int(QualityBit.SATURATED),
    "CR": int(QualityBit.COSMIC_RAY),
    "DETECTED": int(QualityBit.DETECTED),
    "STREAK": int(QualityBit.STREAK),
    "INVALID_VARIANCE": int(QualityBit.INVALID_VARIANCE),
    "NO_DATA": int(QualityBit.NO_DATA),
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
        QualityBit.NO_DATA,
    )
}
DEFAULT_ZERO_WEIGHT_BITS = (
    QualityBit.BAD_PIXEL
    | QualityBit.SATURATED
    | QualityBit.COSMIC_RAY
    | QualityBit.STREAK
    | QualityBit.INVALID_VARIANCE
    | QualityBit.NO_DATA
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
    version: str = __version__
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
        # Distinguish "this product carries no provenance" from "the provenance
        # is there but unreadable". The second means a truncated write or a
        # perturbed FITS header, and collapsing it to an empty payload produced
        # a plausible-looking ProducerMetadata(version="unknown") that a caller
        # could not tell from a genuine "unknown".
        payload: dict = {}
        if payload_raw is None:
            provenance_state = "absent"
        else:
            try:
                decoded = json.loads(payload_raw)
            except (TypeError, json.JSONDecodeError) as exc:
                warnings.warn(
                    f"Provenance cards are present but could not be decoded ({exc}); treating the producer as unknown.",
                    RuntimeWarning,
                )
                provenance_state = "undecodable"
            else:
                if not isinstance(decoded, dict):
                    warnings.warn(
                        f"Provenance payload is {type(decoded).__name__}, not an object; "
                        f"treating the producer as unknown.",
                        RuntimeWarning,
                    )
                    provenance_state = "not_an_object"
                else:
                    payload = decoded
                    provenance_state = "ok"
        producer = payload.get("producer") or {}
        if provenance_state != "ok" and producer:
            warnings.warn(
                f"Provenance is {provenance_state} but a producer record was expected; "
                f"treating the producer as unknown.",
                RuntimeWarning,
            )
        return cls(
            artifact_type=str(header.get("WMART", "unknown")),
            contract_version=str(header.get("WMVERS", CONTRACT_VERSION)),
            producer=ProducerMetadata.from_dict(producer if isinstance(producer, dict) else {}),
            provenance=payload.get("provenance", {}),
            mask_polarity=header.get("WMMASK"),
            semantics=header.get("WMSEM"),
        )


def _json_header_cards(legacy_key: str, prefix: str, value: Mapping[str, Any]) -> dict[str, str]:
    """Encode JSON without relying on unsupported FITS long-string cards."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    if len(encoded) <= _FITS_HEADER_VALUE_CHARS:
        return {legacy_key: encoded}
    # Fixed-size cards avoid a dependency on FITS CONTINUE support.
    chunks = [
        encoded[index : index + _FITS_HEADER_VALUE_CHARS] for index in range(0, len(encoded), _FITS_HEADER_VALUE_CHARS)
    ]
    return {f"{prefix}CNT": str(len(chunks))} | {f"{prefix}{index:02d}": chunk for index, chunk in enumerate(chunks)}


def _json_from_header_cards(header: Mapping[str, Any], legacy_key: str, prefix: str) -> str | None:
    """Read JSON stored in single legacy or chunked FITS header cards."""
    count = header.get(f"{prefix}CNT")
    if count is None:
        return str(header[legacy_key]) if legacy_key in header else None
    try:
        count = int(count)
        if count <= 0:
            return ""
        return "".join(str(header[f"{prefix}{index:02d}"]) for index in range(count))
    except (KeyError, TypeError, ValueError):
        return ""


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

    def __post_init__(self):
        shape = np.shape(self.quality_mask)
        for name, arr in (
            ("inverse_variance", self.inverse_variance),
            ("weight", self.weight),
            ("confidence", self.confidence),
        ):
            if np.shape(arr) != shape:
                raise ValueError(f"WeightMaskProduct {name} shape {np.shape(arr)} != quality_mask shape {shape}")
        if not isinstance(self.quality_mask, np.ndarray) or self.quality_mask.dtype != np.dtype(np.uint32):
            raise TypeError("WeightMaskProduct quality_mask must be a numpy ndarray with dtype uint32")
        if not np.issubdtype(self.inverse_variance.dtype, np.floating):
            raise TypeError("WeightMaskProduct inverse_variance must have a float dtype")
        if not np.issubdtype(self.weight.dtype, np.floating):
            raise TypeError("WeightMaskProduct weight must have a float dtype")
        if not np.issubdtype(self.confidence.dtype, np.floating):
            raise TypeError("WeightMaskProduct confidence must have a float dtype")
        # `x < 0` is False for NaN, so a plain range test lets non-finite
        # values through. inf also passes `weight < 0`.
        if not np.all(np.isfinite(self.weight)):
            raise ValueError("WeightMaskProduct weight must be finite")
        if not np.all(np.isfinite(self.confidence)):
            raise ValueError("WeightMaskProduct confidence must be finite")
        if not np.all(np.isfinite(self.inverse_variance)):
            raise ValueError("WeightMaskProduct inverse_variance must be finite")
        if np.any(self.inverse_variance < 0):
            raise ValueError("WeightMaskProduct inverse_variance must be non-negative")
        if np.any(self.weight < 0):
            raise ValueError("WeightMaskProduct weight must be non-negative")
        if np.any((self.confidence < 0) | (self.confidence > 1)):
            raise ValueError("WeightMaskProduct confidence must be in [0, 1]")


def canonical_quality_mask(mask: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    """Validate and copy a quality mask into the contract uint32 representation.

    The copy is required, not an optimization: :func:`build_weight_product`
    sets ``INVALID_VARIANCE`` bits on the result, so a ``copy=False`` cast of a
    uint32 input would silently mutate the caller's mask.
    """
    array = np.asarray(mask)
    if array.shape != shape:
        raise ValueError(f"quality mask shape {array.shape} does not match data shape {shape}")
    if array.dtype.kind not in "iu":
        raise TypeError("quality mask must have an integer dtype")
    if np.issubdtype(array.dtype, np.signedinteger) and np.any(array < 0):
        raise ValueError("quality mask values must be non-negative")
    if np.any(array > np.iinfo(np.uint32).max):
        raise ValueError(
            "quality mask values must fit within uint32; values wider than uint32 "
            "would be silently discarded by the contract representation"
        )
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


def _validate_confidence_percentile(percentile):
    if isinstance(percentile, (bool, np.bool_)) or not 0 < percentile <= 100:
        raise ValueError("confidence_percentile must be in (0, 100]")


def _priority_keys(identity, pixel_indices):
    encoded = str(identity).encode("utf-8", errors="surrogatepass")
    seed = np.uint64(int.from_bytes(hashlib.blake2b(encoded, digest_size=8, person=b"wm-conf").digest(), "little"))
    with np.errstate(over="ignore"):
        keys = pixel_indices.astype(np.uint64, copy=False) + seed + np.uint64(0x9E3779B97F4A7C15)
        keys = (keys ^ (keys >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        keys = (keys ^ (keys >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return keys ^ (keys >> np.uint64(31))


class _BoundedPrioritySampler:
    def __init__(self, max_samples: int = MAX_CONFIDENCE_SAMPLES):
        if isinstance(max_samples, (bool, np.bool_)) or max_samples <= 0:
            raise ValueError("max_samples must be positive")
        self.max_samples = int(max_samples)
        self._priorities = np.empty(0, dtype=np.uint64)
        self._identity_labels = np.empty(0, dtype=object)
        self._pixel_indices = np.empty(0, dtype=np.uint64)
        self._values = np.empty(0, dtype=np.float64)
        self._offsets = {}

    def update(self, identity, values):
        identity_label = str(identity)
        array = np.asarray(values)
        flat = array.flat
        stream_offset = self._offsets.get(identity_label, 0)
        for start in range(0, array.size, _CONFIDENCE_SAMPLE_CHUNK_SIZE):
            chunk = np.asarray(flat[start : start + _CONFIDENCE_SAMPLE_CHUNK_SIZE])
            offsets = np.flatnonzero(np.isfinite(chunk) & (chunk > 0))
            if not offsets.size:
                continue
            candidate_values = chunk[offsets]
            candidate_pixels = offsets.astype(np.uint64) + np.uint64(stream_offset + start)
            candidate_priorities = _priority_keys(identity_label, candidate_pixels)
            candidate_identities = np.full(offsets.size, identity_label, dtype=object)
            priorities = np.concatenate((self._priorities, candidate_priorities))
            identity_labels = np.concatenate((self._identity_labels, candidate_identities))
            pixel_indices = np.concatenate((self._pixel_indices, candidate_pixels))
            sample_values = np.concatenate((self._values, candidate_values))
            if priorities.size > self.max_samples:
                cutoff = np.partition(priorities, self.max_samples - 1)[self.max_samples - 1]
                below = np.flatnonzero(priorities < cutoff)
                tied = np.flatnonzero(priorities == cutoff)
                needed = self.max_samples - below.size
                if tied.size > needed:
                    tie_order = np.lexsort((pixel_indices[tied], identity_labels[tied]))
                    tied = tied[tie_order[:needed]]
                keep = np.concatenate((below, tied))
                priorities = priorities[keep]
                identity_labels = identity_labels[keep]
                pixel_indices = pixel_indices[keep]
                sample_values = sample_values[keep]
            self._priorities = priorities
            self._identity_labels = identity_labels
            self._pixel_indices = pixel_indices
            self._values = sample_values
        self._offsets[identity_label] = stream_offset + array.size

    def sample(self):
        if not self._priorities.size:
            return self._values
        order = np.lexsort((self._pixel_indices, self._identity_labels, self._priorities))
        return self._values[order]


def _bounded_percentile(values: np.ndarray, percentile: float) -> float:
    """Calculate a deterministic percentile without survey-scale global work."""
    _validate_confidence_percentile(percentile)
    sampler = _BoundedPrioritySampler()
    sampler.update("weight", values)
    sample = sampler.sample()
    if not sample.size:
        return float("nan")
    return float(np.percentile(sample, percentile))


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
    _validate_confidence_percentile(confidence_percentile)

    raw_mask = np.zeros(inverse_variance.shape, dtype=np.uint32) if quality_mask is None else quality_mask
    mask = canonical_quality_mask(raw_mask, inverse_variance.shape)
    # Validity applies to the array we return: a finite float64 can overflow
    # or underflow when represented by the contract's float32 plane.
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        inverse_variance = inverse_variance.astype(np.float32)
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
    normalization = _bounded_percentile(weight, confidence_percentile)
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
