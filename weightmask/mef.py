"""MEF orchestration: per-HDU processing, streaming writers, parallel fan-out."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import threading
import uuid
from typing import Optional, Tuple

import fitsio
import numpy as np
from astropy.wcs import WCS
from scipy import ndimage

from . import __version__
from .bad import _get_global_median, compute_flat_bad_mask_cached, detect_dark_hot_pixels
from .contract import (
    CONFIDENCE_SEMANTICS,
    CONFIDENCE_SEMANTICS_SCALED,
    CONTRACT_VERSION,
    DEFAULT_ZERO_WEIGHT_BITS,
    INVERSE_VARIANCE_SEMANTICS,
    MASK_POLARITY,
    ArtifactMetadata,
    ProducerMetadata,
    QualityBit,
    _BoundedPrioritySampler,
    build_weight_product,
)
from .errors import StageFailure
from .process import _effective_tile_size, process_image


def process_hdu(
    hdu_sci,
    hdu_flat,
    config: dict,
    hdu_index: int,
    tile_size: int = 1024,
    flat_path: Optional[str] = None,
    hdu_badpix=None,
    precomputed_bad_mask: Optional[np.ndarray] = None,
    detector_prior: Optional[np.ndarray] = None,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[dict],
]:
    """Processes a single Science HDU to generate all mask and map products."""
    hdu_name = getattr(hdu_sci, "name", f"HDU{hdu_index}") if hasattr(hdu_sci, "name") else f"HDU{hdu_index}"
    print(f"\n--- Processing HDU {hdu_index} ({hdu_name}) ---")

    try:
        sci_data_full = np.asarray(hdu_sci.read())
        sci_hdr = hdu_sci.read_header()
    except (OSError, fitsio.FITSFormatError) as e:
        print(f"Skipping HDU: Cannot read science data: {e}")
        return None, None, None, None, None, None

    flat_data_full = None
    if hdu_flat is not None:
        # Read the flat unconditionally: process_image uses it as the actual
        # flat field for the variance/weight math, not just for the bad mask.
        try:
            flat_data_full = np.ascontiguousarray(hdu_flat.read().astype(np.float32, copy=False))
        except (OSError, fitsio.FITSFormatError) as e:
            print(f"Skipping HDU: Cannot read flat data: {e}")
            return None, None, None, None, None, None

    bad_mask = None
    if precomputed_bad_mask is not None:
        bad_mask = precomputed_bad_mask
    elif flat_data_full is not None:
        eff = _effective_tile_size(tile_size, sci_data_full.shape)
        bad_mask = compute_flat_bad_mask_cached(
            flat_data_full,
            config.get("flat_masking", {}),
            eff,
            flat_path=flat_path,
            hdu_index=hdu_index,
        )

    badpix_mask = None
    if hdu_badpix is not None:
        try:
            ext = np.asarray(hdu_badpix.read())
            if ext.shape == sci_data_full.shape:
                if not np.all((ext == 0) | (ext == 1)):
                    print("Skipping HDU: Provided keep-map must contain only 0 (bad) and 1 (good).")
                    return None, None, None, None, None, None
                badpix_mask = ext == 0
                frac = float(np.mean(badpix_mask))
                thresh = float(config.get("flat_masking", {}).get("dead_ccd_badpix_fraction", 0.9))
                if frac > thresh:
                    print(f"  NOTE: external mask flags {frac:.1%} of HDU {hdu_index} bad (dead CCD?) -- zero weight.")
            else:
                print(f"Skipping HDU: Provided keep-map shape mismatch {ext.shape} != {sci_data_full.shape}.")
                return None, None, None, None, None, None
        except (OSError, fitsio.FITSFormatError, TypeError, ValueError) as e:
            print(f"Skipping HDU: Cannot read provided keep-map: {e}")
            return None, None, None, None, None, None

    return process_image(
        sci_data_full,
        sci_hdr,
        flat_data_full,
        config,
        tile_size,
        bad_mask=bad_mask,
        badpix_mask=badpix_mask,
        detector_prior=detector_prior,
    )


def line_geometry(ys, xs, shape):
    """CCD-local line of a streak: angle in degrees and offset from the chip centre.

    The normal is sign-canonicalised so two copies of the same column compare
    equal. Common-frame geometry must resolve whether equal local lines are a
    sky-consistent trail or detector-fixed replicas.
    """
    ys = np.asarray(ys)
    xs = np.asarray(xs)
    if xs.size < 24:
        return None
    x = xs.astype(np.float64)
    y = ys.astype(np.float64)
    mx, my = float(x.mean()), float(y.mean())
    xc, yc = x - mx, y - my
    cxx = float(np.dot(xc, xc))
    cyy = float(np.dot(yc, yc))
    cxy = float(np.dot(xc, yc))
    theta = 0.5 * float(np.arctan2(2.0 * cxy, cxx - cyy))
    direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    h, w = shape
    cx, cy = 0.5 * (w - 1), 0.5 * (h - 1)
    offset = float(normal[0] * (mx - cx) + normal[1] * (my - cy))
    if normal[0] < 0.0 or (normal[0] == 0.0 and normal[1] < 0.0):
        offset = -offset
    angle = float(np.degrees(np.arctan2(direction[1], direction[0])) % 180.0)
    return {
        "offset_px": offset,
        "angle_deg": angle,
        "point": [mx, my],
        "direction": direction.tolist(),
    }


def _angle_sep_deg(left, right):
    sep = abs(float(left) - float(right)) % 180.0
    return min(sep, 180.0 - sep)


def replica_indices(geometries, tol_px=25.0, tol_deg=1.5):
    """Indices whose CCD-local streak line is shared with another chip."""
    replicas = set()
    for i in range(len(geometries)):
        left = geometries[i]
        if left is None:
            continue
        for j in range(i + 1, len(geometries)):
            right = geometries[j]
            if right is None:
                continue
            if _angle_sep_deg(left["angle_deg"], right["angle_deg"]) > tol_deg:
                continue
            if abs(left["offset_px"] - right["offset_px"]) > tol_px:
                continue
            replicas.add(i)
            replicas.add(j)
    return replicas


def _replica_component_indices(catalogs, tol_px=25.0, tol_deg=1.5):
    peers = [set() for _ in catalogs]
    detector_fixed_edges = set()
    for i in range(len(catalogs)):
        left = catalogs[i]
        left_local = left.get("local")
        if left_local is None:
            continue
        for j in range(i + 1, len(catalogs)):
            right = catalogs[j]
            if left.get("hdu") == right.get("hdu"):
                continue
            right_local = right.get("local")
            if right_local is None:
                continue
            if _angle_sep_deg(left_local["angle_deg"], right_local["angle_deg"]) > tol_deg:
                continue
            if abs(left_local["offset_px"] - right_local["offset_px"]) > tol_px:
                continue
            peers[i].add(j)
            peers[j].add(i)
            left_common = left.get("common")
            right_common = right.get("common")
            if left_common is None or right_common is None:
                continue
            if np.allclose(left_common["crval"], right_common["crval"], rtol=0.0, atol=1.0e-8):
                common_angle = _angle_sep_deg(left_common["angle_deg"], right_common["angle_deg"])
                scale = max(left_common["scale_deg_per_px"], right_common["scale_deg_per_px"])
                common_offset = abs(left_common["offset_deg"] - right_common["offset_deg"])
                if common_angle > tol_deg or common_offset > tol_px * scale:
                    detector_fixed_edges.add((i, j))
    replicas = set()
    for i, j in detector_fixed_edges:
        if peers[i] == {j} and peers[j] == {i}:
            replicas.update((i, j))
    return replicas


def extract_individual_masks(header_info: dict, mask_data):
    keys = ("bad", "sat", "cr", "obj", "streak", "nodata")
    source = header_info.get("individual_masks", {}) if header_info else {}
    if mask_data is None:
        return tuple(np.array([]) for _ in keys)
    shape = mask_data.shape
    out = []
    for k in keys:
        if k in source:
            out.append(source[k])
        else:
            print(f"  WARNING: individual mask '{k}' missing from header_info; emitting all-False.")
            out.append(np.zeros(shape, dtype=bool))
    return tuple(out)


def _store_individual_masks(
    output_data,
    i,
    hdu_header,
    hdu_name,
    bad_mask,
    sat_mask,
    cr_mask,
    obj_mask,
    streak_mask,
    nodata_mask,
):
    if i not in output_data:
        output_data[i] = {}
    output_data[i]["individual_masks"] = {
        "bad": {
            "data": bad_mask.astype(np.uint8),
            "header": _header_with_contract_metadata(hdu_header, "bad_mask", mask=True, semantics="boolean_mask"),
            "name": f"BAD_{hdu_name}",
        },
        "sat": {
            "data": sat_mask.astype(np.uint8),
            "header": _header_with_contract_metadata(
                hdu_header, "saturation_mask", mask=True, semantics="boolean_mask"
            ),
            "name": f"SAT_{hdu_name}",
        },
        "cr": {
            "data": cr_mask.astype(np.uint8),
            "header": _header_with_contract_metadata(
                hdu_header, "cosmic_ray_mask", mask=True, semantics="boolean_mask"
            ),
            "name": f"CR_{hdu_name}",
        },
        "obj": {
            "data": obj_mask.astype(np.uint8),
            "header": _header_with_contract_metadata(hdu_header, "object_mask", mask=True, semantics="boolean_mask"),
            "name": f"OBJ_{hdu_name}",
        },
        "streak": {
            "data": streak_mask.astype(np.uint8),
            "header": _header_with_contract_metadata(hdu_header, "streak_mask", mask=True, semantics="boolean_mask"),
            "name": f"STREAK_{hdu_name}",
        },
        "nodata": {
            "data": nodata_mask.astype(np.uint8),
            "header": _header_with_contract_metadata(hdu_header, "nodata_mask", mask=True, semantics="boolean_mask"),
            "name": f"NODATA_{hdu_name}",
        },
    }


def _assign_map_if_valid(output_data, i, key, data, header, hdu_name):
    if data is not None:
        if i not in output_data:
            output_data[i] = {}
        output_data[i][key] = {
            "data": data,
            "header": header,
            "name": f"{key.upper()}_{hdu_name}",
        }


# FITS reserved keywords from the tiled-image compression (fpack) convention.
# They describe the *on-disk* encoding of the input image, not the science
# pixels; copying them onto an uncompressed output makes readers apply the
# compression scaling a second time and corrupts every output value.
_COMPRESSION_HEADER_KEYS = (
    "BZERO",
    "BSCALE",
    "BITPIX",
    "ZQUANTIZ",
    "ZBLANK",
    "ZSIMPLE",
    "ZCMPTYPE",
    "ZNAXIS",
    "ZBITPIX",
    "ZDITHER0",
)


def _strip_compression_keywords(header):
    """Remove fpack/quantization keywords from an output header.

    The input HDU header is reused for the output products, but the outputs are
    written as ordinary (uncompressed) FITS. The input's ``BZERO``/``BSCALE``/
    ``BITPIX`` (and the ``Z*`` tile-compression cards) must not be copied,
    otherwise fitsio applies the compression scaling to the already-scaled
    output values -- e.g. a [0,1] weight map came back offset by ``BZERO=32668``
    when the input was fpacked.
    """
    if header is None:
        return header
    out = dict(header)
    for key in list(out.keys()):
        if (
            key in _COMPRESSION_HEADER_KEYS
            or key.startswith("ZTILE")
            or key.startswith("ZNAME")
            or key.startswith("ZVAL")
        ):
            out.pop(key, None)
    return out


def _header_with_contract_metadata(header, artifact_type: str, *, mask: bool = False, semantics: str | None = None):
    """Copy a FITS-style header and add portable Wave 5 contract metadata."""
    try:
        output_header = dict(header or {})
    except (TypeError, ValueError):
        output_header = {}
    metadata = ArtifactMetadata(
        artifact_type,
        ProducerMetadata(version=__version__),
        provenance={"producer_stage": "weightmask.cli"},
        mask_polarity=MASK_POLARITY if mask else None,
        semantics=semantics,
    )
    output_header.update(metadata.to_header())
    return output_header


def _store_output_maps(
    output_data,
    i,
    hdu_header,
    hdu_name,
    config,
    confidence_map,
    weight_map,
    mask_data,
    inv_var_data,
    sky_map,
    args,
    bad_mask,
    sat_mask,
    cr_mask,
    obj_mask,
    streak_mask,
    nodata_mask,
    paths,
    sky_header=None,
):
    if paths["out_map_path"]:
        output_format = config.get("output_params", {}).get("output_map_format", "weight").lower()
        map_data = confidence_map if output_format == "confidence" else weight_map
        if (
            output_format == "confidence"
            and config.get("confidence_params", {}).get("normalize_scope") == "per_exposure"
        ):
            # Normalize once after streaming; per-HDU clipping loses high weights.
            map_data = weight_map
        if output_format != "confidence":
            semantics = "masked_inverse_variance"
        elif config.get("confidence_params", {}).get("scale_to_100", False):
            semantics = CONFIDENCE_SEMANTICS_SCALED
        else:
            semantics = CONFIDENCE_SEMANTICS
        _assign_map_if_valid(
            output_data,
            i,
            "map",
            map_data,
            _header_with_contract_metadata(hdu_header, output_format, semantics=semantics),
            hdu_name,
        )
    if paths["out_mask_path"]:
        _assign_map_if_valid(
            output_data,
            i,
            "mask",
            mask_data,
            _header_with_contract_metadata(hdu_header, "quality_mask", mask=True, semantics="named_quality_bits"),
            hdu_name,
        )
    if paths["out_invvar_path"]:
        _assign_map_if_valid(
            output_data,
            i,
            "invvar",
            inv_var_data,
            _header_with_contract_metadata(hdu_header, "inverse_variance", semantics=INVERSE_VARIANCE_SEMANTICS),
            hdu_name,
        )
    if paths["out_sky_path"]:
        sky_output_header = sky_header if sky_header is not None else hdu_header
        sky_artifact = "sky_mesh" if (sky_header or {}).get("SKYMESH", False) else "sky"
        _assign_map_if_valid(
            output_data,
            i,
            "sky",
            sky_map,
            _header_with_contract_metadata(sky_output_header, sky_artifact, semantics="background_adu"),
            hdu_name,
        )
    if paths["out_weight_raw_path"] and weight_map is not None:
        if i not in output_data:
            output_data[i] = {}
        output_data[i]["weight_raw"] = {
            "data": weight_map,
            "header": _header_with_contract_metadata(hdu_header, "weight", semantics="masked_inverse_variance"),
            "name": f"WEIGHT_{hdu_name}",
        }
    if args.individual_masks and mask_data is not None:
        _store_individual_masks(
            output_data,
            i,
            hdu_header,
            hdu_name,
            bad_mask,
            sat_mask,
            cr_mask,
            obj_mask,
            streak_mask,
            nodata_mask,
        )


def _rescale_confidence_to_global(paths, writers, config):
    """Normalize streamed raw weights once to exposure-global confidence."""
    if config.get("output_params", {}).get("output_map_format", "weight").lower() != "confidence":
        print("  Confidence global norm skipped (map product holds weight, not confidence).")
        return True
    map_path = (paths or {}).get("out_map_path")
    writer = (writers or {}).get("map")
    if not map_path or writer is None:
        return True
    percentile = config.get("confidence_params", {}).get("normalize_percentile", 99.0)
    upper = 100.0 if (config or {}).get("confidence_params", {}).get("scale_to_100", False) else 1.0
    entries = list(writer.positions.items())
    writer_identities = getattr(writer, "identities", {})
    identities = [str(writer_identities.get(hdu_index, f"HDU{hdu_index}")) for hdu_index, _pos in entries]
    if len(set(identities)) != len(identities):
        raise ValueError("duplicate HDU identity in exposure confidence normalization")
    try:
        with fitsio.FITS(map_path, "rw") as f:
            sampler = _BoundedPrioritySampler()
            for identity, (_hdu_index, pos) in zip(identities, entries):
                data = f[pos].read()
                sampler.update(identity, data)
            sample = sampler.sample()
            if not sample.size:
                return True
            normalization = float(np.percentile(sample, percentile))
            if not np.isfinite(normalization) or normalization <= 0:
                print(f"  ERROR: global confidence normalization is invalid ({normalization}).")
                return False
            for _hdu_index, pos in entries:
                data = f[pos].read()
                confidence = np.clip(data / normalization, 0.0, 1.0) * upper
                f[pos].write(confidence.astype(writer.dtype or np.float32, copy=False))
    except OSError as e:
        print(f"  ERROR: confidence global normalization failed: {e}")
        return False
    print(f"  Confidence normalized to exposure-global p{percentile:g} {normalization:.3g}.")
    return True


def _veto_dead_ccd_hdus(flat_bad_masks, flat_meds, flat_cfg, flat_shapes=None):
    """Zero-weight HDUs whose flat level is an outlier across the exposure.

    A whole CCD blanked by Elixir (or failed hardware) shows up as a flat
    median tens of sigma from its siblings. Such an HDU's weights would be
    garbage; mark it fully BAD explicitly instead of letting a partial mask
    through. No-op for small-HDU runs and when all medians agree. Mutates
    flat_bad_masks in place.
    """
    if not flat_cfg.get("dead_ccd_enable", True):
        return
    idxs = [i for i in flat_meds if np.isfinite(float(flat_meds[i]))]
    if len(idxs) < int(flat_cfg.get("dead_ccd_min_hdus", 8)):
        return
    meds = np.array([float(flat_meds[i]) for i in idxs])
    if not np.all(np.isfinite(meds)):
        return
    exp_med = float(np.median(meds))
    mad = float(np.median(np.abs(meds - exp_med)) * 1.4826)
    if not np.isfinite(exp_med) or exp_med <= 0:
        return
    sigma = float(flat_cfg.get("dead_ccd_mad_sigma", 5.0))
    floor = float(flat_cfg.get("dead_ccd_min_rel_dev", 0.10)) * exp_med
    for i, m in zip(idxs, meds):
        if abs(m - exp_med) > max(sigma * mad, floor):
            if i in flat_bad_masks:
                flat_bad_masks[i] = np.ones_like(flat_bad_masks[i], dtype=bool)
            elif flat_shapes is not None and i in flat_shapes:
                flat_bad_masks[i] = np.ones(flat_shapes[i], dtype=bool)
            else:
                continue
            print(f"    HDU {i}: flat median {m:.3f} vs exposure {exp_med:.3f} -- flagging whole CCD BAD.")


def _resolve_max_workers(max_workers: int | None, n_hdus: int) -> int:
    """Resolve worker count; 1 means sequential (reproducible/CI)."""
    if max_workers is None:
        eff = min(8, os.cpu_count() or 4)
    else:
        try:
            eff = int(max_workers)
        except (TypeError, ValueError):
            eff = min(8, os.cpu_count() or 4)
        if eff <= 1:
            return 1
    if eff <= 0:
        return 1
    return max(1, min(eff, max(1, n_hdus)))


def _fits_path_of(hdul, explicit: Optional[str]) -> Optional[str]:
    if isinstance(explicit, str) and explicit:
        return explicit
    try:
        p = getattr(hdul, "_filename", None)
        if isinstance(p, bytes):
            p = p.decode()
        return p if isinstance(p, str) and p else None
    except Exception:
        return None


def _hdu_at(hdul, index: int):
    """Return hdul[index] or None if missing/unreadable."""
    if hdul is None:
        return None
    try:
        return hdul[index] if index < len(hdul) else None
    except Exception:
        return None


# Per-CCD identifier keys, most specific first. MegaCam frames carry the
# physical CCD id in CCDNAME (e.g. '8341-7-5'); CCDNAM is the older spelling
# of the same thing. 'CCD' is deliberately absent: on MegaCam it holds the
# detector model ('Marconi/EEV CCD42-90'), identical for all 36 HDUs, so
# naming from it would give every product in the file the same EXTNAME.
_CCD_IDENTIFIER_KEYS = ("CCDNAME", "CCDNAM")


def _hdu_identifier(hdu, header, index: int) -> str:
    """Output-name token for one HDU: its CCD id, else its own name, else the index.

    The input's ``EXTNAME`` is a tiling-compression artifact and fitsio's image
    HDUs expose no ``name``, so every product used to be named from its
    position alone (``MAP_HDU1``). Naming them after the CCD id instead
    (``MAP_8341-7-5``) makes each HDU self-describing.
    """
    for key in _CCD_IDENTIFIER_KEYS:
        value = None
        if header is not None:
            try:
                value = header.get(key)
            except Exception:
                value = None
        if isinstance(value, bytes):
            value = value.decode(errors="replace")
        if value is None or isinstance(value, str) and not value.strip():
            continue
        token = "-".join(str(value).split()).replace("/", "-")
        if token:
            return token
    name = getattr(hdu, "name", None)
    if isinstance(name, bytes):
        name = name.decode(errors="replace")
    if isinstance(name, str) and name.strip():
        return name.strip()
    return f"HDU{index}"


def _open_hdu_handles(
    i: int,
    *,
    in_path: Optional[str],
    fl_path: Optional[str],
    bp_path: Optional[str],
    hdul_input,
    hdul_flat,
    hdul_badpix,
):
    """Yield (sci, flat, badpix, header, name) for one HDU, from path reopen or open handles.

    When ``in_path`` is set, opens per-worker FITS handles (thread-safe). Otherwise uses
    the already-open ``hdul_*`` objects. Caller must invoke the returned ``close()``.
    """
    opened: list = []

    def close():
        for handle in opened:
            try:
                handle.close()
            except Exception:
                pass

    try:
        if in_path is not None:
            fi = fitsio.FITS(in_path, "r")
            opened.append(fi)
            if i >= len(fi):
                close()
                return None, None, None, None, f"HDU{i}", close, "index out of range"
            hdu_sci = fi[i]
            if hdul_flat is not None and fl_path is not None:
                ff = fitsio.FITS(fl_path, "r")
                opened.append(ff)
                if i >= len(ff):
                    # A flat MEF shorter than the science MEF must not degrade
                    # to "no flat": process_image would substitute a unit flat
                    # and emit unflat-fielded weights that look successful.
                    # Build the reason before close(), which closes ff.
                    reason = f"flat {fl_path} has {len(ff)} HDU(s), need index {i}"
                    close()
                    return None, None, None, None, f"HDU{i}", close, reason
                hdu_flat_obj = ff[i]
            else:
                hdu_flat_obj = _hdu_at(hdul_flat, i)
            if hdul_badpix is not None and bp_path is not None:
                fb = fitsio.FITS(bp_path, "r")
                opened.append(fb)
                if i >= len(fb):
                    reason = f"keep-map {bp_path} has {len(fb)} HDU(s), need index {i}"
                    close()
                    return None, None, None, None, f"HDU{i}", close, reason
                hdu_badpix_obj = fb[i]
            else:
                hdu_badpix_obj = _hdu_at(hdul_badpix, i)
            try:
                hdu_header_raw = fi[i].read_header()
            except Exception:
                hdu_header_raw = None
            hdu_name = _hdu_identifier(hdu_sci, hdu_header_raw, i)
            return hdu_sci, hdu_flat_obj, hdu_badpix_obj, hdu_header_raw, hdu_name, close, None

        hdu_sci = hdul_input[i]
        hdu_flat_obj = _hdu_at(hdul_flat, i)
        hdu_badpix_obj = _hdu_at(hdul_badpix, i)
        try:
            hdu_header_raw = hdul_input[i].read_header()
        except Exception:
            hdu_header_raw = None
        hdu_name = _hdu_identifier(hdu_sci, hdu_header_raw, i)
        return hdu_sci, hdu_flat_obj, hdu_badpix_obj, hdu_header_raw, hdu_name, close, None
    except Exception as e:
        close()
        return None, None, None, None, f"HDU{i}", (lambda: None), str(e)


def _common_wcs_geometry(header):
    if header is None:
        return None
    try:
        has_cd = all(header.get(name) is not None for name in ("CD1_1", "CD1_2", "CD2_1", "CD2_2"))
        has_cdelt = all(header.get(name) is not None for name in ("CDELT1", "CDELT2"))
        if not has_cd and not has_cdelt:
            return None
        # WCS() warns and substitutes a default for a card it cannot parse
        # (e.g. CDELT1 = "not a number"), which would hand back a plausible but
        # wrong plate scale. Refuse such a header instead of failing open.
        for name in (
            "CRVAL1",
            "CRVAL2",
            "CRPIX1",
            "CRPIX2",
            "CDELT1",
            "CDELT2",
            "CD1_1",
            "CD1_2",
            "CD2_1",
            "CD2_2",
            "PC1_1",
            "PC1_2",
            "PC2_1",
            "PC2_2",
        ):
            value = header.get(name)
            if value is None:
                continue
            try:
                float(value)
            except (TypeError, ValueError):
                return None
        cards = {}
        for name in (
            "CTYPE1",
            "CTYPE2",
            "CUNIT1",
            "CUNIT2",
            "CRVAL1",
            "CRVAL2",
            "CRPIX1",
            "CRPIX2",
            "CDELT1",
            "CDELT2",
            "CD1_1",
            "CD1_2",
            "CD2_1",
            "CD2_2",
            "PC1_1",
            "PC1_2",
            "PC2_1",
            "PC2_2",
        ):
            value = header.get(name)
            if value is not None:
                cards[name] = value
        wcs = WCS(cards, naxis=2)
        if not wcs.has_celestial:
            return None
        celestial = wcs.celestial
        ctypes = [str(value).upper() for value in celestial.wcs.ctype]
        if len(ctypes) != 2 or any(not value.endswith("-TAN") for value in ctypes):
            return None
        celestial.wcs.set()
        crval = np.asarray(celestial.wcs.crval, dtype=np.float64)
        crpix = np.asarray(celestial.wcs.crpix, dtype=np.float64) - 1.0
        cd = np.asarray(celestial.pixel_scale_matrix, dtype=np.float64)
        probe = np.asarray(celestial.all_pix2world([[0.0, 0.0]], 0), dtype=np.float64)
    except Exception:
        return None
    if crval.shape != (2,) or crpix.shape != (2,) or cd.shape != (2, 2):
        return None
    if not np.all(np.isfinite(crval)) or not np.all(np.isfinite(crpix)) or not np.all(np.isfinite(cd)):
        return None
    if probe.shape != (1, 2) or not np.all(np.isfinite(probe)):
        return None
    determinant = float(np.linalg.det(cd))
    if not np.isfinite(determinant) or abs(determinant) <= np.finfo(np.float64).tiny:
        return None
    return {
        "crval": crval,
        "crpix": crpix,
        "cd": cd,
        "scale_deg_per_px": float(np.sqrt(abs(determinant))),
    }


def _line_to_common_geometry(local, wcs):
    if local is None or wcs is None:
        return None
    point = np.asarray(local["point"], dtype=np.float64)
    direction = wcs["cd"] @ np.asarray(local["direction"], dtype=np.float64)
    length = float(np.hypot(*direction))
    if not np.isfinite(length) or length <= 0.0:
        return None
    direction /= length
    normal = np.array([-direction[1], direction[0]], dtype=np.float64)
    mosaic_point = wcs["cd"] @ (point - wcs["crpix"])
    offset = float(np.dot(normal, mosaic_point))
    if normal[0] < 0.0 or (normal[0] == 0.0 and normal[1] < 0.0):
        normal = -normal
        offset = -offset
    return {
        "angle_deg": float(np.degrees(np.arctan2(normal[1], normal[0])) % 180.0),
        "normal": normal.tolist(),
        "offset_deg": offset,
        "scale_deg_per_px": wcs["scale_deg_per_px"],
        "crval": wcs["crval"],
    }


def _streak_catalog(hdu_index, mask_data, header=None):
    """Connected STREAK components and their detector/common-frame geometry."""
    if mask_data is None:
        return []
    data = np.asarray(mask_data)
    streak = (data & np.uint32(QualityBit.STREAK)) != 0
    labels, count = ndimage.label(streak, structure=np.ones((3, 3), dtype=np.uint8))
    if count == 0:
        return []
    shape = tuple(data.shape)
    wcs = _common_wcs_geometry(header)
    catalogs = []
    for component, region in enumerate(ndimage.find_objects(labels), 1):
        if region is None:
            continue
        ys, xs = np.nonzero(labels[region] == component)
        if ys.size < 24:
            continue
        ys += region[0].start
        xs += region[1].start
        local = line_geometry(ys, xs, shape)
        catalogs.append(
            {
                "hdu": int(hdu_index),
                "component": int(component),
                "ys": np.asarray(ys, dtype=np.int32),
                "xs": np.asarray(xs, dtype=np.int32),
                "shape": shape,
                "local": local,
                "common": _line_to_common_geometry(local, wcs),
            }
        )
    return catalogs


def _rewrite_hdu(fits_obj, pos, data):
    # Existing HDUs retain their compression; ImageHDU.write has no compress option.
    fits_obj[pos].write(np.ascontiguousarray(data))


def _ensure_hdu_extname(fits_obj, pos, extname):
    current = fits_obj[pos].read_header().get("EXTNAME")
    if str(current or "").strip() != extname:
        fits_obj[pos].write_key("EXTNAME", extname)


def _clear_chip_replicas(catalogs, writers, config):
    """Drop STREAK where the same CCD-local line was written on another chip.

    Coordinates were recorded at flush time, so this does not hold the MEF's
    arrays. Weight is restored from inverse variance only where STREAK was the
    only zero-weight bit and the map product is a weight.
    """
    if len(catalogs) < 2:
        return True
    cleared = _replica_component_indices(catalogs)
    if not cleared:
        return True
    mask_writer = (writers or {}).get("mask")
    if mask_writer is None or not getattr(mask_writer, "out_path", None):
        return True
    output_format = str((config or {}).get("output_params", {}).get("output_map_format", "weight")).lower()
    conf_cfg = (config or {}).get("confidence_params", {})
    per_hdu_confidence = output_format == "confidence" and conf_cfg.get("normalize_scope") != "per_exposure"
    streak_bit = np.uint32(QualityBit.STREAK)
    exclusion = DEFAULT_ZERO_WEIGHT_BITS & ~QualityBit.STREAK
    exclude_detected = bool((config or {}).get("output_params", {}).get("mask_detected_in_weight", False))
    if exclude_detected:
        exclusion |= QualityBit.DETECTED
    other_bits = np.uint32(exclusion)
    map_writer = writers.get("map")
    ivar_writer = writers.get("invvar")
    raw_writer = writers.get("weight_raw")
    ind_writer = writers.get("ind_streak")

    def _open(writer):
        if writer is None:
            return None
        return fitsio.FITS(writer.out_path, "rw")

    handles = []
    try:
        fmask = fitsio.FITS(mask_writer.out_path, "rw")
        handles.append(fmask)
        fmap = _open(map_writer)
        fivar = _open(ivar_writer)
        fraw = _open(raw_writer)
        find = _open(ind_writer)
        handles.extend(handle for handle in (fmap, fivar, fraw, find) if handle is not None)
        individual_positions = {}
        if ind_writer is not None:
            for index in sorted(cleared):
                cat = catalogs[index]
                hdu = cat["hdu"]
                if hdu in individual_positions:
                    continue
                spos = ind_writer.positions.get(hdu)
                if spos is None or find is None or spos >= len(find):
                    print(f"  ERROR: chip-replica veto: missing individual STREAK position for HDU {hdu}.")
                    return False
                if np.shape(find[spos].read()) != cat["shape"]:
                    print(f"  ERROR: chip-replica veto: individual STREAK shape mismatch for HDU {hdu}.")
                    return False
                individual_positions[hdu] = spos
        n_cleared = 0
        for index in sorted(cleared):
            cat = catalogs[index]
            pos = mask_writer.positions.get(cat["hdu"])
            if pos is None or pos >= len(fmask):
                print(f"  WARNING: chip-replica veto: unknown mask position for HDU {cat['hdu']}.")
                continue
            data = fmask[pos].read()
            ys, xs = cat["ys"], cat["xs"]
            if ys.size == 0 or data.shape != cat["shape"]:
                print(f"  WARNING: chip-replica veto: shape mismatch for HDU {cat['hdu']}.")
                continue
            pixels = data[ys, xs]
            only = ((pixels & streak_bit) != 0) & ((pixels & other_bits) == 0)
            needs_restore = bool(np.any(only)) and (fmap is not None or fraw is not None)
            ipos = ivar_writer.positions.get(cat["hdu"]) if ivar_writer is not None else None
            ivar = fivar[ipos].read() if fivar is not None and ipos is not None else None
            if needs_restore and np.shape(ivar) != data.shape:
                print(
                    f"  WARNING: chip-replica veto: weight map could not be restored on HDU {cat['hdu']}; retaining STREAK."
                )
                continue
            data[ys, xs] = pixels & ~np.array(streak_bit, dtype=data.dtype)
            for handle, writer in ((fmap, map_writer), (fraw, raw_writer)):
                if handle is None or not needs_restore:
                    continue
                wpos = writer.positions.get(cat["hdu"])
                if wpos is None:
                    raise OSError(f"missing output position for HDU {cat['hdu']}")
                if writer is map_writer and per_hdu_confidence:
                    product = build_weight_product(
                        ivar,
                        data.astype(np.uint32, copy=False),
                        exclude_detected=exclude_detected,
                        confidence_percentile=conf_cfg.get("normalize_percentile", 99.0),
                    )
                    output = product.confidence * (100.0 if conf_cfg.get("scale_to_100", False) else 1.0)
                else:
                    output = handle[wpos].read()
                    output[ys[only], xs[only]] = ivar[ys[only], xs[only]]
                _rewrite_hdu(handle, wpos, output.astype(writer.dtype or np.float32, copy=False))
            _rewrite_hdu(fmask, pos, data)
            if find is not None:
                spos = individual_positions[cat["hdu"]]
                ind = find[spos].read()
                ind[ys, xs] = 0
                _rewrite_hdu(find, spos, ind.astype(np.uint8, copy=False))
            n_cleared += 1
        if n_cleared:
            print(f"  Chip-replica veto cleared {n_cleared} STREAK components.")
        return True
    except OSError as exc:
        print(f"  ERROR: chip-replica veto failed: {exc}")
        return False
    finally:
        for handle in handles:
            try:
                handle.close()
            except Exception:
                pass


_PRODUCT_PATH_KEYS = (
    "out_map_path",
    "out_mask_path",
    "out_invvar_path",
    "out_sky_path",
    "out_weight_raw_path",
)
_INDIVIDUAL_MASK_KEYS = ("bad", "sat", "cr", "obj", "streak", "nodata")


def _ordered_product_paths(paths):
    products = [(key, paths.get(key)) for key in _PRODUCT_PATH_KEYS]
    individual = paths.get("individual_mask_paths") or {}
    products.extend((f"individual_mask_paths.{key}", individual.get(key)) for key in _INDIVIDUAL_MASK_KEYS)
    return [(key, os.fspath(path)) for key, path in products if path]


class _ProductPublication:
    def __init__(self, paths):
        self.generation_id = uuid.uuid4().hex
        self._entries = []
        self._committed = False
        self._rollback_incomplete = False
        temporary_paths = dict(paths)
        temporary_paths["individual_mask_paths"] = dict(paths.get("individual_mask_paths") or {})
        temporary_paths["_generation_id"] = self.generation_id
        for key, final_path in _ordered_product_paths(paths):
            directory = os.path.dirname(os.path.abspath(final_path))
            basename = os.path.basename(final_path)
            temporary = os.path.join(directory, f".{basename}.wm-tmp-{self.generation_id}")
            backup = os.path.join(directory, f".{basename}.wm-bak-{self.generation_id}")
            self._entries.append((final_path, temporary, backup))
            if key.startswith("individual_mask_paths."):
                temporary_paths["individual_mask_paths"][key.rsplit(".", 1)[1]] = temporary
            else:
                temporary_paths[key] = temporary
        self.paths = temporary_paths

    @property
    def temporary_path(self):
        if len(self._entries) != 1:
            raise ValueError("temporary_path is only defined for a single-product publication")
        return self._entries[0][1]

    def validate(self, expected_hdus):
        expected_count = len(expected_hdus)
        reference_hdu_count = None
        reference_shapes = None
        reference_names = None
        for _final, temporary, _backup in self._entries:
            with fitsio.FITS(temporary, "r") as handle:
                if reference_hdu_count is None:
                    reference_hdu_count = len(handle)
                elif len(handle) != reference_hdu_count:
                    raise ValueError("output products have different HDU counts")
                images = [
                    item for item in handle if item.get_info().get("hdutype") == 0 and item.get_info().get("ndims") == 2
                ]
                if len(images) != expected_count:
                    raise ValueError(f"output contains {len(images)} image HDUs; expected {expected_count}")
                shapes = []
                names = []
                for item in images:
                    header = item.read_header()
                    generation = str(header.get("WMGENID", ""))
                    if generation != self.generation_id:
                        raise ValueError("output product generation ID is missing or inconsistent")
                    if str(header.get("WMVERS", "")) != CONTRACT_VERSION or not header.get("WMART"):
                        raise ValueError("output product contract metadata is missing or inconsistent")
                    shape = tuple(item.get_dims())
                    name = str(header.get("EXTNAME", "")).strip()
                    if not name:
                        raise ValueError("output product EXTNAME is missing")
                    shapes.append(shape)
                    names.append(name.split("_", 1)[-1])
                if len(set(names)) != len(names):
                    raise ValueError("output product EXTNAME values are not unique")
                if reference_shapes is None:
                    reference_shapes = shapes
                    reference_names = names
                elif shapes != reference_shapes or names != reference_names:
                    raise ValueError("output product HDU shapes or EXTNAME values are not synchronized")

    def promote(self):
        moved = []
        try:
            for final, temporary, backup in self._entries:
                had_original = os.path.exists(final)
                if had_original:
                    os.replace(final, backup)
                moved.append((final, temporary, backup, had_original))
                os.replace(temporary, final)
        except OSError as promotion_error:
            rollback_errors = []
            for final, temporary, backup, had_original in reversed(moved):
                try:
                    if had_original:
                        if not os.path.exists(backup):
                            raise OSError(f"rollback backup is missing: {backup}")
                        os.replace(backup, final)
                    elif os.path.exists(final):
                        try:
                            os.replace(final, temporary)
                        except OSError as quarantine_error:
                            try:
                                os.unlink(final)
                            except OSError as unlink_error:
                                raise OSError(
                                    f"could not quarantine ({quarantine_error}) or remove new output: {unlink_error}"
                                ) from quarantine_error
                except OSError as exc:
                    rollback_errors.append(exc)
            if rollback_errors:
                self._rollback_incomplete = True
                raise OSError(
                    f"publication failed ({promotion_error}) and rollback was incomplete: {rollback_errors[0]}"
                ) from promotion_error
            raise
        self._committed = True

    def cleanup(self):
        pending = []
        for _final, temporary, backup in self._entries:
            paths = (temporary,) if self._rollback_incomplete else (temporary, backup)
            for path in paths:
                try:
                    os.unlink(path)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    pending.append((path, backup, exc))
        for path, backup, first_error in pending:
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                kind = "post-commit backup" if self._committed and path == backup else "publication artifact"
                print(f"  WARNING: Could not remove {kind} '{path}': {exc}; first attempt failed: {first_error}")


def _process_all_hdus_to_paths(
    hdus_to_process: list,
    hdul_input,
    hdul_flat,
    config: dict,
    paths: dict,
    args: argparse.Namespace,
    flat_path: Optional[str] = None,
    hdul_badpix=None,
    hdul_dark=None,
    max_workers: int | None = None,
    input_path: Optional[str] = None,
    badpix_path: Optional[str] = None,
    dark_path: Optional[str] = None,
    detector_priors: Optional[dict] = None,
) -> int:
    """Process every requested HDU, streaming each HDU's outputs to disk.
    Each HDU's maps are written to their output files as soon as they are
    produced instead of being accumulated in memory. A 36-CCD MegaPrime MEF
    therefore only ever holds a single CCD's products at a time (~a few
    hundred MB) rather than the whole MEF's (~7 GB), which previously blew
    through the container memory limit when several exposures were processed
    concurrently and OOM-killed the batch.
    Parallel by default: per-HDU ThreadPoolExecutor; single-HDU inputs run
    inline (per-file fast path). Compute runs concurrently, but only one
    worker-window of HDUs sits ahead of the writer, and writes are applied in
    HDU-index order for deterministic output.
    """
    if max_workers is None:
        max_workers = getattr(args, "max_workers", None)
    eff_workers = _resolve_max_workers(max_workers, len(hdus_to_process))
    tile_size = getattr(args, "tile_size", 1024) if hasattr(args, "tile_size") else 1024
    try:
        tile_size = int(tile_size)
    except (TypeError, ValueError):
        tile_size = 1024

    def _usable(p: Optional[str]) -> Optional[str]:
        return p if isinstance(p, str) and p and os.path.exists(p) else None

    in_path_u = _usable(_fits_path_of(hdul_input, input_path))
    fl_path_u = _usable(_fits_path_of(hdul_flat, flat_path)) if hdul_flat is not None else None
    bp_path_u = _usable(_fits_path_of(hdul_badpix, badpix_path)) if hdul_badpix is not None else None
    dk_path_u = _usable(_fits_path_of(hdul_dark, dark_path)) if hdul_dark is not None else None
    process_success_count = 0
    writers = _make_output_writers(paths, hdul_input, config)
    flat_bad_masks = {}
    if hdul_flat is not None and flat_path:
        print(f"  Pre-computing flat bad masks for {len(hdus_to_process)} HDUs...")
        flat_cfg = config.get("flat_masking", {})
        flat_meds: dict = {}
        flat_shapes: dict = {}
        for i in hdus_to_process:
            if i < len(hdul_flat):
                try:
                    flat_data = np.ascontiguousarray(hdul_flat[i].read().astype(np.float32, copy=False))
                    flat_meds[i] = _get_global_median(flat_data)
                    flat_shapes[i] = flat_data.shape
                except Exception as e:
                    print(f"    WARNING: Failed to read flat for HDU {i}: {e}")
        # Tactic D: medians stay sequential (cheap reads for the dead-CCD
        # veto); the 15x15 mask medians move into the ThreadPoolExecutor
        # workers via the precomputed_bad_mask miss path in _compute_one.
        _veto_dead_ccd_hdus(flat_bad_masks, flat_meds, flat_cfg, flat_shapes=flat_shapes)
        print(f"  Flat medians ready for {len(flat_meds)} HDUs ({len(flat_bad_masks)} vetoed dead)")
    conf_scope = config.get("confidence_params", {}).get("normalize_scope", "per_hdu")
    dark_cfg_global = config.get("dark_masking", {})

    def _merge_dark(i, hdu_sci, pre):
        dark_hdu = None
        fd_local = None
        close_fd = False
        try:
            if dk_path_u is not None:
                try:
                    fd_local = fitsio.FITS(dk_path_u, "r")
                    close_fd = True
                    if i < len(fd_local):
                        dark_hdu = fd_local[i]
                    else:
                        print(
                            f"    WARNING: dark frame {dk_path_u} has {len(fd_local)} HDU(s), "
                            f"none for index {i}. No dark hot-pixel mask for this CCD."
                        )
                        dark_hdu = None
                except Exception as e:
                    print(f"    WARNING: dark frame HDU {i} unavailable: {e}. Skipping dark mask.")
                    dark_hdu = None
            elif hdul_dark is not None:
                try:
                    if i < len(hdul_dark):
                        dark_hdu = hdul_dark[i]
                    else:
                        print(
                            f"    WARNING: dark frame has {len(hdul_dark)} HDU(s), "
                            f"none for index {i}. No dark hot-pixel mask for this CCD."
                        )
                        dark_hdu = None
                except Exception as e:
                    print(f"    WARNING: dark frame HDU {i} unavailable: {e}. Skipping dark mask.")
                    dark_hdu = None
            if dark_hdu is None:
                return pre
            try:
                dark_data = np.ascontiguousarray(dark_hdu.read().astype(np.float32, copy=False))
                dark_hot = detect_dark_hot_pixels(dark_data, dark_cfg_global)
                sci_shape = tuple(hdu_sci.get_dims())
                if dark_hot.shape != sci_shape:
                    print(f"    Skipping dark mask for HDU {i}: shape mismatch.")
                else:
                    if pre is not None and pre.shape != sci_shape:
                        pre = None
                    pre = dark_hot if pre is None else (pre | dark_hot)
                    print(f"    Dark frame adds {int(np.count_nonzero(dark_hot))} hot pixels to HDU {i}.")
            except Exception as e:
                print(f"    WARNING: dark mask failed for HDU {i}: {e}")
            return pre
        finally:
            if close_fd and fd_local is not None:
                try:
                    fd_local.close()
                except Exception:
                    pass

    def _compute_one(i):
        pre = flat_bad_masks.get(i) if flat_bad_masks else None
        close = None

        try:
            hdu_sci, hdu_flat_obj, hdu_badpix_obj, hdu_header_raw, hdu_name, close, err = _open_hdu_handles(
                i,
                in_path=in_path_u,
                fl_path=fl_path_u,
                bp_path=bp_path_u,
                hdul_input=hdul_input,
                hdul_flat=hdul_flat,
                hdul_badpix=hdul_badpix,
            )
            if err is not None or hdu_sci is None:
                print(f"Skipping HDU {i}: {err or 'cannot access input HDU'}.")
                return (i, (None, None, None, None, None, None), None, hdu_name)
            pre2 = _merge_dark(i, hdu_sci, pre)
            prior = detector_priors.get(i) if isinstance(detector_priors, dict) else None
            result = process_hdu(
                hdu_sci,
                hdu_flat_obj,
                config,
                i,
                tile_size=tile_size,
                flat_path=flat_path,
                hdu_badpix=hdu_badpix_obj,
                precomputed_bad_mask=pre2,
                detector_prior=prior,
            )
            return (i, result, hdu_header_raw, hdu_name)
        except StageFailure as exc:
            if exc.hdu_index is None:
                exc.hdu_index = i
            raise
        except Exception as e:
            import traceback

            print(f"FATAL ERROR processing HDU {i}: {e}\n{traceback.format_exc()}")
            return (i, (None, None, None, None, None, None), None, f"HDU{i}")
        finally:
            if close is not None:
                close()

    streak_catalogs: list = []

    def _emit(i, result, hdu_header_raw, hdu_name_raw):
        """Store and flush one HDU's products (releases its arrays immediately)."""
        nonlocal process_success_count
        try:
            if result is None or result[0] is None:
                print(f"Skipping HDU {i} due to processing errors.")
                return
            (mask_data, inv_var_data, weight_map, confidence_map, sky_map, header_info) = result
            bad_mask, sat_mask, cr_mask, obj_mask, streak_mask, nodata_mask = extract_individual_masks(
                header_info, mask_data
            )
            streak_catalogs.extend(_streak_catalog(i, mask_data, hdu_header_raw))
            hdu_name = hdu_name_raw
            hdu_header = hdu_header_raw
            if hdu_header is None:
                hdu_header = fitsio.FITSHDR()
            hdu_header = _strip_compression_keywords(hdu_header)
            sky_cards = (header_info or {}).get("sky_cards") or {}
            sky_header = {**(hdu_header or {}), **sky_cards} if sky_cards else hdu_header
            hdu_output: dict = {}
            _store_output_maps(
                hdu_output,
                i,
                hdu_header,
                hdu_name,
                config,
                confidence_map,
                weight_map,
                mask_data,
                inv_var_data,
                sky_map,
                args,
                bad_mask,
                sat_mask,
                cr_mask,
                obj_mask,
                streak_mask,
                nodata_mask,
                paths,
                sky_header=sky_header,
            )
            _flush_hdu_output(writers, hdu_output, i)
            process_success_count += 1
        except Exception as e:
            import traceback

            print(f"FATAL ERROR processing HDU {i}: {e}\n{traceback.format_exc()}")

    # Each HDU leaves memory as soon as it is produced, and at most
    # ``workers_n`` HDUs are ever computed ahead of the writer, so peak memory
    # stays at a few CCDs instead of the whole MEF's (~8 GB for 36 MegaPrime
    # CCDs). Writes still happen in HDU order, so output files are identical to
    # the previously fully-buffered path.
    workers_n = max(1, min(eff_workers, len(hdus_to_process)))
    if workers_n <= 1:
        for i in hdus_to_process:
            _emit(i, *_compute_one(i)[1:])
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers_n) as ex:
            in_flight: dict = {}
            queue = iter(hdus_to_process)

            def _submit_next() -> bool:
                i = next(queue, None)
                if i is None:
                    return False
                in_flight[i] = ex.submit(_compute_one, i)
                return True

            for _ in range(workers_n):
                if not _submit_next():
                    break
            for i in hdus_to_process:
                fut = in_flight.pop(i, None)
                if fut is None:  # pragma: no cover - the window always holds the next index
                    continue
                try:
                    _i, _res, _hdr, _nm = fut.result()
                except StageFailure:
                    for pending in in_flight.values():
                        pending.cancel()
                    raise
                except Exception as e:
                    import traceback

                    print(f"FATAL ERROR processing HDU {i}: {e}\n{traceback.format_exc()}")
                    _i, _res, _hdr, _nm = i, (None, None, None, None, None, None), None, f"HDU{i}"
                _emit(_i, _res, _hdr, _nm)
                _submit_next()
    if not _clear_chip_replicas(streak_catalogs, writers, config):
        return 0
    if conf_scope == "per_exposure" and not _rescale_confidence_to_global(paths, writers, config):
        return 0
    return process_success_count


def process_all_hdus(
    hdus_to_process: list,
    hdul_input,
    hdul_flat,
    config: dict,
    paths: dict,
    args: argparse.Namespace,
    flat_path: Optional[str] = None,
    hdul_badpix=None,
    hdul_dark=None,
    max_workers: int | None = None,
    input_path: Optional[str] = None,
    badpix_path: Optional[str] = None,
    dark_path: Optional[str] = None,
    detector_priors: Optional[dict] = None,
) -> int:
    publication = _ProductPublication(paths)
    try:
        count = _process_all_hdus_to_paths(
            hdus_to_process,
            hdul_input,
            hdul_flat,
            config,
            publication.paths,
            args,
            flat_path=flat_path,
            hdul_badpix=hdul_badpix,
            hdul_dark=hdul_dark,
            max_workers=max_workers,
            input_path=input_path,
            badpix_path=badpix_path,
            dark_path=dark_path,
            detector_priors=detector_priors,
        )
        if count != len(hdus_to_process):
            return count
        publication.validate(hdus_to_process)
        publication.promote()
        return count
    except StageFailure:
        raise
    except (OSError, ValueError) as exc:
        print(f"  ERROR: product publication failed: {exc}")
        return 0
    finally:
        publication.cleanup()


class _StreamingMapWriter:
    """Stream one map product (map/mask/invvar/sky/...) into a MEF file.

    HDUs are appended as they are produced, so a large multi-extension input
    never has to be held in memory at once. The on-disk layout matches the
    previous buffered writer exactly:

    * a single-extension input with the image at HDU 0 -> a one-HDU file;
    * a MEF whose primary is itself an image -> that image becomes the primary
      and the remaining images are appended as extensions;
    * a MEF whose primary is not an image -> an empty primary header is written
      first, then every image is appended as an extension.
    """

    def __init__(
        self,
        out_path: str,
        hdul_input,
        primary_header,
        compress: bool = False,
        dtype=None,
        generation_id: str | None = None,
    ):
        self.out_path = out_path
        self.hdul_input = hdul_input
        self.primary_header = dict(primary_header or {})
        self._opened = False
        self.positions: dict = {}
        self.identities: dict = {}
        self._lock = threading.Lock()
        self.compress = bool(compress)
        self.dtype = np.dtype(dtype) if dtype is not None else None
        self.generation_id = generation_id
        if generation_id:
            self.primary_header["WMGENID"] = generation_id

    def _prep(self, data):
        if data is None or self.dtype is None:
            return data
        try:
            return np.ascontiguousarray(data, dtype=self.dtype)
        except Exception:
            return np.asarray(data, dtype=self.dtype)

    def write(self, hdu_index: int, data, header, extname: str) -> None:
        data = self._prep(data)
        header = dict(header or {})
        header["EXTNAME"] = extname
        if self.generation_id:
            header["WMGENID"] = self.generation_id
        # Only pass ``compress`` when compressing: fitsio warns about (and
        # ignores) a placeholder value such as "NOT_SET".
        kwargs = {"compress": "RICE_1"} if self.compress else {}
        with self._lock:
            self.identities[hdu_index] = extname
            if not self._opened:
                if hdu_index == 0:
                    primary_data = data
                    primary_header = header
                else:
                    primary_data = None
                    primary_header = self.primary_header
                fitsio.write(self.out_path, primary_data, header=primary_header, clobber=True, **kwargs)
                self._opened = True
                if hdu_index == 0:
                    with fitsio.FITS(self.out_path, "rw") as f_out:
                        # Compressed primary images live at HDU 1, not HDU 0.
                        position = len(f_out) - 1
                        _ensure_hdu_extname(f_out, position, extname)
                        self.positions[hdu_index] = position
                    return
            with fitsio.FITS(self.out_path, "rw") as f_out:
                f_out.write(data, header=header, extname=extname, **kwargs)
                position = len(f_out) - 1
                _ensure_hdu_extname(f_out, position, extname)
                self.positions[hdu_index] = position


def _wire_dtype(bitpix, *, is_mask: bool):
    """Map configured bitpix to a numpy wire dtype (mask uint, float otherwise)."""
    try:
        b = int(bitpix)
    except (TypeError, ValueError):
        return np.uint16 if is_mask else np.float32
    if is_mask:
        return {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}.get(b, np.uint16)
    return {32: np.float32, 64: np.float64, -32: np.float32, -64: np.float64}.get(b, np.float32)


def _make_output_writers(paths: dict, hdul_input, config: dict | None = None) -> dict:
    """Build a streaming writer per requested output product."""
    primary_header = None
    if len(hdul_input) > 0:
        try:
            primary_header = _strip_compression_keywords(hdul_input[0].read_header())
        except Exception:
            primary_header = None
    op = (config or {}).get("output_params", {}) if isinstance((config or {}).get("output_params", {}), dict) else {}
    try:
        mask_bp = int(op.get("mask_bitpix", 16))
    except (TypeError, ValueError):
        mask_bp = 16
    try:
        ivar_bp = int(op.get("ivar_bitpix", 32))
    except (TypeError, ValueError):
        ivar_bp = 32
    compress = bool(op.get("compress", False))
    generation_id = paths.get("_generation_id")
    mask_dt = _wire_dtype(mask_bp, is_mask=True)
    float_dt = _wire_dtype(ivar_bp, is_mask=False)
    writers: dict = {}
    for key, path_key in (
        ("map", "out_map_path"),
        ("mask", "out_mask_path"),
        ("invvar", "out_invvar_path"),
        ("sky", "out_sky_path"),
        ("weight_raw", "out_weight_raw_path"),
    ):
        out_path = paths.get(path_key)
        if out_path:
            dt = mask_dt if key == "mask" else float_dt
            writers[key] = _StreamingMapWriter(
                out_path,
                hdul_input,
                primary_header,
                compress=compress,
                dtype=dt,
                generation_id=generation_id,
            )
    for mask_type in ("bad", "sat", "cr", "obj", "streak", "nodata"):
        out_path = (paths.get("individual_mask_paths") or {}).get(mask_type)
        if out_path:
            writers[f"ind_{mask_type}"] = _StreamingMapWriter(
                out_path,
                hdul_input,
                primary_header,
                compress=compress,
                dtype=np.uint8,
                generation_id=generation_id,
            )
    return writers


def _flush_hdu_output(writers: dict, hdu_output: dict, hdu_index: int) -> None:
    """Write a single HDU's buffered maps to their output files and release them."""
    entry_map = hdu_output.get(hdu_index) or {}
    for key in ("map", "mask", "invvar", "sky", "weight_raw"):
        entry = entry_map.get(key)
        if entry is not None and key in writers:
            writers[key].write(hdu_index, entry["data"], entry["header"], entry["name"])

    individual = entry_map.get("individual_masks") or {}
    for mask_type in ("bad", "sat", "cr", "obj", "streak", "nodata"):
        entry = individual.get(mask_type)
        writer_key = f"ind_{mask_type}"
        if entry is not None and writer_key in writers:
            writers[writer_key].write(hdu_index, entry["data"], entry["header"], entry["name"])
