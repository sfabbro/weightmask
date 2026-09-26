"""MEF orchestration: per-HDU processing, streaming writers, parallel fan-out."""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import threading
from typing import Optional, Tuple

import fitsio
import numpy as np

from . import __version__
from .bad import _get_global_median, compute_flat_bad_mask_cached, detect_bad_pixels
from .contract import (
    CONFIDENCE_SEMANTICS,
    DEFAULT_ZERO_WEIGHT_BITS,
    INVERSE_VARIANCE_SEMANTICS,
    MASK_POLARITY,
    ArtifactMetadata,
    ProducerMetadata,
    QualityBit,
)
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
        sci_data_full = np.ascontiguousarray(hdu_sci.read().astype(np.float32))
        sci_hdr = hdu_sci.read_header()
    except (OSError, fitsio.FITSFormatError) as e:
        print(f"Skipping HDU: Cannot read science data: {e}")
        return None, None, None, None, None, None

    flat_data_full = None
    if hdu_flat is not None:
        try:
            flat_data_full = np.ascontiguousarray(hdu_flat.read().astype(np.float32))
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
            ext = hdu_badpix.read()
            if ext.shape == sci_data_full.shape:
                # External mask uses the Elixir keep-map convention: 0 = bad, 1 = good.
                badpix_mask = ext == 0
                frac = float(np.mean(badpix_mask))
                thresh = float(config.get("flat_masking", {}).get("dead_ccd_badpix_fraction", 0.9))
                if frac > thresh:
                    print(f"  NOTE: external mask flags {frac:.1%} of HDU {hdu_index} bad (dead CCD?) -- zero weight.")
            else:
                print(f"  Skipping badpix mask: shape mismatch {ext.shape} != {sci_data_full.shape}")
        except OSError as e:
            print(f"Skipping badpix mask: cannot read: {e}")

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
    equal. A sky line that crosses two chips lands at different CCD-local
    offsets, so this is the same distinction as the curator's chip-replica test.
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
    return {"offset_px": offset, "angle_deg": angle}


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


def extract_individual_masks(header_info: dict, mask_data):
    keys = ("bad", "sat", "cr", "obj", "streak", "nodata")
    source = header_info.get("individual_masks", {}) if header_info else {}
    if mask_data is None:
        return tuple(np.array([]) for _ in keys)
    shape = mask_data.shape
    return tuple(source.get(k, np.zeros(shape, dtype=bool)) for k in keys)


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
            "header": hdu_header,
            "name": f"BAD_{hdu_name}",
        },
        "sat": {
            "data": sat_mask.astype(np.uint8),
            "header": hdu_header,
            "name": f"SAT_{hdu_name}",
        },
        "cr": {
            "data": cr_mask.astype(np.uint8),
            "header": hdu_header,
            "name": f"CR_{hdu_name}",
        },
        "obj": {
            "data": obj_mask.astype(np.uint8),
            "header": hdu_header,
            "name": f"OBJ_{hdu_name}",
        },
        "streak": {
            "data": streak_mask.astype(np.uint8),
            "header": hdu_header,
            "name": f"STREAK_{hdu_name}",
        },
        "nodata": {
            "data": nodata_mask.astype(np.uint8),
            "header": hdu_header,
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
        semantics = CONFIDENCE_SEMANTICS if output_format == "confidence" else "masked_inverse_variance"
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
        _assign_map_if_valid(
            output_data, i, "sky", sky_map, sky_header if sky_header is not None else hdu_header, hdu_name
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


def _rescale_confidence_to_global(paths, writers, conf_samples, conf_p99, config):
    """Rewrite an exposure-global confidence normalization into the map file.

    Per-HDU confidence was normalized by each HDU's own p99 at stream time;
    rescale stored values by p99_hdu/global_p99 so confidence is comparable
    across CCDs. Only applies when the map product holds confidence
    (output_map_format=confidence); weight maps keep physical units.
    """
    if config.get("output_params", {}).get("output_map_format", "weight").lower() != "confidence":
        print("  Confidence global norm skipped (map product holds weight, not confidence).")
        return
    map_path = (paths or {}).get("out_map_path")
    writer = (writers or {}).get("map")
    if not map_path or writer is None:
        return
    pooled = np.concatenate([np.ravel(s) for s in conf_samples.values()])
    global_p99 = float(np.percentile(pooled, 99.0))
    if not np.isfinite(global_p99) or global_p99 <= 0:
        return
    factors = {i: p99 / global_p99 for i, p99 in conf_p99.items() if p99 > 0}
    # Rewriting an existing HDU: only pass ``compress`` when actually
    # compressing. fitsio ignores/ warns about a placeholder value.
    compress = bool((config or {}).get("output_params", {}).get("compress", False)) or str(map_path).endswith(".fz")
    write_kwargs = {"compress": "RICE_1"} if compress else {}
    try:
        with fitsio.FITS(map_path, "rw") as f:
            for hdu_index, factor in factors.items():
                pos = writer.positions.get(hdu_index)
                if pos is None or pos >= len(f):
                    continue
                data = f[pos].read()
                f[pos].write(np.clip(data * factor, 0.0, 1.0).astype(np.float32), **write_kwargs)
    except OSError as e:
        print(f"  WARNING: confidence global rescale failed: {e}")
        return
    print(f"  Confidence renormalized to exposure-global p99 {global_p99:.3g}.")


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
                hdu_flat_obj = ff[i] if i < len(ff) else None
            else:
                hdu_flat_obj = _hdu_at(hdul_flat, i)
            if hdul_badpix is not None and bp_path is not None:
                fb = fitsio.FITS(bp_path, "r")
                opened.append(fb)
                hdu_badpix_obj = fb[i] if i < len(fb) else None
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


def _streak_catalog(hdu_index, mask_data):
    """Pixel coordinates of the STREAK bit. Small enough to keep after the HDU is flushed."""
    if mask_data is None:
        return None
    ys, xs = np.nonzero((np.asarray(mask_data) & np.uint32(QualityBit.STREAK)) != 0)
    if ys.size < 24:
        return None
    return {
        "hdu": int(hdu_index),
        "ys": np.asarray(ys, dtype=np.int32),
        "xs": np.asarray(xs, dtype=np.int32),
        "shape": tuple(np.shape(mask_data)),
    }


def _rewrite_hdu(fits_obj, pos, data, compress):
    kwargs = {"compress": "RICE_1"} if compress else {}
    fits_obj[pos].write(np.ascontiguousarray(data), **kwargs)


def _clear_chip_replicas(catalogs, writers, config):
    """Drop STREAK where the same CCD-local line was written on another chip.

    Coordinates were recorded at flush time, so this does not hold the MEF's
    arrays. Weight is restored from inverse variance only where STREAK was the
    only zero-weight bit and the map product is a weight.
    """
    if len(catalogs) < 2:
        return
    geoms = [line_geometry(cat["ys"], cat["xs"], cat["shape"]) for cat in catalogs]
    cleared = replica_indices(geoms)
    if not cleared:
        return
    mask_writer = (writers or {}).get("mask")
    if mask_writer is None or not getattr(mask_writer, "out_path", None):
        return
    output_format = str((config or {}).get("output_params", {}).get("output_map_format", "weight")).lower()
    weight_format = output_format == "weight"
    # Resolve compression once for every product HDU: explicit config OR the
    # fpack convention that a .fz file carries RICE-compressed HDUs. All products
    # share the same output dir/base, so they are .fz together or not at all.
    compress = bool((config or {}).get("output_params", {}).get("compress", False)) or str(
        mask_writer.out_path
    ).endswith(".fz")
    streak_bit = np.uint32(QualityBit.STREAK)
    other_bits = np.uint32(int(DEFAULT_ZERO_WEIGHT_BITS & ~QualityBit.STREAK))
    map_writer = writers.get("map") if weight_format else None
    ivar_writer = writers.get("invvar") if weight_format else None
    raw_writer = writers.get("weight_raw") if weight_format else None
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
        n_cleared = 0
        for index in sorted(cleared):
            cat = catalogs[index]
            pos = mask_writer.positions.get(cat["hdu"])
            if pos is None or pos >= len(fmask):
                continue
            data = np.array(fmask[pos].read(), copy=True)
            ys, xs = cat["ys"], cat["xs"]
            if ys.size == 0 or data.shape != cat["shape"]:
                continue
            pixels = data[ys, xs]
            only = ((pixels & streak_bit) != 0) & ((pixels & other_bits) == 0)
            data[ys, xs] = pixels & ~np.array(streak_bit, dtype=data.dtype)
            _rewrite_hdu(fmask, pos, data, compress)
            if fmap is not None and fivar is not None and np.any(only):
                wpos = map_writer.positions.get(cat["hdu"])
                ipos = ivar_writer.positions.get(cat["hdu"])
                if wpos is not None and ipos is not None and wpos < len(fmap) and ipos < len(fivar):
                    weight = np.array(fmap[wpos].read(), copy=True)
                    ivar = fivar[ipos].read()
                    if weight.shape == data.shape and np.shape(ivar) == data.shape:
                        weight[ys[only], xs[only]] = ivar[ys[only], xs[only]]
                        _rewrite_hdu(fmap, wpos, weight.astype(np.float32, copy=False), compress)
            if fraw is not None and fivar is not None and np.any(only):
                rpos = raw_writer.positions.get(cat["hdu"])
                ipos = ivar_writer.positions.get(cat["hdu"])
                if rpos is not None and ipos is not None and rpos < len(fraw) and ipos < len(fivar):
                    raw = np.array(fraw[rpos].read(), copy=True)
                    ivar = fivar[ipos].read()
                    if raw.shape == np.shape(ivar):
                        raw[ys[only], xs[only]] = ivar[ys[only], xs[only]]
                        _rewrite_hdu(fraw, rpos, raw.astype(np.float32, copy=False), compress)
            if find is not None:
                spos = ind_writer.positions.get(cat["hdu"])
                if spos is not None and spos < len(find):
                    ind = np.array(find[spos].read(), copy=True)
                    if ind.shape == cat["shape"]:
                        ind[ys, xs] = 0
                        _rewrite_hdu(find, spos, ind.astype(np.uint8, copy=False), compress)
            n_cleared += 1
        if n_cleared:
            print(f"  Chip-replica veto cleared STREAK on {n_cleared} HDUs.")
    except OSError as exc:
        print(f"  WARNING: chip-replica veto failed: {exc}")
    finally:
        for handle in handles:
            try:
                handle.close()
            except Exception:
                pass


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
                    flat_data = np.ascontiguousarray(hdul_flat[i].read().astype(np.float32))
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
    conf_samples: dict = {}
    conf_p99: dict = {}
    dark_cfg_global = config.get("dark_masking")

    def _merge_dark(i, hdu_sci, pre):
        if dark_cfg_global is None:
            return pre
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
                        dark_hdu = None
                except Exception:
                    dark_hdu = None
            elif hdul_dark is not None:
                try:
                    dark_hdu = hdul_dark[i] if i < len(hdul_dark) else None
                except Exception:
                    dark_hdu = None
            if dark_hdu is None:
                return pre
            try:
                dark_data = np.ascontiguousarray(dark_hdu.read().astype(np.float32))
                dark_hot = detect_bad_pixels(dark_data, dark_cfg_global, using_unit_flat=False)
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
        except Exception as e:
            import traceback

            print(f"FATAL ERROR processing HDU {i}: {e}\n{traceback.format_exc()}")
            return (i, (None, None, None, None, None, None), None, f"HDU{i}")
        finally:
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
            if conf_scope == "per_exposure" and weight_map is not None:
                wpos = weight_map[weight_map > 0]
                if wpos.size > 0:
                    step = max(1, wpos.size // 20000)
                    conf_samples[i] = np.ascontiguousarray(wpos[::step])
                    conf_p99[i] = float(np.percentile(conf_samples[i], 99.0))
            bad_mask, sat_mask, cr_mask, obj_mask, streak_mask, nodata_mask = extract_individual_masks(
                header_info, mask_data
            )
            catalog = _streak_catalog(i, mask_data)
            if catalog is not None:
                streak_catalogs.append(catalog)
            process_success_count += 1
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
                except Exception as e:
                    import traceback

                    print(f"FATAL ERROR processing HDU {i}: {e}\n{traceback.format_exc()}")
                    _i, _res, _hdr, _nm = i, (None, None, None, None, None, None), None, f"HDU{i}"
                _emit(_i, _res, _hdr, _nm)
                _submit_next()
    _clear_chip_replicas(streak_catalogs, writers, config)
    if conf_scope == "per_exposure" and conf_samples:
        _rescale_confidence_to_global(paths, writers, conf_samples, conf_p99, config)
    return process_success_count


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

    def __init__(self, out_path: str, hdul_input, primary_header, compress: bool = False, dtype=None):
        self.out_path = out_path
        self.hdul_input = hdul_input
        self.primary_header = primary_header
        self._opened = False
        self.positions: dict = {}
        self._next_data_pos = 0
        self._lock = threading.Lock()
        self.compress = bool(compress)
        self.dtype = np.dtype(dtype) if dtype is not None else None

    def _prep(self, data):
        if data is None or self.dtype is None:
            return data
        try:
            return np.ascontiguousarray(data, dtype=self.dtype)
        except Exception:
            return np.asarray(data, dtype=self.dtype)

    def write(self, hdu_index: int, data, header, extname: str) -> None:
        data = self._prep(data)
        # Only pass ``compress`` when compressing: fitsio warns about (and
        # ignores) a placeholder value such as "NOT_SET".
        kwargs = {"compress": "RICE_1"} if self.compress else {}
        with self._lock:
            if not self._opened:
                single_hdu0 = hdu_index == 0 and len(self.hdul_input) == 1
                if single_hdu0:
                    fitsio.write(self.out_path, data, header=header, clobber=True, **kwargs)
                    self._opened = True
                    self.positions[hdu_index] = 0
                    self._next_data_pos = 1
                    return
                if hdu_index == 0:
                    primary_data = data
                    primary_header = header
                else:
                    primary_data = None
                    primary_header = self.primary_header
                    self._next_data_pos = 1
                fitsio.write(self.out_path, primary_data, header=primary_header, clobber=True, **kwargs)
                self._opened = True
                if hdu_index == 0:
                    self.positions[hdu_index] = 0
                    self._next_data_pos = 1
                    return
            with fitsio.FITS(self.out_path, "rw") as f_out:
                f_out.write(data, header=header, extname=extname, **kwargs)
            self.positions[hdu_index] = self._next_data_pos
            self._next_data_pos += 1


def _wire_dtype(bitpix, *, is_mask: bool):
    """Map configured bitpix to a numpy wire dtype (mask uint, float otherwise)."""
    try:
        b = int(bitpix)
    except (TypeError, ValueError):
        return np.uint16 if is_mask else np.float32
    if is_mask:
        return {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}.get(b, np.uint16)
    return {16: np.float16, 32: np.float32, 64: np.float64, -32: np.float32, -64: np.float64}.get(b, np.float32)


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
            writers[key] = _StreamingMapWriter(out_path, hdul_input, primary_header, compress=compress, dtype=dt)
    for mask_type in ("bad", "sat", "cr", "obj", "streak", "nodata"):
        out_path = (paths.get("individual_mask_paths") or {}).get(mask_type)
        if out_path:
            writers[f"ind_{mask_type}"] = _StreamingMapWriter(
                out_path, hdul_input, primary_header, compress=compress, dtype=np.uint8
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
