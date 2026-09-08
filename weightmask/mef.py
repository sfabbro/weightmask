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
from .bad import _get_global_median, compute_flat_bad_mask, detect_bad_pixels
from .contract import (
    CONFIDENCE_SEMANTICS,
    INVERSE_VARIANCE_SEMANTICS,
    MASK_POLARITY,
    ArtifactMetadata,
    ProducerMetadata,
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
    except OSError as e:
        print(f"Skipping HDU: Cannot read science data: {e}")
        return None, None, None, None, None, None

    flat_data_full = None
    if hdu_flat is not None:
        try:
            flat_data_full = np.ascontiguousarray(hdu_flat.read().astype(np.float32))
        except OSError as e:
            print(f"Skipping HDU: Cannot read flat data: {e}")
            return None, None, None, None, None, None

    bad_mask = None
    if precomputed_bad_mask is not None:
        bad_mask = precomputed_bad_mask
    elif flat_data_full is not None:
        eff = _effective_tile_size(tile_size, sci_data_full.shape)
        bad_mask = compute_flat_bad_mask(flat_data_full, config.get("flat_masking", {}), eff)

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
        sci_data_full, sci_hdr, flat_data_full, config, tile_size, bad_mask=bad_mask, badpix_mask=badpix_mask
    )


# --- Main execution function called by entry point ---


def extract_individual_masks(header_info: dict, mask_data):
    shape = mask_data.shape if mask_data is not None else (0, 0)
    if header_info and "individual_masks" in header_info:
        individual_masks = header_info["individual_masks"]
        bad_mask = individual_masks.get("bad", np.zeros(shape, dtype=bool)) if mask_data is not None else np.array([])
        sat_mask = individual_masks.get("sat", np.zeros(shape, dtype=bool)) if mask_data is not None else np.array([])
        cr_mask = individual_masks.get("cr", np.zeros(shape, dtype=bool)) if mask_data is not None else np.array([])
        obj_mask = individual_masks.get("obj", np.zeros(shape, dtype=bool)) if mask_data is not None else np.array([])
        streak_mask = (
            individual_masks.get("streak", np.zeros(shape, dtype=bool)) if mask_data is not None else np.array([])
        )
    else:
        bad_mask = np.zeros(shape, dtype=bool) if mask_data is not None else np.array([])
        sat_mask = np.zeros(shape, dtype=bool) if mask_data is not None else np.array([])
        cr_mask = np.zeros(shape, dtype=bool) if mask_data is not None else np.array([])
        obj_mask = np.zeros(shape, dtype=bool) if mask_data is not None else np.array([])
        streak_mask = np.zeros(shape, dtype=bool) if mask_data is not None else np.array([])

    return bad_mask, sat_mask, cr_mask, obj_mask, streak_mask


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
    compress = bool((config or {}).get("output_params", {}).get("compress", False))
    wcomp = "RICE_1" if (compress or str(map_path).endswith(".fz")) else "NOT_SET"
    try:
        with fitsio.FITS(map_path, "rw") as f:
            for hdu_index, factor in factors.items():
                pos = writer.positions.get(hdu_index)
                if pos is None or pos >= len(f):
                    continue
                data = f[pos].read()
                f[pos].write(np.clip(data * factor, 0.0, 1.0).astype(np.float32), compress=wcomp)
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
            try:
                tmp_n = fi[i]
                hdu_name = getattr(tmp_n, "name", f"HDU{i}") if hasattr(tmp_n, "name") else f"HDU{i}"
            except Exception:
                hdu_name = f"HDU{i}"
            return hdu_sci, hdu_flat_obj, hdu_badpix_obj, hdu_header_raw, hdu_name, close, None

        hdu_sci = hdul_input[i]
        hdu_flat_obj = _hdu_at(hdul_flat, i)
        hdu_badpix_obj = _hdu_at(hdul_badpix, i)
        try:
            hdu_header_raw = hdul_input[i].read_header()
        except Exception:
            hdu_header_raw = None
        try:
            tmp = hdul_input[i]
            hdu_name = getattr(tmp, "name", f"HDU{i}") if hasattr(tmp, "name") else f"HDU{i}"
        except Exception:
            hdu_name = f"HDU{i}"
        return hdu_sci, hdu_flat_obj, hdu_badpix_obj, hdu_header_raw, hdu_name, close, None
    except Exception as e:
        close()
        return None, None, None, None, f"HDU{i}", (lambda: None), str(e)


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
) -> int:
    """Process every requested HDU, streaming each HDU's outputs to disk.
    Each HDU's maps are written to their output files as soon as they are
    produced instead of being accumulated in memory. A 36-CCD MegaPrime MEF
    therefore only ever holds a single CCD's products at a time (~a few
    hundred MB) rather than the whole MEF's (~7 GB), which previously blew
    through the container memory limit when several exposures were processed
    concurrently and OOM-killed the batch.
    Parallel by default: per-HDU ThreadPoolExecutor; single-HDU inputs run
    inline (per-file fast path). Compute runs concurrently, writes are
    applied in HDU-index order for deterministic output.
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
                sci_shape = hdu_sci.read().shape
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

        def close():
            return None

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
            result = process_hdu(
                hdu_sci,
                hdu_flat_obj,
                config,
                i,
                tile_size=tile_size,
                flat_path=flat_path,
                hdu_badpix=hdu_badpix_obj,
                precomputed_bad_mask=pre2,
            )
            return (i, result, hdu_header_raw, hdu_name)
        except Exception as e:
            import traceback

            print(f"FATAL ERROR processing HDU {i}: {e}\n{traceback.format_exc()}")
            return (i, (None, None, None, None, None, None), None, f"HDU{i}")
        finally:
            close()

    results: dict = {}
    if len(hdus_to_process) <= 1 or eff_workers <= 1:
        for i in hdus_to_process:
            _i, _res, _hdr, _nm = _compute_one(i)
            results[_i] = (_res, _hdr, _nm)
    else:
        npool = max(1, min(eff_workers, len(hdus_to_process)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=npool) as ex:
            futs = {ex.submit(_compute_one, i): i for i in hdus_to_process}
            for fut in concurrent.futures.as_completed(futs):
                try:
                    _i, _res, _hdr, _nm = fut.result()
                except Exception as e:
                    import traceback

                    _i = futs[fut]
                    print(f"FATAL ERROR processing HDU {_i}: {e}\n{traceback.format_exc()}")
                    results[_i] = ((None, None, None, None, None, None), None, f"HDU{_i}")
                    continue
                results[_i] = (_res, _hdr, _nm)
    for i in hdus_to_process:
        _res, _hdr_raw, _nm = results.get(i, ((None, None, None, None, None, None), None, f"HDU{i}"))
        try:
            if _res is None or _res[0] is None:
                print(f"Skipping HDU {i} due to processing errors.")
                continue
            (mask_data, inv_var_data, weight_map, confidence_map, sky_map, header_info) = _res
            if conf_scope == "per_exposure" and weight_map is not None:
                wpos = weight_map[weight_map > 0]
                if wpos.size > 0:
                    step = max(1, wpos.size // 20000)
                    conf_samples[i] = np.ascontiguousarray(wpos[::step])
                    conf_p99[i] = float(np.percentile(conf_samples[i], 99.0))
            process_success_count += 1
            bad_mask, sat_mask, cr_mask, obj_mask, streak_mask = extract_individual_masks(header_info, mask_data)
            hdu_name = _nm if isinstance(_nm, str) and _nm else f"HDU{i}"
            hdu_header = _hdr_raw
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
                paths,
                sky_header=sky_header,
            )
            _flush_hdu_output(writers, hdu_output, i)
        except Exception as e:
            import traceback

            print(f"FATAL ERROR processing HDU {i}: {e}\n{traceback.format_exc()}")
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
        comp = "RICE_1" if self.compress else "NOT_SET"
        with self._lock:
            if not self._opened:
                single_hdu0 = hdu_index == 0 and len(self.hdul_input) == 1
                if single_hdu0:
                    fitsio.write(self.out_path, data, header=header, clobber=True, compress=comp)
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
                fitsio.write(self.out_path, primary_data, header=primary_header, clobber=True, compress=comp)
                self._opened = True
                if hdu_index == 0:
                    self.positions[hdu_index] = 0
                    self._next_data_pos = 1
                    return
            with fitsio.FITS(self.out_path, "rw") as f_out:
                f_out.write(data, header=header, extname=extname, compress=comp)
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
    for mask_type in ("bad", "sat", "cr", "obj", "streak"):
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
    for mask_type in ("bad", "sat", "cr", "obj", "streak"):
        entry = individual.get(mask_type)
        writer_key = f"ind_{mask_type}"
        if entry is not None and writer_key in writers:
            writers[writer_key].write(hdu_index, entry["data"], entry["header"], entry["name"])
