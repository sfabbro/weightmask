#!/usr/bin/env python3
"""
WeightMask CLI using fitsio instead of astropy.io.fits for better performance with MEF files.
"""

import argparse
import concurrent.futures
import hashlib
import os
import threading
import time
from typing import Optional, Tuple

import fitsio
import numpy as np
import yaml

from . import MASK_BITS, MASK_DTYPE, __version__  # Import from __init__.py
from .background import estimate_background

# Import from other modules within the package using relative imports
from .bad import _get_global_median, compute_flat_bad_mask, detect_bad_pixels
from .contract import (
    CONFIDENCE_SEMANTICS,
    INVERSE_VARIANCE_SEMANTICS,
    MASK_POLARITY,
    ArtifactMetadata,
    ProducerMetadata,
)
from .cosmics import detect_cosmic_rays
from .objects import detect_objects
from .satur import detect_saturated_pixels, grow_bleed_trails
from .streaks import detect_streaks
from .utils import clean_config_dict, extract_hdu_spec
from .variance import calculate_inverse_variance
from .weight import generate_weight_and_confidence

_CFG_HASH_KEY = "WMCFGH"


def flat_bad_mask_cache_path(flat_path: str) -> str:
    """On-disk cache path for a flat's bad-pixel mask: ``<flat>.mask.fits``.

    The cache is a MEF whose image HDUs line up one-to-one with the flat's, so
    an exposure's HDU index maps straight onto the cached mask for that CCD.
    """
    for suffix in (".fits.fz", ".fits"):
        if flat_path.endswith(suffix):
            return flat_path[: -len(suffix)] + ".mask.fits"
    return flat_path + ".mask.fits"


def _flat_cfg_hash(flat_cfg: dict, tile_size: int = 1024) -> str:
    material = repr(sorted(flat_cfg.items())) + f"|tile:{int(tile_size)}"
    return hashlib.md5(material.encode("utf-8")).hexdigest()[:8]


def _flat_image_indices(hdul) -> list[int]:
    return [
        i for i in range(len(hdul)) if hdul[i].get_info().get("hdutype") == 0 and hdul[i].get_info().get("ndims") == 2
    ]


def _write_flat_bad_mask_cache(cache_path: str, flat_path: str, flat_cfg: dict, tile_size: int) -> None:
    """Compute every HDU's bad mask for one flat and write them to ``cache_path`` atomically."""
    masks: dict[int, np.ndarray] = {}
    with fitsio.FITS(flat_path, "r") as ff:
        for i in _flat_image_indices(ff):
            masks[i] = compute_flat_bad_mask(ff[i].read().astype(np.float32), flat_cfg, tile_size)

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    primary_header = {_CFG_HASH_KEY: _flat_cfg_hash(flat_cfg, tile_size)}
    idxs = sorted(masks)
    if idxs and idxs[0] == 0:
        # Single-image primary (not the MegaCam layout, but be safe).
        fitsio.write(tmp_path, masks[0].astype(np.uint8), header=primary_header, clobber=True)
        idxs = idxs[1:]
    else:
        fitsio.write(tmp_path, None, header=primary_header, clobber=True)
    for i in idxs:
        with fitsio.FITS(tmp_path, "rw") as f:
            f.write(masks[i].astype(np.uint8), extname=f"BAD_{i}")
    os.replace(tmp_path, cache_path)


def load_flat_bad_mask(flat_path: str, hdu_index: int, config: dict, tile_size: int = 1024) -> Optional[np.ndarray]:
    """Load one HDU's bad mask from the ``<flat>.mask.fits`` cache, or None if missing/stale."""
    cache_path = flat_bad_mask_cache_path(flat_path)
    if not os.path.exists(cache_path):
        return None
    try:
        with fitsio.FITS(cache_path, "r") as f:
            if hdu_index >= len(f):
                return None
            if f[0].read_header().get(_CFG_HASH_KEY) != _flat_cfg_hash(config.get("flat_masking", {}), tile_size):
                return None
            info = f[hdu_index].get_info()
            if info.get("hdutype") != 0 or info.get("ndims") != 2:
                return None
            return f[hdu_index].read().astype(bool)
    except OSError:
        return None


def ensure_flat_bad_mask_cache(flat_path: str, config: dict, tile_size: int = 1024) -> str:
    """Compute (or reuse) a flat's full bad-pixel mask cache and return its path."""
    cache_path = flat_bad_mask_cache_path(flat_path)
    expected_hash = _flat_cfg_hash(config.get("flat_masking", {}), tile_size)
    try:
        with fitsio.FITS(cache_path, "r") as f:
            if f[0].read_header().get(_CFG_HASH_KEY) == expected_hash:
                return cache_path
    except OSError:
        pass
    try:
        _write_flat_bad_mask_cache(cache_path, flat_path, config.get("flat_masking", {}), tile_size)
    except OSError as e:
        print(f"  WARNING: could not write badmask cache {cache_path}: {e}")
    return cache_path


def get_or_compute_flat_bad_mask(
    flat_path: str, hdu_index: int, flat_data: np.ndarray, config: dict, tile_size: int = 1024
) -> np.ndarray:
    """Return the bad-pixel/column mask for one flat HDU, cached in ``<flat>.mask.fits``.

    The mask depends only on the flat (plus ``flat_masking`` config and tile size), so it is
    computed once per flat and reused across every exposure that shares that
    flat. Computing it from scratch dominates weightmask runtime (a 15x15
    median filter per tile), so this cache is the single biggest speedup. The
    cache is written atomically (temp file + rename) so concurrent workers never
    observe a half-written file.
    """
    mask = load_flat_bad_mask(flat_path, hdu_index, config, tile_size)
    if mask is not None and mask.shape == flat_data.shape:
        return mask

    ensure_flat_bad_mask_cache(flat_path, config, tile_size)
    mask = load_flat_bad_mask(flat_path, hdu_index, config, tile_size)
    if mask is not None and mask.shape == flat_data.shape:
        return mask

    # Cache unavailable (e.g. read-only disk): fall back to computing this HDU inline.
    return compute_flat_bad_mask(flat_data, config.get("flat_masking", {}), tile_size)


def validate_fits_file(file_path: str) -> bool:
    """Validate that a file is a proper FITS file."""
    try:
        with fitsio.FITS(file_path, "r") as f:
            if len(f) == 0:
                print(f"ERROR: FITS file {file_path} appears to be empty.")
                return False
            return True
    except OSError as e:
        print(f"ERROR: Cannot open FITS file {file_path}: {e}")
        return False


def validate_config(config: dict) -> bool:
    """Validate configuration parameters."""
    required_sections = [
        "flat_masking",
        "saturation",
        "sep_background",
        "cosmic_ray",
        "sep_objects",
        "streak_masking",
        "variance",
        "confidence_params",
        "output_params",
    ]
    allowed_sections = set(required_sections) | {"dark_masking"}
    for section in required_sections:
        if section not in config:
            print(f"WARNING: Required configuration section '{section}' missing.")

    extra_sections = sorted(set(config) - allowed_sections)
    if extra_sections:
        print(f"ERROR: Unsupported top-level configuration sections: {', '.join(extra_sections)}")
        return False

    dict_sections = [
        "flat_masking",
        "saturation",
        "sep_background",
        "cosmic_ray",
        "sep_objects",
        "streak_masking",
        "variance",
        "confidence_params",
        "output_params",
        "dark_masking",
    ]
    for section in dict_sections:
        if section in config and not isinstance(config[section], dict):
            print(f"ERROR: '{section}' section must be a dictionary.")
            return False

    if "variance" in config:
        var_method = config["variance"].get("method", "theoretical")
        if var_method not in ["theoretical", "rms_map", "empirical_fit"]:
            print(f"ERROR: Invalid variance method '{var_method}'.")
            return False

        misplaced_flat_keys = {
            "local_filter_size",
            "local_low_thresh",
            "local_high_thresh",
            "col_enable",
            "col_deriv_sigma",
            "col_dead_thresh",
        }
        wrong_place = sorted(misplaced_flat_keys & set(config["variance"]))
        if wrong_place:
            print(
                "ERROR: Bad-pixel tuning keys must be under 'flat_masking', not 'variance': " + ", ".join(wrong_place)
            )
            return False

    if "sep_background" in config:
        background_method = config["sep_background"].get("method", "sep")
        if background_method not in ["sep", "median_filter", "robust_median_fallback"]:
            print(f"ERROR: Invalid background method '{background_method}'.")
            return False

    if "streak_masking" in config:
        streak_method = config["streak_masking"].get("method")
        streak_mode = config["streak_masking"].get("mode")
        allowed_values = ["auto_ground", "satdet_only", "mrt_only", "legacy_compare", "satdet", "frangi_legacy"]
        if streak_method is not None and streak_method not in allowed_values:
            print(f"ERROR: Invalid streak masking method '{streak_method}'.")
            return False
        if streak_mode is not None and streak_mode not in allowed_values:
            print(f"ERROR: Invalid streak masking mode '{streak_mode}'.")
            return False
        legacy_keys = {"enable_ransac_trails", "ransac_params", "frangi_params"}
        stale_keys = sorted(legacy_keys & set(config["streak_masking"]))
        if stale_keys:
            print(
                "ERROR: Legacy streak keys are no longer supported. "
                "Use 'enable_sparse_ransac', 'sparse_ransac_params', and 'frangi_legacy_params': "
                + ", ".join(stale_keys)
            )
            return False
    for _sec, _key in (("variance", "gain_keyword"), ("variance", "readnoise_keyword"), ("variance", "rdnoise_keyword"), ("saturation", "keyword")):
        _cfg_sec = config.get(_sec, {}) if isinstance(config.get(_sec, {}), dict) else {}
        if _key in _cfg_sec:
            _v = _cfg_sec[_key]
            _ok = isinstance(_v, str) or (isinstance(_v, (list, tuple)) and all(isinstance(_k, str) for _k in _v))
            if not _ok:
                print(f"ERROR: '{_sec}.{_key}' must be a header keyword string or list of strings.")
                return False
    _op = config.get("output_params", {}) if isinstance(config.get("output_params", {}), dict) else {}
    if "mask_bitpix" in _op and _op["mask_bitpix"] not in (8, 16, 32, 64):
        print("ERROR: 'output_params.mask_bitpix' must be one of 8/16/32/64.")
        return False
    if "ivar_bitpix" in _op and _op["ivar_bitpix"] not in (16, 32, 64, -32, -64):
        print("ERROR: 'output_params.ivar_bitpix' must be one of 16/32/64/-32/-64.")
        return False
    if "compress" in _op and not isinstance(_op["compress"], bool):
        print("ERROR: 'output_params.compress' must be a boolean.")
        return False
    return True


def _header_lookup(header, key_cfg, default):
    """First-present header value for a str-or-list keyword config, else default."""
    if header is None:
        return default
    keys = key_cfg if isinstance(key_cfg, (list, tuple)) else [key_cfg]
    get = getattr(header, "get", None)
    for k in keys:
        if not isinstance(k, str) or not k:
            continue
        try:
            v = get(k, None) if callable(get) else header.get(k, None) if hasattr(header, "get") else None
        except Exception:
            v = None
        if v is None:
            try:
                if k in header:
                    v = header[k]
            except Exception:
                v = None
        if v is not None:
            return v
    return default


def _effective_tile_size(tile_size, shape) -> int:
    try:
        t = int(tile_size)
    except (TypeError, ValueError):
        t = 1024
    try:
        min_dim = int(min(shape))
    except Exception:
        return max(16, t)
    return max(16, min(t, max(1, min_dim // 2)))


def process_image(
    sci_data_full: np.ndarray,
    sci_hdr: dict,
    flat_data_full: Optional[np.ndarray],
    config: dict,
    tile_size: int = 1024,
    bad_mask: Optional[np.ndarray] = None,
    badpix_mask: Optional[np.ndarray] = None,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[dict],
]:
    """Processes a single Science image to generate all mask and map products."""
    hdu_start_time = time.time()
    sci_shape = sci_data_full.shape
    using_unit_flat = True
    eff_tile = _effective_tile_size(tile_size, sci_shape)
    if flat_data_full is not None:
        using_unit_flat = False
        if flat_data_full.shape != sci_shape:
            print("Skipping processing: Flat data shape mismatch.")
            return None, None, None, None, None, None
    else:
        print("  INFO: No flat field provided, assuming flat = 1.0.")
        flat_data_full = np.ones_like(sci_data_full, dtype=np.float32)

    # --- 0. Initial Setup ---
    final_mask_int = np.zeros(sci_shape, dtype=MASK_DTYPE)
    header_info: dict = {}

    # Initialize individual masks
    sat_mask = np.zeros(sci_shape, dtype=bool)
    cr_mask = np.zeros(sci_shape, dtype=bool)
    obj_mask = np.zeros(sci_shape, dtype=bool)
    streak_mask = np.zeros(sci_shape, dtype=bool)

    # --- 1. Tile-based Masking (Bad Pixels, Saturation) ---
    print("  (1/7) Processing Bad Pixel mask...")
    if bad_mask is None:
        bad_mask = np.zeros(sci_shape, dtype=bool)
        if flat_data_full is not None and not using_unit_flat:
            # Compute flat bad mask ONCE for the full HDU (replaces per-tile + cache)
            print("    Computing flat bad-pixel mask (full HDU)...")
            flat_bad_mask = compute_flat_bad_mask(flat_data_full, config.get("flat_masking", {}), eff_tile)
            bad_mask |= flat_bad_mask
        else:
            # No flat provided: unit-flat fallback (generic instruments without
            # flat calibration still produce valid weights).
            for y in range(0, sci_shape[0], eff_tile):
                for x in range(0, sci_shape[1], eff_tile):
                    tile_slice = (slice(y, y + eff_tile), slice(x, x + eff_tile))
                    sci_data_tile = sci_data_full[tile_slice]
                    flat_data_tile = flat_data_full[tile_slice]
                    if not np.isfinite(sci_data_tile).any():
                        continue
                    flat_mask_bool_tile = detect_bad_pixels(flat_data_tile, config.get("flat_masking", {}), using_unit_flat)
                    bad_mask[tile_slice] |= flat_mask_bool_tile
    if badpix_mask is not None:
        bad_mask = bad_mask | badpix_mask
    final_mask_int[bad_mask] |= MASK_BITS["BAD"]

    print("  (1.1/7) Detecting saturation on the full image...")
    saturation_level, sat_method_used, sat_mask = detect_saturated_pixels(
        sci_data_full, sci_hdr, config.get("saturation", {})
    )
    final_mask_int[sat_mask] |= MASK_BITS["SAT"]
    header_info["SAT_LVL"], header_info["SAT_METH"] = saturation_level, sat_method_used

    # --- Full Image Processing Steps ---
    interim_mask_bool = final_mask_int > 0

    # Calculate preliminary background RMS for CR and Bleed masking
    print("  Calculating preliminary background RMS...")
    prelim_bkg_map, prelim_bkg_rms = estimate_background(
        sci_data_full, interim_mask_bool, config.get("sep_background", {})
    )

    # --- 1.5 Bleed Trail (Blooming) Masking ---
    sat_cfg = config.get("saturation", {})
    if sat_cfg.get("mask_bleed_trails", True):
        print("  (1.5/7) Growing Bleed Trails for saturated stars...")
        sat_mask_full = grow_bleed_trails(sci_data_full, sat_mask, prelim_bkg_map, prelim_bkg_rms, sat_cfg)
        new_bleed_pixels = sat_mask_full & ~sat_mask
        sat_mask |= new_bleed_pixels
        final_mask_int[new_bleed_pixels] |= MASK_BITS["SAT"]
        interim_mask_bool |= new_bleed_pixels
        print(f"      Added {np.count_nonzero(new_bleed_pixels)} bleed trail pixels.")

    # --- 2. First-Pass Cosmic Ray Detection ---
    print("  (2/7) Running first-pass Cosmic Ray detection...")
    cosmic_cfg = config.get("cosmic_ray", {})
    variance_cfg = config.get("variance", {})
    gain_raw = _header_lookup(sci_hdr, variance_cfg.get("gain_keyword", "GAIN"), variance_cfg.get("default_gain", 1.0))
    rdnoise_raw = _header_lookup(
        sci_hdr, variance_cfg.get("rdnoise_keyword", variance_cfg.get("readnoise_keyword", "RDNOISE")),
        variance_cfg.get("default_rdnoise", 0.0),
    )
    try:
        gain = float(gain_raw)
    except (ValueError, TypeError):
        gain = float(variance_cfg.get("default_gain", 1.0))
    try:
        read_noise_e = float(rdnoise_raw)
    except (ValueError, TypeError):
        read_noise_e = float(variance_cfg.get("default_rdnoise", 0.0))

    cr_add_mask = detect_cosmic_rays(
        sci_data_full,
        interim_mask_bool,
        saturation_level,
        gain,
        read_noise_e,
        cosmic_cfg,
        bkg_rms_map=prelim_bkg_rms,
    )
    cr_mask |= cr_add_mask
    final_mask_int[cr_add_mask] |= MASK_BITS["CR"]
    interim_mask_bool |= cr_add_mask
    print(f"      Masked {np.count_nonzero(cr_add_mask)} new CR pixels.")

    # --- 3. Iterative Background and Object Detection ---
    print("  (3/7) Starting iterative Background/Object detection...")
    sep_bg_cfg = config.get("sep_background", {})
    object_cfg = config.get("sep_objects", {})
    iterations = sep_bg_cfg.get("iterations", 2)
    current_obj_mask = np.zeros(sci_shape, dtype=bool)
    last_bg_mask = None
    last_bkg_map = None
    last_bkg_rms_map = None
    for i in range(iterations):
        print(f"    Iteration {i + 1}/{iterations}...")
        total_mask_for_bg = interim_mask_bool | current_obj_mask
        bkg_map, bkg_rms_map = estimate_background(sci_data_full, total_mask_for_bg, sep_bg_cfg)
        if bkg_map is None:
            return None, None, None, None, None, None

        data_sub = sci_data_full - bkg_map
        new_obj_add_mask = detect_objects(data_sub, bkg_rms_map, total_mask_for_bg, object_cfg)

        last_bg_mask = total_mask_for_bg
        last_bkg_map, last_bkg_rms_map = bkg_map, bkg_rms_map
        if np.count_nonzero(new_obj_add_mask) == 0 and i > 0:
            print("      No new objects found, ending iteration.")
            break
        current_obj_mask |= new_obj_add_mask

    obj_mask |= current_obj_mask
    final_mask_int[current_obj_mask] |= MASK_BITS["DETECTED"]
    print(f"      Masked {np.count_nonzero(current_obj_mask)} DETECTED pixels total.")

    # --- 4. Final Sky Maps and Object Mask ---
    print("  (4/7) Finalizing sky maps and object mask...")
    final_obj_mask = current_obj_mask
    final_full_mask = interim_mask_bool | final_obj_mask
    if last_bg_mask is not None and np.array_equal(final_full_mask, last_bg_mask):
        print("  Reusing iteration background (mask unchanged)...")
        sky_map, final_bkg_rms_map = last_bkg_map, last_bkg_rms_map
    else:
        sky_map, final_bkg_rms_map = estimate_background(sci_data_full, final_full_mask, sep_bg_cfg)
    if sky_map is None:
        return None, None, None, None, None, None

    # --- 5. Inverse Variance Map ---
    print("  (5/7) Calculating inverse variance map...")
    variance_cfg["gain"] = gain
    variance_cfg["read_noise"] = read_noise_e
    inv_variance_map = calculate_inverse_variance(
        variance_cfg,
        sky_map,
        flat_data_full,
        final_bkg_rms_map,
        sci_data=sci_data_full,
        obj_mask=final_obj_mask,
    )
    if inv_variance_map is None:
        return None, None, None, None, None, None

    # --- 6. Streak Detection ---
    print("  (6/7) Detecting streaks...")
    streak_cfg = config.get("streak_masking", {})
    if streak_cfg.get("enable", False):
        data_sub = sci_data_full - sky_map
        streak_add_mask = detect_streaks(data_sub, final_bkg_rms_map, final_full_mask, streak_cfg)
        streak_mask |= streak_add_mask
        final_mask_int[streak_add_mask] |= MASK_BITS["STREAK"]
        final_full_mask |= streak_add_mask
        print(f"      Masked {np.count_nonzero(streak_add_mask)} new STREAK pixels.")

    # --- 7. Generate Final Weight and Confidence Maps ---
    print("  (7/7) Generating final weight and confidence maps...")
    weight_map, confidence_map = generate_weight_and_confidence(inv_variance_map, final_mask_int, config)
    if weight_map is None:
        return None, None, None, None, None, None

    header_info["individual_masks"] = {
        "bad": bad_mask,
        "sat": sat_mask,
        "cr": cr_mask,
        "obj": obj_mask,
        "streak": streak_mask,
    }

    hdu_elapsed = time.time() - hdu_start_time
    print(f"--- Image processed in {hdu_elapsed:.2f} seconds ---")
    return (
        final_mask_int,
        inv_variance_map,
        weight_map,
        confidence_map,
        sky_map,
        header_info,
    )


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
        # Use pre-computed flat bad mask (computed once per flat HDU in process_all_hdus)
        bad_mask = precomputed_bad_mask
    elif flat_data_full is not None and flat_path:
        # Fallback to cache-based computation (for backward compatibility)
        bad_mask = get_or_compute_flat_bad_mask(flat_path, hdu_index, flat_data_full, config, tile_size)

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


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Mask and Weight/Confidence Maps for FITS files.")
    parser.add_argument("input_file", type=str, help="Path to input FITS file.")
    parser.add_argument(
        "--output_map",
        "-o",
        type=str,
        default=None,
        help="Path for primary output map (Weight or Confidence). Default: <input_base>.weight.fits",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (optional, attempts default locations).",
    )
    parser.add_argument(
        "--flat_image",
        type=str,
        default=None,
        help="Path to input flat field FITS file (optional).",
    )
    parser.add_argument(
        "--dark_image",
        type=str,
        default=None,
        help="Path to input dark frame FITS file (optional). Hot pixels are "
        "detected with the dark_masking section and OR'd into the BAD bit.",
    )
    parser.add_argument(
        "--badpix_mask",
        type=str,
        default=None,
        help="Path to an external bad-pixel mask MEF (Elixir keep-map convention: 0 = bad, 1 = good). "
        "Each HDU's zero-valued pixels are OR'd into the BAD quality bit.",
    )
    parser.add_argument(
        "--output_mask",
        type=str,
        default=None,
        help="Path for output bitmask FITS file (optional).",
    )
    parser.add_argument(
        "--output_invvar",
        type=str,
        default=None,
        help="Path for output inverse variance FITS file (optional).",
    )
    parser.add_argument(
        "--output_sky",
        type=str,
        default=None,
        help="Path for output sky background map file (optional).",
    )
    parser.add_argument(
        "--output_weight_raw",
        type=str,
        default=None,
        help="Path for unnormalized weight map (masked inv_var), if different from primary map.",
    )
    parser.add_argument(
        "--hdu",
        type=int,
        default=None,
        help="HDU index to process (e.g., 0, 1). Processes extensions if omitted.",
    )
    parser.add_argument(
        "--individual_masks",
        action="store_true",
        help="Output individual mask component files.",
    )
    parser.add_argument(
        "--nproc",
        "--max-workers",
        dest="max_workers",
        type=int,
        default=None,
        help="Max parallel HDU workers (default min(8, ncpu); 0/1 = sequential).",
    )
    return parser.parse_args()


def validate_input_files(args: argparse.Namespace) -> bool:
    if not os.path.exists(args.input_file):
        print(f"ERROR: Input file not found: {args.input_file}")
        return False

    if not validate_fits_file(args.input_file):
        print(f"ERROR: Input file validation failed: {args.input_file}")
        return False

    if args.flat_image:
        if not os.path.exists(args.flat_image):
            print(f"ERROR: Flat field file not found: {args.flat_image}")
            return False
        if not validate_fits_file(args.flat_image):
            print(f"ERROR: Flat field file validation failed: {args.flat_image}")
            return False

    if args.dark_image:
        if not os.path.exists(args.dark_image):
            print(f"ERROR: Dark frame file not found: {args.dark_image}")
            return False
        if not validate_fits_file(args.dark_image):
            print(f"ERROR: Dark frame file validation failed: {args.dark_image}")
            return False
    if args.badpix_mask:
        if not os.path.exists(args.badpix_mask):
            print(f"ERROR: Bad pixel mask file not found: {args.badpix_mask}")
            return False
        if not validate_fits_file(args.badpix_mask):
            print(f"ERROR: Bad pixel mask file validation failed: {args.badpix_mask}")
            return False

    return True


def _find_default_config() -> str:
    default_configs = ["weightmask.yml", "config.yml", ".weightmask.yml"]
    for cfg in default_configs:
        if os.path.exists(cfg):
            print(f"Using default config file found at: {cfg}")
            return cfg
    return None


def _read_and_clean_config(config_path: str) -> dict:
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            return clean_config_dict(config)
    except OSError as e:
        print(f"ERROR: Failed to read config file '{config_path}': {e}")
        return None
    except yaml.YAMLError as e:
        print(f"ERROR: Failed to parse config file '{config_path}': {e}")
        return None


def load_configuration(config_path: str) -> dict:
    if config_path is None:
        config_path = _find_default_config()
        if config_path is None:
            print("ERROR: Config file not specified and no default found.")
            return None

    config = _read_and_clean_config(config_path)
    if config is None:
        return None

    if not isinstance(config, dict):
        print(f"ERROR: Config file '{config_path}' must be a YAML dictionary.")
        return None

    if "output_params" not in config:
        config["output_params"] = {}
    if "confidence_params" not in config:
        config["confidence_params"] = {}
    if "output_map_format" not in config["output_params"]:
        config["output_params"]["output_map_format"] = "weight"
    config["output_params"].setdefault("mask_bitpix", 16)
    config["output_params"].setdefault("ivar_bitpix", 32)
    config["output_params"].setdefault("compress", False)

    if not validate_config(config):
        print("ERROR: Configuration validation failed.")
        return None

    return config


def determine_output_paths(args: argparse.Namespace, input_path: str, config: dict | None = None) -> dict:
    out_map_path = args.output_map
    compress = bool((config or {}).get("output_params", {}).get("compress", False))
    if out_map_path is None:
        input_basename = os.path.basename(str(input_path))
        if input_basename.endswith(".fits.fz"):
            base = input_basename[:-8]
        elif input_basename.endswith(".fits"):
            base = input_basename[:-5]
        else:
            base = os.path.splitext(input_basename)[0]
        output_dir = os.path.dirname(str(input_path)) or "."
        default_suffix = ".weight.fits.fz" if compress else ".weight.fits"
        out_map_path = os.path.join(output_dir, f"{base}{default_suffix}")
        print(f"Output map path not specified, using default: {out_map_path}")

    output_dir = os.path.dirname(out_map_path)
    base_out = os.path.splitext(os.path.basename(out_map_path))[0]
    out_mask_path = (
        args.output_mask or os.path.join(output_dir, f"{base_out}.mask.fits") if args.output_mask is not None else None
    )
    out_invvar_path = (
        args.output_invvar or os.path.join(output_dir, f"{base_out}.ivar.fits")
        if args.output_invvar is not None
        else None
    )
    out_sky_path = (
        args.output_sky or os.path.join(output_dir, f"{base_out}.sky.fits") if args.output_sky is not None else None
    )
    out_weight_raw_path = args.output_weight_raw

    individual_mask_paths = {}
    if args.individual_masks:
        individual_mask_paths = {
            "bad": os.path.join(output_dir, f"{base_out}.bad.fits"),
            "sat": os.path.join(output_dir, f"{base_out}.sat.fits"),
            "cr": os.path.join(output_dir, f"{base_out}.cr.fits"),
            "obj": os.path.join(output_dir, f"{base_out}.obj.fits"),
            "streak": os.path.join(output_dir, f"{base_out}.streak.fits"),
        }

    return {
        "out_map_path": out_map_path,
        "out_mask_path": out_mask_path,
        "out_invvar_path": out_invvar_path,
        "out_sky_path": out_sky_path,
        "out_weight_raw_path": out_weight_raw_path,
        "individual_mask_paths": individual_mask_paths,
    }


def open_fits_files(input_path: str, flat_path: str):
    try:
        hdul_input = fitsio.FITS(input_path, "r")
        hdul_flat = fitsio.FITS(flat_path, "r") if flat_path else None
        return hdul_input, hdul_flat
    except OSError as e:
        print(f"ERROR: Could not open input files: {e}")
        return None, None


def get_hdus_to_process(hdul_input, input_hdu: int) -> list:
    if input_hdu is not None:
        if 0 <= input_hdu < len(hdul_input):
            try:
                info = hdul_input[input_hdu].get_info()
                if info.get("hdutype") == 0 and info.get("ndims") == 2:
                    return [input_hdu]
                else:
                    print(
                        f"ERROR: Specified HDU {input_hdu} is not a 2D image "
                        f"(hdutype={info.get('hdutype')}, ndims={info.get('ndims')})."
                    )
                    return []
            except Exception as e:
                print(f"ERROR: Cannot inspect specified HDU {input_hdu}: {e}")
                return []
        else:
            print(f"ERROR: Specified HDU {input_hdu} not found.")
            return []
    else:
        hdus = []
        for idx, hdu in enumerate(hdul_input):
            try:
                info = hdu.get_info()
                if info.get("hdutype") == 0 and info.get("ndims") == 2:
                    hdus.append(idx)
            except Exception:
                continue
        if not hdus:
            print("ERROR: No suitable Image HDUs found.")
        return hdus


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
        _assign_map_if_valid(output_data, i, "sky", sky_map, hdu_header, hdu_name)
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


def _veto_dead_ccd_hdus(flat_bad_masks, flat_meds, flat_cfg):
    """Zero-weight HDUs whose flat level is an outlier across the exposure.

    A whole CCD blanked by Elixir (or failed hardware) shows up as a flat
    median tens of sigma from its siblings. Such an HDU's weights would be
    garbage; mark it fully BAD explicitly instead of letting a partial mask
    through. No-op for small-HDU runs and when all medians agree. Mutates
    flat_bad_masks in place.
    """
    if not flat_cfg.get("dead_ccd_enable", True):
        return
    idxs = [i for i in flat_bad_masks if i in flat_meds]
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
            flat_bad_masks[i] = np.ones_like(flat_bad_masks[i], dtype=bool)
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
        for i in hdus_to_process:
            if i < len(hdul_flat):
                try:
                    flat_data = np.ascontiguousarray(hdul_flat[i].read().astype(np.float32))
                    print(f"    Pre-computing flat bad mask for HDU {i}...")
                    flat_meds[i] = _get_global_median(flat_data)
                    flat_bad_masks[i] = compute_flat_bad_mask(flat_data, flat_cfg, tile_size)
                except Exception as e:
                    print(f"    WARNING: Failed to pre-compute flat bad mask for HDU {i}: {e}")
        _veto_dead_ccd_hdus(flat_bad_masks, flat_meds, flat_cfg)
        print(f"  Pre-computed {len(flat_bad_masks)} flat bad masks")
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
        try:
            if in_path_u is not None:
                fi = None
                ff = None
                fb = None
                try:
                    fi = fitsio.FITS(in_path_u, "r")
                    if i >= len(fi):
                        print(f"Skipping HDU {i}: index out of range.")
                        return (i, (None, None, None, None, None, None), None, f"HDU{i}")
                    hdu_sci = fi[i]
                    if hdul_flat is not None:
                        if fl_path_u is not None:
                            ff = fitsio.FITS(fl_path_u, "r")
                            hdu_flat_obj = ff[i] if i < len(ff) else None
                        else:
                            try:
                                hdu_flat_obj = hdul_flat[i] if i < len(hdul_flat) else None
                            except Exception:
                                hdu_flat_obj = None
                    else:
                        hdu_flat_obj = None
                    if hdul_badpix is not None:
                        if bp_path_u is not None:
                            fb = fitsio.FITS(bp_path_u, "r")
                            hdu_badpix_obj = fb[i] if i < len(fb) else None
                        else:
                            try:
                                hdu_badpix_obj = hdul_badpix[i] if i < len(hdul_badpix) else None
                            except Exception:
                                hdu_badpix_obj = None
                    else:
                        hdu_badpix_obj = None
                    pre2 = _merge_dark(i, hdu_sci, pre)
                    result = process_hdu(hdu_sci, hdu_flat_obj, config, i, tile_size=tile_size, flat_path=flat_path, hdu_badpix=hdu_badpix_obj, precomputed_bad_mask=pre2)
                    try:
                        hdu_header_raw = fi[i].read_header()
                    except Exception:
                        hdu_header_raw = None
                    try:
                        tmp_n = fi[i]
                        hdu_name = getattr(tmp_n, "name", f"HDU{i}") if hasattr(tmp_n, "name") else f"HDU{i}"
                    except Exception:
                        hdu_name = f"HDU{i}"
                    return (i, result, hdu_header_raw, hdu_name)
                finally:
                    for _h in (fi, ff, fb):
                        try:
                            if _h is not None:
                                _h.close()
                        except Exception:
                            pass
            else:
                try:
                    hdu_sci = hdul_input[i]
                except Exception as e:
                    print(f"Skipping HDU {i}: cannot access input HDU: {e}")
                    return (i, (None, None, None, None, None, None), None, f"HDU{i}")
                try:
                    hdu_flat_obj = hdul_flat[i] if hdul_flat is not None and i < len(hdul_flat) else None
                except Exception:
                    hdu_flat_obj = None
                try:
                    hdu_badpix_obj = hdul_badpix[i] if hdul_badpix is not None and i < len(hdul_badpix) else None
                except Exception:
                    hdu_badpix_obj = None
                pre2 = pre
                if dark_cfg_global is not None and hdul_dark is not None:
                    try:
                        if i < len(hdul_dark):
                            try:
                                dark_data = np.ascontiguousarray(hdul_dark[i].read().astype(np.float32))
                                dark_hot = detect_bad_pixels(dark_data, dark_cfg_global, using_unit_flat=False)
                                sci_shape = hdu_sci.read().shape
                                if dark_hot.shape != sci_shape:
                                    print(f"    Skipping dark mask for HDU {i}: shape mismatch.")
                                else:
                                    if pre2 is not None and pre2.shape != sci_shape:
                                        pre2 = None
                                    pre2 = dark_hot if pre2 is None else (pre2 | dark_hot)
                                    print(f"    Dark frame adds {int(np.count_nonzero(dark_hot))} hot pixels to HDU {i}.")
                            except Exception as e:
                                print(f"    WARNING: dark mask failed for HDU {i}: {e}")
                    except Exception:
                        pass
                result = process_hdu(hdu_sci, hdu_flat_obj, config, i, tile_size=tile_size, flat_path=flat_path, hdu_badpix=hdu_badpix_obj, precomputed_bad_mask=pre2)
                try:
                    hdu_header_raw = hdul_input[i].read_header()
                except Exception:
                    hdu_header_raw = None
                try:
                    tmp = hdul_input[i]
                    hdu_name = getattr(tmp, "name", f"HDU{i}") if hasattr(tmp, "name") else f"HDU{i}"
                except Exception:
                    hdu_name = f"HDU{i}"
                return (i, result, hdu_header_raw, hdu_name)
        except Exception as e:
            import traceback
            print(f"FATAL ERROR processing HDU {i}: {e}\n{traceback.format_exc()}")
            return (i, (None, None, None, None, None, None), None, f"HDU{i}")
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
            hdu_output: dict = {}
            _store_output_maps(hdu_output, i, hdu_header, hdu_name, config, confidence_map, weight_map, mask_data, inv_var_data, sky_map, args, bad_mask, sat_mask, cr_mask, obj_mask, streak_mask, paths)
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
            writers[f"ind_{mask_type}"] = _StreamingMapWriter(out_path, hdul_input, primary_header, compress=compress, dtype=np.uint8)
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


def _cleanup_hdul(hdul_input, hdul_flat, hdul_badpix=None, hdul_dark=None):
    if hdul_input:
        hdul_input.close()
    if hdul_flat:
        hdul_flat.close()
    if hdul_badpix:
        hdul_badpix.close()
    if hdul_dark:
        hdul_dark.close()


def run_pipeline() -> int:
    """Main function to parse arguments and run the pipeline."""
    args = parse_arguments()

    print("Starting WeightMask Pipeline...")
    start_pipeline_time = time.time()

    if not validate_input_files(args):
        return 1

    config = load_configuration(args.config)
    if config is None:
        return 1

    input_path, input_hdu = extract_hdu_spec(args.input_file)
    flat_path, flat_hdu = extract_hdu_spec(args.flat_image) if args.flat_image else (None, None)
    badpix_path, _ = extract_hdu_spec(args.badpix_mask) if args.badpix_mask else (None, None)
    if args.hdu is not None:
        input_hdu = args.hdu

    paths = determine_output_paths(args, input_path, config)

    hdul_input, hdul_flat = open_fits_files(input_path, flat_path)
    if hdul_input is None:
        return 1

    hdus_to_process = get_hdus_to_process(hdul_input, input_hdu)
    if not hdus_to_process:
        _cleanup_hdul(hdul_input, hdul_flat)
        return 1
    print(f"Processing {len(hdus_to_process)} Image HDU(s): {hdus_to_process}")

    dark_path, _ = extract_hdu_spec(args.dark_image) if getattr(args, "dark_image", None) else (None, None)
    hdul_dark = None
    if dark_path:
        try:
            hdul_dark = fitsio.FITS(dark_path, "r")
        except OSError as e:
            print(f"ERROR: Could not open dark frame {dark_path}: {e}")
            _cleanup_hdul(hdul_input, hdul_flat)
            return 1

    hdul_badpix = None
    if badpix_path:
        try:
            hdul_badpix = fitsio.FITS(badpix_path, "r")
        except OSError as e:
            print(f"ERROR: Could not open badpix mask {badpix_path}: {e}")
            _cleanup_hdul(hdul_input, hdul_flat, None, hdul_dark)
            return 1

    process_success_count = process_all_hdus(
        hdus_to_process, hdul_input, hdul_flat, config, paths, args, flat_path=flat_path, hdul_badpix=hdul_badpix,
        hdul_dark=hdul_dark, max_workers=getattr(args, "max_workers", None),
        input_path=input_path, badpix_path=badpix_path, dark_path=dark_path,
    )

    _cleanup_hdul(hdul_input, hdul_flat, hdul_badpix, hdul_dark)

    import warnings

    warnings.filterwarnings("default", category=UserWarning)
    warnings.filterwarnings("default", category=RuntimeWarning)

    if process_success_count == 0:
        print("\nNo HDUs processed successfully. No output files written.")
        return 1

    print(f"\nPipeline finished in {time.time() - start_pipeline_time:.2f} seconds.")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(run_pipeline())
