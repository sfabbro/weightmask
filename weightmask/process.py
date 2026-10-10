"""Science pipeline: config validation and per-image mask/weight generation."""

import time as _time
from contextlib import contextmanager
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

from . import MASK_BITS, MASK_DTYPE
from .background import estimate_background
from .bad import compute_flat_bad_mask, detect_non_illuminated
from .config import CONFIG_SCHEMA, configuration_errors
from .cosmics import detect_cosmic_rays
from .objects import detect_objects
from .satur import detect_saturated_pixels, grow_bleed_trails
from .streaks import _resolve_streak_mode, detect_streaks
from .variance import _OBSOLETE_VARIANCE_KEYS, amplifier_gain_map, calculate_inverse_variance
from .weight import generate_weight_and_confidence


@contextmanager
def _timed(store: dict, key: str):
    """Accumulate wall-clock seconds for one pipeline stage (stdlib only)."""
    t0 = _time.perf_counter()
    try:
        yield
    finally:
        store[key] = float(store.get(key, 0.0)) + (_time.perf_counter() - t0)


def _format_sky_output(sky_map, output_params, mesh_box):
    sky_cards = {}
    if str((output_params or {}).get("sky_format", "full")).lower() != "mesh":
        return sky_map, sky_cards
    if mesh_box is None:
        print("    WARNING: sky_format=mesh but no SEP box available; writing full sky map.")
        return sky_map, sky_cards
    from .background import sky_to_mesh

    sky_out, sky_cards = sky_to_mesh(sky_map, mesh_box)
    print(f"    Sky mesh product: {sky_out.shape} (box={mesh_box})")
    return sky_out, sky_cards


def validate_config(config: dict) -> bool:
    """Validate configuration parameters through the canonical nested schema."""
    if not isinstance(config, dict):
        print("ERROR: Configuration must be a dictionary.")
        return False
    errors = configuration_errors(config)
    for section in CONFIG_SCHEMA:
        if section != "dark_masking" and section not in config:
            print(f"WARNING: Configuration section '{section}' missing; defaults will be used.")
    for error in errors:
        print(f"ERROR: {error}")
    return not errors


def _first_present_keyword(header, key_cfg):
    """Keyword name that ``_header_lookup`` would read, or None."""
    if header is None:
        return None
    keys = key_cfg if isinstance(key_cfg, (list, tuple)) else [key_cfg]
    get = getattr(header, "get", None)
    for k in keys:
        if not isinstance(k, str) or not k:
            continue
        try:
            v = get(k, None) if callable(get) else None
        except (TypeError, AttributeError, KeyError):
            v = None
        if v is None:
            try:
                if k in header:
                    v = header[k]
            except (TypeError, AttributeError, KeyError):
                v = None
        if v is not None:
            return k
    return None


def _detection_background_views(sky_cal, rms_cal, candidate_mask, mesh_box):
    """Build detector-only sky/RMS views from calibrated maps."""
    sky_det = np.array(sky_cal, copy=True)
    rms_det = np.array(rms_cal, copy=True)
    support = np.asarray(candidate_mask, dtype=bool)
    if not np.any(support):
        return sky_det, rms_det
    sigma = max(float(mesh_box), 1.0)
    outside = (~support) & np.isfinite(sky_cal)
    weights = gaussian_filter(outside.astype(np.float32), sigma=sigma, mode="nearest")
    values = gaussian_filter(np.where(outside, sky_cal, 0.0).astype(np.float32), sigma=sigma, mode="nearest")
    valid_weights = weights > 1e-6
    interpolated = np.divide(values, weights, out=np.array(sky_cal, copy=True), where=valid_weights)
    sky_det[support] = interpolated[support]
    invalid_rms = support & (~np.isfinite(rms_cal) | (rms_cal <= 0))
    if np.any(invalid_rms):
        rms_outside = outside & np.isfinite(rms_cal) & (rms_cal > 0)
        rms_weights = gaussian_filter(rms_outside.astype(np.float32), sigma=sigma, mode="nearest")
        rms_values = gaussian_filter(
            np.where(rms_outside, rms_cal, 0.0).astype(np.float32), sigma=sigma, mode="nearest"
        )
        rms_interpolated = np.divide(
            rms_values, rms_weights, out=np.array(rms_cal, copy=True), where=rms_weights > 1e-6
        )
        rms_det[invalid_rms] = rms_interpolated[invalid_rms]
    return sky_det, rms_det


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
        except (TypeError, AttributeError, KeyError):
            v = None
        if v is None:
            try:
                if k in header:
                    v = header[k]
            except (TypeError, AttributeError, KeyError):
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
    return min(max(16, t), max(16, min_dim // 2), min_dim)


def process_image(
    sci_data_full: np.ndarray,
    sci_hdr: dict,
    flat_data_full: Optional[np.ndarray],
    config: dict,
    tile_size: int = 1024,
    bad_mask: Optional[np.ndarray] = None,
    badpix_mask: Optional[np.ndarray] = None,
    detector_prior: Optional[np.ndarray] = None,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[dict],
]:
    """Processes a single Science image to generate all mask and map products."""
    config_errors = configuration_errors(config)
    if config_errors:
        raise ValueError("Invalid configuration: " + " ".join(config_errors))
    hdu_start_time = _time.time()
    variance_cfg = dict(config.get("variance", {}))
    obsolete = sorted(_OBSOLETE_VARIANCE_KEYS & variance_cfg.keys())
    if obsolete:
        raise ValueError("Unsupported variance keys: " + ", ".join(obsolete))
    streak_cfg = dict(config.get("streak_masking", {}))
    _resolve_streak_mode(streak_cfg)
    sci_data_full = np.asarray(sci_data_full)
    if sci_data_full.ndim != 2 or sci_data_full.size == 0:
        raise ValueError("science data must be a nonempty 2-D image")
    saturation_data = sci_data_full
    sci_data_full = np.ascontiguousarray(sci_data_full, dtype=np.float32)
    sci_shape = sci_data_full.shape
    detector_prior_array = None
    if detector_prior is not None:
        detector_prior_array = np.asarray(detector_prior, dtype=bool)
        if detector_prior_array.shape != sci_shape:
            raise ValueError("detector prior shape must match science data")
    using_unit_flat = True
    eff_tile = _effective_tile_size(tile_size, sci_shape)
    if flat_data_full is not None:
        using_unit_flat = False
        if flat_data_full.shape != sci_shape:
            print("Skipping processing: Flat data shape mismatch.")
            return None, None, None, None, None, None
        flat_data_full = np.ascontiguousarray(flat_data_full, dtype=np.float32)
    else:
        print("  INFO: No flat field provided, assuming flat = 1.0.")
        flat_data_full = np.ones_like(sci_data_full, dtype=np.float32)

    if bad_mask is not None:
        bad_mask = np.asarray(bad_mask, dtype=bool)
        if bad_mask.shape != sci_shape:
            raise ValueError("bad mask shape must match science data")
    if badpix_mask is not None:
        badpix_mask = np.asarray(badpix_mask, dtype=bool)
        if badpix_mask.shape != sci_shape:
            raise ValueError("badpix mask shape must match science data")

    # --- 0. Initial Setup ---
    final_mask_int = np.zeros(sci_shape, dtype=MASK_DTYPE)
    header_info: dict = {}
    timings: dict = {}
    header_info["timings"] = timings

    # Initialize individual masks
    sat_mask = np.zeros(sci_shape, dtype=bool)
    cr_mask = np.zeros(sci_shape, dtype=bool)
    obj_mask = np.zeros(sci_shape, dtype=bool)
    streak_mask = np.zeros(sci_shape, dtype=bool)
    nodata_mask = np.zeros(sci_shape, dtype=bool)

    print("  (1/7) Processing Bad Pixel mask...")
    with _timed(timings, "bad_flat"):
        if bad_mask is None:
            bad_mask = np.zeros(sci_shape, dtype=bool)
            if not using_unit_flat:
                # Compute flat bad mask ONCE for the full HDU (replaces per-tile + cache)
                print("    Computing flat bad-pixel mask (full HDU)...")
                flat_bad_mask = compute_flat_bad_mask(flat_data_full, config.get("flat_masking", {}), eff_tile)
                bad_mask |= flat_bad_mask
            else:
                print("  Skipping flat bad-pixel detection (using unit flat).")
        if badpix_mask is not None:
            bad_mask = bad_mask | badpix_mask
        nodata_mask = ~np.isfinite(sci_data_full) | detect_non_illuminated(sci_shape, sci_hdr)
        n_nodata = int(np.count_nonzero(nodata_mask))
        if n_nodata:
            print(f"    Masking {n_nodata} non-finite or non-illuminated pixels NO_DATA.")
    final_mask_int[bad_mask] |= MASK_BITS["BAD"]
    final_mask_int[nodata_mask] |= MASK_BITS["NO_DATA"]

    print("  (1.1/7) Detecting saturation on the full image...")
    sat_cfg = dict(config.get("saturation", {}))
    with _timed(timings, "saturation"):
        saturation_level, sat_method_used, sat_mask = detect_saturated_pixels(saturation_data, sci_hdr, sat_cfg)
    del saturation_data
    final_mask_int[sat_mask] |= MASK_BITS["SAT"]
    header_info["SAT_LVL"], header_info["SAT_METH"] = saturation_level, sat_method_used

    # --- Full Image Processing Steps ---
    interim_mask_bool = final_mask_int > 0
    sep_bg_cfg = dict(config.get("sep_background", {}))
    # Calculate preliminary background RMS for CR and Bleed masking
    print("  Calculating preliminary background RMS...")
    with _timed(timings, "background_prelim"):
        prelim_bkg_map, prelim_bkg_rms = estimate_background(sci_data_full, interim_mask_bool, sep_bg_cfg)

    # --- 1.5 Bleed Trail (Blooming) Masking ---
    with _timed(timings, "bleed"):
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
    cosmic_cfg = dict(config.get("cosmic_ray", {}))
    gain_raw = _header_lookup(sci_hdr, variance_cfg.get("gain_keyword", "GAIN"), variance_cfg.get("default_gain", 1.0))
    rdnoise_raw = _header_lookup(
        sci_hdr,
        variance_cfg.get("rdnoise_keyword", "RDNOISE"),
        variance_cfg.get("default_rdnoise", 0.0),
    )
    try:
        gain = float(gain_raw)
        if not np.isfinite(gain) or gain <= 0:
            raise ValueError("gain must be finite and positive")
    except (ValueError, TypeError):
        gain = float(variance_cfg.get("default_gain", 1.0))
        print(f"  WARNING: GAIN header value {gain_raw!r} is invalid; using default {gain}.")
    try:
        read_noise_e = float(rdnoise_raw)
        if not np.isfinite(read_noise_e) or read_noise_e < 0:
            raise ValueError("read noise must be finite and non-negative")
    except (ValueError, TypeError):
        read_noise_e = float(variance_cfg.get("default_rdnoise", 0.0))
        print(f"  WARNING: RDNOISE header value {rdnoise_raw!r} is invalid; using default {read_noise_e}.")
    with _timed(timings, "cosmics"):
        cr_add_mask = detect_cosmic_rays(
            sci_data_full,
            interim_mask_bool,
            saturation_level,
            gain,
            read_noise_e,
            cosmic_cfg,
            bkg_rms_map=prelim_bkg_rms,
            sky_map=prelim_bkg_map,
            header=sci_hdr,
        )
    cr_mask |= cr_add_mask
    final_mask_int[cr_add_mask] |= MASK_BITS["CR"]
    interim_mask_bool |= cr_add_mask
    print(f"      Masked {np.count_nonzero(cr_add_mask)} new CR pixels.")

    # --- 3. Iterative Background and Object Detection ---
    print("  (3/7) Starting iterative Background/Object detection...")
    object_cfg = dict(config.get("sep_objects", {}))
    iterations = sep_bg_cfg.get("iterations", 2)
    baseline_object_mask = np.zeros(sci_shape, dtype=bool)
    elongated_fallback_mask = np.zeros(sci_shape, dtype=bool)
    elongated_candidate_mask = np.zeros(sci_shape, dtype=bool)
    handoff_enabled = bool(object_cfg.get("handoff_elongated_to_streak", True))
    bg_diag: dict = {}
    last_bg_mask = None
    last_bkg_map = None
    last_bkg_rms_map = None
    with _timed(timings, "bgobj_loop"):
        for i in range(iterations):
            print(f"    Iteration {i + 1}/{iterations}...")
            total_mask_for_bg = interim_mask_bool | baseline_object_mask
            with _timed(timings, f"bg_iter_{i}"):
                # Reusing the preliminary background here would be wrong: it was
                # estimated with the pre-bleed/pre-CR mask, and iteration 0 masks
                # out the bleed and cosmic-ray pixels found since.
                bkg_map, bkg_rms_map = estimate_background(
                    sci_data_full, total_mask_for_bg, {**sep_bg_cfg, "_diagnostics": bg_diag}
                )
            if bkg_map is None:
                return None, None, None, None, None, None
            data_sub = sci_data_full - bkg_map
            with _timed(timings, f"obj_iter_{i}"):
                new_obj_add_mask = detect_objects(data_sub, bkg_rms_map, total_mask_for_bg, object_cfg)
            last_bg_mask = total_mask_for_bg
            last_bkg_map, last_bkg_rms_map = bkg_map, bkg_rms_map
            fallback = object_cfg.pop("_elongated_fallback_mask", None)
            candidate = object_cfg.pop("_elongated_candidate_mask", None)
            if candidate is None:
                candidate = object_cfg.pop("_elongated_for_sky", None)
            else:
                object_cfg.pop("_elongated_for_sky", None)
            if handoff_enabled and isinstance(candidate, np.ndarray) and candidate.shape == sci_shape:
                elongated_candidate_mask |= candidate.astype(bool, copy=False)
            if isinstance(fallback, np.ndarray) and fallback.shape == sci_shape:
                if handoff_enabled:
                    elongated_fallback_mask |= fallback
            baseline_object_mask |= new_obj_add_mask
            if handoff_enabled and isinstance(fallback, np.ndarray) and fallback.shape == sci_shape:
                baseline_object_mask |= fallback
            if np.count_nonzero(new_obj_add_mask) == 0 and i > 0:
                print("      No new objects found, ending iteration.")
                break

    baseline_object_mask &= ~interim_mask_bool
    elongated_fallback_mask &= baseline_object_mask & ~interim_mask_bool
    elongated_candidate_mask &= elongated_fallback_mask
    ordinary_object_mask = baseline_object_mask & ~elongated_fallback_mask
    final_mask_int[baseline_object_mask] |= MASK_BITS["DETECTED"]
    print(f"      Masked {np.count_nonzero(baseline_object_mask)} DETECTED pixels total.")
    # --- 4. Final Sky Maps and Object Mask ---
    print("  (4/7) Finalizing sky maps and object mask...")
    final_full_mask = interim_mask_bool | baseline_object_mask
    with _timed(timings, "background_final"):
        if last_bg_mask is not None and np.array_equal(final_full_mask, last_bg_mask):
            print("  Reusing iteration background (mask unchanged)...")
            sky_map, final_bkg_rms_map = last_bkg_map, last_bkg_rms_map
        else:
            sky_map, final_bkg_rms_map = estimate_background(
                sci_data_full, final_full_mask, {**sep_bg_cfg, "_diagnostics": bg_diag}
            )
    if sky_map is None:
        return None, None, None, None, None, None

    # --- 5. Inverse Variance Map ---
    print("  (5/7) Calculating inverse variance map...")
    gain_map = amplifier_gain_map(sci_hdr, sci_shape, gain)
    if gain_map is not None:
        variance_cfg["gain"] = gain_map
        header_info["GAIN_SRC"] = "GAINA+GAINB"
    else:
        variance_cfg["gain"] = gain
        header_info["GAIN_SRC"] = _first_present_keyword(sci_hdr, variance_cfg.get("gain_keyword", "GAIN"))
    variance_cfg["read_noise"] = read_noise_e
    with _timed(timings, "variance"):
        inv_variance_map = calculate_inverse_variance(
            variance_cfg,
            sky_map,
            flat_data_full,
            final_bkg_rms_map,
            sci_data=sci_data_full,
            obj_mask=final_full_mask,
        )
    if inv_variance_map is None:
        return None, None, None, None, None, None
    # --- 6. Streak Detection ---
    print("  (6/7) Detecting streaks...")
    with _timed(timings, "streaks"):
        if streak_cfg.get("enable", False):
            sky_det = np.array(sky_map, copy=True)
            rms_det = np.array(final_bkg_rms_map, copy=True)
            effective_box = bg_diag.get("box_size")
            if handoff_enabled and effective_box is not None:
                sky_det, rms_det = _detection_background_views(
                    sky_map, final_bkg_rms_map, elongated_candidate_mask, effective_box
                )
            data_sub = sci_data_full - sky_det
            streak_exclude = interim_mask_bool | ordinary_object_mask
            if handoff_enabled:
                streak_exclude |= elongated_fallback_mask & ~elongated_candidate_mask
            if detector_prior_array is not None:
                streak_exclude = streak_exclude | detector_prior_array
            streak_add_mask = detect_streaks(data_sub, rms_det, streak_exclude, streak_cfg)
            if detector_prior_array is not None:
                streak_add_mask = streak_add_mask & ~detector_prior_array
            accepted = streak_add_mask & elongated_fallback_mask & ~ordinary_object_mask
            if np.any(accepted):
                detected_bit = np.asarray(
                    ~int(MASK_BITS["DETECTED"]) & np.iinfo(final_mask_int.dtype).max, dtype=final_mask_int.dtype
                )
                final_mask_int[accepted] &= detected_bit
            streak_mask |= streak_add_mask
            final_mask_int[streak_add_mask] |= MASK_BITS["STREAK"]
            final_full_mask |= streak_add_mask
            print(f"      Masked {np.count_nonzero(streak_add_mask)} new STREAK pixels.")

    accepted = streak_mask & elongated_fallback_mask & ~ordinary_object_mask
    final_detected = ordinary_object_mask | (elongated_fallback_mask & ~accepted)
    final_mask_int[final_detected] |= MASK_BITS["DETECTED"]
    obj_mask |= final_detected

    # --- 7. Generate Final Weight and Confidence Maps ---
    print("  (7/7) Generating final weight and confidence maps...")
    with _timed(timings, "weight_confidence"):
        weight_map, confidence_map, contract_product = generate_weight_and_confidence(
            inv_variance_map, final_mask_int, config
        )
    if weight_map is None:
        return None, None, None, None, None, None

    # Sky *output* product (internal sky_map above stays full-res for variance/streaks).
    sky_out = sky_map
    sky_cards: dict = {}
    with _timed(timings, "sky_mesh"):
        sky_out, sky_cards = _format_sky_output(sky_map, config.get("output_params", {}), bg_diag.get("box_size"))
    header_info["sky_cards"] = sky_cards
    header_info["contract_product"] = contract_product

    header_info["individual_masks"] = {
        "bad": bad_mask,
        "sat": sat_mask,
        "cr": cr_mask,
        "obj": obj_mask,
        "streak": streak_mask,
        "nodata": nodata_mask,
    }

    hdu_elapsed = _time.time() - hdu_start_time
    timings["hdu_total"] = float(hdu_elapsed)
    _top = sorted(
        (
            (k, v)
            for k, v in timings.items()
            if k != "hdu_total" and not k.startswith("bg_iter_") and not k.startswith("obj_iter_")
        ),
        key=lambda kv: kv[1],
        reverse=True,
    )[:3]
    _top_str = " ".join(f"{k}:{v:.1f}s" for k, v in _top)
    print(f"--- Image processed in {hdu_elapsed:.2f} seconds --- top={_top_str}")
    out_mask = contract_product.quality_mask if contract_product is not None else final_mask_int
    out_ivar = contract_product.inverse_variance if contract_product is not None else inv_variance_map
    return (
        out_mask,
        out_ivar,
        weight_map,
        confidence_map,
        sky_out,
        header_info,
    )
