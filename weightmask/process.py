"""Science pipeline: config validation and per-image mask/weight generation."""

import time as _time
from contextlib import contextmanager
from typing import Optional, Tuple

import numpy as np

from . import MASK_BITS, MASK_DTYPE
from .background import estimate_background
from .bad import compute_flat_bad_mask, detect_bad_pixels, detect_non_illuminated
from .cosmics import detect_cosmic_rays
from .objects import detect_objects
from .satur import detect_saturated_pixels, grow_bleed_trails
from .streaks import detect_streaks
from .variance import calculate_inverse_variance
from .weight import generate_weight_and_confidence


@contextmanager
def _timed(store: dict, key: str):
    """Accumulate wall-clock seconds for one pipeline stage (stdlib only)."""
    t0 = _time.perf_counter()
    try:
        yield
    finally:
        store[key] = float(store.get(key, 0.0)) + (_time.perf_counter() - t0)


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
        allowed_values = ["auto_ground"]
        if streak_method is not None and streak_method not in allowed_values:
            print(f"ERROR: Invalid streak masking method '{streak_method}'.")
            return False
        if streak_mode is not None and streak_mode not in allowed_values:
            print(f"ERROR: Invalid streak masking mode '{streak_mode}'.")
            return False
        legacy_keys = {"enable_ransac_trails", "ransac_params", "frangi_params", "frangi_legacy_params"}
        stale_keys = sorted(legacy_keys & set(config["streak_masking"]))
        if stale_keys:
            print(
                "ERROR: Legacy streak keys are no longer supported. "
                "Use 'enable_sparse_ransac' and 'sparse_ransac_params' "
                "(Frangi comparison: benchmarks.frangi_legacy): " + ", ".join(stale_keys)
            )
            return False
    for _sec, _key in (
        ("variance", "gain_keyword"),
        ("variance", "readnoise_keyword"),
        ("variance", "rdnoise_keyword"),
        ("saturation", "keyword"),
    ):
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
    if "sky_format" in _op and str(_op["sky_format"]).lower() not in ("full", "mesh"):
        print("ERROR: 'output_params.sky_format' must be 'full' or 'mesh'.")
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
    import time

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
                        if not np.isfinite(sci_data_tile).any():
                            continue
                        flat_mask_bool_tile = detect_bad_pixels(
                            flat_data_full[tile_slice], config.get("flat_masking", {}), using_unit_flat
                        )
                        bad_mask[tile_slice] |= flat_mask_bool_tile
        if badpix_mask is not None:
            bad_mask = bad_mask | badpix_mask
        nodata_mask = detect_non_illuminated(sci_shape, sci_hdr)
        n_nodata = int(np.count_nonzero(nodata_mask))
        if n_nodata:
            print(f"    Masking {n_nodata} non-illuminated (DATASEC-exterior) pixels NO_DATA.")
    final_mask_int[bad_mask] |= MASK_BITS["BAD"]
    final_mask_int[nodata_mask] |= MASK_BITS["NO_DATA"]

    print("  (1.1/7) Detecting saturation on the full image...")
    with _timed(timings, "saturation"):
        saturation_level, sat_method_used, sat_mask = detect_saturated_pixels(
            sci_data_full, sci_hdr, config.get("saturation", {})
        )
    final_mask_int[sat_mask] |= MASK_BITS["SAT"]
    header_info["SAT_LVL"], header_info["SAT_METH"] = saturation_level, sat_method_used

    # --- Full Image Processing Steps ---
    interim_mask_bool = final_mask_int > 0
    # Calculate preliminary background RMS for CR and Bleed masking
    print("  Calculating preliminary background RMS...")
    with _timed(timings, "background_prelim"):
        prelim_bkg_map, prelim_bkg_rms = estimate_background(
            sci_data_full, interim_mask_bool, config.get("sep_background", {})
        )

    # --- 1.5 Bleed Trail (Blooming) Masking ---
    sat_cfg = config.get("saturation", {})
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
    cosmic_cfg = config.get("cosmic_ray", {})
    variance_cfg = dict(config.get("variance", {}))
    gain_raw = _header_lookup(sci_hdr, variance_cfg.get("gain_keyword", "GAIN"), variance_cfg.get("default_gain", 1.0))
    rdnoise_raw = _header_lookup(
        sci_hdr,
        variance_cfg.get("rdnoise_keyword", variance_cfg.get("readnoise_keyword", "RDNOISE")),
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
    with _timed(timings, "cosmics"):
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
    bg_diag: dict = {}
    last_bg_mask = None
    last_bkg_map = None
    last_bkg_rms_map = None
    with _timed(timings, "bgobj_loop"):
        for i in range(iterations):
            print(f"    Iteration {i + 1}/{iterations}...")
            total_mask_for_bg = interim_mask_bool | current_obj_mask
            with _timed(timings, f"bg_iter_{i}"):
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
    variance_cfg["gain"] = gain
    variance_cfg["read_noise"] = read_noise_e
    with _timed(timings, "variance"):
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
    with _timed(timings, "streaks"):
        if streak_cfg.get("enable", False):
            data_sub = sci_data_full - sky_map
            streak_add_mask = detect_streaks(data_sub, final_bkg_rms_map, final_full_mask, streak_cfg)
            streak_mask |= streak_add_mask
            final_mask_int[streak_add_mask] |= MASK_BITS["STREAK"]
            final_full_mask |= streak_add_mask
            print(f"      Masked {np.count_nonzero(streak_add_mask)} new STREAK pixels.")

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
        if str(config.get("output_params", {}).get("sky_format", "full")).lower() == "mesh":
            mesh_box = bg_diag.get("box_size")
            if mesh_box is None:
                print("    WARNING: sky_format=mesh but no SEP box available; writing full sky map.")
            else:
                from .background import sky_to_mesh

                sky_out, sky_cards = sky_to_mesh(sky_map, mesh_box)
                print(f"    Sky mesh product: {sky_out.shape} (box={mesh_box})")
        else:
            timings["sky_mesh"] = float(timings.get("sky_mesh", 0.0))
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

    hdu_elapsed = time.time() - hdu_start_time
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
