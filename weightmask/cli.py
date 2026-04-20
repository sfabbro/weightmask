#!/usr/bin/env python3
"""
WeightMask CLI using fitsio instead of astropy.io.fits for better performance with MEF files.
"""

import argparse
import os
import time
from typing import Optional, Tuple

import fitsio
import numpy as np
import yaml

from . import MASK_BITS, MASK_DTYPE  # Import from __init__.py
from .background import estimate_background

# Import from other modules within the package using relative imports
from .bad import detect_bad_pixels
from .cosmics import detect_cosmic_rays
from .objects import detect_objects
from .satur import detect_saturated_pixels, grow_bleed_trails
from .streaks import detect_streaks
from .utils import clean_config_dict, extract_hdu_spec
from .variance import calculate_inverse_variance
from .weight import generate_weight_and_confidence


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
    for section in required_sections:
        if section not in config:
            print(f"WARNING: Required configuration section '{section}' missing.")

    if "variance" in config:
        if not isinstance(config["variance"], dict):
            print("ERROR: 'variance' section must be a dictionary.")
            return False
        var_method = config["variance"].get("method", "theoretical")
        if var_method not in ["theoretical", "rms_map", "empirical_fit"]:
            print(f"ERROR: Invalid variance method '{var_method}'.")
            return False
    return True


def process_image(
    sci_data_full: np.ndarray,
    sci_hdr: dict,
    flat_data_full: Optional[np.ndarray],
    config: dict,
    tile_size: int = 1024,
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
    bad_mask = np.zeros(sci_shape, dtype=bool)
    sat_mask = np.zeros(sci_shape, dtype=bool)
    cr_mask = np.zeros(sci_shape, dtype=bool)
    obj_mask = np.zeros(sci_shape, dtype=bool)
    streak_mask = np.zeros(sci_shape, dtype=bool)

    # --- 1. Tile-based Masking (Bad Pixels, Saturation) ---
    print("  (1/7) Processing tiles for Bad Pixel and Saturation masks...")
    saturation_level, sat_method_used = -1, "unknown"
    for y in range(0, sci_shape[0], tile_size):
        for x in range(0, sci_shape[1], tile_size):
            tile_slice = (slice(y, y + tile_size), slice(x, x + tile_size))
            sci_data_tile = sci_data_full[tile_slice]
            flat_data_tile = flat_data_full[tile_slice]

            if not np.isfinite(sci_data_tile).any():
                continue

            flat_mask_bool_tile = detect_bad_pixels(flat_data_tile, config.get("flat_masking", {}), using_unit_flat)
            bad_mask[tile_slice] |= flat_mask_bool_tile
            final_mask_int[tile_slice][flat_mask_bool_tile] |= MASK_BITS["BAD"]

            sat_level_tile, sat_meth_tile, sat_mask_bool_tile = detect_saturated_pixels(
                sci_data_tile, sci_hdr, config.get("saturation", {})
            )
            sat_mask[tile_slice] |= sat_mask_bool_tile
            final_mask_int[tile_slice][sat_mask_bool_tile] |= MASK_BITS["SAT"]

            if saturation_level < 0:  # Use first tile's result as representative
                saturation_level, sat_method_used = sat_level_tile, sat_meth_tile
                header_info["SAT_LVL"], header_info["SAT_METH"] = (
                    saturation_level,
                    sat_method_used,
                )

    # --- Full Image Processing Steps ---
    interim_mask_bool = final_mask_int > 0

    # Calculate preliminary background RMS for CR and Bleed masking
    print("  Calculating preliminary background RMS...")
    _, prelim_bkg_rms = estimate_background(sci_data_full, interim_mask_bool, config.get("sep_background", {}))

    # --- 1.5 Bleed Trail (Blooming) Masking ---
    sat_cfg = config.get("saturation", {})
    if sat_cfg.get("mask_bleed_trails", True):
        print("  (1.5/7) Growing Bleed Trails for saturated stars...")
        sat_mask_full = grow_bleed_trails(sci_data_full, sat_mask, prelim_bkg_rms * 0, prelim_bkg_rms, sat_cfg)
        new_bleed_pixels = sat_mask_full & ~sat_mask
        sat_mask |= new_bleed_pixels
        final_mask_int[new_bleed_pixels] |= MASK_BITS["SAT"]
        interim_mask_bool |= new_bleed_pixels
        print(f"      Added {np.sum(new_bleed_pixels)} bleed trail pixels.")

    # --- 2. First-Pass Cosmic Ray Detection ---
    print("  (2/7) Running first-pass Cosmic Ray detection...")
    cosmic_cfg = config.get("cosmic_ray", {})
    variance_cfg = config.get("variance", {})
    gain = sci_hdr.get(variance_cfg.get("gain_keyword", "GAIN"), variance_cfg.get("default_gain", 1.0))
    read_noise_e = sci_hdr.get(
        variance_cfg.get("rdnoise_keyword", "RDNOISE"),
        variance_cfg.get("default_rdnoise", 0.0),
    )

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
    print(f"      Masked {np.sum(cr_add_mask)} new CR pixels.")

    # --- 3. Iterative Background and Object Detection ---
    print("  (3/7) Starting iterative Background/Object detection...")
    sep_bg_cfg = config.get("sep_background", {})
    object_cfg = config.get("sep_objects", {})
    iterations = sep_bg_cfg.get("iterations", 2)
    current_obj_mask = np.zeros(sci_shape, dtype=bool)

    for i in range(iterations):
        print(f"    Iteration {i + 1}/{iterations}...")
        total_mask_for_bg = interim_mask_bool | current_obj_mask
        bkg_map, bkg_rms_map = estimate_background(sci_data_full, total_mask_for_bg, sep_bg_cfg)
        if bkg_map is None:
            return None, None, None, None, None, None

        data_sub = sci_data_full - bkg_map
        new_obj_add_mask = detect_objects(data_sub, bkg_rms_map, total_mask_for_bg, object_cfg)

        if np.sum(new_obj_add_mask) == 0 and i > 0:
            print("      No new objects found, ending iteration.")
            break
        current_obj_mask |= new_obj_add_mask

    obj_mask |= current_obj_mask
    final_mask_int[current_obj_mask] |= MASK_BITS["DETECTED"]
    print(f"      Masked {np.sum(current_obj_mask)} DETECTED pixels total.")

    # --- 4. Final Sky Maps and Object Mask ---
    print("  (4/7) Finalizing sky maps and object mask...")
    final_obj_mask = current_obj_mask
    final_full_mask = interim_mask_bool | final_obj_mask
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
        print(f"      Masked {np.sum(streak_add_mask)} new STREAK pixels.")

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
    hdu_sci, hdu_flat, config: dict, hdu_index: int, tile_size: int = 1024
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

    return process_image(sci_data_full, sci_hdr, flat_data_full, config, tile_size)


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

    if not validate_config(config):
        print("ERROR: Configuration validation failed.")
        return None

    return config


def determine_output_paths(args: argparse.Namespace, input_path: str) -> dict:
    out_map_path = args.output_map
    if out_map_path is None:
        input_basename = os.path.basename(str(input_path))
        if input_basename.endswith(".fits.fz"):
            base = input_basename[:-8]
        elif input_basename.endswith(".fits"):
            base = input_basename[:-5]
        else:
            base = os.path.splitext(input_basename)[0]
        output_dir = os.path.dirname(str(input_path)) or "."
        default_suffix = ".weight.fits"
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
            return [input_hdu]
        else:
            print(f"ERROR: Specified HDU {input_hdu} not found.")
            return []
    else:
        hdus = list(range(len(hdul_input)))
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
        _assign_map_if_valid(output_data, i, "map", map_data, hdu_header, hdu_name)
    if paths["out_mask_path"]:
        _assign_map_if_valid(output_data, i, "mask", mask_data, hdu_header, hdu_name)
    if paths["out_invvar_path"]:
        _assign_map_if_valid(output_data, i, "invvar", inv_var_data, hdu_header, hdu_name)
    if paths["out_sky_path"]:
        _assign_map_if_valid(output_data, i, "sky", sky_map, hdu_header, hdu_name)
    if paths["out_weight_raw_path"] and weight_map is not None:
        if i not in output_data:
            output_data[i] = {}
        output_data[i]["weight_raw"] = {
            "data": weight_map,
            "header": hdu_header,
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


def process_all_hdus(
    hdus_to_process: list,
    hdul_input,
    hdul_flat,
    config: dict,
    paths: dict,
    args: argparse.Namespace,
) -> tuple:
    process_success_count = 0
    output_data = {}

    for i in hdus_to_process:
        try:
            hdu_sci = hdul_input[i]
            hdu_flat_obj = hdul_flat[i] if hdul_flat and i < len(hdul_flat) else None

            result = process_hdu(hdu_sci, hdu_flat_obj, config, i)
            if result[0] is None:
                print(f"Skipping HDU {i} due to processing errors.")
                continue
            (
                mask_data,
                inv_var_data,
                weight_map,
                confidence_map,
                sky_map,
                header_info,
            ) = result
            process_success_count += 1

            bad_mask, sat_mask, cr_mask, obj_mask, streak_mask = extract_individual_masks(header_info, mask_data)

            hdu_name = getattr(hdu_sci, "name", f"HDU{i}") if hasattr(hdu_sci, "name") else f"HDU{i}"

            try:
                hdu_header = hdul_input[i].read_header()
            except Exception:
                hdu_header = None
            if hdu_header is None:
                hdu_header = fitsio.FITSHDR()

            _store_output_maps(
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
            )

        except Exception as e:
            import traceback

            print(f"FATAL ERROR processing HDU {i}: {e}\n{traceback.format_exc()}")

    return process_success_count, output_data


def write_single_output_file(out_path: str, output_data: dict, hdul_input, key: str):
    primary_data = hdul_input[0].read() if len(hdul_input) > 0 else None
    fitsio.write(out_path, primary_data, clobber=True)
    with fitsio.FITS(out_path, "rw") as f_out:
        for i in sorted(output_data.keys()):
            if key in output_data[i]:
                f_out.write(
                    output_data[i][key]["data"],
                    header=output_data[i][key]["header"],
                    extname=output_data[i][key]["name"],
                )


def write_individual_mask_files(individual_mask_paths: dict, output_data: dict, hdul_input):
    for mask_type in ["bad", "sat", "cr", "obj", "streak"]:
        mask_path = individual_mask_paths.get(mask_type)
        if mask_path:
            primary_data = hdul_input[0].read() if len(hdul_input) > 0 else None
            fitsio.write(mask_path, primary_data, clobber=True)
            with fitsio.FITS(mask_path, "rw") as f_out:
                for i in sorted(output_data.keys()):
                    if "individual_masks" in output_data[i] and mask_type in output_data[i]["individual_masks"]:
                        mask_info = output_data[i]["individual_masks"][mask_type]
                        f_out.write(
                            mask_info["data"],
                            header=mask_info["header"],
                            extname=mask_info["name"],
                        )
            print(f"  {mask_type.capitalize()} mask file written: {mask_path}")


def write_all_output_files(paths: dict, output_data: dict, hdul_input, process_success_count: int) -> bool:
    if process_success_count == 0:
        print("\nNo HDUs processed successfully. No output files written.")
        return True

    print("\nWriting output files...")
    try:
        if paths["out_map_path"] and output_data:
            write_single_output_file(paths["out_map_path"], output_data, hdul_input, "map")
            print(f"  Map file written: {paths['out_map_path']}")

        if paths["out_mask_path"] and output_data:
            write_single_output_file(paths["out_mask_path"], output_data, hdul_input, "mask")
            print(f"  Mask file written: {paths['out_mask_path']}")

        if paths["out_invvar_path"] and output_data:
            write_single_output_file(paths["out_invvar_path"], output_data, hdul_input, "invvar")
            print(f"  Inverse variance file written: {paths['out_invvar_path']}")

        if paths["out_sky_path"] and output_data:
            write_single_output_file(paths["out_sky_path"], output_data, hdul_input, "sky")
            print(f"  Sky file written: {paths['out_sky_path']}")

        if paths["out_weight_raw_path"] and output_data:
            write_single_output_file(paths["out_weight_raw_path"], output_data, hdul_input, "weight_raw")
            print(f"  Raw weight file written: {paths['out_weight_raw_path']}")

        if paths["individual_mask_paths"] and output_data:
            write_individual_mask_files(paths["individual_mask_paths"], output_data, hdul_input)

        return True
    except OSError as e:
        print(f"ERROR: Failed to write output files: {e}")
        import traceback

        traceback.print_exc()
        return False


def _cleanup_hdul(hdul_input, hdul_flat):
    if hdul_input:
        hdul_input.close()
    if hdul_flat:
        hdul_flat.close()


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
    if args.hdu is not None:
        input_hdu = args.hdu

    paths = determine_output_paths(args, input_path)

    hdul_input, hdul_flat = open_fits_files(input_path, flat_path)
    if hdul_input is None:
        return 1

    hdus_to_process = get_hdus_to_process(hdul_input, input_hdu)
    if not hdus_to_process:
        _cleanup_hdul(hdul_input, hdul_flat)
        return 1
    print(f"Processing {len(hdus_to_process)} Image HDU(s): {hdus_to_process}")

    process_success_count, output_data = process_all_hdus(hdus_to_process, hdul_input, hdul_flat, config, paths, args)

    success = write_all_output_files(paths, output_data, hdul_input, process_success_count)

    _cleanup_hdul(hdul_input, hdul_flat)

    import warnings

    warnings.filterwarnings("default", category=UserWarning)
    warnings.filterwarnings("default", category=RuntimeWarning)

    if not success:
        return 1

    print(f"\nPipeline finished in {time.time() - start_pipeline_time:.2f} seconds.")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(run_pipeline())
