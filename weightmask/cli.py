#!/usr/bin/env python3
"""
WeightMask CLI using fitsio instead of astropy.io.fits for better performance with MEF files.
"""

import argparse
import os
import sys
import time

import fitsio
import yaml

from . import __version__
from .mef import process_all_hdus
from .process import validate_config
from .utils import clean_config_dict, extract_hdu_spec


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


def parse_arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build weight, mask, inverse-variance, and sky maps for astronomical "
            "FITS/MEF images. A YAML config is required."
        ),
        epilog=(
            "Examples:\n"
            "  weightmask science.fits --config weightmask.yml --flat_image flat.fits "
            "-o out.weight.fits --output_mask out.mask.fits\n"
            "  weightmask science.fits --config weightmask.yml --output_sky sky.fits "
            "--output_invvar invvar.fits\n"
            "  weightmask-reconstruct-sky sky_mesh.fits -o sky_full.fits\n"
            "\n"
            "Config keys: weightmask.yml (copy into the working directory; not installed "
            "with the package). Usage: docs/usage.md\n"
            "Mesh skies: weightmask-reconstruct-sky (also: weightmask reconstruct-sky ...)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    inputs = parser.add_argument_group("Inputs")
    outputs = parser.add_argument_group("Outputs")
    run = parser.add_argument_group("Run")

    inputs.add_argument("input_file", type=str, help="Path to the science FITS/MEF.")
    inputs.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML config. If omitted, looks for weightmask.yml in the current "
        "directory (also config.yml, .weightmask.yml). Not bundled in the wheel.",
    )
    inputs.add_argument(
        "--flat_image",
        type=str,
        default=None,
        help="Flat-field FITS/MEF used for BAD pixels and the F² weight term.",
    )
    inputs.add_argument(
        "--dark_image",
        type=str,
        default=None,
        help="Dark frame FITS/MEF. Hot pixels from dark_masking are OR'd into BAD.",
    )
    inputs.add_argument(
        "--badpix_mask",
        type=str,
        default=None,
        help="External keep-map MEF (0 = bad, 1 = good). Zeros are OR'd into BAD.",
    )
    outputs.add_argument(
        "--output_map",
        "-o",
        type=str,
        default=None,
        help="Primary map (weight or confidence). Default: <input_base>.weight.fits",
    )
    outputs.add_argument(
        "--output_mask",
        type=str,
        default=None,
        help="Combined integer quality mask FITS.",
    )
    outputs.add_argument(
        "--output_invvar",
        type=str,
        default=None,
        help="Inverse-variance FITS (sanitized plane).",
    )
    outputs.add_argument(
        "--output_sky",
        type=str,
        default=None,
        help="Sky FITS (full map or SKYMESH, from output_params.sky_format).",
    )
    outputs.add_argument(
        "--output_weight_raw",
        type=str,
        default=None,
        help="Unnormalized masked inverse variance, if different from the primary map.",
    )
    run.add_argument(
        "--hdu",
        type=int,
        default=None,
        help="HDU index to process. Default: every 2-D image extension.",
    )
    run.add_argument(
        "--individual_masks",
        action="store_true",
        help="Also write per-component mask FITS files.",
    )
    run.add_argument(
        "--nproc",
        "--max-workers",
        dest="max_workers",
        type=int,
        default=None,
        help="Max parallel HDU workers (default min(8, ncpu); 0/1 = sequential).",
    )
    run.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    return parser.parse_args(argv)


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
    config["output_params"].setdefault("sky_format", "full")

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


def _cleanup_hdul(hdul_input, hdul_flat, hdul_badpix=None, hdul_dark=None):
    if hdul_input:
        hdul_input.close()
    if hdul_flat:
        hdul_flat.close()
    if hdul_badpix:
        hdul_badpix.close()
    if hdul_dark:
        hdul_dark.close()


def run_pipeline(argv=None) -> int:
    """Main function to parse arguments and run the pipeline."""
    argv = list(sys.argv[1:] if argv is None else argv)
    # Compat: `weightmask reconstruct-sky ...` delegates to the dedicated entry.
    if argv and argv[0] == "reconstruct-sky":
        from .reconstruct_sky import main as reconstruct_sky_main

        return reconstruct_sky_main(argv[1:])

    args = parse_arguments(argv)

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
    sys.exit(run_pipeline())
