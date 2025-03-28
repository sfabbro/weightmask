import os
import time
import warnings
import argparse
import yaml
from astropy.io import fits
import numpy as np

# Import from other modules within the package using relative imports
from .bad import detect_bad_pixels
from .satur import detect_saturated_pixels
from .cosmics import detect_cosmic_rays
from .streaks import detect_streaks
from .background import estimate_background
from .variance import calculate_inverse_variance
from .weight import generate_weight_and_confidence
from .objects import detect_objects
from .utils import extract_hdu_spec, create_binary_mask
from . import MASK_BITS, MASK_DTYPE # Import from __init__.py


def process_hdu(hdu_sci, hdu_flat, config, hdu_index):
    """
    Processes a single Science HDU to generate mask, inv variance, weight, conf, sky maps.

    Args:
        hdu_sci (fits.ImageHDU): Science image HDU.
        hdu_flat (fits.ImageHDU or None): Flat field HDU, or None if using flat=1.
        config (dict): Dictionary containing configuration parameters.
        hdu_index (int): Index of the HDU in the original FITS file.

    Returns:
        tuple: (mask_data, inv_var_data, weight_map, confidence_map, sky_map, header_info)
               Returns (None, None, None, None, None, None) on failure for this HDU.
               header_info is a dict of key parameters used (for output headers).
    """
    hdu_start_time = time.time()
    print(f"\n--- Processing HDU {hdu_index} ({hdu_sci.name}) ---")

    # --- 1. Load Data ---
    if hdu_sci.data is None or not isinstance(hdu_sci, (fits.ImageHDU, fits.CompImageHDU)):
         print("Skipping HDU: Science data missing or not an image HDU.")
         return None, None, None, None, None, None
    sci_shape = hdu_sci.data.shape
    if hdu_flat is not None and (hdu_flat.data is None or hdu_flat.data.shape != sci_shape):
         print("Skipping HDU: Flat data missing or shape mismatch.")
         return None, None, None, None, None, None
    sci_data = np.ascontiguousarray(hdu_sci.data.astype(np.float32))
    sci_hdr = hdu_sci.header
    if hdu_flat is not None:
        flat_data = np.ascontiguousarray(hdu_flat.data.astype(np.float32))
        using_unit_flat = False
    else:
        print("  INFO: No flat field provided, assuming flat = 1.0.")
        flat_data = np.ones_like(sci_data, dtype=np.float32)
        using_unit_flat = True

    final_mask_int = np.zeros(sci_data.shape, dtype=MASK_DTYPE)
    header_info = {}

    # --- 2. Get Noise Parameters ---
    variance_cfg_safe = config.get('variance', {}) # Ensure variance section exists
    gain = sci_hdr.get(variance_cfg_safe.get('gain_keyword','GAIN'), variance_cfg_safe.get('default_gain', 1.0))
    read_noise_e = sci_hdr.get(variance_cfg_safe.get('rdnoise_keyword','RDNOISE'), variance_cfg_safe.get('default_rdnoise', 0.0))
    if variance_cfg_safe.get('gain_keyword','GAIN') not in sci_hdr: print(f"  WARNING: Keyword '{variance_cfg_safe.get('gain_keyword','GAIN')}' not found. Using default: {gain:.2f} e-/ADU.")
    if variance_cfg_safe.get('rdnoise_keyword','RDNOISE') not in sci_hdr: print(f"  WARNING: Keyword '{variance_cfg_safe.get('rdnoise_keyword','RDNOISE')}' not found. Using default: {read_noise_e:.2f} e-.")
    header_info['GAIN_USD'] = gain
    header_info['RDNS_USD'] = read_noise_e

    # --- 3-7. Masking Steps (Bad, Sat, CR, Obj, Streak) ---
    print("Creating initial masks (Bad Flat, Saturation)...")
    flat_mask_bool = detect_bad_pixels(flat_data, config.get('flat_masking',{}), using_unit_flat)
    final_mask_int[flat_mask_bool] |= MASK_BITS['BAD']
    saturation_level, sat_method_used, sat_mask_bool = detect_saturated_pixels(sci_data, sci_hdr, config.get('saturation',{}))
    final_mask_int[sat_mask_bool] |= MASK_BITS['SAT']
    print(f"  Masked {np.sum(flat_mask_bool)} BAD pixels, {np.sum(sat_mask_bool)} SAT pixels.")
    header_info['SAT_LVL'] = saturation_level
    header_info['SAT_METH'] = sat_method_used
    interim_mask_1_bool = flat_mask_bool | sat_mask_bool

    print("Estimating initial background/RMS...")
    sep_bg_cfg_safe = config.get('sep_background', {})
    bkg_map1, bkg_rms_map1 = estimate_background(sci_data, interim_mask_1_bool, sep_bg_cfg_safe)
    if bkg_map1 is None or bkg_rms_map1 is None: # Add check
         print("  ERROR: Initial background estimation failed.")
         return None, None, None, None, None, None
    initial_bkg_rms_map = bkg_rms_map1

    print(f"Detecting Cosmic Rays...")
    cosmic_cfg_safe = config.get('cosmic_ray', {})
    header_info['CR_SIG'] = cosmic_cfg_safe.get('sigclip', 4.5)
    cr_add_mask = detect_cosmic_rays(sci_data, interim_mask_1_bool, saturation_level,
                                    gain, read_noise_e, cosmic_cfg_safe)
    final_mask_int[cr_add_mask] |= MASK_BITS['CR']
    print(f"  Masked {np.sum(cr_add_mask)} new CR pixels.")
    interim_mask_2_bool = interim_mask_1_bool | cr_add_mask

    print(f"Detecting Objects...")
    object_cfg_safe = config.get('sep_objects', {})
    header_info['SEP_THR'] = object_cfg_safe.get('extract_thresh', 1.5)
    data_sub = sci_data - bkg_map1
    obj_add_mask = detect_objects(data_sub, initial_bkg_rms_map, interim_mask_2_bool, object_cfg_safe)
    final_mask_int[obj_add_mask] |= MASK_BITS['DETECTED']
    print(f"  Masked {np.sum(obj_add_mask)} new DETECTED pixels.")
    interim_mask_3_bool = interim_mask_2_bool | obj_add_mask

    streak_cfg = config.get('streak_masking', {})
    header_info['STRK_EN'] = streak_cfg.get('enable', False)
    if streak_cfg.get('enable', False):
        print(f"Detecting Streaks using method: {streak_cfg.get('method', 'ransac')}...")
        streak_start_time = time.time()
        streak_add_mask = detect_streaks(data_sub, initial_bkg_rms_map, interim_mask_3_bool, streak_cfg)
        final_mask_int[streak_add_mask] |= MASK_BITS['STREAK']
        print(f"  Masked {np.sum(streak_add_mask)} new STREAK pixels.")
        print(f"  Streak detection finished in {time.time() - streak_start_time:.2f} seconds.")
    else:
        print("Streak masking disabled.")


    # --- 8. Final Background Estimation ---
    print("Estimating final background and RMS map...")
    final_combined_bool_mask = (final_mask_int > 0)
    sky_map, final_bkg_rms_map = estimate_background(sci_data, final_combined_bool_mask, sep_bg_cfg_safe)
    if sky_map is None or final_bkg_rms_map is None:
         print("  ERROR: Final background estimation failed.")
         return None, None, None, None, None, None
    print(f"  Final background global mean: {np.mean(sky_map):.3f}")

    # --- 9. Inverse Variance Map ---
    print("Calculating inverse variance map...")
    variance_method = variance_cfg_safe.get('method', 'theoretical').lower()
    header_info['VAR_METH'] = variance_method
    inv_variance_map = calculate_inverse_variance(
        method=variance_method,
        sky_map=sky_map, flat_map=flat_data, gain=gain, read_noise_e=read_noise_e,
        bkg_rms_map=final_bkg_rms_map, epsilon=variance_cfg_safe.get('epsilon', 1e-9)
    )
    if inv_variance_map is None:
        print("  ERROR: Inverse variance calculation failed.")
        return None, None, None, None, None, None
    print(f"  Inverse variance map calculated (mean non-zero: {np.mean(inv_variance_map[inv_variance_map > 0]):.4g})")

    # --- 10. Generate Weight and Confidence Maps ---
    weight_map, confidence_map = generate_weight_and_confidence(
        inv_variance_map, final_mask_int, config # Pass main config
    )
    if weight_map is None or confidence_map is None:
         print("  ERROR: Weight/Confidence map generation failed.")
         return None, None, None, None, None, None

    # --- End Processing ---
    hdu_elapsed = time.time() - hdu_start_time
    print(f"--- HDU processed in {hdu_elapsed:.2f} seconds ---")
    return final_mask_int, inv_variance_map, weight_map, confidence_map, sky_map, header_info


# --- Main execution function called by entry point ---
def run_pipeline():
    """ Main function to parse arguments and run the pipeline. """
    parser = argparse.ArgumentParser(description="Generate Mask and Weight/Confidence Maps for FITS files.")
    parser.add_argument('input_file', type=str, help="Path to input FITS file.")
    parser.add_argument('--output_map', '-o', type=str, default=None,
                        help="Path for primary output map (Weight or Confidence). Default: <input_base>.weight.fits")
    parser.add_argument('--config', type=str, default=None,
                        help="Path to YAML configuration file (optional, attempts default locations).")
    parser.add_argument('--flat_image', type=str, default=None, help="Path to input flat field FITS file (optional).")
    parser.add_argument('--output_mask', type=str, default=None, help="Path for output bitmask FITS file (optional).")
    parser.add_argument('--output_invvar', type=str, default=None, help="Path for output inverse variance FITS file (optional).")
    parser.add_argument('--output_sky', type=str, default=None, help="Path for output sky background map file (optional).")
    parser.add_argument('--output_weight_raw', type=str, default=None, help="Path for unnormalized weight map (masked inv_var), if different from primary map.")
    parser.add_argument('--hdu', type=int, default=None, help="HDU index to process (e.g., 0, 1). Processes extensions if omitted.")
    parser.add_argument('--individual_masks', action='store_true', help="Output individual mask component files.")
    args = parser.parse_args()

    print("Starting WeightMask Pipeline...")
    start_pipeline_time = time.time()

    # --- Handle Configuration File ---
    config_path = args.config
    if config_path is None:
        # Try finding default config in current dir or package dir
        # Assumes cli.py is inside weightmask package
        script_dir = os.path.dirname(__file__)
        possible_paths = ['weightmask.yml', os.path.join(script_dir, '../..', 'weightmask.yml')] # Check cwd, then project root
        for p in possible_paths:
            if os.path.exists(p):
                config_path = p
                print(f"Using default config file found at: {config_path}")
                break
        if config_path is None:
             print("ERROR: Config file not specified (--config) and no default 'weightmask.yml' found in CWD or project root.")
             return 1 # Indicate error

    # --- Load Configuration ---
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        # Ensure default sub-dicts exist if missing entirely
        if 'output_params' not in config: config['output_params'] = {}
        if 'confidence_params' not in config: config['confidence_params'] = {}
        # Set default output format if not specified
        if 'output_map_format' not in config['output_params']:
            config['output_params']['output_map_format'] = 'weight'
    except Exception as e:
        print(f"ERROR: Failed to load or parse config file '{config_path}': {e}")
        return 1 # Indicate error

    # --- Parse Input/Flat Files and HDU ---
    input_path, input_hdu = extract_hdu_spec(args.input_file)
    flat_path, flat_hdu = (extract_hdu_spec(args.flat_image) if args.flat_image else (None, None))
    if args.hdu is not None:
        input_hdu = args.hdu
        flat_hdu = args.hdu

    # --- Determine Default Output Path if Needed ---
    out_map_path = args.output_map
    if out_map_path is None:
        base = os.path.basename(input_path)
        if base.lower().endswith(('.fits.gz', '.fits.fz', '.fit.gz', '.fit.fz')): base = os.path.splitext(os.path.splitext(base)[0])[0]
        elif base.lower().endswith(('.fits', '.fit')): base = os.path.splitext(base)[0]
        elif '.' in base: base = os.path.splitext(base)[0]
        output_dir = os.path.dirname(input_path) or '.'
        default_suffix = ".weight.fits" # Sensible default name
        out_map_path = os.path.join(output_dir, f"{base}{default_suffix}")
        print(f"Output map path not specified, using default: {out_map_path}")

    # --- Determine Output Format ---
    output_format_config = config.get('output_params',{}).get('output_map_format', 'weight')
    if args.output_map is None and output_format_config != 'weight':
         print(f"NOTE: Default output filename implies 'weight' format, but config specifies '{output_format_config}'. Saving as '{output_format_config}'.")


    # --- Define Other Output Paths ---
    output_dir = os.path.dirname(out_map_path) or '.' # Base other optional outputs on primary map dir
    base_out_map = os.path.basename(out_map_path)
    base_out = os.path.splitext(base_out_map)[0]
    if base_out.endswith('.map'): base_out = os.path.splitext(base_out)[0] # Remove .map too

    out_mask_path = args.output_mask or os.path.join(output_dir, f"{base_out}.mask.fits") if args.output_mask != "" else None
    out_invvar_path = args.output_invvar or os.path.join(output_dir, f"{base_out}.ivar.fits") if args.output_invvar != "" else None
    out_sky_path = args.output_sky or os.path.join(output_dir, f"{base_out}.sky.fits") if args.output_sky != "" else None
    out_weight_raw_path = args.output_weight_raw # Remains None if not specified

    # --- Open Input FITS ---
    try:
        hdul_input = fits.open(input_path, memmap=True)
        hdul_flat = fits.open(flat_path, memmap=True) if flat_path else None
    except FileNotFoundError as e: print(f"ERROR: Input file not found: {e}"); return 1
    except Exception as e: print(f"ERROR: Could not open input files: {e}"); return 1

    # --- Handle HDU Selection ---
    hdus_to_process = []
    hdu_name_suffix = "" # Suffix only relevant if saving multiple optional files *and* processing single HDU
    if input_hdu is not None:
        if 0 <= input_hdu < len(hdul_input):
            hdus_to_process = [input_hdu]
            # Get name only if needed for optional outputs and processing single HDU
            if args.individual_masks or out_weight_raw_path:
                hdu_name = hdul_input[input_hdu].name if hasattr(hdul_input[input_hdu], 'name') and hdul_input[input_hdu].name else f"HDU{input_hdu}"
                hdu_name_suffix = f"_{hdu_name}" if input_hdu > 0 or hdu_name != "" else ""
            print(f"Processing single HDU: {input_hdu}")
        else:
            print(f"ERROR: Specified HDU {input_hdu} not found."); hdul_input.close(); return 1
    else:
        hdus_to_process = [i for i, hdu in enumerate(hdul_input) if isinstance(hdu, (fits.ImageHDU, fits.CompImageHDU)) and i > 0]
        if not hdus_to_process and len(hdul_input)>0 and isinstance(hdul_input[0], (fits.ImageHDU, fits.CompImageHDU)):
             print("No extensions found, processing Primary HDU (0)."); hdus_to_process = [0]
        elif not hdus_to_process: print("ERROR: No suitable Image HDUs found."); hdul_input.close(); return 1
        print(f"Processing {len(hdus_to_process)} Image HDU(s): {hdus_to_process}")

    # --- Print final paths ---
    print(f"\nInput File: {input_path}{f'[{input_hdu}]' if input_hdu is not None else ''}")
    print(f"Flat Image: {flat_path}{f'[{flat_hdu}]' if flat_path and flat_hdu is not None else '' if flat_path else 'None (using 1.0)'}")
    print(f"Output Map:    {out_map_path} (Format: {output_format_config})")
    if out_mask_path: print(f"Output Mask:   {out_mask_path}")
    if out_invvar_path: print(f"Output InvVar: {out_invvar_path}")
    if out_sky_path: print(f"Output Sky:    {out_sky_path}")
    if out_weight_raw_path: print(f"Output Raw Wt: {out_weight_raw_path}")

    # --- Prepare Output HDU Lists ---
    hdul_map_out = fits.HDUList([hdul_input[0].copy()]) if out_map_path else None
    hdul_mask_out = fits.HDUList([hdul_input[0].copy()]) if out_mask_path else None
    hdul_invvar_out = fits.HDUList([hdul_input[0].copy()]) if out_invvar_path else None
    hdul_sky_out = fits.HDUList([hdul_input[0].copy()]) if out_sky_path else None
    hdul_weight_raw_out = fits.HDUList([hdul_input[0].copy()]) if out_weight_raw_path else None
    if args.individual_masks:
        hdul_bad_out, hdul_satur_out, hdul_cosmics_out, hdul_streaks_out = (fits.HDUList([hdul_input[0].copy()]) for _ in range(4))
        # Use base_out for individual mask names
        out_bad_path = os.path.join(output_dir, f"{base_out}_bad.fits")
        out_satur_path = os.path.join(output_dir, f"{base_out}_satur.fits")
        out_cosmics_path = os.path.join(output_dir, f"{base_out}_cosmic.fits")
        out_streaks_path = os.path.join(output_dir, f"{base_out}_streak.fits")
        print(f"Individual mask components base name: {base_out}")

    print(f"\nProcessing {len(hdus_to_process)} science extension(s)...")
    warnings.filterwarnings('ignore', category=UserWarning, append=True)
    warnings.filterwarnings('ignore', category=RuntimeWarning, append=True)

    # --- Process HDUs ---
    process_success_count = 0
    for i in hdus_to_process:
        try:
            hdu_sci = hdul_input[i]
            hdu_flat = hdul_flat[i] if hdul_flat and i < len(hdul_flat) else None
            mask_data, inv_var_data, weight_map, confidence_map, sky_map, header_info = process_hdu(hdu_sci, hdu_flat, config, i)
            if mask_data is None: print(f"Skipping HDU {i} due to processing errors."); continue
            process_success_count += 1

            # --- Create and append HDUs based on which files are requested ---
            output_format = config['output_params']['output_map_format'].lower()
            conf_cfg = config.get('confidence_params', {})

            # Primary Output Map HDU (Potentially scaled int16)
            if hdul_map_out is not None:
                primary_map_save_data, primary_map_comment, primary_map_bunit = (None, "", "")
                bscale, bzero = (1.0, 0.0) # Default for float/unscaled

                if output_format == 'weight':
                    primary_map_save_data = weight_map.astype(np.float32) # Ensure float32
                    primary_map_comment = 'Weight Map (masked inverse variance)'
                    primary_map_bunit = 'adu**-2' # Adjust if variance units change
                elif output_format == 'confidence':
                    primary_map_data_float = confidence_map # Should be float 0-1 or 0-100
                    primary_map_comment = 'Confidence Map (Normalized Weight Map, scaled int16)'
                    min_phys, max_phys = (0.0, 100.0 if conf_cfg.get('scale_to_100', False) else 1.0)
                    min_int, max_int = (-32768, 32767); int_range = max_int - min_int
                    phys_range = max_phys - min_phys
                    if abs(phys_range) > 1e-9 and int_range > 0:
                        bscale = phys_range / int_range
                        bzero = min_phys - bscale * min_int
                        int_data = np.round((primary_map_data_float - bzero) / (bscale + 1e-30))
                        primary_map_save_data = np.clip(int_data, min_int, max_int).astype(np.int16)
                    else: # Handle cases where range is zero or invalid
                        bscale = 1.0; bzero = min_phys
                        primary_map_save_data = np.full(primary_map_data_float.shape, min_int, dtype=np.int16) # Save as min int value

                    primary_map_comment += ' Scaled 0-100' if conf_cfg.get('scale_to_100', False) else ' Scaled 0-1'
                    primary_map_bunit = 'percent' if conf_cfg.get('scale_to_100', False) else ''
                else: # Default to weight
                     primary_map_save_data = weight_map.astype(np.float32)
                     primary_map_comment = 'Weight Map (masked inverse variance)'
                     primary_map_bunit = 'adu**-2'

                map_hdu = fits.ImageHDU(data=primary_map_save_data, header=hdu_sci.header.copy(), name=f'MAP_{hdu_sci.name}')
                map_hdu.header['COMMENT'] = primary_map_comment
                if primary_map_bunit: map_hdu.header['BUNIT'] = primary_map_bunit
                if output_format == 'confidence':
                    map_hdu.header['BSCALE'] = bscale; map_hdu.header['BZERO'] = bzero
                map_hdu.header['VARMETH'] = (header_info.get('VAR_METH'), 'Variance calculation method used'); map_hdu.header['ORIG_HDU'] = (i, 'Original HDU index')
                hdul_map_out.append(map_hdu)

            # Optional outputs (Mask, InvVar, Sky, Raw Weight)
            if hdul_mask_out is not None:
                mask_hdu = fits.ImageHDU(data=mask_data, header=hdu_sci.header.copy(), name=f'MASK_{hdu_sci.name}')
                mask_hdu.header['COMMENT'] = 'Combined Mask (Bitmask)'; hdul_mask_out.append(mask_hdu)
            if hdul_invvar_out is not None:
                invvar_hdu = fits.ImageHDU(data=inv_var_data.astype(np.float32), header=hdu_sci.header.copy(), name=f'INVVAR_{hdu_sci.name}')
                invvar_hdu.header['COMMENT'] = 'Pure Inverse Variance Map'; invvar_hdu.header['BUNIT'] = 'adu**-2'; hdul_invvar_out.append(invvar_hdu)
            if hdul_sky_out is not None:
                sky_hdu = fits.ImageHDU(data=sky_map.astype(np.float32), header=hdu_sci.header.copy(), name=f'SKY_{hdu_sci.name}')
                sky_hdu.header['COMMENT'] = 'Smoothed Sky Background Map'; sky_hdu.header['BUNIT'] = 'adu'; hdul_sky_out.append(sky_hdu)
            if hdul_weight_raw_out is not None:
                 weight_raw_hdu = fits.ImageHDU(data=weight_map.astype(np.float32), header=hdu_sci.header.copy(), name=f'WEIGHT_{hdu_sci.name}')
                 weight_raw_hdu.header['COMMENT'] = 'Weight Map (masked inv_var, unnormalized)'; weight_raw_hdu.header['BUNIT'] = 'adu**-2'; hdul_weight_raw_out.append(weight_raw_hdu)

            # Individual Masks
            if args.individual_masks:
                 # ... (creation and appending as before) ...
                 bad_mask = create_binary_mask(mask_data, MASK_BITS['BAD']); bad_hdu = fits.ImageHDU(data=bad_mask, name=f'BAD_{hdu_sci.name}'); hdul_bad_out.append(bad_hdu)
                 satur_mask = create_binary_mask(mask_data, MASK_BITS['SAT']); satur_hdu = fits.ImageHDU(data=satur_mask, name=f'SATUR_{hdu_sci.name}'); hdul_satur_out.append(satur_hdu)
                 cosmic_mask = create_binary_mask(mask_data, MASK_BITS['CR']); cosmic_hdu = fits.ImageHDU(data=cosmic_mask, name=f'COSMIC_{hdu_sci.name}'); hdul_cosmics_out.append(cosmic_hdu)
                 streak_mask = create_binary_mask(mask_data, MASK_BITS['STREAK']); streak_hdu = fits.ImageHDU(data=streak_mask, name=f'STREAK_{hdu_sci.name}'); hdul_streaks_out.append(streak_hdu)

        except Exception as e:
            import traceback
            print(f"FATAL ERROR processing HDU {i} ({hdu_sci.name if 'hdu_sci' in locals() else 'Unknown'}): {e}")
            print(traceback.format_exc())
            print(f"Skipping output for this HDU.")

    # --- Write Output Files ---
    # Check if any HDUs were successfully processed before writing
    if process_success_count == 0:
        print("\nNo HDUs processed successfully. No output files will be written.")
    else:
        if hdul_map_out:
            print(f"\nWriting primary map file to: {out_map_path}")
            try: hdul_map_out.writeto(out_map_path, overwrite=True, checksum=True); print(" Map file OK.")
            except Exception as e: print(f" ERROR writing map file: {e}")
        # ... (rest of file writing, checking if list is not None and len > 1) ...
        if hdul_mask_out and len(hdul_mask_out) > 1:
            print(f"Writing mask file to: {out_mask_path}")
            try: hdul_mask_out.writeto(out_mask_path, overwrite=True, checksum=True); print(" Mask file OK.")
            except Exception as e: print(f" ERROR writing mask file: {e}")
        if hdul_invvar_out and len(hdul_invvar_out) > 1:
            print(f"Writing inverse variance file to: {out_invvar_path}")
            try: hdul_invvar_out.writeto(out_invvar_path, overwrite=True, checksum=True); print(" InvVar file OK.")
            except Exception as e: print(f" ERROR writing inverse variance file: {e}")
        if hdul_sky_out and len(hdul_sky_out) > 1:
            print(f"Writing sky background file to: {out_sky_path}")
            try: hdul_sky_out.writeto(out_sky_path, overwrite=True, checksum=True); print(" Sky background file OK.")
            except Exception as e: print(f" ERROR writing sky background file: {e}")
        if hdul_weight_raw_out and len(hdul_weight_raw_out) > 1:
            print(f"Writing raw weight file to: {out_weight_raw_path}")
            try: hdul_weight_raw_out.writeto(out_weight_raw_path, overwrite=True, checksum=True); print(" Raw weight file OK.")
            except Exception as e: print(f" ERROR writing raw weight file: {e}")
        if args.individual_masks:
            print(f"Writing individual mask component files...")
            # Check len > 1 for each individual mask list before writing
            if hdul_bad_out and len(hdul_bad_out) > 1:
                try: hdul_bad_out.writeto(out_bad_path, overwrite=True); print("  Bad pixel mask OK.")
                except Exception as e: print(f"  ERROR writing bad pixel mask: {e}")
            # ... etc for satur, cosmics, streaks ...


    # --- Cleanup ---
    if hdul_input: hdul_input.close()
    if hdul_flat: hdul_flat.close()
    warnings.filterwarnings('default', category=UserWarning)
    warnings.filterwarnings('default', category=RuntimeWarning)
    pipeline_elapsed = time.time() - start_pipeline_time
    print(f"\nPipeline finished in {pipeline_elapsed:.2f} seconds.")
    return 0 # Indicate success


if __name__ == "__main__":
    # This allows running the script directly for debugging if needed
    # but the primary execution path is via the entry point calling run_pipeline()
    import sys
    sys.exit(run_pipeline())