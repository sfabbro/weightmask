import os
import time
import warnings
import argparse
import yaml
from astropy.io import fits
import numpy as np
import sep
from astroscrappy import detect_cosmics
from skimage.transform import hough_line, hough_line_peaks
from skimage.draw import line_aa
from skimage.morphology import disk, binary_dilation
import re

# --- Mask Bit Definitions ---
# Defined here for use in the code, mirroring the YAML comments
MASK_BITS = {
    'BAD':      1 << 0,  # 1
    'SAT':      1 << 1,  # 2
    'CR':       1 << 2,  # 4
    'DETECTED': 1 << 3,  # 8
    'STREAK':   1 << 4,  # 16
}
MASK_DTYPE = np.int16 # Data type for the mask image

# --- Helper Functions ---

def extract_hdu_spec(filepath):
    """
    Extract HDU specifier from CFITSIO-style filename (e.g., 'file.fits[1]')
    
    Args:
        filepath (str): Path with potential HDU specifier
        
    Returns:
        tuple: (clean_path, hdu_index) where hdu_index is None if not specified
    """
    match = re.match(r'^(.*?)(?:\[(\d+)\])?$', filepath)
    if match:
        path, hdu_spec = match.groups()
        if hdu_spec is not None:
            return path, int(hdu_spec)
    return filepath, None


def create_binary_mask(mask_data, bit_flag):
    """
    Create a binary mask (0/1) from a bitmask for a specific flag.
    
    Args:
        mask_data (ndarray): Bitmask array
        bit_flag (int): Bit flag to extract
        
    Returns:
        ndarray: Binary mask (0=not set, 1=set)
    """
    return np.where((mask_data & bit_flag) > 0, 1, 0).astype(np.uint8)


def estimate_saturation_from_histogram(data, min_adu=None, max_adu=None, min_counts=None, drop_factor=None):
    """
    Estimates saturation level from histogram, with automatic parameter determination.
    
    Args:
        data (ndarray): Image data array
        min_adu (float, optional): Lower bound ADU value for histogram analysis. If None, auto-determined.
        max_adu (float, optional): Upper bound ADU value for histogram analysis. If None, auto-determined.
        min_counts (int, optional): Minimum counts threshold. If None, auto-determined.
        drop_factor (float, optional): Counts ratio threshold between adjacent bins. If None, auto-determined.
    
    Returns:
        float or None: Estimated saturation level, or None if not found
    """
    # Auto-determine parameters if not provided
    try:
        # Filter out infinities and NaNs for percentile calculations
        finite_data = data[np.isfinite(data)]
        if finite_data.size == 0:
            print("  Histogram analysis: No finite data found.")
            return None
            
        # Auto-determine min_adu and max_adu
        if min_adu is None or max_adu is None:
            p95 = np.percentile(finite_data, 95)
            p99_9 = np.percentile(finite_data, 99.9)
            p_max = np.max(finite_data)
            
            auto_min_adu = p95 * 0.9  # Start a bit below 95th percentile
            auto_max_adu = min(p_max * 1.1, p99_9 * 1.5)  # Don't go too far beyond max value
            
            min_adu = min_adu if min_adu is not None else auto_min_adu
            max_adu = max_adu if max_adu is not None else auto_max_adu
            
            print(f"  Auto-determined histogram range: [{min_adu:.1f},{max_adu:.1f}] ADU")
        
        # Auto-determine min_counts
        if min_counts is None:
            # Calculate based on image size - larger images need higher thresholds
            img_size = data.size
            auto_min_counts = max(10, int(img_size * 1e-5))  # Adjust factor as needed
            min_counts = auto_min_counts
            print(f"  Auto-determined minimum counts threshold: {min_counts}")
        
        # Auto-determine drop_factor
        if drop_factor is None:
            # Default based on typical CCD/CMOS saturation behavior
            drop_factor = 5.0
            print(f"  Using default drop factor: {drop_factor:.1f}")
    
    except Exception as e:
        print(f"  Parameter auto-determination failed: {e}")
        # Fall back to reasonable defaults
        min_adu = min_adu if min_adu is not None else 30000
        max_adu = max_adu if max_adu is not None else 65000
        min_counts = min_counts if min_counts is not None else 10
        drop_factor = drop_factor if drop_factor is not None else 5.0
    
    # Log final parameters
    print(f"  Attempting histogram analysis: range=[{min_adu:.1f},{max_adu:.1f}], min_counts={min_counts}, drop_factor={drop_factor:.1f}")
    
    # Main histogram analysis (keeping existing logic with minor improvements)
    try:
        valid_data = data[(data >= min_adu) & (data <= max_adu) & np.isfinite(data)]
        if valid_data.size < min_counts * 5:
            print("  Histogram analysis: Not enough valid pixels in range.")
            return None
            
        bin_edges = np.arange(min_adu, max_adu + 2, 1)
        counts, _ = np.histogram(valid_data, bins=bin_edges)
        
        potential_sat_indices = np.where(counts >= min_counts)[0]
        if len(potential_sat_indices) == 0:
            print("  Histogram analysis: No bins found with counts above threshold.")
            return None
            
        for idx in range(len(potential_sat_indices) - 1, -1, -1):
            i = potential_sat_indices[idx]
            if i == len(counts) - 1:
                estimated_level = bin_edges[i]
                print(f"  Histogram analysis: Found saturation plateau ending at highest analyzed bin: {estimated_level:.1f} ADU")
                return float(estimated_level)
                
            counts_current = counts[i]
            counts_next = counts[i+1]
            
            if counts_next <= 0:
                if counts_current >= min_counts:
                    estimated_level = bin_edges[i]
                    print(f"  Histogram analysis: Found sharp drop to zero after bin {i}. Level: {estimated_level:.1f} ADU")
                    return float(estimated_level)
                else:
                    continue
                    
            if counts_current / counts_next >= drop_factor:
                estimated_level = bin_edges[i]
                print(f"  Histogram analysis: Found drop factor > {drop_factor:.1f} after bin {i}. Level: {estimated_level:.1f} ADU")
                return float(estimated_level)
                
        print("  Histogram analysis: No significant drop found.")
        return None
        
    except Exception as e:
        print(f"  Histogram analysis failed with error: {e}")
        return None


def calculate_inverse_variance(sky_map_ff, flat_map, gain, read_noise_e, epsilon):
    """ Calculates inverse variance map. """
    # [Function code remains the same - snipped for brevity]
    valid_flat_mask = (flat_map > epsilon); safe_flat = np.where(valid_flat_mask, flat_map, epsilon)
    safe_sky_ff = np.maximum(sky_map_ff, 0.0); read_noise_var_e = read_noise_e**2
    signal_var_e = safe_sky_ff * safe_flat * gain; total_var_e = read_noise_var_e + signal_var_e
    gain_sq = gain**2 + epsilon; variance_adu_ff = total_var_e / ((gain_sq * safe_flat**2) + epsilon)
    inv_variance = 1.0 / (variance_adu_ff + epsilon)
    inv_variance[~valid_flat_mask] = 0.0; inv_variance[total_var_e < epsilon] = 0.0
    return inv_variance.astype(np.float32)


def process_hdu(hdu_sci, hdu_flat, config, hdu_index):
    """
    Processes a single Science HDU to generate mask, inv variance, and conf map data.

    Args:
        hdu_sci (fits.ImageHDU): Science image HDU.
        hdu_flat (fits.ImageHDU or None): Flat field HDU, or None if using flat=1.
        config (dict): Dictionary containing configuration parameters.
        hdu_index (int): Index of the HDU in the original FITS file.

    Returns:
        tuple: (mask_data, inv_var_data, weight_data, conf_data, sky_map, header_info)
               Returns (None, None, None, None, None, None) on failure for this HDU.
               header_info is a dict of key parameters used (for output headers).
    """
    hdu_start_time = time.time()
    print(f"\n--- Processing HDU {hdu_index} ({hdu_sci.name}) ---")

    # --- 1. Load Data ---
    if hdu_sci.data is None or not isinstance(hdu_sci, (fits.ImageHDU, fits.CompImageHDU)):
         print("Skipping HDU: Science data missing or not an image HDU.")
         return None, None, None, None, None, None
    
    # Fix: Define shape variable before using it
    sci_shape = hdu_sci.data.shape
    if hdu_flat is not None and (hdu_flat.data is None or hdu_flat.data.shape != sci_shape):
         print("Skipping HDU: Flat data missing or shape mismatch.")
         return None, None, None, None, None, None

    sci_data = np.ascontiguousarray(hdu_sci.data.astype(np.float32))
    sci_hdr = hdu_sci.header
    # Load flat or create array of ones
    if hdu_flat is not None:
        flat_data = np.ascontiguousarray(hdu_flat.data.astype(np.float32))
        using_unit_flat = False
    else:
        print("  INFO: No flat field provided, assuming flat = 1.0.")
        flat_data = np.ones_like(sci_data, dtype=np.float32)
        using_unit_flat = True

    final_mask_int = np.zeros(sci_data.shape, dtype=MASK_DTYPE)
    header_info = {} # To store parameters used

    # --- 2. Get Noise Parameters ---
    gain = sci_hdr.get(config['variance']['gain_keyword'], config['variance']['default_gain'])
    read_noise_e = sci_hdr.get(config['variance']['rdnoise_keyword'], config['variance']['default_rdnoise'])
    if config['variance']['gain_keyword'] not in sci_hdr: print(f"  WARNING: Keyword '{config['variance']['gain_keyword']}' not found. Using default: {gain:.2f} e-/ADU.")
    if config['variance']['rdnoise_keyword'] not in sci_hdr: print(f"  WARNING: Keyword '{config['variance']['rdnoise_keyword']}' not found. Using default: {read_noise_e:.2f} e-.")
    header_info['GAIN_USD'] = gain
    header_info['RDNS_USD'] = read_noise_e

    # --- 3. Initial Masks (Bad Flat, Saturation) ---
    print("Creating initial masks (Bad Flat, Saturation)...")
    # Bad Flat Pixels (skip if flat is unity)
    flat_mask_bool = np.zeros(sci_data.shape, dtype=bool)
    if not using_unit_flat:
        with np.errstate(invalid='ignore'): median_flat = np.nanmedian(flat_data[flat_data > 0])
        if not np.isfinite(median_flat) or median_flat <= 0: median_flat = 1.0
        flat_low = config['flat_masking']['low_thresh'] * median_flat
        flat_high = config['flat_masking']['high_thresh'] * median_flat
        flat_mask_bool = ( (flat_data <= flat_low) |
                           (flat_data >= flat_high) |
                           (flat_data <= 0) | (~np.isfinite(flat_data)) )
        final_mask_int[flat_mask_bool] |= MASK_BITS['BAD']
        print(f"  Masked {np.sum(flat_mask_bool)} BAD pixels from Flat (Low={flat_low:.2f}, High={flat_high:.2f}).")
    else:
        print("  Skipping bad flat pixel mask (using unit flat).")

    # Saturation Detection
    cfg_sat = config['saturation']
    saturation_level = None; sat_method_used = 'none'
    if cfg_sat['method'] == 'histogram':
        saturation_level = estimate_saturation_from_histogram(sci_data, cfg_sat['hist_min_adu'], cfg_sat['hist_max_adu'], cfg_sat['hist_min_counts'], cfg_sat['hist_drop_factor'])
        if saturation_level is not None: sat_method_used = 'histogram'
        else:
            print("  Histogram failed. Trying header fallback...")
            if cfg_sat['keyword'] and cfg_sat['keyword'] in sci_hdr:
                try: saturation_level = float(sci_hdr[cfg_sat['keyword']]); sat_method_used = 'header (fallback)'; print(f"  Using saturation from header: {saturation_level:.1f} ADU.")
                except (ValueError, TypeError): print(f"  Header fallback failed (parse error)."); saturation_level = None
            else: print(f"  Header fallback failed (keyword missing).")
    elif cfg_sat['method'] == 'header':
        if cfg_sat['keyword'] and cfg_sat['keyword'] in sci_hdr:
            try: saturation_level = float(sci_hdr[cfg_sat['keyword']]); sat_method_used = 'header'; print(f"  Using saturation from header: {saturation_level:.1f} ADU.")
            except (ValueError, TypeError): print(f"  Header method failed (parse error)."); saturation_level = None
        else: print(f"  Header method failed (keyword missing).")
    if saturation_level is None:
        saturation_level = cfg_sat['fallback_level']; sat_method_used = 'default'
        print(f"  WARNING: Using fallback saturation: {saturation_level:.1f} ADU.")
    header_info['SAT_LVL'] = saturation_level
    header_info['SAT_METH'] = sat_method_used

    sat_mask_bool = (sci_data >= saturation_level)
    final_mask_int[sat_mask_bool] |= MASK_BITS['SAT']
    print(f"  Masked {np.sum(sat_mask_bool)} SAT pixels.")
    interim_mask_1_bool = flat_mask_bool | sat_mask_bool # Initial combined boolean mask

    # --- 4. Initial Background/RMS Estimation (SEP) ---
    print("Estimating initial background/RMS...")
    cfg_bkg = config['sep_background']
    try:
        bkg1 = sep.Background(sci_data, mask=interim_mask_1_bool, bw=cfg_bkg['box_size'], bh=cfg_bkg['box_size'], fw=cfg_bkg['filter_size'], fh=cfg_bkg['filter_size'])
        bkg_map1 = bkg1.back(); bkg_rms_map = bkg1.rms()
        print(f"  Initial background global RMS: {bkg1.globalrms:.3f}")
    except Exception as e: # Fallback
        print(f"  Initial background estimation failed: {e}. Using global."); bkg_val, bkg_rms_global = sep.background(sci_data, mask=interim_mask_1_bool); bkg_map1 = np.full(sci_data.shape, bkg_val, dtype=np.float32); bkg_rms_map = np.full(sci_data.shape, bkg_rms_global, dtype=np.float32); print(f"  Using global RMS: {bkg_rms_global:.3f}")
    bkg_rms_map = np.where(np.isfinite(bkg_rms_map) & (bkg_rms_map > 0), bkg_rms_map, np.inf)

    # --- 5. Cosmic Ray Detection (astroscrappy) ---
    print(f"Detecting Cosmic Rays...")
    cfg_cr = config['cosmic_ray']
    header_info['CR_SIG'] = cfg_cr['sigclip']
    try:
        crmask_bool, _ = detect_cosmics(sci_data, inmask=interim_mask_1_bool, satlevel=saturation_level, gain=gain, readnoise=read_noise_e, sigclip=cfg_cr['sigclip'], objlim=cfg_cr['objlim'], verbose=False)
        cr_add_mask = crmask_bool & (~interim_mask_1_bool)
        final_mask_int[cr_add_mask] |= MASK_BITS['CR']
        print(f"  Masked {np.sum(cr_add_mask)} new CR pixels (sigclip={cfg_cr['sigclip']}).")
    except Exception as e: print(f"  Astroscrappy failed: {e}"); crmask_bool = np.zeros(sci_data.shape, dtype=bool)
    interim_mask_2_bool = interim_mask_1_bool | crmask_bool # Include CRs

    # --- 6. Object Detection (SEP) ---
    print(f"Detecting Objects...")
    cfg_obj = config['sep_objects']
    header_info['SEP_THR'] = cfg_obj['extract_thresh']
    object_mask_bool = np.zeros(sci_data.shape, dtype=bool)
    try:
        data_sub = sci_data - bkg_map1
        objects = sep.extract(data_sub, thresh=cfg_obj['extract_thresh'], err=bkg_rms_map, mask=interim_mask_2_bool, minarea=cfg_obj['min_area'], segmentation_map=False)
        print(f"  Detected {len(objects)} objects (thresh={cfg_obj['extract_thresh']} sigma).")
        if len(objects) > 0:
            sep.mask_ellipse(object_mask_bool, objects['x'], objects['y'], objects['a'], objects['b'], objects['theta'], r=cfg_obj['ellipse_k'])
            obj_add_mask = object_mask_bool & (~interim_mask_2_bool)
            final_mask_int[obj_add_mask] |= MASK_BITS['DETECTED']
            print(f"  Masked {np.sum(obj_add_mask)} new DETECTED pixels (k={cfg_obj['ellipse_k']}).")
    except Exception as e: print(f"  SEP extraction failed: {e}")
    interim_mask_3_bool = interim_mask_2_bool | object_mask_bool # Include objects

    # --- 7. Streak Detection (Hough Transform) ---
    cfg_streak = config['streak_masking']
    header_info['STRK_EN'] = cfg_streak['enable']
    streak_mask_final_bool = np.zeros(sci_data.shape, dtype=bool)
    if cfg_streak['enable']:
        print("Detecting Streaks using Hough Transform...")
        streak_start_time = time.time()
        header_info['STRK_ITHR'] = cfg_streak['input_threshold_sigma']
        header_info['STRK_PKTH'] = cfg_streak['hough_peak_threshold_factor']
        header_info['STRK_DIL'] = cfg_streak['dilation_radius']
        try:
            hough_input_threshold = cfg_streak['input_threshold_sigma'] * bkg_rms_map
            hough_input_image = (data_sub > hough_input_threshold)
            hough_input_image[interim_mask_3_bool] = False # Mask known stuff
            print(f"  Hough input: {np.sum(hough_input_image)} pixels > {cfg_streak['input_threshold_sigma']:.1f} sigma.")

            if np.any(hough_input_image):
                tested_angles = np.linspace(-np.pi / 2, np.pi / 2, cfg_streak['hough_angles'], endpoint=False)
                h_space, h_angles, h_dists = hough_line(hough_input_image, theta=tested_angles)
                hough_threshold = cfg_streak['hough_peak_threshold_factor'] * np.max(h_space) if np.max(h_space) > 0 else 0
                h_peaks = hough_line_peaks(h_space, h_angles, h_dists, threshold=hough_threshold, min_distance=cfg_streak['hough_min_distance'], min_angle=cfg_streak['hough_min_angle'])
                num_peaks = len(h_peaks[0])
                print(f"  Found {num_peaks} potential line peaks in Hough space.")

                if num_peaks > 0:
                    streak_lines_mask = np.zeros(sci_data.shape, dtype=bool)
                    img_rows, img_cols = sci_data.shape
                    for _, angle, dist in zip(*h_peaks):
                        # Simplified line drawing - find points on border
                        pts = []
                        # Top border (y=0)
                        if np.cos(angle) != 0: x_at_y0 = (dist - 0 * np.sin(angle)) / np.cos(angle);
                        if 0 <= x_at_y0 < img_cols: pts.append((0, int(round(x_at_y0))))
                        # Bottom border (y=rows-1)
                        if np.cos(angle) != 0: x_at_yN = (dist - (img_rows-1) * np.sin(angle)) / np.cos(angle);
                        if 0 <= x_at_yN < img_cols: pts.append((img_rows - 1, int(round(x_at_yN))))
                        # Left border (x=0)
                        if np.sin(angle) != 0 and len(pts)<2: y_at_x0 = (dist - 0 * np.cos(angle)) / np.sin(angle);
                        if 0 <= y_at_x0 < img_rows: pts.append((int(round(y_at_x0)), 0))
                        # Right border (x=cols-1)
                        if np.sin(angle) != 0 and len(pts)<2: y_at_xN = (dist - (img_cols-1) * np.cos(angle)) / np.sin(angle);
                        if 0 <= y_at_xN < img_rows: pts.append((int(round(y_at_xN)), img_cols-1))

                        if len(pts) >= 2:
                            r0, c0 = pts[0]; r1, c1 = pts[-1] # Use first and last found border point
                            r0, c0 = max(0, min(r0, img_rows-1)), max(0, min(c0, img_cols-1))
                            r1, c1 = max(0, min(r1, img_rows-1)), max(0, min(c1, img_cols-1))
                            rr, cc, val = line_aa(r0, c0, r1, c1)
                            valid_idx = (rr>=0) & (rr<img_rows) & (cc>=0) & (cc<img_cols)
                            streak_lines_mask[rr[valid_idx], cc[valid_idx]] = True

                    if np.any(streak_lines_mask):
                         print(f"  Dilating {np.sum(streak_lines_mask)} line pixels by radius {cfg_streak['dilation_radius']}...")
                         selem = disk(cfg_streak['dilation_radius'])
                         streak_mask_final_bool = binary_dilation(streak_lines_mask, structure=selem)
            else:
                print("  No candidate pixels found for Hough input.")

            # Add streak flag to the main mask, avoiding previously masked pixels
            streak_add_mask = streak_mask_final_bool & (~interim_mask_3_bool)
            final_mask_int[streak_add_mask] |= MASK_BITS['STREAK']
            print(f"  Masked {np.sum(streak_add_mask)} new STREAK pixels.")
            print(f"  Streak detection finished in {time.time() - streak_start_time:.2f} seconds.")
        except Exception as e: print(f"  Streak detection failed: {e}")
    else: print("Streak masking disabled.")

    # --- 8. Final Background Estimation (SEP) ---
    print("Estimating final background...")
    final_combined_bool_mask = (final_mask_int > 0) # Includes ALL bits now
    try:
        bkg_final = sep.Background(sci_data, mask=final_combined_bool_mask, bw=cfg_bkg['box_size'], bh=cfg_bkg['box_size'], fw=cfg_bkg['filter_size'], fh=cfg_bkg['filter_size'])
        sky_map = bkg_final.back(); print(f"  Final background global mean: {np.mean(sky_map):.3f}")
    except Exception as e: print(f"  Final background estimation failed: {e}. Using initial map."); sky_map = bkg_map1

    # --- 9. Inverse Variance Map ---
    print("Calculating inverse variance map...")
    # Pure inverse variance map - represents inverse of sky background variance without masking
    inv_variance_map = calculate_inverse_variance(sky_map, flat_data, gain, read_noise_e, config['variance']['epsilon'])
    print(f"  Inverse variance map calculated (mean non-zero: {np.mean(inv_variance_map[inv_variance_map > 0]):.4g})")
    
    # --- 10. Weight Map (Inverse Variance with bad pixels masked) ---
    print("Calculating weight map...")
    # Create a mask for bad/defective pixels that should have zero weight
    bad_pixel_mask = (
        (final_mask_int & MASK_BITS['BAD']) |      # Bad pixels
        (final_mask_int & MASK_BITS['SAT']) |      # Saturated pixels
        (final_mask_int & MASK_BITS['CR']) |       # Cosmic rays
        (final_mask_int & MASK_BITS['STREAK'])     # Streaks
    ) > 0
    
    # Create weight map from inverse variance but mask out problematic pixels
    weight_map = inv_variance_map.copy()
    weight_map[bad_pixel_mask] = 0.0
    print(f"  Weight map calculated - masked {np.sum(bad_pixel_mask)} bad pixels")
    print(f"  Note: Object pixels remain valid in weight map")

    # --- 11. Confidence Map ---
    print("Generating confidence map...")
    cfg_conf = config['confidence_map']
    conf_dtype = getattr(np, cfg_conf['dtype'], np.uint8) # Use numpy dtype
    conf_map_data = np.full(sci_data.shape, cfg_conf['bad_value'], dtype=conf_dtype)
    # Good = No mask bits set AND InvVar > 0
    good_pixels_bool = (~final_combined_bool_mask) & (inv_variance_map > 0)
    conf_map_data[good_pixels_bool] = cfg_conf['good_value']
    num_good_pixels = np.sum(good_pixels_bool)
    percent_good = 100 * num_good_pixels / conf_map_data.size
    print(f"  Confidence map generated: {num_good_pixels} good pixels ({percent_good:.2f}%).")

    hdu_elapsed = time.time() - hdu_start_time
    print(f"--- HDU processed in {hdu_elapsed:.2f} seconds ---")

    return final_mask_int, inv_variance_map, weight_map, conf_map_data, sky_map, header_info


def main():
    """ Main function to parse arguments and run the pipeline. """
    parser = argparse.ArgumentParser(description="Generate Mask, Inverse Variance, and Confidence Maps for FITS files.")
    parser.add_argument('input_file', type=str, help="Path to input FITS file (can include CFITSIO HDU specifier, e.g., file.fits[1]).")
    parser.add_argument('--config', type=str, default="weightmask.yml", required=True, help="Path to YAML configuration file.")
    parser.add_argument('--flat_image', type=str, default=None, help="Path to input flat field FITS file (optional, assumes flat=1 if omitted).")
    parser.add_argument('--output_mask', type=str, default=None, help="Path for output bitmask FITS file (optional, default based on input file).")
    parser.add_argument('--output_invvar', type=str, default=None, help="Path for output inverse variance FITS file (optional, default based on input file).")
    parser.add_argument('--output_conf', type=str, default=None, help="Path for output confidence map FITS file (optional, default based on input file).")
    parser.add_argument('--output_weight', type=str, default=None, help="Path for output weight map file (inv variance with masked pixels=0).")
    parser.add_argument('--output_sky', type=str, default=None, help="Path for output smoothed sky background map file.")
    parser.add_argument('--hdu', type=int, default=None, help="HDU index to process (0=primary, 1=first extension). Overrides HDU specified in filename.")
    parser.add_argument('--individual_masks', action='store_true', help="Output individual mask component files (bad, saturated, cosmic, streak).")
    args = parser.parse_args()

    print("Starting Integrated Pipeline...")
    start_pipeline_time = time.time()

    # --- Load Configuration ---
    print(f"Loading configuration from: {args.config}")
    try:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        # Basic validation could be added here - check presence of main keys
        if not all(k in config for k in ['flat_masking', 'saturation', 'sep_background', 'cosmic_ray', 'sep_objects', 'streak_masking', 'variance', 'confidence_map']):
             warnings.warn("Config file might be missing some expected top-level keys.")
    except Exception as e:
        print(f"ERROR: Failed to load or parse config file '{args.config}': {e}")
        return

    # --- Parse CFITSIO-style HDU specifiers ---
    input_path, input_hdu = extract_hdu_spec(args.input_file)
    flat_path, flat_hdu = (extract_hdu_spec(args.flat_image) if args.flat_image else (None, None))
    
    # Command-line --hdu overrides filename HDU specifier
    if args.hdu is not None:
        input_hdu = args.hdu
        flat_hdu = args.hdu  # Also use --hdu for flat if not otherwise specified
    
    # --- Determine Output Filenames ---
    # Get the base filename without any extension
    base = os.path.basename(input_path)
    
    # Handle various extension cases, including compressed FITS formats
    # First, check for common FITS compression extensions
    if base.lower().endswith(('.fits.gz', '.fits.fz', '.fit.gz', '.fit.fz')):
        # Remove both the .fits/.fit and compression extensions (.gz/.fz)
        base = os.path.splitext(os.path.splitext(base)[0])[0]
    elif base.lower().endswith(('.fits', '.fit')):
        # Remove just the .fits/.fit extension
        base = os.path.splitext(base)[0]
    elif '.' in base:
        # For any other extension, remove it
        base = os.path.splitext(base)[0]
    
    output_dir = os.path.dirname(input_path)

    # --- Open Input FITS ---
    try:
        hdul_input = fits.open(input_path, memmap=True)
        hdul_flat = fits.open(flat_path, memmap=True) if flat_path else None
    except FileNotFoundError as e:
        print(f"ERROR: Input file not found: {e}")
        return
    except Exception as e:
        print(f"ERROR: Could not open input files: {e}")
        if 'hdul_input' in locals() and hdul_input: hdul_input.close()
        return

    # --- Handle HDU Selection ---
    hdus_to_process = []
    hdu_name_suffix = ""
    if input_hdu is not None:
        # Process only the specific HDU
        if input_hdu < len(hdul_input):
            hdus_to_process = [input_hdu]
            # Get HDU name for output filenames
            hdu_name = hdul_input[input_hdu].name if hasattr(hdul_input[input_hdu], 'name') and hdul_input[input_hdu].name else f"HDU{input_hdu}"
            hdu_name_suffix = f"_{hdu_name}"
            print(f"Processing single HDU: {input_hdu} (name: {hdu_name})")
        else:
            print(f"ERROR: Specified HDU {input_hdu} not found in input file with {len(hdul_input)} HDUs.")
            return
    else:
        # Process all HDUs except the primary (0)
        hdus_to_process = range(1, len(hdul_input))
        print(f"Processing all HDUs: {len(hdus_to_process)} extension(s)")

    # Create output filenames with HDU name suffix if needed
    out_mask_path = args.output_mask or os.path.join(output_dir, f"{base}{hdu_name_suffix}.mask.fits")
    out_invvar_path = args.output_invvar or os.path.join(output_dir, f"{base}{hdu_name_suffix}.ivar.fits")
    out_conf_path = args.output_conf or os.path.join(output_dir, f"{base}{hdu_name_suffix}.conf.fits")
    out_weight_path = args.output_weight or os.path.join(output_dir, f"{base}{hdu_name_suffix}.weight.fits")
    out_sky_path = args.output_sky or os.path.join(output_dir, f"{base}{hdu_name_suffix}.sky.fits")
    
    # Individual mask component files if requested
    if args.individual_masks:
        out_bad_path = os.path.join(output_dir, f"{base}{hdu_name_suffix}.bad.fits")
        out_satur_path = os.path.join(output_dir, f"{base}{hdu_name_suffix}.satur.fits")
        out_cosmics_path = os.path.join(output_dir, f"{base}{hdu_name_suffix}.cosmics.fits")
        out_streaks_path = os.path.join(output_dir, f"{base}{hdu_name_suffix}.streaks.fits")
        print(f"Individual mask components will be written to:")
        print(f"  Bad Pixels:    {out_bad_path}")
        print(f"  Saturation:    {out_satur_path}")
        print(f"  Cosmic Rays:   {out_cosmics_path}")
        print(f"  Streaks:       {out_streaks_path}")

    print(f"Input File: {input_path}{f'[{input_hdu}]' if input_hdu is not None else ''}")
    print(f"Flat Image: {flat_path}{f'[{flat_hdu}]' if flat_path and flat_hdu is not None else '' if flat_path else 'None (using 1.0)'}")
    print(f"Output Mask:   {out_mask_path}")
    print(f"Output InvVar: {out_invvar_path}")
    print(f"Output Conf:   {out_conf_path}")
    print(f"Output Weight: {out_weight_path}")
    print(f"Output Sky:    {out_sky_path}")

    # --- Prepare Output HDU Lists ---
    hdul_mask_out = fits.HDUList([hdul_input[0].copy()])
    hdul_invvar_out = fits.HDUList([hdul_input[0].copy()])
    hdul_conf_out = fits.HDUList([hdul_input[0].copy()])
    hdul_weight_out = fits.HDUList([hdul_input[0].copy()])
    hdul_sky_out = fits.HDUList([hdul_input[0].copy()])
    
    # Individual mask HDULists if requested
    if args.individual_masks:
        hdul_bad_out = fits.HDUList([hdul_input[0].copy()])
        hdul_satur_out = fits.HDUList([hdul_input[0].copy()])
        hdul_cosmics_out = fits.HDUList([hdul_input[0].copy()])
        hdul_streaks_out = fits.HDUList([hdul_input[0].copy()])

    # --- Process HDUs ---
    print(f"\nProcessing {len(hdus_to_process)} science extension(s)...")
    warnings.filterwarnings('ignore', category=UserWarning, append=True)
    warnings.filterwarnings('ignore', category=RuntimeWarning, append=True)

    for i in hdus_to_process:
        try:
            hdu_sci = hdul_input[i]
            # Get corresponding flat HDU or None
            hdu_flat = hdul_flat[i] if hdul_flat and i < len(hdul_flat) else None

            # Process this HDU with explicit index
            mask_data, inv_var_data, weight_data, conf_data, sky_map, header_info = process_hdu(hdu_sci, hdu_flat, config, i)

            if mask_data is not None: # Check if processing was successful
                # Create and append HDUs
                mask_hdu = fits.ImageHDU(data=mask_data, header=hdu_sci.header.copy(), name=f'MASK_{hdu_sci.name}')
                mask_hdu.header['COMMENT'] = 'Combined Mask (Bitmask)'; # Add more comments...
                for name, bit in MASK_BITS.items(): mask_hdu.header[f'BIT_{name}'] = (bit, f'Mask bit for {name}')
                # Add parameters from header_info
                mask_hdu.header['SATURATE'] = (header_info.get('SAT_LVL'), f"Saturation level used ({header_info.get('SAT_METH')})")
                mask_hdu.header['CRSIGMA'] = (header_info.get('CR_SIG'), 'Astroscrappy sigclip parameter used')
                mask_hdu.header['SEPTHRES'] = (header_info.get('SEP_THR'), 'SEP extraction threshold (sigma)')
                if header_info.get('STRK_EN', False):
                     mask_hdu.header['STREAKEN'] = (True, 'Streak masking enabled')
                     mask_hdu.header['STRKTHRS'] = (header_info.get('STRK_ITHR'), 'Streak Hough input sigma thresh')
                     mask_hdu.header['STRKPKTH'] = (header_info.get('STRK_PKTH'), 'Streak Hough peak factor')
                     mask_hdu.header['STRKDILR'] = (header_info.get('STRK_DIL'), 'Streak dilation radius (pix)')
                else: mask_hdu.header['STREAKEN'] = (False, 'Streak masking disabled')
                
                # Add original HDU index
                mask_hdu.header['ORIG_HDU'] = (i, 'Original HDU index in input file')

                hdul_mask_out.append(mask_hdu)

                # Pure inverse variance map (no masking)
                invvar_hdu = fits.ImageHDU(data=inv_var_data, header=hdu_sci.header.copy(), name=f'INVVAR_{hdu_sci.name}')
                invvar_hdu.header['COMMENT'] = 'Pure Inverse Variance Map (no masking, represents sky background uncertainty)'
                invvar_hdu.header['BUNIT'] = ('adu**-2', 'Inverse variance units')
                invvar_hdu.header['GAIN'] = (header_info.get('GAIN_USD'), 'Gain (e-/ADU) used for variance')
                invvar_hdu.header['RDNOISE'] = (header_info.get('RDNS_USD'), 'Read Noise (e-) used for variance')
                invvar_hdu.header['ORIG_HDU'] = (i, 'Original HDU index in input file')
                hdul_invvar_out.append(invvar_hdu)

                # Weight map (inverse variance with masked pixels)
                weight_hdu = fits.ImageHDU(data=weight_data, header=hdu_sci.header.copy(), name=f'WEIGHT_{hdu_sci.name}')
                weight_hdu.header['COMMENT'] = 'Weight Map (inverse variance with bad/saturated/CR/streak pixels set to 0)'
                weight_hdu.header['BUNIT'] = ('adu**-2', 'Weight units')
                weight_hdu.header['ORIG_HDU'] = (i, 'Original HDU index in input file')
                hdul_weight_out.append(weight_hdu)
                
                # Create sky background map
                sky_hdu = fits.ImageHDU(data=sky_map, header=hdu_sci.header.copy(), name=f'SKY_{hdu_sci.name}')
                sky_hdu.header['COMMENT'] = 'Smoothed Sky Background Map'
                sky_hdu.header['BUNIT'] = ('adu', 'Sky background value units')
                sky_hdu.header['BKGBOX'] = (config['sep_background']['box_size'], 'Background box size used')
                sky_hdu.header['BKGFILT'] = (config['sep_background']['filter_size'], 'Background filter size used')
                sky_hdu.header['ORIG_HDU'] = (i, 'Original HDU index in input file')
                hdul_sky_out.append(sky_hdu)
                
                conf_hdu = fits.ImageHDU(data=conf_data, header=hdu_sci.header.copy(), name=f'CONF_{hdu_sci.name}')
                conf_hdu.header['COMMENT'] = f"Confidence Map ({config['confidence_map']['good_value']}=Good, {config['confidence_map']['bad_value']}=Bad)"# Add more comments...
                conf_hdu.header['ORIG_HDU'] = (i, 'Original HDU index in input file')
                hdul_conf_out.append(conf_hdu)
                
                # Create individual mask files if requested
                if args.individual_masks:
                    # Create binary masks for each component
                    bad_mask = create_binary_mask(mask_data, MASK_BITS['BAD'])
                    satur_mask = create_binary_mask(mask_data, MASK_BITS['SAT'])
                    cosmic_mask = create_binary_mask(mask_data, MASK_BITS['CR'])
                    streak_mask = create_binary_mask(mask_data, MASK_BITS['STREAK'])
                    
                    # Create HDUs for each component
                    bad_hdu = fits.ImageHDU(data=bad_mask, header=hdu_sci.header.copy(), name=f'BAD_{hdu_sci.name}')
                    bad_hdu.header['COMMENT'] = 'Binary mask for bad pixels (1=bad, 0=good)'
                    bad_hdu.header['ORIG_HDU'] = (i, 'Original HDU index in input file')
                    hdul_bad_out.append(bad_hdu)
                    
                    satur_hdu = fits.ImageHDU(data=satur_mask, header=hdu_sci.header.copy(), name=f'SATUR_{hdu_sci.name}')
                    satur_hdu.header['COMMENT'] = 'Binary mask for saturated pixels (1=saturated, 0=good)'
                    satur_hdu.header['SATURATE'] = (header_info.get('SAT_LVL'), f"Saturation level used ({header_info.get('SAT_METH')})")
                    satur_hdu.header['ORIG_HDU'] = (i, 'Original HDU index in input file')
                    hdul_satur_out.append(satur_hdu)
                    
                    cosmic_hdu = fits.ImageHDU(data=cosmic_mask, header=hdu_sci.header.copy(), name=f'COSMIC_{hdu_sci.name}')
                    cosmic_hdu.header['COMMENT'] = 'Binary mask for cosmic ray pixels (1=CR, 0=good)'
                    cosmic_hdu.header['CRSIGMA'] = (header_info.get('CR_SIG'), 'Astroscrappy sigclip parameter used')
                    cosmic_hdu.header['ORIG_HDU'] = (i, 'Original HDU index in input file')
                    hdul_cosmics_out.append(cosmic_hdu)
                    
                    streak_hdu = fits.ImageHDU(data=streak_mask, header=hdu_sci.header.copy(), name=f'STREAK_{hdu_sci.name}')
                    streak_hdu.header['COMMENT'] = 'Binary mask for streak pixels (1=streak, 0=good)'
                    streak_hdu.header['STREAKEN'] = (header_info.get('STRK_EN', False), 'Streak masking enabled')
                    streak_hdu.header['ORIG_HDU'] = (i, 'Original HDU index in input file')
                    hdul_streaks_out.append(streak_hdu)

        except Exception as e:
            # Catch-all for unexpected errors during HDU processing call
            import traceback
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            print(f"FATAL ERROR processing HDU {i} ({hdu_sci.name if 'hdu_sci' in locals() else 'Unknown'}): {e}")
            print(traceback.format_exc())
            print(f"Skipping output for this HDU.")
            print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    # --- Write Output Files ---
    print(f"\nWriting combined mask file to: {out_mask_path}")
    try: hdul_mask_out.writeto(out_mask_path, overwrite=True, checksum=True); print(" Mask file OK.")
    except Exception as e: print(f" ERROR writing mask file: {e}")

    print(f"Writing inverse variance map file to: {out_invvar_path}")
    try: hdul_invvar_out.writeto(out_invvar_path, overwrite=True, checksum=True); print(" InvVar file OK.")
    except Exception as e: print(f" ERROR writing inverse variance file: {e}")

    print(f"Writing confidence map file to: {out_conf_path}")
    try: hdul_conf_out.writeto(out_conf_path, overwrite=True, checksum=True); print(" Conf file OK.")
    except Exception as e: print(f" ERROR writing confidence file: {e}")
    
    print(f"Writing weight map file to: {out_weight_path}")
    try: hdul_weight_out.writeto(out_weight_path, overwrite=True, checksum=True); print(" Weight file OK.")
    except Exception as e: print(f" ERROR writing weight file: {e}")
    
    print(f"Writing sky background map file to: {out_sky_path}")
    try: hdul_sky_out.writeto(out_sky_path, overwrite=True, checksum=True); print(" Sky background file OK.")
    except Exception as e: print(f" ERROR writing sky background file: {e}")

    # Write individual mask files if requested
    if args.individual_masks:
        print(f"Writing individual mask component files...")
        
        try: hdul_bad_out.writeto(out_bad_path, overwrite=True); print(" Bad pixel mask OK.")
        except Exception as e: print(f" ERROR writing bad pixel mask: {e}")
        
        try: hdul_satur_out.writeto(out_satur_path, overwrite=True); print(" Saturation mask OK.")
        except Exception as e: print(f" ERROR writing saturation mask: {e}")
        
        try: hdul_cosmics_out.writeto(out_cosmics_path, overwrite=True); print(" Cosmic ray mask OK.")
        except Exception as e: print(f" ERROR writing cosmic ray mask: {e}")
        
        try: hdul_streaks_out.writeto(out_streaks_path, overwrite=True); print(" Streak mask OK.")
        except Exception as e: print(f" ERROR writing streak mask: {e}")

    # --- Close Files & Finish ---
    if hdul_input: hdul_input.close()
    if hdul_flat: hdul_flat.close()
    warnings.filterwarnings('default', category=UserWarning)
    warnings.filterwarnings('default', category=RuntimeWarning)
    pipeline_elapsed = time.time() - start_pipeline_time
    print(f"\nPipeline finished in {pipeline_elapsed:.2f} seconds.")


if __name__ == "__main__":
    main()