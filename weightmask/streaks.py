import numpy as np
from skimage.measure import label, regionprops, LineModelND, ransac
from skimage.morphology import binary_dilation, disk, opening
from skimage.feature import canny
from skimage.transform import probabilistic_hough_line
from skimage.draw import line_aa
import warnings
import time

# --- RANSAC-based Detection Method ---
def _detect_streaks_ransac(data_sub, bkg_rms_map, existing_mask, config):
    """
    Internal function for streak detection using region properties and RANSAC.
    (Code based on the previous response)
    """
    streak_mask_final_bool = np.zeros(data_sub.shape, dtype=bool)
    img_rows, img_cols = data_sub.shape

    # --- Configuration Parameters ---
    cfg = config.get('ransac_params', {}) # Get RANSAC specific params
    # Candidate Selection
    input_threshold_sigma = cfg.get('input_threshold_sigma', 3.0)
    use_canny = cfg.get('use_canny', True)
    canny_sigma = cfg.get('canny_sigma', 1.0)
    canny_low_threshold = cfg.get('canny_low_threshold', 0.1)
    canny_high_threshold = cfg.get('canny_high_threshold', 0.5)
    # Region Filtering
    min_elongation = cfg.get('min_elongation', 5.0)
    min_pixels = cfg.get('min_pixels', 20)
    max_pixels = cfg.get('max_pixels', 10000)
    # RANSAC Parameters
    ransac_min_samples = cfg.get('ransac_min_samples', 5)
    ransac_residual_threshold = cfg.get('ransac_residual_threshold', 1.0)
    ransac_max_trials = cfg.get('ransac_max_trials', 100)
    # Validation
    min_inliers = cfg.get('min_inliers', 15)
    # Dilation
    dilation_radius = config.get('dilation_radius', 5) # Use common dilation radius

    print("--> Using RANSAC Method for Streak Detection")
    print(f"    Config: Elong>{min_elongation}, Pix=[{min_pixels}-{max_pixels}], RANSAC thresh={ransac_residual_threshold}, Min Inliers={min_inliers}")

    start_time = time.time()
    try:
        # --- 1. Candidate Pixel Selection ---
        threshold = input_threshold_sigma * bkg_rms_map
        candidate_mask = (data_sub > threshold) & (~existing_mask)
        if use_canny:
            with warnings.catch_warnings():
                 warnings.simplefilter("ignore", category=UserWarning)
                 edges = canny(data_sub, sigma=canny_sigma,
                               low_threshold=canny_low_threshold,
                               high_threshold=canny_high_threshold,
                               mask=~existing_mask)
            candidate_mask = candidate_mask & edges
            print(f"    Canny edges identified {np.sum(edges)} pixels.")
        num_candidates = np.sum(candidate_mask)
        print(f"    Found {num_candidates} initial candidate pixels.")
        if num_candidates < min_pixels: return np.zeros(data_sub.shape, dtype=bool)

        # --- 2. Connected Components ---
        labeled_mask, num_labels = label(candidate_mask, connectivity=2, return_num=True)
        if num_labels == 0: return np.zeros(data_sub.shape, dtype=bool)

        # --- 3. Region Filtering & RANSAC ---
        regions = regionprops(labeled_mask)
        potential_streaks = []
        for region in regions:
            if not (min_pixels <= region.area <= max_pixels): continue
            if region.minor_axis_length > 1e-3:
                elongation = region.major_axis_length / region.minor_axis_length
            else: elongation = np.inf
            if elongation >= min_elongation: potential_streaks.append(region)
        print(f"    Filtered down to {len(potential_streaks)} potential streak regions.")

        streak_inlier_coords_rows = []
        streak_inlier_coords_cols = []
        num_ransac_success = 0
        for region in potential_streaks:
            coords = region.coords
            if len(coords) < ransac_min_samples: continue
            try:
                model_robust, inliers = ransac(coords, LineModelND,
                                               min_samples=ransac_min_samples,
                                               residual_threshold=ransac_residual_threshold,
                                               max_trials=ransac_max_trials)
                num_inlier_pixels = np.sum(inliers)
                if num_inlier_pixels >= min_inliers:
                    num_ransac_success += 1
                    inlier_coords = coords[inliers]
                    streak_inlier_coords_rows.extend(inlier_coords[:, 0])
                    streak_inlier_coords_cols.extend(inlier_coords[:, 1])
            except Exception: continue
        print(f"    Validated {num_ransac_success} regions as streaks using RANSAC.")

        # --- 4. Mask Generation & Dilation ---
        if streak_inlier_coords_rows:
            streak_core_mask = np.zeros(data_sub.shape, dtype=bool)
            valid_rows = np.array(streak_inlier_coords_rows)
            valid_cols = np.array(streak_inlier_coords_cols)
            idx = (valid_rows >= 0) & (valid_rows < img_rows) & (valid_cols >= 0) & (valid_cols < img_cols)
            streak_core_mask[valid_rows[idx], valid_cols[idx]] = True
            print(f"    Dilating {np.sum(streak_core_mask)} streak core pixels by radius {dilation_radius}...")
            selem = disk(dilation_radius)
            if selem.size > 0 : streak_mask_final_bool = binary_dilation(streak_core_mask, structure=selem)
            else: streak_mask_final_bool = streak_core_mask

    except Exception as e:
        import traceback
        print(f"    RANSAC streak detection failed: {e}")
        print(traceback.format_exc())
        return np.zeros(data_sub.shape, dtype=bool)

    elapsed = time.time() - start_time
    print(f"    RANSAC method finished in {elapsed:.2f} seconds.")
    return streak_mask_final_bool


# --- Probabilistic Hough Transform Method ---
def _detect_streaks_hough(data_sub, bkg_rms_map, existing_mask, config):
    """
    Internal function for streak detection using Probabilistic Hough Transform.
    """
    streak_mask_final_bool = np.zeros(data_sub.shape, dtype=bool)
    img_rows, img_cols = data_sub.shape

    # --- Configuration Parameters ---
    cfg = config.get('hough_params', {}) # Get Hough specific params
    input_threshold_sigma = cfg.get('input_threshold_sigma', 3.0)
    use_canny = cfg.get('use_canny', False) # Canny optional for Hough input
    canny_sigma = cfg.get('canny_sigma', 1.0)
    canny_low_threshold = cfg.get('canny_low_threshold', 0.1)
    canny_high_threshold = cfg.get('canny_high_threshold', 0.5)
    # Morphological filtering options
    use_morph_open = cfg.get('use_morph_open', False)
    morph_kernel_size = cfg.get('morph_kernel_size', 5)
    # Probabilistic Hough parameters
    prob_hough_threshold = cfg.get('prob_hough_threshold', 10) # Min votes to count line
    prob_hough_line_length = cfg.get('prob_hough_line_length', 50) # Min line length
    prob_hough_line_gap = cfg.get('prob_hough_line_gap', 10) # Max gap between segments
    # Dilation
    dilation_radius = config.get('dilation_radius', 5) # Use common dilation radius

    print("--> Using Probabilistic Hough Method for Streak Detection")
    print(f"    Config: Threshold={prob_hough_threshold}, Length={prob_hough_line_length}, Gap={prob_hough_line_gap}")

    start_time = time.time()
    try:
        # --- 1. Candidate Pixel Selection ---
        threshold = input_threshold_sigma * bkg_rms_map
        hough_input_image = (data_sub > threshold) & (~existing_mask)

        if use_canny:
            with warnings.catch_warnings():
                 warnings.simplefilter("ignore", category=UserWarning)
                 edges = canny(data_sub, sigma=canny_sigma,
                               low_threshold=canny_low_threshold,
                               high_threshold=canny_high_threshold,
                               mask=~existing_mask)
            hough_input_image = hough_input_image & edges # Combine threshold and edges
            print(f"    Using Canny edges ({np.sum(edges)} pixels) for Hough input.")
        elif use_morph_open:
             # Morphological opening can help remove small noise points
             # Use a linear kernel if you expect streaks in certain orientations,
             # otherwise a disk/square might work. Using disk here for generality.
             selem = disk(morph_kernel_size // 2) # Approx size
             if selem.size > 0:
                  opened_mask = opening(hough_input_image, structure=selem)
                  print(f"    Applied morphological opening, {np.sum(opened_mask)} pixels remain.")
                  hough_input_image = opened_mask
             else:
                  print("    Morphological opening skipped (kernel size too small).")


        num_candidates = np.sum(hough_input_image)
        print(f"    Found {num_candidates} candidate pixels for Hough input.")
        if num_candidates < prob_hough_line_length: # Need enough pixels for longest line
            print("    Too few candidate pixels to proceed.")
            return np.zeros(data_sub.shape, dtype=bool)

        # --- 2. Probabilistic Hough Transform ---
        # Generate theta values for Hough transform (all angles)
        tested_angles = np.linspace(-np.pi / 2, np.pi / 2, 360, endpoint=False) # Standard angle range

        lines = probabilistic_hough_line(hough_input_image,
                                         threshold=prob_hough_threshold,
                                         line_length=prob_hough_line_length,
                                         line_gap=prob_hough_line_gap,
                                         theta=tested_angles)

        print(f"    Found {len(lines)} potential line segments.")

        # --- 3. Mask Generation ---
        if lines:
            streak_core_mask = np.zeros(data_sub.shape, dtype=bool)
            for p0, p1 in lines:
                 # Use line_aa for anti-aliased line drawing (integer mask)
                 rr, cc, val = line_aa(p0[1], p0[0], p1[1], p1[0]) # Note: skimage uses row, col (y, x)
                 # Clip coordinates to be within image bounds
                 valid_idx = (rr >= 0) & (rr < img_rows) & (cc >= 0) & (cc < img_cols)
                 streak_core_mask[rr[valid_idx], cc[valid_idx]] = True # Set valid pixels to True

            # --- 4. Dilation ---
            print(f"    Dilating {np.sum(streak_core_mask)} streak core pixels by radius {dilation_radius}...")
            selem = disk(dilation_radius)
            if selem.size > 0:
                 streak_mask_final_bool = binary_dilation(streak_core_mask, structure=selem)
            else:
                 streak_mask_final_bool = streak_core_mask

    except Exception as e:
        import traceback
        print(f"    Probabilistic Hough streak detection failed: {e}")
        print(traceback.format_exc())
        return np.zeros(data_sub.shape, dtype=bool)

    elapsed = time.time() - start_time
    print(f"    Probabilistic Hough method finished in {elapsed:.2f} seconds.")
    return streak_mask_final_bool


# --- Main Detection Function ---
def detect_streaks(data_sub, bkg_rms_map, existing_mask, config):
    """
    Detect linear streaks using the method specified in the configuration.

    Args:
        data_sub (ndarray): Background-subtracted image data.
        bkg_rms_map (ndarray): Background RMS map.
        existing_mask (ndarray): Boolean mask of already masked pixels.
        config (dict): Configuration dictionary for streak detection.
                       Must contain 'enable' (bool) and 'method' ('ransac' or 'hough').
                       Should also contain method-specific parameter sub-dicts
                       ('ransac_params', 'hough_params') and 'dilation_radius'.

    Returns:
        ndarray: Boolean mask of newly detected streak pixels.
    """
    if not config.get('enable', False):
        print("Streak masking disabled in main config.")
        return np.zeros(data_sub.shape, dtype=bool)

    method = config.get('method', 'ransac').lower() # Default to RANSAC if not specified

    if method == 'ransac':
        streak_mask_bool = _detect_streaks_ransac(data_sub, bkg_rms_map, existing_mask, config)
    elif method == 'hough':
        streak_mask_bool = _detect_streaks_hough(data_sub, bkg_rms_map, existing_mask, config)
    else:
        print(f"  ERROR: Unknown streak detection method '{method}'. Choose 'ransac' or 'hough'.")
        return np.zeros(data_sub.shape, dtype=bool)

    # Only return newly detected pixels (not already in existing_mask)
    streak_add_mask = streak_mask_bool & (~existing_mask)
    num_new_pixels = np.sum(streak_add_mask)
    if num_new_pixels > 0:
        print(f"  Final streak mask includes {num_new_pixels} new pixels (Method: {method}).")
    else:
        print(f"  No new streak pixels added by method '{method}'.")

    return streak_add_mask