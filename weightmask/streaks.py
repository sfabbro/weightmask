import numpy as np
import time
import warnings
import concurrent.futures
from skimage.measure import label, regionprops, ransac, LineModelND
from skimage.morphology import binary_dilation, disk, white_tophat
from skimage.filters import frangi, apply_hysteresis_threshold

def _detect_streaks_frangi(data_sub, bkg_rms_map, existing_mask, config):
    """
    Internal function for streak detection using Frangi ridge filter and hysteresis.
    This works in the continuous domain and is much more robust for single noisy exposures.
    """
    img_rows, img_cols = data_sub.shape
    streak_mask_final_bool = np.zeros(data_sub.shape, dtype=bool)

    # --- Configuration Parameters ---
    cfg = config.get('frangi_params', {})
    
    # 1. Background / Star Removal
    tophat_radius = cfg.get('tophat_radius', 10)  # Should be wider than streaks
    
    # 2. Frangi Filter Params
    sigmas = cfg.get('sigmas', [1, 2, 3])  # Scales for the ridge filter (determines streak width)
    black_ridges = cfg.get('black_ridges', False)
    
    # 3. Hysteresis Thresholding 
    # High threshold = strict undeniable streak cores (e.g. 99th percentile of filter response)
    # Low threshold = growing the mask along the ridge until it fades into noise
    high_threshold_sig = cfg.get('high_threshold_sig', 5.0)  # Multiplier on median absolute deviation of frangi map
    low_threshold_sig = cfg.get('low_threshold_sig', 1.5)
    
    # 4. Region limits
    min_area = cfg.get('min_area', 50)

    
    # 5. Dilation
    dilation_radius = config.get('dilation_radius', 3)

    print("--> Using Frangi Ridge Method for Streak Detection")
    start_time = time.time()
    
    try:
        # Step 1: Pre-process (Remove stars/large background with Top-Hat)
        # Fill existing mask with local median to avoid edge artifacts during filtering
        # Step 1: Pre-process (Remove stars/large background with Top-Hat)
        # We NO LONGER median-fill existing masks, because bright streaks might 
        # have been partially flagged as objects/cosmics. Median-filling them 
        # literally erases the streak before Frangi runs!
        data_clean = data_sub.copy()
                 
        print(f"    Applying White Top-Hat (radius={tophat_radius}) to flatten sky/stars...")
        selem = disk(tophat_radius)
        if selem.size > 0:
            tophat_img = white_tophat(data_clean, footprint=selem)
        else:
            tophat_img = data_clean

        # Normalize by background RMS so the noise floor is ~1.0. 
        # This makes Frangi output amplitudes consistent regardless of exposure time or noise.
        if bkg_rms_map is not None:
            # avoid division by zero
            safe_rms = np.where(bkg_rms_map <= 0, 1.0, bkg_rms_map)
            tophat_img = tophat_img / safe_rms
            
        # Step 2: Frangi Ridge Filter (Block Processing for large images)
        print(f"    Applying Frangi Filter (sigmas={sigmas})...")
        ridge_map = np.zeros_like(tophat_img)
        
        block_size = cfg.get('block_size', 1024)
        pad = cfg.get('block_pad', 32)
        
        # Determine if we should block-process
        if img_rows > block_size or img_cols > block_size:
            print(f"    Image is large ({img_rows}x{img_cols}). Using parallel block processing (size={block_size}, pad={pad}).")

            def process_block(r, c):
                r_start_pad = max(0, r - pad)
                r_end_pad = min(img_rows, r + block_size + pad)
                c_start_pad = max(0, c - pad)
                c_end_pad = min(img_cols, c + block_size + pad)

                block = tophat_img[r_start_pad:r_end_pad, c_start_pad:c_end_pad]

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    block_ridge = frangi(block, sigmas=sigmas, black_ridges=black_ridges)

                # Calculate valid interior indices to paste back
                valid_r_start = r - r_start_pad
                valid_r_end = valid_r_start + min(block_size, img_rows - r)
                valid_c_start = c - c_start_pad
                valid_c_end = valid_c_start + min(block_size, img_cols - c)

                h = valid_r_end - valid_r_start
                w = valid_c_end - valid_c_start
                interior = block_ridge[valid_r_start:valid_r_end, valid_c_start:valid_c_end]

                return r, c, h, w, interior

            futures = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for r in range(0, img_rows, block_size):
                    for c in range(0, img_cols, block_size):
                        futures.append(executor.submit(process_block, r, c))

                for f in concurrent.futures.as_completed(futures):
                    r, c, h, w, interior = f.result()
                    ridge_map[r:r+h, c:c+w] = interior
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                ridge_map = frangi(tophat_img, sigmas=sigmas, black_ridges=black_ridges)
             
        # Step 3: Hysteresis Thresholding
        # Since the input to the Frangi filter is normalized by the local background RMS, 
        # the output amplitudes are scale-invariant and physically meaningful.
        # We prioritize absolute thresholds from the config, with MAD as a fallback.
        high_thresh = cfg.get('high_threshold')
        low_thresh = cfg.get('low_threshold')
        
        if high_thresh is None or low_thresh is None:
            # Fallback to MAD-based thresholding if absolute ones aren't provided
            high_threshold_sig = cfg.get('high_threshold_sig', 3.0)
            low_threshold_sig = cfg.get('low_threshold_sig', 1.0)
            
            # Isolate background pixels (less than 1-sigma and not already masked)
            bkg_mask = tophat_img < 1.0
            if existing_mask is not None:
                # We MUST exclude existing masks from the threshold estimation sample
                # otherwise star ridges will bias our noise statistics upwards.
                bkg_mask &= (~existing_mask)
            
            if np.sum(bkg_mask) < 1000:
                bkg_mask = np.ones_like(tophat_img, dtype=bool) # Fallback if entirely crowded
                
            bkg_ridge = ridge_map[bkg_mask]
            
            # Use percentiles for robust estimation in extremely non-Gaussian backgrounds
            if len(bkg_ridge) > 1000:
                p50, p90, p99 = np.percentile(bkg_ridge, [50, 90, 99])
                # Empirically scale the thresholds based on the long tail of the distribution
                tail_spread = p99 - p50
                if tail_spread == 0:
                    tail_spread = 1e-9
                
                high_thresh = p50 + high_threshold_sig * tail_spread
                low_thresh = p50 + low_threshold_sig * tail_spread
            else:
                high_thresh = 1e-5 # Fallback constants
                low_thresh = 5e-6
                
            # Floor to ensure we don't pick up infinite 0s
            high_thresh = max(high_thresh, 1e-6)
            low_thresh = max(low_thresh, 1e-7)
            
        print(f"    Hysteresis Thresholds: Low={low_thresh:.2e}, High={high_thresh:.2e}")
        
        hyst_mask = apply_hysteresis_threshold(ridge_map, low_thresh, high_thresh)
        print(f"    Hysteresis found {np.sum(hyst_mask)} candidate pixel groups.")

        # Step 4: Region Filtering (Elongation & Size)
        labeled_mask, num_labels = label(hyst_mask, connectivity=2, return_num=True)
        if num_labels == 0:
            return np.zeros(data_sub.shape, dtype=bool)
            
        regions = regionprops(labeled_mask)
        streak_core_mask = np.zeros(data_sub.shape, dtype=bool)
        
        num_valid_streaks = 0
        for region in regions:
            if region.area < min_area:
                continue
                
            # Discard streaks that are predominantly already masked (e.g. saturation bleed trails)
            if existing_mask is not None:
                coords = region.coords
                valid_rows = coords[:, 0]
                valid_cols = coords[:, 1]
                idx = (valid_rows >= 0) & (valid_rows < img_rows) & (valid_cols >= 0) & (valid_cols < img_cols)
                r_mask = existing_mask[valid_rows[idx], valid_cols[idx]]
                if np.sum(r_mask) > 0.5 * region.area:
                    continue
            
            # Elongation check
            # For ridges, elongation can be misleading if the segment breaks up.
            # We trust the Frangi filter's shape response and instead just check
            # if the region is long enough.
            if region.major_axis_length >= 10:
                num_valid_streaks += 1
                coords = region.coords
                valid_rows = coords[:, 0]
                valid_cols = coords[:, 1]
                idx = (valid_rows >= 0) & (valid_rows < img_rows) & (valid_cols >= 0) & (valid_cols < img_cols)
                streak_core_mask[valid_rows[idx], valid_cols[idx]] = True
                
        print(f"    Validated {num_valid_streaks} regions as streaks based on length.")

        # Step 5: Dilation
        print(f"    Dilating {np.sum(streak_core_mask)} streak core pixels by radius {dilation_radius}...")
        selem = disk(dilation_radius)
        if selem.size > 0:
            streak_mask_final_bool = binary_dilation(streak_core_mask, footprint=selem)
        else:
            streak_mask_final_bool = streak_core_mask

    except Exception as e:
        import traceback
        print(f"    Frangi streak detection failed: {e}")
        print(traceback.format_exc())
        return np.zeros(data_sub.shape, dtype=bool)

    elapsed = time.time() - start_time
    print(f"    Frangi method finished in {elapsed:.2f} seconds.")
    return streak_mask_final_bool

def _detect_trails_ransac(data_sub, bkg_rms_map, existing_mask, config):
    """
    Internal function for sparse trail detection using RANSAC line fitting.
    Ideal for faint "glint" trails (dotted lines of point sources).
    """
    cfg = config.get('ransac_params', {})
    base_detect_thresh_sig = cfg.get('detect_thresh_sig', 5.0)
    base_min_inliers = cfg.get('min_inliers', 10)
    residual_threshold = cfg.get('residual_threshold', 2.0)
    
    detect_thresh_sig = base_detect_thresh_sig
    min_inliers = base_min_inliers
    
    # --- Adaptive RANSAC Thresholds (Density/Clutter Scaling) ---
    # We inspect the global background to see how 'cluttered' or heavy-tailed the noise is.
    # A purely Gaussian background has a P99 around 2.3 sigma.
    valid_mask = ~existing_mask if existing_mask is not None else np.ones(data_sub.shape, dtype=bool)
    if bkg_rms_map is not None:
        valid_mask &= (bkg_rms_map > 0)
    valid_data = data_sub[valid_mask]
    
    if len(valid_data) > 1000:
        # ⚡ Bolt: Subsample large arrays before calculating global robust statistics
        step = max(1, len(valid_data) // 100000)
        sampled_data = valid_data[::step]
        p50, p99 = np.percentile(sampled_data, [50, 99])
        mad_approx = np.median(np.abs(sampled_data - p50)) * 1.4826
        if mad_approx > 0:
            tail_ratio = (p99 - p50) / mad_approx
            if tail_ratio > 3.0:
                # Highly cluttered field (many stars/artifacts). We must stiffen RANSAC requirements.
                clutter_penalty = min(tail_ratio / 3.0, 4.0)
                detect_thresh_sig *= np.sqrt(clutter_penalty)
                min_inliers = int(min_inliers * clutter_penalty)
                print(f"    [Adaptive RANSAC] Clutter penalty {clutter_penalty:.2f}x applied (tail ratio {tail_ratio:.1f}). New thresholds: sig={detect_thresh_sig:.1f}, min_inliers={min_inliers}")
    
    # 1. Candidate points: Bright points above noise floor
    if bkg_rms_map is not None:
        median_rms = np.median(bkg_rms_map)
        # Logarithmic threshold scaling for extremely noisy backgrounds
        if median_rms > 15.0:
            effective_sig = detect_thresh_sig * (1.0 + 0.5 * np.log10(median_rms / 15.0))
            thresh = effective_sig * bkg_rms_map
        else:
            thresh = detect_thresh_sig * bkg_rms_map
    else:
        from astropy.stats import mad_std

        thresh = detect_thresh_sig * mad_std(data_sub)
        
    candidate_mask = (data_sub > thresh) & (~existing_mask)
    
    # Optional: Filter tiny noise spikes before RANSAC to improve performance
    # but be careful not to kill thin streaks. 
    # Instead of opening, we can use a minimum clump size filter if needed.
    # For now, let's keep it raw but increase the coordinate limit more aggressively.
        
    coords = np.argwhere(candidate_mask) # (N, 2) array of [y, x]
    
    # Optional: if points are overwhelmingly noise (e.g. >30% of image), give up RANSAC
    if len(coords) > 0.3 * data_sub.size:
         print(f"    WARNING: Too many points for RANSAC ({len(coords)}). Skipping RANSAC.")
         return np.zeros(data_sub.shape, dtype=bool)
    
    if len(coords) < min_inliers:
        return np.zeros(data_sub.shape, dtype=bool)

    print(f"--> Using RANSAC Method for Sparse Trail Detection ({len(coords)} candidate points)")
    trail_mask = np.zeros(data_sub.shape, dtype=bool)
    
    try:
        # We might have multiple trails. Try to find the dominant one.
        # For now, just find one dominant trail per call.
        model_robust, inliers = ransac(coords, LineModelND, min_samples=2,
                                       residual_threshold=residual_threshold, max_trials=1000)
        
        if inliers is not None and np.sum(inliers) >= min_inliers:
            inlier_coords = coords[inliers]
            # Calculate trail extent: find the dimension (Y or X) with largest spread
            # and use that to find the actual endpoints of the trail.
            diffs = inlier_coords.max(axis=0) - inlier_coords.min(axis=0)
            sort_dim = np.argmax(diffs)
            p0_idx = np.argmin(inlier_coords[:, sort_dim])
            p1_idx = np.argmax(inlier_coords[:, sort_dim])
            p0 = inlier_coords[p0_idx]
            p1 = inlier_coords[p1_idx]
            
            length = np.sqrt(np.sum((p1 - p0)**2))
            
            if length > 0:
                density = np.sum(inliers) / length
            else:
                density = 0.0
                
            min_line_density = cfg.get("min_line_density", 0.2)
            if length > cfg.get("min_length", 100) and density > min_line_density:
                print(f"    RANSAC found trail: length={length:.1f} pix, inliers={np.sum(inliers)}, density={density:.3f}")
                # Draw the line in the mask
                from skimage.draw import line
                y0, x0 = p0.astype(int)
                y1, x1 = p1.astype(int)
                # Ensure within bounds
                h, w = data_sub.shape
                y0, x0 = np.clip([y0, x0], [0, 0], [h-1, w-1])
                y1, x1 = np.clip([y1, x1], [0, 0], [h-1, w-1])
                
                rr, cc = line(y0, x0, y1, x1)
                trail_mask[rr, cc] = True
                
                # Dilate the thin line
                dilation_radius = config.get('dilation_radius', 3)
                selem = disk(dilation_radius)
                trail_mask = binary_dilation(trail_mask, footprint=selem)
                
    except Exception as e:
        print(f"    RANSAC trail detection failed: {e}")
        
    return trail_mask

# --- Main Detection Function ---
def detect_streaks(data_sub, bkg_rms_map, existing_mask, config):
    """
    Detect linear streaks using the method specified in the configuration.

    Args:
        data_sub (ndarray): Background-subtracted image data.
        bkg_rms_map (ndarray): Background RMS map.
        existing_mask (ndarray): Boolean mask of already masked pixels.
        config (dict): Configuration dictionary for streak detection.
                       Must contain 'enable' (bool) and 'method' ('frangi').
                       Should also contain method-specific parameter sub-dicts
                       ('frangi_params') and 'dilation_radius'.

    Returns:
        ndarray: Boolean mask of newly detected streak pixels.
    """
    if not config.get('enable', False):
        print("Streak masking disabled in main config.")
        return np.zeros(data_sub.shape, dtype=bool)

    method = config.get('method', 'frangi').lower()

    streak_mask_bool = np.zeros(data_sub.shape, dtype=bool)
    
    # Run primary selected method
    if method == 'frangi':
        streak_mask_bool |= _detect_streaks_frangi(data_sub, bkg_rms_map, existing_mask, config)
    
    # Optionally run RANSAC trail detection to catch sparse tracks
    if config.get('enable_ransac_trails', True):
        streak_mask_bool |= _detect_trails_ransac(data_sub, bkg_rms_map, existing_mask, config)

    # We return the full boolean mask of the detected streak. 
    # If the user wants to know how many *new* pixels were added, we can compute it,
    # but the function must return the full mask so the benchmark can evaluate it properly.
    if existing_mask is not None:
        num_new_pixels = np.sum(streak_mask_bool & (~existing_mask))
    else:
        num_new_pixels = np.sum(streak_mask_bool)
        
    if num_new_pixels > 0:
        print(f"  Final streak mask includes {num_new_pixels} new pixels (Method: {method}).")
    else:
        print(f"  No new streak pixels added by method '{method}'.")

    return streak_mask_bool