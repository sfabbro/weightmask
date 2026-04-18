import numpy as np
import warnings

import scipy.signal
import scipy.ndimage


def estimate_saturation_robust_clump(data, min_adu=None, max_adu=None):
    """
    Robust Detrended Saturation Detection.
    Finds smeared saturation limits without causing artificial saturation on empty fields.
    Uses 1D peak finding on the extreme right tail of the intensity distribution.
    If no isolated clump is found (i.e. smooth exponential drop-off), returns None.

    Args:
        data (ndarray): Image data array (should be float).
        min_adu (float, optional): Lower bound ADU value for analysis. If None, auto-determined.
        max_adu (float, optional): Upper bound ADU value for analysis. If None, auto-determined.

    Returns:
        float or None: Estimated saturation level, or None if no saturation is present.
    """
    try:
        # Filter out infinities and NaNs
        finite_data = data[np.isfinite(data)]
        if finite_data.size == 0:
            print("  Robust Clump: No finite data found.")
            return None

        p_max = np.max(finite_data)

        # Auto-determine min_adu and max_adu if not provided
        if min_adu is None or max_adu is None:
            # We want to exclude the vast majority of normal sky/star pixels.
            # Start analysis above the 99th percentile, but ensure we don't start too high
            # if the field is sparse.

            # ⚡ Bolt: Subsample large arrays before calculating global robust statistics
            step = max(1, finite_data.size // 100000)
            sampled_data = finite_data[::step]

            p99 = np.percentile(sampled_data, 99)
            p99_9 = np.percentile(sampled_data, 99.9)

            # Heuristic: start halfway between the 99th percentile and max,
            # or 80% of the 99.9th percentile, whichever is more conservative.
            auto_min_adu = min(p99 + (p_max - p99) * 0.3, p99_9 * 0.8)
            auto_max_adu = min(
                p_max * 1.01, p99_9 * 1.5
            )  # Allow some room above 99.9th

            min_adu = min_adu if min_adu is not None else auto_min_adu
            max_adu = max_adu if max_adu is not None else auto_max_adu

            # Absolute sanity check: if max is small, we shouldn't be looking for saturation
            if p_max < 10000:
                print(
                    f"  Robust Clump: Max ADU ({p_max:.1f}) is very low. Assuming no saturation."
                )
                return None

            if max_adu <= min_adu:
                max_adu = min_adu + 100.0

            print(
                f"  Auto-determined histogram range: [{min_adu:.1f}, {max_adu:.1f}] ADU"
            )

        print(
            f"  Attempting robust clump analysis: range=[{min_adu:.1f}, {max_adu:.1f}]"
        )

        # Bin the extreme tail into ~100 bins.
        # This prevents the histogram from dissolving into noise for smeared clumps.
        bins = np.linspace(min_adu, max_adu, 100)
        counts, bin_edges = np.histogram(finite_data, bins=bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        if np.sum(counts) < 20:
            print(
                "  Robust Clump: Not enough pixels in extreme tail (empty field). No saturation detected."
            )
            return None

        # Smooth the histogram heavily to find the macro-structure (the clump)
        smoothed_counts = scipy.ndimage.gaussian_filter1d(
            counts.astype(float), sigma=2.0
        )

        # Find peaks.
        # Expected prominence is at least a few percent of the max smoothed count in this tail.
        min_prominence = max(2.0, np.max(smoothed_counts) * 0.05)

        peaks, properties = scipy.signal.find_peaks(
            smoothed_counts, prominence=min_prominence
        )

        if len(peaks) == 0:
            print(
                "  Robust Clump: Smooth intensity tail with no anomalous clumps. No saturation detected."
            )
            return None

        # The saturation clump is usually the most prominent peak in this extreme tail
        best_peak_idx = peaks[np.argmax(properties["prominences"])]
        peak_adu = bin_centers[best_peak_idx]

        # The saturation threshold should be set at the START of the clump,
        # which is approximated by the left base of the peak.
        left_base_idx = properties["left_bases"][np.argmax(properties["prominences"])]

        estimated_level = bin_centers[left_base_idx]

        # Sanity check: don't let it fall all the way back to min_adu if the base is poorly defined
        estimated_level = max(estimated_level, min_adu + (peak_adu - min_adu) * 0.2)

        print(f"  Robust Clump: Found anomalous clump at ~{peak_adu:.1f} ADU.")
        print(
            f"  Robust Clump: Setting threshold near the base of the clump: {estimated_level:.1f} ADU."
        )

        return float(estimated_level)

    except Exception as e:
        import traceback

        print(f"  Robust Clump analysis failed with error: {e}")
        print(traceback.format_exc())
        return None


def _get_saturation_from_header(sci_hdr, header_keyword):
    """Attempt to extract saturation level from the header."""
    if not header_keyword or sci_hdr is None or header_keyword not in sci_hdr:
        print(f"  Header method failed (keyword '{header_keyword}' missing or not specified).")
        return None

    try:
        saturation_level = float(sci_hdr[header_keyword])
        print(f"  Using saturation from header keyword '{header_keyword}': {saturation_level:.1f} ADU.")
        return saturation_level
    except (ValueError, TypeError):
        print(f"  Header method failed (parse error for keyword '{header_keyword}').")
        return None


def detect_saturated_pixels(sci_data, sci_hdr, config):
    """
    Detect saturated pixels in the science data using the configured method.

    Args:
        sci_data (ndarray): Science image data array (float32 recommended).
        sci_hdr (fits.Header): Science image header.
        config (dict): Configuration dictionary for saturation detection.

    Returns:
        tuple: (saturation_level, sat_method_used, sat_mask_bool)
               saturation_level (float): The determined saturation level in ADU.
               sat_method_used (str): Method used ('histogram', 'header', 'default').
               sat_mask_bool (ndarray): Boolean mask where True indicates saturated pixels.
    """
    saturation_level = None
    sat_method_used = "none"
    hist_params = config.get(
        "histogram_params", {}
    )  # Get sub-dictionary for histogram params

    # Ensure data is float for calculations
    if not np.issubdtype(sci_data.dtype, np.floating):
        warnings.warn(
            "Science data is not float, casting to float32 for saturation check.",
            UserWarning,
        )
        sci_data = sci_data.astype(np.float32)

    method = config.get("method", "histogram")
    header_keyword = config.get("keyword")

    if method == "histogram":
        print("Attempting robust detrended saturation detection (Clump method)...")
        # Pass specific histogram parameters if they exist in the config
        saturation_level = estimate_saturation_robust_clump(
            sci_data,
            min_adu=hist_params.get("hist_min_adu"),  # Use .get for safety
            max_adu=hist_params.get("hist_max_adu"),
        )

        if saturation_level is not None:
            sat_method_used = "histogram (robust clump)"
        else:
            print(
                "  Robust Clump detection yielded no saturation. Trying header fallback..."
            )
            saturation_level = _get_saturation_from_header(sci_hdr, header_keyword)
            if saturation_level is not None:
                sat_method_used = "header (fallback)"

    elif method == "header":
        print("Attempting saturation detection via header keyword...")
        saturation_level = _get_saturation_from_header(sci_hdr, header_keyword)
        if saturation_level is not None:
            sat_method_used = "header"

    # Final fallback to default value if all methods fail
    if saturation_level is None:
        fallback_level = config.get(
            "fallback_level", 65535.0
        )  # Default fallback if not in config
        saturation_level = fallback_level
        sat_method_used = "default fallback"
        print(
            f"  WARNING: Using fallback saturation level: {saturation_level:.1f} ADU."
        )

    # Ensure saturation_level is a float before comparison
    saturation_level = float(saturation_level)

    # Create the boolean mask for core saturated pixels
    # Handle potential NaNs/Infs in input data safely
    with np.errstate(invalid="ignore"):  # Suppress warnings from comparing with NaN/Inf
        sat_mask_bool = (sci_data >= saturation_level) & np.isfinite(sci_data)

    print(
        f"  Final saturation level used: {saturation_level:.1f} ADU (Method: {sat_method_used})"
    )

    return saturation_level, sat_method_used, sat_mask_bool


def _grow_bleed_up(sci_data, stop_thresh, x, y_min, max_grow, new_mask):
    """Helper to grow bleed trail upward from a saturated segment."""
    if y_min <= 0:
        return

    start_idx = y_min - 1
    end_idx = max(-1, start_idx - max_grow)

    if end_idx == -1:
        sci_slice = sci_data[start_idx::-1, x]
        thresh_slice = stop_thresh[start_idx::-1]
    else:
        sci_slice = sci_data[start_idx:end_idx:-1, x]
        thresh_slice = stop_thresh[start_idx:end_idx:-1]

    cond = sci_slice > thresh_slice
    if not np.all(cond):
        grown = np.argmin(cond)
    else:
        grown = len(cond)

    if grown > 0:
        end_mask = start_idx - grown
        if end_mask == -1:
            new_mask[start_idx::-1, x] = True
        else:
            new_mask[start_idx:end_mask:-1, x] = True


def _grow_bleed_down(sci_data, stop_thresh, h, x, y_max, max_grow, new_mask):
    """Helper to grow bleed trail downward from a saturated segment."""
    if y_max >= h - 1:
        return

    start_idx = y_max + 1
    end_idx = min(h, start_idx + max_grow)

    sci_slice = sci_data[start_idx:end_idx, x]
    thresh_slice = stop_thresh[start_idx:end_idx]

    cond = sci_slice > thresh_slice
    if not np.all(cond):
        grown = np.argmin(cond)
    else:
        grown = len(cond)

    if grown > 0:
        new_mask[start_idx : start_idx + grown, x] = True


def grow_bleed_trails(sci_data, sat_mask, sky_map, bkg_rms_map, config):
    """
    Grow saturated regions vertically to mask bleed trails (blooming).
    Uses a region-growing algorithm that stops when flux hits the background level.

    Args:
        sci_data (ndarray): Science image data.
        sat_mask (ndarray): Boolean mask of saturated cores.
        sky_map (ndarray): Background sky map.
        bkg_rms_map (ndarray): Background RMS map.
        config (dict): Configuration for bleed masking.

    Returns:
        ndarray: Boolean mask with expanded bleed trails.
    """
    if not config.get("mask_bleed_trails", True):
        return sat_mask

    print("  Growing bleed trails based on local flux levels...")
    h, w = sci_data.shape
    new_mask = sat_mask.copy()

    # Identify columns with saturation
    sat_cols = np.where(np.any(sat_mask, axis=0))[0]

    for x in sat_cols:
        col_sat = sat_mask[:, x]
        # Find contiguous segments of saturated pixels in this column
        labeled_segments, num_segments = scipy.ndimage.label(col_sat)

        for s in range(1, num_segments + 1):
            segment_indices = np.where(labeled_segments == s)[0]
            y_min, y_max = segment_indices.min(), segment_indices.max()

            # Get background levels for this column
            col_bkg = sky_map[:, x] if sky_map is not None else np.zeros(h)
            col_rms = bkg_rms_map[:, x] if bkg_rms_map is not None else np.full(h, 10.0)

            # Use a conservative threshold (e.g. 5 sigma) to prevent over-growing into noise
            stop_thresh = col_bkg + config.get("bleed_thresh_sigma", 5.0) * col_rms

            max_grow = config.get("bleed_grow_vertical", 50)

            _grow_bleed_up(sci_data, stop_thresh, x, y_min, max_grow, new_mask)
            _grow_bleed_down(sci_data, stop_thresh, h, x, y_max, max_grow, new_mask)

    # Horizontal dilation for safety (optional)
    h_dilation = config.get("bleed_grow_horizontal", 2)
    if h_dilation > 0:
        selem = np.ones((1, 2 * h_dilation + 1), dtype=bool)
        new_mask = scipy.ndimage.binary_dilation(new_mask, structure=selem)

    print(f"    Bleed trail growth added {np.sum(new_mask & ~sat_mask)} pixels.")
    return new_mask
