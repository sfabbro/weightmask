import numpy as np
import warnings

def estimate_saturation_from_histogram_stable_peak(data, min_adu=None, max_adu=None, num_iterations=20):
    """
    Estimates saturation level from histogram by finding a stable peak below the max,
    inspired by the logic in fitssatur.c.

    Args:
        data (ndarray): Image data array (should be float).
        min_adu (float, optional): Lower bound ADU value for histogram analysis. If None, auto-determined.
        max_adu (float, optional): Upper bound ADU value for histogram analysis. If None, auto-determined.
        num_iterations (int): Number of iterations for zeroing out histogram top end.

    Returns:
        float or None: Estimated saturation level, or None if not found
    """
    try:
        # Filter out infinities and NaNs for percentile/max calculations
        finite_data = data[np.isfinite(data)]
        if finite_data.size == 0:
            print("  Histogram (Stable Peak): No finite data found.")
            return None

        p_max = np.max(finite_data)

        # Auto-determine min_adu and max_adu if not provided
        if min_adu is None or max_adu is None:
            # Use percentiles to set a reasonable range, avoiding extreme outliers influencing max_adu too much
            p95 = np.percentile(finite_data, 95)
            p99_9 = np.percentile(finite_data, 99.9)

            auto_min_adu = p95 * 0.8 # Start below the main distribution tail
            auto_max_adu = min(p_max * 1.01, p99_9 * 1.2) # Cap near max value or slightly above 99.9th

            min_adu = min_adu if min_adu is not None else auto_min_adu
            max_adu = max_adu if max_adu is not None else auto_max_adu

            # Ensure max_adu is slightly larger than min_adu
            if max_adu <= min_adu:
                max_adu = min_adu + 100 # Add a small buffer
            print(f"  Auto-determined histogram range: [{min_adu:.1f},{max_adu:.1f}] ADU")

    except Exception as e:
        print(f"  Parameter auto-determination failed: {e}")
        # Fall back to reasonable defaults if auto-determination fails
        min_adu = min_adu if min_adu is not None else 30000.0
        max_adu = max_adu if max_adu is not None else 65535.0
        if max_adu <= min_adu: max_adu = min_adu + 100.0


    print(f"  Attempting histogram analysis (Stable Peak): range=[{min_adu:.1f},{max_adu:.1f}]")

    try:
        # Use integer bins for direct comparison
        bins = np.arange(int(np.floor(min_adu)), int(np.ceil(max_adu)) + 2)
        counts, bin_edges = np.histogram(finite_data, bins=bins)

        if np.sum(counts) < 10: # Need some minimum number of pixels in range
            print("  Histogram (Stable Peak): Not enough pixels in analysis range.")
            return None

        # Find the initial peak (highest bin with counts)
        valid_counts_indices = np.where(counts > 0)[0]
        if len(valid_counts_indices) == 0:
             print("  Histogram (Stable Peak): No counts found in analysis range.")
             return None
        initial_peak_idx = valid_counts_indices[np.argmax(counts[valid_counts_indices])]
        initial_peak_val = bin_edges[initial_peak_idx]
        print(f"  Initial histogram peak value: {initial_peak_val:.1f} ADU (counts: {counts[initial_peak_idx]})")

        # Iteratively zero out top bins and find the peak, looking for stability
        longest_stable_run = 0
        stable_run_start_idx = -1
        current_stable_run = 0
        last_peak_idx = -1

        # Start zeroing from just below max_adu down towards the initial peak value
        # This avoids issues if the initial peak *is* the saturation level
        zero_start_val = max_adu

        for i in range(num_iterations):
            # Calculate the value to zero above for this iteration
            # Linearly decrease the zeroing threshold from max_adu towards initial_peak_val
            zero_above_val = zero_start_val - (zero_start_val - initial_peak_val) * (i + 1) / num_iterations

            # Ensure zero_above_val doesn't go below the start of our bins
            zero_above_val = max(zero_above_val, bin_edges[0])

            temp_counts = counts.copy()
            zero_above_idx = np.searchsorted(bin_edges, zero_above_val, side='left')
            if zero_above_idx < len(temp_counts):
                 temp_counts[zero_above_idx:] = 0
            # Also ensure the initial peak itself isn't considered in stability check unless it persists
            # temp_counts[initial_peak_idx] = 0 # Optional: Force check below initial peak

            valid_indices = np.where(temp_counts > 0)[0]
            if len(valid_indices) == 0:
                # print(f"  Iter {i+1}: No peak found after zeroing above {zero_above_val:.1f}")
                break # Stop if histogram becomes empty

            current_peak_idx = valid_indices[np.argmax(temp_counts[valid_indices])]

            # Check stability (ignoring the absolute initial peak unless it's the only one left)
            if current_peak_idx == last_peak_idx and current_peak_idx != initial_peak_idx:
                current_stable_run += 1
            else:
                # New peak or first iteration
                if current_stable_run > longest_stable_run:
                    longest_stable_run = current_stable_run
                    stable_run_start_idx = last_peak_idx # Store the start index of the longest run

                # Reset for the new peak
                current_stable_run = 1
                last_peak_idx = current_peak_idx

            # print(f"  Iter {i+1}: Zeroed above {zero_above_val:.1f}, Peak Idx: {current_peak_idx} (Val: {bin_edges[current_peak_idx]:.1f}), Counts: {temp_counts[current_peak_idx]}, Stable run: {current_stable_run}")


        # Check if the last run was the longest
        if current_stable_run > longest_stable_run and last_peak_idx != initial_peak_idx:
            longest_stable_run = current_stable_run
            stable_run_start_idx = last_peak_idx

        # Require a minimum run length to be considered stable (e.g., 3 iterations)
        min_stable_run = 3
        if longest_stable_run >= min_stable_run and stable_run_start_idx != -1:
            estimated_level = bin_edges[stable_run_start_idx] # The start value of the stable plateau
            print(f"  Histogram (Stable Peak): Found stable plateau at {estimated_level:.1f} ADU (run length {longest_stable_run}).")
            # Add a small buffer (e.g., half a bin width)
            return float(estimated_level + 0.5 * np.mean(np.diff(bin_edges)))
        elif initial_peak_idx != -1:
             # Fallback if no stable run found: return the initial peak value.
             # This might happen if saturation is very sharp or data is noisy.
             estimated_level = bin_edges[initial_peak_idx]
             print(f"  Histogram (Stable Peak): No stable plateau found (longest run {longest_stable_run}). Falling back to initial peak: {estimated_level:.1f} ADU.")
             return float(estimated_level + 0.5 * np.mean(np.diff(bin_edges)))
        else:
            print("  Histogram (Stable Peak): Could not identify a stable peak.")
            return None

    except Exception as e:
        import traceback
        print(f"  Histogram (Stable Peak) analysis failed with error: {e}")
        print(traceback.format_exc())
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
    sat_method_used = 'none'
    hist_params = config.get('histogram_params', {}) # Get sub-dictionary for histogram params

    # Ensure data is float for calculations
    if not np.issubdtype(sci_data.dtype, np.floating):
        warnings.warn("Science data is not float, casting to float32 for saturation check.", UserWarning)
        sci_data = sci_data.astype(np.float32)

    if config.get('method', 'histogram') == 'histogram':
        print("Attempting saturation detection via histogram (Stable Peak method)...")
        # Pass specific histogram parameters if they exist in the config
        saturation_level = estimate_saturation_from_histogram_stable_peak(
            sci_data,
            min_adu=hist_params.get('hist_min_adu'), # Use .get for safety
            max_adu=hist_params.get('hist_max_adu'),
            num_iterations=hist_params.get('hist_iterations', 20) # Add num_iterations to config?
        )

        if saturation_level is not None:
            sat_method_used = 'histogram (stable peak)'
        else:
            print("  Histogram (Stable Peak) failed. Trying header fallback...")
            # Fallback to header keyword
            header_keyword = config.get('keyword')
            if header_keyword and header_keyword in sci_hdr:
                try:
                    saturation_level = float(sci_hdr[header_keyword])
                    sat_method_used = 'header (fallback)'
                    print(f"  Using saturation from header keyword '{header_keyword}': {saturation_level:.1f} ADU.")
                except (ValueError, TypeError):
                    print(f"  Header fallback failed (parse error for keyword '{header_keyword}').")
                    saturation_level = None # Ensure it's None if parse fails
            else:
                print(f"  Header fallback failed (keyword '{header_keyword}' missing or not specified).")

    elif config.get('method') == 'header':
        print("Attempting saturation detection via header keyword...")
        header_keyword = config.get('keyword')
        if header_keyword and header_keyword in sci_hdr:
            try:
                saturation_level = float(sci_hdr[header_keyword])
                sat_method_used = 'header'
                print(f"  Using saturation from header keyword '{header_keyword}': {saturation_level:.1f} ADU.")
            except (ValueError, TypeError):
                print(f"  Header method failed (parse error for keyword '{header_keyword}').")
                saturation_level = None
        else:
             print(f"  Header method failed (keyword '{header_keyword}' missing or not specified).")

    # Final fallback to default value if all methods fail
    if saturation_level is None:
        fallback_level = config.get('fallback_level', 65535.0) # Default fallback if not in config
        saturation_level = fallback_level
        sat_method_used = 'default fallback'
        print(f"  WARNING: Using fallback saturation level: {saturation_level:.1f} ADU.")

    # Ensure saturation_level is a float before comparison
    saturation_level = float(saturation_level)

    # Create the boolean mask
    # Handle potential NaNs/Infs in input data safely
    with np.errstate(invalid='ignore'): # Suppress warnings from comparing with NaN/Inf
        sat_mask_bool = (sci_data >= saturation_level) & np.isfinite(sci_data)

    print(f"  Final saturation level used: {saturation_level:.1f} ADU (Method: {sat_method_used})")

    return saturation_level, sat_method_used, sat_mask_bool