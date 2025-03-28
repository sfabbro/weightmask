import numpy as np
import warnings

def detect_bad_pixels(flat_data, config, using_unit_flat=False):
    """
    Detect bad pixels and columns in the flat field.

    Combines thresholding on individual pixel values with detection
    of columns showing anomalous median values or low variance.

    Args:
        flat_data (ndarray): Flat field data array.
        config (dict): Configuration dictionary for flat masking.
                       Should contain 'low_thresh', 'high_thresh', and
                       optionally 'col_enable', 'col_low_var_factor',
                       'col_median_dev_factor'.
        using_unit_flat (bool): Whether a unit flat (all 1.0) is being used.

    Returns:
        ndarray: Boolean mask of bad pixels and columns (True = bad).
    """
    if not np.isfinite(flat_data).any():
        warnings.warn("Flat data contains no finite values. Returning empty mask.", RuntimeWarning)
        return np.zeros(flat_data.shape, dtype=bool)

    # --- 1. Individual Pixel Thresholding ---
    pixel_mask_bool = np.zeros(flat_data.shape, dtype=bool)
    if not using_unit_flat:
        print("  Detecting bad pixels via thresholds...")
        try:
            # Use nanmedian/nanmean for robustness against existing NaNs/Infs
            with warnings.catch_warnings(): # Suppress RuntimeWarning from empty slices
                warnings.simplefilter("ignore", category=RuntimeWarning)
                median_flat = np.nanmedian(flat_data[flat_data > 0]) # Avoid zero/negative values for median calc

            if not np.isfinite(median_flat) or median_flat <= 0:
                warnings.warn("Could not determine a valid median flat value (>0). Using 1.0. Pixel thresholding may be unreliable.", RuntimeWarning)
                median_flat = 1.0

            flat_low = config.get('low_thresh', 0.5) * median_flat
            flat_high = config.get('high_thresh', 2.0) * median_flat

            # Create mask for pixels outside acceptable range or non-finite
            pixel_mask_bool = (
                (flat_data <= flat_low) |
                (flat_data >= flat_high) |
                (flat_data <= 0) | # Explicitly mask non-positive
                (~np.isfinite(flat_data)) # Mask NaNs/Infs
            )
            print(f"    Found {np.sum(pixel_mask_bool)} bad pixels (Low<{flat_low:.3f}, High>{flat_high:.3f}).")

        except Exception as e:
            print(f"  WARNING: Pixel thresholding failed: {e}. Skipping.")
            pixel_mask_bool.fill(False) # Reset mask if calculation failed
    else:
         print("  Skipping individual pixel thresholding (using unit flat).")

    # --- 2. Bad Column Detection ---
    column_mask_bool = np.zeros(flat_data.shape, dtype=bool)
    if not using_unit_flat and config.get('col_enable', True): # Check if enabled
        print("  Detecting bad columns via statistics...")
        try:
            # Calculate global statistics again if needed (median_flat might be 1.0 from above)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                global_flat_median = np.nanmedian(flat_data)
                global_flat_var = np.nanvar(flat_data)

            if not np.isfinite(global_flat_median) or not np.isfinite(global_flat_var):
                 warnings.warn("Could not determine valid global flat median/variance. Skipping column detection.", RuntimeWarning)
            else:
                # Calculate column statistics (ignoring NaNs)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning) # Ignore warnings from all-NaN columns
                    column_medians = np.nanmedian(flat_data, axis=0)
                    column_vars = np.nanvar(flat_data, axis=0)

                # Get thresholds from config or use defaults
                low_var_factor = config.get('col_low_var_factor', 0.05) # Factor of global variance
                median_dev_factor = config.get('col_median_dev_factor', 0.1) # Factor of global median

                # Thresholds
                var_threshold = low_var_factor * global_flat_var
                median_deviation_threshold = median_dev_factor * abs(global_flat_median) # Use abs for safety

                # Identify bad columns
                # Condition 1: Column variance is significantly lower than global variance (potential dead column)
                # Condition 2: Column median deviates significantly from global median
                # Condition 3: Column median or variance calculation failed (all NaNs in column)
                bad_cols_idx = np.where(
                    (column_vars < var_threshold) |
                    (np.abs(column_medians - global_flat_median) > median_deviation_threshold) |
                    (~np.isfinite(column_medians)) |
                    (~np.isfinite(column_vars))
                )[0]

                if len(bad_cols_idx) > 0:
                    column_mask_bool[:, bad_cols_idx] = True
                    print(f"    Found {len(bad_cols_idx)} bad columns (Var<{var_threshold:.3g}, MedDev>{median_deviation_threshold:.3g}).")
                else:
                    print("    No bad columns detected.")

        except Exception as e:
             print(f"  WARNING: Bad column detection failed: {e}. Skipping.")
             # No need to reset column_mask_bool, it's already False

    elif not config.get('col_enable', True):
        print("  Bad column detection disabled in config.")

    # --- 3. Combine Masks ---
    final_mask_bool = pixel_mask_bool | column_mask_bool
    total_bad = np.sum(final_mask_bool)
    print(f"  Total BAD pixels/columns identified in flat: {total_bad}")

    return final_mask_bool