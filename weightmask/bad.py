import numpy as np
import warnings


def detect_bad_pixels(flat_data, config, using_unit_flat=False):
    """
    Detect bad pixels and columns in the flat field using local structural analysis.

    Instead of global thresholds which fail on vignetted fields, this applies
    a median filter to create a structural model of the flat, and flags pixels
    that deviate significantly from that model. Bad columns are found using
    horizontal derivatives to find sharp discontinuities.

    Args:
        flat_data (ndarray): Flat field data array.
        config (dict): Configuration dictionary for flat masking.
        using_unit_flat (bool): Whether a unit flat (all 1.0) is being used.

    Returns:
        ndarray: Boolean mask of bad pixels and columns (True = bad).
    """
    if not np.isfinite(flat_data).any():
        warnings.warn(
            "Flat data contains no finite values. Returning empty mask.", RuntimeWarning
        )
        return np.zeros(flat_data.shape, dtype=bool)

    pixel_mask_bool = np.zeros(flat_data.shape, dtype=bool)
    column_mask_bool = np.zeros(flat_data.shape, dtype=bool)

    if using_unit_flat:
        print("  Skipping bad pixel/column detection (using unit flat).")
        return pixel_mask_bool

    # --- 1. Local Structural Pixel Thresholding ---
    print("  Detecting bad pixels via local median deviation...")
    try:
        from scipy.ndimage import median_filter

        filter_size = config.get("local_filter_size", 15)
        local_low_thresh = config.get("local_low_thresh", 0.5)
        local_high_thresh = config.get("local_high_thresh", 2.0)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            # Create a heavily smoothed version of the flat to serve as the "true" illumination model.
            # Replace NaNs/Infs with median so they don't corrupt the filter
            clean_flat = flat_data.copy()

            # ⚡ Bolt: Subsample large arrays before calculating global robust statistics
            valid_flat = clean_flat[clean_flat > 0]
            step = max(1, valid_flat.size // 100000)
            global_med = (
                np.nanmedian(valid_flat[::step]) if valid_flat.size > 0 else 1.0
            )

            if not np.isfinite(global_med):
                global_med = 1.0
            clean_flat[~np.isfinite(clean_flat)] = global_med

            smoothed_flat = median_filter(clean_flat, size=filter_size)

            # Avoid division by zero
            smoothed_flat[smoothed_flat <= 0] = 1e-6

            # Calculate the ratio between the raw flat and the local smoothed model
            ratio = flat_data / smoothed_flat

        # Flag pixels where the ratio is outside the acceptable bounds
        pixel_mask_bool = (
            (ratio <= local_low_thresh)
            | (ratio >= local_high_thresh)
            | (flat_data <= 0)
            | (~np.isfinite(flat_data))
        )
        print(
            f"    Found {np.sum(pixel_mask_bool)} bad pixels (Ratio < {local_low_thresh:.2f} or > {local_high_thresh:.2f})."
        )

    except Exception as e:
        print(f"  WARNING: Local pixel thresholding failed: {e}. Skipping.")
        pixel_mask_bool.fill(False)

    # --- 2. Derivative-Based Column Detection ---
    if config.get("col_enable", True):
        print("  Detecting bad columns via horizontal derivatives...")
        try:
            # A completely dead column will have zero variance and low median,
            # but a partially bad column will just cause a sharp jump.
            # We take the derivative across columns (axis 1) of the column medians.

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                column_medians = np.nanmedian(flat_data, axis=0)

            # Mask entirely NaN/Inf columns immediately
            invalid_cols = ~np.isfinite(column_medians)
            column_mask_bool[:, invalid_cols] = True
            num_invalid = np.sum(invalid_cols)

            if num_invalid < len(column_medians):
                # Calculate horizontal derivative (difference between adjacent columns)
                valid_medians = column_medians.copy()
                valid_medians[invalid_cols] = np.nanmedian(
                    valid_medians
                )  # Patch for diff

                col_diffs = np.abs(np.diff(valid_medians, prepend=valid_medians[0]))

                # Robust statistics of the differences
                med_diff = np.nanmedian(col_diffs)
                mad_diff = np.nanmedian(np.abs(col_diffs - med_diff)) * 1.4826
                if mad_diff == 0:
                    mad_diff = np.nanstd(col_diffs)

                sigma_thresh = config.get("col_deriv_sigma", 10.0)
                thresh = med_diff + sigma_thresh * mad_diff

                # Columns where the jump from the neighbor is huge
                jump_cols_idx = np.where(col_diffs > thresh)[0]

                # Also catch columns that are completely dead (very close to 0)
                dead_thresh = config.get("col_dead_thresh", 0.1) * global_med
                dead_cols_idx = np.where(valid_medians < dead_thresh)[0]

                bad_cols_combined = np.unique(
                    np.concatenate([jump_cols_idx, dead_cols_idx])
                )

                if len(bad_cols_combined) > 0:
                    column_mask_bool[:, bad_cols_combined] = True
                    print(
                        f"    Found {len(bad_cols_combined)} bad columns (Deriv > {thresh:.3g} or Med < {dead_thresh:.3g})"
                    )
                    if num_invalid > 0:
                        print(
                            f"    Plus {num_invalid} columns masked due to being mostly NaNs/Infs."
                        )
                else:
                    print("    No bad columns detected.")

        except Exception as e:
            import traceback

            print(f"  WARNING: Bad column detection failed: {e}. Skipping.")
            print(traceback.format_exc())

    else:
        print("  Bad column detection disabled in config.")

    # --- 3. Combine Masks ---
    final_mask_bool = pixel_mask_bool | column_mask_bool
    total_bad = np.sum(final_mask_bool)
    print(f"  Total BAD pixels/columns identified in flat: {total_bad}")

    return final_mask_bool
