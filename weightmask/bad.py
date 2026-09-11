import re
import warnings

import numpy as np


def _get_global_median(flat_data):
    """Calculate a robust global median of valid pixels."""
    valid_flat = flat_data[flat_data > 0]
    step = max(1, valid_flat.size // 100000)
    global_med = np.nanmedian(valid_flat[::step]) if valid_flat.size > 0 else 1.0
    if not np.isfinite(global_med):
        global_med = 1.0
    return global_med


def _detect_bad_pixels_local(flat_data, config, global_med):
    """Detect bad pixels via local median deviation."""
    pixel_mask_bool = np.zeros(flat_data.shape, dtype=bool)
    print("  Detecting bad pixels via local median deviation...")
    try:
        from scipy.ndimage import median_filter

        filter_size = config.get("local_filter_size", config.get("filter_size", 15))
        local_low_thresh = config.get("local_low_thresh", config.get("low_thresh", 0.5))
        local_high_thresh = config.get("local_high_thresh", config.get("high_thresh", 2.0))

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)

            # Create a heavily smoothed version of the flat to serve as the "true" illumination model.
            # Replace NaNs/Infs with median so they don't corrupt the filter
            clean_flat = flat_data.copy()

            clean_flat[~np.isfinite(clean_flat)] = global_med

            smoothed_flat = median_filter(clean_flat, size=filter_size)

            # Avoid division by zero
            smoothed_flat[smoothed_flat <= 0] = 1e-6

            # Calculate the ratio between the raw flat and the local smoothed model
            ratio = flat_data / smoothed_flat

        # Flag pixels where the ratio is outside the acceptable bounds
        pixel_mask_bool = (
            (ratio <= local_low_thresh) | (ratio >= local_high_thresh) | (flat_data <= 0) | (~np.isfinite(flat_data))
        )
        print(
            f"    Found {np.count_nonzero(pixel_mask_bool)} bad pixels (Ratio < {local_low_thresh:.2f} or > {local_high_thresh:.2f})."
        )

    except Exception as e:
        print(f"  WARNING: Local pixel thresholding failed: {e}. Skipping.")
        pixel_mask_bool.fill(False)

    return pixel_mask_bool


def _detect_bad_columns_derivative(flat_data, config, global_med):
    """Detect bad columns via horizontal derivatives."""
    column_mask_bool = np.zeros(flat_data.shape, dtype=bool)

    if not config.get("col_enable", True):
        print("  Bad column detection disabled in config.")
        return column_mask_bool

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
        num_invalid = np.count_nonzero(invalid_cols)

        if num_invalid < len(column_medians):
            # Calculate horizontal derivative (difference between adjacent columns)
            valid_medians = column_medians.copy()
            valid_medians[invalid_cols] = np.nanmedian(valid_medians)  # Patch for diff

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
            dead_thresh = config.get("col_dead_thresh", config.get("col_median_dev_factor", 0.1)) * global_med
            dead_cols_idx = np.where(valid_medians < dead_thresh)[0]

            bad_cols_combined = np.unique(np.concatenate([jump_cols_idx, dead_cols_idx]))

            if len(bad_cols_combined) > 0:
                column_mask_bool[:, bad_cols_combined] = True
                print(
                    f"    Found {len(bad_cols_combined)} bad columns (Deriv > {thresh:.3g} or Med < {dead_thresh:.3g})"
                )
                if num_invalid > 0:
                    print(f"    Plus {num_invalid} columns masked due to being mostly NaNs/Infs.")
            else:
                print("    No bad columns detected.")

    except Exception as e:
        import traceback

        print(f"  WARNING: Bad column detection failed: {e}. Skipping.")
        print(traceback.format_exc())

    return column_mask_bool


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
        warnings.warn("Flat data contains no finite values. Returning empty mask.", RuntimeWarning)
        return np.zeros(flat_data.shape, dtype=bool)

    if using_unit_flat:
        print("  Skipping bad pixel/column detection (using unit flat).")
        return np.zeros(flat_data.shape, dtype=bool)

    global_med = _get_global_median(flat_data)

    # --- 1. Local Structural Pixel Thresholding ---
    pixel_mask_bool = _detect_bad_pixels_local(flat_data, config, global_med)

    # --- 2. Derivative-Based Column Detection ---
    column_mask_bool = _detect_bad_columns_derivative(flat_data, config, global_med)

    # --- 3. Combine Masks ---
    final_mask_bool = pixel_mask_bool | column_mask_bool
    total_bad = np.count_nonzero(final_mask_bool)
    print(f"  Total BAD pixels/columns identified in flat: {total_bad}")

    return final_mask_bool


def compute_flat_bad_mask(flat_data, config, tile_size=1024):
    """Compute the full bad-pixel/column mask for one flat HDU, tiled.

    Runs ``detect_bad_pixels`` over the same 1024x1024 tiles used by
    ``process_image`` and ORs the results into a full-HDU mask. The result
    depends only on the flat (and config), so callers cache it per flat and
    reuse it across every exposure that shares that flat instead of
    recomputing the expensive 15x15 median filter each time.
    """
    bad_mask = np.zeros(flat_data.shape, dtype=bool)
    for y in range(0, flat_data.shape[0], tile_size):
        for x in range(0, flat_data.shape[1], tile_size):
            tile = (slice(y, y + tile_size), slice(x, x + tile_size))
            flat_tile = flat_data[tile]
            if not np.isfinite(flat_tile).any():
                continue
            bad_mask[tile] = detect_bad_pixels(flat_tile, config, using_unit_flat=False)
    return bad_mask


def _parse_section(section):
    """Parse a FITS '[x1:x2,y1:y2]' section string to 0-based exclusive (r0, r1, c0, c1); None on failure."""
    if not isinstance(section, str):
        return None
    m = re.match(r"\[\s*(-?\d+)\s*:\s*(-?\d+)\s*,\s*(-?\d+)\s*:\s*(-?\d+)\s*\]", section.strip())
    if not m:
        return None
    x1, x2, y1, y2 = (int(v) for v in m.groups())
    if x1 == 0 or x2 == 0 or y1 == 0 or y2 == 0:
        return None
    return (min(y1, y2) - 1, max(y1, y2), min(x1, x2) - 1, max(x1, x2))


def detect_non_illuminated(shape, header):
    """Complement of the header DATASEC illuminated rectangle as a bool mask.

    MegaCam frames carry per-HDU overscan/prescan strips outside DATASEC
    (e.g. 2112x4644 with DATASEC '[33:2080,1:4612]'). The parsed rectangle is
    clamped to the image and must overlap CCDSIZE when present; otherwise the
    header is treated as non-MegaCam and an all-False mask is returned, i.e.
    no behavior change. Malformed or missing sections never raise.
    """
    h, w = shape
    try:
        get = header.get if hasattr(header, "get") else (lambda k, d=None: header[k] if k in header else d)
        illum = _parse_section(get("DATASEC", None))
        if illum is None:
            return np.zeros(shape, dtype=bool)
        r0, r1, c0, c1 = illum
        r0, r1 = max(r0, 0), min(r1, h)
        c0, c1 = max(c0, 0), min(c1, w)
        if r1 <= r0 or c1 <= c0:
            return np.zeros(shape, dtype=bool)
        ccd = _parse_section(get("CCDSIZE", None))
        if ccd is not None:
            cr0, cr1, cc0, cc1 = ccd
            if r0 >= cr1 or r1 <= cr0 or c0 >= cc1 or c1 <= cc0:
                return np.zeros(shape, dtype=bool)
        mask = np.ones(shape, dtype=bool)
        mask[r0:r1, c0:c1] = False
    except Exception:
        return np.zeros(shape, dtype=bool)
    return mask
