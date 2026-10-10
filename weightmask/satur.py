import numpy as np
import scipy.ndimage
from scipy.signal import find_peaks

_MIN_CLUMP_PIXELS = 20


def _tail_histogram(values, edges):
    """Counts per bin, identical to ``np.histogram(values, bins=edges)``.

    ``np.histogram`` with explicit bin edges sorts the entire input, yet
    discards everything outside ``[edges[0], edges[-1]]``. On a real CCD only a
    few hundred of ~10^7 pixels lie in the saturation tail, so restrict to the
    tail first and bin only that. Counts are integers, so there is no
    floating-point summation order to preserve: the result is exact, not
    approximate.
    """
    if not np.all(np.diff(edges) > 0):
        # Degenerate edges: let numpy raise its own monotonicity error so the
        # failure mode for malformed config is unchanged.
        counts, _ = np.histogram(values, bins=edges)
        return counts, edges
    n_bins = len(edges) - 1
    lo = edges[0]
    hi = edges[-1]
    tail = values[(values >= lo) & (values <= hi)]
    if tail.size == 0:
        return np.zeros(n_bins, dtype=np.intp), edges
    idx = np.searchsorted(edges, tail, side="right") - 1
    np.clip(idx, 0, n_bins - 1, out=idx)
    return np.bincount(idx, minlength=n_bins).astype(np.intp, copy=False), edges


def estimate_saturation_robust_clump(data, min_adu=None, max_adu=None, finite_data=None):
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
        if finite_data is None:
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

            # Subsample large arrays before calculating global robust statistics
            step = max(1, finite_data.size // 100000)
            sampled_data = finite_data[::step]

            p99 = np.percentile(sampled_data, 99)
            p99_9 = np.percentile(sampled_data, 99.9)

            # Heuristic: start halfway between the 99th percentile and max,
            # or 80% of the 99.9th percentile, whichever is more conservative.
            auto_min_adu = min(p99 + (p_max - p99) * 0.3, p99_9 * 0.8)
            auto_max_adu = p_max * 1.01  # Allow the histogram analysis to cover the entire tail up to max ADU

            min_adu = min_adu if min_adu is not None else auto_min_adu
            max_adu = max_adu if max_adu is not None else auto_max_adu

            # Absolute sanity check: if max is small, we shouldn't be looking for saturation
            if p_max < 10000:
                print(f"  Robust Clump: Max ADU ({p_max:.1f}) is very low. Assuming no saturation.")
                return None

            if max_adu <= min_adu:
                max_adu = min_adu + 100.0

            print(f"  Auto-determined histogram range: [{min_adu:.1f}, {max_adu:.1f}] ADU")

        print(f"  Attempting robust clump analysis: range=[{min_adu:.1f}, {max_adu:.1f}]")

        # Bin the extreme tail into ~100 bins.
        # This prevents the histogram from dissolving into noise for smeared clumps.
        bins = np.linspace(min_adu, max_adu, 100)
        counts, bin_edges = _tail_histogram(finite_data, bins)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        if np.sum(counts) < _MIN_CLUMP_PIXELS:
            print("  Robust Clump: Not enough pixels in extreme tail (empty field). No saturation detected.")
            return None

        # Smooth the histogram heavily to find the macro-structure (the clump)
        smoothed_counts = scipy.ndimage.gaussian_filter1d(counts.astype(float), sigma=2.0)

        # Find peaks.
        # Expected prominence is at least a few percent of the max smoothed count in this tail.
        min_prominence = max(2.0, np.max(smoothed_counts) * 0.05)

        peaks, properties = find_peaks(smoothed_counts, prominence=min_prominence)

        if len(peaks) == 0:
            print("  Robust Clump: Smooth intensity tail with no anomalous clumps. No saturation detected.")
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
        print(f"  Robust Clump: Setting threshold near the base of the clump: {estimated_level:.1f} ADU.")

        return float(estimated_level)

    except Exception as e:
        print(f"  Robust Clump analysis failed with error: {e}")
        return None


def _get_saturation_from_header(sci_hdr, header_keyword):
    """Attempt to extract saturation level from the header (str-or-list keyword)."""
    keys = header_keyword if isinstance(header_keyword, (list, tuple)) else [header_keyword]
    keys = [k for k in keys if isinstance(k, str) and k]
    if not keys or sci_hdr is None:
        print(f"  Header advisory unavailable (keyword '{header_keyword}' missing or not specified).")
        return None
    for _k in keys:
        try:
            if _k not in sci_hdr:
                continue
            saturation_level = float(sci_hdr[_k])
            print(f"  Header advisory from keyword '{_k}': {saturation_level:.1f} ADU.")
            return saturation_level
        except (ValueError, TypeError, KeyError):
            print(f"  Header advisory failed (parse error for keyword '{_k}').")
            continue
    print(f"  Header advisory unavailable (keywords {keys} missing).")
    return None


def _estimate_effective_full_scale(sci_data, sci_hdr, config, header_keyword, finite_data=None):
    """Estimate a plausible upper-scale bound for guarded saturation detection."""
    hist_params = config.get("histogram_params", {})
    candidates = []

    explicit = config.get("effective_full_scale")
    if explicit is not None:
        candidates.append(float(explicit))

    if hist_params.get("hist_max_adu") is not None:
        candidates.append(float(hist_params["hist_max_adu"]))

    fallback_level = config.get("fallback_level", 65535.0)
    candidates.append(float(fallback_level))

    advisory = _get_saturation_from_header(sci_hdr, header_keyword)
    if advisory is not None and advisory > 0:
        candidates.append(float(advisory))

    if np.issubdtype(sci_data.dtype, np.integer):
        candidates.append(float(np.iinfo(sci_data.dtype).max))

    if finite_data is None:
        finite_data = sci_data[np.isfinite(sci_data)]
    data_max = float(np.max(finite_data)) if finite_data.size > 0 else 0.0
    positive = [value for value in candidates if np.isfinite(value) and value > 0]
    if not positive:
        return max(data_max, 65535.0), advisory

    # Prefer the smallest plausible scale that still safely clears the current data range.
    plausible = [value for value in positive if value >= max(data_max, 1.0) * 0.75]
    effective = min(plausible) if plausible else max(positive)
    return float(effective), advisory


def _estimate_plateau_tail(sci_data, effective_full_scale, config, finite_data=None):
    """Return the highest qualifying repeated upper-tail level and its multiplicity."""
    hist_params = config.get("histogram_params", {})
    min_tail_pixels = hist_params.get("min_tail_pixels", 8)
    if (
        isinstance(min_tail_pixels, (bool, np.bool_))
        or not isinstance(min_tail_pixels, (int, np.integer))
        or min_tail_pixels <= 0
    ):
        raise ValueError("saturation.histogram_params.min_tail_pixels must be a positive integer")
    min_tail_pixels = int(min_tail_pixels)

    if finite_data is None:
        finite_data = sci_data[np.isfinite(sci_data)]
    if finite_data.size == 0:
        return None, 0

    guard_fraction = float(hist_params.get("guard_fraction", 0.75))
    lower_guard = guard_fraction * effective_full_scale

    if np.max(finite_data) < lower_guard:
        return None, 0

    tail = finite_data[finite_data >= lower_guard]
    if tail.size < min_tail_pixels:
        return None, 0

    levels, counts = np.unique(tail, return_counts=True)
    qualifying = np.flatnonzero(counts >= min_tail_pixels)
    if qualifying.size == 0:
        return None, 0
    plateau_index = int(qualifying[-1])
    return float(levels[plateau_index]), int(counts[plateau_index])


def _choose_saturation_level(hist_level, plateau_level, plateau_support, effective_full_scale, advisory, config):
    """Pick the final saturation level using guarded histogram logic."""
    hist_params = config.get("histogram_params", {})
    guard_fraction = float(hist_params.get("guard_fraction", 0.75))
    upper_factor = float(hist_params.get("max_upper_factor", 1.05))
    lower_guard = guard_fraction * effective_full_scale
    upper_guard = upper_factor * effective_full_scale

    if hist_level is not None and lower_guard <= hist_level <= upper_guard:
        return float(hist_level), "histogram (guarded)"

    if (
        plateau_level is not None
        and plateau_support >= _MIN_CLUMP_PIXELS
        and lower_guard <= plateau_level <= upper_guard
    ):
        return float(plateau_level), "plateau-tail fallback"

    if advisory is not None and lower_guard <= advisory <= upper_guard:
        return float(advisory), "header advisory fallback"

    if plateau_level is not None and lower_guard <= plateau_level <= upper_guard:
        return float(plateau_level), "plateau-tail fallback"

    return float(effective_full_scale), "default guarded fallback"


def _saturation_for_region(sci_data, config, effective_full_scale, advisory, finite_data=None):
    """Estimate the full-frame saturation level."""
    hist_params = config.get("histogram_params", {})
    hist_level = estimate_saturation_robust_clump(
        sci_data,
        min_adu=hist_params.get("hist_min_adu"),
        max_adu=hist_params.get("hist_max_adu"),
        finite_data=finite_data,
    )
    plateau_level, plateau_support = _estimate_plateau_tail(
        sci_data, effective_full_scale, config, finite_data=finite_data
    )
    return _choose_saturation_level(hist_level, plateau_level, plateau_support, effective_full_scale, advisory, config)


def detect_saturated_pixels(sci_data, sci_hdr, config):
    """
    Detect saturated pixels using the guarded histogram cascade.

    Args:
        sci_data (ndarray): Science image data array (float32 recommended).
        sci_hdr (fits.Header): Science image header.
        config (dict): Configuration dictionary for saturation detection.

    Returns:
        tuple: (saturation_level, sat_method_used, sat_mask_bool)
               saturation_level (float): The determined saturation level in ADU.
               sat_method_used (str): Label from the guarded cascade
               ('histogram (guarded)', 'plateau-tail fallback',
               'header advisory fallback', or 'default guarded fallback').
               sat_mask_bool (ndarray): Boolean mask where True indicates saturated pixels.
    """
    if "method" in config:
        raise ValueError("saturation.method is unsupported; saturation uses the guarded histogram cascade")
    header_keyword = config.get("keyword")

    print("Attempting guarded histogram-based saturation detection...")
    finite_data = sci_data[np.isfinite(sci_data)]
    effective_full_scale, advisory = _estimate_effective_full_scale(
        sci_data, sci_hdr, config, header_keyword, finite_data=finite_data
    )
    saturation_level, sat_method_used = _saturation_for_region(
        sci_data, config, effective_full_scale, advisory, finite_data=finite_data
    )
    if sat_method_used == "default guarded fallback":
        print(f"  WARNING: Falling back to guarded full-scale saturation level: {saturation_level:.1f} ADU.")
    saturation_level = float(saturation_level)

    # Create the boolean mask for core saturated pixels
    # Handle potential NaNs/Infs in input data safely
    with np.errstate(invalid="ignore"):  # Suppress warnings from comparing with NaN/Inf
        sat_mask_bool = (sci_data >= saturation_level) & np.isfinite(sci_data)

    print(f"  Final saturation level used: {saturation_level:.1f} ADU (Method: {sat_method_used})")

    return saturation_level, sat_method_used, sat_mask_bool


def _grow_bleed(sci_data, stop_thresh, x, rows, new_mask):
    """Mask the leading run of ``rows`` whose flux stays above ``stop_thresh``.

    ``rows`` are the candidate pixels along column ``x`` in growth order, so
    the caller decides direction simply by how it builds the index array.
    Growth stops at the first sample at or below its threshold.
    """
    if rows.size == 0:
        return

    cond = sci_data[rows, x] > stop_thresh[rows]
    grown = int(np.argmin(cond)) if not np.all(cond) else int(cond.size)
    if grown > 0:
        new_mask[rows[:grown], x] = True


def _grow_bleed_up(sci_data, stop_thresh, x, y_min, max_grow, new_mask):
    """Grow bleed upward from the saturated segment starting at ``y_min``."""
    _grow_bleed(sci_data, stop_thresh, x, np.arange(y_min - 1, max(-1, y_min - 1 - max_grow), -1), new_mask)


def _grow_bleed_down(sci_data, stop_thresh, h, x, y_max, max_grow, new_mask):
    """Grow bleed downward from the saturated segment ending at ``y_max``."""
    _grow_bleed(sci_data, stop_thresh, x, np.arange(y_max + 1, min(h, y_max + 1 + max_grow)), new_mask)


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

    if len(sat_cols) > 0:
        min_x, max_x = np.min(sat_cols), np.max(sat_cols)
        sliced_sat_mask = sat_mask[:, min_x : max_x + 1]

        # Extract configuration lookups and pre-allocate fallback arrays outside the loop
        # to avoid redundant dict lookups and allocations inside the tight iteration over saturation segments
        bleed_thresh_sigma = config.get("bleed_thresh_sigma", 5.0)
        max_grow = config.get("bleed_grow_vertical", 50)
        fallback_bkg = np.zeros(h) if sky_map is None else None
        fallback_rms = np.full(h, 10.0) if bkg_rms_map is None else None

        # Find contiguous vertical segments in 2D to avoid 1D looping overhead
        struct = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        labeled_mask, num_features = scipy.ndimage.label(sliced_sat_mask, structure=struct)
        slices = scipy.ndimage.find_objects(labeled_mask)

        for s in slices:
            if s is None:
                continue
            sy, sx = s
            x = sx.start + min_x
            y_min = sy.start
            y_max = sy.stop - 1

            # Get background levels for this column
            col_bkg = sky_map[:, x] if sky_map is not None else fallback_bkg
            col_rms = bkg_rms_map[:, x] if bkg_rms_map is not None else fallback_rms

            # Use a conservative threshold (e.g. 5 sigma) to prevent over-growing into noise.
            # Where ``col_rms`` carries the ``inf`` sentinel (no RMS measurement)
            # ``stop_thresh`` becomes infinite and the trail simply does not grow
            # into that column -- the same inert reading the detection thresholds
            # use, so no substitute RMS is introduced here.
            stop_thresh = col_bkg + bleed_thresh_sigma * col_rms

            core_rows = y_max - y_min + 1
            if config.get("bleed_adaptive_cap", False):
                cap = core_rows * float(config.get("bleed_cap_core_factor", 3.0))
                cap = min(max(cap, float(config.get("bleed_cap_min", 20))), float(config.get("bleed_cap_max", 200)))
                seg_grow = int(min(cap, max_grow))
            else:
                seg_grow = max_grow
            _grow_bleed_up(sci_data, stop_thresh, x, y_min, seg_grow, new_mask)
            _grow_bleed_down(sci_data, stop_thresh, h, x, y_max, seg_grow, new_mask)

    # Horizontal dilation for safety (optional)
    h_dilation = config.get("bleed_grow_horizontal", 2)
    if h_dilation > 0:
        selem = np.ones((1, 2 * h_dilation + 1), dtype=bool)
        new_mask = scipy.ndimage.binary_dilation(new_mask, structure=selem)

    print(f"    Bleed trail growth added {np.count_nonzero(new_mask & ~sat_mask)} pixels.")
    return new_mask
