import hashlib
import json
import os
import re
import warnings

import numpy as np


def _get_global_median(flat_data):
    """Calculate a robust global median of valid pixels."""
    flat = np.asarray(flat_data).reshape(-1)
    chunk_size = 100000
    valid_count = sum(
        np.count_nonzero(flat[start : start + chunk_size] > 0) for start in range(0, flat.size, chunk_size)
    )
    step = max(1, valid_count // 100000)
    samples = []
    seen = 0
    for start in range(0, flat.size, chunk_size):
        valid = flat[start : start + chunk_size]
        valid = valid[valid > 0]
        sample_start = (-seen) % step
        samples.append(valid[sample_start::step])
        seen += valid.size
    global_med = np.nanmedian(np.concatenate(samples)) if valid_count > 0 else 1.0
    if not np.isfinite(global_med):
        global_med = 1.0
    return global_med


def _detect_bad_pixels_local(flat_data, config, global_med):
    """Detect bad pixels via local median deviation."""
    pixel_mask_bool = np.zeros(flat_data.shape, dtype=bool)
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

    except (ValueError, KeyError, TypeError) as e:
        print(f"  WARNING: Local pixel thresholding failed: {e}. Skipping.")
        pixel_mask_bool.fill(False)

    return pixel_mask_bool


def _detect_bad_columns_derivative(flat_data, config, global_med):
    """Return indices of bad columns found via horizontal derivatives."""
    bad_columns = np.empty(0, dtype=int)

    if not config.get("col_enable", True):
        print("  Bad column detection disabled in config.")
        return bad_columns

    print("  Detecting bad columns via horizontal derivatives...")
    try:
        # A completely dead column will have zero variance and low median,
        # but a partially bad column will just cause a sharp jump.
        # We take the derivative across columns (axis 1) of the column medians.

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            column_medians = np.array([np.nanmedian(flat_data[:, column]) for column in range(flat_data.shape[1])])

        # Mask entirely NaN/Inf columns immediately
        invalid_cols = ~np.isfinite(column_medians)
        num_invalid = np.count_nonzero(invalid_cols)
        bad_columns = np.flatnonzero(invalid_cols)

        if num_invalid < len(column_medians):
            # Calculate horizontal derivative (difference between adjacent columns)
            valid_medians = column_medians.copy()
            valid_medians[invalid_cols] = np.nanmedian(valid_medians)  # Patch for diff

            col_diffs = np.abs(np.diff(valid_medians))

            # Robust statistics of the differences
            med_diff = np.nanmedian(col_diffs)
            mad_diff = np.nanmedian(np.abs(col_diffs - med_diff)) * 1.4826
            if mad_diff == 0:
                mad_diff = np.nanstd(col_diffs)

            sigma_thresh = config.get("col_deriv_sigma", 10.0)
            thresh = med_diff + sigma_thresh * mad_diff

            # Columns where the jump from the neighbor is huge
            jump_edges = np.where(col_diffs > thresh)[0]
            radius = max(2, int(config.get("local_filter_size", 15)) // 2)
            local_baseline = np.empty_like(valid_medians)
            for column in range(len(valid_medians)):
                start = max(0, column - radius)
                stop = min(len(valid_medians), column + radius + 1)
                neighbors = np.concatenate((valid_medians[start:column], valid_medians[column + 1 : stop]))
                local_baseline[column] = np.nanmedian(neighbors) if neighbors.size else valid_medians[column]
            deviations = np.abs(valid_medians - local_baseline)
            jump_cols_idx = []
            for edge in jump_edges:
                left, right = edge, edge + 1
                jump_cols_idx.append(left if deviations[left] > deviations[right] else right)

            # Also catch columns that are completely dead (very close to 0)
            dead_thresh = config.get("col_dead_thresh", 0.1) * global_med
            dead_cols_idx = np.where(valid_medians < dead_thresh)[0]

            bad_cols_combined = np.unique(np.concatenate([jump_cols_idx, dead_cols_idx])).astype(int)
            bad_columns = np.union1d(bad_columns, bad_cols_combined)

            if len(bad_cols_combined) > 0:
                print(
                    f"    Found {len(bad_cols_combined)} bad columns (Deriv > {thresh:.3g} or Med < {dead_thresh:.3g})"
                )
                if num_invalid > 0:
                    print(f"    Plus {num_invalid} columns masked due to being mostly NaNs/Infs.")
            else:
                print("    No bad columns detected.")

    except Exception as e:
        print(f"  WARNING: Bad column detection failed: {e}. Skipping.")

    return bad_columns


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
        warnings.warn("Flat data contains no finite values. Masking the entire flat.", RuntimeWarning)
        return np.ones(flat_data.shape, dtype=bool)

    if using_unit_flat:
        print("  Skipping bad pixel/column detection (using unit flat).")
        return np.zeros(flat_data.shape, dtype=bool)

    global_med = _get_global_median(flat_data)

    # --- 1. Local Structural Pixel Thresholding ---
    pixel_mask_bool = _detect_bad_pixels_local(flat_data, config, global_med)

    # --- 2. Derivative-Based Column Detection ---
    bad_columns = _detect_bad_columns_derivative(flat_data, config, global_med)

    # --- 3. Combine Masks ---
    pixel_mask_bool[:, bad_columns] = True
    total_bad = np.count_nonzero(pixel_mask_bool)
    print(f"  Total BAD pixels/columns identified in flat: {total_bad}")

    return pixel_mask_bool


def compute_flat_bad_mask(flat_data, config, tile_size=1024):
    """Compute the full bad-pixel/column mask for one flat HDU, tiled.

    Local filtering uses halo-expanded tiles and writes only their cores, while
    column statistics are computed once over the full HDU. The result depends
    only on the flat HDU and the ``flat_masking`` settings;
    ``compute_flat_bad_mask_cached`` reuses it across every exposure that shares
    a flat instead of recomputing the expensive median filter each time.
    """
    bad_mask = np.zeros(flat_data.shape, dtype=bool)
    global_med = _get_global_median(flat_data)
    filter_size = config.get("local_filter_size", 15)
    try:
        halo = max(0, int(filter_size) // 2)
    except (TypeError, ValueError):
        halo = 0
    for y in range(0, flat_data.shape[0], tile_size):
        for x in range(0, flat_data.shape[1], tile_size):
            y_stop = min(y + tile_size, flat_data.shape[0])
            x_stop = min(x + tile_size, flat_data.shape[1])
            expanded_y = slice(max(0, y - halo), min(flat_data.shape[0], y_stop + halo))
            expanded_x = slice(max(0, x - halo), min(flat_data.shape[1], x_stop + halo))
            expanded = flat_data[expanded_y, expanded_x]
            expanded_mask = _detect_bad_pixels_local(expanded, config, global_med)
            core_y = slice(y - expanded_y.start, y_stop - expanded_y.start)
            core_x = slice(x - expanded_x.start, x_stop - expanded_x.start)
            bad_mask[y:y_stop, x:x_stop] = expanded_mask[core_y, core_x]
    bad_columns = _detect_bad_columns_derivative(flat_data, config, global_med)
    bad_mask[:, bad_columns] = True
    return bad_mask


def detect_dark_hot_pixels(dark_data, config):
    """Upper-tail hot pixels in a mostly healthy, bias-subtracted dark HDU."""
    if set(config) - {"hot_sigma"}:
        raise ValueError("dark_masking accepts only hot_sigma")
    hot_sigma = float(config.get("hot_sigma", 8.0))
    if isinstance(config.get("hot_sigma"), bool) or not np.isfinite(hot_sigma) or hot_sigma <= 0:
        raise ValueError("dark_masking.hot_sigma must be finite and positive")
    valid = np.isfinite(dark_data)
    if not np.any(valid):
        return ~valid
    values = dark_data[valid]
    # ponytail: one global median/MAD assumes a uniform dark pedestal and a
    # healthy majority; use per-amplifier/local baselines for structured darks.
    center = np.median(values)
    scatter = 1.4826 * np.median(np.abs(values - center))
    threshold = center + hot_sigma * scatter
    return (~valid) | (dark_data > threshold)


# Bump when the cache layout or the mask computation changes incompatibly.
_FLAT_BAD_CACHE_VERSION = 4
# flat_masking keys that configure the cache rather than the mask.
_FLAT_BAD_CACHE_CONTROL_KEYS = ("bad_mask_cache", "bad_mask_cache_dir")


def _flat_bad_settings(flat_cfg) -> str:
    """Canonical digest of the flat_masking settings that shape the mask.

    Every key is included (bar the cache controls) so that a retuned or newly
    added setting can never read a mask computed under the old settings.
    """
    if not isinstance(flat_cfg, dict):
        return "{}"
    payload = {k: v for k, v in flat_cfg.items() if k not in _FLAT_BAD_CACHE_CONTROL_KEYS}
    try:
        return json.dumps(payload, sort_keys=True, default=str)
    except (TypeError, ValueError):  # pragma: no cover - exotic config values
        return str(sorted((str(k), str(v)) for k, v in payload.items()))


def flat_bad_mask_cache_file(flat_cfg, flat_path, hdu_index, shape, tile_size):
    """Cache file for one flat HDU, or None when caching does not apply.

    The identity covers the flat's absolute path, byte size and nanosecond
    mtime, the HDU index, its shape and the flat_masking settings, so a replaced
    flat or retuned masking cannot read a stale mask. ``tile_size`` remains an
    accepted argument for compatibility but does not affect the tile-invariant
    mask identity.
    """
    if not isinstance(flat_cfg, dict) or not flat_cfg.get("bad_mask_cache", True):
        return None
    if not flat_path:
        return None
    try:
        stat = os.stat(str(flat_path))
    except OSError:
        return None
    cache_dir = flat_cfg.get("bad_mask_cache_dir")
    if not cache_dir:
        cache_dir = os.path.join(os.path.dirname(os.path.abspath(str(flat_path))), ".weightmask_cache")
    identity = "|".join(
        (
            f"v{_FLAT_BAD_CACHE_VERSION}",
            os.path.abspath(str(flat_path)),
            str(stat.st_size),
            str(stat.st_mtime_ns),
            str(hdu_index),
            "x".join(str(int(d)) for d in shape),
            _flat_bad_settings(flat_cfg),
        )
    )
    digest = hashlib.sha256(identity.encode()).hexdigest()[:32]
    return os.path.join(str(cache_dir), f"flatbad_{digest}.npy")


def compute_flat_bad_mask_cached(flat_data, flat_cfg, tile_size=1024, *, flat_path=None, hdu_index=None):
    """``compute_flat_bad_mask``, reused across exposures that share a flat.

    One flat HDU's mask costs ~20 s on a 9.8 Mpix CCD, yet a survey that
    processes N exposures through one flat computes each HDU's mask N times.
    Caching it on disk (next to the flat, or in ``flat_masking.bad_mask_cache_dir``)
    turns that into one computation per flat HDU. Any cache miss, disabled
    cache or I/O error falls back to the plain computation, so products never
    depend on the cache being present or writable.
    """
    cache_file = flat_bad_mask_cache_file(flat_cfg, flat_path, hdu_index, getattr(flat_data, "shape", ()), tile_size)
    if cache_file is None:
        return compute_flat_bad_mask(flat_data, flat_cfg, tile_size)
    if os.path.exists(cache_file):
        try:
            cached = np.load(cache_file, allow_pickle=False)
            if cached.dtype == np.bool_ and cached.shape == flat_data.shape:
                print(f"    Reusing cached flat bad-pixel mask for HDU {hdu_index} ({cache_file}).")
                return cached
        except (OSError, ValueError, EOFError):
            # Unreadable, truncated or zero-byte entry: recompute and overwrite
            # it. ``EOFError`` is what numpy raises for an empty file, which is
            # not an ``OSError`` subclass, so it has to be named explicitly.
            pass
    bad_mask = compute_flat_bad_mask(flat_data, flat_cfg, tile_size)
    try:
        directory = os.path.dirname(cache_file)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp_path = f"{cache_file}.tmp{os.getpid()}"
        with open(tmp_path, "wb") as handle:
            np.save(handle, bad_mask)
        os.replace(tmp_path, cache_file)  # atomic: concurrent readers see one or the other
    except OSError as exc:
        # Results are still correct, but the ~20 s/HDU median filter will be
        # recomputed on every run from here on. The usual cause is a read-only
        # mount (VOSpace), where the cache cannot live next to the flat, so
        # say so instead of failing silently.
        print(
            f"    WARNING: could not cache the flat bad-pixel mask ({exc}). "
            f"Set flat_masking.bad_mask_cache_dir to a writable path, or expect this "
            f"computation to repeat on every run."
        )
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
