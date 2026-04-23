import warnings

import numpy as np
import sep
from astropy.stats import mad_std
from scipy import linalg
from scipy.ndimage import median_filter


def _estimate_global_sep(sci_data, mask):
    """Estimate a single global background using a single SEP box."""
    try:
        bkg = sep.Background(sci_data, mask=mask, bw=sci_data.shape[1], bh=sci_data.shape[0])
        if bkg.globalback == 0 and np.any(~mask):
            return None, None
        bkg_map = np.full(sci_data.shape, bkg.globalback, dtype=np.float32)
        bkg_rms_map = np.full(sci_data.shape, bkg.globalrms, dtype=np.float32)
        return bkg_map, bkg_rms_map
    except Exception as e:
        warnings.warn(
            f"Global SEP fallback failed: {e}. Trying robust median.",
            RuntimeWarning,
        )
        return None, None


def _check_and_fix_edge_artifacts(bkg_map, sci_data_shape):
    """Check for edge artifacts and apply smoothing if necessary."""
    edge_width = 50
    h, w = sci_data_shape
    if h <= 2 * edge_width or w <= 2 * edge_width:
        return bkg_map
    edge_regions = [
        bkg_map[:edge_width, :],
        bkg_map[-edge_width:, :],
        bkg_map[:, :edge_width],
        bkg_map[:, -edge_width:],
    ]
    center_median = np.median(bkg_map[edge_width:-edge_width, edge_width:-edge_width])
    for edge_region in edge_regions:
        edge_median = np.median(edge_region)
        if abs(edge_median - center_median) > 50:
            from scipy.ndimage import gaussian_filter

            bkg_map = gaussian_filter(bkg_map, sigma=2.0)
            print("    Applied Gaussian smoothing to reduce edge artifacts")
            break
    return bkg_map


def _estimate_sep_tiered(sci_data, mask, box_size, filter_size, max_box_size):
    """Estimate background using SEP with tiered retries."""
    current_box = box_size
    attempt = 0
    while current_box <= max_box_size:
        try:
            if attempt > 0:
                print(f"    Retrying SEP with larger box size: {current_box}")

            bkg = sep.Background(
                sci_data,
                mask=mask,
                bw=current_box,
                bh=current_box,
                fw=filter_size,
                fh=filter_size,
                maskthresh=0.0,
            )
            bkg_map = bkg.back()
            bkg_rms_map = bkg.rms()
            print(f"    SEP background global RMS: {bkg.globalrms:.3f} (box={current_box})")

            bkg_map = _check_and_fix_edge_artifacts(bkg_map, sci_data.shape)
            return bkg_map, bkg_rms_map

        except Exception as e:
            next_box = current_box * 2
            if next_box > max_box_size:
                warnings.warn(
                    f"All SEP background retries failed: {e}. Falling back to robust global median.",
                    RuntimeWarning,
                )
                break
            current_box = next_box
            attempt += 1
    return None, None


def _estimate_robust_median(sci_data, mask, method, config):
    """Estimate background using median filter or robust median fallback."""
    try:
        if method == "robust_median_fallback":
            print("    Using robust global median fallback.")
            kernel_size = 0
        else:
            default_size = max(15, int(min(sci_data.shape) / 20) // 2 * 2 + 1)
            kernel_size = config.get("median_kernel_size", default_size)
            print(f"    Using median filter with kernel size: {kernel_size}")

        if kernel_size > 0:
            bkg_map = median_filter(sci_data, size=kernel_size)
        else:
            valid_data = sci_data[~mask] if np.any(~mask) else sci_data
            step = max(1, valid_data.size // 100000)
            bkg_val = np.median(valid_data.ravel()[::step])
            bkg_map = np.full(sci_data.shape, bkg_val, dtype=np.float32)

        data_sub = sci_data - bkg_map
        valid_sub = data_sub[~mask] if np.any(~mask) else data_sub
        step_sub = max(1, valid_sub.size // 100000)
        global_rms = mad_std(valid_sub.ravel()[::step_sub], ignore_nan=True)
        bkg_rms_map = np.full(sci_data.shape, global_rms, dtype=np.float32)
        print(f"    Final RMS (mad_std): {global_rms:.3f}")
        return bkg_map, bkg_rms_map

    except Exception as e:
        warnings.warn(f"Final background fallback failed: {e}", RuntimeWarning)
        return None, None


def _estimate_smooth_surface(sci_data, mask, config):
    """Estimate a smooth low-order background surface for gradient-dominated fields."""
    valid = np.isfinite(sci_data) & (~mask)
    if np.sum(valid) < 100:
        return None, None

    y_idx, x_idx = np.indices(sci_data.shape, dtype=np.float32)
    sample_y = y_idx[valid]
    sample_x = x_idx[valid]
    sample_v = sci_data[valid].astype(np.float32)
    step = max(1, sample_v.size // 50000)
    sample_y = sample_y[::step]
    sample_x = sample_x[::step]
    sample_v = sample_v[::step]

    x_norm = (sample_x - 0.5 * (sci_data.shape[1] - 1)) / max(float(sci_data.shape[1]), 1.0)
    y_norm = (sample_y - 0.5 * (sci_data.shape[0] - 1)) / max(float(sci_data.shape[0]), 1.0)
    design = np.column_stack(
        [
            np.ones_like(x_norm),
            x_norm,
            y_norm,
            x_norm * y_norm,
            x_norm**2,
            y_norm**2,
        ]
    )
    try:
        coeffs, *_ = linalg.lstsq(design, sample_v)
    except Exception as e:
        warnings.warn(f"Smooth-surface background fit failed: {e}", RuntimeWarning)
        return None, None

    full_x = (x_idx - 0.5 * (sci_data.shape[1] - 1)) / max(float(sci_data.shape[1]), 1.0)
    full_y = (y_idx - 0.5 * (sci_data.shape[0] - 1)) / max(float(sci_data.shape[0]), 1.0)
    full_design = np.stack(
        [
            np.ones_like(full_x),
            full_x,
            full_y,
            full_x * full_y,
            full_x**2,
            full_y**2,
        ],
        axis=0,
    )
    bkg_map = np.tensordot(coeffs, full_design, axes=(0, 0)).astype(np.float32)
    residual = sci_data - bkg_map
    valid_resid = residual[valid]
    step_r = max(1, valid_resid.size // 100000)
    global_rms = mad_std(valid_resid[::step_r], ignore_nan=True)
    if not np.isfinite(global_rms) or global_rms <= 0:
        global_rms = float(np.nanstd(valid_resid))
    bkg_rms_map = np.full(sci_data.shape, global_rms, dtype=np.float32)
    print(f"    Smooth-surface fallback RMS: {global_rms:.3f}")
    return bkg_map, bkg_rms_map


def _auto_box_size(sci_data_shape, mask_fraction, config):
    """Choose a background box size from image scale and crowding."""
    min_dim = min(sci_data_shape)
    base = max(32, min(256, int(round(min_dim / 8.0))))
    if mask_fraction > 0.5:
        base = min(512, base * 2)
    base = int(config.get("box_size", base))
    return max(16, base)


def estimate_background_with_diagnostics(sci_data, mask, config):
    """Estimate background and return diagnostic metadata for benchmarking."""
    diagnostics = {
        "requested_method": config.get("method", "sep").lower(),
        "mask_fraction": float(np.mean(mask)),
        "effective_method": None,
        "fallback": None,
        "box_size": None,
    }
    bkg_map, bkg_rms_map = estimate_background(sci_data, mask, {**config, "_diagnostics": diagnostics})
    return bkg_map, bkg_rms_map, diagnostics


def estimate_background(sci_data, mask, config):
    """
    Estimate background and background RMS using a configured method.

    Args:
        sci_data (ndarray): Science image data array
        mask (ndarray): Boolean mask of pixels to exclude from background estimation
        config (dict): Configuration dictionary for background estimation

    Returns:
        tuple: (background_map, background_rms_map) or (None, None) on failure.
    """
    method = config.get("method", "sep").lower()
    diagnostics = config.get("_diagnostics")
    print(f"  Estimating background using method: '{method}'")

    bkg_map, bkg_rms_map = None, None

    if method == "sep":
        mask_fraction = np.mean(mask)
        mask_threshold = float(config.get("mask_threshold", 0.8))
        if mask_fraction > mask_threshold:
            print(f"    WARNING: High mask coverage ({mask_fraction:.1%}). Proactively falling back to global mode.")
            bkg_map, bkg_rms_map = _estimate_global_sep(sci_data, mask)
            if diagnostics is not None:
                diagnostics["fallback"] = "global_sep"
            if bkg_map is None:
                print("    Global SEP fallback failed; switching to robust median fallback.")
                method = "robust_median_fallback"
        else:
            box_size = (
                _auto_box_size(sci_data.shape, mask_fraction, config)
                if config.get("auto_box_scaling", True)
                else config.get("box_size", 128)
            )
            filter_size = config.get("filter_size", 3)
            max_box_size = config.get("max_box_size", max(box_size, 1024))
            if diagnostics is not None:
                diagnostics["box_size"] = int(box_size)
            bkg_map, bkg_rms_map = _estimate_sep_tiered(sci_data, mask, box_size, filter_size, max_box_size)
            if bkg_map is None:
                print("    SEP tiered retries failed; switching to robust median fallback.")
                method = "robust_median_fallback"

    if method in ("robust_median_fallback", "median_filter"):
        bkg_map, bkg_rms_map = _estimate_robust_median(sci_data, mask, method, config)
        if bkg_map is None and config.get("smooth_surface_fallback", True):
            bkg_map, bkg_rms_map = _estimate_smooth_surface(sci_data, mask, config)
            method = "smooth_surface"
    elif method != "sep":
        warnings.warn(f"Unknown background estimation method: '{method}'", RuntimeWarning)
        return None, None

    if bkg_rms_map is not None:
        invalid = ~np.isfinite(bkg_rms_map) | (bkg_rms_map <= 0)
        if np.any(invalid):
            bkg_rms_map = np.where(~invalid, bkg_rms_map, np.inf)

    if diagnostics is not None:
        diagnostics["effective_method"] = method

    return bkg_map, bkg_rms_map
