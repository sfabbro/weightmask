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
    """Estimate background using SEP with tiered retries.

    Returns ``(bkg_map, bkg_rms_map, box_used)``; ``box_used`` is None on failure.
    """
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
            return bkg_map, bkg_rms_map, int(current_box)

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
    return None, None, None


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
    if np.count_nonzero(valid) < 100:
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
    auto = max(32, min(256, int(round(min_dim / 8.0))))
    if mask_fraction > 0.5:
        auto = min(512, auto * 2)
    try:
        cfg_box = int(config.get("box_size", auto))
    except (TypeError, ValueError):
        cfg_box = auto
    # Config box is a ceiling: small detectors scale down to 64/32 automatically.
    return max(16, min(auto, cfg_box))


def _as_box(box, default=128):
    try:
        return max(1, int(box))
    except (TypeError, ValueError):
        return default


def sky_to_mesh(sky_map, box):
    """Downsample full-res sky to SEP mesh nodes + FITS rebuild cards.

    ``n = (size - 1) // box + 1`` nodes at ``(k + 0.5) * box`` (clipped).
    Returns ``(mesh_float32, cards_dict)``.
    """
    box = _as_box(box)
    arr = np.ascontiguousarray(sky_map, dtype=np.float32)
    h, w = arr.shape
    ny = max(1, (h - 1) // box + 1) if h > 0 else 1
    nx = max(1, (w - 1) // box + 1) if w > 0 else 1
    yi = np.clip(np.rint((np.arange(ny) + 0.5) * box).astype(np.intp), 0, max(h - 1, 0))
    xi = np.clip(np.rint((np.arange(nx) + 0.5) * box).astype(np.intp), 0, max(w - 1, 0))
    mesh = arr[np.ix_(yi, xi)] if h > 0 and w > 0 else np.zeros((ny, nx), dtype=np.float32)
    cards = {"SKYMESH": True, "MESHBW": box, "MESHBH": box, "SKYH": h, "SKYW": w}
    return np.ascontiguousarray(mesh, dtype=np.float32), cards


def reconstruct_sky_mesh(mesh, shape, box):
    """Rebuild full-res sky from mesh (SEP node phase + natural cubic)."""
    # ponytail: scipy CubicSpline, not SEP C bicubic; sub-ADU on MegaPrime.
    # Swap back to a SEP port if bit-identical back() is required.
    from scipy.interpolate import CubicSpline

    box = _as_box(box)
    m = np.atleast_2d(np.asarray(mesh, dtype=np.float64))
    h, w = int(shape[0]), int(shape[1])
    if m.size == 0 or h <= 0 or w <= 0:
        return np.zeros((max(h, 0), max(w, 0)), dtype=np.float32)
    ny, nx = m.shape
    if ny == 1 and nx == 1:
        return np.full((h, w), float(m[0, 0]), dtype=np.float32)
    node_y = (np.arange(ny) + 0.5) * box
    node_x = (np.arange(nx) + 0.5) * box
    ys = np.arange(h, dtype=np.float64)
    xs = np.arange(w, dtype=np.float64)
    if ny > 1:
        cols = np.column_stack([CubicSpline(node_y, m[:, j], bc_type="natural")(ys) for j in range(nx)])
    else:
        cols = np.broadcast_to(m, (h, nx)).copy()
    if nx > 1:
        out = np.vstack([CubicSpline(node_x, cols[i], bc_type="natural")(xs) for i in range(h)])
    else:
        out = cols
    return np.ascontiguousarray(out, dtype=np.float32)


def parse_sky_mesh_header(header):
    """Parse SKYMESH cards into ``((h, w), box)``."""
    if header is None:
        raise ValueError("Header is missing SKYMESH=T (not a sky mesh product)")
    try:
        skymesh = header["SKYMESH"]
    except Exception as e:
        raise ValueError("Header is missing SKYMESH=T (not a sky mesh product)") from e
    if skymesh is not True and str(skymesh).strip().upper() not in ("T", "TRUE", "1"):
        raise ValueError("Header is missing SKYMESH=T (not a sky mesh product)")
    try:
        h, w = int(header["SKYH"]), int(header["SKYW"])
        box = int(header["MESHBW"] if "MESHBW" in header else header["MESHBH"])
    except Exception as e:
        raise ValueError("SKYMESH header requires SKYH, SKYW, and MESHBW/MESHBH") from e
    if h <= 0 or w <= 0:
        raise ValueError(f"Invalid SKYH/SKYW shape ({h}, {w})")
    return (h, w), max(1, box)


def reconstruct_sky_from_header(mesh, header):
    """Rebuild full-res sky from a mesh array and its SKYMESH FITS header cards."""
    shape, box = parse_sky_mesh_header(header)
    return reconstruct_sky_mesh(mesh, shape, box)


def _repair_negative_dips(bkg_map, sci_data, bkg_rms_map, mask, config, diagnostics=None):
    """Nearest-valid fill for unphysical negative-sky interpolation overshoots.

    SEP's mesh interpolation can undershoot below zero next to masked bright
    sources and image edges (measured: 0.2-0.5% of a MegaPrime CCD at -100s of
    ADU while the data sit at +1700). A negative *level* is only repaired when
    the data agree with the repaired sky: the pixel must be inconsistent with
    the map (data far above it) and consistent with the fill (data near
    neighbor sky). Faint regions, blanks, and rms-unknown pixels never qualify.
    """
    try:
        enable = bool(config.get("dip_repair_enable", True))
    except Exception:
        enable = True
    if not enable or bkg_map is None:
        return bkg_map
    try:
        sigma = float(config.get("dip_repair_sigma", 5.0))
        max_frac = float(config.get("dip_repair_max_fraction", 0.05))
    except (TypeError, ValueError):
        sigma, max_frac = 5.0, 0.05
    try:
        data = np.asarray(sci_data, dtype=np.float64)
        guide = np.isfinite(np.asarray(bkg_map, dtype=np.float64))
    except Exception:
        return bkg_map
    try:
        candidate = (
            guide
            & (np.asarray(bkg_map) < 0)
            & np.isfinite(data)
            & (~np.asarray(mask, dtype=bool))
        )
    except Exception:
        return bkg_map
    n_cand = int(np.count_nonzero(candidate))
    if n_cand == 0:
        return bkg_map
    frac = n_cand / bkg_map.size
    if not np.isfinite(frac) or frac > max_frac:
        print(f"    WARNING: negative sky over {frac:.1%} of image exceeds repair cap; leaving map as-is.")
        return bkg_map
    try:
        from scipy.ndimage import distance_transform_edt

        rms = np.asarray(bkg_rms_map, dtype=np.float64)
        finite_rms = np.isfinite(rms) & (rms > 0)
        rms_ref = np.median(rms[finite_rms]) if np.any(finite_rms) else np.nan
        thresh = np.where(finite_rms, sigma * rms, sigma * rms_ref)
        skymap = np.asarray(bkg_map, dtype=np.float64)
        skymed = np.median(skymap[np.isfinite(skymap)]) if np.any(np.isfinite(skymap)) else np.nan
        invalid = candidate | ~guide
        _, idx = distance_transform_edt(invalid, return_indices=True)
        fill_nn = skymap[tuple(idx)]
        bad_map = (data - skymap) > thresh
        # Tier 1: neighbor fill where the data agree with it.
        accept = (
            candidate
            & np.isfinite(thresh)
            & np.isfinite(fill_nn)
            & bad_map
            & (np.abs(data - fill_nn) <= thresh)
        )
        # Tier 2: deep interiors whose border fill is still depressed fall back
        # to the global median when the data agree with that instead.
        if np.isfinite(skymed):
            accept2 = (
                candidate
                & ~accept
                & np.isfinite(thresh)
                & bad_map
                & (np.abs(data - skymed) <= thresh)
            )
        else:
            accept2 = np.zeros_like(candidate, dtype=bool)
    except Exception as e:
        print(f"    WARNING: dip repair failed ({e}); leaving map as-is.")
        return bkg_map
    n_bad = int(np.count_nonzero(accept)) + int(np.count_nonzero(accept2))
    if n_bad == 0:
        return bkg_map
    out = skymap.copy()
    out[accept] = fill_nn[accept]
    out[accept2] = skymed
    print(f"    Repaired {n_bad} negative-sky pixels ({n_bad / bkg_map.size:.3%}) via nearest-valid fill.")
    if diagnostics is not None:
        diagnostics["dip_repaired"] = n_bad
    return out.astype(np.asarray(bkg_map).dtype, copy=False)


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
    # Mesh encoding needs the SEP box that actually built the map (incl. retries).
    # Non-SEP / global fallbacks have no mesh-compatible box -> leave unset/None.
    if diagnostics is not None:
        diagnostics.pop("box_size", None)

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
            try:
                box_size = max(1, int(box_size))
            except (TypeError, ValueError):
                box_size = 128
            filter_size = config.get("filter_size", 3)
            max_box_size = config.get("max_box_size", max(box_size, 1024))
            bkg_map, bkg_rms_map, used_box = _estimate_sep_tiered(
                sci_data, mask, box_size, filter_size, max_box_size
            )
            if diagnostics is not None and used_box is not None:
                diagnostics["box_size"] = used_box
            if bkg_map is None:
                print("    SEP tiered retries failed; switching to robust median fallback.")
                method = "robust_median_fallback"

    if method in ("robust_median_fallback", "median_filter"):
        bkg_map, bkg_rms_map = _estimate_robust_median(sci_data, mask, method, config)
        if diagnostics is not None:
            diagnostics.pop("box_size", None)
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

    if bkg_map is not None and bkg_rms_map is not None:
        bkg_map = _repair_negative_dips(bkg_map, sci_data, bkg_rms_map, mask, config, diagnostics)

    if diagnostics is not None:
        diagnostics["effective_method"] = method

    return bkg_map, bkg_rms_map
