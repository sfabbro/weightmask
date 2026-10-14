import time
import warnings

import numpy as np
import scipy.ndimage as ndi
from astropy.stats import mad_std
from skimage.draw import line
from skimage.measure import LineModelND, label, ransac
from skimage.morphology import dilation, disk

from .utils import rms_or_robust, rms_valid_mask, robust_rms


def _profile_outliers(profile, sigma):
    """True where a 1-D profile exceeds ``sigma`` robust standard deviations."""
    profile = np.asarray(profile, dtype=np.float64)
    finite = np.isfinite(profile)
    outliers = np.zeros(profile.shape, dtype=bool)
    if not np.any(finite):
        return outliers
    values = profile[finite]
    med = float(np.median(values))
    mad = 1.4826 * float(np.median(np.abs(values - med)))
    if not np.isfinite(mad) or mad <= 1e-12:
        mad = 1e-12
    outliers[finite] = values > med + float(sigma) * mad
    return outliers


def _finite_axis_profiles(frame):
    if np.all(np.isfinite(frame)):
        return np.median(frame, axis=0), np.median(frame, axis=1)
    values = frame.copy()
    values[~np.isfinite(values)] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(values, axis=0), np.nanmedian(values, axis=1)


class PersistenceProfileAccumulator:
    def __init__(self, shape, min_other=2, sigma=3.0):
        if len(shape) != 2:
            raise ValueError("persistent_axis_mask frames must be 2-D")
        self.shape = tuple(shape)
        self.min_other = int(min_other)
        self.sigma = float(sigma)
        self.frame_count = 0
        self.col_hits = np.zeros(self.shape[1], dtype=np.intp)
        self.row_hits = np.zeros(self.shape[0], dtype=np.intp)

    def add(self, frame):
        frame = np.asarray(frame, dtype=np.float64)
        if frame.shape != self.shape:
            raise ValueError("persistent_axis_mask frames must share a shape")
        col_profile, row_profile = _finite_axis_profiles(frame)
        self.col_hits += _profile_outliers(col_profile, self.sigma)
        self.row_hits += _profile_outliers(row_profile, self.sigma)
        self.frame_count += 1

    def mask(self):
        mask = np.zeros(self.shape, dtype=bool)
        hot_cols = np.nonzero(self.col_hits >= self.min_other)[0]
        hot_rows = np.nonzero(self.row_hits >= self.min_other)[0]
        if hot_cols.size:
            mask[:, hot_cols] = True
        if hot_rows.size:
            mask[hot_rows, :] = True
        return mask


def persistent_axis_mask(frames, min_other=2, sigma=3.0):
    """Columns and rows that are bright outliers in at least ``min_other`` frames.

    ``frames`` are other exposures of one CCD (not the frame being masked). An
    axis is detector-fixed when its median is a robust outlier in that many
    frames — the same count the trail curator requires of other epochs. A trail
    that exists in only one exposure does not qualify.
    """
    accumulator = None
    for frame in frames:
        array = np.asarray(frame, dtype=np.float64)
        if accumulator is None:
            accumulator = PersistenceProfileAccumulator(array.shape, min_other=min_other, sigma=sigma)
        accumulator.add(array)
        del array, frame
    if accumulator is None:
        raise ValueError("persistent_axis_mask needs at least one frame")
    return accumulator.mask()


def _normalize_angle_deg(angle_deg):
    """Normalize an angle to the [0, 180) degree range."""
    return (angle_deg + 180.0) % 180.0


def _bin_array(data_sub, bin_factor):
    """Mean-bin an array by an integer factor, trimming the leftover border."""
    bh = data_sub.shape[0] // bin_factor
    bw = data_sub.shape[1] // bin_factor
    return data_sub[: bh * bin_factor, : bw * bin_factor].reshape(bh, bin_factor, bw, bin_factor).mean(axis=(1, 3))


_OBSOLETE_STREAK_KEYS = frozenset(
    {
        "method",
        "dilation_radius",
        "enable_ransac_trails",
        "ransac_params",
        "frangi_params",
        "frangi_legacy_params",
        "mrt_rescue_params",
        "satdet_params",
        "retry_without_existing_mask",
        "retry_if_area_fraction_exceeds",
        "retry_if_support_width_exceeds",
        "sparse_on_primary_weak_only",
    }
)


def _resolve_streak_mode(config):
    """Production accepts only ``mode: auto_ground``."""
    obsolete = sorted(_OBSOLETE_STREAK_KEYS & config.keys())
    if obsolete:
        raise ValueError(f"Unsupported streak parameters: {', '.join(obsolete)}")
    requested = config.get("mode", "auto_ground")
    if requested != "auto_ground":
        raise ValueError(f"Unknown streak detection mode {requested!r}. Valid mode: 'auto_ground'.")
    return "auto_ground"


_EDGE_BUFFER_DEFAULT = 32


def _clip_line_to_image(anchor, direction, shape):
    """Clip an infinite line to the image bounds."""
    h, w = shape
    ax, ay = anchor
    dx, dy = direction
    points = []

    if abs(dx) > 1e-6:
        for x in (0.0, float(w - 1)):
            t = (x - ax) / dx
            y = ay + t * dy
            if 0.0 <= y <= h - 1:
                points.append((x, y))

    if abs(dy) > 1e-6:
        for y in (0.0, float(h - 1)):
            t = (y - ay) / dy
            x = ax + t * dx
            if 0.0 <= x <= w - 1:
                points.append((x, y))

    unique_points = []
    for point in points:
        if not any(np.allclose(point, existing, atol=1e-6) for existing in unique_points):
            unique_points.append(point)

    if len(unique_points) < 2:
        return None

    max_pair = None
    max_dist = -1.0
    for i, p0 in enumerate(unique_points):
        for p1 in unique_points[i + 1 :]:
            dist = np.hypot(p1[0] - p0[0], p1[1] - p0[1])
            if dist > max_dist:
                max_dist = dist
                max_pair = (p0, p1)
    return max_pair


def _edges_touched(endpoints, shape, edge_buffer):
    """Return the set of image edges touched by the endpoints."""
    h, w = shape
    touched = set()
    for x, y in endpoints:
        if x <= edge_buffer:
            touched.add("left")
        if x >= (w - 1 - edge_buffer):
            touched.add("right")
        if y <= edge_buffer:
            touched.add("top")
        if y >= (h - 1 - edge_buffer):
            touched.add("bottom")
    return touched


def _adjust_confidence_for_mask(confidence_threshold, existing_mask):
    """Raise the accept threshold in proportion to the excluded-pixel fraction.

    A heavily masked image has fewer independent pixels to confirm a trail, so
    the confidence bar is raised by a capped amount.
    """
    if existing_mask is not None:
        return confidence_threshold + min(0.25, 20.0 * float(np.mean(existing_mask)))
    return confidence_threshold


def _sample_trail_strip(image, endpoints, strip_length, strip_width, interpolation_order):
    """Sample a trail-aligned strip using interpolation in local trail coordinates."""
    (x0, y0), (x1, y1) = endpoints
    dx = x1 - x0
    dy = y1 - y0
    length = np.hypot(dx, dy)
    if length <= 1.0:
        return None

    direction = np.array([dx, dy], dtype=np.float32) / length
    normal = np.array([-direction[1], direction[0]], dtype=np.float32)
    center = np.array([(x0 + x1) * 0.5, (y0 + y1) * 0.5], dtype=np.float32)

    sample_length = int(max(strip_length, np.ceil(length) + 1))
    sample_width = int(max(strip_width, 5))
    t = np.linspace(-0.5 * (sample_length - 1), 0.5 * (sample_length - 1), sample_length, dtype=np.float32)
    v = np.linspace(-0.5 * (sample_width - 1), 0.5 * (sample_width - 1), sample_width, dtype=np.float32)

    x_coords = center[0] + t[:, None] * direction[0] + v[None, :] * normal[0]
    y_coords = center[1] + t[:, None] * direction[1] + v[None, :] * normal[1]
    inside = (x_coords >= 0.0) & (x_coords <= image.shape[1] - 1) & (y_coords >= 0.0) & (y_coords <= image.shape[0] - 1)

    sampled = ndi.map_coordinates(
        image,
        [y_coords.ravel(), x_coords.ravel()],
        order=int(interpolation_order),
        mode="constant",
        cval=0.0,
    ).reshape(sample_length, sample_width)
    sampled = np.where(inside, sampled, np.nan)

    return {
        "sampled": sampled,
        "inside": inside,
        "x_coords": x_coords,
        "y_coords": y_coords,
    }


def _largest_near_center(mask_1d):
    """Keep the contiguous support region closest to the strip center."""
    if not np.any(mask_1d):
        return mask_1d

    labels, n = ndi.label(mask_1d.astype(np.uint8))
    if n <= 1:
        return mask_1d

    center = 0.5 * (len(mask_1d) - 1)
    best_label = None
    best_score = None
    for label_idx in range(1, n + 1):
        idx = np.where(labels == label_idx)[0]
        centroid = idx.mean()
        score = (abs(centroid - center), -len(idx))
        if best_score is None or score < best_score:
            best_score = score
            best_label = label_idx
    return labels == best_label


def _largest_contiguous_run(mask_1d, max_gap=0, min_run_length=1, valid=None):
    """Return the longest coherent group of runs in a 1D mask.

    ``max_gap`` is the largest number of false samples allowed between two
    support runs before they are treated as unrelated. ``min_run_length``
    discards isolated short runs before grouping, which keeps a few noise pixels
    from joining otherwise separate fragments. The selected runs are returned
    without filling their gaps, so this preserves the measured support while
    preventing a faint-but-real dashed trail from being truncated to its
    longest bright fragment. With ``max_gap=0`` and the default minimum run
    length, this retains the historical longest-contiguous-run behavior.
    Optional valid samples exclude occluded locations from run selection,
    without filling or detecting anything in those locations.
    """
    mask_1d = np.asarray(mask_1d, dtype=bool)
    if valid is not None:
        keep = np.zeros_like(mask_1d)
        keep[valid] = _largest_contiguous_run(mask_1d[valid], max_gap, min_run_length)
        return keep
    if not np.any(mask_1d):
        return mask_1d

    labels, n = ndi.label(mask_1d.astype(np.uint8))
    if n <= 1:
        return mask_1d if np.count_nonzero(mask_1d) >= max(1, int(min_run_length)) else np.zeros_like(mask_1d)

    runs = []
    for label_idx in range(1, n + 1):
        indices = np.flatnonzero(labels == label_idx)
        runs.append((int(indices[0]), int(indices[-1]), int(indices.size)))

    minimum = max(1, int(min_run_length))
    if minimum > 1:
        runs = [run for run in runs if run[2] >= minimum]
        if not runs:
            return np.zeros_like(mask_1d)

    gap = max(0, int(max_gap))
    groups = []
    for start, end, count in runs:
        if groups and start - groups[-1][1] - 1 <= gap:
            groups[-1] = (groups[-1][0], end, groups[-1][2] + count)
        else:
            groups.append((start, end, count))

    # Match the old primary criterion (number of true samples), with span and
    # earliest start as deterministic tie-breakers.
    best = max(groups, key=lambda group: (group[2], group[1] - group[0] + 1, -group[0]))
    keep = np.zeros_like(mask_1d)
    best_start, best_end, _best_count = best
    for start, end, count in runs:
        if count >= minimum and start >= best_start and end <= best_end:
            keep[start : end + 1] = True
    return keep


def _refit_endpoints_from_strip(strip, centered, inside):
    """Refit trail endpoints from across-strip flux centroids (one robust pass).

    Hough-segment clustering leaves ~1-2° angle errors; over a 256px strip the
    trail drifts ~10px across the strip and the support blooms to 12-20 rows.
    A centroid fit per strip row recovers the true line; callers re-sample the
    strip once at the corrected angle. Returns new clipped endpoints or None.
    """
    h, w = centered.shape
    weights = np.where(inside & np.isfinite(centered), np.clip(centered, 0.0, None), 0.0)
    wsum = weights.sum(axis=1)
    valid = wsum > 0
    if int(np.count_nonzero(valid)) < 8:
        return None
    cols = np.arange(w, dtype=np.float64)
    cent = (weights[valid] @ cols) / wsum[valid]
    rows = np.nonzero(valid)[0].astype(np.float64)
    # Consensus Theil-Sen: every valid row votes once (bright stars span few
    # rows and lose the median), then one inlier refit. No consensus -> None.
    sel = np.linspace(0, len(rows) - 1, min(len(rows), 41)).astype(int)
    r, c = rows[sel], cent[sel]
    slopes = []
    for stride in (1, 2, 4):
        for k in range(0, len(r) - stride, stride):
            dr = r[k + stride] - r[k]
            if dr >= 4:
                slopes.append((c[k + stride] - c[k]) / dr)
    if not slopes:
        return None
    slope = float(np.median(slopes))
    icept = float(np.median(c - slope * r))
    resid = np.abs(c - (slope * r + icept))
    inl = resid <= max(3.0, 2.0 * float(np.median(resid)))
    if float(np.mean(inl)) < 0.5 or int(np.count_nonzero(inl)) < 8:
        return None  # rows disagree: star soup, not one line; keep geometry
    if float(r[inl].max() - r[inl].min()) < 100:
        return None  # consensus spans a star, not the strip; keep geometry
    r2, c2 = r[inl], c[inl]
    slopes2 = []
    for stride in (1, 2, 4):
        for k in range(0, len(r2) - stride, stride):
            dr = r2[k + stride] - r2[k]
            if dr >= 4:
                slopes2.append((c2[k + stride] - c2[k]) / dr)
    if slopes2:
        slope = float(np.median(slopes2))
        icept = float(np.median(c2 - slope * r2))
    if abs(slope) * h < 0.25:
        return None  # sub-pixel drift over the strip; resampling buys nothing
    x_coords, y_coords = strip["x_coords"], strip["y_coords"]

    def _image_at(row_f, col_f):
        ri = min(h - 1, max(0, int(round(row_f))))
        ci = min(w - 1, max(0, int(round(col_f))))
        return float(np.median(x_coords[max(0, ri - 1) : ri + 2, ci])), float(
            np.median(y_coords[max(0, ri - 1) : ri + 2, ci])
        )

    x0, y0 = _image_at(0, icept)
    x1, y1 = _image_at(h - 1, icept + slope * (h - 1))
    if not all(np.isfinite(v) for v in (x0, y0, x1, y1)):
        return None
    if np.hypot(x1 - x0, y1 - y0) < 20:
        return None
    return ((x0, y0), (x1, y1))


def _refine_trail_mask(data_sub, bkg_rms_map, candidate, mask_cfg, existing_mask=None):
    """Refine a Hough candidate by fitting a trail-aligned strip mask."""
    if bkg_rms_map is not None:
        # Detection threshold: where the background RMS is unknown (the ``inf``
        # sentinel from ``estimate_background``) there is no significance to
        # assert, so those pixels are made inert exactly like existing-mask
        # pixels below -- *not* normalised by a substitute RMS, which would let
        # an unmeasurable region manufacture strip support.
        valid_rms = rms_valid_mask(bkg_rms_map)
        usable_rms = np.where(valid_rms, bkg_rms_map, 1.0)
        detect_img = np.where(valid_rms, data_sub / np.where(usable_rms > 0, usable_rms, 1.0), np.nan)
    else:
        detect_img = data_sub
    if existing_mask is not None:
        detect_img = np.where(existing_mask, np.nan, detect_img)

    strip_length = int(mask_cfg.get("strip_length", 256))
    strip_width = int(mask_cfg.get("strip_width", 96))
    profile_sigma = float(mask_cfg.get("profile_sigma_threshold", 3.0))
    interpolation_order = int(mask_cfg.get("rotation_interpolation_order", 1))
    padding = int(mask_cfg.get("padding", 4))
    profile_percentile = float(mask_cfg.get("profile_percentile", 85.0))
    min_mask_pixels = int(mask_cfg.get("min_mask_pixels", 64))
    min_row_hits = int(mask_cfg.get("min_row_hits", 8))
    min_row_hit_fraction = float(mask_cfg.get("min_row_hit_fraction", 0.5))
    min_col_hit_fraction = float(mask_cfg.get("min_col_hit_fraction", 0.2))
    max_support_width = int(mask_cfg.get("max_support_width", 16))
    # Only the full-span Hough-peak candidate has independent line evidence
    # strong enough to bridge intermittent support. Contour candidates can
    # follow curved star/galaxy structure; extending those across gaps creates
    # long false-positive trails.
    max_row_gap = int(mask_cfg.get("max_row_gap", 0)) if candidate.get("source") == "houghpeaks" else 0
    min_row_run = int(mask_cfg.get("min_row_run", 0)) if max_row_gap > 0 else 0

    def _support_count(centered, inside, bg_std):
        # Padded support width for one sampled geometry (argmin criterion).
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            profile = np.nanpercentile(centered, profile_percentile, axis=0)
        hot = inside & np.isfinite(centered) & (centered > profile_sigma * bg_std)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            col_hit = np.nanmean(hot, axis=0)
        sup = np.isfinite(profile) & (profile > profile_sigma * bg_std)
        sup &= np.isfinite(col_hit) & (col_hit >= min_col_hit_fraction)
        sup = _largest_near_center(sup)
        if np.any(sup) and padding > 0:
            sup = ndi.binary_dilation(sup, structure=np.ones(2 * padding + 1, dtype=bool))
        # A narrower refit is useful only if it still supports a coherent trail.
        # Otherwise a short noise fragment can replace and erase a valid line.
        hits = np.any(hot & sup[np.newaxis, :], axis=1)
        hits = ndi.binary_closing(hits, structure=np.ones(2 * padding + 1, dtype=bool))
        observed = np.any(np.isfinite(centered[:, sup]), axis=1) if candidate.get("source") == "houghpeaks" else None
        hits = _largest_contiguous_run(hits, max_gap=max_row_gap, min_run_length=min_row_run, valid=observed)
        if np.count_nonzero(hits) < min_row_hits or np.mean(hits) < min_row_hit_fraction:
            return 0
        return int(np.count_nonzero(sup))

    def _better(w_new, w_cur):
        # Empty support (no detection) never wins: prefer any detection,
        # then the narrowest one. Fixes argmin-toward-empty collapse.
        if w_cur == 0:
            return w_new != 0
        return 0 < w_new < w_cur

    endpoints = candidate["clipped_endpoints"]
    strip = centered = inside = bg_std = None
    for _pass in range(2):
        strip = _sample_trail_strip(detect_img, endpoints, strip_length, strip_width, interpolation_order)
        if strip is None:
            return np.zeros(data_sub.shape, dtype=bool), {
                "support_width": 0,
                "row_hit_fraction": 0.0,
                "mask_pixels": 0,
                "reject_reason": "strip_none",
            }
        sampled = strip["sampled"]
        inside = strip["inside"]
        width_axis = np.arange(sampled.shape[1], dtype=np.float32)
        center_col = 0.5 * (sampled.shape[1] - 1)
        sideband = np.abs(width_axis - center_col) >= max(3.0, 0.25 * sampled.shape[1])
        background_pixels = sampled[:, sideband]
        bg_values = background_pixels[np.isfinite(background_pixels)]
        if bg_values.size < 50:
            return np.zeros(data_sub.shape, dtype=bool), {
                "support_width": 0,
                "row_hit_fraction": 0.0,
                "mask_pixels": 0,
                "reject_reason": "bg_starved",
            }
        bg_med = np.median(bg_values)
        bg_std = mad_std(bg_values, ignore_nan=True)
        if not np.isfinite(bg_std) or bg_std <= 1e-6:
            bg_std = np.std(bg_values)
        if not np.isfinite(bg_std) or bg_std <= 1e-6:
            bg_std = np.nanstd(sampled)
        if not np.isfinite(bg_std) or bg_std <= 1e-6:
            bg_std = 1e-3
        # Check for asymmetric step discontinuity (e.g. amplifier boundary) across the strip
        left_band = sampled[:, : max(1, int(0.25 * sampled.shape[1]))]
        right_band = sampled[:, max(1, int(0.75 * sampled.shape[1])) :]
        left_finite = left_band[np.isfinite(left_band)]
        right_finite = right_band[np.isfinite(right_band)]
        left_med = np.median(left_finite) if left_finite.size > 0 else np.nan
        right_med = np.median(right_finite) if right_finite.size > 0 else np.nan
        if np.isfinite(left_med) and np.isfinite(right_med):
            step_diff = abs(left_med - right_med)
            if step_diff > 3.5 * bg_std:
                return np.zeros(data_sub.shape, dtype=bool), {
                    "support_width": 0,
                    "row_hit_fraction": 0.0,
                    "mask_pixels": 0,
                    "reject_reason": "step_discontinuity",
                }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            row_bg = np.nanmedian(background_pixels, axis=1)
        row_bg = np.where(np.isfinite(row_bg), row_bg, bg_med)
        centered = sampled - row_bg[:, np.newaxis]
        if _pass == 0:
            width0 = _support_count(centered, inside, bg_std)
            bundle0 = (strip, sampled, inside, bg_med, bg_std, row_bg, centered)
            refit = _refit_endpoints_from_strip(strip, centered, inside)
            if refit is None:
                break
            endpoints = refit
        else:
            width1 = _support_count(centered, inside, bg_std)
            best_w = width1 if _better(width1, width0) else width0
            if _better(width0, width1):
                strip, sampled, inside, bg_med, bg_std, row_bg, centered = bundle0
            # Near-miss fan: sub-degree grid errors still drift a 2000px strip
            # ~10px wide. Rotate the original endpoints around their midpoint
            # and keep the narrowest support. Bounded to near-misses only.
            fan_slack = float(mask_cfg.get("angle_fan_slack", 13.0))
            if (best_w == 0 and candidate.get("source") == "houghpeaks") or (
                max_support_width < best_w <= max_support_width + fan_slack
            ):
                (fx0, fy0), (fx1, fy1) = candidate["clipped_endpoints"]
                fmx, fmy = 0.5 * (fx0 + fx1), 0.5 * (fy0 + fy1)
                flen = max(np.hypot(fx1 - fx0, fy1 - fy0), 1.0)
                fnx, fny = -(fy1 - fy0) / flen, (fx1 - fx0) / flen
                fan_geoms = []
                for d_ang in (0.25, -0.25, 0.5, -0.5, 1.0, -1.0):
                    ca = np.cos(np.radians(d_ang))
                    sa = np.sin(np.radians(d_ang))
                    rot = []
                    for qx, qy in ((fx0, fy0), (fx1, fy1)):
                        dx, dy = qx - fmx, qy - fmy
                        rot.append((fmx + dx * ca - dy * sa, fmy + dx * sa + dy * ca))
                    fan_geoms.append((rot[0], rot[1]))
                # Lateral offsets catch rho-blended peaks whose line runs
                # parallel to the trail but outside the strip center.
                for d_off in (8.0, -8.0, 16.0, -16.0):
                    fan_geoms.append(
                        (
                            (fx0 + fnx * d_off, fy0 + fny * d_off),
                            (fx1 + fnx * d_off, fy1 + fny * d_off),
                        )
                    )
                for rot in fan_geoms:
                    fs = _sample_trail_strip(
                        detect_img, (rot[0], rot[1]), strip_length, strip_width, interpolation_order
                    )
                    if fs is None:
                        continue
                    fsmp = fs["sampled"]
                    fins = fs["inside"]
                    fbackground = fsmp[:, sideband]
                    fbg_values = fbackground[np.isfinite(fbackground)]
                    if fbg_values.size < 50:
                        continue
                    fbg_med = np.median(fbg_values)
                    fbg_std = mad_std(fbg_values, ignore_nan=True)
                    if not np.isfinite(fbg_std) or fbg_std <= 1e-6:
                        fbg_std = np.std(fbg_values)
                    if not np.isfinite(fbg_std) or fbg_std <= 1e-6:
                        fbg_std = np.nanstd(fsmp)
                    if not np.isfinite(fbg_std) or fbg_std <= 1e-6:
                        fbg_std = 1e-3
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        frbg = np.nanmedian(fbackground, axis=1)
                    frbg = np.where(np.isfinite(frbg), frbg, fbg_med)
                    fcen = fsmp - frbg[:, np.newaxis]
                    wfan = _support_count(fcen, fins, fbg_std)
                    if _better(wfan, best_w):
                        best_w = wfan
                        strip, sampled, inside, bg_med, bg_std, row_bg, centered = (
                            fs,
                            fsmp,
                            fins,
                            fbg_med,
                            fbg_std,
                            frbg,
                            fcen,
                        )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        profile = np.nanpercentile(centered, profile_percentile, axis=0)

    hot_pixels = inside & np.isfinite(centered) & (centered > profile_sigma * bg_std)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        col_hit_fraction = np.nanmean(hot_pixels, axis=0)

    support_cols = np.isfinite(profile) & (profile > profile_sigma * bg_std)
    support_cols &= np.isfinite(col_hit_fraction) & (col_hit_fraction >= min_col_hit_fraction)
    support_cols = _largest_near_center(support_cols)
    if np.any(support_cols) and padding > 0:
        support_cols = ndi.binary_dilation(support_cols, structure=np.ones(2 * padding + 1, dtype=bool))

    if not np.any(support_cols):
        return np.zeros(data_sub.shape, dtype=bool), {
            "support_width": 0,
            "row_hit_fraction": 0.0,
            "mask_pixels": 0,
            "reject_reason": "no_support",
        }
    if np.count_nonzero(support_cols) > max_support_width:
        return np.zeros(data_sub.shape, dtype=bool), {
            "support_width": int(np.count_nonzero(support_cols)),
            "row_hit_fraction": 0.0,
            "mask_pixels": 0,
            "reject_reason": "width_over_max",
        }

    hot_pixels = hot_pixels & support_cols[np.newaxis, :]
    row_hits = np.any(hot_pixels, axis=1)
    row_hits = ndi.binary_closing(row_hits, structure=np.ones(2 * padding + 1, dtype=bool))
    observed = (
        np.any(np.isfinite(centered[:, support_cols]), axis=1) if candidate.get("source") == "houghpeaks" else None
    )
    row_hits = _largest_contiguous_run(row_hits, max_gap=max_row_gap, min_run_length=min_row_run, valid=observed)
    if np.count_nonzero(row_hits) < min_row_hits:
        return np.zeros(data_sub.shape, dtype=bool), {
            "support_width": int(np.count_nonzero(support_cols)),
            "row_hit_fraction": float(np.mean(row_hits)),
            "mask_pixels": 0,
            "reject_reason": "few_row_hits",
        }
    row_hit_fraction = float(np.mean(row_hits))
    if row_hit_fraction < min_row_hit_fraction:
        return np.zeros(data_sub.shape, dtype=bool), {
            "support_width": int(np.count_nonzero(support_cols)),
            "row_hit_fraction": row_hit_fraction,
            "mask_pixels": 0,
            "reject_reason": "low_row_hit_fraction",
        }

    occluded_geometry = None
    if observed is not None and existing_mask is not None and np.any(row_hits):
        # Save interior source crossings for prediction after every evidence
        # gate; occluded pixels never contribute detection support.
        yy = np.clip(np.rint(strip["y_coords"][:, support_cols]).astype(int), 0, data_sub.shape[0] - 1)
        xx = np.clip(np.rint(strip["x_coords"][:, support_cols]).astype(int), 0, data_sub.shape[1] - 1)
        known = np.isfinite(data_sub[yy, xx]) & strip["inside"][:, support_cols]
        if bkg_rms_map is not None:
            known &= valid_rms[yy, xx]
        occluded = np.all(existing_mask[yy, xx] & known, axis=1)
        ends = np.flatnonzero(row_hits)
        occluded[: ends[0]] = False
        occluded[ends[-1] + 1 :] = False
        if np.any(occluded):
            occluded_geometry = yy, xx, occluded

    refined_strip = np.zeros_like(hot_pixels, dtype=bool)
    refined_strip[row_hits, :] = support_cols[np.newaxis, :]
    refined_strip &= inside

    if padding > 0:
        refined_strip = ndi.binary_dilation(refined_strip, structure=np.ones((3, 2 * padding + 1), dtype=bool))

    yy = np.rint(strip["y_coords"][refined_strip]).astype(int)
    xx = np.rint(strip["x_coords"][refined_strip]).astype(int)
    valid = (yy >= 0) & (yy < data_sub.shape[0]) & (xx >= 0) & (xx < data_sub.shape[1])
    if np.count_nonzero(valid) < min_mask_pixels:
        return np.zeros(data_sub.shape, dtype=bool), {
            "support_width": int(np.count_nonzero(support_cols)),
            "row_hit_fraction": row_hit_fraction,
            "mask_pixels": int(np.count_nonzero(valid)),
            "reject_reason": "few_mask_pixels",
        }

    mask = np.zeros(data_sub.shape, dtype=bool)
    mask[yy[valid], xx[valid]] = True
    return mask, {
        "support_width": int(np.count_nonzero(support_cols)),
        "row_hit_fraction": row_hit_fraction,
        "mask_pixels": int(np.count_nonzero(mask)),
        "occluded_geometry": occluded_geometry,
    }


def _score_candidate(candidate, refined_mask, refine_info, existing_mask):
    """Score a refined candidate using support, continuity, and overlap penalties."""
    if refined_mask is None or np.count_nonzero(refined_mask) == 0:
        return 0.0

    span = max(float(candidate.get("span", 1.0)), 1.0)
    support_density = np.count_nonzero(refined_mask) / span
    segment_score = min(1.0, len(candidate.get("segments", [])) / 6.0)
    edge_score = min(1.0, float(candidate.get("edge_touches", 0)) / 2.0)
    support_width = float(refine_info.get("support_width", 0))
    row_hit_fraction = float(refine_info.get("row_hit_fraction", 0.0))
    width_penalty = max(0.0, (support_width - 8.0) / 12.0)
    overlap_penalty = 0.0
    if existing_mask is not None:
        overlap_penalty = float(np.mean(existing_mask[refined_mask])) if np.any(refined_mask) else 0.0
    corridor_penalty = float(candidate.get("corridor_overlap", 0.0))
    score = (
        0.45 * min(1.0, support_density / 6.0)
        + 0.35 * segment_score
        + 0.20 * edge_score
        + 0.25 * min(1.0, row_hit_fraction / 0.6)
        - 0.35 * overlap_penalty
        - 0.80 * corridor_penalty
        - 0.45 * width_penalty
    )
    return float(score)


def _candidate_from_rho_theta(rho, theta_deg, shape):
    """Build a representative candidate from Radon-space coordinates."""
    theta = np.radians(theta_deg)
    normal = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    direction = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float32)
    center = np.array([0.5 * (shape[1] - 1), 0.5 * (shape[0] - 1)], dtype=np.float32)
    anchor = center + rho * normal
    clipped = _clip_line_to_image(anchor, direction, shape)
    if clipped is None:
        return None
    (x0, y0), (x1, y1) = clipped
    span = np.hypot(x1 - x0, y1 - y0)
    return {
        "segments": [((float(x0), float(y0)), (float(x1), float(y1)))],
        "angle_deg": float(_normalize_angle_deg(theta_deg)),
        "endpoints": clipped,
        "clipped_endpoints": clipped,
        "span": float(span),
        "raw_span": float(span),
        "edge_touches": len(_edges_touched(clipped, shape, _EDGE_BUFFER_DEFAULT)),
    }


def _refine_hough_peak(H, thetas, rhos, ti, ri):
    """Parabolic sub-pixel refinement of a Hough accumulator peak.

    The 1-degree theta grid alone leaves ~0.3-degree errors, which drift a
    2200px strip ~12px across and bloat support past the width gate. A local
    parabola fit recovers ~0.05-degree precision. Returns (theta_deg, rho).
    """
    orient = H.shape == (len(rhos), len(thetas))
    n_rho, n_th = (len(rhos), len(thetas)) if orient else (H.shape[1], H.shape[0])
    # Clamp the peak before indexing the grids below: the neighbour probes are
    # clamped, so an out-of-range index here would raise on a grid edge.
    ti = min(n_th - 1, max(0, ti))
    ri = min(n_rho - 1, max(0, ri))

    def _at(r, t):
        r = min(n_rho - 1, max(0, r))
        t = min(n_th - 1, max(0, t))
        return float(H[r, t] if orient else H[t, r])

    def _parabolic(vals):
        a, b, c = vals
        denom = a - 2.0 * b + c
        if abs(denom) < 1e-9:
            return 0.0
        return max(-1.0, min(1.0, 0.5 * (a - c) / denom))

    d_theta = float(np.degrees(thetas[1] - thetas[0])) if len(thetas) > 1 else 1.0
    d_rho = float(rhos[1] - rhos[0]) if len(rhos) > 1 else 1.0
    off_t = _parabolic([_at(ri, ti - 1), _at(ri, ti), _at(ri, ti + 1)])
    off_r = _parabolic([_at(ri - 1, ti), _at(ri, ti), _at(ri + 1, ti)])
    return float(np.degrees(thetas[ti])) + off_t * d_theta, float(rhos[ri]) + off_r * d_rho


def _detect_streaks_houghpeaks(data_sub, bkg_rms_map, existing_mask, config, predictions=None):
    """Find full-span trails as accumulator peaks of a binned standard Hough.

    Probabilistic Hough segments drown in star clutter (130k segments on a
    real HDU) and cluster into full-span phantoms. A standard Hough on a 4x
    binned SNR image integrates along whole lines instead: a real trail is
    the dominant peak (846 vs 288 votes measured), found in ~2s. Peak lines
    go through the same strip confirm + score gates as every other candidate.
    Misses dashed/intermittent trails by construction (left to RANSAC).
    """
    from skimage.transform import hough_line, hough_line_peaks

    cfg = config.get("houghpeak_params", {})
    mask_cfg = config.get("mask_params", {})
    if not cfg.get("enable", True):
        return np.zeros(data_sub.shape, dtype=bool), [], {"peaks": 0}
    bfac = max(1, int(cfg.get("bin", 4)))
    thresh_sig = float(cfg.get("thresh_sig", 2.5))
    min_votes = int(cfg.get("min_votes", 100))
    max_candidates = int(cfg.get("max_candidates", 6))
    confidence_threshold = float(config.get("houghpeak_params", {}).get("confidence_threshold", 0.40))
    min_refined_mask_pixels = int(mask_cfg.get("min_mask_pixels", 64))
    confidence_threshold = _adjust_confidence_for_mask(confidence_threshold, existing_mask)

    h, w = data_sub.shape
    bh, bw = h // bfac, w // bfac
    if bh < 16 or bw < 16:
        return np.zeros(data_sub.shape, dtype=bool), [], {"peaks": 0}
    binned = _bin_array(data_sub, bfac)
    if bkg_rms_map is not None:
        brms = _bin_array(bkg_rms_map, bfac)
        snr = binned / np.maximum(brms / bfac, 1e-6)
    else:
        snr = binned
    if existing_mask is not None:
        bex = existing_mask[: bh * bfac, : bw * bfac].reshape(bh, bfac, bw, bfac).any(axis=(1, 3))
        snr = np.where(bex, 0.0, snr)
    streak_mask = np.zeros(data_sub.shape, dtype=bool)
    accepted = []
    used = []
    n_peaks_total = 0
    # Two strata: strict restores the experiment's proven normalization
    # (binned-mean / rms-mean, where an 8-sigma trail outvotes clutter 3:1);
    # nominal uses proper binned SNR for fainter trails.
    for sname, simg in (("strict", snr / bfac), ("nominal", snr)):
        binary = np.isfinite(simg) & (simg > thresh_sig)
        H, thetas, rhos = hough_line(binary)
        peaks = hough_line_peaks(H, thetas, rhos, min_distance=1, threshold=min_votes, num_peaks=2 * max_candidates)
        n_peaks_total += len(peaks[0])
        top = ", ".join(
            "(%.1f,%.0f:%d)" % (float(np.degrees(t)), float(r), int(v)) for v, t, r in list(zip(*peaks))[:5]
        )
        print(f"    [houghpeaks:{sname}] {int(binary.sum())} px, {len(peaks[0])} peaks top=[{top}]...")
        for _, theta, rho_b in zip(*peaks):
            ti = int(np.argmin(np.abs(thetas - theta)))
            ri = int(np.argmin(np.abs(rhos - rho_b)))
            theta_deg, rho_b = _refine_hough_peak(H, thetas, rhos, ti, ri)
            theta = np.radians(theta_deg)
            # Binned index rho to full-res centered rho. Binned pixel j spans
            # full pixels [jB, jB+B-1]; skimage rhos are in binned index units
            # while _candidate_from_rho_theta wants centered full-res units.
            rho = float(
                rho_b * bfac
                - ((w - 1) / 2.0 - (bfac - 1) / 2.0) * np.cos(theta)
                - ((h - 1) / 2.0 - (bfac - 1) / 2.0) * np.sin(theta)
            )
            if any(abs(theta_deg - t0) < 2.0 and abs(rho - r0) < 40.0 for r0, t0 in used):
                continue
            candidate = _candidate_from_rho_theta(rho, theta_deg, data_sub.shape)
            if candidate is None:
                continue
            candidate["source"] = "houghpeaks"
            refined, refine_info = _refine_trail_mask(
                data_sub, bkg_rms_map, candidate, mask_cfg, existing_mask=existing_mask
            )
            conf = _score_candidate(candidate, refined, refine_info, existing_mask)
            if np.count_nonzero(refined) >= min_refined_mask_pixels and conf >= confidence_threshold:
                streak_mask |= refined
                accepted.append({"theta_deg": theta_deg, "rho": rho, "confidence": conf})
                used.append((rho, theta_deg))
                if predictions is not None and refine_info.get("occluded_geometry") is not None:
                    # ponytail: one boolean image per occluded candidate; use
                    # cropped masks if crowded-field memory becomes limiting.
                    predictions.append((refine_info["occluded_geometry"], refined))
    print(f"    Hough peaks accepted {len(accepted)} trail(s).")
    return streak_mask, accepted, {"peaks": n_peaks_total, "accepted_count": len(accepted)}


def _contour_candidates(data_sub, existing_mask, bkg_rms_map, config):
    """Propose trail candidates from contour morphology (ASTRiDE pattern).

    Connected-component borders above threshold, gated on circularity, area
    and radius deviation, with PCA angle + extreme-point span. Catches long
    bright trails the Hough stages splinter or miss, at ~3s/HDU. Survivors
    go through the same strip confirm + score gates as every candidate.
    """
    from skimage import measure as _measure

    cfg = config.get("contour_params", {})
    thresh_sig = float(cfg.get("thresh_sig", 2.5))
    min_span = float(cfg.get("min_span", 150.0))
    shape_cut = float(cfg.get("shape_cut", 0.2))
    area_cut = float(cfg.get("area_cut", 100.0))
    radius_dev_cut = float(cfg.get("radius_dev_cut", 0.5))
    img = np.clip(data_sub, 0.0, None).astype(np.float32)
    if existing_mask is not None:
        img = np.where(existing_mask, 0.0, img)
    if bkg_rms_map is not None:
        # Detection threshold: the contour level is set from the median of the
        # *measured* RMS values, so sentinel pixels are excluded rather than
        # counted into the scale (which ``np.nanmedian`` would not do -- it
        # ignores NaN but not ``inf``).
        scale = robust_rms(bkg_rms_map, default=1.0)
    else:
        scale = 1.0
    if not np.isfinite(scale) or scale <= 0:
        return []
    contours = _measure.find_contours(img, thresh_sig * scale, fully_connected="high")
    h, w = data_sub.shape
    out = []
    for contour in contours:
        y, x = contour[:, 0], contour[:, 1]
        if len(x) < 10:
            continue
        # Shape factor 4*pi*A/P^2 via poly area/perimeter on the path.
        peri = float(np.sum(np.hypot(np.diff(x), np.diff(y)))) + float(np.hypot(x[0] - x[-1], y[0] - y[-1]))
        if peri <= 0:
            continue
        area = 0.5 * abs(float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])))
        if area < area_cut:
            continue
        shape = 4.0 * np.pi * area / (peri * peri)
        if shape > shape_cut:
            continue
        cx, cy = float(np.mean(x)), float(np.mean(y))
        dist = np.hypot(x - cx, y - cy)
        rad = float(np.median(dist))
        if rad <= 0 or float(np.std(dist)) / rad < radius_dev_cut:
            continue
        # PCA angle + extreme span.
        xc, yc = x - cx, y - cy
        theta = 0.5 * np.arctan2(2.0 * float(np.sum(xc * yc)), float(np.sum(xc * xc) - np.sum(yc * yc)))
        direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
        proj = np.column_stack([xc, yc]) @ direction
        span = float(proj.max() - proj.min())
        if span < min_span:
            continue
        p0 = (cx + proj.min() * direction[0], cy + proj.min() * direction[1])
        p1 = (cx + proj.max() * direction[0], cy + proj.max() * direction[1])
        clipped = _clip_line_to_image(
            np.array([(p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0]), direction, data_sub.shape
        )
        if clipped is None:
            continue
        (ex0, ey0), (ex1, ey1) = clipped
        seg = ((float(ex0), float(ey0)), (float(ex1), float(ey1)))
        out.append(
            {
                "segments": [seg],
                "angle_deg": float(_normalize_angle_deg(np.degrees(theta))),
                "endpoints": seg,
                "clipped_endpoints": seg,
                "span": float(np.hypot(ex1 - ex0, ey1 - ey0)),
                "raw_span": span,
                "edge_touches": len(_edges_touched(clipped, data_sub.shape, _EDGE_BUFFER_DEFAULT)),
                "corridor_overlap": 0.0,
                "source": "contours",
            }
        )
    return out


def _detect_streaks_contours(data_sub, bkg_rms_map, existing_mask, config):
    """Confirm contour-morphology candidates with the strip profiler."""
    cfg = config.get("contour_params", {})
    mask_cfg = config.get("mask_params", {})
    if not cfg.get("enable", True):
        return np.zeros(data_sub.shape, dtype=bool), [], {"candidates": 0}
    confidence_threshold = float(config.get("contour_params", {}).get("confidence_threshold", 0.40))
    min_refined_mask_pixels = int(mask_cfg.get("min_mask_pixels", 64))
    confidence_threshold = _adjust_confidence_for_mask(confidence_threshold, existing_mask)
    print("--> Contour-morphology candidate search...")
    candidates = _contour_candidates(data_sub, existing_mask, bkg_rms_map, config)
    print(f"    Contour stage proposed {len(candidates)} candidate(s).")
    streak_mask = np.zeros(data_sub.shape, dtype=bool)
    accepted = []
    for candidate in candidates:
        refined, refine_info = _refine_trail_mask(
            data_sub, bkg_rms_map, candidate, mask_cfg, existing_mask=existing_mask
        )
        conf = _score_candidate(candidate, refined, refine_info, existing_mask)
        if np.count_nonzero(refined) >= min_refined_mask_pixels and conf >= confidence_threshold:
            streak_mask |= refined
            accepted.append({"angle_deg": candidate["angle_deg"], "confidence": conf})
    print(f"    Contour stage accepted {len(accepted)} trail(s).")
    return streak_mask, accepted, {"candidates": len(candidates), "accepted_count": len(accepted)}


def _detect_trails_sparse_ransac(data_sub, bkg_rms_map, existing_mask, config):
    """Detect intermittent trails using iterative RANSAC on residual candidate points."""
    cfg = config.get("sparse_ransac_params", {})
    detect_thresh_sig = float(cfg.get("detect_thresh_sig", 5.0))
    residual_threshold = float(cfg.get("residual_threshold", 2.0))
    min_inliers = int(cfg.get("min_inliers", 10))
    min_length = float(cfg.get("min_length", 100))
    min_line_density = float(cfg.get("min_line_density", 0.2))
    max_trials = int(cfg.get("max_trials", 1000))
    max_trails = int(cfg.get("max_trails", 3))

    if bkg_rms_map is not None:
        # Detection threshold: no measurable RMS means no detectable excess, so
        # unknown pixels get an infinite threshold (never detect).
        thresh = detect_thresh_sig * rms_or_robust(bkg_rms_map, fallback=np.inf)
    else:
        thresh = detect_thresh_sig * mad_std(data_sub, ignore_nan=True)

    residual_mask = (data_sub > thresh) & np.isfinite(data_sub)
    if existing_mask is not None:
        residual_mask &= ~existing_mask

    trail_mask = np.zeros(data_sub.shape, dtype=bool)
    trail_idx = 0
    # ponytail: each rejected model consumes at least min_inliers pixels;
    # heavily crowded fields need a candidate cap before the RANSAC search.
    while trail_idx < max_trails:
        coords = np.argwhere(residual_mask)
        if len(coords) < min_inliers:
            break

        print(f"--> Using sparse RANSAC trail detection ({len(coords)} candidate points)")
        try:
            _, inliers = ransac(
                coords,
                LineModelND,
                min_samples=2,
                residual_threshold=residual_threshold,
                max_trials=max_trials,
                rng=0,
            )
        except Exception as e:
            print(f"    Sparse RANSAC failed: {e}")
            break

        if inliers is None or np.count_nonzero(inliers) < min_inliers:
            break

        inlier_coords = coords[inliers]
        diffs = inlier_coords.max(axis=0) - inlier_coords.min(axis=0)
        sort_dim = int(np.argmax(diffs))
        p0 = inlier_coords[np.argmin(inlier_coords[:, sort_dim])]
        p1 = inlier_coords[np.argmax(inlier_coords[:, sort_dim])]
        length = np.hypot(*(p1 - p0))
        density = np.count_nonzero(inliers) / max(length, 1.0)
        if length < min_length or density < min_line_density:
            residual_mask[inlier_coords[:, 0], inlier_coords[:, 1]] = False
            continue

        rr, cc = line(int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1]))
        current = np.zeros_like(trail_mask)
        current[rr, cc] = True
        dilation_r = int(cfg.get("dilation_radius", 1))
        current = dilation(current, footprint=disk(dilation_r)) if dilation_r > 0 else current
        trail_mask |= current
        residual_mask[inlier_coords[:, 0], inlier_coords[:, 1]] = False
        residual_mask &= ~current
        trail_idx += 1
        print(
            f"    Sparse RANSAC found trail {trail_idx}: length={length:.1f} px, "
            f"inliers={np.count_nonzero(inliers)}, density={density:.3f}"
        )

    return trail_mask


def _label_components(mask):
    """Connected components of a boolean mask plus their sizes, or ``None`` if empty."""
    labeled, n_components = label(np.asarray(mask, dtype=bool), connectivity=2, return_num=True)
    if n_components == 0:
        return None
    sizes = np.bincount(labeled.ravel(), minlength=n_components + 1)
    return labeled, n_components, sizes


def _keep_by_component(labeled, n_components, sizes, min_pixels, keep_component):
    """Apply ``keep_component(index, size)`` to each labelled component.

    ``keep_component`` is consulted only for components of at least ``min_pixels``
    pixels. A smaller one is always kept: a size floor is a poor judge on its own,
    and the profile gate downstream has the geometry to be a better one.
    """
    keep = np.ones(n_components + 1, dtype=bool)
    keep[0] = False
    for index in range(1, n_components + 1):
        if sizes[index] < int(min_pixels):
            continue
        keep[index] = bool(keep_component(index, int(sizes[index])))
    return keep[labeled]


def _drop_bright_components(mask, data_sub, bkg_rms_map, max_sigma, min_pixels=24, existing_mask=None):
    """Drop streak components too bright to be a trail.

    Measured on real MegaCam amps, as a multiple of the local background RMS:

    | feature | p90 of the component |
    |---|---|
    | unconfirmed linear feature (996195p HDU 35/36) | 3.0 / 3.2 sigma |
    | bright star arm (1013721p HDU 33) | 155 sigma |
    | near-saturated column group (1013719p/1013720p HDU 5) | 1138 / 1345 sigma |

    The last case is a group of columns at 97-99.5% of the ``SATURATE`` level. The
    saturation stage tests against the keyword itself, so it misses them by a
    hair, and ``bad.py`` works from the flat, where those columns are ordinary.
    Nothing upstream owns them, and a feature over 1000 sigma above the sky is not
    a trail.

    This is a stopgap for a threshold that arguably belongs in the saturation
    stage; it lives here because it cannot mask a real trail, and because a
    satellite trail in a 560 s exposure has no headroom to spare. 6x margin
    above the two trails measured so far, 7x below the brightest false positive.
    """
    if max_sigma is None or bkg_rms_map is None or not np.any(mask):
        return mask
    components = _label_components(mask)
    if components is None:
        return mask
    labeled, n_components, sizes = components

    def keep_component(index, _size):
        sel = labeled == index
        values = data_sub[sel].astype(np.float64)
        noise = bkg_rms_map[sel].astype(np.float64)
        good = np.isfinite(values) & np.isfinite(noise) & (noise > 0)
        if existing_mask is not None:
            good &= ~existing_mask[sel]
        if not np.any(good):
            return True  # nothing measurable here, so nothing to judge it by
        sigma = float(np.percentile(values[good] / noise[good], 90))
        return sigma <= float(max_sigma)

    return _keep_by_component(labeled, n_components, sizes, min_pixels, keep_component)


def _drop_premasked_components(mask, existing_mask, max_fraction, min_pixels=24):
    """Drop streak components that lie mostly inside the mask the pipeline already has.

    A component the upstream stages already flagged is not a new finding, and on
    real MegaCam amps the dominant false positive is a saturated star's bleed:
    45% of such a component's pixels are already masked, against 2% for a real
    satellite trail. Both the product (median ``data_sub`` 1675 e- vs 45 e-,
    i.e. 11x the sky) and the geometry (a hairline with a 1-row step where the
    bleed wing changes intensity, versus a constant-width band) separate the two
    classes, but the pre-masked fraction is free -- both masks are already in
    hand here -- and separates them by a factor of 22.

    Applied per component, not to the whole mask, so a stage that returns a bleed
    *and* a trail keeps the trail.
    """
    if existing_mask is None or max_fraction is None or max_fraction >= 1.0:
        return mask
    if not np.any(mask):
        return mask
    components = _label_components(mask)
    if components is None:
        return mask
    labeled, n_components, sizes = components
    overlap = np.bincount(labeled.ravel(), weights=existing_mask.ravel().astype(np.float64), minlength=n_components + 1)
    return _keep_by_component(
        labeled,
        n_components,
        sizes,
        min_pixels,
        lambda index, _size: (overlap[index] / sizes[index]) <= float(max_fraction),
    )


def _gate_mask_by_profile(mask, max_half_width=6.0, min_pixels=24):
    """Keep pixels that lie on a thin line. Drop a component that is too wide.

    The Hough or Radon vote is a proposal. A component is a streak only when its
    pixels concentrate about one axis within ``max_half_width``. The returned
    mask is that band, not the whole voted corridor.
    """
    components = _label_components(mask)
    if components is None:
        return np.zeros(np.shape(mask), dtype=bool)
    labeled, n_components, _sizes = components
    out = np.zeros(labeled.shape, dtype=bool)
    for index in range(1, n_components + 1):
        ys, xs = np.nonzero(labeled == index)
        if ys.size < int(min_pixels):
            continue
        y = ys.astype(np.float64)
        x = xs.astype(np.float64)
        yc = y - y.mean()
        xc = x - x.mean()
        cov_xx = float(np.dot(xc, xc))
        cov_yy = float(np.dot(yc, yc))
        cov_xy = float(np.dot(xc, yc))
        theta = 0.5 * float(np.arctan2(2.0 * cov_xy, cov_xx - cov_yy))
        direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
        transverse = xc * normal[0] + yc * normal[1]
        along = xc * direction[0] + yc * direction[1]
        med = float(np.median(transverse))
        mad = 1.4826 * float(np.median(np.abs(transverse - med)))
        if not np.isfinite(mad) or mad < 0.5:
            mad = 0.5
        if mad > float(max_half_width):
            continue
        span = float(along.max() - along.min())
        if span < 4.0 * max(mad, 1.0):
            continue
        half = min(float(max_half_width), max(2.0 * mad, 1.5))
        keep = np.abs(transverse - med) <= half
        if int(np.count_nonzero(keep)) < int(min_pixels):
            continue
        out[ys[keep], xs[keep]] = True
    return out


def detect_streaks(data_sub, bkg_rms_map, existing_mask, config):
    """
    Detect linear streaks via the production ``auto_ground`` path plus optional sparse RANSAC.

    Raises ValueError on unknown modes (fail fast on config typos).
    Mode ``auto_ground``: binned-Hough peaks and
    contour morphology, then residual RANSAC.

    Benchmark-only Frangi lives in ``benchmarks.frangi_legacy``.
    """
    mode = _resolve_streak_mode(config)
    if not config.get("enable", False):
        print("Streak masking disabled in main config.")
        return np.zeros(data_sub.shape, dtype=bool)

    streak_mask_bool = np.zeros(data_sub.shape, dtype=bool)
    debug_info = {"mode": mode, "houghpeaks": {}, "contours": {}, "sparse_ransac": None}
    min_streak_px = int(config.get("mask_params", {}).get("min_mask_pixels", 64))
    # A component the upstream stages already flagged is not a new finding, and a
    # component orders of magnitude above the sky is not a trail.
    max_premasked = config.get("mask_params", {}).get("max_premasked_fraction", 0.25)
    max_sigma = config.get("mask_params", {}).get("max_component_sigma", 20.0)

    def _veto(mask):
        mask = _drop_bright_components(mask, data_sub, bkg_rms_map, max_sigma, min_streak_px, existing_mask)
        return _drop_premasked_components(mask, existing_mask, max_premasked, min_streak_px)

    print(f"  Detecting streaks (mode: {mode})...")
    streak_t0 = time.time()
    predictions = []
    hp_mask, hp_accepted, hp_debug = _detect_streaks_houghpeaks(
        data_sub, bkg_rms_map, existing_mask, config, predictions
    )
    hp_mask = _veto(hp_mask)
    streak_mask_bool |= hp_mask
    debug_info["houghpeaks"] = {"accepted": hp_accepted, **hp_debug}
    ct_mask, ct_accepted, ct_debug = _detect_streaks_contours(data_sub, bkg_rms_map, existing_mask, config)
    ct_mask = _veto(ct_mask)
    streak_mask_bool |= ct_mask
    debug_info["contours"] = {"accepted": ct_accepted, **ct_debug}
    n_prescreen_accepted = len(hp_accepted) + len(ct_accepted)
    print(
        f"    [streak] prescreen done in {time.time() - streak_t0:.1f}s: "
        f"{n_prescreen_accepted} accepted, {int(np.count_nonzero(streak_mask_bool))} px."
    )
    # The Radon rescue used to sit here as the *sensitive* stage -- the thing that
    # finds what the cheap prescreen misses. It is gone. Measured on 83 real amps
    # it accepted on 3, all three false positives, while houghpeaks explained the
    # historical real detections. On injected trails at 4/6/8/12 sigma over
    # two lengths and two seeds it changed recall_line by +0.000 in 8 of 8 cells.
    # It cost 121.7 s/amp of a 125.7 s/amp stage. See benchmarks/streak_recall_floor.py
    # and benchmarks/streak_stage_sweep.py; both must be re-run, not assumed, if
    # anyone proposes restoring a faint-trail stage.

    if config.get("enable_sparse_ransac", True):
        residual_existing = streak_mask_bool.copy()
        if existing_mask is not None:
            residual_existing |= existing_mask
        print(f"    [streak] sparse RANSAC pass (t+{time.time() - streak_t0:.1f}s)...")
        sparse_mask = _detect_trails_sparse_ransac(data_sub, bkg_rms_map, residual_existing, config)
        sparse_mask = _veto(sparse_mask)
        streak_mask_bool |= sparse_mask
        debug_info["sparse_ransac"] = int(np.count_nonzero(sparse_mask))
    else:
        debug_info["sparse_ransac"] = 0

    if config.get("profile_accept", True):
        mask_cfg = config.get("mask_params", {})
        half_width = float(mask_cfg.get("max_support_width", 16)) / 2.0
        min_pixels = int(mask_cfg.get("min_mask_pixels", 64))
        streak_mask_bool = _gate_mask_by_profile(streak_mask_bool, max_half_width=half_width, min_pixels=min_pixels)

    if predictions:
        # Predictions are added after every evidence gate. They cannot reject
        # a confirmed component or supply support for another prediction.
        inferred_mask = np.zeros_like(streak_mask_bool)
        for (yy, xx, occluded), candidate_mask in predictions:
            # Unrelated crossing trails cannot revive a vetoed candidate.
            confirmed = _veto(candidate_mask)
            if config.get("profile_accept", True):
                confirmed = _gate_mask_by_profile(confirmed, max_half_width=half_width, min_pixels=min_pixels)
            confirmed &= streak_mask_bool
            mask_cfg = config.get("mask_params", {})
            max_gap = int(mask_cfg.get("max_row_gap", 0))
            row_hits = _largest_contiguous_run(
                np.any(confirmed[yy, xx], axis=1),
                max_gap=max_gap,
                min_run_length=int(mask_cfg.get("min_row_run", 0)) if max_gap > 0 else 0,
                valid=~occluded,
            )
            if np.count_nonzero(row_hits) < int(mask_cfg.get("min_row_hits", 8)) or np.mean(row_hits) < float(
                mask_cfg.get("min_row_hit_fraction", 0.5)
            ):
                continue
            rows = np.flatnonzero(row_hits)
            if not rows.size:
                continue
            selected = occluded.copy()
            selected[: rows[0]] = False
            selected[rows[-1] + 1 :] = False
            inferred_mask[yy[selected], xx[selected]] = True
        streak_mask_bool |= inferred_mask

    if existing_mask is not None:
        num_new_pixels = int(np.count_nonzero(streak_mask_bool & (~existing_mask)))
    else:
        num_new_pixels = int(np.count_nonzero(streak_mask_bool))

    if config.get("debug", False):
        config["_last_run"] = debug_info

    streak_elapsed = time.time() - streak_t0
    if num_new_pixels > 0:
        print(f"  Final streak mask includes {num_new_pixels} new pixels (Mode: {mode}, {streak_elapsed:.1f}s).")
    else:
        print(f"  No new streak pixels added by mode '{mode}' ({streak_elapsed:.1f}s).")

    return streak_mask_bool
