import concurrent.futures
import warnings

import numpy as np
import scipy.ndimage as ndi
from astropy.stats import mad_std
from skimage.draw import line
from skimage.feature import canny
from skimage.filters import apply_hysteresis_threshold, frangi
from skimage.measure import LineModelND, label, ransac, regionprops
from skimage.morphology import dilation, disk, white_tophat
from skimage.transform import probabilistic_hough_line, radon


def _normalize_angle_deg(angle_deg):
    """Normalize an angle to the [0, 180) degree range."""
    return (angle_deg + 180.0) % 180.0


def _extract_valid_pixels(data, existing_mask=None):
    valid = np.isfinite(data)
    if existing_mask is not None:
        valid &= ~existing_mask
    return valid


def _robust_scale_image(data_sub, existing_mask, percentiles):
    """Rescale sky-subtracted data into a 0-1 range for edge detection."""
    valid = _extract_valid_pixels(data_sub, existing_mask)
    if np.sum(valid) < 100:
        return np.zeros_like(data_sub, dtype=np.float32)

    sampled = data_sub[valid]
    step = max(1, sampled.size // 100000)
    lo, hi = np.percentile(sampled[::step], percentiles)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(data_sub, dtype=np.float32)

    scaled = (data_sub - lo) / (hi - lo)
    scaled = np.clip(scaled, 0.0, 1.0)
    scaled[~np.isfinite(scaled)] = 0.0
    return scaled.astype(np.float32)


def _prepare_streak_image(data_sub, existing_mask):
    """Suppress compact-source residuals before edge extraction."""
    positive = np.clip(data_sub, 0.0, None)
    filtered = ndi.median_filter(positive, size=15)
    prepared = np.clip(positive - filtered, 0.0, None)
    prepared = white_tophat(prepared, footprint=disk(3))
    if existing_mask is not None:
        prepared = np.where(existing_mask, 0.0, prepared)
    return prepared.astype(np.float32)


def _resolve_streak_mode(config):
    """Resolve the public streak mode while preserving method backward compatibility."""
    requested = (config.get("mode") or config.get("method") or "auto_ground").lower()
    aliases = {
        "satdet": "satdet_only",
        "frangi_legacy": "legacy_compare",
    }
    return aliases.get(requested, requested)


def _extract_multiscale_segments(data_sub, existing_mask, cfg):
    """Extract Hough segments across several smoothing scales."""
    percentiles = tuple(cfg.get("rescale_percentiles", [4.5, 93.0]))
    gaussian_sigmas = cfg.get("gaussian_sigmas")
    if gaussian_sigmas is None:
        base_sigma = float(cfg.get("gaussian_sigma", 2.0))
        gaussian_sigmas = sorted(set([max(0.0, base_sigma * factor) for factor in (0.75, 1.0, 1.5)]))
    canny_low = float(cfg.get("canny_low_threshold", 0.1))
    canny_high = float(cfg.get("canny_high_threshold", 0.35))
    min_edge_perimeter = float(cfg.get("small_edge_perimeter", 60.0))
    line_len = int(cfg.get("hough_min_line_length", 120))
    line_gap = int(cfg.get("hough_max_line_gap", 30))
    hough_threshold = int(cfg.get("hough_threshold", 10))
    rng_seed = int(cfg.get("hough_rng_seed", 0))

    prepared = _prepare_streak_image(data_sub, existing_mask)
    positive_raw = np.clip(data_sub, 0.0, None).astype(np.float32)
    if existing_mask is not None:
        positive_raw = np.where(existing_mask, 0.0, positive_raw)
    source_images = [("prepared", prepared), ("raw", positive_raw)]
    all_segments = []
    debug_scales = []
    for source_idx, (source_name, source_image) in enumerate(source_images):
        for idx, sigma in enumerate(gaussian_sigmas):
            scaled = _robust_scale_image(source_image, existing_mask, percentiles)
            if sigma > 0:
                scaled = ndi.gaussian_filter(scaled, sigma)
            edges = canny(scaled, sigma=0.0, low_threshold=canny_low, high_threshold=canny_high)
            if existing_mask is not None:
                edges &= ~existing_mask
            edges = _prune_small_edges(edges, min_edge_perimeter)
            segments = probabilistic_hough_line(
                edges,
                threshold=hough_threshold,
                line_length=line_len,
                line_gap=line_gap,
                rng=rng_seed + idx + 100 * source_idx,
            )
            all_segments.extend(segments)
            debug_scales.append(
                {
                    "source": source_name,
                    "sigma": float(sigma),
                    "segments": len(segments),
                    "edge_pixels": int(np.sum(edges)),
                }
            )

    return all_segments, debug_scales


def _prune_small_edges(edge_mask, min_perimeter):
    """Drop tiny edge fragments before Hough extraction."""
    if min_perimeter <= 0:
        return edge_mask

    labeled = label(edge_mask, connectivity=2)
    cleaned = np.zeros_like(edge_mask, dtype=bool)
    for region in regionprops(labeled):
        if region.perimeter >= min_perimeter:
            coords = region.coords
            cleaned[coords[:, 0], coords[:, 1]] = True
    return cleaned


def _segment_angle_and_rho(segment):
    """Return the normalized line angle and midpoint rho for a segment."""
    (x0, y0), (x1, y1) = segment
    dx = x1 - x0
    dy = y1 - y0
    angle_deg = _normalize_angle_deg(np.degrees(np.arctan2(dy, dx)))
    normal_theta = np.radians(_normalize_angle_deg(angle_deg + 90.0))
    mx = 0.5 * (x0 + x1)
    my = 0.5 * (y0 + y1)
    rho = mx * np.cos(normal_theta) + my * np.sin(normal_theta)
    return angle_deg, rho


def _cluster_segments(segments, angle_tol_deg, rho_tol_px):
    """Greedily cluster Hough segments by angle and offset."""
    clusters = []
    for segment in segments:
        angle_deg, rho = _segment_angle_and_rho(segment)
        assigned = False
        for cluster in clusters:
            angle_diff = abs(angle_deg - cluster["angle_deg"])
            angle_diff = min(angle_diff, 180.0 - angle_diff)
            if angle_diff <= angle_tol_deg and abs(rho - cluster["rho"]) <= rho_tol_px:
                cluster["segments"].append(segment)
                cluster["angle_deg"] = np.mean([_segment_angle_and_rho(s)[0] for s in cluster["segments"]])
                cluster["rho"] = np.mean([_segment_angle_and_rho(s)[1] for s in cluster["segments"]])
                assigned = True
                break
        if not assigned:
            clusters.append({"segments": [segment], "angle_deg": angle_deg, "rho": rho})
    return clusters


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


def _representative_line(cluster, shape):
    """Convert a cluster of Hough segments into one clipped representative line."""
    endpoints = np.asarray(cluster["segments"], dtype=np.float32).reshape(-1, 2)
    midpoint = endpoints.mean(axis=0)
    theta = np.radians(cluster["angle_deg"])
    direction = np.array([np.cos(theta), np.sin(theta)], dtype=np.float32)
    projections = (endpoints - midpoint) @ direction
    p0 = midpoint + projections.min() * direction
    p1 = midpoint + projections.max() * direction
    clipped = _clip_line_to_image(midpoint, direction, shape)
    return {
        "endpoints": ((float(p0[0]), float(p0[1])), (float(p1[0]), float(p1[1]))),
        "clipped_endpoints": clipped,
        "direction": direction,
        "midpoint": midpoint,
        "raw_span": float(projections.max() - projections.min()),
    }


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


def _line_corridor_mask(endpoints, shape, radius):
    """Create a coarse corridor mask around a representative line."""
    (x0, y0), (x1, y1) = endpoints
    rr, cc = line(int(round(y0)), int(round(x0)), int(round(y1)), int(round(x1)))
    mask = np.zeros(shape, dtype=bool)
    mask[rr, cc] = True
    if radius > 0:
        mask = dilation(mask, footprint=disk(radius))
    return mask


def _build_satdet_candidates(segments, data_sub, shape, cfg, existing_mask):
    """Cluster and filter Hough segments into plausible trail candidates."""
    clusters = _cluster_segments(
        segments,
        float(cfg.get("cluster_angle_tol_deg", 3.0)),
        float(cfg.get("cluster_rho_tol_px", 30.0)),
    )
    edge_buffer = int(cfg.get("edge_buffer", 32))
    min_segments = int(cfg.get("min_cluster_segments", 3))
    max_existing_mask_fraction = float(cfg.get("max_existing_mask_fraction", 0.6))
    max_candidates = int(cfg.get("max_candidates", 8))
    min_interior_span = float(cfg.get("min_interior_span", 120.0))
    min_edge_touches = int(cfg.get("min_edge_touches", 1))
    min_segment_density = float(cfg.get("min_segment_density", 0.015))
    corridor_radius = int(cfg.get("candidate_corridor_radius", 12))
    prepared = _prepare_streak_image(data_sub, existing_mask)
    candidates = []

    for cluster in clusters:
        if len(cluster["segments"]) < min_segments:
            continue

        rep = _representative_line(cluster, shape)
        clipped = rep["clipped_endpoints"]
        if clipped is None:
            continue

        raw_endpoints = np.asarray(cluster["segments"], dtype=np.float32).reshape(-1, 2)
        raw_edge_touches = len(_edges_touched(raw_endpoints, shape, edge_buffer))
        raw_span = rep["raw_span"]
        segment_density = len(cluster["segments"]) / max(raw_span, 1.0)
        if segment_density < min_segment_density:
            continue

        (c0, c1) = clipped
        span = np.hypot(c1[0] - c0[0], c1[1] - c0[1])
        if raw_edge_touches < min_edge_touches and raw_span < min_interior_span:
            continue

        if existing_mask is not None:
            corridor = _line_corridor_mask(clipped, shape, corridor_radius)
            masked_fraction = np.mean(existing_mask[corridor]) if np.any(corridor) else 0.0
            if masked_fraction > max_existing_mask_fraction:
                continue
        else:
            corridor = _line_corridor_mask(clipped, shape, corridor_radius)
            masked_fraction = 0.0

        corridor_signal = prepared[corridor]
        corridor_signal = corridor_signal[np.isfinite(corridor_signal)]
        if corridor_signal.size > 0:
            corridor_response = float(np.percentile(corridor_signal, 90))
            corridor_mean = float(np.mean(corridor_signal))
        else:
            corridor_response = 0.0
            corridor_mean = 0.0

        candidates.append(
            {
                "segments": cluster["segments"],
                "angle_deg": cluster["angle_deg"],
                "endpoints": rep["endpoints"],
                "clipped_endpoints": clipped,
                "span": span,
                "raw_span": raw_span,
                "edge_touches": raw_edge_touches,
                "corridor_overlap": float(masked_fraction),
                "corridor_response": corridor_response,
                "corridor_mean": corridor_mean,
            }
        )

    candidates.sort(
        key=lambda item: (
            item["corridor_response"],
            item["corridor_mean"],
            len(item["segments"]),
            item["span"],
            item["edge_touches"],
            -item["corridor_overlap"],
        ),
        reverse=True,
    )
    if max_candidates > 0:
        candidates = candidates[:max_candidates]

    print(f"    Clustered {len(segments)} Hough segments into {len(candidates)} trail candidate(s).")
    return candidates


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


def _largest_contiguous_run(mask_1d):
    """Return only the longest contiguous run in a 1D mask."""
    if not np.any(mask_1d):
        return mask_1d

    labels, n = ndi.label(mask_1d.astype(np.uint8))
    if n <= 1:
        return mask_1d

    best_label = 1 + np.argmax([np.sum(labels == idx) for idx in range(1, n + 1)])
    return labels == best_label


def _refine_trail_mask(data_sub, bkg_rms_map, candidate, mask_cfg, existing_mask=None):
    """Refine a Hough candidate by fitting a trail-aligned strip mask."""
    if bkg_rms_map is not None:
        safe_rms = np.where((bkg_rms_map > 0) & np.isfinite(bkg_rms_map), bkg_rms_map, np.nanmedian(bkg_rms_map))
        detect_img = data_sub / np.where(safe_rms > 0, safe_rms, 1.0)
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

    strip = _sample_trail_strip(
        detect_img,
        candidate["clipped_endpoints"],
        strip_length,
        strip_width,
        interpolation_order,
    )
    if strip is None:
        return np.zeros(data_sub.shape, dtype=bool)

    sampled = strip["sampled"]
    inside = strip["inside"]
    width_axis = np.arange(sampled.shape[1], dtype=np.float32)
    center_col = 0.5 * (sampled.shape[1] - 1)
    sideband = np.abs(width_axis - center_col) >= max(3.0, 0.25 * sampled.shape[1])
    background_pixels = sampled[:, sideband]
    bg_values = background_pixels[np.isfinite(background_pixels)]
    if bg_values.size < 50:
        return np.zeros(data_sub.shape, dtype=bool), {"support_width": 0, "row_hit_fraction": 0.0, "mask_pixels": 0}

    bg_med = np.median(bg_values)
    bg_std = mad_std(bg_values, ignore_nan=True)
    if not np.isfinite(bg_std) or bg_std <= 1e-6:
        bg_std = np.std(bg_values)
    if not np.isfinite(bg_std) or bg_std <= 1e-6:
        bg_std = np.nanstd(sampled)
    if not np.isfinite(bg_std) or bg_std <= 1e-6:
        bg_std = 1e-3

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        row_bg = np.nanmedian(background_pixels, axis=1)
    row_bg = np.where(np.isfinite(row_bg), row_bg, bg_med)
    centered = sampled - row_bg[:, np.newaxis]

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
        return np.zeros(data_sub.shape, dtype=bool), {"support_width": 0, "row_hit_fraction": 0.0, "mask_pixels": 0}
    if np.sum(support_cols) > max_support_width:
        return np.zeros(data_sub.shape, dtype=bool), {
            "support_width": int(np.sum(support_cols)),
            "row_hit_fraction": 0.0,
            "mask_pixels": 0,
        }

    hot_pixels = hot_pixels & support_cols[np.newaxis, :]
    row_hits = np.sum(hot_pixels, axis=1) > 0
    row_hits = ndi.binary_closing(row_hits, structure=np.ones(2 * padding + 1, dtype=bool))
    row_hits = _largest_contiguous_run(row_hits)
    if np.sum(row_hits) < min_row_hits:
        return np.zeros(data_sub.shape, dtype=bool), {
            "support_width": int(np.sum(support_cols)),
            "row_hit_fraction": float(np.mean(row_hits)),
            "mask_pixels": 0,
        }
    row_hit_fraction = float(np.mean(row_hits))
    if row_hit_fraction < min_row_hit_fraction:
        return np.zeros(data_sub.shape, dtype=bool), {
            "support_width": int(np.sum(support_cols)),
            "row_hit_fraction": row_hit_fraction,
            "mask_pixels": 0,
        }

    refined_strip = np.zeros_like(hot_pixels, dtype=bool)
    refined_strip[row_hits, :] = support_cols[np.newaxis, :]
    refined_strip &= inside

    if padding > 0:
        refined_strip = ndi.binary_dilation(refined_strip, structure=np.ones((3, 2 * padding + 1), dtype=bool))

    yy = np.rint(strip["y_coords"][refined_strip]).astype(int)
    xx = np.rint(strip["x_coords"][refined_strip]).astype(int)
    valid = (yy >= 0) & (yy < data_sub.shape[0]) & (xx >= 0) & (xx < data_sub.shape[1])
    if np.sum(valid) < min_mask_pixels:
        return np.zeros(data_sub.shape, dtype=bool), {
            "support_width": int(np.sum(support_cols)),
            "row_hit_fraction": row_hit_fraction,
            "mask_pixels": int(np.sum(valid)),
        }

    mask = np.zeros(data_sub.shape, dtype=bool)
    mask[yy[valid], xx[valid]] = True
    return mask, {
        "support_width": int(np.sum(support_cols)),
        "row_hit_fraction": row_hit_fraction,
        "mask_pixels": int(np.sum(mask)),
    }


def _score_candidate(candidate, refined_mask, refine_info, existing_mask):
    """Score a refined candidate using support, continuity, and overlap penalties."""
    if refined_mask is None or np.sum(refined_mask) == 0:
        return 0.0

    span = max(float(candidate.get("span", 1.0)), 1.0)
    support_density = np.sum(refined_mask) / span
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
        "edge_touches": len(_edges_touched(clipped, shape, int(16))),
    }


def _detect_streaks_mrt_like(data_sub, bkg_rms_map, existing_mask, config):
    """Run a lightweight MRT-like Radon rescue for low-confidence ground-based streaks."""
    cfg = config.get("mrt_rescue_params", {})
    mask_cfg = config.get("mask_params", {})
    theta_step = float(cfg.get("theta_step_deg", 1.0))
    peak_threshold = float(cfg.get("peak_threshold_sig", 4.5))
    max_candidates = int(cfg.get("max_candidates", 4))
    confidence_threshold = float(cfg.get("confidence_threshold", 0.35))

    if bkg_rms_map is not None:
        safe_rms = np.where((bkg_rms_map > 0) & np.isfinite(bkg_rms_map), bkg_rms_map, np.nanmedian(bkg_rms_map))
        normalized = data_sub / np.maximum(safe_rms, 1e-6)
    else:
        normalized = data_sub
    if existing_mask is not None:
        normalized = np.where(existing_mask, 0.0, normalized)
    normalized = np.clip(normalized, 0.0, None)
    normalized -= np.nanmedian(normalized)
    normalized = np.clip(normalized, 0.0, None)

    thetas = np.arange(0.0, 180.0, theta_step, dtype=np.float32)
    if thetas.size == 0:
        return np.zeros(data_sub.shape, dtype=bool), [], {"theta_step_deg": theta_step, "peaks": 0}

    sinogram = radon(normalized, theta=thetas, circle=False)
    med = np.nanmedian(sinogram, axis=0, keepdims=True)
    sigma = mad_std(sinogram - med, axis=0, ignore_nan=True)
    sigma = np.where(np.isfinite(sigma) & (sigma > 1e-6), sigma, 1.0)
    snr = (sinogram - med) / sigma[np.newaxis, :]

    flat_indices = np.argpartition(snr.ravel(), -max_candidates)[-max_candidates:]
    order = flat_indices[np.argsort(snr.ravel()[flat_indices])[::-1]]

    streak_mask = np.zeros(data_sub.shape, dtype=bool)
    accepted = []
    used = []
    rho_coords = np.arange(sinogram.shape[0], dtype=np.float32) - 0.5 * (sinogram.shape[0] - 1)
    for flat_idx in order:
        rho_idx, theta_idx = np.unravel_index(int(flat_idx), snr.shape)
        score_peak = float(snr[rho_idx, theta_idx])
        if not np.isfinite(score_peak) or score_peak < peak_threshold:
            continue
        rho = float(rho_coords[rho_idx])
        theta_deg = float(thetas[theta_idx])
        if any(abs(theta_deg - t0) < 2.0 and abs(rho - r0) < 20.0 for r0, t0 in used):
            continue

        candidate = _candidate_from_rho_theta(rho, theta_deg, data_sub.shape)
        if candidate is None:
            continue
        refined, refine_info = _refine_trail_mask(
            data_sub, bkg_rms_map, candidate, mask_cfg, existing_mask=existing_mask
        )
        conf = _score_candidate(candidate, refined, refine_info, existing_mask)
        if conf >= confidence_threshold:
            streak_mask |= refined
            accepted.append({"rho": rho, "theta_deg": theta_deg, "peak_snr": score_peak, "confidence": conf})
            used.append((rho, theta_deg))

    return streak_mask, accepted, {"theta_step_deg": theta_step, "accepted_count": len(accepted)}


def _detect_streaks_satdet(data_sub, bkg_rms_map, existing_mask, config):
    """
    Detect streaks using a satdet-inspired Hough candidate extractor plus strip refiner.
    """
    cfg = config.get("satdet_params", {})
    mask_cfg = config.get("mask_params", {})
    min_refined_mask_pixels = int(mask_cfg.get("min_mask_pixels", 64))
    confidence_threshold = float(cfg.get("confidence_threshold", 0.40))
    min_segment_accept = int(cfg.get("min_segment_accept", 3))
    if existing_mask is not None:
        confidence_threshold += min(0.25, 20.0 * float(np.mean(existing_mask)))

    print("--> Using satdet-inspired multi-scale streak detection")
    segments, debug_scales = _extract_multiscale_segments(data_sub, existing_mask, cfg)
    print(f"    Probabilistic Hough returned {len(segments)} segment(s).")
    if not segments:
        return np.zeros(data_sub.shape, dtype=bool), [], {"scales": debug_scales, "accepted_count": 0}

    candidates = _build_satdet_candidates(segments, data_sub, data_sub.shape, cfg, existing_mask)
    streak_mask = np.zeros(data_sub.shape, dtype=bool)
    accepted = []
    for candidate in candidates:
        refined, refine_info = _refine_trail_mask(
            data_sub, bkg_rms_map, candidate, mask_cfg, existing_mask=existing_mask
        )
        confidence = _score_candidate(candidate, refined, refine_info, existing_mask)
        if (
            np.sum(refined) >= min_refined_mask_pixels
            and confidence >= confidence_threshold
            and len(candidate["segments"]) >= min_segment_accept
        ):
            streak_mask |= refined
            accepted.append(
                {
                    "angle_deg": candidate["angle_deg"],
                    "span": candidate["span"],
                    "segments": len(candidate["segments"]),
                    "confidence": confidence,
                    "support_width": refine_info.get("support_width", 0),
                }
            )

    print(f"    satdet-style refinement produced {np.sum(streak_mask)} streak pixels.")
    return (
        streak_mask,
        accepted,
        {"scales": debug_scales, "accepted_count": len(accepted), "candidates": len(candidates)},
    )


def _apply_frangi_filter(tophat_img, sigmas, black_ridges, block_size, pad, img_rows, img_cols):
    """Legacy Frangi helper retained for benchmark comparisons."""
    print(f"    Applying Frangi Filter (sigmas={sigmas})...")
    ridge_map = np.zeros_like(tophat_img)

    if img_rows > block_size or img_cols > block_size:
        print(
            f"    Image is large ({img_rows}x{img_cols}). "
            f"Using parallel block processing (size={block_size}, pad={pad})."
        )

        def process_block(r, c):
            r_start_pad = max(0, r - pad)
            r_end_pad = min(img_rows, r + block_size + pad)
            c_start_pad = max(0, c - pad)
            c_end_pad = min(img_cols, c + block_size + pad)
            block = tophat_img[r_start_pad:r_end_pad, c_start_pad:c_end_pad]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                block_ridge = frangi(block, sigmas=sigmas, black_ridges=black_ridges)

            valid_r_start = r - r_start_pad
            valid_r_end = valid_r_start + min(block_size, img_rows - r)
            valid_c_start = c - c_start_pad
            valid_c_end = valid_c_start + min(block_size, img_cols - c)
            return r, c, block_ridge[valid_r_start:valid_r_end, valid_c_start:valid_c_end]

        futures = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for r in range(0, img_rows, block_size):
                for c in range(0, img_cols, block_size):
                    futures.append(executor.submit(process_block, r, c))
            for future in concurrent.futures.as_completed(futures):
                r, c, block = future.result()
                ridge_map[r : r + block.shape[0], c : c + block.shape[1]] = block
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ridge_map = frangi(tophat_img, sigmas=sigmas, black_ridges=black_ridges)

    return ridge_map


def _calculate_hysteresis_thresholds(cfg, tophat_img, ridge_map, existing_mask):
    """Legacy Frangi threshold helper retained for comparison runs."""
    high_thresh = cfg.get("high_threshold")
    low_thresh = cfg.get("low_threshold")

    if high_thresh is None or low_thresh is None:
        high_threshold_sig = cfg.get("high_threshold_sig", 3.0)
        low_threshold_sig = cfg.get("low_threshold_sig", 1.0)

        bkg_mask = tophat_img < 1.0
        if existing_mask is not None:
            bkg_mask &= ~existing_mask
        if np.sum(bkg_mask) < 1000:
            bkg_mask = np.ones_like(tophat_img, dtype=bool)

        bkg_ridge = ridge_map[bkg_mask]
        if len(bkg_ridge) > 1000:
            p50, p99 = np.percentile(bkg_ridge, [50, 99])
            tail_spread = max(p99 - p50, 1e-9)
            high_thresh = max(p50 + high_threshold_sig * tail_spread, 1e-6)
            low_thresh = max(p50 + low_threshold_sig * tail_spread, 1e-7)
        else:
            high_thresh = 1e-5
            low_thresh = 5e-6

    return low_thresh, high_thresh


def _filter_streak_regions(hyst_mask, min_area, min_elongation, existing_mask, data_sub_shape):
    """Legacy Frangi region filter with explicit elongation gating."""
    img_rows, img_cols = data_sub_shape
    labeled_mask, num_labels = label(hyst_mask, connectivity=2, return_num=True)
    if num_labels == 0:
        return np.zeros(data_sub_shape, dtype=bool)

    regions = regionprops(labeled_mask)
    streak_core_mask = np.zeros(data_sub_shape, dtype=bool)
    num_valid_streaks = 0

    for region in regions:
        if region.area < min_area or region.axis_major_length < 10:
            continue

        elongation = region.axis_major_length / max(region.axis_minor_length, 1e-6)
        if elongation < min_elongation:
            continue

        if existing_mask is not None:
            coords = region.coords
            existing_fraction = np.mean(existing_mask[coords[:, 0], coords[:, 1]])
            if existing_fraction > 0.5:
                continue

        num_valid_streaks += 1
        coords = region.coords
        idx = (coords[:, 0] >= 0) & (coords[:, 0] < img_rows) & (coords[:, 1] >= 0) & (coords[:, 1] < img_cols)
        streak_core_mask[coords[idx, 0], coords[idx, 1]] = True

    print(f"    Validated {num_valid_streaks} regions as streaks based on geometry.")
    return streak_core_mask


def _detect_streaks_frangi_legacy(data_sub, bkg_rms_map, existing_mask, config):
    """Retained legacy Frangi detector for internal benchmarking only."""
    cfg = config.get("frangi_legacy_params", {})
    img_rows, img_cols = data_sub.shape
    streak_mask_final_bool = np.zeros(data_sub.shape, dtype=bool)
    tophat_radius = cfg.get("tophat_radius", 10)
    sigmas = cfg.get("sigmas", [1, 2, 3])
    black_ridges = cfg.get("black_ridges", False)
    min_area = cfg.get("min_area", 50)
    min_elongation = float(cfg.get("min_elongation", 5.0))
    dilation_radius = config.get("dilation_radius", 3)

    print("--> Using legacy Frangi streak detection")
    try:
        selem = disk(tophat_radius)
        tophat_img = white_tophat(data_sub, footprint=selem) if selem.size > 0 else data_sub
        if bkg_rms_map is not None:
            safe_rms = np.where(bkg_rms_map <= 0, 1.0, bkg_rms_map)
            tophat_img = tophat_img / safe_rms

        ridge_map = _apply_frangi_filter(
            tophat_img,
            sigmas,
            black_ridges,
            int(cfg.get("block_size", 1024)),
            int(cfg.get("block_pad", 32)),
            img_rows,
            img_cols,
        )
        low_thresh, high_thresh = _calculate_hysteresis_thresholds(cfg, tophat_img, ridge_map, existing_mask)
        hyst_mask = apply_hysteresis_threshold(ridge_map, low_thresh, high_thresh)
        streak_core_mask = _filter_streak_regions(
            hyst_mask,
            min_area,
            min_elongation,
            existing_mask,
            data_sub.shape,
        )
        selem = disk(dilation_radius)
        streak_mask_final_bool = dilation(streak_core_mask, footprint=selem) if selem.size > 0 else streak_core_mask
    except Exception as e:
        print(f"    Legacy Frangi streak detection failed: {e}")
        return np.zeros(data_sub.shape, dtype=bool)

    return streak_mask_final_bool


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
        median_rms = np.nanmedian(bkg_rms_map)
        if np.isfinite(median_rms) and median_rms > 15.0:
            detect_thresh_sig *= 1.0 + 0.5 * np.log10(median_rms / 15.0)
        thresh = detect_thresh_sig * np.where(
            bkg_rms_map > 0, bkg_rms_map, median_rms if np.isfinite(median_rms) else 1.0
        )
    else:
        thresh = detect_thresh_sig * mad_std(data_sub, ignore_nan=True)

    residual_mask = (data_sub > thresh) & np.isfinite(data_sub)
    if existing_mask is not None:
        residual_mask &= ~existing_mask

    trail_mask = np.zeros(data_sub.shape, dtype=bool)
    for trail_idx in range(max_trails):
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
            )
        except Exception as e:
            print(f"    Sparse RANSAC failed: {e}")
            break

        if inliers is None or np.sum(inliers) < min_inliers:
            break

        inlier_coords = coords[inliers]
        diffs = inlier_coords.max(axis=0) - inlier_coords.min(axis=0)
        sort_dim = int(np.argmax(diffs))
        p0 = inlier_coords[np.argmin(inlier_coords[:, sort_dim])]
        p1 = inlier_coords[np.argmax(inlier_coords[:, sort_dim])]
        length = np.hypot(*(p1 - p0))
        density = np.sum(inliers) / max(length, 1.0)
        if length < min_length or density < min_line_density:
            break

        rr, cc = line(int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1]))
        current = np.zeros_like(trail_mask)
        current[rr, cc] = True
        current = dilation(current, footprint=disk(int(config.get("dilation_radius", 3))))
        trail_mask |= current
        residual_mask &= ~current
        print(
            f"    Sparse RANSAC found trail {trail_idx + 1}: length={length:.1f} px, "
            f"inliers={np.sum(inliers)}, density={density:.3f}"
        )

    return trail_mask


def detect_streaks(data_sub, bkg_rms_map, existing_mask, config):
    """
    Detect linear streaks using the configured primary method plus optional sparse RANSAC.

    Supported methods:
    - 'satdet': satdet-inspired Hough candidate extraction and strip mask refinement.
    - 'frangi_legacy': retained only for internal benchmark comparisons.
    """
    if not config.get("enable", False):
        print("Streak masking disabled in main config.")
        return np.zeros(data_sub.shape, dtype=bool)

    mode = _resolve_streak_mode(config)
    streak_mask_bool = np.zeros(data_sub.shape, dtype=bool)
    debug_info = {"mode": mode, "primary": {}, "retry_unmasked": {}, "mrt": {}, "sparse_ransac": None}

    if mode in {"auto_ground", "satdet_only"}:
        satdet_mask, accepted, primary_debug = _detect_streaks_satdet(data_sub, bkg_rms_map, existing_mask, config)
        streak_mask_bool |= satdet_mask
        debug_info["primary"] = {"accepted": accepted, **primary_debug}
        low_confidence = len(accepted) == 0 or np.sum(satdet_mask) < int(
            config.get("mask_params", {}).get("min_mask_pixels", 64)
        )
        primary_area_fraction = float(np.mean(satdet_mask)) if satdet_mask.size > 0 else 0.0
        suspicious_primary = False
        if accepted:
            support_widths = [float(item.get("support_width", 0.0)) for item in accepted]
            suspicious_primary = primary_area_fraction > float(
                config.get("retry_if_area_fraction_exceeds", 0.03)
            ) or np.median(support_widths) > float(config.get("retry_if_support_width_exceeds", 10.0))
        if (
            mode == "auto_ground"
            and low_confidence
            and existing_mask is not None
            and config.get("retry_without_existing_mask", True)
        ):
            retry_mask, retry_accepted, retry_debug = _detect_streaks_satdet(data_sub, bkg_rms_map, None, config)
            if len(retry_accepted) > 0:
                streak_mask_bool |= retry_mask
            debug_info["retry_unmasked"] = {"accepted": retry_accepted, **retry_debug}
            low_confidence = low_confidence and len(retry_accepted) == 0
        elif (
            mode == "auto_ground"
            and suspicious_primary
            and existing_mask is not None
            and config.get("retry_without_existing_mask", True)
        ):
            retry_mask, retry_accepted, retry_debug = _detect_streaks_satdet(data_sub, bkg_rms_map, None, config)
            retry_pixels = int(np.sum(retry_mask))
            primary_pixels = int(np.sum(satdet_mask))
            if len(retry_accepted) > 0 and retry_pixels > 0 and retry_pixels < primary_pixels:
                streak_mask_bool = retry_mask.copy()
            debug_info["retry_unmasked"] = {"accepted": retry_accepted, **retry_debug}
        if mode == "auto_ground" and low_confidence:
            mrt_mask, mrt_candidates, mrt_debug = _detect_streaks_mrt_like(data_sub, bkg_rms_map, existing_mask, config)
            streak_mask_bool |= mrt_mask
            debug_info["mrt"] = {"accepted": mrt_candidates, **mrt_debug}
    elif mode == "mrt_only":
        mrt_mask, mrt_candidates, mrt_debug = _detect_streaks_mrt_like(data_sub, bkg_rms_map, existing_mask, config)
        streak_mask_bool |= mrt_mask
        debug_info["mrt"] = {"accepted": mrt_candidates, **mrt_debug}
    elif mode == "legacy_compare":
        streak_mask_bool |= _detect_streaks_frangi_legacy(data_sub, bkg_rms_map, existing_mask, config)
    else:
        print(f"Unknown streak detection mode '{mode}'.")
        return np.zeros(data_sub.shape, dtype=bool)

    run_sparse_ransac = bool(config.get("enable_sparse_ransac", True))
    if config.get("sparse_on_primary_weak_only", True):
        primary_accept_count = 0
        primary_debug = debug_info.get("primary", {})
        if isinstance(primary_debug, dict):
            accepted_value = primary_debug.get("accepted", [])
            if isinstance(accepted_value, list):
                primary_accept_count = len(accepted_value)
            elif isinstance(primary_debug.get("accepted"), int):
                primary_accept_count = int(primary_debug["accepted"])
        retry_debug = debug_info.get("retry_unmasked", {})
        if isinstance(retry_debug, dict):
            accepted_value = retry_debug.get("accepted", [])
            if isinstance(accepted_value, list):
                primary_accept_count += len(accepted_value)
            elif isinstance(retry_debug.get("accepted"), int):
                primary_accept_count += int(retry_debug["accepted"])
        run_sparse_ransac = run_sparse_ransac and (
            np.sum(streak_mask_bool) < int(config.get("mask_params", {}).get("min_mask_pixels", 64))
            or (mode in {"auto_ground", "satdet_only"} and primary_accept_count == 0)
        )

    if run_sparse_ransac:
        residual_existing = streak_mask_bool.copy()
        if existing_mask is not None:
            residual_existing |= existing_mask
        sparse_mask = _detect_trails_sparse_ransac(data_sub, bkg_rms_map, residual_existing, config)
        streak_mask_bool |= sparse_mask
        debug_info["sparse_ransac"] = int(np.sum(sparse_mask))
    else:
        debug_info["sparse_ransac"] = 0

    if existing_mask is not None:
        num_new_pixels = int(np.sum(streak_mask_bool & (~existing_mask)))
    else:
        num_new_pixels = int(np.sum(streak_mask_bool))

    if num_new_pixels > 0:
        print(f"  Final streak mask includes {num_new_pixels} new pixels (Mode: {mode}).")
    else:
        print(f"  No new streak pixels added by mode '{mode}'.")

    if config.get("debug", False):
        config["_last_run"] = debug_info

    return streak_mask_bool
