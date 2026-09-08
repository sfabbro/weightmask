import time
import warnings

import numpy as np
import scipy.ndimage as ndi
from astropy.stats import mad_std
from skimage.draw import line
from skimage.feature import canny
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
    if np.count_nonzero(valid) < 100:
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
    """Production accepts only ``auto_ground`` (``method`` is a legacy alias for ``mode``)."""
    requested = (config.get("mode") or config.get("method") or "auto_ground").lower()
    if requested != "auto_ground":
        raise ValueError(
            f"Unknown streak detection mode {requested!r}. "
            "Valid mode: ('auto_ground',). Benchmark Frangi lives in benchmarks.frangi_legacy."
        )
    return "auto_ground"


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

    # Masked stars leave hard holes whose tophat/Canny halos spawn tens of
    # thousands of phantom Hough segments. Dilate the exclusion once here so
    # every downstream use (prepare, raw, rescale, edges) sees quiet margins.
    hole_margin = int(cfg.get("mask_hole_dilation", 5))
    work_mask = existing_mask
    if existing_mask is not None and hole_margin > 0:
        work_mask = dilation(existing_mask, footprint=disk(hole_margin))
    prepared = _prepare_streak_image(data_sub, work_mask)
    positive_raw = np.clip(data_sub, 0.0, None).astype(np.float32)
    if work_mask is not None:
        positive_raw = np.where(work_mask, 0.0, positive_raw)
    source_images = [("prepared", prepared), ("raw", positive_raw)]
    all_segments = []
    debug_scales = []
    for source_idx, (source_name, source_image) in enumerate(source_images):
        for idx, sigma in enumerate(gaussian_sigmas):
            scaled = _robust_scale_image(source_image, work_mask, percentiles)
            if sigma > 0:
                scaled = ndi.gaussian_filter(scaled, sigma)
            edges = canny(scaled, sigma=0.0, low_threshold=canny_low, high_threshold=canny_high)
            if work_mask is not None:
                edges &= ~work_mask
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
                    "edge_pixels": int(np.count_nonzero(edges)),
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
    """Greedily cluster Hough segments by angle and offset.

    The original implementation rescanned every existing cluster per segment and
    re-averaged each cluster's angle/rho over all its members on every
    assignment -- O(n^2) and pathological for the ~10^4-10^5 Hough segments a
    crowded real CCD produces (the probabilistic Hough step alone can return
    tens of thousands of segments).

    This version buckets segments by quantized (angle, rho) and compares each
    segment only against clusters in the neighbouring buckets, maintaining each
    cluster's running mean incrementally. The result is O(n) amortized with the
    same greedy first-match-wins semantics.
    """
    if not segments:
        return []

    angle_bin_w = max(float(angle_tol_deg), 1e-6)
    rho_bin_w = max(float(rho_tol_px), 1e-6)
    n_angle_bins = max(1, int(round(180.0 / angle_bin_w)))

    # Precompute line parameters once (the original recomputed them repeatedly).
    params = [_segment_angle_and_rho(s) for s in segments]

    clusters: list[dict] = []
    # Quantized (angle_bin, rho_bin) -> list of cluster indices registered there.
    bucket_map: dict[tuple[int, int], list[int]] = {}

    def bucket_key(angle_deg: float, rho: float) -> tuple[int, int]:
        a = int(round(angle_deg / angle_bin_w)) % n_angle_bins
        r = int(round(rho / rho_bin_w))
        return (a, r)

    def register(ci: int, key: tuple[int, int]) -> None:
        bucket_map.setdefault(key, []).append(ci)

    def rekey(ci: int, old_key: tuple[int, int], new_key: tuple[int, int]) -> None:
        if old_key == new_key:
            return
        lst = bucket_map.get(old_key)
        if lst is not None:
            try:
                lst.remove(ci)
            except ValueError:
                pass
        register(ci, new_key)

    for segment, (angle_deg, rho) in zip(segments, params):
        a, r = bucket_key(angle_deg, rho)
        assigned = False
        for da in (-1, 0, 1):
            aa = (a + da) % n_angle_bins
            for dr in (-1, 0, 1):
                for ci in bucket_map.get((aa, r + dr), ()):
                    cluster = clusters[ci]
                    # Join iff compatible with the SEED segment. Comparing
                    # against the running mean lets the center random-walk
                    # across the corridor and absorb unrelated segments into
                    # full-span phantoms; the mean is still maintained below
                    # for the representative angle.
                    seed_ang = cluster["seed_angle"]
                    ang_diff_seed = abs(angle_deg - seed_ang)
                    ang_diff_seed = min(ang_diff_seed, 180.0 - ang_diff_seed)
                    if ang_diff_seed <= angle_tol_deg and abs(rho - cluster["seed_rho"]) <= rho_tol_px:
                        cluster["segments"].append(segment)
                        cluster["angle_sum"] += angle_deg
                        cluster["rho_sum"] += rho
                        n = len(cluster["segments"])
                        cluster["angle_deg"] = cluster["angle_sum"] / n
                        cluster["rho"] = cluster["rho_sum"] / n
                        rekey(ci, cluster["bin"], bucket_key(cluster["angle_deg"], cluster["rho"]))
                        cluster["bin"] = bucket_key(cluster["angle_deg"], cluster["rho"])
                        assigned = True
                        break
                if assigned:
                    break
            if assigned:
                break
        if not assigned:
            key = (a, r)
            ci = len(clusters)
            clusters.append(
                {
                    "segments": [segment],
                    "angle_sum": angle_deg,
                    "rho_sum": rho,
                    "angle_deg": angle_deg,
                    "rho": rho,
                    "seed_angle": angle_deg,
                    "seed_rho": rho,
                    "bin": key,
                }
            )
            register(ci, key)

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
    """Return the (rr, cc) pixel coordinates of a corridor around a line.

    The corridor is the set of pixels within ``radius`` of the line segment.
    It is computed on a tight bounding box so the per-candidate cost is
    proportional to the corridor length rather than the full image area (the
    previous full-image mask + dilation was O(width*height) per candidate, which
    dominates when hundreds of Hough clusters reach this stage on a real CCD).
    """
    (x0, y0), (x1, y1) = endpoints
    h, w = shape
    rr, cc = line(int(round(y0)), int(round(x0)), int(round(y1)), int(round(x1)))
    if len(rr) == 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    r = int(radius)
    r0 = max(0, int(rr.min()) - r - 1)
    r1 = min(h, int(rr.max()) + r + 2)
    c0 = max(0, int(cc.min()) - r - 1)
    c1 = min(w, int(cc.max()) + r + 2)

    local = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    local[rr - r0, cc - c0] = True
    if r > 0:
        local = dilation(local, footprint=disk(r))

    grr, gcc = np.nonzero(local)
    return grr + r0, gcc + c0


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

    # Only ``max_candidates`` clusters survive the final sort, but every
    # qualifying cluster triggers an expensive corridor dilation. On a
    # streak-free star field the Hough step yields tens of thousands of
    # spurious segments -> thousands of clusters, making this loop the dominant
    # runtime. Real trails produce many collinear segments, so evaluate only the
    # largest ``corridor_work_cap`` clusters and rank those by corridor signal.
    corridor_work_cap = max(int(max_candidates), 1) * 10
    qualifying = [c for c in clusters if len(c["segments"]) >= min_segments]
    qualifying.sort(key=lambda c: len(c["segments"]), reverse=True)

    for cluster in qualifying[:corridor_work_cap]:
        rep = _representative_line(cluster, shape)
        clipped = rep["clipped_endpoints"]
        if clipped is None:
            continue

        # Transverse scatter of member midpoints around the representative
        # line: a real trail's segments agree to ~±3px, while percolation
        # phantoms spray across the corridor. Reject the spray.
        _pts = np.asarray(cluster["segments"], dtype=np.float32).reshape(-1, 2)
        _d = rep["direction"]
        _rel = _pts - rep["midpoint"]
        _rms = float(np.sqrt(np.mean((_rel[:, 0] * _d[1] - _rel[:, 1] * _d[0]) ** 2)))
        if _rms > float(cfg.get("max_transverse_rms", 8.0)):
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

        corridor_rr, corridor_cc = _line_corridor_mask(clipped, shape, corridor_radius)
        masked_fraction = 0.0
        if existing_mask is not None and corridor_rr.size:
            masked_fraction = float(np.mean(existing_mask[corridor_rr, corridor_cc]))
        if masked_fraction > max_existing_mask_fraction:
            continue

        corridor_signal = prepared[corridor_rr, corridor_cc]
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

    best_label = 1 + np.argmax([np.count_nonzero(labels == idx) for idx in range(1, n + 1)])
    return labels == best_label


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
            if max_support_width < best_w <= max_support_width + fan_slack:
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
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        frbg = np.nanmedian(fsmp[:, sideband], axis=1)
                    frbg = np.where(np.isfinite(frbg), frbg, bg_med)
                    fcen = fsmp - frbg[:, np.newaxis]
                    wfan = _support_count(fcen, fins, bg_std)
                    if _better(wfan, best_w):
                        best_w = wfan
                        strip, sampled, inside, row_bg, centered = fs, fsmp, fins, frbg, fcen

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
    row_hits = _largest_contiguous_run(row_hits)
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


def _refine_hough_peak(H, thetas, rhos, ti, ri):
    """Parabolic sub-pixel refinement of a Hough accumulator peak.

    The 1-degree theta grid alone leaves ~0.3-degree errors, which drift a
    2200px strip ~12px across and bloat support past the width gate. A local
    parabola fit recovers ~0.05-degree precision. Returns (theta_deg, rho).
    """
    orient = H.shape == (len(rhos), len(thetas))
    n_rho, n_th = (len(rhos), len(thetas)) if orient else (H.shape[1], H.shape[0])

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


def _detect_streaks_houghpeaks(data_sub, bkg_rms_map, existing_mask, config):
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
    confidence_threshold = float(config.get("satdet_params", {}).get("confidence_threshold", 0.40))
    min_refined_mask_pixels = int(mask_cfg.get("min_mask_pixels", 64))
    if existing_mask is not None:
        confidence_threshold += min(0.25, 20.0 * float(np.mean(existing_mask)))

    h, w = data_sub.shape
    bh, bw = h // bfac, w // bfac
    if bh < 16 or bw < 16:
        return np.zeros(data_sub.shape, dtype=bool), [], {"peaks": 0}
    _tw, _th_trim = w - bfac * bw, h - bfac * bh
    binned = data_sub[: bh * bfac, : bw * bfac].reshape(bh, bfac, bw, bfac).mean(axis=(1, 3))
    if bkg_rms_map is not None:
        brms = bkg_rms_map[: bh * bfac, : bw * bfac].reshape(bh, bfac, bw, bfac).mean(axis=(1, 3))
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
            refined, refine_info = _refine_trail_mask(
                data_sub, bkg_rms_map, candidate, mask_cfg, existing_mask=existing_mask
            )
            conf = _score_candidate(candidate, refined, refine_info, existing_mask)
            if np.count_nonzero(refined) >= min_refined_mask_pixels and conf >= confidence_threshold:
                streak_mask |= refined
                accepted.append({"theta_deg": theta_deg, "rho": rho, "confidence": conf})
                used.append((rho, theta_deg))
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
    config.get("mask_params", {})
    thresh_sig = float(cfg.get("thresh_sig", 2.5))
    min_span = float(cfg.get("min_span", 150.0))
    shape_cut = float(cfg.get("shape_cut", 0.2))
    area_cut = float(cfg.get("area_cut", 100.0))
    radius_dev_cut = float(cfg.get("radius_dev_cut", 0.5))
    img = np.clip(data_sub, 0.0, None).astype(np.float32)
    if existing_mask is not None:
        img = np.where(existing_mask, 0.0, img)
    if bkg_rms_map is not None:
        rms = np.where((bkg_rms_map > 0) & np.isfinite(bkg_rms_map), bkg_rms_map, np.nan)
        scale = float(np.nanmedian(rms)) if np.any(np.isfinite(rms)) else 1.0
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
                "edge_touches": len(_edges_touched(clipped, data_sub.shape, 16)),
                "corridor_overlap": 0.0,
            }
        )
    return out


def _detect_streaks_contours(data_sub, bkg_rms_map, existing_mask, config):
    """Confirm contour-morphology candidates with the strip profiler."""
    cfg = config.get("contour_params", {})
    mask_cfg = config.get("mask_params", {})
    if not cfg.get("enable", True):
        return np.zeros(data_sub.shape, dtype=bool), [], {"candidates": 0}
    confidence_threshold = float(config.get("satdet_params", {}).get("confidence_threshold", 0.40))
    min_refined_mask_pixels = int(mask_cfg.get("min_mask_pixels", 64))
    if existing_mask is not None:
        confidence_threshold += min(0.25, 20.0 * float(np.mean(existing_mask)))
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
    bin_factor = int(cfg.get("bin_factor", 1))
    if bin_factor > 1:
        print(f"    Binning {bin_factor}x{bin_factor} for Hough prescreen (confirm at full res)...")
        bh, bw = data_sub.shape[0] // bin_factor, data_sub.shape[1] // bin_factor
        binned = (
            data_sub[: bh * bin_factor, : bw * bin_factor].reshape(bh, bin_factor, bw, bin_factor).mean(axis=(1, 3))
        )
        binned_mask = None
        if existing_mask is not None:
            binned_mask = (
                existing_mask[: bh * bin_factor, : bw * bin_factor]
                .reshape(bh, bin_factor, bw, bin_factor)
                .any(axis=(1, 3))
            )
        bin_cfg = dict(cfg)
        for _k in ("hough_min_line_length", "hough_max_line_gap", "small_edge_perimeter"):
            if _k in bin_cfg:
                bin_cfg[_k] = max(2, int(bin_cfg[_k] // bin_factor))
        if "gaussian_sigmas" in bin_cfg and bin_cfg["gaussian_sigmas"] is not None:
            bin_cfg["gaussian_sigmas"] = [float(s) / bin_factor for s in bin_cfg["gaussian_sigmas"]]
        if "gaussian_sigma" in bin_cfg:
            bin_cfg["gaussian_sigma"] = float(bin_cfg["gaussian_sigma"]) / bin_factor
        segments, debug_scales = _extract_multiscale_segments(binned, binned_mask, bin_cfg)
        segments = [
            ((x0 * bin_factor, y0 * bin_factor), (x1 * bin_factor, y1 * bin_factor)) for (x0, y0), (x1, y1) in segments
        ]
    else:
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
            np.count_nonzero(refined) >= min_refined_mask_pixels
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

    print(f"    satdet-style refinement produced {np.count_nonzero(streak_mask)} streak pixels.")
    return (
        streak_mask,
        accepted,
        {"scales": debug_scales, "accepted_count": len(accepted), "candidates": len(candidates)},
    )


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
            break

        rr, cc = line(int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1]))
        current = np.zeros_like(trail_mask)
        current[rr, cc] = True
        dilation_r = int(cfg.get("dilation_radius", config.get("dilation_radius", 1)))
        current = dilation(current, footprint=disk(dilation_r)) if dilation_r > 0 else current
        trail_mask |= current
        residual_mask &= ~current
        print(
            f"    Sparse RANSAC found trail {trail_idx + 1}: length={length:.1f} px, "
            f"inliers={np.count_nonzero(inliers)}, density={density:.3f}"
        )

    return trail_mask


def detect_streaks(data_sub, bkg_rms_map, existing_mask, config):
    """
    Detect linear streaks via the production ``auto_ground`` path plus optional sparse RANSAC.

    Raises ValueError on unknown modes (fail fast on config typos).
    Mode ``auto_ground`` (also via legacy ``method`` alias): binned-Hough peaks + satdet
    primary + unmasked retry + MRT rescue + conditional RANSAC.

    Benchmark-only Frangi lives in ``benchmarks.frangi_legacy``.
    """
    if not config.get("enable", False):
        print("Streak masking disabled in main config.")
        return np.zeros(data_sub.shape, dtype=bool)

    mode = _resolve_streak_mode(config)
    streak_mask_bool = np.zeros(data_sub.shape, dtype=bool)
    debug_info = {"mode": mode, "primary": {}, "retry_unmasked": {}, "mrt": {}, "sparse_ransac": None}

    print(f"  Detecting streaks (mode: {mode})...")
    streak_t0 = time.time()
    hp_mask, hp_accepted, hp_debug = _detect_streaks_houghpeaks(data_sub, bkg_rms_map, existing_mask, config)
    streak_mask_bool |= hp_mask
    debug_info["houghpeaks"] = {"accepted": hp_accepted, **hp_debug}
    ct_mask, ct_accepted, ct_debug = _detect_streaks_contours(data_sub, bkg_rms_map, existing_mask, config)
    streak_mask_bool |= ct_mask
    debug_info["contours"] = {"accepted": ct_accepted, **ct_debug}
    print("    [streak] satdet primary pass...")
    satdet_mask, accepted, primary_debug = _detect_streaks_satdet(data_sub, bkg_rms_map, existing_mask, config)
    streak_mask_bool |= satdet_mask
    debug_info["primary"] = {"accepted": accepted, **primary_debug}
    # Tactic C: "primary accepted" spans every primary stage (houghpeaks +
    # contours + satdet), each of which applies its own accept thresholds.
    # Counting only satdet kept rescue/RANSAC running on HDUs where an
    # earlier stage had already accepted trails.
    n_early_accepted = len(hp_accepted) + len(ct_accepted)
    print(f"    [streak] primary done in {time.time() - streak_t0:.1f}s: {len(accepted)} accepted.")
    low_confidence = len(accepted) == 0 or np.count_nonzero(satdet_mask) < int(
        config.get("mask_params", {}).get("min_mask_pixels", 64)
    )
    primary_area_fraction = float(np.mean(satdet_mask)) if satdet_mask.size > 0 else 0.0
    suspicious_primary = False
    if accepted:
        support_widths = [float(item.get("support_width", 0.0)) for item in accepted]
        suspicious_primary = primary_area_fraction > float(
            config.get("retry_if_area_fraction_exceeds", 0.03)
        ) or np.median(support_widths) > float(config.get("retry_if_support_width_exceeds", 10.0))
    # Same all-primary rule for the unmasked retry: skip it when houghpeaks /
    # contours already accepted trails. The suspicious-phantom recovery below
    # is untouched, as is the retry when no primary stage accepted anything.
    if (
        low_confidence
        and n_early_accepted == 0
        and existing_mask is not None
        and config.get("retry_without_existing_mask", True)
    ):
        print(f"    [streak] retrying satdet without existing mask (t+{time.time() - streak_t0:.1f}s)...")
        retry_mask, retry_accepted, retry_debug = _detect_streaks_satdet(data_sub, bkg_rms_map, None, config)
        if len(retry_accepted) > 0:
            streak_mask_bool |= retry_mask
        debug_info["retry_unmasked"] = {"accepted": retry_accepted, **retry_debug}
        low_confidence = low_confidence and len(retry_accepted) == 0
    elif suspicious_primary and existing_mask is not None and config.get("retry_without_existing_mask", True):
        print(f"    [streak] retrying satdet without existing mask (t+{time.time() - streak_t0:.1f}s)...")
        retry_mask, retry_accepted, retry_debug = _detect_streaks_satdet(data_sub, bkg_rms_map, None, config)
        retry_pixels = int(np.count_nonzero(retry_mask))
        primary_pixels = int(np.count_nonzero(satdet_mask))
        if len(retry_accepted) > 0 and retry_pixels > 0 and retry_pixels < primary_pixels:
            streak_mask_bool = retry_mask.copy()
        debug_info["retry_unmasked"] = {"accepted": retry_accepted, **retry_debug}
    min_streak_px = int(config.get("mask_params", {}).get("min_mask_pixels", 64))
    n_primary_total = (
        n_early_accepted + len(accepted) + len(debug_info.get("retry_unmasked", {}).get("accepted", []) or [])
    )
    if low_confidence and (n_primary_total == 0 or np.count_nonzero(streak_mask_bool) < min_streak_px):
        print(f"    [streak] MRT rescue pass (t+{time.time() - streak_t0:.1f}s)...")
        mrt_mask, mrt_candidates, mrt_debug = _detect_streaks_mrt_like(data_sub, bkg_rms_map, existing_mask, config)
        streak_mask_bool |= mrt_mask
        debug_info["mrt"] = {"accepted": mrt_candidates, **mrt_debug}

    run_sparse_ransac = bool(config.get("enable_sparse_ransac", True))
    if config.get("sparse_on_primary_weak_only", True):
        primary_accept_count = n_early_accepted
        primary_debug = debug_info.get("primary", {})
        if isinstance(primary_debug, dict):
            accepted_value = primary_debug.get("accepted", [])
            if isinstance(accepted_value, list):
                primary_accept_count += len(accepted_value)
            elif isinstance(primary_debug.get("accepted"), int):
                primary_accept_count += int(primary_debug["accepted"])
        retry_debug = debug_info.get("retry_unmasked", {})
        if isinstance(retry_debug, dict):
            accepted_value = retry_debug.get("accepted", [])
            if isinstance(accepted_value, list):
                primary_accept_count += len(accepted_value)
            elif isinstance(retry_debug.get("accepted"), int):
                primary_accept_count += int(retry_debug["accepted"])
        run_sparse_ransac = run_sparse_ransac and (
            np.count_nonzero(streak_mask_bool) < int(config.get("mask_params", {}).get("min_mask_pixels", 64))
            or primary_accept_count == 0
        )

    if run_sparse_ransac:
        residual_existing = streak_mask_bool.copy()
        if existing_mask is not None:
            residual_existing |= existing_mask
        print(f"    [streak] sparse RANSAC pass (t+{time.time() - streak_t0:.1f}s)...")
        sparse_mask = _detect_trails_sparse_ransac(data_sub, bkg_rms_map, residual_existing, config)
        streak_mask_bool |= sparse_mask
        debug_info["sparse_ransac"] = int(np.count_nonzero(sparse_mask))
    else:
        debug_info["sparse_ransac"] = 0

    if existing_mask is not None:
        num_new_pixels = int(np.count_nonzero(streak_mask_bool & (~existing_mask)))
    else:
        num_new_pixels = int(np.count_nonzero(streak_mask_bool))

    streak_elapsed = time.time() - streak_t0
    if num_new_pixels > 0:
        print(f"  Final streak mask includes {num_new_pixels} new pixels (Mode: {mode}, {streak_elapsed:.1f}s).")
    else:
        print(f"  No new streak pixels added by mode '{mode}' ({streak_elapsed:.1f}s).")

    if config.get("debug", False):
        config["_last_run"] = debug_info

    return streak_mask_bool
