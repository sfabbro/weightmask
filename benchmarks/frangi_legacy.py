"""Legacy Frangi streak detector — benchmark comparison only.

Moved out of weightmask.streaks so production carries one detector path.
"""

import concurrent.futures
import warnings

import numpy as np
from skimage.filters import apply_hysteresis_threshold, frangi
from skimage.measure import label, regionprops
from skimage.morphology import dilation, disk, white_tophat


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
        if np.count_nonzero(bkg_mask) < 1000:
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


