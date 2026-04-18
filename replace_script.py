import re

with open("weightmask/streaks.py", "r") as f:
    content = f.read()

helpers_code = """
def _apply_frangi_filter(
    tophat_img, sigmas, black_ridges, block_size, pad, img_rows, img_cols
):
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
                block_ridge = frangi(
                    block, sigmas=sigmas, black_ridges=black_ridges
                )

            valid_r_start = r - r_start_pad
            valid_r_end = valid_r_start + min(block_size, img_rows - r)
            valid_c_start = c - c_start_pad
            valid_c_end = valid_c_start + min(block_size, img_cols - c)

            h = valid_r_end - valid_r_start
            w = valid_c_end - valid_c_start
            interior = block_ridge[
                valid_r_start:valid_r_end, valid_c_start:valid_c_end
            ]

            return r, c, h, w, interior

        futures = []
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for r in range(0, img_rows, block_size):
                for c in range(0, img_cols, block_size):
                    futures.append(executor.submit(process_block, r, c))

            for f in concurrent.futures.as_completed(futures):
                r, c, h, w, interior = f.result()
                ridge_map[r : r + h, c : c + w] = interior
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ridge_map = frangi(
                tophat_img, sigmas=sigmas, black_ridges=black_ridges
            )

    return ridge_map


def _calculate_hysteresis_thresholds(cfg, tophat_img, ridge_map, existing_mask):
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
            p50, p90, p99 = np.percentile(bkg_ridge, [50, 90, 99])
            tail_spread = p99 - p50
            if tail_spread == 0:
                tail_spread = 1e-9

            high_thresh = p50 + high_threshold_sig * tail_spread
            low_thresh = p50 + low_threshold_sig * tail_spread
        else:
            high_thresh = 1e-5
            low_thresh = 5e-6

        high_thresh = max(high_thresh, 1e-6)
        low_thresh = max(low_thresh, 1e-7)

    return low_thresh, high_thresh


def _filter_streak_regions(hyst_mask, min_area, existing_mask, data_sub_shape):
    img_rows, img_cols = data_sub_shape
    labeled_mask, num_labels = label(hyst_mask, connectivity=2, return_num=True)
    if num_labels == 0:
        return np.zeros(data_sub_shape, dtype=bool)

    regions = regionprops(labeled_mask)
    streak_core_mask = np.zeros(data_sub_shape, dtype=bool)

    num_valid_streaks = 0
    for region in regions:
        if region.area < min_area:
            continue

        if existing_mask is not None:
            coords = region.coords
            valid_rows = coords[:, 0]
            valid_cols = coords[:, 1]
            idx = (
                (valid_rows >= 0)
                & (valid_rows < img_rows)
                & (valid_cols >= 0)
                & (valid_cols < img_cols)
            )
            r_mask = existing_mask[valid_rows[idx], valid_cols[idx]]
            if np.sum(r_mask) > 0.5 * region.area:
                continue

        if hasattr(region, "axis_major_length"):
            major_length = region.axis_major_length
        else:
            major_length = getattr(region, "major_axis_length", 0)

        if major_length >= 10:
            num_valid_streaks += 1
            coords = region.coords
            valid_rows = coords[:, 0]
            valid_cols = coords[:, 1]
            idx = (
                (valid_rows >= 0)
                & (valid_rows < img_rows)
                & (valid_cols >= 0)
                & (valid_cols < img_cols)
            )
            streak_core_mask[valid_rows[idx], valid_cols[idx]] = True

    print(f"    Validated {num_valid_streaks} regions as streaks based on length.")
    return streak_core_mask


def _detect_streaks_frangi(data_sub, bkg_rms_map, existing_mask, config):
"""

new_main_func = """def _detect_streaks_frangi(data_sub, bkg_rms_map, existing_mask, config):
    \"\"\"
    Internal function for streak detection using Frangi ridge filter and hysteresis.
    This works in the continuous domain and is much more robust for single noisy exposures.
    \"\"\"
    img_rows, img_cols = data_sub.shape
    streak_mask_final_bool = np.zeros(data_sub.shape, dtype=bool)

    # --- Configuration Parameters ---
    cfg = config.get("frangi_params", {})

    tophat_radius = cfg.get("tophat_radius", 10)
    sigmas = cfg.get("sigmas", [1, 2, 3])
    black_ridges = cfg.get("black_ridges", False)
    min_area = cfg.get("min_area", 50)
    dilation_radius = config.get("dilation_radius", 3)

    print("--> Using Frangi Ridge Method for Streak Detection")
    start_time = time.time()

    try:
        data_clean = data_sub.copy()
        print(f"    Applying White Top-Hat (radius={tophat_radius}) to flatten sky/stars...")
        selem = disk(tophat_radius)
        if selem.size > 0:
            tophat_img = white_tophat(data_clean, footprint=selem)
        else:
            tophat_img = data_clean

        if bkg_rms_map is not None:
            safe_rms = np.where(bkg_rms_map <= 0, 1.0, bkg_rms_map)
            tophat_img = tophat_img / safe_rms

        block_size = cfg.get("block_size", 1024)
        pad = cfg.get("block_pad", 32)

        ridge_map = _apply_frangi_filter(
            tophat_img, sigmas, black_ridges, block_size, pad, img_rows, img_cols
        )

        low_thresh, high_thresh = _calculate_hysteresis_thresholds(
            cfg, tophat_img, ridge_map, existing_mask
        )

        print(f"    Hysteresis Thresholds: Low={low_thresh:.2e}, High={high_thresh:.2e}")

        hyst_mask = apply_hysteresis_threshold(ridge_map, low_thresh, high_thresh)
        print(f"    Hysteresis found {np.sum(hyst_mask)} candidate pixel groups.")

        streak_core_mask = _filter_streak_regions(
            hyst_mask, min_area, existing_mask, data_sub.shape
        )

        print(f"    Dilating {np.sum(streak_core_mask)} streak core pixels by radius {dilation_radius}...")

        from skimage.morphology import dilation
        selem = disk(dilation_radius)
        if selem.size > 0:
            streak_mask_final_bool = dilation(streak_core_mask, footprint=selem)
        else:
            streak_mask_final_bool = streak_core_mask

    except Exception as e:
        import traceback

        print(f"    Frangi streak detection failed: {e}")
        print(traceback.format_exc())
        return np.zeros(data_sub.shape, dtype=bool)

    elapsed = time.time() - start_time
    print(f"    Frangi method finished in {elapsed:.2f} seconds.")
    return streak_mask_final_bool
"""

pattern = (
    r"def _detect_streaks_frangi\(data_sub, bkg_rms_map, existing_mask, config\):.*?return streak_mask_final_bool\n"
)

new_content = re.sub(
    pattern,
    helpers_code.replace("def _detect_streaks_frangi(data_sub, bkg_rms_map, existing_mask, config):\n", "")
    + new_main_func,
    content,
    flags=re.DOTALL,
)

with open("weightmask/streaks.py", "w") as f:
    f.write(new_content)
