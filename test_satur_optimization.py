import numpy as np
import scipy.ndimage
from weightmask.satur import grow_bleed_trails

def create_mock_data():
    h, w = 4000, 4000
    sci_data = np.random.normal(100, 10, (h, w)).astype(np.float32)
    sat_mask = np.zeros((h, w), dtype=bool)
    sky_map = np.full((h, w), 100.0, dtype=np.float32)
    bkg_rms_map = np.full((h, w), 10.0, dtype=np.float32)

    # Add multiple saturated columns
    for col in np.random.choice(w, 2000, replace=False):
        # 10 segments per column
        for i in range(10):
            start = np.random.randint(100, h - 500)
            length = np.random.randint(20, 50)
            sat_mask[start:start+length, col] = True
            sci_data[start-50:start+length+50, col] = np.linspace(150, 100, length+100)

    config = {
        "mask_bleed_trails": True,
        "bleed_thresh_sigma": 5.0,
        "bleed_grow_vertical": 50,
        "bleed_grow_horizontal": 2,
    }
    return sci_data, sat_mask, sky_map, bkg_rms_map, config

sci_data, sat_mask, sky_map, bkg_rms_map, config = create_mock_data()

import copy
import time

sat_mask_orig = copy.deepcopy(sat_mask)

start_orig = time.time()
res_orig = grow_bleed_trails(sci_data, sat_mask_orig, sky_map, bkg_rms_map, config)
end_orig = time.time()

print(f"Orig time: {end_orig - start_orig:.4f}s")

def new_grow_bleed_trails(sci_data, sat_mask, sky_map, bkg_rms_map, config):
    if not config.get("mask_bleed_trails", True):
        return sat_mask

    print("  Growing bleed trails based on local flux levels...")
    h, w = sci_data.shape
    new_mask = sat_mask.copy()

    # Identify columns with saturation
    sat_cols = np.where(np.any(sat_mask, axis=0))[0]

    if len(sat_cols) == 0:
        return new_mask

    # Get a 2D mask of just the saturated columns to label
    sat_mask_cols = sat_mask[:, sat_cols]

    # 2D structure for labeling (connects only vertically)
    # The structure shape must be (3, 3) because sat_mask_cols is 2D
    struct = np.array([[0, 1, 0],
                       [0, 1, 0],
                       [0, 1, 0]])

    labeled_cols, num_segments = scipy.ndimage.label(sat_mask_cols, structure=struct)

    # Pre-calculate background levels if missing to avoid checking per segment
    if sky_map is None:
        sky_map = np.zeros((h, w))
    if bkg_rms_map is None:
        bkg_rms_map = np.full((h, w), 10.0)

    # we can use slices to find coordinates using labeled_cols
    # Find all segments at once
    slices = scipy.ndimage.find_objects(labeled_cols)

    # Pre-calculate thresholds for all saturated columns at once
    # Use broadcasting
    col_bkgs = sky_map[:, sat_cols]
    col_rmss = bkg_rms_map[:, sat_cols]
    stop_threshes = col_bkgs + config.get("bleed_thresh_sigma", 5.0) * col_rmss

    for s_idx, slc in enumerate(slices):
        if slc is None:
            continue

        y_slice, x_slice = slc
        y_min = y_slice.start
        y_max = y_slice.stop - 1

        # In a purely vertical structure, the x_slice should span exactly 1 element
        col_idx = x_slice.start
        x = sat_cols[col_idx]

        stop_thresh = stop_threshes[:, col_idx]

        max_grow = config.get("bleed_grow_vertical", 50)

        # Grow up
        if y_min > 0:
            start_idx = y_min - 1
            end_idx = max(-1, start_idx - max_grow)

            if end_idx == -1:
                sci_slice = sci_data[start_idx::-1, x]
                thresh_slice = stop_thresh[start_idx::-1]
            else:
                sci_slice = sci_data[start_idx:end_idx:-1, x]
                thresh_slice = stop_thresh[start_idx:end_idx:-1]

            cond = sci_slice > thresh_slice
            if not np.all(cond):
                grown = np.argmin(cond)
            else:
                grown = len(cond)

            if grown > 0:
                end_mask = start_idx - grown
                if end_mask == -1:
                    new_mask[start_idx::-1, x] = True
                else:
                    new_mask[start_idx:end_mask:-1, x] = True

        # Grow down
        if y_max < h - 1:
            start_idx = y_max + 1
            end_idx = min(h, start_idx + max_grow)

            sci_slice = sci_data[start_idx:end_idx, x]
            thresh_slice = stop_thresh[start_idx:end_idx]

            cond = sci_slice > thresh_slice
            if not np.all(cond):
                grown = np.argmin(cond)
            else:
                grown = len(cond)

            if grown > 0:
                new_mask[start_idx : start_idx + grown, x] = True

    # Horizontal dilation for safety (optional)
    h_dilation = config.get("bleed_grow_horizontal", 2)
    if h_dilation > 0:
        selem = np.ones((1, 2 * h_dilation + 1), dtype=bool)
        new_mask = scipy.ndimage.binary_dilation(new_mask, structure=selem)

    print(f"    Bleed trail growth added {np.sum(new_mask & ~sat_mask)} pixels.")
    return new_mask

start_new = time.time()
res_new = new_grow_bleed_trails(sci_data, sat_mask, sky_map, bkg_rms_map, config)
end_new = time.time()

print(f"New time: {end_new - start_new:.4f}s")
print("Arrays equal?", np.array_equal(res_orig, res_new))
