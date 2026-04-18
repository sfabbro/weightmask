import re

with open('weightmask/satur.py', 'r') as f:
    content = f.read()

old_imports = """import scipy.ndimage"""
new_imports = """import scipy.ndimage\nfrom scipy.ndimage import label, find_objects"""
content = content.replace(old_imports, new_imports)

old_loop = """
    # Identify columns with saturation
    sat_cols = np.where(np.any(sat_mask, axis=0))[0]

    for x in sat_cols:
        col_sat = sat_mask[:, x]
        # Find contiguous segments of saturated pixels in this column
        labeled_segments, num_segments = scipy.ndimage.label(col_sat)

        for s in range(1, num_segments + 1):
            segment_indices = np.where(labeled_segments == s)[0]
            y_min, y_max = segment_indices.min(), segment_indices.max()

            # Get background levels for this column
            col_bkg = sky_map[:, x] if sky_map is not None else np.zeros(h)
            col_rms = bkg_rms_map[:, x] if bkg_rms_map is not None else np.full(h, 10.0)

            # Use a conservative threshold (e.g. 5 sigma) to prevent over-growing into noise
            stop_thresh = col_bkg + config.get("bleed_thresh_sigma", 5.0) * col_rms

            max_grow = config.get("bleed_grow_vertical", 50)

            _grow_bleed_up(sci_data, stop_thresh, x, y_min, max_grow, new_mask)
            _grow_bleed_down(sci_data, stop_thresh, h, x, y_max, max_grow, new_mask)
"""
new_loop = """
    # Identify columns with saturation
    y_any, x_any = np.where(sat_mask)
    if len(y_any) > 0:
        min_y, max_y = np.min(y_any), np.max(y_any)
        min_x, max_x = np.min(x_any), np.max(x_any)

        # A vertical 3x3 struct ensures we only connect pixels in the same column
        struct = np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        sub_mask = sat_mask[min_y:max_y+1, min_x:max_x+1]

        labeled, num_features = label(sub_mask, structure=struct)
        slices = find_objects(labeled)

        max_grow = config.get("bleed_grow_vertical", 50)
        bleed_thresh_sigma = config.get("bleed_thresh_sigma", 5.0)

        for sl in slices:
            if sl is not None:
                y_sl, x_sl = sl
                # sl contains slices for the bounding box in the sub_mask
                x = min_x + x_sl.start
                y_min = min_y + y_sl.start
                y_max = min_y + y_sl.stop - 1

                # Get background levels for this column
                col_bkg = sky_map[:, x] if sky_map is not None else np.zeros(h)
                col_rms = bkg_rms_map[:, x] if bkg_rms_map is not None else np.full(h, 10.0)

                # Use a conservative threshold (e.g. 5 sigma) to prevent over-growing into noise
                stop_thresh = col_bkg + bleed_thresh_sigma * col_rms

                _grow_bleed_up(sci_data, stop_thresh, x, y_min, max_grow, new_mask)
                _grow_bleed_down(sci_data, stop_thresh, h, x, y_max, max_grow, new_mask)
"""
content = content.replace(old_loop, new_loop)

with open('weightmask/satur.py', 'w') as f:
    f.write(content)
