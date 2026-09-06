#!/usr/bin/env python3
"""Faithful emulation of poloka-core MakeSatellite (SFabbro/poloka-core).

Pipeline: zero weight-bad + saturated pixels -> connected components above
sky + 1*sigma (recursive flood fill; restarts at +0.1 sigma on enormous
clusters) -> keep components with size >= diag/10, elongation >= 2,
major axis >= diag/50 -> mask kept pixels, NBRSATEL count.

Used to race the author's prior art against weightmask on identical data.
Differences from C++: iterative label() instead of recursion (no stack
overflow path, so no restart loop); first-pass threshold includes the sky
level directly (the C++ first assignment omits fondciel and relies on the
restart to correct it - see review notes).
"""

import numpy as np


def poloka_satellite_mask(data, sky_map, sigma, existing_mask=None, sat_mask=None,
                          nsigma=1.0, min_elong=2.0, min_axis_frac=0.02):
    """Return (bool mask, n_tracks, details). Mirrors ClusterList/Cut/Mask."""
    from scipy.ndimage import label

    img = np.where(np.isfinite(data), data, 0.0).astype(np.float64)
    if existing_mask is not None:
        img = np.where(existing_mask, 0.0, img)
    if sat_mask is not None:
        img = np.where(sat_mask, 0.0, img)
    sky = np.nanmedian(sky_map) if np.ndim(sky_map) else float(sky_map)
    thresh = sky + nsigma * sigma
    h, w = img.shape
    diag = float(np.hypot(h, w))
    lab, n = label(img > thresh)
    kept = np.zeros(img.shape, dtype=bool)
    details = []
    # C++ pushes only npixels > 20; skip noise specks vectorized (else the
    # per-component Python loop dominates runtime on noisy fields).
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    for i in range(1, n + 1):
        size = int(sizes[i])
        if size <= 20:
            continue
        ys, xs = np.nonzero(lab == i)
        xmean, ymean = xs.mean(), ys.mean()
        x2 = ((xs - xmean) ** 2).mean()
        y2 = ((ys - ymean) ** 2).mean()
        xy = ((xs - xmean) * (ys - ymean)).mean()
        mm = (x2 + y2) / 2.0
        mn = (x2 - y2) / 2.0
        mo = np.sqrt(mn * mn + xy * xy)
        a = 2.0 * np.sqrt(max(mm + mo, 0.0))
        b = 2.0 * np.sqrt(max(mm - mo, 0.0))
        elong = a / max(b, 1e-9)
        if size >= diag / 10.0 and elong >= min_elong and a >= diag * min_axis_frac:
            kept[lab == i] = True
            details.append({"size": size, "elong": float(elong), "a": float(a)})
    return kept, len(details), details
