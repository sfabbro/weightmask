#!/usr/bin/env python3
"""Extraction-level recall: per-(source,sigma) Hough segments vs injected truth.

Replicates _extract_multiscale_segments' inner loop per (source, sigma) so
each pass is attributed exactly: total segments, segments near the injected
trail, edge pixels, and time.

Usage:
    pixi run python benchmarks/streak_extract_recall.py <exposure.fits.fz> <hdu> [seed]
"""

import sys
import time

import numpy as np


def main():
    import fitsio
    import yaml
    from scipy import ndimage as ndi
    from skimage.feature import canny
    from skimage.transform import probabilistic_hough_line

    from weightmask.background import estimate_background
    from weightmask.streaks import (
        _prepare_streak_image,
        _prune_small_edges,
        _robust_scale_image,
    )

    path, hdu_idx = sys.argv[1], int(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    base_cfg = yaml.safe_load(open("weightmask.yml"))
    satcfg = base_cfg["streak_masking"]["satdet_params"]

    with fitsio.FITS(path, "r") as f:
        sci = np.ascontiguousarray(f[hdu_idx].read().astype(np.float32))
    with fitsio.FITS("/arc/projects/mlao/cfhtcast/cfhtcast-data/wm/" + path.split("/")[-1].split(".")[0] + ".mask.fits") as f:
        existing = (f[hdu_idx].read() & 55) > 0
    sky, rms = estimate_background(sci, existing, base_cfg["sep_background"])
    sub = sci - sky
    from astropy.stats import mad_std

    noise = float(mad_std(sub[::100, ::100]))
    h, w = sub.shape
    angle = 0.35
    dx, dy = np.cos(angle), np.sin(angle)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    across = np.abs((xx - w / 2) * (-dy) + (yy - h / 2) * dx)
    along = (xx - w / 2) * dx + (yy - h / 2) * dy
    trail = np.where((across <= 3) & (np.abs(along) <= 400), 8.0 * noise * np.exp(-0.5 * (across / 1.2) ** 2), 0.0)
    data = sub + trail
    truth = (across <= 5) & (np.abs(along) <= 400)

    percentiles = tuple(satcfg.get("rescale_percentiles", [4.5, 93.0]))
    prepared = _prepare_streak_image(data, existing)
    positive_raw = np.clip(data, 0.0, None).astype(np.float32)
    positive_raw = np.where(existing, 0.0, positive_raw)
    sources = {"prepared": prepared, "raw": positive_raw}
    for src_name, src_img in sources.items():
        for sigma in satcfg.get("gaussian_sigmas", [1.5, 2.0, 3.0]):
            t0 = time.time()
            scaled = _robust_scale_image(src_img, existing, percentiles)
            if sigma > 0:
                scaled = ndi.gaussian_filter(scaled, sigma)
            edges = canny(scaled, sigma=0.0, low_threshold=float(satcfg.get("canny_low_threshold", 0.1)),
                          high_threshold=float(satcfg.get("canny_high_threshold", 0.35)))
            edges &= ~existing
            edges = _prune_small_edges(edges, float(satcfg.get("small_edge_perimeter", 60.0)))
            segs = probabilistic_hough_line(
                edges,
                threshold=int(satcfg.get("hough_threshold", 10)),
                line_length=int(satcfg.get("hough_min_line_length", 120)),
                line_gap=int(satcfg.get("hough_max_line_gap", 30)),
                rng=int(satcfg.get("hough_rng_seed", 0)),
            )
            near = 0
            for (x0, y0), (x1, y1) in segs:
                ix, iy = int(round((x0 + x1) / 2)), int(round((y0 + y1) / 2))
                if 0 <= iy < h and 0 <= ix < w and truth[max(0, iy - 5) : iy + 6, max(0, ix - 5) : ix + 6].any():
                    near += 1
            print(f"{src_name} sig={sigma}: segs={len(segs)} near-truth={near} edgepx={int(edges.sum())} {time.time()-t0:.1f}s",
                  flush=True)


if __name__ == "__main__":
    main()
