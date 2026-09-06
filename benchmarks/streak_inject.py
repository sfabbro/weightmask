#!/usr/bin/env python3
"""Streak injection-recovery harness on a real MegaPrime HDU.

Injects synthetic trails (continuous + dashed, several lengths/brightnesses)
into a background-subtracted real HDU, runs detect_streaks, and reports
per-trail recall, false-positive pixels, and runtime.

Usage:
    pixi run python benchmarks/streak_inject.py <exposure.fits.fz> <hdu> [seed]
"""

import sys
import time

import numpy as np

from weightmask.background import estimate_background
from weightmask.streaks import detect_streaks


def inject_trails(shape, sky, rng):
    truth = np.zeros(shape, dtype=bool)
    img = np.zeros(shape, dtype=np.float32)
    h, w = shape
    specs = [
        (1500, 8.0, False), (1500, 5.0, False),
        (800, 8.0, False), (800, 5.0, False),
        (400, 6.0, False), (400, 4.0, True),
    ]
    trails = []
    placed = []
    for length, peak_sig, dashed in specs:
        # Non-overlapping placements: crossings merge into tangles that no
        # line-finder should fully recover, confounding recall.
        for _ in range(50):
            angle = rng.uniform(0, np.pi)
            dx, dy = np.cos(angle), np.sin(angle)
            cx, cy = rng.uniform(w * 0.2, w * 0.8), rng.uniform(h * 0.2, h * 0.8)
            if all(max(abs(cx - px), abs(cy - py)) > 500 for px, py, _ in placed):
                break
        placed.append((cx, cy, angle))
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        along = (xx - cx) * dx + (yy - cy) * dy
        across = np.abs((xx - cx) * (-dy) + (yy - cy) * dx)
        on_trail = (np.abs(along) <= length / 2) & (across <= 2.0)
        if dashed:
            on_trail &= ((along % 50) < 30)
        prof = np.exp(-0.5 * (across / 1.2) ** 2)
        flux = np.where(on_trail, peak_sig * prof, 0.0)
        # local rms ~ use global median sky noise proxy passed via sky sig
        img += flux
        truth |= on_trail & (flux > 0.3 * peak_sig)
        trails.append((length, peak_sig, dashed))
    return img, truth, trails


def main():
    import fitsio
    import yaml

    from astropy.stats import mad_std

    path, hdu_idx = sys.argv[1], int(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    rng = np.random.default_rng(seed)
    base_cfg = yaml.safe_load(open("weightmask.yml"))
    streak_cfg = dict(base_cfg["streak_masking"])
    streak_cfg["enable"] = True
    if len(sys.argv) > 4:
        streak_cfg["mode"] = sys.argv[4]
    if len(sys.argv) > 5:
        import json as _json

        for _k, _v in _json.loads(sys.argv[5]).items():
            if "." in _k:
                _top, _sub = _k.split(".", 1)
                streak_cfg[_top] = {**streak_cfg.get(_top, {}), _sub: _v}
            else:
                streak_cfg[_k] = _v

    with fitsio.FITS(path, "r") as f:
        sci = np.ascontiguousarray(f[hdu_idx].read().astype(np.float32))
    import glob as _glob

    _wm = _glob.glob("/arc/projects/mlao/cfhtcast/cfhtcast-data/wm/" + path.split("/")[-1].split(".")[0] + ".mask.fits")
    if _wm:
        with fitsio.FITS(_wm[0], "r") as _f:
            _m = _f[hdu_idx].read() if hdu_idx < len(_f) else None
        existing = ((_m & 55) > 0) if _m is not None and _m.shape == sci.shape else np.zeros(sci.shape, dtype=bool)
        print(f"production existing_mask: {int(existing.sum())} px")
    else:
        existing = np.zeros(sci.shape, dtype=bool)
    sky, rms = estimate_background(sci, existing, base_cfg["sep_background"])
    data_sub = sci - sky
    noise = float(mad_std(data_sub[~np.isnan(data_sub)][::100]))
    trail_flux, truth, trails = inject_trails(sci.shape, sky, rng)
    data_test = data_sub + trail_flux * noise
    print(f"trails={len(trails)} truth_px={int(truth.sum())} noise={noise:.1f}")

    import hashlib
    import os

    key = hashlib.md5(repr(sorted((k, repr(v)) for k, v in streak_cfg.items() if k != "debug")).encode()).hexdigest()[:8]
    cache = f"/tmp/streak_base_{path.split('/')[-1]}_{hdu_idx}_{key}.npy"
    if os.path.exists(cache):
        baseline = np.load(cache).astype(bool)
        print(f"baseline cached: {int(baseline.sum())} px")
        dt_base = 0.0
    else:
        t0 = time.time()
        baseline = detect_streaks(data_sub, rms, existing, dict(streak_cfg))
        dt_base = time.time() - t0
        np.save(cache, baseline.astype(np.uint8))
        print(f"baseline fresh: {int(baseline.sum())} px {dt_base:.1f}s")

    t0 = time.time()
    mask = detect_streaks(data_test, rms, existing, streak_cfg)
    dt = time.time() - t0
    # per-trail recall via connected components of truth
    from scipy.ndimage import binary_dilation, label

    lab, n = label(truth)
    recalls = []
    recalls5 = []
    for i in range(1, n + 1):
        t = lab == i
        recalls.append(np.count_nonzero(mask & t) / np.count_nonzero(t))
        recalls5.append(min(1.0, np.count_nonzero(mask & binary_dilation(t, iterations=5)) / np.count_nonzero(t)))
    novel = mask & ~baseline
    from scipy.ndimage import binary_dilation as _bd
    fp = int(np.count_nonzero(novel & ~truth))
    fp5 = int(np.count_nonzero(novel & ~_bd(truth, iterations=5)))
    print(f"mode={streak_cfg.get('mode')} recall={[round(r, 2) for r in recalls]} recall5={[round(r, 2) for r in recalls5]} fp_px={fp} fp5_px={fp5} {dt:.1f}s")


if __name__ == "__main__":
    main()
