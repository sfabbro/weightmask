#!/usr/bin/env python3
"""CR completeness / false-positive curves by injection into a real MegaPrime HDU.

Injects single-pixel CRs and multi-pixel worms at known positions into one
real exposure HDU, runs detect_cosmic_rays under several configs, and reports
completeness per morphology plus false-positive pixels and runtime.

Usage:
    pixi run python benchmarks/cr_faint_curves.py <exposure.fits.fz> <hdu> [seed]
"""

import sys
import time

import numpy as np

from weightmask.background import estimate_background
from weightmask.cosmics import detect_cosmic_rays


def inject(sci, sky, rms, rng, n_single=150, n_worm=60):
    truth_single = np.zeros(sci.shape, dtype=bool)
    truth_worm = np.zeros(sci.shape, dtype=bool)
    h, w = sci.shape
    out = sci.copy()
    for _ in range(n_single):
        y, x = rng.integers(10, h - 10), rng.integers(10, w - 10)
        out[y, x] = sky[y, x] + rng.uniform(4, 8) * rms[y, x]
        truth_single[y, x] = True
    for _ in range(n_worm):
        y, x = rng.integers(10, h - 10), rng.integers(10, w - 10)
        angle = rng.uniform(0, np.pi)
        length = int(rng.integers(3, 9))
        amp = rng.uniform(4, 8)
        for t in range(length):
            yy, xx = int(y + t * np.sin(angle)), int(x + t * np.cos(angle))
            if 0 <= yy < h and 0 <= xx < w:
                out[yy, xx] = sky[yy, xx] + amp * rms[yy, xx]
                truth_worm[yy, xx] = True
    return out, truth_single, truth_worm


def score(flag, truth_single, truth_worm):
    from scipy.ndimage import binary_dilation

    near_any = binary_dilation(truth_single | truth_worm, iterations=1)
    tp_single = np.count_nonzero(flag & truth_single)
    tp_worm = np.count_nonzero(flag & truth_worm)
    fp = np.count_nonzero(flag & ~near_any)
    return (
        tp_single / max(1, np.count_nonzero(truth_single)),
        tp_worm / max(1, np.count_nonzero(truth_worm)),
        fp,
        int(np.count_nonzero(flag)),
    )


def main():
    import fitsio
    import yaml

    path, hdu_idx = sys.argv[1], int(sys.argv[2])
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    rng = np.random.default_rng(seed)
    base_cfg = yaml.safe_load(open("weightmask.yml"))
    cosmic_base = base_cfg["cosmic_ray"]

    with fitsio.FITS(path, "r") as f:
        sci = np.ascontiguousarray(f[hdu_idx].read().astype(np.float32))
        hdr = f[hdu_idx].read_header()
    sat = float(hdr.get("SATURATE", 60000.0))
    gain = float(hdr.get("GAIN", 1.5))
    rn = float(hdr.get("RDNOISE", 5.0))
    empty = np.zeros(sci.shape, dtype=bool)
    sky, rms = estimate_background(sci, empty, base_cfg["sep_background"])
    assert sky is not None and rms is not None

    injected, t_single, t_worm = inject(sci, sky, rms, rng)
    print(f"HDU {hdu_idx}: shape={sci.shape} singles={t_single.sum()} wormpx={t_worm.sum()}")

    variants = {
        "main8.5": {},
        "main5.0-nogate": {"sigclip": 5.0},
        "main+faint5.0-e2": {"faint_cr": {"enable": True}},
        "main+faint5.0-e1": {"faint_cr": {"enable": True, "min_elongation": 1.0}},
    }
    for name, override in variants.items():
        cfg = {**cosmic_base, **override}
        if isinstance(override.get("faint_cr"), dict):
            cfg["faint_cr"] = {**cosmic_base["faint_cr"], **override["faint_cr"]}
        t0 = time.time()
        flag = detect_cosmic_rays(injected, empty, sat, gain, rn, cfg, bkg_rms_map=rms)
        dt = time.time() - t0
        cs, cw, fp, n = score(flag, t_single, t_worm)
        print(f"{name:18s} single={cs:.3f} worm={cw:.3f} fp_px={fp:6d} total={n:6d} {dt:5.1f}s")


if __name__ == "__main__":
    main()
