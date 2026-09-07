#!/usr/bin/env python3
"""Race poloka-emulation vs weightmask streaks on identical inputs.

Cases: clean bright trail (crop), full-HDU diagonal (production mask),
real HDU no-injection (FP check), curved arc (poloka should win).
"""

import sys
import time

import numpy as np


def load(pid, hdu, crop=None):
    import fitsio

    from weightmask.background import estimate_background

    with fitsio.FITS(f"/scratch/sfabbro/downloads/{pid}.fits.fz") as f:
        sci = np.ascontiguousarray(f[hdu].read().astype(np.float32))
        hdr = f[hdu].read_header()
    with fitsio.FITS(f"/arc/projects/mlao/cfhtcast/cfhtcast-data/wm/{pid}.mask.fits") as f:
        m = f[hdu].read() if hdu < len(f) else None
    if crop is not None:
        sci = sci[crop].copy()
        if m is not None:
            m = m[crop].copy()
    ex = np.zeros(sci.shape, dtype=bool)
    if m is not None and m.shape == sci.shape:
        ex = (m & 55) > 0
    import yaml

    cfg = yaml.safe_load(open("weightmask.yml"))
    sky, rms = estimate_background(sci, ex, cfg["sep_background"])
    sub = sci - sky
    from astropy.stats import mad_std

    noise = float(mad_std(sub[::10, ::10]))
    return sub, sky, rms, ex, float(hdr.get("SATURATE", 60000.0)), noise, cfg


def race(name, data, sky, rms, ex, sat, truth, scfg):
    import sys

    sys.path.insert(0, "benchmarks")
    from poloka_tracks import poloka_satellite_mask
    from weightmask.streaks import detect_streaks

    # NOTE poloka works on raw (sky-level) images: re-add the background.
    raw = data + sky
    t0 = time.time()
    pm, pn, det = poloka_satellite_mask(
        raw, sky, rms, existing_mask=ex, sat_mask=raw >= sat)
    dt0 = time.time() - t0
    t0 = time.time()
    sm = detect_streaks(data, rms, ex, scfg)
    dt1 = time.time() - t0
    denom = int(truth.sum())
    print(f"{name}: POLOKA rec={float((pm & truth).sum() / denom) if denom else float('nan'):.2f} "
          f"fp={int((pm & ~truth).sum())} ntrk={pn} {dt0:.1f}s | "
          f"OURS rec={float((sm & truth).sum() / denom) if denom else float('nan'):.2f} "
          f"fp={int((sm & ~truth).sum())} {dt1:.1f}s", flush=True)


def main():
    import yaml

    cfg = yaml.safe_load(open("weightmask.yml"))
    scfg = dict(cfg["streak_masking"])
    scfg["enable"] = True
    scfg["mode"] = "auto_ground"

    # (a) clean 10-sigma horizontal trail in quiet crop
    sub, sky, rms, ex, sat, noise, _ = load("719016p", 1, np.s_[0:1024, 1024:2048])
    h, w = sub.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    across = np.abs(yy - 0.5 * h)
    data = sub + np.where(across <= 3, 10.0 * noise * np.exp(-0.5 * (across / 1.2) ** 2), 0.0)
    race("clean10sig", data, sky, rms, ex, sat, across <= 2, scfg)

    # (b) full-HDU 8-sigma diagonal, production mask
    sub, sky, rms, ex, sat, noise, _ = load("719016p", 1)
    h, w = sub.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    across = np.abs((xx - 0.5 * w) * 0.3 + (yy - 0.5 * h))
    data = sub + np.where(across <= 3, 8.0 * noise * np.exp(-0.5 * (across / 1.2) ** 2), 0.0)
    race("full8sig", data, sky, rms, ex, sat, across <= 2, scfg)

    # (c) real HDU, no injection (FP check)
    race("real-noinj", sub, sky, rms, ex, sat, np.zeros(sub.shape, dtype=bool), scfg)

    # (d) curved arc (constant curvature radius 3000px, 8 sigma)
    rr = np.sqrt((xx - 0.5 * w) ** 2 + (yy - 3 * h) ** 2)
    arc = np.abs(rr - 2.5 * h)
    data = sub + np.where(arc <= 3, 8.0 * noise * np.exp(-0.5 * (arc / 1.2) ** 2), 0.0)
    race("curved8sig", data, sky, rms, ex, sat, arc <= 2, scfg)


if __name__ == "__main__":
    main()
