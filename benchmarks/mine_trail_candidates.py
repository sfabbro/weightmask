#!/usr/bin/env python3
"""Mine trail candidates from real exposures for ground-truth curation.

Runs the cheap houghpeaks stage over every HDU of the given exposures (with
production masks as existing_mask where available, else zeros) and writes one
TSV line per accepted trail: pid, hdu, theta, rho, confidence, mask pixels.
A human reviews thumbnails of candidates -> curated ground truth set.

Usage:
    pixi run python benchmarks/mine_trail_candidates.py <pid> [pid ...]
"""

import sys
import time

import numpy as np


def main():
    import fitsio
    import glob
    import yaml

    from weightmask.background import estimate_background
    from weightmask.streaks import _detect_streaks_houghpeaks

    base = yaml.safe_load(open("weightmask.yml"))
    scfg = dict(base["streak_masking"])
    scfg["enable"] = True
    print("pid\thdu\ttheta\trho\tconf\tmaskpx\ttime_s", flush=True)
    for pid in sys.argv[1:]:
        src = None
        for cand in (f"/scratch/sfabbro/downloads/{pid}.fits.fz", f"/scratch/sfabbro/downloads/{pid}.fits"):
            import os

            if os.path.exists(cand):
                src = cand
                break
        if src is None:
            print(f"{pid}\tNONE\t-\t-\t-\t-\t-", flush=True)
            continue
        wm = glob.glob(f"/arc/projects/mlao/cfhtcast/cfhtcast-data/wm/{pid}.mask.fits")
        with fitsio.FITS(src, "r") as f:
            idxs = [i for i in range(len(f)) if f[i].get_info().get("ndims") == 2]
            for i in idxs:
                t0 = time.time()
                try:
                    sci = np.ascontiguousarray(f[i].read().astype(np.float32))
                    existing = np.zeros(sci.shape, dtype=bool)
                    if wm:
                        with fitsio.FITS(wm[0], "r") as mf:
                            if i < len(mf):
                                m = mf[i].read()
                                if m.shape == sci.shape:
                                    existing = (m & 55) > 0
                    sky, rms = estimate_background(sci, existing, base["sep_background"])
                    mask, acc, _ = _detect_streaks_houghpeaks(sci - sky, rms, existing, scfg)
                    dt = time.time() - t0
                    if acc:
                        for a in acc:
                            print(f"{pid}\t{i}\t{a['theta_deg']:.2f}\t{a['rho']:.0f}\t"
                                  f"{a['confidence']:.2f}\t{int((mask > 0).sum())}\t{dt:.0f}", flush=True)
                    else:
                        print(f"{pid}\t{i}\t-\t-\t-\t0\t{dt:.0f}", flush=True)
                except Exception as e:
                    print(f"{pid}\t{i}\tERROR\t{e}\t-\t-\t-", flush=True)


if __name__ == "__main__":
    main()
