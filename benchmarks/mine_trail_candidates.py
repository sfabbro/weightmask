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


def _hdu_identity(header):
    for key in ("CCDNAME", "CCDNAM", "CCDID", "CHIPID"):
        value = header.get(key)
        if value not in (None, ""):
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            return ("CCD", str(value).strip().upper())
    extname = header.get("EXTNAME")
    if extname not in (None, ""):
        if isinstance(extname, bytes):
            extname = extname.decode("utf-8", errors="replace")
        return ("EXTNAME", str(extname).strip().upper())
    return None


def _read_header(hdu):
    return hdu.read_header() if hasattr(hdu, "read_header") else hdu.header


def _matching_hdu(handle, target_header):
    identity = _hdu_identity(target_header)
    if identity is None:
        raise ValueError("science HDU has no EXTNAME or CCD identity")
    matches = [index for index, hdu in enumerate(handle) if _hdu_identity(_read_header(hdu)) == identity]
    if len(matches) != 1:
        raise ValueError(f"expected one mask HDU for identity {identity}, found {len(matches)}")
    return handle[matches[0]]


def main():
    import glob

    import fitsio
    import yaml

    from weightmask.background import estimate_background
    from weightmask.streaks import _detect_streaks_houghpeaks

    base = yaml.safe_load(open("weightmask.yml"))
    scfg = dict(base["streak_masking"])
    scfg["enable"] = True
    print("pid\thdu\ttheta\trho\tconf\tmaskpx\ttime_s", flush=True)
    failures = False
    for pid in sys.argv[1:]:
        src = None
        for cand in (f"/scratch/sfabbro/downloads/{pid}.fits.fz", f"/scratch/sfabbro/downloads/{pid}.fits"):
            import os

            if os.path.exists(cand):
                src = cand
                break
        if src is None:
            print(f"{pid}\tNONE\t-\t-\t-\t-\t-", flush=True)
            failures = True
            continue
        wm = glob.glob(f"/arc/projects/mlao/cfhtcast/cfhtcast-data/wm/{pid}.mask.fits")
        if not wm:
            print(f"{pid}\tERROR\tmissing mask product\t-\t-\t-\t-", flush=True)
            failures = True
            continue
        with fitsio.FITS(src, "r") as f:
            mask_handle = fitsio.FITS(wm[0], "r")
            try:
                idxs = [i for i in range(len(f)) if f[i].get_info().get("ndims") == 2]
                for i in idxs:
                    t0 = time.time()
                    try:
                        sci = np.ascontiguousarray(f[i].read().astype(np.float32))
                        science_header = f[i].read_header()
                        existing = np.zeros(sci.shape, dtype=bool)
                        if mask_handle is not None:
                            mask_hdu = _matching_hdu(mask_handle, science_header)
                            m = mask_hdu.read()
                            if m.shape != sci.shape:
                                raise ValueError(f"mask shape {m.shape} does not match science shape {sci.shape}")
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
                        failures = True
            finally:
                mask_handle.close()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
