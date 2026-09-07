#!/usr/bin/env python3
"""Background-pull comparison: old batch weight maps vs current code.

For each (pid, hdu): loads science + OLD mask/weight/sky from wm/, runs the
CURRENT process_hdu for new mask/weight/sky, then compares pull =
(sci - sky) * sqrt(weight) on pixels unmasked in BOTH versions.
Reports std, robust sigma, tail fractions, KS distance, and quantiles.

Usage:
    pixi run python benchmarks/weight_pull_compare.py <pid> <hdu> [pid hdu ...]
"""

import sys

import numpy as np


def pull_stats(pull):
    from scipy.stats import kstest

    v = pull[np.isfinite(pull)]
    out = {
        "n": int(v.size),
        "std": float(np.std(v)),
        "rsig": float(1.4826 * np.median(np.abs(v))),
        "f3": float(np.mean(np.abs(v) > 3)),
        "f4": float(np.mean(np.abs(v) > 4)),
        "f5": float(np.mean(np.abs(v) > 5)),
        "ks": float(kstest(v[:: max(1, v.size // 20000)], "norm").statistic),
    }
    q = np.percentile(v, [1, 5, 50, 95, 99])
    out["q"] = [round(float(x), 3) for x in q]
    return out


def main():
    import fitsio
    import yaml

    pairs = [(sys.argv[i], int(sys.argv[i + 1])) for i in range(1, len(sys.argv), 2)]
    dl = "/scratch/sfabbro/downloads"
    wmd = "/arc/projects/mlao/cfhtcast/cfhtcast-data/wm"
    cfg = yaml.safe_load(open("weightmask.yml"))
    flat_of = {}
    with open("/arc/projects/mlao/cfhtcast/cfhtcast-data/flat_map_22_only.txt") as fh:
        fh.readline()
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                flat_of[p[0]] = p[2]
    with open("/arc/projects/mlao/cfhtcast/cfhtcast-data/flat_map.txt") as fh:
        fh.readline()
        for line in fh:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 3:
                flat_of.setdefault(p[0], p[2])

    agg = {"old": [], "new": []}
    for pid, hdu in pairs:
        with fitsio.FITS(f"{dl}/{pid}.fits.fz") as f:
            sci = np.ascontiguousarray(f[hdu].read().astype(np.float32))
        with fitsio.FITS(f"{wmd}/{pid}.weight.fits") as f:
            ow = f[hdu].read().astype(np.float64)
        with fitsio.FITS(f"{wmd}/{pid}.mask.fits") as f:
            om = f[hdu].read()
        with fitsio.FITS(f"{wmd}/{pid}.sky.fits") as f:
            osky = f[hdu].read().astype(np.float64)
        with fitsio.FITS(f"{dl}/{pid}.fits.fz") as hi, fitsio.FITS(flat_of[pid]) as hf:
            from weightmask.mef import process_hdu

            res = process_hdu(hi[hdu], hf[hdu] if hdu < len(hf) else None, cfg, hdu, tile_size=1024)
        nmask, ninv, nweight, nconf, nsky, _ = res
        print(f"{pid} hdu{hdu}: oldmask={float(((om & 55) > 0).mean()):.4f} "
              f"newmask={float(((nmask & 55) > 0).mean()):.4f}", flush=True)
        common = (om == 0) & (nmask == 0)
        for tag, wmap, skymap in (("old", ow, osky), ("new", nweight.astype(np.float64), nsky.astype(np.float64))):
            sel = common & np.isfinite(sci) & np.isfinite(skymap) & np.isfinite(wmap) & (wmap > 0)
            pull = (sci[sel].astype(np.float64) - skymap[sel]) * np.sqrt(wmap[sel])
            st = pull_stats(pull)
            agg[tag].append(pull)
            print(f"  {tag}: n={st['n']} std={st['std']:.3f} rsig={st['rsig']:.3f} "
                  f"f3={st['f3']:.4f} f4={st['f4']:.5f} f5={st['f5']:.5f} ks={st['ks']:.4f} q={st['q']}", flush=True)
    for tag in ("old", "new"):
        st = pull_stats(np.concatenate(agg[tag]))
        print(f"POOLED {tag}: n={st['n']} std={st['std']:.3f} rsig={st['rsig']:.3f} "
              f"f3={st['f3']:.4f} f4={st['f4']:.5f} f5={st['f5']:.5f} ks={st['ks']:.4f} q={st['q']}", flush=True)


if __name__ == "__main__":
    main()
