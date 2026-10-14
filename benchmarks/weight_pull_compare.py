#!/usr/bin/env python3
"""Background-pull comparison: old batch weight maps vs current code.

For each (pid, hdu): loads science + OLD mask/inverse-variance/sky from wm/,
runs the CURRENT process_hdu for new products, then compares physical Gaussian
pulls on pixels unmasked in BOTH versions.
Reports std, robust sigma, tail fractions, KS distance, and quantiles.

Usage:
    pixi run python benchmarks/weight_pull_compare.py <pid> <hdu> [pid hdu ...]
"""

import os
import sys

import numpy as np


def gaussian_pulls(science, sky, inverse_variance, mask=None):
    science = np.asarray(science, dtype=np.float64)
    sky = np.asarray(sky, dtype=np.float64)
    inverse_variance = np.asarray(inverse_variance, dtype=np.float64)
    if science.shape != sky.shape or science.shape != inverse_variance.shape:
        raise ValueError("science, sky, and inverse_variance must have matching shapes")
    usable = np.isfinite(science) & np.isfinite(sky) & np.isfinite(inverse_variance) & (inverse_variance > 0)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != science.shape:
            raise ValueError("mask must have the same shape as science")
        usable &= ~mask
    return (science[usable] - sky[usable]) * np.sqrt(inverse_variance[usable])


def _pull_series(science, common, old_ivar, old_sky, new_ivar, new_sky):
    return (
        ("old", gaussian_pulls(science, old_sky, old_ivar, mask=~common)),
        ("new", gaussian_pulls(science, new_sky, new_ivar, mask=~common)),
    )


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
        raise ValueError(f"expected one product HDU for identity {identity}, found {len(matches)}")
    return handle[matches[0]]


def _validate_inverse_variance_header(header):
    artifact = header.get("WMART")
    semantics = header.get("WMSEM")
    if isinstance(artifact, bytes):
        artifact = artifact.decode("utf-8", errors="replace")
    if isinstance(semantics, bytes):
        semantics = semantics.decode("utf-8", errors="replace")
    if str(artifact).strip().lower() != "inverse_variance" or str(semantics).strip() != "inverse_variance_adu^-2":
        raise ValueError("product is not declared as inverse_variance with inverse_variance_adu^-2 semantics")


def _validate_product_header(header, artifact, semantics):
    actual_artifact = header.get("WMART")
    actual_semantics = header.get("WMSEM")
    if isinstance(actual_artifact, bytes):
        actual_artifact = actual_artifact.decode("utf-8", errors="replace")
    if isinstance(actual_semantics, bytes):
        actual_semantics = actual_semantics.decode("utf-8", errors="replace")
    actual_artifact = str(actual_artifact).strip().lower()
    expected_artifacts = {artifact} if isinstance(artifact, str) else set(artifact)
    if actual_artifact not in expected_artifacts or str(actual_semantics).strip() != semantics:
        expected = "/".join(sorted(expected_artifacts))
        raise ValueError(f"product metadata must declare {expected}/{semantics}")


def pull_stats(pull):
    from scipy.stats import kstest

    v = pull[np.isfinite(pull)]
    out = {
        "n": int(v.size),
        "std": float(np.std(v)),
        "rsig": float(1.4826 * np.median(np.abs(v - np.median(v)))),
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
            science_header = f[hdu].read_header()
        old_ivar_path = f"{wmd}/{pid}.ivar.fits"
        if not os.path.exists(old_ivar_path):
            old_ivar_path = f"{wmd}/{pid}.weight.fits"
        with (
            fitsio.FITS(old_ivar_path) as f_ivar,
            fitsio.FITS(f"{wmd}/{pid}.mask.fits") as f_mask,
            fitsio.FITS(f"{wmd}/{pid}.sky.fits") as f_sky,
        ):
            old_ivar_hdu = _matching_hdu(f_ivar, science_header)
            old_mask_hdu = _matching_hdu(f_mask, science_header)
            old_sky_hdu = _matching_hdu(f_sky, science_header)
            _validate_inverse_variance_header(_read_header(old_ivar_hdu))
            _validate_product_header(_read_header(old_mask_hdu), "quality_mask", "named_quality_bits")
            _validate_product_header(_read_header(old_sky_hdu), ("sky", "sky_mesh"), "background_adu")
            ow = old_ivar_hdu.read().astype(np.float64)
            om = old_mask_hdu.read()
            osky = old_sky_hdu.read().astype(np.float64)
        with fitsio.FITS(f"{dl}/{pid}.fits.fz") as hi, fitsio.FITS(flat_of[pid]) as hf:
            from weightmask.mef import process_hdu

            science_hdu = hi[hdu]
            flat_hdu = _matching_hdu(hf, science_header)
            res = process_hdu(science_hdu, flat_hdu, cfg, hdu, tile_size=1024)
        nmask, ninv, _nweight, _nconf, nsky, _ = res
        print(f"{pid} hdu{hdu}: oldmask={float(((om & 55) > 0).mean()):.4f} "
              f"newmask={float(((nmask & 55) > 0).mean()):.4f}", flush=True)
        common = (om == 0) & (nmask == 0)
        for tag, pull in _pull_series(sci, common, ow, osky, ninv.astype(np.float64), nsky.astype(np.float64)):
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
