#!/usr/bin/env python3
"""CR completeness / false-positive curves by injection into a real HDU.

Injects single-pixel CRs and multi-pixel worms at known positions into a real
image, runs ``detect_cosmic_rays`` under several configs, and reports completeness
per morphology plus false-positive pixels and runtime. Each (variant, seed) is one
TSV row, so two runs can be diffed instead of eyeballed.

The interesting variant is ``single-pass``: the production two-pass arrangement
runs full L.A.Cosmic twice (a strict base pass plus a loose niter=1 "faint" pass)
when a single looser pass followed by the same morphology gate might reach the
same completeness for less than half the cost. This benchmark is what decides it.

Usage:
    pixi run python benchmarks/cr_faint_curves.py <exposure.fits.fz> [--hdu N]
        [--seeds 1] [--out results.tsv] [--summary-only]
"""

import argparse
import os
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
        int(fp),
        int(np.count_nonzero(flag)),
    )


def variants(cosmic_base):
    """Config variants to compare, as ``name -> (override, description)``.

    ``cosmic_base`` is the ``cosmic_ray`` section of the project config.
    """
    base = cosmic_base
    faint_base = base["faint_cr"]
    return {
        "main": ({"faint_cr": {**faint_base, "enable": False}}, "base pass alone, no second pass"),
        "main+faint": (
            {"faint_cr": {**faint_base, "enable": True}},
            "production two-pass arrangement",
        ),
        "single-pass": (
            {"single_pass": True, "faint_cr": {**faint_base, "enable": True}},
            "one loose pass, morphology gate instead of a second L.A.Cosmic",
        ),
    }


TSV_COLUMNS = ["exposure", "hdu", "seed", "variant", "single_recall", "worm_recall", "fp_px", "total_px", "dt_s"]


def format_tsv(rows):
    lines = ["\t".join(TSV_COLUMNS)]
    for row in rows:
        lines.append("\t".join(str(row.get(column, "")) for column in TSV_COLUMNS))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exposure")
    parser.add_argument("--hdu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--out", default=None)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args(argv)

    import fitsio
    import yaml

    base_cfg = yaml.safe_load(open("weightmask.yml"))
    cosmic_base = base_cfg["cosmic_ray"]
    exposure_id = os.path.basename(args.exposure).split(".")[0]

    with fitsio.FITS(args.exposure, "r") as handle:
        sci = np.ascontiguousarray(handle[args.hdu].read().astype(np.float32))
        header = handle[args.hdu].read_header()
    saturation = float(header.get("SATURATE", 60000.0))
    gain = float(header.get("GAIN", 1.5))
    read_noise = float(header.get("RDNOISE", 5.0))
    empty = np.zeros(sci.shape, dtype=bool)
    sky, rms = estimate_background(sci, empty, base_cfg["sep_background"])
    assert sky is not None and rms is not None

    rows = []
    for seed in range(args.seed, args.seed + args.seeds):
        rng = np.random.default_rng(seed)
        injected, truth_single, truth_worm = inject(sci, sky, rms, rng)
        print(f"seed {seed}: singles={int(truth_single.sum())} worm_px={int(truth_worm.sum())}")
        for name, (override, description) in variants(cosmic_base).items():
            cfg = {**cosmic_base, **override}
            if isinstance(override.get("faint_cr"), dict):
                cfg["faint_cr"] = {**cosmic_base["faint_cr"], **override["faint_cr"]}
            t0 = time.time()
            flag = detect_cosmic_rays(injected, empty, saturation, gain, read_noise, cfg, bkg_rms_map=rms)
            dt = time.time() - t0
            single, worm, fp, total = score(flag, truth_single, truth_worm)
            print(f"  {name:18s} single={single:.3f} worm={worm:.3f} fp_px={fp:6d} total={total:6d} {dt:5.1f}s")
            rows.append(
                {
                    "exposure": exposure_id,
                    "hdu": args.hdu,
                    "seed": seed,
                    "variant": name,
                    "single_recall": round(single, 3),
                    "worm_recall": round(worm, 3),
                    "fp_px": fp,
                    "total_px": total,
                    "dt_s": round(dt, 1),
                    "_description": description,
                }
            )

    if args.summary_only:
        for name, (_, description) in variants(cosmic_base).items():
            subset = [row for row in rows if row["variant"] == name]
            if not subset:
                continue
            print(
                f"{name:18s} single={np.mean([r['single_recall'] for r in subset]):.3f} "
                f"worm={np.mean([r['worm_recall'] for r in subset]):.3f} "
                f"fp_px={np.mean([r['fp_px'] for r in subset]):.0f} "
                f"dt={np.mean([r['dt_s'] for r in subset]):.1f}s  # {description}"
            )
    if args.out:
        with open(args.out, "w") as handle:
            handle.write(format_tsv(rows) + "\n")
        print(f"wrote {len(rows)} rows to {args.out}")
    elif not args.summary_only:
        print(format_tsv(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
