#!/usr/bin/env python3
"""Which streak stage does the work on real MegaCam amps, and what does it cost?

Per HDU, on **production** inputs -- captured from the ``process_image`` streak
stage, not a re-derivation -- record for every stage: wall time, how many
candidates it accepted, and how many of the *final* mask's pixels its own mask
explains. Also characterise each surviving component (span, width, brightness,
star attachment) so bleed and satellite trails can be told apart on evidence.

Sampling is validated, not assumed. Every-4th-HDU is a reasonable default for a
wide survey but it missed the only real trails in the local corpus, twice, and
both misses silently understated what the cheap prescreen could do. So the HDU
list is checked against the amps that actually matter before any work starts,
and the run aborts if they are missing.

Usage:
    pixi run python benchmarks/streak_stage_sweep.py --out sweeps.json
    pixi run python benchmarks/streak_stage_sweep.py --no-rescue --out no_rescue.json
    pixi run python benchmarks/streak_stage_sweep.py --dense --out dense.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import fitsio
import numpy as np
import yaml

MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
if MODULE_DIR not in sys.path:
    sys.path.insert(0, MODULE_DIR)
REPO = os.path.dirname(MODULE_DIR)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from production_inputs import capture_detector_inputs, streak_config  # noqa: E402

import weightmask.streaks as ST  # noqa: E402

DATA = os.path.join(REPO, "benchmark_data", "megacam")
FLAT = os.path.join(DATA, "perf", "flat_08Bm01_r.fits.fz")
STRIDE = 4

EXPOSURES = [
    ("long/996195p.fits.fz", "996195p"),
    ("known/2079618p.fits.fz", "2079618p"),
    ("perf/1013719p.fits.fz", "1013719p"),
    ("perf/1013720p.fits.fz", "1013720p"),
    ("perf/1013721p.fits.fz", "1013721p"),
    ("megacam_streak_case.fits", "megacam_streak_case"),
]

# The only two amps in the local corpus carrying a real linear feature, and the
# only reason a stride of 4 is unsafe. Keep this in step with the amp list.
REQUIRED = {("996195p", 35), ("996195p", 36)}

STAGES = (
    "_detect_streaks_houghpeaks",
    "_detect_streaks_contours",
    "_detect_streaks_mrt_like",
    "_detect_trails_sparse_ransac",
)


def component_table(mask, data_sub, rms):
    """Geometry, brightness and star attachment for each sizeable component."""
    from scipy import ndimage as ndi

    labeled, n = ndi.label(mask, structure=np.ones((3, 3)))
    rows = []
    bright = (data_sub > 10.0 * np.nanmedian(rms)) & np.isfinite(data_sub)
    blab, bn = ndi.label(bright)
    blobs = [np.nonzero(blab == i) for i in range(1, bn + 1)] if bn else []
    for i in range(1, n + 1):
        ys, xs = np.nonzero(labeled == i)
        if ys.size < 80:
            continue
        y = ys.astype(float) - ys.mean()
        x = xs.astype(float) - xs.mean()
        theta = 0.5 * np.arctan2(2.0 * np.dot(x, y), np.dot(x, x) - np.dot(y, y))
        across = x * (-np.sin(theta)) + y * np.cos(theta)
        along = x * np.cos(theta) + y * np.sin(theta)
        mad = 1.4826 * float(np.median(np.abs(across - np.median(across))))
        values = data_sub[ys, xs]
        attach = 0
        for by, bx in blobs:
            if by.size < 6:
                continue
            bxc, byc = bx.mean() - xs.mean(), by.mean() - ys.mean()
            t = abs(bxc * (-np.sin(theta)) + byc * np.cos(theta))
            s = abs(bxc * np.cos(theta) + byc * np.sin(theta))
            if t <= 25.0 and s <= 0.5 * max(1.0, along.max() - along.min()):
                attach += 1
        rows.append(
            {
                "px": int(ys.size),
                "span": round(float(along.max() - along.min()), 1),
                "angle_deg": round(float(np.degrees(theta)), 2),
                "across_mad": round(mad, 2),
                "median_sub": round(float(np.median(values)), 1),
                "p90_over_rms": round(
                    float(np.percentile(values / np.maximum(rms[ys, xs], 1e-6), 90)), 2
                ),
                "bright_blobs_on_line": attach,
                "centroid": [int(ys.mean()), int(xs.mean())],
            }
        )
    rows.sort(key=lambda r: -r["px"])
    return rows


def run_one(job):
    path, hdu, exposure, no_rescue = job
    out = {"exposure": exposure, "hdu": int(hdu), "error": None, "no_rescue": bool(no_rescue)}
    try:
        with fitsio.FITS(path) as handle:
            science = np.ascontiguousarray(handle[hdu].read().astype(np.float32))
            header = handle[hdu].read_header()
        flat = None
        if os.path.exists(FLAT):
            with fitsio.FITS(FLAT) as handle:
                if hdu < len(handle) and handle[hdu].read().shape == science.shape:
                    flat = np.ascontiguousarray(handle[hdu].read().astype(np.float32))
        cfg = yaml.safe_load(open(os.path.join(REPO, "weightmask.yml")))
        scfg = streak_config(cfg)
        if no_rescue:
            scfg["mrt_rescue_params"] = {**scfg["mrt_rescue_params"], "enable": False}
        with contextlib.redirect_stdout(io.StringIO()):
            data_sub, rms, existing = capture_detector_inputs(science, header, cfg, flat=flat)

        originals = {name: getattr(ST, name) for name in STAGES}
        record = {}

        def wrap(name):
            fn = originals[name]

            def inner(*args, **kwargs):
                started = time.perf_counter()
                try:
                    result = fn(*args, **kwargs)
                finally:
                    slot = record.setdefault(name, {"s": 0.0, "calls": 0, "accepted": [], "mask": None})
                    slot["s"] += time.perf_counter() - started
                    slot["calls"] += 1
                if isinstance(result, tuple) and len(result) >= 2 and isinstance(result[1], list):
                    slot["accepted"].append(len(result[1]))
                    slot["mask"] = np.asarray(result[0], dtype=bool)
                else:
                    slot["accepted"].append(-1)
                    slot["mask"] = np.asarray(result, dtype=bool)
                return result

            setattr(ST, name, inner)

        for name in STAGES:
            wrap(name)
        try:
            clock = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                mask = ST.detect_streaks(data_sub, rms, existing, dict(scfg))
            out["streak_s"] = round(time.perf_counter() - clock, 1)
        finally:
            for name, fn in originals.items():
                setattr(ST, name, fn)

        out["mask_px"] = int(mask.sum())
        out["premasked_frac"] = round(float((mask & existing).sum() / max(1, mask.sum())), 3)
        out["existing_px"] = int(existing.sum())
        out["stages"] = {}
        for name in STAGES:
            slot = record.get(name)
            if slot is None:
                out["stages"][name] = {"ran": False}
                continue
            stage_mask = slot["mask"]
            out["stages"][name] = {
                "ran": True,
                "s": round(slot["s"], 1),
                "calls": slot["calls"],
                "accepted": slot["accepted"],
                "own_px": int(stage_mask.sum()),
                "explains_final_px": int((stage_mask & mask).sum()),
            }
        out["components"] = component_table(mask, data_sub, rms)
    except Exception as exc:  # pragma: no cover - surfaced, never swallowed
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def build_jobs(dense, no_rescue):
    jobs = []
    for relative, exposure in EXPOSURES:
        path = os.path.join(DATA, relative)
        if not os.path.exists(path):
            continue
        with fitsio.FITS(path) as handle:
            count = len(handle)
        if dense:
            hdus = list(range(1, count))
        else:
            hdus = list(range(1, count, STRIDE))
        for hdu in hdus:
            jobs.append((path, hdu, exposure, no_rescue))
    return jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=os.path.join(MODULE_DIR, "streak_stage_sweep.json"))
    parser.add_argument("--dense", action="store_true", help="scan every HDU of every exposure")
    parser.add_argument("--no-rescue", action="store_true", help="disable the Radon rescue")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)

    jobs = build_jobs(args.dense, args.no_rescue)
    sampled = {(job[2], job[1]) for job in jobs}
    missing = REQUIRED - sampled
    if missing:
        print(
            f"ERROR: the HDU sample omits {sorted(missing)}, which carry the only real "
            f"linear features in the local corpus. A stride of {STRIDE} hides them and "
            f"understates what the prescreen can do. Re-run with --dense.",
            file=sys.stderr,
        )
        return 2
    print(f"{len(jobs)} HDUs across {len({j[2] for j in jobs})} exposures, rescue "
          f"{'off' if args.no_rescue else 'on'}; required amps {sorted(REQUIRED)} present")
    results = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for index, record in enumerate(pool.map(run_one, jobs), 1):
            results.append(record)
            flag = record["error"] or f"{record['mask_px']:6d}px {record['streak_s']:5.1f}s"
            print(f"  [{index:2d}/{len(jobs)}] {record['exposure']}:{record['hdu']:<3d} {flag}", flush=True)
    print(f"done in {time.time() - started:.0f}s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(results, handle, indent=1)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
