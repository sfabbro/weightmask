#!/usr/bin/env python3
"""Where does the shipped detector stop finding injected trails?

This started as a rescue on/off comparison and outlived its subject. The Radon
rescue it toggled is gone -- measured dead on both axes before removal (83 real
amps: 3 acceptances, all false positives; injected trails at 4/6/8/12 sigma:
+0.000 ``recall_line`` in 8 of 8 cells; 121.7 s/amp of a 125.7 s/amp stage).

What is worth keeping is the other half: the recall curve itself. The shipped
detector finds both real trails in 996195p in 0.3 s and masks nothing at all on
81 of the other 82 amps, so there is no local measurement of where it *stops*
working. This supplies one, and it is the number any future sensitive stage has
to beat before it earns its cost back.

Trails are injected into ``data_sub`` after the production background, so the
detector sees its real input plus signal. That overstates recall slightly: a
trail bright enough to have influenced the background would be partly absorbed
by it. Recorded as a caveat, not corrected for.

``recall_line`` is the distance-based recall (any mask pixel within 5 px of the
injected line counts); the width-convention ``recall`` is reported beside it and
is not comparable across ``bin`` settings.

Usage:
    pixi run streak-recall-floor                        # the shipped floor
    pixi run streak-recall-floor -- --disable contours  # what contours uniquely finds
    pixi run streak-recall-floor -- --disable ransac    # what RANSAC uniquely finds (dashed trails)
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
for path in (REPO, MODULE_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from production_inputs import capture_detector_inputs, streak_config  # noqa: E402
from streak_inject import inject_grid, trail_recall  # noqa: E402

import weightmask.streaks as ST  # noqa: E402

LONG = os.path.join(REPO, "benchmark_data", "megacam", "long", "996195p.fits.fz")
PERF = os.path.join(REPO, "benchmark_data", "megacam", "perf")
FLAT = os.path.join(PERF, "flat_08Bm01_r.fits.fz")
# 996195p 35/36 carry the only real linear features; 1013719p 9 is clean.
AMPS = [
    (LONG, 35, "real linear feature"),
    (LONG, 36, "real linear feature"),
    (f"{PERF}/1013719p.fits.fz", 9, "clean"),
]
SIGMAS = (4.0, 6.0, 8.0, 12.0)
LENGTHS = (800, 1500)
SEEDS = (0, 1)
STAGES = ("contours", "houghpeaks", "ransac")
# Dashed trails are the sparse RANSAC stage's whole remit. Measuring it on
# continuous trails only would repeat the mistake this benchmark already caught
# once: a stage judged on inputs it was never built for.
DASHED_KINDS = (False, True)


def run_one(job):
    path, hdu, label, sigmas, lengths, seeds, disable, dashed_kinds = job
    out = {
        "exposure": os.path.basename(path).split(".")[0],
        "hdu": int(hdu),
        "label": label,
        "disabled": list(disable),
        "rows": [],
    }
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
        # Marginal-value measurement: switch a named stage off and re-run the same
        # grid, so a stage's cost can be weighed against what it uniquely finds.
        # Stages are switched by their own config block, never by monkeypatching,
        # so what is measured is what ships.
        for stage in disable:
            if stage == "contours":
                scfg["contour_params"] = {**scfg["contour_params"], "enable": False}
            elif stage == "houghpeaks":
                scfg["houghpeak_params"] = {**scfg["houghpeak_params"], "enable": False}
            elif stage == "ransac":
                scfg["enable_sparse_ransac"] = False
            else:
                raise ValueError(f"unknown stage {stage!r}; stages are {STAGES}")
        with contextlib.redirect_stdout(io.StringIO()):
            data_sub, rms, existing = capture_detector_inputs(science, header, cfg, flat=flat)
        finite = data_sub[np.isfinite(data_sub)]
        noise = float(np.median(np.abs(finite - np.median(finite))) * 1.4826)
        for length in lengths:
            for sigma in sigmas:
                for dashed in dashed_kinds:
                    for seed in seeds:
                        flux, _truth, trails = inject_grid(
                            data_sub.shape,
                            [(length, sigma, dashed)],
                            np.random.default_rng(seed),
                            500.0,
                        )
                        test = data_sub + flux * noise
                        started = time.perf_counter()
                        with contextlib.redirect_stdout(io.StringIO()):
                            mask = ST.detect_streaks(test, rms, existing, dict(scfg))
                        elapsed = time.perf_counter() - started
                        exact, tolerant, line = trail_recall(mask, trails[0]["truth"])
                        out["rows"].append(
                            {
                                "length": length,
                                "sigma": sigma,
                                "dashed": bool(dashed),
                                "seed": seed,
                                "recall": round(exact, 3),
                                "recall5": round(tolerant, 3),
                                "recall_line": round(line, 3),
                                "px": int(mask.sum()),
                                "s": round(elapsed, 1),
                            }
                        )
    except Exception as exc:  # pragma: no cover - surfaced, never swallowed
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=os.path.join(MODULE_DIR, "streak_recall_floor.json"))
    parser.add_argument(
        "--disable",
        action="append",
        default=[],
        choices=STAGES,
        help="switch a stage off for the whole grid; repeatable. Measures what a stage "
        "uniquely finds, at its true cost. Omit to measure the shipped floor.",
    )
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args(argv)
    jobs = [(p, h, lab, SIGMAS, LENGTHS, SEEDS, tuple(args.disable), DASHED_KINDS) for p, h, lab in AMPS]
    print(
        f"{len(jobs)} amps x {len(LENGTHS)} lengths x {len(SIGMAS)} sigmas x {len(DASHED_KINDS)} "
        f"trail kinds x {len(SEEDS)} seeds through the production path; "
        f"disabled stages: {args.disable or 'none'}",
        flush=True,
    )
    results = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for record in pool.map(run_one, jobs):
            results.append(record)
            print(f"  {record['exposure']}:{record['hdu']} done ({len(record.get('rows', []))} rows)", flush=True)
    print(f"done in {time.time() - started:.0f}s", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as handle:
        json.dump(results, handle, indent=1)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
