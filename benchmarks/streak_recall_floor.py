#!/usr/bin/env python3
"""Where does the shipped detector stop finding injected trails?

This started as a rescue on/off comparison and outlived its subject. The Radon
rescue it toggled is gone -- measured dead on both axes before removal (83 real
amps: 3 acceptances, all false positives; injected trails at 4/6/8/12 sigma:
+0.000 ``recall_line`` in 8 of 8 cells; 121.7 s/amp of a 125.7 s/amp stage).

What is worth keeping is the other half: the recall curve itself. The shipped
detector's earlier snapshot found both unconfirmed linear features in 996195p in 0.3 s and
masked nothing on 81 of the other 82 amps. Those are historical measurements,
not current timings or a measurement of where it stops working. This supplies
that curve, and it is the number any future sensitive stage has
to beat before it earns its cost back.

Trails are injected into raw science, then the production detector boundary is
captured again. Truth pixels overlapped by recomputed upstream exclusions are
reported separately from scoreable pixels.

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
import hashlib
import io
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

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

from production_inputs import (  # noqa: E402
    capture_detector_inputs,
    production_gain_read_noise,
    streak_config,
)
from streak_inject import inject_grid, inject_source_poisson, scoreable_truth, trail_recall  # noqa: E402

import weightmask.streaks as ST  # noqa: E402

LONG = os.path.join(REPO, "benchmark_data", "megacam", "long", "996195p.fits.fz")
PERF = os.path.join(REPO, "benchmark_data", "megacam", "perf")
FLAT = os.path.join(PERF, "flat_08Bm01_r.fits.fz")
# 996195p 35/36 carry the only unconfirmed linear features; 1013719p 9 is clean.
AMPS = [
    (LONG, 35, "unconfirmed linear feature"),
    (LONG, 36, "unconfirmed linear feature"),
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
METRIC_REVISION = "stage-recovery-v1"
EVIDENCE_SCHEMA = "stage-evidence-v1"


def _hash_value(value):
    digest = hashlib.sha256()
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(memoryview(array))
    elif isinstance(value, (bytes, bytearray, memoryview)):
        digest.update(bytes(value))
    elif isinstance(value, (list, tuple)):
        for item in value:
            digest.update(_hash_value(item).encode())
    else:
        digest.update(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode())
    return digest.hexdigest()


def _hash_paths(paths):
    digest = hashlib.sha256()
    for path in sorted((os.fspath(path) for path in paths)):
        digest.update(os.path.basename(path).encode())
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def _stage_source_paths():
    return (
        Path(REPO) / "weightmask" / "streaks.py",
        Path(REPO) / "weightmask" / "process.py",
    )


def _metric_code_paths():
    return (Path(REPO) / "benchmarks" / "streak_inject.py",)


def build_stage_evidence(records, input_data, config, source_paths=None):
    """Attach hashes needed to revalidate archived stage qualification."""
    sources = tuple(source_paths or _stage_source_paths())
    evidence = {
        "schema": EVIDENCE_SCHEMA,
        "input_hash": _hash_value(input_data),
        "config_hash": _hash_value(config),
        "source_hash": _hash_paths(sources),
        "metric_revision": METRIC_REVISION,
        "code_hash": _hash_paths(
            (Path(__file__), Path(__file__).with_name("streak_stage_sweep.py"), *_metric_code_paths())
        ),
        "metric_code_hash": _hash_paths(_metric_code_paths()),
        "records_hash": _hash_value(records),
    }
    return {"evidence": evidence, "records": records}


def verify_stage_evidence(artifact, input_data, config, source_paths=None):
    """Reject archived stage evidence that no longer describes current code."""
    evidence = artifact.get("evidence") if isinstance(artifact, dict) else None
    required = (
        "schema",
        "input_hash",
        "config_hash",
        "source_hash",
        "metric_revision",
        "code_hash",
        "metric_code_hash",
        "records_hash",
    )
    if not isinstance(evidence, dict) or any(key not in evidence for key in required):
        raise ValueError("archived stage evidence is missing provenance hashes")
    records = artifact.get("records")
    if records is None:
        records = {key: value for key, value in artifact.items() if key != "evidence"}
    expected = build_stage_evidence(records, input_data, config, source_paths)["evidence"]
    for key in required:
        if evidence[key] != expected[key]:
            label = "metric revision" if key == "metric_revision" else key.replace("_", " ")
            raise ValueError(f"archived stage evidence has stale {label}")
    return True


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
        gain, _read_noise = production_gain_read_noise(header, cfg)
        for length in lengths:
            for sigma in sigmas:
                for dashed in dashed_kinds:
                    for seed in seeds:
                        clean_baseline = ST.detect_streaks(data_sub, rms, existing, dict(scfg))
                        flux, _truth, trails, rejected = inject_grid(
                            data_sub.shape,
                            [(length, sigma, dashed)],
                            np.random.default_rng(seed),
                            500.0,
                            invalid_mask=~np.isfinite(science) | clean_baseline,
                            return_rejected=True,
                            rms_map=rms,
                        )
                        raw = inject_source_poisson(
                            science,
                            flux,
                            gain,
                            np.random.default_rng(seed + 10000),
                            invalid_mask=~np.isfinite(science),
                        )
                        with contextlib.redirect_stdout(io.StringIO()):
                            test, test_rms, test_existing = capture_detector_inputs(
                                raw, header, cfg, flat=flat
                            )
                        started = time.perf_counter()
                        with contextlib.redirect_stdout(io.StringIO()):
                            mask = ST.detect_streaks(test, test_rms, test_existing, dict(scfg))
                        elapsed = time.perf_counter() - started
                        score_truth, rejected_truth = scoreable_truth(
                            trails[0]["truth"], test, test_existing, clean_baseline
                        )
                        exact, tolerant, line = trail_recall(mask, score_truth)
                        out["rows"].append(
                            {
                                "length": length,
                                "sigma": sigma,
                                "dashed": bool(dashed),
                                "seed": seed,
                                "recall": round(exact, 3),
                                "recall5": round(tolerant, 3),
                                "recall_line": round(line, 3),
                                "scoreable_px": int(np.count_nonzero(score_truth)),
                                "rejected_px": int(np.count_nonzero(rejected_truth)),
                                "rejected_cells": len(rejected),
                                "px": int(mask.sum()),
                                "s": round(elapsed, 1),
                            }
                        )
        records = {key: value for key, value in out.items() if key != "evidence"}
        out["evidence"] = build_stage_evidence(records, (science, data_sub, rms, existing), scfg)["evidence"]
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
    return int(any(record.get("error") for record in results))


if __name__ == "__main__":
    sys.exit(main())
