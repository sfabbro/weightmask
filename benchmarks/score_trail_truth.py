#!/usr/bin/env python3
"""Score a streak detector against the curated real-MegaCam label set.

The fixture written by ``benchmarks/curate_trail_truth.py`` carries, for each
labelled entry, a line in *CCD pixel coordinates* and the evidence behind its
label.  This tool turns each line into the band of pixels it covers, runs a
detector on the same exposure/HDU, and reports:

* **recall** on ``trail`` entries -- did the detector mask the real thing?
* **false positives** on ``artefact`` entries -- did it mask clutter it must
  leave alone?  This is the number that matters while the fixture's ``trail``
  bucket is empty: injected Gaussian trails never test it, because they do not
  reproduce bad columns, register rows or the chip-edge band.

Because the labels are geometry rather than stored pixels, the science files are
required (they are not committed).  Any entry whose exposure cannot be found is
reported as skipped, never silently dropped.

Usage
-----
    pixi run python benchmarks/score_trail_truth.py --detector streaks
    pixi run python benchmarks/score_trail_truth.py --detector none --detector poloka
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
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

from production_inputs import capture_detector_inputs, streak_config  # noqa: E402

DEFAULT_FIXTURE = os.path.join("benchmarks", "trail_truth", "megacam_real_labels.json")
DEFAULT_REPORT = os.path.join("benchmarks", "trail_truth", "megacam_real_labels.score.json")
DETECTORS = ("streaks", "houghpeaks", "poloka", "none")


def labelled_band(entry, shape, corridor_px):
    """Rasterise an entry's labelled line into the pixels it actually covers."""
    from curate_trail_truth import _clip_line_to_shape

    point = entry["geometry"]["point"]
    direction = entry["geometry"]["direction"]
    clipped = _clip_line_to_shape(point, direction, shape)
    if clipped is None:
        return None
    (x0, y0), (x1, y1) = clipped
    h, w = shape
    length = float(np.hypot(x1 - x0, y1 - y0))
    steps = max(int(length), 2)
    t = np.linspace(0.0, 1.0, steps)
    xs = np.clip(np.round(x0 + (x1 - x0) * t).astype(np.intp), 0, w - 1)
    ys = np.clip(np.round(y0 + (y1 - y0) * t).astype(np.intp), 0, h - 1)
    band = np.zeros(shape, dtype=bool)
    band[ys, xs] = True
    try:
        from scipy import ndimage as ndi

        band = ndi.binary_dilation(band, iterations=max(1, int(round(corridor_px))))
    except Exception:  # pragma: no cover - scipy is a hard dependency elsewhere
        pass
    return band


def run_detector(detector, science, data_sub, rms, existing, header, streak_cfg):
    """Return a boolean mask from the named detector."""
    if detector == "none":
        return np.zeros(science.shape, dtype=bool)
    from weightmask.streaks import _detect_streaks_houghpeaks, detect_streaks

    if detector == "streaks":
        return detect_streaks(data_sub, rms, existing, streak_cfg)
    if detector == "houghpeaks":
        mask, _, _ = _detect_streaks_houghpeaks(data_sub, rms, existing, streak_cfg)
        return mask
    if detector == "poloka":
        from poloka_tracks import poloka_satellite_mask

        mask, _, _ = poloka_satellite_mask(
            science,
            np.full(science.shape, float(np.median(science)), dtype=np.float32),
            float(np.median(rms)),
            existing_mask=existing,
            sat_mask=science >= float(header.get("SATURATE", 0.0) or np.inf),
        )
        return mask
    raise ValueError(f"unknown detector {detector!r}")


def score_hdu(job):
    """Run every detector on one (exposure, HDU) and score the entries there."""
    import fitsio

    path, hdu, entries, config_path, detectors, corridor_px, quiet = job[:7]
    apply_persistence = bool(job[7]) if len(job) > 7 else False
    out = {"exposure": entries[0]["exposure"], "hdu": int(hdu), "results": {}, "error": None}
    try:
        with fitsio.FITS(path) as handle:
            science = np.ascontiguousarray(handle[hdu].read().astype(np.float32))
            header = handle[hdu].read_header()
        with contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext():
            config = yaml.safe_load(open(config_path))
            streak_cfg = streak_config(config)
            prior = _persistence_prior(path, header, science.shape) if apply_persistence else None
            data_sub, rms, existing = capture_detector_inputs(science, header, config, detector_prior=prior)
        # Rasterise every band once: the detectors share them.
        bands = [labelled_band(entry, science.shape, corridor_px) for entry in entries]
        for detector in detectors:
            started = time.time()
            with contextlib.redirect_stdout(io.StringIO()) if quiet else contextlib.nullcontext():
                mask = run_detector(detector, science, data_sub, rms, existing, header, streak_cfg)
            elapsed = time.time() - started
            rows = []
            for entry, band in zip(entries, bands):
                if band is None:
                    rows.append(
                        {
                            "key": entry_key(entry),
                            "label": entry["label"],
                            "band_px": 0,
                            "covered_px": 0,
                            "coverage": 0.0,
                        }
                    )
                    continue
                total = int(band.sum())
                covered = int(np.count_nonzero(mask & band))
                rows.append(
                    {
                        "key": entry_key(entry),
                        "label": entry["label"],
                        "band_px": total,
                        "covered_px": covered,
                        "coverage": (covered / total) if total else 0.0,
                    }
                )
            out["results"][detector] = {"seconds": round(elapsed, 2), "rows": rows}
    except Exception as exc:  # pragma: no cover - surfaced, not swallowed
        out["error"] = f"{type(exc).__name__}: {exc}"
    return out


def _ccd_name(header):
    for key in ("CCDNAME", "CCDNAM"):
        if key in header and str(header[key]).strip():
            return str(header[key]).strip()
    return None


def _calibration_frame(header):
    kinds = {"FLAT", "DARK", "BIAS", "ZERO", "DOMEFLAT", "SKYFLAT", "TWILIGHTFLAT"}
    return any(
        kinds.intersection(str(header.get(key, "")).upper().replace("_", " ").replace("-", " ").split())
        for key in ("OBSTYPE", "IMAGETYP", "EXPTYPE")
    )


def _persistence_prior(path, header, shape):
    """Columns and rows bright on two distinct other science files of this CCD."""
    from weightmask.streaks import persistent_axis_mask
    from weightmask.utils import paths_alias

    name = _ccd_name(header)
    if not name:
        return None
    directory = os.path.dirname(os.path.abspath(path))
    frames = []
    distinct_paths = []
    for fname in sorted(os.listdir(directory)):
        lower = fname.lower()
        if not lower.endswith((".fits", ".fz")) or lower.startswith(("flat_", "dark_", "bias_", "zero_")):
            continue
        other = os.path.join(directory, fname)
        if paths_alias(other, path) or any(paths_alias(other, previous) for previous in distinct_paths):
            continue
        distinct_paths.append(other)
        try:
            with fitsio.FITS(other) as handle:
                if _calibration_frame(handle[0].read_header()):
                    continue
                for ext in range(len(handle)):
                    try:
                        other_header = handle[ext].read_header()
                    except Exception:
                        continue
                    if _ccd_name(other_header) != name or _calibration_frame(other_header):
                        continue
                    data = np.ascontiguousarray(handle[ext].read(), dtype=np.float32)
                    if data.shape == shape:
                        frames.append(data)
                        break
        except Exception:
            continue
    if len(frames) < 2:
        return None
    return persistent_axis_mask(frames, min_other=2)


def entry_key(entry) -> str:
    return f"{entry['exposure']}:{entry['hdu']}:{entry['ccdname']}:{entry['geometry']['offset_px']}"


def _detected(row, min_coverage):
    return row["coverage"] >= min_coverage


def summarise(fixture, per_hdu, detectors, min_coverage, skip_px):
    """Aggregate per-HDU results into per-detector metrics."""
    totals = {
        detector: {
            "trail_entries": 0,
            "trail_detected": 0,
            "artefact_entries": 0,
            "artefact_false_positive": 0,
            "artefact_band_px": 0,
            "artefact_covered_px": 0,
            "seconds": 0.0,
            "hdus_scored": 0,
        }
        for detector in detectors
    }
    skipped = []
    for record in per_hdu:
        if record["error"]:
            skipped.append({"exposure": record["exposure"], "hdu": record["hdu"], "reason": record["error"]})
            continue
        for detector, result in record["results"].items():
            bucket = totals[detector]
            bucket["hdus_scored"] += 1
            bucket["seconds"] += result["seconds"]
            for row in result["rows"]:
                label = row["label"]
                if row.get("band_px", 0) < skip_px:
                    continue
                if label == "trail":
                    bucket["trail_entries"] += 1
                    bucket["trail_detected"] += int(_detected(row, min_coverage))
                elif label == "artefact":
                    bucket["artefact_entries"] += 1
                    bucket["artefact_false_positive"] += int(_detected(row, min_coverage))
                    bucket["artefact_band_px"] += row["band_px"]
                    bucket["artefact_covered_px"] += row["covered_px"]
    for detector, bucket in totals.items():
        entries = bucket["trail_entries"]
        bucket["trail_recall"] = (bucket["trail_detected"] / entries) if entries else None
        artefacts = bucket["artefact_entries"]
        bucket["artefact_fp_rate"] = (bucket["artefact_false_positive"] / artefacts) if artefacts else None
        bucket["artefact_pixel_coverage"] = (
            bucket["artefact_covered_px"] / bucket["artefact_band_px"] if bucket["artefact_band_px"] else None
        )
    return totals, skipped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--fixture", default=DEFAULT_FIXTURE)
    parser.add_argument("--config", default="weightmask.yml")
    parser.add_argument("--detector", action="append", choices=list(DETECTORS), help="repeatable; default 'streaks'")
    parser.add_argument("--data-root", help="override the fixture's data_root")
    parser.add_argument("--corridor-px", type=float, default=2.0, help="half-width of the labelled band")
    parser.add_argument("--min-coverage", type=float, default=0.25, help="band fraction that counts as 'masked'")
    parser.add_argument("--skip-px", type=int, default=8, help="ignore bands smaller than this")
    parser.add_argument(
        "--max-artefact-fp-rate", type=float, default=0.10, help="gate on the artefact false-positive rate"
    )
    parser.add_argument(
        "--min-trail-recall", type=float, default=0.0, help="gate on trail recall (vacuous while there are no trails)"
    )
    parser.add_argument("--exposures", help="comma-separated subset")
    parser.add_argument(
        "--hdu-list",
        help="comma-separated exposure:hdu pairs to restrict scoring to, e.g. 1013719p:17,1013720p:5",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument(
        "--apply-persistence",
        action="store_true",
        help="Exclude columns/rows bright on two other same-CCD exposures in this directory.",
    )
    parser.add_argument("--quiet", action="store_true", default=True)
    parser.add_argument("--verbose", dest="quiet", action="store_false")
    args = parser.parse_args(argv)

    if any(
        not math.isfinite(value) or not 0 <= value <= 1
        for value in (args.min_coverage, args.max_artefact_fp_rate, args.min_trail_recall)
    ):
        parser.error("coverage and recall/FP thresholds must be finite and between 0 and 1")
    if not math.isfinite(args.corridor_px) or args.corridor_px <= 0 or args.skip_px < 1:
        parser.error("--corridor-px and --skip-px must be positive")

    detectors = args.detector or ["streaks"]
    with open(args.fixture) as handle:
        fixture = json.load(handle)
    data_root = args.data_root or fixture["data_requirements"]["data_root"]
    wanted = set(args.exposures.split(",")) if args.exposures else None

    grouped = {}
    for entry in fixture["entries"]:
        if wanted and entry["exposure"] not in wanted:
            continue
        grouped.setdefault((entry["exposure"], entry["hdu"]), []).append(entry)

    if args.hdu_list:
        requested = set()
        for item in args.hdu_list.split(","):
            item = item.strip()
            if not item:
                continue
            exposure, _, hdu = item.rpartition(":")
            if not exposure or not hdu.isdigit():
                print(f"ERROR: --hdu-list entry {item!r} is not exposure:hdu", file=sys.stderr)
                return 2
            requested.add((exposure, int(hdu)))
        absent = requested - set(grouped)
        if absent:
            print(f"WARNING: no labelled entries for {sorted(absent)}", file=sys.stderr)
        grouped = {key: value for key, value in grouped.items() if key in requested}

    plan, jobs, missing = [], [], set()
    for (exposure, hdu), entries in sorted(grouped.items()):
        info = fixture["data_requirements"]["exposures"].get(exposure)
        path = info["path"] if info else os.path.join(data_root, exposure + ".fits")
        if args.data_root:
            path = os.path.join(data_root, os.path.basename(path))
        if not os.path.exists(path):
            missing.add(exposure)
            continue
        plan.append(((exposure, hdu), entries))
        job = (path, hdu, entries, args.config, detectors, args.corridor_px, args.quiet)
        if args.apply_persistence:
            job = (*job, True)
        jobs.append(job)

    if missing:
        print(f"WARNING: no science file for exposure(s): {', '.join(sorted(missing))}", file=sys.stderr)
    if not jobs:
        print(f"ERROR: nothing scoreable under {data_root}", file=sys.stderr)
        return 2

    print(f"Scoring {len(detectors)} detector(s) on {len(jobs)} annotated HDU(s)")
    started = time.time()
    per_hdu = []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for _, record in zip(plan, pool.map(score_hdu, jobs, chunksize=1)):
            per_hdu.append(record)
    print(f"Scored in {time.time() - started:.0f} s")

    totals, skipped = summarise(fixture, per_hdu, detectors, args.min_coverage, args.skip_px)

    # Record the config that produced these numbers. The 2026-09-18 score
    # artifact carried no config, so its timings could not be reconciled with a
    # direct measurement and the baseline was not reproducible.
    config_source = open(args.config).read()
    config_record = {
        "path": args.config,
        "sha256": hashlib.sha256(config_source.encode()).hexdigest(),
        "detector_inputs": "production (process_image streak stage)",
        "streak_masking": yaml.safe_load(config_source).get("streak_masking", {}),
    }

    lines = [
        "# Real-MegaCam label-set score",
        "",
        f"Fixture: `{args.fixture}` (schema {fixture['schema']}). Detectors: {', '.join(detectors)}.",
        f"Coverage threshold {args.min_coverage:.2f} of a {2 * args.corridor_px:.0f} px-wide labelled band.",
        f"Config `{config_record['path']}` sha256 `{config_record['sha256'][:16]}`. Detector inputs are captured from the",
        "`process_image` streak stage, so the exclusion mask matches production.",
        "",
        "| detector | trail recall | artefact entries | artefact FPs | FP rate | artefact pixel coverage | HDUs | seconds |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for detector in detectors:
        bucket = totals[detector]
        recall = "n/a" if bucket["trail_recall"] is None else f"{bucket['trail_recall']:.3f}"
        rate = "n/a" if bucket["artefact_fp_rate"] is None else f"{bucket['artefact_fp_rate']:.3f}"
        pixel = "n/a" if bucket["artefact_pixel_coverage"] is None else f"{bucket['artefact_pixel_coverage']:.5f}"
        lines.append(
            f"| {detector} | {recall} | {bucket['artefact_entries']} | {bucket['artefact_false_positive']} | "
            f"{rate} | {pixel} | {bucket['hdus_scored']} | {bucket['seconds']:.0f} |"
        )
    if fixture["summary"]["by_label"]["trail"] == 0:
        lines += [
            "",
            "Trail recall is `n/a`: the fixture carries no confirmed real trail. The line that can be scored is the",
            "false-positive rate on real clutter, which is what the audit's injected-trail suite cannot measure.",
        ]
    failures = [f"missing science exposure: {exposure}" for exposure in sorted(missing)]
    failures.extend(f"unscored {record['exposure']}:{record['hdu']}: {record['reason']}" for record in skipped)
    for detector in detectors:
        bucket = totals[detector]
        if bucket["hdus_scored"] != len(jobs) or not (bucket["trail_entries"] + bucket["artefact_entries"]):
            failures.append(f"{detector}: incomplete labelled evidence ({bucket['hdus_scored']}/{len(jobs)} HDUs)")
        if bucket["artefact_fp_rate"] is not None and bucket["artefact_fp_rate"] > args.max_artefact_fp_rate:
            failures.append(
                f"{detector}: artefact FP rate {bucket['artefact_fp_rate']:.3f} > {args.max_artefact_fp_rate:.3f}"
            )
        if bucket["trail_recall"] is not None and bucket["trail_recall"] < args.min_trail_recall:
            failures.append(f"{detector}: trail recall {bucket['trail_recall']:.3f} < {args.min_trail_recall:.3f}")
    if failures:
        lines += ["", "## Gate failures", ""] + [f"- {item}" for item in failures]

    report = {
        "metric_revision": "truth-band-recall-v2",
        "fixture": args.fixture,
        "detectors": detectors,
        "config": config_record,
        "thresholds": {"min_coverage": args.min_coverage, "corridor_px": args.corridor_px, "skip_px": args.skip_px},
        "totals": totals,
        "skipped": skipped + [{"exposure": exposure, "reason": "missing science file"} for exposure in sorted(missing)],
        "gate_failures": failures,
        "per_hdu": per_hdu,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
    with open(args.report, "w") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    with open(os.path.splitext(args.report)[0] + ".md", "w") as handle:
        handle.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Wrote {args.report}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
