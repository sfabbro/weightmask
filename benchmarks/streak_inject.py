#!/usr/bin/env python3
"""Streak injection-recovery harness on a real HDU, as a swept grid.

Injects synthetic trails (continuous and dashed, several lengths/brightnesses and
angles) into raw science, recomputes production detector inputs, runs
``detect_streaks``, and
reports per-trail recall plus novel false-positive pixels. Every grid cell is one
TSV row, so a threshold change is judged by diffing two runs instead of by
reading a log.

The injection and scoring helpers are importable, which is what
``tests/test_streak_recall_floor.py`` uses to pin a recall floor in CI.

Usage:
    pixi run python benchmarks/streak_inject.py <exposure.fits.fz> [--hdu N]
        [--seeds 3] [--out results.tsv]
        [--exposure-id NAME] [--mode auto_ground] [--set key=value ...]
        [--quick] [--no-cache]

Detector inputs are captured from the ``process_image`` streak stage, so the
background iteration and the exclusion mask match production. One background pass
over an unmasked frame leaves chip-fixed columns in ``data_sub``, which
manufactures false positives.
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SOURCE_DIR = Path(__file__).resolve().parents[1] / "weightmask"

# (length_px, peak_sigma, dashed): the compact grid used by ``--quick``.
QUICK_SPECS = [
    (1500, 8.0, False),
    (1500, 5.0, False),
    (800, 8.0, False),
    (800, 5.0, False),
    (400, 6.0, False),
    (400, 4.0, True),
]

SWEEP_LENGTHS = (300, 800, 1500)
SWEEP_SIGMAS = (4.0, 6.0, 8.0)
SWEEP_DASHED = (False, True)
SWEEP_DASHED_SIGMAS = (4.0, 6.0)


def sweep_specs(lengths=SWEEP_LENGTHS, sigmas=SWEEP_SIGMAS, dashed=SWEEP_DASHED, dashed_sigmas=SWEEP_DASHED_SIGMAS):
    """The full grid (15 trails by default).

    Dashed trails are only injected at the fainter levels: a dashed 8-sigma trail
    is a row of bright dashes, and grading a line-finder on it would say more
    about the grid than about the detector.
    """
    specs = []
    for length in lengths:
        for sigma in sigmas:
            specs.append((length, sigma, False))
            if dashed and sigma in dashed_sigmas:
                specs.append((length, sigma, True))
    return specs


def _point_segment_distance(point, start, end):
    segment = end - start
    length_sq = float(np.dot(segment, segment))
    if length_sq <= 0:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(np.dot(point - start, segment) / length_sq, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + fraction * segment)))


def _segment_distance(left, right):
    left_start, left_end = left
    right_start, right_end = right
    left_vector = left_end - left_start
    right_vector = right_end - right_start

    def cross(first, second):
        return float(first[0] * second[1] - first[1] * second[0])

    denominator = cross(left_vector, right_vector)
    offset = right_start - left_start
    if abs(denominator) > 1e-12:
        left_fraction = cross(offset, right_vector) / denominator
        right_fraction = cross(offset, left_vector) / denominator
        if 0.0 <= left_fraction <= 1.0 and 0.0 <= right_fraction <= 1.0:
            return 0.0
    return min(
        _point_segment_distance(left_start, right_start, right_end),
        _point_segment_distance(left_end, right_start, right_end),
        _point_segment_distance(right_start, left_start, left_end),
        _point_segment_distance(right_end, left_start, left_end),
    )


def _trail_segment(length, angle, cx, cy):
    direction = np.array([np.cos(angle), np.sin(angle)], dtype=np.float64)
    centre = np.array([cx, cy], dtype=np.float64)
    half = 0.5 * float(length) * direction
    return centre - half, centre + half


def place_trails(
    shape,
    specs,
    rng,
    min_separation=500.0,
    margin=0.15,
    attempts=200,
    rejected=None,
    band_half_width=2.0,
):
    """Pick non-overlapping trail placements, yielding the ones that fit.

    Crossings merge into tangles that no line-finder should fully recover, so
    overlapping placements would confound recall.
    """
    h, w = shape
    placed = []
    for length, peak_sig, dashed in specs:
        angle = float(rng.uniform(0.0, np.pi))
        found = None
        for _ in range(attempts):
            cand_cx = float(rng.uniform(w * margin, w * (1.0 - margin)))
            cand_cy = float(rng.uniform(h * margin, h * (1.0 - margin)))
            candidate_segment = _trail_segment(length, angle, cand_cx, cand_cy)
            if all(
                max(abs(cand_cx - pcx), abs(cand_cy - pcy)) > min_separation
                and _segment_distance(candidate_segment, _trail_segment(pl, pa, pcx, pcy))
                > 2.0 * band_half_width
                for pl, _, _, pa, pcx, pcy in placed
            ):
                found = (cand_cx, cand_cy)
                break
        if found is None:
            if rejected is not None:
                rejected.append(
                    {
                        "length": length,
                        "peak_sig": peak_sig,
                        "dashed": dashed,
                        "reason": "placement_failed",
                        "truth_px": 0,
                    }
                )
            continue
        placed.append((length, peak_sig, dashed, angle, found[0], found[1]))
        yield length, peak_sig, dashed, angle, found[0], found[1]


def trail_flux(
    shape,
    length,
    peak_sig,
    dashed,
    angle,
    cx,
    cy,
    half_width=2.0,
    dash_period=50.0,
    dash_on=30.0,
    profile_width=1.2,
    rms_map=None,
):
    """Flux in sky-sigma units for one trail, plus the mask of its main body."""
    h, w = shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dx, dy = np.cos(angle), np.sin(angle)
    along = (xx - cx) * dx + (yy - cy) * dy
    across = np.abs((xx - cx) * (-dy) + (yy - cy) * dx)
    on_trail = (np.abs(along) <= length / 2.0) & (across <= half_width)
    if dashed:
        on_trail &= np.mod(along, dash_period) < dash_on
    profile = np.exp(-0.5 * (across / profile_width) ** 2)
    body = on_trail & (profile > 0.3)
    flux = np.where(on_trail, peak_sig * profile, 0.0)
    if rms_map is not None:
        rms_map = np.asarray(rms_map, dtype=np.float32)
        if rms_map.shape != shape:
            raise ValueError("rms_map shape differs from the injection shape")
        scaled = np.zeros_like(flux)
        valid = on_trail & np.isfinite(rms_map) & (rms_map > 0)
        scaled[valid] = flux[valid] * rms_map[valid]
        flux = scaled
    return flux, body


def inject_grid(
    shape,
    specs,
    rng,
    min_separation=500.0,
    margin=0.15,
    invalid_mask=None,
    return_rejected=False,
    profile_width=1.6,
    rms_map=None,
):
    """Return ``(flux, truth, trails)``; ``trails`` carries each trail's own truth.

    Per-trail truth is tracked explicitly rather than recovered from connected
    components of the union, so recall is attributed to the trail that was
    actually placed there.
    """
    flux = np.zeros(shape, dtype=np.float32)
    truth = np.zeros(shape, dtype=bool)
    trails = []
    rejected = []
    valid_rms = None
    if rms_map is not None:
        rms_map = np.asarray(rms_map, dtype=np.float32)
        if rms_map.shape != shape:
            raise ValueError("rms_map shape differs from the injection shape")
        valid_rms = np.isfinite(rms_map) & (rms_map > 0)
    for length, peak_sig, dashed, angle, cx, cy in place_trails(
        shape, specs, rng, min_separation, margin, rejected=rejected, band_half_width=2.0
    ):
        trail, body = trail_flux(
            shape,
            length,
            peak_sig,
            dashed,
            angle,
            cx,
            cy,
            profile_width=profile_width,
            rms_map=rms_map,
        )
        overlap = np.any(body & truth)
        invalid = (invalid_mask is not None and np.any(body & invalid_mask)) or (
            valid_rms is not None and np.any(body & ~valid_rms)
        )
        if overlap or invalid:
            rejected.append(
                {
                    "length": length,
                    "peak_sig": peak_sig,
                    "dashed": dashed,
                    "reason": "truth_overlap" if overlap else "invalid_or_excluded",
                    "truth_px": int(np.count_nonzero(body)),
                }
            )
            continue
        flux += trail
        truth |= body
        trails.append(
            {
                "length": length,
                "peak_sig": peak_sig,
                "dashed": dashed,
                "angle_deg": round(float(np.degrees(angle)), 1),
                "cx": round(cx, 1),
                "cy": round(cy, 1),
                "truth": body,
            }
        )
    if return_rejected:
        return flux, truth, trails, rejected
    return flux, truth, trails


def inject_source_poisson(raw_science, source_adu, gain, rng, invalid_mask=None):
    """Add a source in ADU with Poisson counting noise in electrons."""
    gain = float(gain)
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("gain must be finite and positive")
    source = np.maximum(np.asarray(source_adu, dtype=np.float32), 0.0)
    if invalid_mask is not None:
        source = np.where(invalid_mask, 0.0, source)
    electrons = source.astype(np.float64) * gain
    realization = rng.poisson(electrons).astype(np.float32) / gain
    return np.asarray(raw_science, dtype=np.float32) + realization


def trail_recall(mask, body, dilation=5, line_half_width=2.0):
    """Three recalls for one trail's own truth mask.

    ``exact``
        Fraction of the truth band the mask covers directly.
    ``tolerant``
        Fraction of truth pixels within ``dilation`` pixels of the mask.
    ``line``
        Fraction of the truth band lying within ``line_half_width`` px of *any*
        mask pixel.

    ``line`` is the one that answers "was this trail found and localised". A
    detector emits a fitted centre line, typically 1 px wide, while the injected
    body is a 3.7 px-wide Gaussian band (``across < 1.86`` px). Comparing the two
    directly caps the score near 0.74 no matter how well centred the line is, so
    ``exact`` measures a width convention rather than a detection. Measured on
    1013719p HDU 19 at 20 sigma: ``exact`` 0.669 with a median transverse offset
    of 0.96 px and an along-trail span of 121% of the truth -- i.e. found dead
    centre and slightly long, yet scored as two-thirds recovered.
    """
    from scipy.ndimage import binary_dilation, distance_transform_edt

    n_px = int(np.count_nonzero(body))
    if n_px == 0 or not np.any(mask):
        return 0.0, 0.0, 0.0
    exact = float(np.count_nonzero(mask & body)) / n_px
    near_mask = binary_dilation(mask, iterations=dilation) if dilation > 0 else mask
    hits = int(np.count_nonzero(body & near_mask))
    distance = distance_transform_edt(~mask)
    line = float(np.count_nonzero(distance[body] <= line_half_width)) / n_px
    return exact, hits / n_px, line


def score_mask(mask, baseline, truth, dilation=5):
    """Novel-pixel false positives: exact, and outside a tolerance ring of truth."""
    from scipy.ndimage import binary_dilation

    novel = mask & ~baseline
    near = binary_dilation(truth, iterations=dilation)
    return int(np.count_nonzero(novel & ~truth)), int(np.count_nonzero(novel & ~near)), int(np.count_nonzero(novel))


def scoreable_truth(body, data_sub, existing, baseline):
    """Partition a trail truth mask after production recomputation."""
    rejected = ~np.isfinite(data_sub) | existing | baseline
    return body & ~rejected, body & rejected


def apply_overrides(config, pairs):
    """Apply ``key=value`` (optionally dotted) overrides in place and return it."""
    from weightmask.config import clean_config_dict

    for item in pairs or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, value = item.split("=", 1)
        parsed = clean_config_dict({key: value})[key]
        if isinstance(parsed, str):  # keep strings as strings for enumerated knobs
            parsed = value
        if "." in key:
            top, sub = key.split(".", 1)
            config[top] = {**config.get(top, {}), sub: parsed}
        else:
            config[key] = parsed
    return config


def baseline_key(data_sub, rms, existing, config):
    """Invalidate cached masks when production inputs, settings, or code change."""
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode())
    for value in (data_sub, rms, existing):
        if value is None:
            digest.update(b"None")
            continue
        array = np.ascontiguousarray(value)
        digest.update(str((array.shape, array.dtype.str)).encode())
        digest.update(memoryview(array))
    for path in sorted(SOURCE_DIR.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:24]


TSV_COLUMNS = [
    "exposure",
    "hdu",
    "seed",
    "trail",
    "length",
    "peak_sig",
    "dashed",
    "angle_deg",
    "cx",
    "cy",
    "truth_px",
    "scoreable_px",
    "rejected_px",
    "rejected_cells",
    "recall",
    "recall5",
    "recall_line",
    "fp_px",
    "fp5_px",
    "novel_px",
    "dt_s",
    "mode",
    "note",
    "metric_revision",
]


def format_tsv(rows):
    lines = ["\t".join(TSV_COLUMNS)]
    for row in rows:
        lines.append("\t".join(str(row.get(column, "")) for column in TSV_COLUMNS))
    return "\n".join(lines)


def summarise(rows):
    """Aggregate into ``(mean recall, mean 5px recall, mean line recall, mean fp5)``."""
    if not rows:
        return 0.0, 0.0, 0.0, 0.0
    recalls = [float(row["recall"]) for row in rows if row.get("recall") != ""]
    tolerant = [float(row["recall5"]) for row in rows if row.get("recall5") != ""]
    line = [float(row["recall_line"]) for row in rows if row.get("recall_line") != ""]
    fp5 = [float(row["fp5_px"]) for row in rows]
    return (
        sum(recalls) / max(1, len(recalls)),
        sum(tolerant) / max(1, len(tolerant)),
        sum(line) / max(1, len(line)),
        sum(fp5) / max(1, len(fp5)),
    )


def load_config(args):
    import yaml

    base_cfg = yaml.safe_load(open("weightmask.yml"))
    streak_cfg = dict(base_cfg["streak_masking"])
    streak_cfg["enable"] = True
    if args.mode:
        streak_cfg["mode"] = args.mode
    apply_overrides(streak_cfg, args.set)
    return base_cfg, streak_cfg


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exposure")
    parser.add_argument("--hdu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0, help="first seed; a run uses seed .. seed+N-1")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--out", default=None, help="write the grid to this TSV")
    parser.add_argument("--exposure-id", default=None, help="label for the TSV (default: the file stem)")
    parser.add_argument(
        "--flat",
        default=None,
        help="flat-field FITS for the same HDU. Strongly recommended: the flat is what lets the "
        "upstream bad-column and bleed stages mask chip-fixed structure, and without it the cheap "
        "prescreen can accept that structure as a 'trail'.",
    )
    parser.add_argument("--mode", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--quick", action="store_true", help="six-trail grid instead of the full sweep")
    parser.add_argument("--no-cache", action="store_true", help="recompute the injected-blank baseline")
    parser.add_argument("--cache-dir", default=os.path.join("/tmp", "wm_streak_baseline"))
    parser.add_argument("--min-separation", type=float, default=500.0)
    parser.add_argument("--summary-only", action="store_true", help="print the aggregate instead of the TSV")
    args = parser.parse_args(argv)
    if args.seeds < 1:
        parser.error("--seeds must be positive")

    import fitsio
    from production_inputs import capture_detector_inputs, production_gain_read_noise

    from weightmask.streaks import detect_streaks

    base_cfg, streak_cfg = load_config(args)
    exposure_id = args.exposure_id or os.path.basename(args.exposure).split(".")[0]
    specs = QUICK_SPECS if args.quick else sweep_specs()

    with fitsio.FITS(args.exposure, "r") as handle:
        sci = np.ascontiguousarray(handle[args.hdu].read().astype(np.float32))
        header = handle[args.hdu].read_header()

    flat = None
    if args.flat:
        with fitsio.FITS(args.flat, "r") as handle:
            flat = np.ascontiguousarray(handle[args.hdu].read().astype(np.float32))

    # Detector inputs must come from the production pipeline. A single background
    # pass over an unmasked frame leaves the chip-fixed columns and rows in
    # ``data_sub``: that manufactures false streak positives.
    # The recorded ``bin`` recall curve was non-monotonic for exactly that reason
    # -- it measured whether the sensitive stages ran, not resolution.
    base_cfg["streak_masking"] = dict(streak_cfg)
    clean_data_sub, clean_rms, clean_existing = capture_detector_inputs(sci, header, base_cfg, flat=flat)
    note = "production streak stage" + ("" if flat is not None else " (NO FLAT)")
    if flat is None:
        print("WARNING: no --flat given; upstream bad-column/bleed masking is much weaker than production.")
    print(f"existing_mask: {int(clean_existing.sum())} px ({note})")

    gain, _read_noise = production_gain_read_noise(header, base_cfg)
    cache_key = baseline_key(clean_data_sub, clean_rms, clean_existing, base_cfg)

    cache_path = os.path.join(args.cache_dir, f"{exposure_id}_{args.hdu}_{cache_key}_prod.npy")
    if not args.no_cache and os.path.exists(cache_path):
        clean_baseline = np.load(cache_path, allow_pickle=False).astype(bool)
        if clean_baseline.shape != sci.shape:
            raise ValueError("cached baseline shape differs from the science frame")
        print(f"baseline cached ({int(clean_baseline.sum())} px)")
    else:
        t0 = time.time()
        clean_baseline = detect_streaks(clean_data_sub, clean_rms, clean_existing, dict(streak_cfg))
        print(f"baseline fresh ({int(clean_baseline.sum())} px, {time.time() - t0:.1f}s)")
        if not args.no_cache:
            os.makedirs(args.cache_dir, exist_ok=True)
            np.save(cache_path, clean_baseline.astype(np.uint8))

    rows = []
    for seed in range(args.seed, args.seed + args.seeds):
        rng = np.random.default_rng(seed)
        flux, truth, trails, rejected = inject_grid(
            sci.shape,
            specs,
            rng,
            args.min_separation,
            invalid_mask=~np.isfinite(sci) | clean_baseline,
            return_rejected=True,
            rms_map=clean_rms,
        )
        raw_test = inject_source_poisson(sci, flux, gain, rng, invalid_mask=~np.isfinite(sci))
        data_sub, rms, existing = capture_detector_inputs(raw_test, header, base_cfg, flat=flat)

        t0 = time.time()
        mask = detect_streaks(data_sub, rms, existing, dict(streak_cfg))
        dt = time.time() - t0

        scoreable = np.zeros(sci.shape, dtype=bool)
        trail_partitions = []
        for trail in trails:
            accepted_truth, rejected_pixels = scoreable_truth(
                trail["truth"], data_sub, existing, clean_baseline
            )
            scoreable |= accepted_truth
            trail_partitions.append((accepted_truth, rejected_pixels))
        fp_px, fp5_px, novel_px = score_mask(mask, clean_baseline, scoreable)
        print(
            f"seed {seed}: trails={len(trails)} rejected_cells={len(rejected)} truth_px={int(truth.sum())} "
            f"scoreable_px={int(scoreable.sum())} fp_px={fp_px} fp5_px={fp5_px} {dt:.1f}s"
        )

        for index, (trail, (accepted_truth, rejected_pixels)) in enumerate(zip(trails, trail_partitions)):
            exact, tolerant, line = trail_recall(mask, accepted_truth)
            row = {key: value for key, value in trail.items() if key != "truth"}
            row.update(
                {
                    "exposure": exposure_id,
                    "hdu": args.hdu,
                    "seed": seed,
                    "trail": index,
                    "truth_px": int(np.count_nonzero(trail["truth"])),
                    "scoreable_px": int(np.count_nonzero(accepted_truth)),
                    "rejected_px": int(np.count_nonzero(rejected_pixels)),
                    "rejected_cells": len(rejected),
                    "recall": round(exact, 3),
                    "recall5": round(tolerant, 3),
                    "recall_line": round(line, 3),
                    "fp_px": fp_px,
                    "fp5_px": fp5_px,
                    "novel_px": novel_px,
                    "dt_s": round(dt, 1),
                    "mode": streak_cfg.get("mode", "auto_ground"),
                    "note": note,
                    "metric_revision": "truth-pixel-recall-v3",
                }
            )
            rows.append(row)

    mean_recall, mean_recall5, mean_line, mean_fp5 = summarise(rows)
    if args.summary_only:
        print(
            f"grid summary: trails={len(rows)} mean_recall={mean_recall:.3f} "
            f"mean_recall5={mean_recall5:.3f} mean_recall_line={mean_line:.3f} mean_fp5_px={mean_fp5:.0f}"
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
