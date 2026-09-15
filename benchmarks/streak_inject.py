#!/usr/bin/env python3
"""Streak injection-recovery harness on a real HDU, as a swept grid.

Injects synthetic trails (continuous and dashed, several lengths/brightnesses and
angles) into a background-subtracted real image, runs ``detect_streaks``, and
reports per-trail recall plus novel false-positive pixels. Every grid cell is one
TSV row, so a threshold change is judged by diffing two runs instead of by
reading a log.

The injection and scoring helpers are importable, which is what
``tests/test_streak_recall_floor.py`` uses to pin a recall floor in CI.

Usage:
    pixi run python benchmarks/streak_inject.py <exposure.fits.fz> [--hdu N]
        [--seeds 3] [--out results.tsv] [--mask-path MASK.fits]
        [--exposure-id NAME] [--mode auto_ground] [--set key=value ...]
        [--quick] [--no-cache]

``--mask-path`` replaces the previously hardcoded production location; when it is
omitted a mask is looked for beside the exposure, and when none exists an empty
mask is used (recorded in the TSV's ``note`` column).
"""

import argparse
import hashlib
import os
import sys
import time

import numpy as np

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


def place_trails(shape, specs, rng, min_separation=500.0, margin=0.15, attempts=200):
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
            if all(max(abs(cand_cx - px), abs(cand_cy - py)) > min_separation for px, py, _ in placed):
                found = (cand_cx, cand_cy)
                break
        if found is None:
            continue
        placed.append((found[0], found[1], angle))
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
    flux = np.where(on_trail, peak_sig * profile, 0.0)
    return flux, on_trail & (flux > 0.3 * peak_sig)


def inject_grid(shape, specs, rng, min_separation=500.0, margin=0.15):
    """Return ``(flux, truth, trails)``; ``trails`` carries each trail's own truth.

    Per-trail truth is tracked explicitly rather than recovered from connected
    components of the union, so recall is attributed to the trail that was
    actually placed there.
    """
    flux = np.zeros(shape, dtype=np.float32)
    truth = np.zeros(shape, dtype=bool)
    trails = []
    for length, peak_sig, dashed, angle, cx, cy in place_trails(shape, specs, rng, min_separation, margin):
        trail, body = trail_flux(shape, length, peak_sig, dashed, angle, cx, cy)
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
    return flux, truth, trails


def trail_recall(mask, body, dilation=5):
    """Recall and 5px-tolerant recall for one trail's own truth mask."""
    from scipy.ndimage import binary_dilation

    n_px = int(np.count_nonzero(body))
    if n_px == 0:
        return 0.0, 0.0
    exact = float(np.count_nonzero(mask & body)) / n_px
    hits = int(np.count_nonzero(mask & binary_dilation(body, iterations=dilation)))
    return exact, min(1.0, hits / n_px)


def score_mask(mask, baseline, truth, dilation=5):
    """Novel-pixel false positives: exact, and outside a tolerance ring of truth."""
    from scipy.ndimage import binary_dilation

    novel = mask & ~baseline
    near = binary_dilation(truth, iterations=dilation)
    return int(np.count_nonzero(novel & ~truth)), int(np.count_nonzero(novel & ~near)), int(np.count_nonzero(novel))


def load_existing_mask(mask_path, hdu, shape):
    """Read a production mask HDU if one is available, else an empty mask."""
    if not mask_path:
        return np.zeros(shape, dtype=bool), "no mask file"
    import fitsio

    with fitsio.FITS(mask_path, "r") as handle:
        if hdu >= len(handle):
            return np.zeros(shape, dtype=bool), f"mask has no HDU {hdu}"
        raw = handle[hdu].read()
    if raw.shape != shape:
        return np.zeros(shape, dtype=bool), f"mask HDU shape {raw.shape} != {shape}"
    return (raw & 55) > 0, "production mask"


def default_mask_path(exposure_path):
    """Look for ``<pid>.mask.fits`` beside the exposure."""
    directory, name = os.path.split(exposure_path)
    stem = name.split(".")[0]
    for candidate in (os.path.join(directory, f"{stem}.mask.fits"), os.path.join(directory, f"{stem}.mask.fits.fz")):
        if os.path.exists(candidate):
            return candidate
    return None


def apply_overrides(config, pairs):
    """Apply ``key=value`` (optionally dotted) overrides in place and return it."""
    from weightmask.utils import clean_config_dict

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


def config_key(config):
    """Stable short hash of a config, used to cache the injected-blank baseline."""
    payload = repr(sorted((k, repr(v)) for k, v in config.items() if k != "debug"))
    return hashlib.md5(payload.encode()).hexdigest()[:8]


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
    "recall",
    "recall5",
    "fp_px",
    "fp5_px",
    "novel_px",
    "dt_s",
    "mode",
    "note",
]


def format_tsv(rows):
    lines = ["\t".join(TSV_COLUMNS)]
    for row in rows:
        lines.append("\t".join(str(row.get(column, "")) for column in TSV_COLUMNS))
    return "\n".join(lines)


def summarise(rows):
    """Aggregate the grid into ``(mean recall, mean 5px recall, mean fp5)``."""
    if not rows:
        return 0.0, 0.0, 0.0
    recalls = [float(row["recall"]) for row in rows if row.get("recall") != ""]
    tolerant = [float(row["recall5"]) for row in rows if row.get("recall5") != ""]
    fp5 = [float(row["fp5_px"]) for row in rows]
    return (
        sum(recalls) / max(1, len(recalls)),
        sum(tolerant) / max(1, len(tolerant)),
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
    parser.add_argument("--mask-path", default=None, help="production mask (default: beside the exposure)")
    parser.add_argument("--exposure-id", default=None, help="label for the TSV (default: the file stem)")
    parser.add_argument("--mode", default=None)
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--quick", action="store_true", help="six-trail grid instead of the full sweep")
    parser.add_argument("--no-cache", action="store_true", help="recompute the injected-blank baseline")
    parser.add_argument("--cache-dir", default=os.path.join("/tmp", "wm_streak_baseline"))
    parser.add_argument("--min-separation", type=float, default=500.0)
    parser.add_argument("--summary-only", action="store_true", help="print the aggregate instead of the TSV")
    args = parser.parse_args(argv)

    import fitsio
    from astropy.stats import mad_std

    from weightmask.background import estimate_background
    from weightmask.streaks import detect_streaks

    base_cfg, streak_cfg = load_config(args)
    exposure_id = args.exposure_id or os.path.basename(args.exposure).split(".")[0]
    specs = QUICK_SPECS if args.quick else sweep_specs()

    with fitsio.FITS(args.exposure, "r") as handle:
        sci = np.ascontiguousarray(handle[args.hdu].read().astype(np.float32))

    mask_path = args.mask_path if args.mask_path is not None else default_mask_path(args.exposure)
    existing, note = load_existing_mask(mask_path, args.hdu, sci.shape)
    print(f"existing_mask: {int(existing.sum())} px ({note})")

    sky, rms = estimate_background(sci, existing, base_cfg["sep_background"])
    data_sub = sci - sky
    finite = data_sub[np.isfinite(data_sub)]
    noise = float(mad_std(finite[:: max(1, finite.size // 200000)]))

    rows = []
    for seed in range(args.seed, args.seed + args.seeds):
        rng = np.random.default_rng(seed)
        flux, truth, trails = inject_grid(sci.shape, specs, rng, args.min_separation)
        data_test = data_sub + flux * noise

        cache_path = os.path.join(args.cache_dir, f"{exposure_id}_{args.hdu}_{seed}_{config_key(streak_cfg)}.npy")
        if not args.no_cache and os.path.exists(cache_path):
            baseline = np.load(cache_path).astype(bool)
            print(f"seed {seed}: baseline cached ({int(baseline.sum())} px)")
        else:
            t0 = time.time()
            baseline = detect_streaks(data_sub, rms, existing, dict(streak_cfg))
            print(f"seed {seed}: baseline fresh ({int(baseline.sum())} px, {time.time() - t0:.1f}s)")
            if not args.no_cache:
                os.makedirs(args.cache_dir, exist_ok=True)
                np.save(cache_path, baseline.astype(np.uint8))

        t0 = time.time()
        mask = detect_streaks(data_test, rms, existing, dict(streak_cfg))
        dt = time.time() - t0

        fp_px, fp5_px, novel_px = score_mask(mask, baseline, truth)
        print(f"seed {seed}: trails={len(trails)} truth_px={int(truth.sum())} fp_px={fp_px} fp5_px={fp5_px} {dt:.1f}s")

        for index, trail in enumerate(trails):
            exact, tolerant = trail_recall(mask, trail["truth"])
            row = {key: value for key, value in trail.items() if key != "truth"}
            row.update(
                {
                    "exposure": exposure_id,
                    "hdu": args.hdu,
                    "seed": seed,
                    "trail": index,
                    "truth_px": int(np.count_nonzero(trail["truth"])),
                    "recall": round(exact, 3),
                    "recall5": round(tolerant, 3),
                    "fp_px": fp_px,
                    "fp5_px": fp5_px,
                    "novel_px": novel_px,
                    "dt_s": round(dt, 1),
                    "mode": streak_cfg.get("mode", "auto_ground"),
                    "note": note,
                }
            )
            rows.append(row)

    mean_recall, mean_recall5, mean_fp5 = summarise(rows)
    if args.summary_only:
        print(
            f"grid summary: trails={len(rows)} mean_recall={mean_recall:.3f} "
            f"mean_recall5={mean_recall5:.3f} mean_fp5_px={mean_fp5:.0f}"
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
