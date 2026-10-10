#!/usr/bin/env python3
"""CR completeness / false-positive curves by injection into a real HDU.

Injects single-pixel CRs and multi-pixel worms at known positions into a real
image, runs ``detect_cosmic_rays`` under several configs, and reports completeness
per morphology plus novel off-truth pixels and runtime. Each (variant, seed) is one
TSV row, so two runs can be diffed instead of eyeballed.

The interesting variant is ``single-pass``: the production two-pass arrangement
runs full L.A.Cosmic twice (a strict base pass plus a loose niter=1 "faint" pass)
when a single looser pass followed by the same morphology gate might reach the
same completeness for less than half the cost. This benchmark is what decides it.

Usage:
    pixi run python benchmarks/cr_faint_curves.py <exposure.fits.fz> [--hdu N]
        [--seeds 1] [--out results.tsv] [--summary-only] [--config PATH]
    pixi run python benchmarks/cr_faint_curves.py --normalized-residuals [--config PATH]
"""

import argparse
import copy
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "weightmask.yml"


def load_config(path=DEFAULT_CONFIG):
    return yaml.safe_load(Path(path).read_text())


def load_shipped_sigclip(config_path=DEFAULT_CONFIG):
    return float(load_config(config_path)["cosmic_ray"]["sigclip"])


def inject(sci, sky, rms, rng, n_single=150, n_worm=60, gain=1.0, invalid_mask=None, return_rejected=False):
    truth_single = np.zeros(sci.shape, dtype=bool)
    truth_worm = np.zeros(sci.shape, dtype=bool)
    rejected = []
    h, w = sci.shape
    out = sci.copy()
    gain = float(gain)
    if not np.isfinite(gain) or gain <= 0:
        raise ValueError("gain must be finite and positive")
    valid = np.isfinite(sci) & np.isfinite(sky) & np.isfinite(rms) & (rms > 0)
    if invalid_mask is not None:
        valid &= ~np.asarray(invalid_mask, dtype=bool)
    def deposit(amplitude, sigma):
        electrons = max(float(amplitude) * float(sigma) * gain, 0.0)
        return float(rng.poisson(electrons)) / gain
    interior = valid.copy()
    interior[:10] = interior[-10:] = False
    interior[:, :10] = interior[:, -10:] = False
    locations = np.flatnonzero(interior)
    if not locations.size:
        raise ValueError("no finite positive-RMS interior pixels for CR injection")
    occupied = np.zeros(sci.shape, dtype=bool)
    sampled_origins = np.zeros(sci.shape, dtype=bool)
    for index in range(n_single):
        available = locations[~sampled_origins.ravel()[locations]]
        if not available.size:
            rejected.extend(
                {"kind": "single", "reason": "placement_failed"} for _ in range(n_single - index)
            )
            break
        y, x = np.unravel_index(rng.choice(available), sci.shape)
        sampled_origins[y, x] = True
        out[y, x] += deposit(rng.uniform(4, 8), rms[y, x])
        truth_single[y, x] = True
        occupied[y, x] = True
    for _ in range(n_worm):
        placed = False
        rejected_reason = None
        for _attempt in range(100):
            available = locations[~sampled_origins.ravel()[locations]]
            if not available.size:
                rejected_reason = "placement_failed"
                break
            y, x = np.unravel_index(rng.choice(available), sci.shape)
            sampled_origins[y, x] = True
            angle = rng.uniform(0, np.pi)
            length = int(rng.integers(3, 9))
            coords = []
            seen = set()
            blocked = False
            for t in range(length):
                yy, xx = int(y + t * np.sin(angle)), int(x + t * np.cos(angle))
                if not (0 <= yy < h and 0 <= xx < w):
                    continue
                if (yy, xx) in seen:
                    continue
                seen.add((yy, xx))
                if not valid[yy, xx] or occupied[yy, xx]:
                    blocked = True
                    break
                coords.append((yy, xx))
            if blocked or not coords:
                rejected_reason = "truth_overlap" if blocked else "placement_failed"
                continue
            amp = rng.uniform(4, 8)
            for yy, xx in coords:
                out[yy, xx] += deposit(amp, rms[yy, xx])
                truth_worm[yy, xx] = True
                occupied[yy, xx] = True
            placed = True
            break
        if not placed:
            rejected.append({"kind": "worm", "reason": rejected_reason or "placement_failed"})
    if return_rejected:
        return out, truth_single, truth_worm, rejected
    return out, truth_single, truth_worm


def score(flag, truth_single, truth_worm, baseline=None):
    from scipy.ndimage import binary_dilation

    near_any = binary_dilation(truth_single | truth_worm, iterations=1)
    tp_single = np.count_nonzero(flag & truth_single)
    tp_worm = np.count_nonzero(flag & truth_worm)
    novel = flag if baseline is None else flag & ~baseline
    fp = np.count_nonzero(novel & ~near_any)
    return (
        tp_single / max(1, np.count_nonzero(truth_single)),
        tp_worm / max(1, np.count_nonzero(truth_worm)),
        int(fp),
        int(np.count_nonzero(flag)),
    )


def scoreable_truth(truth_single, truth_worm, detector_input, baseline=None):
    """Partition truth against invalid/upstream-excluded pixels."""
    data, existing = detector_input
    rejected = ~np.isfinite(data) | np.asarray(existing, dtype=bool)
    if baseline is not None:
        rejected |= np.asarray(baseline, dtype=bool)
    return (
        truth_single & ~rejected,
        truth_worm & ~rejected,
        truth_single & rejected,
        truth_worm & rejected,
    )


def score_variants(flags, truth_single, truth_worm, detector_input, baseline):
    """Score variant masks against one pre-variant truth denominator."""
    score_single, score_worm, rejected_single, rejected_worm = scoreable_truth(
        truth_single, truth_worm, detector_input, baseline
    )
    rows = {}
    for name, flag in flags.items():
        single, worm, fp, total = score(flag, score_single, score_worm, baseline)
        rows[name] = {
            "single": single,
            "worm": worm,
            "fp": fp,
            "total": total,
            "single_scoreable": int(score_single.sum()),
            "single_rejected": int(rejected_single.sum()),
            "worm_scoreable": int(score_worm.sum()),
            "worm_rejected": int(rejected_worm.sum()),
        }
    return rows


def normalized_residual_sigclip_curve(
    *,
    fixed_sigclip,
    seed=0,
    n_noise=1_000_000,
    n_cosmic=10_000,
    rms_adu_values=(1.0, 10.0, 100.0),
):
    """Compare fixed sigma clipping with the retired raw-ADU RMS rule.

    The normalized noise and cosmic residuals are held exactly fixed while only
    their ADU representation changes. A valid significance rule must therefore
    return identical thresholds and metrics at every scale.
    """
    rng = np.random.default_rng(seed)
    noise_sigma = rng.normal(0.0, 1.0, int(n_noise))
    cosmic_sigma = rng.uniform(3.0, 12.0, int(n_cosmic))
    rows = []
    for rms_adu in rms_adu_values:
        rms_adu = float(rms_adu)
        normalized_noise = (noise_sigma * rms_adu) / rms_adu
        normalized_cosmic = (cosmic_sigma * rms_adu) / rms_adu
        thresholds = {
            "fixed_dimensionless": float(fixed_sigclip),
            "legacy_adu_rms": float(np.clip(4.5 * (10.0 / (rms_adu + 1.0)), 3.0, 8.0)),
        }
        for variant, sigclip in thresholds.items():
            rows.append(
                {
                    "variant": variant,
                    "rms_adu": rms_adu,
                    "sigclip": sigclip,
                    "recall": float(np.mean(normalized_cosmic >= sigclip)),
                    "false_positive_rate": float(np.mean(normalized_noise >= sigclip)),
                    "n_noise": int(n_noise),
                    "n_cosmic": int(n_cosmic),
                }
            )
    return rows


NORMALIZED_COLUMNS = ["variant", "rms_adu", "sigclip", "recall", "false_positive_rate", "n_noise", "n_cosmic"]


def format_normalized_tsv(rows):
    lines = ["\t".join(NORMALIZED_COLUMNS)]
    for row in rows:
        lines.append("\t".join(str(row[column]) for column in NORMALIZED_COLUMNS))
    return "\n".join(lines)


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


TSV_COLUMNS = [
    "exposure",
    "hdu",
    "seed",
    "variant",
    "single_recall",
    "worm_recall",
    "single_scoreable",
    "single_rejected",
    "worm_scoreable",
    "worm_rejected",
    "placement_rejected",
    "novel_fp_px",
    "total_px",
    "dt_s",
    "metric_revision",
]


def format_tsv(rows):
    lines = ["\t".join(TSV_COLUMNS)]
    for row in rows:
        lines.append("\t".join(str(row.get(column, "")) for column in TSV_COLUMNS))
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exposure", nargs="?")
    parser.add_argument("--hdu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--out", default=None)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--summary-only", action="store_true")
    parser.add_argument("--normalized-residuals", action="store_true")
    args = parser.parse_args(argv)
    if args.seeds < 1:
        parser.error("--seeds must be positive")
    if args.normalized_residuals:
        fixed_sigclip = load_shipped_sigclip(args.config)
        text = format_normalized_tsv(normalized_residual_sigclip_curve(seed=args.seed, fixed_sigclip=fixed_sigclip))
        if args.out:
            with open(args.out, "w") as handle:
                handle.write(text + "\n")
        print(text)
        return 0
    if args.exposure is None:
        parser.error("exposure is required unless --normalized-residuals is used")

    import fitsio

    base_cfg = load_config(args.config)
    cosmic_base = base_cfg["cosmic_ray"]
    exposure_id = os.path.basename(args.exposure).split(".")[0]

    with fitsio.FITS(args.exposure, "r") as handle:
        sci = np.ascontiguousarray(handle[args.hdu].read().astype(np.float32))
        header = handle[args.hdu].read_header()
    from production_inputs import capture_detector_calls, production_gain_read_noise

    gain, _read_noise = production_gain_read_noise(header, base_cfg)
    clean_calls = capture_detector_calls(sci, header, base_cfg, stop_after="cosmics")
    clean_cosmic = clean_calls["cosmics"][-1]
    sky = clean_cosmic["kwargs"]["sky_map"]
    rms = clean_cosmic["kwargs"]["bkg_rms_map"]
    assert sky is not None and rms is not None

    rows = []
    common_baseline = clean_cosmic["result"]
    for seed in range(args.seed, args.seed + args.seeds):
        rng = np.random.default_rng(seed)
        injected, truth_single, truth_worm, rejected = inject(
            sci, sky, rms, rng, gain=gain, invalid_mask=~np.isfinite(sci), return_rejected=True
        )
        print(f"seed {seed}: singles={int(truth_single.sum())} worm_px={int(truth_worm.sum())}")
        common_injected_calls = capture_detector_calls(injected, header, base_cfg, stop_after="cosmics")
        common_injected = common_injected_calls["cosmics"][-1]
        flags = {}
        timings = {}
        descriptions = {}
        for name, (override, description) in variants(cosmic_base).items():
            cfg = {**cosmic_base, **override}
            if isinstance(override.get("faint_cr"), dict):
                cfg["faint_cr"] = {**cosmic_base["faint_cr"], **override["faint_cr"]}
            full_cfg = copy.deepcopy(base_cfg)
            full_cfg["cosmic_ray"] = cfg
            t0 = time.time()
            injected_calls = capture_detector_calls(injected, header, full_cfg, stop_after="cosmics")
            injected_cosmic = injected_calls["cosmics"][-1]
            flags[name] = injected_cosmic["result"]
            timings[name] = time.time() - t0
            descriptions[name] = description
        denominator = score_variants(
            flags,
            truth_single,
            truth_worm,
            (common_injected["args"][0], common_injected["args"][1]),
            common_baseline,
        )
        for name, metrics in denominator.items():
            single = metrics["single"]
            worm = metrics["worm"]
            fp = metrics["fp"]
            total = metrics["total"]
            dt = timings[name]
            print(f"  {name:18s} single={single:.3f} worm={worm:.3f} novel_fp_px={fp:6d} total={total:6d} {dt:5.1f}s")
            rows.append(
                {
                    "exposure": exposure_id,
                    "hdu": args.hdu,
                    "seed": seed,
                    "variant": name,
                    "single_recall": round(single, 3),
                    "worm_recall": round(worm, 3),
                    "single_scoreable": metrics["single_scoreable"],
                    "single_rejected": metrics["single_rejected"],
                    "worm_scoreable": metrics["worm_scoreable"],
                    "worm_rejected": metrics["worm_rejected"],
                    "placement_rejected": len(rejected),
                    "novel_fp_px": fp,
                    "total_px": total,
                    "dt_s": round(dt, 1),
                    "_description": descriptions[name],
                    "metric_revision": "additive-finite-injection-v3",
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
                f"novel_fp_px={np.mean([r['novel_fp_px'] for r in subset]):.0f} "
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
