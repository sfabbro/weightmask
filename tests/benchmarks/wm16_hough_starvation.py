#!/usr/bin/env python3
"""Measure what actually reaches the synthetic_v2 binned-Hough accumulator.

WM-16 records the aggregate failure as ``houghpeak_params.min_votes = 100``
sitting above the accumulator ceiling at 384 px, so the stage is structurally
dead. The stage logs are consistent with that (``0 peaks``), but they do not
separate it from causes *upstream* of the accumulator: a floor cannot bind on a
binary image that is already empty.

This instrument recovers the exact ``(data_sub, bkg_rms_map, existing_mask)``
the streak stage receives, re-derives the stage's own binary images, and
attributes the starvation of the ground-truth trail:

``premasked``
    Share of the trail's own pixels already inside ``existing_mask`` -- the
    union of the saturation, cosmic-ray and object masks -- before the streak
    stage runs. Those pixels are zeroed in the binned SNR image, so they can
    never vote.
``lit_masked`` / ``lit_unmasked``
    Binned cells that lie on the trail's line and clear ``thresh_sig``, split by
    whether the pre-mask zeroed them. This is the vote budget the trail
    actually gets.
``on_trail_cells``
    Binned cells the trail's line crosses. An upper bound on the trail's votes
    even if every cell were lit and unmasked, and the quantity to compare
    against ``min_votes``.

Read-only: it patches ``_detect_streaks_houghpeaks`` with a spy that computes
statistics and then calls the original, so the pipeline's result is unchanged.

    pixi run python tests/benchmarks/wm16_hough_starvation.py
    pixi run python tests/benchmarks/wm16_hough_starvation.py --cases synthetic_complex
"""

from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402

from tests.benchmarks.run import _synthetic_v2_cases  # noqa: E402
from tests.simulate_and_test import create_simulated_data, run_masking_test  # noqa: E402
from weightmask import streaks as S  # noqa: E402

GT: dict = {}
RECORD: list[dict] = []


def _line(trail):
    """Least-squares x(y) fit through the trail pixels."""
    ys, xs = np.nonzero(trail)
    return np.polyfit(ys.astype(float), xs.astype(float), 1)


def _spy(data_sub, bkg_rms_map, existing_mask, config, predictions=None):
    cfg = config.get("houghpeak_params", {})
    bfac = max(1, int(cfg.get("bin", 4)))
    thresh_sig = float(cfg.get("thresh_sig", 2.5))
    trail = GT["trail"]
    masked = np.zeros_like(existing_mask, dtype=bool) if existing_mask is None else existing_mask
    h, w = data_sub.shape
    bh, bw = h // bfac, w // bfac
    binned = S._bin_array(data_sub, bfac)
    brms = S._bin_array(bkg_rms_map, bfac)
    snr = binned / np.maximum(brms / bfac, 1e-6)
    bmasked = masked[: bh * bfac, : bw * bfac].reshape(bh, bfac, bw, bfac).any(axis=(1, 3))
    # Binned cell centres in full-res pixels, then distance to the trail's line.
    yy, xx = np.mgrid[0:bh, 0:bw]
    slope, intercept = _line(trail)
    yc = yy * bfac + (bfac - 1) / 2.0
    xc = xx * bfac + (bfac - 1) / 2.0
    on_trail = np.abs(xc - (slope * yc + intercept)) / np.sqrt(1.0 + slope**2) <= 1.5
    row = {
        "trail_px": int(np.count_nonzero(trail)),
        "trail_premasked": int(np.count_nonzero(trail & masked)),
        "premask_frac": float(np.mean(bmasked)),
        "on_trail_cells": int(np.count_nonzero(on_trail)),
        "min_votes": int(cfg.get("min_votes", 100)),
        "bin": bfac,
        "thresh_sig": thresh_sig,
    }
    for label, simg in (("strict", snr / bfac), ("nominal", snr)):
        lit = np.isfinite(simg) & (simg > thresh_sig)
        row[f"{label}_lit_unmasked"] = int(np.count_nonzero(lit & on_trail & ~bmasked))
        row[f"{label}_lit_masked"] = int(np.count_nonzero(lit & on_trail & bmasked))
    RECORD.append(row)
    return _real(data_sub, bkg_rms_map, existing_mask, config, predictions)


_real = S._detect_streaks_houghpeaks


def _f1(pred, truth):
    inter = int(np.count_nonzero(pred & truth))
    precision = inter / max(1, int(np.count_nonzero(pred)))
    recall = inter / max(1, int(np.count_nonzero(truth)))
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", nargs="*", help="restrict to these synthetic_v2 case names")
    parser.add_argument("--config", default=os.path.join(ROOT, "weightmask.yml"))
    args = parser.parse_args(argv)

    S._detect_streaks_houghpeaks = _spy
    try:
        header = (
            f"{'case':30s} {'trailPx':>7s} {'premasked':>10s} {'frame%':>7s} {'cells':>6s} "
            f"{'strictUM':>9s} {'strictM':>8s} {'nominalUM':>10s} {'nominalM':>9s} {'minVotes':>9s} {'F1':>6s}"
        )
        print(header, flush=True)
        print("-" * len(header), flush=True)
        for case in _synthetic_v2_cases():
            if args.cases and case["name"] not in args.cases:
                continue
            GT.clear()
            RECORD.clear()
            _, _, gt = create_simulated_data(
                size=case["size"],
                noise_level=case["noise"],
                num_stars=case["stars"],
                streak_flux=case["streak"],
                regime_type=case["regime_type"],
                seed=case["seed"],
            )
            GT["trail"] = gt["streak"]
            _metrics, products = run_masking_test(
                args.config, SimpleNamespace(**case), save_fits=False, return_products=True
            )
            truth = products["ground_truth"]["streak"]
            if not np.array_equal(truth, gt["streak"]):
                print(f"{case['name']}: generator is not reproducible from case parameters", file=sys.stderr)
                return 2
            row = RECORD[-1] if RECORD else {}
            pct = 100.0 * row.get("trail_premasked", 0) / max(1, row.get("trail_px", 1))
            print(
                f"{case['name']:30s} {row.get('trail_px', -1):7d} {pct:9.1f}% "
                f"{row.get('premask_frac', float('nan')) * 100:6.1f}% {row.get('on_trail_cells', -1):6d} "
                f"{row.get('strict_lit_unmasked', -1):9d} {row.get('strict_lit_masked', -1):8d} "
                f"{row.get('nominal_lit_unmasked', -1):10d} {row.get('nominal_lit_masked', -1):9d} "
                f"{row.get('min_votes', -1):9d} {_f1(products['masks']['streaks'], truth):6.3f}",
                flush=True,
            )
    finally:
        S._detect_streaks_houghpeaks = _real
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
