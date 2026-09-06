# WeightMask review, fixes & pilot-100 verdict (2026-09-05/06)

## Pilot-100 recommendation: `patat_photutils` (unchanged)

90/90 exposures, 39,792 rows. 74→90 exposure split is ranking-stable, so the
interim call stands. `sep_masked`+weight was briefly recommended on
weight-response/speed grounds, then withdrawn: the 0.35px SEP-vs-photutils
seeing offset dwarfs the 0.04–0.06 weight gains, and photutils≈patat agreement
is shared code, not corroboration. The independent anchor (imcore, 0.14 MAD)
sits on the patat side. See “Caveats”.

| method | seeing base→weight | agree/patat | see_ok | s/HDU |
|---|---|---|---|---|
| patat_photutils | 5.311→5.362 | ref | 99.7% | ~7.0 |
| pandey_sep | 5.779→5.665 | 0.36 | 99.7% | ~3.6 |
| sep_masked | 5.737→5.665 | 0.33 | 99.8% | ~3.1 |
| photutils_masked | 5.459→5.366 | 0.006* | 99.7% | ~8.9 |
| casutools_imcore | 5.475→5.425 | 0.14 | 96.2% | ~3.8 |
| sextractor | 6.014→6.102 | 0.51 | 98.5% | ~1.9 |

\* shares FWHM machinery with patat — circular, not corroboration.

Caveats: seeing-only metric; weight legs used old masks; the 0.35px
family offset is now the dominant archive-scale uncertainty (needs
injection-recovery or DIMM comparison to resolve, bigger than any mask effect).

## What shipped (measured on real 719016p-class data)

- Flat cache: deduped double 36-HDU precompute; hash covers tile_size.
- CR: faint second pass (worm recall 0.12→0.31, +0.004% px), niter pinned 2.
- Streaks: seed-anchored clustering + transverse-RMS gate (recall 0→0.71,
  FP 3089→21), strip angle refit + fan, binned Hough-peak stage, RANSAC
  seeded; 1500px trails recall 0.6–1.0. 800px/dashed remain below floor.
- Masks: Elixir parity IoU 0.92 ex-HDU4, misses 0.04%; dead-CCD veto exact
  on HDU4; `--dark_image` leg end-to-end (cfhtcast `--dark-map` wired).
- Weights: flat-noise term (−3.5%/−4.9% edge), exposure-global confidence
  norm; background pulls old≈new (rsig 0.983 vs 0.990, n=52M) — gains live
  in masks, not values, as designed.
- Speed: 199s→147s per HDU (−26%) plus ~6–15 min/exposure dedup saving.
- Suite: 148 passed, 1 pre-existing skip.

## Open gaps

Short/dashed-trail floor (harnesses kept: `benchmarks/streak_inject.py`,
`mine_trail_candidates.py`); crosstalk (blocked: no coefficients);
Space-Track account + TLE harness for real ground truth (proposal in
session notes); 4 missing `wm/` masks (workers dead, needs batch resume).
