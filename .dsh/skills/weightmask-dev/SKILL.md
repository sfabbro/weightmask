---
name: weightmask-dev
description: Develop weightmask (weight/confidence maps and defect masks) — YAML config surface, quality-bit contract, streak detection, synthetic benchmarks
metadata:
  project: weightmask
  stack: python, numpy, pixi
---

# weightmask development

Builds weight maps, confidence maps, and defect masks for astronomical FITS
images. Those products are for coaddition and for single-exposure work (shape
measurement, forced photometry, profile fitting, difference imaging). See `docs/`.

## Contract

- Canonical YAML config surface: `flat_masking`, `sep_background`,
  `sep_objects`, `streak_masking`, `variance`.
- NumPy array/header contract with named quality bits, explicit
  `set_means_flagged` mask polarity, and versioned provenance metadata.
- Primary outputs: weight or confidence map, combined bitmask, inverse-variance
  map, sky map, per-contaminant masks (`--individual_masks`).

## Detection

- Default streak detector is `auto_ground` (percentile rescaling + smoothing,
  compact-source suppression, multi-scale Canny + Hough/KHT segment extraction,
  trail-aligned strip refinement, MRT-like Radon rescue, optional sparse RANSAC).
- Frangi comparison lives in `benchmarks/frangi_legacy.py` — not in the package.
- Bad pixels from flats, saturation/bleed trails, cosmic rays, objects, and
  linear streaks.

## Benchmarks (statistics-principled)

- `pixi run benchmark-synthetic` — synthetic suite vs baselines
  (`--with-baselines`): inject known defects and verify recovery statistics.
- `pixi run benchmark-megacam` — Megacam real data
- `pixi run benchmark-acs` — ACS comparison
- `pixi run test`, `pixi run lint`, `pixi run format`

## CLI

`weightmask science.fits --config weightmask.yml`; MEF inputs supported.
