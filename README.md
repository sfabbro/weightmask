# WeightMask

WeightMask builds weight maps, confidence maps, and defect masks for astronomical FITS images. It is aimed at stacking and coaddition workflows where you need a practical mask plane, a usable inverse-variance estimate, and separate outputs for key contaminants.

## What It Does

- Builds weight maps, confidence maps, inverse-variance maps, and sky maps.
- Detects bad pixels from flats, saturation and bleed trails, cosmic rays, astronomical objects, and linear streaks.
- Supports MEF inputs through the CLI.
- Uses a canonical YAML config surface centered on `flat_masking`, `sep_background`, `sep_objects`, `streak_masking`, and `variance`.

## Streak Detection

The default streak detector is `auto_ground`:

- percentile rescaling and Gaussian smoothing
- compact-source suppression
- multi-scale Canny and Hough/KHT-style segment extraction
- trail-aligned strip refinement with adaptive width growth
- MRT-like Radon rescue when the primary detector is low-confidence
- optional sparse RANSAC recovery for intermittent trails

`frangi_legacy` remains available only as a comparison path.

## Quick Start

```bash
pip install -e .
weightmask science.fits --config weightmask.yml
```

Useful outputs:

- primary map: weight or confidence
- combined bitmask
- inverse-variance map
- sky map
- per-contaminant masks with `--individual_masks`

## Repo Layout

- `weightmask/`: library code
- `tests/`: unit tests and synthetic benchmark harness
- `docs/`: installation, usage, and API notes
- `examples/`: runnable synthetic and robustness examples
- `weightmask.yml`: canonical example configuration

Generated products from the synthetic examples and benchmark harness belong under `test_outputs/`; they are not source files.

## Documentation

- Usage: `docs/usage.md`
- Installation: `docs/installation.md`
- API: `docs/api.md`

## Benchmarks

The repo now includes a benchmark runner with synthetic-v2 and manifest-driven real-data suites:

```bash
uv run python -m tests.benchmarks.run --suite synthetic_v2 --with-baselines
uv run python -m tests.benchmarks.run --suite megacam_real
uv run python -m tests.benchmarks.run --suite acs_compare
```

## License

GPL-3.0
