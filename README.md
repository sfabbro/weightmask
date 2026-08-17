# WeightMask

WeightMask builds weight maps, confidence maps, and defect masks for astronomical FITS images. It is aimed at stacking and coaddition workflows where you need a practical mask plane, a usable inverse-variance estimate, and separate outputs for key contaminants.

## What It Does

- Builds weight maps, confidence maps, inverse-variance maps, and sky maps.
- Detects bad pixels from flats, saturation and bleed trails, cosmic rays, astronomical objects, and linear streaks.
- Supports MEF inputs through the CLI.
- Uses a canonical YAML config surface centered on `flat_masking`, `sep_background`, `sep_objects`, `streak_masking`, and `variance`.
- Publishes a NumPy array/header contract with named quality bits, explicit `set_means_flagged` mask polarity, and versioned provenance metadata.

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
- `docs/`: installation, usage, API notes, and research plans
- `examples/`: runnable synthetic and robustness examples
- `weightmask.yml`: canonical example configuration

Generated products from the synthetic examples and benchmark harness belong under `test_outputs/`; they are not source files.

## Array/Header Contract

`weightmask.contract` is a small, transport-neutral interoperability layer.
`build_weight_product()` returns uint32 quality flags, non-negative inverse
variance and weight arrays, normalized `[0, 1]` confidence, and metadata for
each artifact. Set quality bits mean the named condition is present; detected
objects remain informative by default, while defects and invalid variance have
zero weight. `TorchfitsArrayHeaderIO` is an optional adapter that uses only the
public `torchfits.read` and `torchfits.write` APIs when torchfits is installed.
It does not add a torchfits dependency or generate learned masks or weights.

## Documentation

- Usage: `docs/usage.md`
- Installation: `docs/installation.md`
- API: `docs/api.md`
- Next-generation research and implementation plan: `docs/nextgen_weightmask.md`

## Benchmarks

The repo now includes a benchmark runner with synthetic-v2 and manifest-driven real-data suites:

```bash
pixi run benchmark-synthetic
pixi run benchmark-megacam
pixi run benchmark-acs
```

The real-data tasks are acceptance gates: they exit nonzero when a science
exposure, required manual trail label, or matching ACS/WFC ERR/DQ plane is
missing or invalid. Each manual label is a finite binary full-frame FITS mask;
its byte-exact SHA-256 and the SHA-256 of its source exposure are pinned in the
suite manifest. A null label hash intentionally keeps the gate closed until the
reviewed external artifact exists. Label paths and the polarity/inverse-variance,
overmasking, flux-bias, and noise-calibration thresholds are fixed in the
suite manifests under `tests/benchmarks/manifests/`.

## License

MIT
