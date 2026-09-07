# WeightMask

WeightMask takes a detrended FITS or MEF science image and writes a coadd
weight, quality mask, inverse-variance map, and sky. It is built for stacking.

## Install

Pixi (recommended):

```bash
git clone https://github.com/astroai/weightmask.git
cd weightmask
pixi install
pixi run weightmask science.fits --config weightmask.yml --flat_image flat.fits \
  -o out.weight.fits --output_mask out.mask.fits
```

Or `pip install -e .` (Python 3.10+), then the same `weightmask ...` command
without `pixi run`. Details: [docs/installation.md](docs/installation.md).

## Command

```bash
weightmask science.fits --config weightmask.yml --flat_image flat.fits \
  -o out.weight.fits --output_mask out.mask.fits
```

A YAML config is required. Copy [`weightmask.yml`](weightmask.yml) into the
working directory; it is not bundled in the wheel.

## Products

| Product | What |
|---|---|
| Weight | Masked inverse variance (primary map by default) |
| Mask | Integer quality bits |
| Inverse variance | Same plane after sanitizing non-finite / non-positive values |
| Sky | Background map, or a compact mesh rebuildable with `weightmask-reconstruct-sky` |

Quality bits (`set_means_flagged`: a set bit means the condition is present):

| Bit | Name | Zero weight? |
|---|---|---|
| 1 | `BAD` | yes |
| 2 | `SAT` | yes |
| 4 | `CR` | yes |
| 8 | `DETECTED` | no (unless `mask_detected_in_weight`) |
| 16 | `STREAK` | yes |
| 32 | `INVALID_VARIANCE` | yes |

The default theoretical plane is an Elixir-style F² coadd weight,
`ivar = g² F² / (S g + RN²)`. At `F = 1` this is Poisson plus read noise; at
`F ≠ 1` it is a sensitivity weight, not a flat-fielded Poisson identity.

## Docs

- [Installation](docs/installation.md)
- [Usage](docs/usage.md)
- [Algorithms](docs/algorithms.md)
- [API](docs/api.md)
- [CHANGELOG](CHANGELOG.md)

## License

MIT
