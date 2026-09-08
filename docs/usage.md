# Usage

WeightMask reads a detrended science FITS (single extension or MEF) and writes
weight, mask, inverse-variance, and sky products. Algorithms are in
[algorithms.md](algorithms.md). The supported library surface is in
[api.md](api.md).

## Config is required

Pass `--config path.yml`, or put a file in the current directory. Search order
if `--config` is omitted: `weightmask.yml`, `config.yml`, `.weightmask.yml`.
The wheel does not bundle a config.

The only key list is the repo-root [`weightmask.yml`](../weightmask.yml). Extra
top-level keys fail validation. Those YAML values are the 0.1 product
defaults. Missing keys warn and fall back to in-code defaults, which are not
the same file: in particular `streak_masking.enable` is false, `rescale_variance`
is false, `default_gain` is 1.0 e⁻/ADU, and `default_rdnoise` is 0 e⁻. Copy
the YAML. `WeightMapGenerator` raises `ValueError` on invalid config.

Each HDU uses one gain and one read-noise value (first present header keyword
in the configured lists). Dual-amp `GAINA`/`GAINB` are not split.

## Command

```bash
weightmask science.fits --config weightmask.yml --flat_image flat.fits \
  -o out.weight.fits --output_mask out.mask.fits
```

Run `weightmask --help` for Inputs / Outputs / Run groups. Rebuild a compact
sky mesh with the separate program `weightmask-reconstruct-sky` (also
`weightmask reconstruct-sky ...`).

## Cookbook

### MEF

Omit `--hdu` to process every 2-D image extension. Pin one extension with
`--hdu 1` or a CFITSIO-style name `science.fits[1]`. Parallel HDU workers:
`--nproc` / `--max-workers` (default `min(8, ncpu)`; `0` or `1` is sequential).

### Flat, dark, keep-map

```bash
weightmask science.fits --config weightmask.yml \
  --flat_image flat.fits \
  --dark_image dark.fits \
  --badpix_mask bpm.fits \
  -o out.weight.fits --output_mask out.mask.fits
```

`--badpix_mask` is an Elixir keep-map: `0` = bad, `1` = good. Those zeros are
OR'd into `BAD`. `--dark_image` ORs hot pixels into `BAD` only if the config
has a `dark_masking` section (the canonical YAML does). Both, and the MEF
`dead_ccd_*` veto, are CLI/MEF orchestration: `WeightMapGenerator.process`
does not take a dark or keep-map. Without a flat, `F = 1` and flat-based `BAD`
detection is skipped.

### Outputs

```bash
weightmask science.fits --config weightmask.yml \
  -o out.weight.fits \
  --output_mask out.mask.fits \
  --output_invvar out.invvar.fits \
  --output_sky out.sky.fits \
  --individual_masks
```

- Primary map (`-o`): weight or confidence, from `output_params.output_map_format`.
- `--output_mask`: combined integer quality mask.
- `--output_invvar`: sanitized inverse-variance plane.
- `--output_sky`: sky map (`output_params.sky_format: full`) or compact mesh
  (`sky_format: mesh`).
- `--output_weight_raw`: unnormalized masked inverse variance if it should
  differ from the primary map.
- `--individual_masks`: one FITS file per component (bad, sat, cr, obj, streak).

Default primary path is `<input_base>.weight.fits` if `-o` is omitted, or
`.weight.fits.fz` when `output_params.compress` is true.

### Quality bits

| Bit | Name | Zero weight? |
|---|---|---|
| 1 | `BAD` | yes |
| 2 | `SAT` | yes |
| 4 | `CR` | yes |
| 8 | `DETECTED` | no (set `output_params.mask_detected_in_weight` to zero it) |
| 16 | `STREAK` | yes |
| 32 | `INVALID_VARIANCE` | yes |

Polarity is `set_means_flagged`. See [algorithms.md](algorithms.md).

### Compact sky mesh

In `weightmask.yml` set `output_params.sky_format: mesh`. The product stores
SEP-aligned nodes plus `SKYMESH` / `MESHBW` / `MESHBH` / `SKYH` / `SKYW` cards (~2.5 KB
per CCD). Rebuild:

```bash
weightmask-reconstruct-sky sky_mesh.fits -o sky_full.fits
weightmask-reconstruct-sky sky_mesh.fits -o sky_full.fits --hdu 1
```

Compatibility dispatch: `weightmask reconstruct-sky sky_mesh.fits -o sky_full.fits`.

### Weight plane

Default `variance.method: theoretical`. The core plane is

```
ivar = g² F² / (S g + RN²)
```

Elixir-style F² coadd weight. Exact Poisson plus read noise at `F = 1`. At
vignette (`F ≠ 1`) this is a sensitivity weight, not `g² F² / (S g F + RN²)`.
Canonical `weightmask.yml` then adds `flat_rel_noise: 0.003` and
`rescale_variance: true`. Omit those keys and you get the bare formula.

### Python

```python
from astropy.io import fits
from weightmask import WeightMapGenerator
import yaml

with open("weightmask.yml") as f:
    config = yaml.safe_load(f)

sci = fits.getdata("science.fits", ext=1)
hdr = dict(fits.getheader("science.fits", ext=1))
flat = fits.getdata("flat.fits", ext=1)

out = WeightMapGenerator(config).process(sci, header=hdr, flat_data=flat)
weight, mask = out["weight_map"], out["flag_map"]
```

That call is one array. Dark frames, keep-maps, and dead-CCD veto live on the
CLI/MEF path. Exposure-global confidence rescale also does, and only when the
primary map is confidence.
