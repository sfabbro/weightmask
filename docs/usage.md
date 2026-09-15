# Usage

weightmask reads a detrended science FITS (single extension or MEF) and writes
weight, mask, inverse-variance, and sky products. Use them for stacking,
shape measurement, forced photometry, profile fitting, or difference imaging.
Algorithms are in [algorithms.md](algorithms.md). The supported library
surface is in [api.md](api.md).

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

`--badpix_mask` is a keep-map: `0` = bad, `1` = good. Those zeros are
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

Each output HDU is named after the CCD identifier from its header (`MAP_8341-7-5`,
`MASK_8341-7-5`, `STREAK_8341-7-5`, ...), taken from `CCDNAME`, then `CCDNAM`.
`CCD` is deliberately not used: on MegaCam it holds the detector model
(`Marconi/EEV CCD42-90`), which is the same for every HDU in the file, so it
would give all products one EXTNAME. Product HDU order still follows the input,
and files whose headers carry no per-CCD identifier fall back to the positional
name (`MAP_HDU1`).

### Flat bad-pixel mask cache

One flat HDU's bad-pixel mask costs about 20 s on a 9.8 Mpix CCD and depends only
on that flat HDU, the tile size and the `flat_masking` settings. It is therefore
computed once and reused by every exposure processed through the same flat:

```yaml
flat_masking:
  bad_mask_cache: true         # default
  bad_mask_cache_dir: null     # null -> <flat directory>/.weightmask_cache
```

The cache key covers the flat's absolute path, size, nanosecond mtime, HDU index,
shape, tile size and every `flat_masking` setting, so replacing the flat or
retuning the masking cannot reuse a stale mask. A missing, partial, corrupt or
unwritable cache entry simply falls back to the normal computation, and
`bad_mask_cache: false` disables the cache entirely.

### Streak detection cost and the Radon rescue

The streak stage dominates a CCD's time, and two settings decide most of it:

```yaml
streak_masking:
  satdet_params:
    skip_when_prescreen_confirmed: true   # default; -24.5 s/CCD
  mrt_rescue_params:
    enable: true
    bin: 1                                # 4 = ~1.6 s instead of ~20 s
    sinogram_highpass: 101                # samples; 0 disables
    sigma_rel_floor: 0.001
```

`skip_when_prescreen_confirmed` skips the full-resolution multi-scale Canny/Hough
sweep when the cheap binned-Hough prescreen has already accepted a trail in the same
HDU. It cannot suppress a field where the sweep is the stage that finds the trail,
because it only fires once the prescreen has already confirmed one; what it does
depend on is the sweep adding nothing to an HDU the prescreen has already accepted.
Measured on real MegaCam HDUs the streak mask is bit-identical either way on a clean
field, a bright-trail field, and a bright+faint field where the second trail is the
adversarial case. Set it to false to keep the sweep unconditional.

`mrt_rescue_params` is a Radon rescue for faint trails, and it is the most expensive
thing in the stage. `bin` mean-bins the projection image before the transform; the
peak it finds is confirmed at full resolution by the same strip refiner as every
other candidate, so binning costs only `bin` pixels of rho quantisation. The measured
trade-off is in the config comment: `bin: 4` is ~12x faster with equal *average*
recall but a wider per-case spread, which is why full resolution is the default.
`enable: false` removes the stage entirely and leaves detection to the Hough paths,
which is what surveys whose fields are known to carry no intermittent trails should
set.

If a field is diagnosed as slow, `streak_masking.debug: true` puts the per-stage
evidence (segments per scale, candidates, accepted lines with angle/rho/confidence,
which passes ran) into `config['_last_run']`.

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

F² sensitivity weight (same convention used in coadds). Exact Poisson plus
read noise at `F = 1`. At vignette (`F ≠ 1`) this is a sensitivity weight, not
`g² F² / (S g F + RN²)`. Canonical `weightmask.yml` then adds
`flat_rel_noise: 0.003` and `rescale_variance: true`. Omit those
keys and you get the bare formula.

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
