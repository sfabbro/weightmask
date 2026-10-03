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
in the configured lists) unless both `GAINA` and `GAINB` exist and a section
keyword splits the HDU. Then the variance plane uses that gain map.

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
`--hdu 1` or a CFITSIO-style name `science.fits[1]` (the spec must be trailing;
`--hdu` wins if both are given, and says so). Parallel HDU workers:
`--nproc` / `--max-workers` (default `min(8, ncpu)`; `0` or `1` is sequential).

The flat, dark and keep-map are matched to each science HDU **by index**, so
`--flat_image`, `--dark_image` and `--badpix_mask` reject an `[N]` spec rather
than ignoring it. A flat or dark MEF shorter than the science MEF is an error
(a unit flat would silently produce unflat-fielded weights) or, for the dark, a
warning, since only hot-pixel rejection is lost. If the bad-mask cache
directory is not writable -- a read-only VOSpace flat, typically -- the run
still succeeds but says so, because the ~20 s/HDU median filter is then
recomputed every time.

### Persistence priors

`--exposures` builds a per-CCD column/row prior from other exposures of the
same detector. It is a **prior on the streak detector only**: those pixels are
withheld from streak detection so detector-fixed structure is not reported as a
sky trail. They are *not* bad pixels -- they keep their normal weight and carry
no quality bit, because the flux really was recorded there.

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

On a complete run, data extension *k* is the product for science HDU *k*, and
`EXTNAME` records the CCD identifier (falling back to the HDU index). If any HDU
fails, the run **exits non-zero** and says how many were written: the products
then hold fewer data extensions than there are science HDUs, so extension
position no longer lines up. Use `EXTNAME` to identify a product's source HDU
rather than its position.
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

### Streak detection cost

The streak stage dominates a CCD's time. What is left is two binned prescreens
and a conditional RANSAC pass:

```yaml
streak_masking:
  houghpeak_params:
    enable: true
    bin: 4                               # spatial binning before the peak search
    thresh_sig: 2.5
  contour_params:
    enable: true
  mask_params:
    max_premasked_fraction: 0.25          # drop a component the pipeline already flagged
    max_component_sigma: 20.0             # drop a component far brighter than a trail; null disables
  enable_sparse_ransac: true
```

`profile_accept` keeps only components that concentrate about a fitted line
narrower than `mask_params.max_support_width`. `max_premasked_fraction` drops a
streak component lying mostly inside the mask the earlier stages already
produced, because that is not a new finding: on real MegaCam amps a saturated
star's bleed measures 0.45 and a genuine satellite trail 0.02.
### The brightness veto is optional, and uncalibrated

`max_component_sigma` drops a component whose 90th percentile exceeds the given
number of background RMS. **Set it to `null` to disable it.** Measured p90 on
MegaCam, in sigma:

| feature | p90 |
|---|---|
| satellite trail, 996195p HDU 35 / 36 | 3.0 / 3.2 |
| saturated-star bleed | 1138 / 1345 |
| bright star arm | 155 |
| hot column group | ~2000 |

The threshold is a **6x margin below a sample of two trails**. That is a
deliberate provisional setting, not a measured constant: a bright satellite
constellation would be deleted by this rule and nothing in the local corpus
would ever reveal it. Raise it, or null it, if that trade is the wrong way round.

What disabling it costs, measured (veto on -> off, pixels masked):

| feature | veto on | veto off |
|---|---|---|
| real satellite trails | 12,001 / 11,034 | 12,001 / 11,034 -- unaffected |
| saturated-star bleed | 0 | 0 -- the pre-masked veto still suppresses it |
| hot column group | 0 | **10,734** -- returns |
| bright star arm | 0 | **4,385** -- returns |

The two that return sit below `max_premasked_fraction` (0.24 and 0.15 against a
0.25 threshold), so the pre-masked veto cannot catch them either.

The column can instead be caught without reference to brightness, because it is
a static defect: it persists at **0.90** across epochs of the same field against
**0.02** for a real trail. The CLI's `--persistence` uses exactly that. The star
arm persists at only 0.41 and is not reliably caught that way, so on a
single-exposure run the veto is currently the only thing suppressing it.

`tests/test_brightness_veto.py` pins that both real trails survive at every
setting including `null`, and that the veto still suppresses both artefacts.

There used to be a third stage, an angle-binned Radon rescue
(`mrt_rescue_params`), intended as the sensitive one -- the thing that finds
what the cheap prescreen misses. It is gone. Measured on 83 real amps it
accepted on three, all three false positives, while `houghpeaks` explained all
23,035 px of real detections; on injected trails at 4, 6, 8 and 12 sigma, two
lengths and two seeds, it moved `recall_line` by +0.000 in eight of eight cells.
It cost 121.7 s/amp of a 125.7 s/amp stage, and removing it took the stage to
4.0 s/amp with both real trails byte-identical. `pixi run streak-sweep` and
`pixi run streak-recall-floor` re-derive those numbers; anyone proposing a replacement
sensitive stage has to re-run them rather than argue from this file.

`pixi run streak-recall-floor` measures where the stage stops finding injected
trails, across {solid, dashed} trails at 4-12 sigma, two lengths and two seeds,
on the production input path. `--disable {contours,houghpeaks,ransac}` repeats the
same grid with one stage switched off, which is how a stage's cost is weighed
against what only it finds. Both remaining prescreens are dormant on the local
corpus -- contours and RANSAC accept on 0 of 224 real amps each -- yet each is the
sole finder in the cells it helps and costs recall in none, which is why neither
was removed.

`pixi run exposure-time` answers the question at the unit a survey pays in --
the whole exposure, not one chip. On a 36-chip MegaPrime MEF (996195p, 353 Mpix,
one core, `--nproc 1`):

| | wall | per chip |
|---|---|---|
| `streak_masking.enable: false` | **389.9 s** | 10.83 s |
| `streak_masking.enable: true` | **420.4 s** | 11.68 s |
| difference | **+30.4 s** | +0.85 s |

Streak detection is **7% of a full exposure**, not the majority of it. Cosmics is
67%. The remaining streaks-attributable cost is mostly reading the 350 MB flat,
which happens either way.

`--scaling 1 2 4 8` sweeps the worker count with **streaks on**; it has no off-arm.
On this 8-core M2 (4 performance + 4 efficiency):

| `--nproc` | wall | speedup | efficiency |
|---|---|---|---|
| 1 | 430.1 s | 1.00x | 100% |
| 2 | 245.7 s | 1.75x | 88% |
| 4 | 168.9 s | 2.55x | 64% |
| 8 | 139.6 s | 3.08x | 39% |

These absolute times sit ~2% above the 420.4 s single-core figure above for the
same streaks-on configuration. That is run-to-run variance across separate
invocations, not a difference in what was measured; the speedup ratios are taken
within one session and are the trustworthy part of this table.

The efficiency column falls because **the machine saturates, not because the
pipeline serialises**. Per-chip wall time grows with the thread count -- median
11.06 s at `--nproc 1`, 15.64 s at 4, 27.14 s at 8 -- so eight threads on four
performance cores each run ~2.45x slower than one thread alone. The two
candidate serial sections were measured and are negligible: the flat-median
prologue that runs before any worker starts is 2.4 s, and writing every product
serially is 1.0 s, together 0.8% of the single-core run.

An Amdahl fit to the three parallel points implies a 101 s serial floor, which
both direct measurements contradict. The fit is wrong here and is not quoted: with
four threads sharing four cores, a three-point fit cannot separate "serial code"
from "machine saturation", and the per-chip timings can.

Two things that make the comparison number trustworthy and would otherwise have made it
wrong. Both arms are warmed up and the warmup discarded: the first run of either
arm pays for importing skimage/scipy, populating the on-disk flat-bad-pixel cache,
and faulting the flat into the page cache. Measured in sequence without a warmup,
the arm that ran second appeared 2.6x *faster*, which is an ordering artifact and
not a property of streak detection. And the reported streak pixel count is
checked per arm, so a run where the toggle silently failed to take cannot be
mistaken for a measurement.

The synthetic benchmark suite has a known, pre-existing gate failure
(`Synthetic-v2 average streak F1 0.159 < 0.200`) that is unchanged by any of
the above -- the per-case F1 values are identical with and without the rescue.

If a field is diagnosed as slow, `streak_masking.debug: true` puts the per-stage
evidence (candidates, accepted lines with angle/rho/confidence, which passes
ran) into `config['_last_run']`.

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

Also spelled `weightmask reconstruct-sky sky_mesh.fits -o sky_full.fits`, which
dispatches to the same program. That was the original interface; the
`weightmask-reconstruct-sky` console script was added later, and is the one to
use.

### Weight plane

Default `variance.method: theoretical`. The core plane is

```
ivar = g² F² / (S g F + RN²)
```

Flat-fielded Poisson weight. Exact Poisson plus read noise at `F = 1`.
Canonical `weightmask.yml` sets `variance.flat_fielded_poisson: true` because
a spatially varying flat moved a weighted aperture by more than the fixture
read-noise floor. Omitting the key keeps the older `S g` denominator.
Canonical `weightmask.yml` then adds
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
