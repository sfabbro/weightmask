# Usage

weightmask reads a detrended science FITS (single extension or MEF) and writes
weight, mask, inverse-variance, and sky products. Use them for stacking,
shape measurement, forced photometry, profile fitting, or difference imaging.
Algorithms are in [algorithms.md](algorithms.md). The supported library
surface is in [api.md](api.md).

## Config is required

Pass `--config path.yml`, or put `weightmask.yml` in the current directory.
The wheel does not bundle a config.

The only key list is the repo-root [`weightmask.yml`](../weightmask.yml). Extra
top-level keys fail validation. Those YAML values are the product
defaults. Missing keys warn and fall back to in-code defaults, which are not
the same file: in particular `streak_masking.enable` is false, `rescale_variance`
is false, `default_gain` is 1.0 e⁻/ADU, and `default_rdnoise` is 0 e⁻. Copy
the YAML. `validate_config(config)` returns `False` on invalid config.

Each HDU uses one gain and one read-noise value (first present header keyword
in the configured lists) unless both `GAINA` and `GAINB` exist and a section
keyword splits the HDU. Then the variance plane uses that gain map.

## Command

```bash
weightmask science.fits --config weightmask.yml --flat_image flat.fits \
  -o out.weight.fits --output_mask out.mask.fits
```

Run `weightmask --help` for Inputs / Outputs / Run groups. Rebuild a compact
sky mesh with the separate program `weightmask-reconstruct-sky`.

## Cookbook

### MEF

Omit `--hdu` to process every 2-D image extension. Pin one extension with
`--hdu 1` or a CFITSIO-style name `science.fits[1]` (the spec must be trailing;
`--hdu` wins if both are given, and says so). Parallel HDU workers:
`--nproc` (default `min(8, ncpu)`; `0` or `1` is sequential).

The flat, dark and keep-map are matched to each science HDU **by index**, so
`--flat_image`, `--dark_image` and `--badpix_mask` reject an `[N]` spec rather
than ignoring it. A flat or dark MEF shorter than the science MEF is an error
(a unit flat would silently produce unflat-fielded weights) or, for the dark, a
warning, since only hot-pixel rejection is lost. If the bad-mask cache
directory is not writable -- a read-only VOSpace flat, typically -- the run
still succeeds but says so, because the ~20 s/HDU median filter is then
recomputed every time.

### Persistence priors

`--persistence` builds a per-CCD column/row prior from other exposures of the
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
OR'd into `BAD`. `--dark_image` ORs hot pixels into `BAD`, using
`dark_masking.hot_sigma` (default 8). Both, and the MEF
`dead_ccd_*` veto, are CLI/MEF orchestration: single-array `process_image`
does not take a dark or keep-map. Without a flat, `F = 1` and flat-based `BAD`
detection is skipped.

`dark_masking.hot_sigma` selects positive dark outliers above the HDU median
in robust-sigma units (default 8). Zero and negative dark values are valid;
non-finite values are bad.

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
- `--individual_masks`: one FITS file per component (bad, sat, cr, obj, streak, nodata).

Products follow processed science images in HDU order. `EXTNAME` records the
CCD identifier (falling back to the source HDU index). Extension positions can
differ when the science has non-image extensions or a primary image is
compressed. If an HDU or a write fails, the run **exits non-zero**; products may
be incomplete and have different extension counts. Use `EXTNAME` to identify
a product's source HDU.

Outputs must differ from every input and from each other, including through
symbolic or hard links. FITS floating products support 32 or 64 bits;
`ivar_bitpix: 16` is rejected because FITS has no float16 image type.

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

The streak detector combines two fast prescreens
(binned `houghpeaks` and full-resolution `contours`) and a residual RANSAC pass:

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
number of background RMS, using pixels outside the existing mask. A known star
crossed by a confirmed fitted trail does not supply that brightness statistic.
**Set it to `null` to disable it.** Historical measured p90 on MegaCam, in sigma:

| feature | p90 |
|---|---|
| satellite trail, 996195p HDU 35 / 36 | 3.0 / 3.2 |
| saturated-star bleed | 1138 / 1345 |
| bright star arm | 155 |
| hot column group | ~2000 |

The threshold is a **6x margin above a sample of two trails**. That is a
deliberate provisional setting, not a measured constant: a bright satellite
constellation would be deleted by this rule and nothing in the local corpus
would ever reveal it. Raise it, or null it, if that trade is the wrong way round.

The following measurements are historical 0.2.0 results, before the 0.2.1
object-mask fixes (veto on -> off, pixels masked):

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
single-exposure run the old pipeline needed the veto to suppress it. That star
arm has not been requalified after the 0.2.1 object-mask changes.

`tests/test_brightness_veto.py` now pins the two real trails at 17,231 and 9,751
pixels with identical masks at every setting including `null`. Correct upstream
object masks suppress the known column group even with the veto off.

There used to be a third stage, an angle-binned Radon rescue
(`mrt_rescue_params`), intended as the sensitive one -- the thing that finds
what the cheap prescreen misses. It is gone. Measured on 83 real amps it
accepted on three, all three false positives, while `houghpeaks` explained all
23,035 px of real detections; on injected trails at 4, 6, 8 and 12 sigma, two
lengths and two seeds, it moved `recall_line` by +0.000 in eight of eight cells.
It cost 121.7 s/amp of a 125.7 s/amp stage, and removing it took the stage to
4.0 s/amp with both real trails byte-identical at that revision. These stage
counts and timings predate the 0.2.1 fixes. `pixi run streak-sweep` and
`pixi run streak-recall-floor` re-derive those numbers; anyone proposing a replacement
sensitive stage has to re-run them rather than argue from this file.

`pixi run streak-recall-floor` measures where the stage stops finding injected
trails, across {solid, dashed} trails at 4-12 sigma, two lengths and two seeds,
on the production input path. `--disable {contours,houghpeaks,ransac}` repeats the
same grid with one stage switched off, which is how a stage's cost is weighed
against what only it finds. In the historical corpus measurement, contours and
RANSAC accepted on 0 of 224 real amps each, yet each was the
sole finder in the cells it helps and costs recall in none, which is why neither
was removed.

`pixi run exposure-time` answers the question at the unit a survey pays in --
the whole exposure, not one chip. Before the 0.2.1 changes, on a 36-chip MegaPrime MEF (996195p, 353 Mpix,
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

The historical synthetic benchmark reported a gate failure
(`Synthetic-v2 average streak F1 0.159 < 0.200`) with identical per-case scores
with and without the rescue. Those scores used the earlier generator and
recall metric. Version 0.2.1 corrects generator settings and measures recall on
the original truth pixels; its full synthetic gates remain unqualified until
an explicit rerun. The acceptance thresholds are unchanged.

The streak and blank-control cases in `tests.benchmarks` are stage diagnostics:
they use a global median sky, global RMS, and an empty initial mask. They do
not reproduce the calibrated sky and upstream exclusions used by
`process_image`. Production-input evidence comes from the capture helper in
`benchmarks/production_inputs.py`; these two forms of evidence are distinct.

For a direct `weightmask.streaks.detect_streaks` call, setting `debug: true` in
the detector config puts the per-stage evidence (candidates, accepted lines with
angle/rho/confidence, which passes ran) into that config's `['_last_run']`.
`process_image` copies the `streak_masking` config, so it does not expose those
diagnostics to the caller. Its returned `header_info['timings']` reports stage
durations.

### Quality bits

| Bit | Name | Zero weight? |
|---|---|---|
| 1 | `BAD` | yes |
| 2 | `SAT` | yes |
| 4 | `CR` | yes |
| 8 | `DETECTED` | no (set `output_params.mask_detected_in_weight` to zero it) |
| 16 | `STREAK` | yes |
| 32 | `INVALID_VARIANCE` | yes |
| 64 | `NO_DATA` | yes |

Polarity is `set_means_flagged`. See [algorithms.md](algorithms.md).

### Compact sky mesh

In `weightmask.yml` set `output_params.sky_format: mesh`. The product stores
SEP-aligned nodes plus `SKYMESH` / `MESHBW` / `MESHBH` / `SKYH` / `SKYW` cards (~2.5 KB
per CCD). Rebuild:

```bash
weightmask-reconstruct-sky sky_mesh.fits -o sky_full.fits
weightmask-reconstruct-sky sky_mesh.fits -o sky_full.fits --hdu 1
```

### Weight plane

Default `variance.method: theoretical`. The core plane is

```
ivar = g² F² / (S g F + RN²)
```

Flat-fielded Poisson weight. Exact Poisson plus read noise at `F = 1`.
Canonical `weightmask.yml` then adds
`flat_rel_noise: 0.003` and `rescale_variance: true`. Omit those
keys and you get the bare formula.

### Python

```python
from astropy.io import fits
import yaml

from weightmask.process import process_image, validate_config
from weightmask.utils import clean_config_dict

with open("weightmask.yml") as f:
    config = clean_config_dict(yaml.safe_load(f))
assert validate_config(config)

sci = fits.getdata("science.fits", ext=1)
hdr = dict(fits.getheader("science.fits", ext=1))
flat = fits.getdata("flat.fits", ext=1)

mask, ivar, weight, confidence, sky, header_info = process_image(
    sci, hdr, flat, config
)
```

That call is one array. Dark frames, keep-maps, and dead-CCD veto live on the
CLI/MEF path. Exposure-global confidence rescale also does, and only when the
primary map is confidence.
