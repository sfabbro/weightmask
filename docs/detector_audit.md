# Detector and classifier audit

Every stage that flags pixels was reviewed for correctness, cost and latency, and
the findings were either fixed (with a regression test) or measured and left
alone. Numbers come from the real CFHT MegaPrime data in `benchmark_data/megacam/perf`
(2112x4644 = 9.8 Mpix per CCD, 36-CCD MEFs) using `benchmarks/perf_megacam.py`,
on an 8-core Apple machine, single worker, no flat unless noted. The reference
run is the pre-change code with the same harness (see
`benchmarks/perf_megacam.py --compare-baseline`).

## How the reference run is produced

Every claim of "identical products" below is a `--compare-baseline` run of the
same exposure, same HDUs and same flags against the pre-change code:

```bash
# reference: previous revision of the tree, same harness, separate output dir
git worktree add /tmp/base HEAD          # or stash the changes
pixi run -w /tmp/base python benchmarks/perf_megacam.py \
    --out-dir /tmp/base_out --tag ref --exposure-ids 1013719p --hdu-limit 2 --all-products

# current code, compared HDU by HDU against it
pixi run python benchmarks/perf_megacam.py \
    --out-dir /tmp/new_out --tag ref --exposure-ids 1013719p --hdu-limit 2 --all-products \
    --compare-baseline /tmp/base_out
```

A comparison failure exits non-zero and lists the differing HDUs. When a change
is *meant* to alter a contract field, `--compare-ignore-extname` tolerates
exactly the EXTNAME field and still requires identical shapes, dtypes and pixel
values everywhere else (used here for the HDU-naming change). During this work
the reference was produced in-process by restoring HEAD's primitives, which
avoids switching trees; the observable result is the same.

## Measured baseline and result

Per CCD, exposure `1013719p`, mean over the two sampled HDUs, default products
(`--all-products` writes ~235 MB per CCD; that heavier variant measured
115.3 s -> 77.5 s, -32.8 %):

| stage | before | after | note |
|---|---|---|---|
| streaks | 98.8 s | 66.2 s | shared preparation + vectorized pruning |
| cosmics | 8.1 s | 7.9 s | astroscrappy dominates; loops measured negligible |
| background (prelim + 2 iterations + final) | 1.4 s | 1.4 s | unchanged |
| objects | 1.7 s | 1.7 s | unchanged |
| saturation + bleed | 0.5 s | 0.5 s | unchanged |
| variance + weight | 0.5 s | 0.5 s | unchanged |
| **per HDU** | **109.0 s** | **76.2 s** | **-30.1 %**, pixel data identical (only the HDU name changes) |
| flat bad mask | 21.6 s | ~0 s | 2nd exposure through the same flat (cache hit) |

Re-verified on the committed tree with a fresh baseline run, 2 HDUs of
`1013719p`, `--all-products`: 110.4 s/HDU -> 77.1 s/HDU (**-30.1 %**), streaks
100.1 s -> 67.1 s, 11/11 product files identical in shape, dtype and pixel
values. The single EXTNAME difference is the intended one (`INVVAR_HDU1` ->
`INVVAR_8341-7-5`, matching the input `CCDNAME`), which
`--compare-ignore-extname` tolerates while still requiring everything else to
match.

Thread scaling, 4 HDUs of one exposure: 80.9 s/HDU sequentially, 128.7 s wall at
4 workers = **2.51x** with 3.08 cores busy. The previously recorded figure was
1.37x (about 1.4 cores busy); removing Python-level per-region loops and the
duplicated preparation took most of the GIL pressure with it, so the
process-based backend this plan deferred is not needed yet.

## What changed

### `streaks._prepare_streak_image` — duplicated preparation (fixed)

The 15x15 median filter plus `disk(3)` white top-hat is mask-independent (only
the final `np.where` uses the exclusion mask), yet one `detect_streaks` call ran
it up to four times on the same `data_sub`: the binned Hough prescreen, the
corridor build, and both again for the unmasked retry pass. Measured 22.5 s for
the binned + full-resolution pair on a real CCD (21.7 s of it in
`scipy.ndimage.rank_filter`), previously paid twice.

`_streak_image_core` splits the mask-independent part out and `_StreakImageCache`
memoizes it per source array (keyed on identity, holding a strong reference, so a
recycled `id()` cannot alias a stale entry). One prepared image per array now
serves every pass, with the masks applied afterwards.

### `streaks._prune_small_edges` — per-region Python loop (fixed)

`regionprops(label(edge_mask))` with `region.perimeter` ran once per HDU
component: 55k components (measured on a comparable 2.1 Mpix mask) took 1.5-1.9 s
per call, and the multiscale extractor calls it six times per satdet pass.
`_region_perimeters` reproduces skimage's computation for every region at once
(same 4-connected border, same `[[10,2,10],[2,1,2],[10,2,10]]` kernel, same
per-region bounding-box semantics) in 0.069 s per call, 12 calls totalling
0.83 s/HDU. Regions whose perimeter sits within 1e-6 of the cut are re-measured
with skimage's own `perimeter()` so the keep/drop decision is bit-for-bit the one
`regionprops` makes.

### `streaks._detect_streaks_mrt_like` — Radon rescue latency (gated, not reworked)

A single call costs 35.7 s and ~350 MB: skimage's `radon` pads a 2112x4644 CCD to
a 6568x6568 square and warps it at each of 90 angles (0.38 s per `warp`). It
fires exactly on the low-confidence fields the rescue exists for, so on a clean
field it is now the largest single item in the pipeline.

Two bit-identical speedups were investigated and rejected as impossible within
skimage's API: restricting the warp to the rows that can contain content (the
`R` translation would have to absorb the row offset, which changes the float32
rounding of every source coordinate), and shrinking the padded square (the
padding sets the sampling grid). The angle-binned variant that would cut it to
~9 s changes the sinogram, so it needs the injection-recall suite before it can
ship — see follow-ups.

What ships instead is `mrt_rescue_params.enable` (default `true`, so behaviour is
unchanged), so a survey that knows its fields carry no intermittent trails can
skip a 35 s/350 MB stage per CCD.

### `bad.compute_flat_bad_mask` — recomputed per exposure (fixed)

Each flat HDU's mask costs 21.6 s, depends only on that flat HDU, the tile size
and the `flat_masking` settings, and was recomputed for every exposure sharing
the flat: 36 HDUs x 10 exposures x 21.6 s of redundant work per survey run.

`compute_flat_bad_mask_cached` stores it as `.npy` next to the flat (or in
`flat_masking.bad_mask_cache_dir`), keyed on the flat's absolute path, byte size,
nanosecond mtime, HDU index, shape, tile size and every `flat_masking` setting.
A replaced flat or retuned masking therefore cannot read a stale mask; a
corrupt, partial, missing or unwritable entry falls back to the plain
computation, so products never depend on the cache. Set
`flat_masking.bad_mask_cache: false` to disable it.

### `mef.process_all_hdus` — whole-MEF retention (fixed)

The per-HDU products (`mask`, `ivar`, `weight`, `confidence`, `sky` and the six
component masks, ~235 MB per MegaPrime CCD) were accumulated in a `results` dict
until every HDU finished, so a 36-CCD MEF peaked at ~8 GB before any write. The
writer now consumes each HDU as it is produced and at most one worker-window of
HDUs is computed ahead of it; writes still happen in HDU order, so the output
files are unchanged. A regression test asserts the held-HDU count stays within
`max_workers + 1` for a 24-HDU MEF.

### `mef._hdu_identifier` — output naming (changed, contract-visible)

Products were named from HDU position alone (`MAP_HDU1`) because the input
`EXTNAME` is a tiling-compression artifact (`COMPRESSED_IMAGE` on every HDU) and
fitsio's image HDUs expose no `name`. They now carry the header's CCD identifier
(`MAP_8341-7-5`), taken from `CCDNAME`, then `CCDNAM`, falling back to the
previous scheme. `CCD` is deliberately excluded: on MegaCam it holds the
detector model (`Marconi/EEV CCD42-90`), identical for all 36 HDUs, so using it
would have given every product in the file the same EXTNAME. This is the one
intentional product change in this work; data values are unaffected.

### `cosmics._filter_faint_components` — deprecated skimage API (fixed)

`RegionProperties.major_axis_length` / `minor_axis_length` are deprecated in
skimage 0.26 in favour of `axis_major_length` / `axis_minor_length` and are
scheduled for removal in 2.0, which would have broken the faint-CR morphology
gate. `_axis_lengths` prefers the new spelling and falls back for older skimage.

### `mef` — bogus `compress="NOT_SET"` (fixed)

`_StreamingMapWriter.write` and `_rescale_confidence_to_global` passed
`compress="NOT_SET"` to fitsio when not compressing, which fitsio warns about and
ignores. The keyword is now passed only when actually compressing, so the file
format and bytes are unchanged.

## Second pass: the Radon rescue, the prescreen, and one real bug

Same exposure (`1013719p`, 2112x4644 CCDs). The headline figure is the whole thing, not a
sample: the reference is a pristine `git archive HEAD` export in `/tmp`, both sides are run
over **all 36 science HDUs** with 4 workers and `--all-products`, and every product is
compared HDU by HDU.

| full exposure, 36 CCDs, 353 Mpix | HEAD | after |
|---|---|---|
| wall time (4 workers) | 1676.2 s | **1179.9 s** (-29.6 %) |
| per CCD | 46.6 s | **32.8 s** |
| CPU time (sum over HDUs) | 6504 s | 4571 s (-29.7 %) |
| streak stage | 5701.9 s | 3819.4 s (-33.0 %) |
| cosmics (defaults unchanged) | 613.7 s | 572.8 s |
| peak RSS | 4.8 GB | 4.6 GB |
| products | — | **11 products x 36 CCDs (407 HDU comparisons) identical**, 0 differing pixels |

The 36-CCD comparison replaces the earlier two-HDU spot check: no product of any kind
moved, so the rms convention, the Radon rewrite and the prescreen gate are all
output-neutral on real data. It is also not a vacuous comparison -- 7 of the 36 CCDs
detected trails (30 416 mask pixels, up to 19 151 px on one CCD) and 109 090 cosmic-ray
pixels were flagged, and every one of those pixels is in the same place on both sides. The
per-HDU spot checks below are what attributed the change to individual stages.

One caveat on the wall-time figure: the two runs were sequential on a shared workstation,
so a few percent of the difference is ambient load. The products, not the clock, are what
make the change safe; the streak-stage CPU total (5702 -> 3819 s) is the load-independent
part of the speedup.

### `streaks._detect_streaks_mrt_like` — the candidate lines were mirrored (fixed)

`radon` and `_candidate_from_rho_theta` do not share a convention. The builder speaks
Hesse (`theta` = the line's normal direction, `rho` = signed distance along it), which is
what `hough_line` produces; `radon` parameterises the same line *reflected*,
`theta_radon = 90 - phi`. The rescue passed radon coordinates straight into the Hesse
builder, so at every oblique angle the strip refiner was handed the mirror of the
candidate — a line that does not exist in the image, rejected as `no_support`. The two
conventions agree only near `theta = 0`, which is exactly where a chip's bright column
bands project, so the rescue appeared to work while proposing nothing but artefacts.

Calibrated on injected lines at eight (angle, offset) combinations; the conversion is
`theta_hesse = 180 - theta_radon`, `rho_hesse = -rho_radon`, and after it every injected
line is reconstructed to within a pixel:

| injected line | radon peak | builder before | after |
|---|---|---|---|
| phi=90, x = cx-200 | theta=0, rho=-200.5 | correct | correct |
| phi=0, y = cy+150 | theta=90, rho=-149.5 | 300 px away | correct |
| phi=45, offset +120 | theta=45, rho=-119.5 | direction 135 deg | direction 45 deg |
| phi=22.9, offset +80 | theta=67, rho=-79.5 | 135 px away | correct |
| phi=140, offset -60 | theta=130, rho=-58.5 | 69 px away | correct |

Effect on the case the rescue exists for — injected trails on a flat synthetic field, same
configuration, `accepted=0` and `recall=0.000` in all four rows before:

| trail | bin=1 recall | bin=4 recall |
|---|---|---|
| 60 sigma | 0.904 | 0.394 |
| 12 sigma | 0.905 | 0.902 |
| 6 sigma | 0.358 | 0.938 |
| 4 sigma | 0.904 | 0.797 |

`bin: 1` is the default because the per-case spread at `bin: 4` includes a *bright* trail
found only half as well; `bin: 4` is the documented speed lever (~1.6 s vs ~20 s).

### `streaks` Radon geometry and significance (fixed)

- skimage pads to `sqrt(2) * max(shape)`: 6568 per side for a 4644x2112 CCD whose true
diagonal is 5102. Transforming on a diagonal-sized pad keeps the algorithm (same bilinear
`warp`, same centring) but removes 40 % of the pixels per angle; the measured stage cost
falls from ~74 s to ~20 s at `bin: 1`.
- The per-column `mad_std` used to set significance was taken over a column that is mostly
zero padding — 4628 of 6568 entries at theta = 0 — so the MAD was 0, a hard-coded `1.0`
fallback took over, and that column's values read as million-sigma peaks. Statistics are
now computed over the geometrically valid rho range (`|rho| <= h|cos t|/2 + w|sin t|/2`),
with a relative floor instead of the fixed one. The same peaks then report their true
~2.6e3 sigma rather than ~2.6e6.
- A moving-median high-pass along rho (`sinogram_highpass`, default 101) suppresses the
broad star bumps that dominate the sinogram: per-angle MAD 180 -> 9 and a trail's own
significance 51 -> 1468 on the synthetic field.

What the rescue still cannot do is worth stating plainly: on a real CCD the two brightest
columns are genuinely the strongest lines in the image, so a global top-k search spends its
four candidates on them. They are then rejected by the strip refiner's
existing `step_discontinuity` guard, which is what stops this path from masking a good
column as a satellite trail. Restricting the search to non-axis angles does **not** fix the
ranking — a one-pixel column projects into every angle (its rho width grows as the angle
leaves the axis), so the artefacts simply move to theta = +-4 deg; measured, not assumed.

### `streaks` prescreen gating (`satdet_params.skip_when_prescreen_confirmed`, default true)

The pipeline already runs the cheap binned-Hough accumulator stage before the
full-resolution multi-scale Canny/Hough sweep. When it confirms a trail, the sweep costs
~24.5 s per CCD to produce the ~10^5 segments that only re-derive what the accumulator peak
found. Measured on real HDUs, with the gate on and off:

| field | mask pixels (off / on) | time | differing pixels |
|---|---|---|---|
| clean | 10 945 / 10 945 | 27.4 s -> 3.0 s | **0** |
| bright trail (8 sigma, 2500 px) | 20 651 / 20 651 | 28.1 s -> 3.5 s | **0** |
| bright + faint trail | 41 838 / 41 838 | 28.3 s -> 3.7 s | **0** |

The gate only fires when the prescreen has *already* accepted, so a field where the sweep
is the stage that finds the trail is unaffected by construction.

### `cosmics` single pass (`cosmic_ray.single_pass`, default false)

Two real HDUs, injected CRs, `benchmarks/cr_faint_curves.py`:

| variant | single-pixel recall | worm recall | fp px | time |
|---|---|---|---|---|
| base pass alone | 0.197 | 0.086 | 3140 | 5.0 s |
| production two-pass | 0.207 | **0.275** | 3202 | 7.8 s |
| single pass + morphology gate | **0.010** | 0.313 | 1850 | 4.9 s |

The two-pass arrangement earns its 2.8 s: multi-pixel worm recall more than triples
(0.086 -> 0.275) for 62 extra false-positive pixels. The single-pass variant is faster and
cleaner but cannot see single-pixel hits at all, because the morphology gate requires
3-12 px elongated components — which is the reason the second pass exists. Shipped gated
off, with the numbers.

### Background RMS sentinel

`estimate_background` marks pixels whose local RMS it could not measure with `inf` (7.07 %
of pixels, 150 of 2112 whole columns on `1013719p` HDU 6). Four call sites used
`np.nanmedian` to substitute a typical value; `nanmedian` ignores NaN but not `inf`, so once
more than half a chip carried the sentinel the substitution silently became `inf` and every
one of those sites flipped from "judge with a substitute" to "never detect" as a function
of the data. Each site now states its intent through `utils.rms_valid_mask` /
`utils.robust_rms` / `utils.rms_or_robust`: detection thresholds treat the sentinel as inert
(`streaks._detect_streaks_mrt_like`, `streaks._refine_trail_mask`, the contour scale,
`cosmics._apply_psf_protection`, the sparse RANSAC threshold), while rejection and quality
gates substitute a robust value (`cosmics._post_filter_components`,
`cosmics._filter_faint_components`). SEP was verified to honour `err = inf` (no objects in
that region) and is pinned by a test, as is `variance`'s zero weight and `satur`'s refusal
to grow a bleed trail through a sentinel column.

### Thread scaling, re-measured (current code)

Four HDUs of `1013719p`, `process_image` across a thread pool:

| workers | wall | per HDU | speedup |
|---|---|---|---|
| 1 | 198.1 s | 49.5 s | 1.00x |
| 2 | 129.8 s | 32.5 s | 1.53x |
| 4 | 82.7 s | 20.7 s | **2.40x** |
| 8 | 91.2 s | 22.8 s | 2.17x |

The w8 regression is the pool running out of work (four tasks, eight threads), not a code
defect. 2.40x at four workers is unchanged from the earlier measurement, so a process-based
backend remains unjustified.

### Left alone in this pass

- **`objects` and the Hough stages at full resolution.** The sweep is skipped only when the
  prescreen has already confirmed; reducing it further (one source, one sigma) is a recall
  trade that needs the real-trail ground truth below.
- **`sepmed=False`.** Measured *slower* than the separable default (11.2 s vs 5.2 s for one
  L.A.Cosmic pass), so the knob is a dead end rather than a tuning opportunity.
- **`niter`.** Still the dominant astroscrappy cost and still forwarded; the audit found no
  cheaper equivalent.

## Checked and deliberately left alone

- **`cosmics` (`CR`)** — the two `regionprops` post-filters look like the streaks
  antipattern but are not: 1970 regions cost 0.15 s (`_post_filter_components`)
  and 0.11 s (`_filter_faint_components`) against 7.47 s inside astroscrappy's
  `detect_cosmics` (attributed to the caller because the Cython callee reports no
  frame). The dominant knob is `niter`, already forwarded.
- **`objects` (`DETECTED`)** — the `for i in range(len(objects))` in
  `_apply_vectorized_ellipse_mask` is the fallback used only when SEP's vectorized
  `mask_ellipse` raises; the production path is one batched call. Total object
  stage cost 1.8-2.0 s/HDU.
- **`satur` (`SAT`)** — histogram plateau-tail detection plus a `ndimage.label` +
  dilation bleed growth: 0.3 s/HDU, no loops over pixels, no deprecated APIs.
- **`background` (`sky`)** — four `estimate_background` calls per HDU (preliminary,
  two iterations, final) at 0.35-0.5 s each; the final call is skipped when the
  mask is unchanged. `sky_to_mesh` encode is ~0 ms.
- **`variance` / `weight`** — 0.3-0.4 s/HDU, no hot loops; the patch loop in
  `rescale_variance` covers a 9.8 Mpix plane in well under a second.
- **`bad.detect_bad_pixels` / `detect_non_illuminated`** — tile-parallel by
  design; the DATASEC complement is a slice assignment. No changes.
- **I/O** — FITS reads are 0.06-0.09 s per HDU, so the audit confirmed CPU, not
  I/O, remains the constraint.

## Follow-ups (not done here, with the evidence to prioritise them)

1. ~~**Angle-binned Radon rescue.**~~ **Done in the second pass** (`mrt_rescue_params.bin`),
   together with the tight diagonal pad, valid-rho statistics and a star-clutter high-pass:
   the stage is ~1.6 s at `bin: 4` and ~20 s at the default `bin: 1`, against ~35-74 s before,
   and it now actually detects trails (recall 0.36-0.94 on injected 4-60 sigma trails, from
   0.00 in every case before the convention fix). What remains for this path is candidate
   *ranking*: on a real CCD the two brightest columns outrank every trail by orders of
   magnitude, so a global top-k search still spends its candidates on them. Fixing that
   needs either a per-angle-subtracted sinogram (a robust regression against the column-band
   basis) or cross-exposure confirmation, not a threshold tweak.
2. **Process-based backend for `--workers`.** Re-measured at 2.40x for 4 workers and 2.17x for
   8 (the 8-worker figure is the pool running out of work, not GIL contention). Still below
   the core count, but the remaining per-HDU Python work is now a few seconds, so the ceiling
   this would buy is small. Deferred again, deliberately.
3. **Curated real-trail ground truth.** `benchmarks/mine_trail_candidates.py` mines
   candidates and `benchmarks/race_poloka.py` cross-checks Poloka's track catalogue, but no
   reviewed label set is committed, so every remaining detection decision (the sweep
   reduction above, the `max_candidates` cap, the confidence thresholds) is being tuned
   against injected trails, which do not reproduce the clutter that actually drives them.
   This is the highest-value remaining detection investment.
4. **Per-trail catalogue output.** Detection currently emits a bit; the stage-level evidence
   (angle, span, support width, confidence, which pass accepted) already exists in
   `_last_run` under `debug: true`. Writing it as a FITS table per CCD would make tuning
   auditable and let downstream consumers verify a trail rather than trust a bit.
5. **`mef._get_global_median` flat re-read.** The dead-CCD veto re-reads all 36
   flat HDUs once per exposure purely for their medians (36 reads x 10 exposures
   per run). It could share the flat cache identity used by
   `compute_flat_bad_mask_cached`.
6. **`reconstruct_sky_mesh` cubic-spline loop.** 0.96 s per CCD because it builds
   one `CubicSpline` per output row (4644 of them) and per mesh column. Off the
   default path (`sky_format: full`), but a banded-solve vectorization would
   remove it.
