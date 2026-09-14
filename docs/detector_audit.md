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

1. **Angle-binned Radon rescue.** Estimate ~9 s instead of ~35.7 s on a clean
   field (2x-binned input => 3284x3284 pad vs 6568x6568). Requires
   `benchmarks/streak_inject.py` recall/false-positive validation before/after,
   because it changes the sinogram. This is the largest remaining per-HDU item.
2. **Process-based backend for `--workers`.** 2.51x at 4 workers is still below
   the core count; the measurement says threads have room left, so revisit only
   if a wider sweep plateaus.
3. **`mef._get_global_median` flat re-read.** The dead-CCD veto re-reads all 36
   flat HDUs once per exposure purely for their medians (36 reads x 10 exposures
   per run). It could share the flat cache identity used by
   `compute_flat_bad_mask_cached`.
4. **`reconstruct_sky_mesh` cubic-spline loop.** 0.96 s per CCD because it builds
   one `CubicSpline` per output row (4644 of them) and per mesh column. Off the
   default path (`sky_format: full`), but a banded-solve vectorization would
   remove it.
