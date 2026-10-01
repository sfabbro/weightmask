# Changelog

## Unreleased

### Contours keeps its place; the instrument that decides is now reusable

The Radon rescue was removed below because it cost 121.7 s/amp and found nothing.
Contours has the same two surface signals -- 0 acceptances on 224 real amps, and
44% of what the stage now costs -- and the opposite answer underneath:

| injected trail | with contours | without | |
|---|---|---|---|
| 800 px, 6 sigma | 0.167 | **0.000** | contours is the only finder |
| 800 px, 8 sigma | 0.167 | **0.000** | contours is the only finder |
| other 6 cells | equal | equal | no effect |
| mean gain | **+0.042** | | costs 3.3 s/run, 56% of the stage |

Contours costs recall on **0 of 8** cells. That asymmetry is the whole result:
the rescue moved recall by +0.000 everywhere, contours is the sole reason some
injected trails are found at all. Per-amp, the whole gain is on 1013719p:9 at
seed 0, where houghpeaks reaches 0.000 and contours reaches 1.000.

The two stage runs of `benchmarks/streak_recall_floor.py` that produced this are
the same instrument that killed the rescue, so the comparison is like-for-like:
same amps, same grid, same production input path, stages switched by their own
config block rather than by monkeypatching.

`benchmarks/streak_recall_floor.py` now takes `--disable {contours,houghpeaks,ransac}`
(repeatable). A stage switch that is not in that list is a `ValueError` rather
than a silently ignored flag -- the failure mode this whole section is about is a
knob that looks connected and is not.

### Delete the Canny/Hough stage, and stop a re-mask of known-bad pixels

Measured over 56 real MegaCam amps (every 4th HDU of six exposures, production
inputs captured from the `process_image` streak stage), the `satdet` stage --
multi-scale Canny + probabilistic Hough + segment clustering -- **ran on 56 of
56 amps, accepted a candidate on none, and explained zero pixels of any final
mask: 1,543 s across the sweep, 27.5 s per amp.**

It is not mis-tuned. Sweeping `confidence_threshold` from 0.22 down to 0.02
produced zero acceptances on six amps including both known real trails. Sweeping
the Canny thresholds from 0.06/0.22 up to 0.60/1.30 left acceptance at zero
too. Internally the stage is not idle -- it produces ~131,000 Hough segments and
~5,300 clusters on a 9.8 Mpix amp -- but only 16 clusters survive its own gates
and **the closest survivor is 1,357 px from the known satellite trail**, so the
trail is not in the stage's own candidate list at any setting. Its
`min_cluster_segments` gate is also undocumented: the config's `min_segment_
accept` gates a different, later check.

Removed: `_detect_streaks_satdet` and 13 helpers only it called, plus
`_StreakImageCache` and the three unmasked-retry config knobs.
`_refine_trail_mask`, `_sample_trail_strip` and the rest of the strip machinery
survive -- houghpeaks, contours and the rescue all use them.

**All four real detections in 996195p are bit-identical before and after** (1286,
584, 12001, 11034 px), so nothing was lost. HDU 35's streak stage went 37 s ->
1.2 s, because the prescreen's own detection is now enough to skip the rescue;
HDU 1 is 37 s -> 31.5 s, where the rescue still runs.

The rescue gate was rewritten while the stage it depended on was being deleted.
It used to be `low_confidence`, a flag owned by satdet, combined with a
`prescreen_confirmed` heuristic -- a cost optimisation that could veto a more
sensitive detector. It is now a statement about the image: the rescue runs unless
the prescreen has already masked enough to have handled the frame.

**Pre-masked veto.** A component lying mostly inside the mask the pipeline already
carries is not a new finding. The dominant false positive on real amps is a
saturated star's bleed: 45% of its pixels are already flagged, against **2%** for
a real satellite trail -- a factor of 22, and free to compute because both masks
are already in hand. Applied per component so a stage returning both a bleed and
a trail keeps the trail, and to all four stages. On 996195p this removes the two
false positives (HDUs 1 and 16) and leaves both real trails untouched.

**Who owns the survivors?** Bit attribution in the final mask says 97% of the
dead-column pixels carry `STREAK` and nothing else, so this is not a duplicate
label -- `bad.py` genuinely does not own them. It detects bad columns *from the
flat*, and in `1013719p` HDU 5 those columns are ordinary in the flat
(median 1.0108 vs a flat median of 0.9967). They are instead at 97-99.5% of the
`SATURATE` level (column medians 62,069-63,663 against `SATURATE = 63,973`), so
the saturation stage misses them by a hair. The same structure appears in all
three epochs of the QW322 pointing, so it is static.

**Brightness veto** (`mask_params.max_component_sigma`, default 20). As a
multiple of the local background RMS at p90, measured on real amps:

| feature | p90 |
|---|---|
| satellite trail (996195p HDU 35/36) | ~2 sigma |
| saturated-star bleed | 54-66 sigma |
| near-saturated column group | ~2000 sigma |

Both real trails survive **byte-identically** (12,001 and 11,034 px). The
column components fall 10,773 -> 7,182 and 9,256 -> 5,785: the veto removes the
saturated core but not the ~3-sigma halo around it, so it is a partial fix for
that class. This is a stopgap; a pixel at 99% of `SATURATE` arguably belongs to
the saturation stage, which is a wider change than this stage should make on
its own.

**No confirmed satellite trail exists locally, and an attempt to build one
failed.** The two faint linear features in `996195p` HDUs 35/36 are real
(60-90 sigma after line integration, row median 34 e- against 40 e- noise,
i.e. genuinely marginal) but nothing confirms either as a satellite: their
fitted sky position angles differ by 1.74 deg once the 0.15 deg chip rotation is
removed, roughly 18x the fit uncertainty, so they are probably two separate
features rather than one trail, and there is no sibling epoch locally to test
persistence against. A first attempt at a stability fixture was discarded
because its independent line-integration fit reported the feature 66 px from
where the detector puts it; the cause was a scoring bug in the fit, which
rewarded raw line sums so a few bright pixels at y=3220 (row median 0.49 e-
outlived the genuine trail at y=3180, row median 34 e-). The detector was right.

Consequence: **the repository still cannot measure real-data trail recall.**
The veto thresholds are currently justified by one confirmed-by-eye feature and a
0.25 pre-masked fraction, not by a recall measurement. That is the honest state.

### Streak benchmarks were scoring a code path production never runs

`score_trail_truth.py` and `streak_inject.py` each built their own detector
inputs: one `estimate_background` pass over the raw frame with an all-False
exclusion. Production does neither. It iterates the background (preliminary pass,
then cosmic-ray and bleed masking, then a final pass over the accumulated mask)
and it hands `detect_streaks` a populated
`interim_mask_bool | final_obj_mask | detector_prior`. Both benchmarks therefore
measured something the pipeline never executes.

Three separate things followed from that, and the first two were wrong:

1. **The committed false-positive baseline was the background stage's error
   attributed to the streak stage.** A single unmasked pass leaves the chip-fixed
   columns and rows in `data_sub` as bright linear features. Re-scored on the
   production path, on the same 8 HDUs and the same 32 labelled entries: **8
   false positives (rate 0.250) -> 1 (0.031)**, gate 0.100, so the gate passes
   where the committed run recorded 0.124 and failed. Seven of the eight HDUs go
   to zero. `houghpeaks` scores 0.000 on the same inputs for 12 s against 567 s
   for the full cascade.
2. **The `bin` recall table in `weightmask.yml` cannot be used to pick a knob.**
   It was measured on the same non-production path, so the documented
   non-monotonicity (bin=1 scoring 0.36 at 6 sigma where bin=4 scores 0.94) was
   never a resolution trade. The pick in that comment is withdrawn pending
   re-measurement.
3. **The `prescreen_confirmed` suppression is not the recall defect it looked
   like.** It was the natural suspect -- the cheap prescreen accepting something
   skips satdet and the Radon rescue. Flipping
   `satdet_params.skip_when_prescreen_confirmed` to `false` on injected trails
   moved the cost from 6.0 s to 22.6 s and left recall **exactly unchanged** at
   0.101. Rejected.

The scoring tally was checked against the committed totals before any comparison
was drawn: the method reproduces 30 FP / 241 entries on the full set exactly.

`benchmarks/production_inputs.py` replaces the hand-rolled inputs with a capture
of the real `process_image` streak stage, so the harnesses cannot drift from the
pipeline again. It raises rather than falling back if the streak stage is not
reached, because a silent fallback is the defect. The score artifact now records
the config path, its sha256 and the `streak_masking` block; the 2026-09-18
artifact recorded none, which is why its timings could not be reconciled with a
direct measurement. Five new tests in `tests/test_production_inputs.py` pin the
capture to production and were each mutation-checked.

### Verified against real data

Every performance change in the audit below was checked end-to-end, not just per
change: mask, inverse variance, weight, confidence and sky products for three
real MegaCam amps of `1013719p` (4644x2112, 29.4 Mpix total), streaks enabled,
`weightmask.yml` unmodified, against the pre-audit revision `0811166`.

**All 15 arrays bit-identical: zero differing pixels, max|diff| exactly 0.** The
throughput work cost nothing in output on real data.

- Saturation tail histogram: 1285 ms -> 47 ms per 9.8 Mpix amp, by binning only
  the pixels in the tail instead of letting `np.histogram` sort all 9,811,968 of
  them to count the 510 that lie in it. Exact -- counts are integers, so there is
  no summation order to preserve.
- Sky mesh reconstruction: 4.4x and bit-identical, via the `axis=0`/`axis=1`
  form of the same cubic spline. Note this is the mesh<->full-res inverse used
  when reading a stored sky product, not the per-CCD path, so it is not counted
  toward CCD throughput.
- `variance`: `np.divide(..., where=)` in place of boolean fancy-indexing.
- `mef`: dropped redundant `np.array(..., copy=True)` around FITS reads, and
  `astype(copy=False)` on reads. `fitsio.read()` returns an owning, writable
  array for plain and RICE_1 tile-compressed images alike, so this is not an
  mmap alias.

### Correctness and performance audit (bugs found and fixed)

Continuation of the audit below, covering the MEF/CLI input layer:

- **Fixed: the documented CFITSIO input form did not work.** `validate_input_files`
  stat'ed the raw string, so `science.fits[1]` failed with "Input file not found"
  before the spec was ever parsed — while `docs/usage.md` advertised exactly that
  form and `weightmask-reconstruct-sky` honoured it. The spec is now stripped
  before the filesystem is touched.
- **Fixed: a silent trap in the auxiliary inputs.** `--flat_image x.fits[2]` was
  parsed into a variable nothing read, so the flat was matched by index anyway
  and the explicit HDU was silently ignored. An `[N]` on `--flat_image`,
  `--dark_image` or `--badpix_mask` is now rejected, since those are matched to
  the science HDU by index. Also removed the dead binding, and `--hdu` now says
  when it overrides an `[N]` in the input spec.
- **Fixed: a wrong-shaped sky product.** `reconstruct_sky_mesh` returned `(h, 1)`
  when the mesh had a single node column — reachable for any HDU narrower than
  the mesh box, e.g. 30 px wide with `box 32` — and wrote it without complaint.
- **Fixed: a silent repeated cost.** An unwritable bad-mask cache directory (the
  normal case for a read-only VOSpace flat, where the cache cannot live next to
  the file) produced no output at all, while the ~20 s/HDU median filter was
  recomputed on every run. It now warns and names `bad_mask_cache_dir`.
- **Fixed: a partial run exited 0.** `run_pipeline` only failed when *nothing* was
  processed, so a run with a failed CCD wrote fewer data extensions than there
  were science HDUs and still reported success — and from the first skipped HDU
  onward, extension position no longer matched the science HDU. A short run now
  exits non-zero and says how many were written; `EXTNAME` is documented as the
  authoritative source index.
- **Fixed: the contract could launder a mask or a non-finite weight.**
  `canonical_quality_mask` cast an `int64` mask with `astype`, wrapping modulo
  2^32, so any flag at bit >= 32 vanished and that pixel was given **full
  weight** — the exact failure the quality mask exists to prevent. Wider-than-
  uint32 inputs are now refused. `WeightMaskProduct` also accepted `inf` and
  `NaN` weights, because `x < 0` is False for both; the validator now requires
  finite weight and confidence.
- **Documented:** the persistence prior is a prior on the *streak detector* only.
  Those pixels are withheld from streak detection so detector-fixed structure is
  not reported as a trail; they are not bad pixels, keep their normal weight and
  carry no quality bit.

- **Fixed: a silent unit flat.** A flat MEF shorter than the science MEF left
  `hdu_flat_obj = None`, so `process_image` substituted `F = 1` and printed one
  `INFO` line. Every affected CCD got unflat-fielded weights and the run still
  reported success. A short flat or keep-map now fails the HDU with a reason
  and writes no products. This also exposed a flaw in an existing test that had
  been passing *because* of the bug (science written with an empty primary
  against a single-image flat, off by one).
- **Fixed: a silent dark.** A dark frame shorter than the science MEF was
  ignored with no log output at all, losing dark-derived hot-pixel rejection.
- **Fixed: `scale_to_100` produced a mislabelled, then destroyed, product.**
  With `confidence_params.scale_to_100: true` the file holds 0-100 but the
  header claimed `normalized_weight_0_to_1`; and because
  `normalize_scope: per_exposure` is the default, the global rescale then ran
  `np.clip(data * factor, 0, 1)` over it, flattening everything above 1 % of
  the normalisation to exactly 1.0. The clip now follows the map's actual range
  and the semantics card is `normalized_weight_0_to_100` when scaled.
- **Fixed: `detect_objects` raised `UnboundLocalError` on every frame with zero
  detections**, reporting a normal outcome as `ERROR: Object detection failed`.
  The returned mask was always correct, which is why the existing
  empty-input test never caught it — it asserts the mask, not the log.
- **Fixed: `"--5"` in a config value aborted config loading** with a traceback
  instead of passing through as a string.

The two boldest claims from the review that preceded these — that
`ellipse_k: 3.0` empties the `DETECTED` mask, and that `amplifier_gain_map`
cannot handle a `fitsio.FITSHDR` — were both measured and refuted.

### Correctness and performance audit (round one)

A two-round review of every module except `streaks.py` (which is 65 % of per-CCD wall time
and is being left for its own round). Both rounds were about checking that an optimisation
is actually behaviour-preserving, which turned out to matter more than the optimisations.

**Three latent bugs, each invisible to the test suite as it stood:**

- **Flat field silently dropped.** An attempt to skip re-reading the flat when the bad-pixel
  mask was precomputed was wrong: `process_image` uses the flat as the *actual* flat field
  for the variance/weight math, not just for the bad mask, so skipping it substituted a
  unit flat and disabled flat-fielding on every parallel run. Undetectable by the suite
  because every test flat is an array of ones. Pinned by
  `test_process_hdu_still_passes_the_real_flat_when_bad_mask_is_precomputed`.
- **`build_weight_product` mutated the caller's quality mask.** A `copy=False` cast in
  `canonical_quality_mask` aliased a `uint32` input, so the `INVALID_VARIANCE` pass wrote
  bits into the caller's array -- including the mask that gets written to disk. The copy is
  load-bearing and is now documented as such; pinned by
  `test_build_weight_product_does_not_mutate_the_callers_quality_mask`.
- **Stale inverse-variance position in the chip-replica veto.** The raw-weight branch reused
  an `ipos` that was only assigned inside the weight-map branch, so
  `--out-weight-raw` without a weight map raised `NameError` on the first HDU or restored
  from the *previous* HDU on later ones. Pinned by
  `test_raw_weight_restore_does_not_depend_on_the_weight_map`.

**Performance (all verified output-preserving):**

- **Saturation: 1285 ms -> 47 ms per 9.8 Mpix amp.** `estimate_saturation_robust_clump`
  built its 100-bin histogram with `np.histogram(finite_data, bins=<edges>)`. With explicit
  bin edges numpy sorts the entire input while discarding everything outside the edge
  range -- sorting 9,811,968 float32 values in order to count the 510 that lie in the
  saturation tail. Restricting to the tail first and binning only that is exact (counts are
  integers, so there is no summation order to preserve), asserted equal to `np.histogram`
  over 300 fuzzed bin-edge cases. `_estimate_plateau_tail` already did this; the two are
  now consistent. This stage was *absent* from the cProfile rather than cheap, because at
  ~1.3 s/amp it fell below the top-35 cutoff.
- **Sky mesh: 4.4x, bit-identical.** `reconstruct_sky_mesh` built one `CubicSpline` object
  per column and per row; the `axis=0`/`axis=1` vector form is the same spline evaluated
  in bulk. Note this is the mesh<->full-res inverse used when reading a stored sky product,
  not the per-CCD path.
- `variance`: `np.divide(..., where=)` instead of boolean fancy-indexing, which also stops
  dividing by zero outside the valid region.
- `mef`: dropped redundant `np.array(..., copy=True)` around FITS reads in the chip-replica
  veto and used `astype(copy=False)` on reads. `fitsio.read()` returns an owning, writable
  array for both plain and RICE_1 tile-compressed images, so this is not an mmap alias.
- `satur`: the finite-pixel sample is computed once and threaded through instead of
  re-running `np.isfinite` in each of the three estimators.

**Debuggability:**

- The perf harness could report a physically impossible throughput. `--resume` appends each
  checkpointed exposure to the report (so its pixels and stage times count) but `continue`s
  without running it, while `total_wall` times only the checkpoint walk: the committed
  `test_outputs/perf/megacam_perf_after.md` shows `total_wall=0.0s` (really 5.479e-5s, hidden
  by the `:.1f` format) beside `mpix/s=64442323.708`, against a true 0.073. Throughput is now
  refused rather than faked, resumed records are tagged, and `hdu_wall_s > 1.5x total_wall_s`
  raises a disagreement warning. Those "after" artifacts were a mixed-provenance measurement,
  so the earlier round's headline perf numbers are re-measured rather than trusted.
- `cosmics` reports which astroscrappy pass failed instead of one generic message, and the
  faint-CR enhancement mode is validated before any detection so a bad config value raises
  instead of being swallowed. `objects` likewise names the failing SEP pass.
- Previously silent skip paths now warn: chip-replica position/shape mismatches, missing
  individual masks, an invalid global confidence p99, unavailable dark HDUs, unusable
  GAIN/RDNOISE headers, and a weight restore that fails after the mask was already rewritten.
- `WeightMaskProduct` now validates shape, dtype and range on construction.
- New `sep_background.edge_artifact_thresh` (default 50.0) replaces a hard-coded constant in
  the edge-artifact check, which also now uses `nanmedian` so NaNs cannot poison the result.

### Curated real-MegaCam label set (new)

Closes the gap the first two passes named: every detection threshold was still being tuned
against injected trails, which do not reproduce the clutter that drives them.

- Added `benchmarks/curate_trail_truth.py`, which labels real pixels from evidence the
  scored detector cannot produce on its own: four independent proposers, cross-CCD
  corroboration in the shared tangent plane of the MegaCam mosaic, then chip-replication,
  cross-exposure-persistence and axis vetoes, and finally an independent along-line ridge
  measurement in raw counts against a 25-60 px off-line background band.
- Committed `benchmarks/trail_truth/megacam_real_labels.json` (241 labelled real-detector
  regions over 4 exposures and 2 readout formats), its row-by-row review sheet, and
  `benchmarks/score_trail_truth.py` to score any detector against it.
- **Result: no confirmable real trail in the local MegaCam corpus.** Over ten exposures and
  364 HDUs every long bright line resolves to a chip-level defect: CCD `8351-11-4` has 3072
  of its 4644 rows above 20 sigma in one column (66 %), and every chip carries a bright band
  at `y ~ 4590` across the full width. The production detector's earlier positives on
  `1013719p` (30,416 px over 7 CCDs, 19,151 on one) are consistent with a single full-height
  4-px column band, not a trail. The committed set is therefore a false-positive benchmark.
- Measured with it: `houghpeaks` and the production `streaks` path each mask **30 of 241**
  labelled real-clutter regions (12.4 %, ~6 % of artefact pixels) -- mostly chip-fixed
  structure the cross-exposure layer proves is on the detector. The extra ~570 s of
  production work over the prescreen buys nothing on this axis. This is the first
  real-clutter streak measurement in the project; the default gate
  (`--max-artefact-fp-rate 0.10`) documents it as currently failed.
- Fixed three defects the curation exposed in its own tooling, each of which had produced
  convincing wrong labels first: the Radon proposer re-introduced the **mirror** bug in the
  line convention; `line_to_mosaic` canonicalised the line's normal sign but not its offset,
  so mirrored chips let two parallel lines on opposite sides of the mosaic share a
  representation; and the cross-CCD test initially produced sixteen bogus trail groups from
  pairs of bad columns.
- Added `tests/test_trail_truth.py` (22 tests): fixture invariants that make a label
  unaddable without evidence, plus unit tests for the mosaic-geometry and ridge helpers.

Second pass on the same data, verified end-to-end on one **complete 36-CCD MegaCam
exposure** (`1013719p`, 353 Mpix, 4 workers, `--all-products`): **1676 s -> 1180 s wall
(-29.6 %, 46.6 -> 32.8 s/CCD)**, streak stage 5702 -> 3819 CPU-s, and all **11 products x
36 CCDs pixel-identical** to the previous revision (407 HDU comparisons, same EXTNAME, same
dtype, 0 differing pixels) -- the speedups below cost nothing in output on real data.

- Fixed: the Radon rescue built the **mirror** of every candidate line. `radon`
  parameterises a line reflected relative to the Hesse convention
  `_candidate_from_rho_theta` assumes, and the rescue passed radon coordinates straight
  in, so at every oblique angle the strip refiner was handed a line that does not exist
  in the image (rejected as `no_support`). Only near theta = 0 do the two conventions
  agree, which is why the rescue could only ever propose the chip's near-vertical column
  artefacts. Injected-trail recall for this stage went from 0/4 cases to 4/4 (recall
  0.90/0.91/0.36/0.90 for 60/12/6/4 sigma trails); the conversion is pinned by a
  calibration test over eight (angle, offset) injections.
- Streaks: the Radon transform now pads to the true diagonal (`ceil(hypot(h, w))`) instead
  of skimage's `sqrt(2) * max(shape)` -- 5102 vs 6568 per side on a MegaCam CCD, ~40 %
  fewer pixels per angle.
- Streaks: peak significance is measured over the geometrically valid rho range only. The
  old per-column `mad_std` was taken over a column that is mostly zero padding (4628 of
  6568 entries at theta = 0), so its MAD was 0, a hard-coded `1.0` fallback took over, and
  every value in that column read as a million-sigma detection that no threshold could
  reject. A relative floor (`mrt_rescue_params.sigma_rel_floor`) replaces the fixed one.
- Streaks: new `mrt_rescue_params.sinogram_highpass` (default 101 samples) subtracts a
  moving median along rho before significance is measured, removing the broad star bumps
  that otherwise dominate the per-angle MAD (measured 180 -> 9 on a synthetic field, and a
  trail's own significance 51 -> 1468).
- Streaks: new `mrt_rescue_params.bin` (default 1) mean-bins the projection image; 4 gives
  ~1.6 s instead of ~20 s per CCD, with the measured per-case recall table in the config
  comment.
- Streaks: new `satdet_params.skip_when_prescreen_confirmed` (default true) skips the
  full-resolution multi-scale Canny/Hough sweep when the binned-Hough prescreen already
  accepted a trail -- **-24.5 s per CCD**, with bit-identical streak masks (0 differing
  pixels) on a clean field, a bright-trail field and a bright+faint field.
- Background RMS: the `inf` sentinel that `estimate_background` writes for unmeasurable
  pixels (7.1 % of pixels and 150 of 2112 columns on a real MegaCam HDU) now has one
  documented reading per consumer: detection thresholds treat those pixels as inert,
  rejection/quality gates substitute a robust value. This removes the four call sites that
  used `np.nanmedian` on a sentinel-bearing map, which silently flipped meaning once more
  than half a chip was unmeasured (`nanmedian` ignores NaN, not `inf`). SEP's own handling
  of `err=inf` was verified correct and pinned by a test.
- Fixed: a divide-by-zero warning from the `rms_map` variance method on degenerate input.
- Cosmics: new `cosmic_ray.single_pass` (default false) runs one loose L.A.Cosmic pass
  gated by the morphology filter instead of two passes. Measured on a real HDU it is 37 %
  faster with lower false positives but single-pixel completeness collapses from 0.207 to
  0.013, so it is off by default; the numbers are in the audit.
- Benchmarks: `streak_inject.py` is a swept grid with TSV output, `--mask-path`/`--seeds`
  and no hardcoded data paths; `cr_faint_curves.py` gained the same treatment plus the
  config variants the cosmics decision needed. A fast recall-floor / false-positive
  ceiling test (`tests/test_streak_recall_floor.py`) now runs in CI.

Throughput work on real MegaPrime MEFs (measured -32.8 % per-HDU wall time, products
byte-identical apart from the documented naming change).

- Streaks: the mask-independent preparation (15x15 median filter + disk(3) top-hat) is
  computed once per array and shared across the satdet prescreen, the corridor build and
  the unmasked retry, instead of up to four times.
- Streaks: `regionprops`-per-component edge pruning replaced by a vectorized reproduction
  of skimage's perimeter (12x faster per call, identical keep/drop decisions).
- Streaks: the Radon MRT rescue is now switchable via `streak_masking.mrt_rescue_params.enable`
  (~35 s and ~350 MB per 9.8 Mpix HDU); default `true` keeps existing behaviour.
- Flat bad-pixel masks are cached on disk per flat HDU (`flat_masking.bad_mask_cache`,
  `flat_masking.bad_mask_cache_dir`), removing ~20 s per HDU per exposure that shares a flat.
- MEF: each HDU's products are written and released as they are produced; peak memory no
  longer scales with the number of CCDs in the file.
- Output HDUs are named after the CCD identifier (`MAP_8341-7-5`) instead of the HDU index
  (`MAP_HDU1`), falling back to the old scheme when the header carries none.
- Fixed: faint-CR morphology gate used skimage's deprecated axis-length properties
  (removed in skimage 2.0).
- Fixed: fitsio was passed `compress="NOT_SET"` for uncompressed output, which it warns
  about and ignores.

## 0.1.0 - 2026-09-08

First tagged release of the classical weightmask pipeline (`pip install weightmask`).

- CLI: `weightmask` (MEF-capable) and `weightmask-reconstruct-sky`
- Products: weight or confidence, quality mask, inverse-variance, sky, optional per-contaminant masks
- Quality bits: BAD, SAT, CR, DETECTED, STREAK, INVALID_VARIANCE (`set_means_flagged`)
- Theoretical core plane is an F² sensitivity weight `g² F² / (S g + RN²)` (Poisson+RN at F=1). Canonical YAML also enables `flat_rel_noise` and `rescale_variance`.
- Sky mesh encode/decode uses the same clipped SEP node abscissae
- Config: canonical `weightmask.yml`; unknown top-level keys rejected; `WeightMapGenerator` fails closed

Known limitations: one gain/readnoise per HDU; real-data benchmark suites need external labels.
