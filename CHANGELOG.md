# Changelog

## Unreleased

### Correctness and performance audit (bugs found and fixed)

Continuation of the audit below, covering the MEF/CLI input layer:

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
