# Changelog

## Unreleased

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
