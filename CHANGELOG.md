# Changelog

## Unreleased

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
