# Changelog

## 0.1.0 - 2026-09-07

First tagged release of the classical WeightMask pipeline.

- CLI: `weightmask` (MEF-capable) and `weightmask-reconstruct-sky`
- Products: weight or confidence, quality mask, inverse-variance, sky, optional per-contaminant masks
- Quality bits: BAD, SAT, CR, DETECTED, STREAK, INVALID_VARIANCE (`set_means_flagged`)
- Theoretical plane is an Elixir-style F² coadd weight `g² F² / (S g + RN²)` (Poisson+RN at F=1)
- Sky mesh encode/decode uses the same clipped SEP node abscissae
- Config: canonical `weightmask.yml`; unknown top-level keys rejected; `WeightMapGenerator` fails closed

Known limitations: one gain/readnoise per HDU; real-data benchmark suites need external labels; PyPI upload is not part of this tag.
