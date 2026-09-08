# Changelog

## 0.1.0 - 2026-09-08

First tagged release of the classical weightmask pipeline (`pip install weightmask`).

- CLI: `weightmask` (MEF-capable) and `weightmask-reconstruct-sky`
- Products: weight or confidence, quality mask, inverse-variance, sky, optional per-contaminant masks
- Quality bits: BAD, SAT, CR, DETECTED, STREAK, INVALID_VARIANCE (`set_means_flagged`)
- Theoretical core plane is an Elixir-style F² sensitivity weight `g² F² / (S g + RN²)` (Poisson+RN at F=1). Canonical YAML also enables `flat_rel_noise` and `rescale_variance`.
- Sky mesh encode/decode uses the same clipped SEP node abscissae
- Config: canonical `weightmask.yml`; unknown top-level keys rejected; `WeightMapGenerator` fails closed

Known limitations: one gain/readnoise per HDU; real-data benchmark suites need external labels.
