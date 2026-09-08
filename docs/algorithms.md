# Algorithms

weightmask flags contaminants on a detrended FITS image and attaches a
per-pixel weight. Use that weight for stacking or for single-exposure work
(shape measurement, forced photometry, profile fitting, difference imaging).
Processing is per HDU, in this order: bad pixels, saturation cores, a
preliminary sky (for bleed and CRs), bleed grow, cosmic rays, iterative sky and
objects, inverse variance, streaks, then weight and confidence. Config keys
live in [`weightmask.yml`](../weightmask.yml).

Quality bits use `set_means_flagged` polarity. `DETECTED` is informational by
default and does not zero weight. `BAD`, `SAT`, `CR`, `STREAK`, and
`INVALID_VARIANCE` do.

## Bad pixels (`BAD`)

**What.** Dead or hot pixels, dead columns, and blanked CCDs. They have no
usable response and receive zero weight.

**Method.** On a real flat, a local median filter estimates the illumination.
Pixels whose ratio to that surface is outside `[local_low_thresh,
local_high_thresh]` are flagged. Optional column detection marks low-response
columns from the derivative of the column median. With no flat, the pipeline
uses `F = 1` and skips this flat-based `BAD` step.

These extra `BAD` sources run in the CLI/MEF path only, not in
`WeightMapGenerator.process`:

- `dead_ccd_*`: a CCD whose flat median is a MAD outlier versus its siblings
  is flagged entirely.
- `--dark_image`: same local/column logic on a dark, OR'd into `BAD` if
  `dark_masking` is in the config (canonical YAML includes it).
- `--badpix_mask`: Elixir keep-map (`0` = bad, `1` = good).

**Config.** `flat_masking`, `dark_masking`.

## Saturation and bleed (`SAT`)

**What.** Pixels at the analog full well, plus charge that bloomed along the
column.

**Method.** A guarded histogram clump on the high-ADU tail, then a plateau-tail
percentile if that clump fails the guard, then the first present header
keyword in `saturation.keyword` as an advisory fallback, then
`effective_full_scale` / `fallback_level`. `saturation.method` accepts
`histogram` or `header` but does not skip the histogram path. Saturated cores
are grown up and down the CCD column until the data fall back toward sky, then
dilated horizontally (`bleed_grow_horizontal`).

**Config.** `saturation`.

## Sky

**What.** A smooth background map used for object detection, streak
detection, and the Poisson term in the weight. Optional compact mesh for
archive storage.

**Method.** Default is SEP's SExtractor-style mesh background
(`sep.Background`) with iterative object masking (`sep_background.iterations`).
Fallbacks when SEP cannot run: crowded frames (`mask_threshold` is a masked
*fraction*, not an object cut) switch to global SEP; failed mesh retries go
to `robust_median_fallback`, then optional `smooth_surface`. `median_filter`
is an explicit method, not a fallback rung. Negative interpolation overshoots
next to bright masks can be filled from neighbors when the science pixels
agree with the fill (`dip_repair_*`).

Mesh codec: `n = (size - 1) // box + 1` nodes at
`clip(rint((k + 0.5) * box))`.
Rebuild with natural cubic splines (`weightmask-reconstruct-sky`). This
matches SEP's node phase, not SEP's C bicubic interpolant.

**Config.** `sep_background`, `output_params.sky_format`.

## Cosmic rays (`CR`)

**What.** Compact, sharp hits from ionizing particles. They get zero weight.

**Method.** [L.A.Cosmic](#references) Laplacian detection via astroscrappy,
with a PSF-peakiness gate so stellar cores are not taken as CRs, a size and
contrast cut on connected components, and an optional fainter second pass
that keeps only elongated multi-pixel “worms”.

**Config.** `cosmic_ray`.

## Detected objects (`DETECTED`)

**What.** Stars and galaxies. The bit is kept in the mask for downstream
photometry; it does not zero the weight unless
`output_params.mask_detected_in_weight` is true.

**Method.** SEP `extract` on the background-subtracted image, ellipse masks
scaled by object flux (halo), a small binary dilation, and optional
axis-aligned diffraction-spike bars on the brightest compact sources.
Highly elongated detections can be withheld from `DETECTED` and handed to
the streak stage.

**Config.** `sep_objects`.

## Streaks (`STREAK`)

**What.** Linear trails (satellites, aircraft, meteors). Zero weight.

**Method.** Production mode is `auto_ground` only. Candidates are the union of
three extractors (order does not matter; they OR together), then a
trail-aligned strip is refined on the full-resolution image:

- Binned Hough-peak search.
- Elongated contour morphology.
- Multi-scale Canny edges and a probabilistic Hough transform (ACS
  SATDET-inspired; not a bit-identical port).
- Strip profile growth and geometric gates (`mask_params`).
- If primary confidence is low, a Radon-transform peak search (MRT-like
  rescue).
- Optional sparse RANSAC on residual bright pixels for dashed trails.

Frangi-ridge comparison code is not in the package; it lives in
`benchmarks/frangi_legacy.py`.

**Config.** `streak_masking`.

## Inverse variance, weight, and confidence

**What.** A per-pixel weight and a normalized confidence map. The inverse-variance
FITS product is the same plane after sanitizing non-finite or non-positive
values (`INVALID_VARIANCE`).

**Method.** Default `variance.method: theoretical`, in ADU⁻²:

```
ivar = g² F² / (S g + r²)
```

`S` is sky in ADU, `F` the flat, `g` gain in e⁻/ADU, `r` read noise in e⁻.
At `F = 1` this is Poisson plus read noise. At `F ≠ 1` it is an Elixir-style
F² sensitivity weight, not the flat-fielded identity `g² F² / (S g F + r²)`.
That expression is the core plane. Canonical `weightmask.yml` then adds
`flat_rel_noise` (`(S g · rel)²` in the electron denominator, with `rel`
increased where the flat is below its median) and `rescale_variance` (scale
so background SNR has robust standard deviation 1). Omit those keys and the
in-code fallbacks leave both off.

Weight is masked inverse variance. Confidence is that weight divided by its
configured percentile (default 99th), clipped to `[0, 1]` unless
`confidence_params.scale_to_100` is true. `normalize_scope: per_exposure` is
applied by the MEF CLI only when the primary map is confidence
(`output_map_format: confidence`); the canonical YAML writes weight, so that
rescale is a no-op. `WeightMapGenerator` does not apply it. The in-code
fallback is `per_hdu`.

Gain and read noise are one scalar per HDU (first present header keyword).
Dual-amp `GAINA`/`GAINB` are not split.

**Config.** `variance`, `confidence_params`, `output_params`.

## Out of scope in 0.1

Not modelled: persistence, CTI trails, IPC, amplifier crosstalk, ghosts,
scattered light as a separate class, fringing, correlated read-noise
templates, or a learned detector. Future work on calibration-conditioned and
probabilistic products is sketched in
[`research/nextgen_weightmask.md`](research/nextgen_weightmask.md).

## References

- Bertin, E., & Arnouts, S. 1996, A&AS, 117, 393 (SExtractor background and
  extraction; SEP is a Python implementation of that model).
- Barbary, K. 2016, JOSS, 1, 58, [sep](https://github.com/kbarbary/sep).
- van Dokkum, P. G. 2001, PASP, 113, 1420 (L.A.Cosmic).
- McCully, C., et al., [astroscrappy](https://github.com/astropy/astroscrappy)
  (ASCL:1609.012).
- Canny, J. 1986, IEEE Trans. Pattern Anal. Mach. Intell., 8, 679.
- Duda, R. O., & Hart, P. E. 1972, Commun. ACM, 15, 11 (Hough transform).
- Galambos, C., Kittler, J., & Matas, J. 1999, CAIP (probabilistic Hough;
  scikit-image `probabilistic_hough_line`).
- Fischler, M. A., & Bolles, R. C. 1981, Commun. ACM, 24, 381 (RANSAC).
- Magnier, E. A., & Cuillandre, J.-C. 2004, PASP, 116, 449 (CFHT Elixir
  detrending and weight maps).
- STScI `acstools.satdet` (HST/ACS satellite-trail tools; weightmask's Hough
  path is inspired by this style of detector, not a verbatim port).
