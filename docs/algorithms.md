# Algorithms

weightmask flags contaminants on a detrended FITS image and attaches a
per-pixel weight. Use that weight for stacking or for single-exposure work
(shape measurement, forced photometry, profile fitting, difference imaging).
Processing is per HDU, in this order: bad pixels, saturation cores, a
preliminary sky (for bleed and CRs), bleed grow, cosmic rays, iterative sky and
objects, inverse variance, streaks, then weight and confidence. Config keys
live in the canonical [`weightmask.yml`](../weightmask.yml).

Quality bits use `set_means_flagged` polarity. `DETECTED` is informational by
default and does not zero weight. `BAD`, `SAT`, `CR`, `STREAK`,
`INVALID_VARIANCE`, and `NO_DATA` do.

## Bad pixels (`BAD`) and missing data (`NO_DATA`)

**What.** Dead or hot pixels, dead columns, blanked CCDs (`BAD`), and
non-finite science pixels and prescan/overscan regions outside `DATASEC` (`NO_DATA`). They have no usable
sky response and receive zero weight.

**Method.** On a real flat, a local median filter estimates the illumination.
Pixels whose ratio to that surface is outside `[local_low_thresh,
local_high_thresh]` are flagged. Optional column detection marks low-response
columns from the derivative of the column median. With no flat, the pipeline
uses `F = 1` and skips this flat-based `BAD` step. Pixels outside the header's
`DATASEC` bounding box (when smaller than the HDU array) are flagged as
`NO_DATA` (`detect_non_illuminated`), as are non-finite science values. Non-finite
flat pixels remain `BAD`, including entire invalid tiles.

These extra `BAD` sources run in the CLI/MEF path (`process_all_hdus`), not in
single-array `process_image`:

- `dead_ccd_*`: a CCD whose flat median is a MAD outlier versus its siblings
  is flagged entirely.
- `--dark_image`: dark values above the HDU median plus `hot_sigma` times
  the robust scatter (`1.4826 × MAD`), and non-finite dark values, become `BAD`
  using `dark_masking.hot_sigma` (default 8). A zero dark is valid. This criterion assumes
  a mostly healthy HDU with a uniform pedestal; structured darks need a
  calibrated per-amplifier or local baseline.
- `--badpix_mask`: keep-map (`0` = bad, `1` = good).

**Config.** `flat_masking`, `dark_masking`.

## Saturation and bleed (`SAT`)

**What.** Pixels at the analog full well, plus charge that bloomed along the
column.

**Method.** A guarded histogram clump on the high-ADU tail, then a plateau-tail
percentile if that clump fails the guard, then the first present header
keyword in `saturation.keyword` as an advisory fallback, then
`effective_full_scale` / `fallback_level`. Saturated cores
are grown up and down the CCD column until the data fall back toward sky, then
dilated horizontally (`bleed_grow_horizontal`).

**Config.** `saturation`.

## Sky

**What.** A smooth background map used for object detection, streak
detection, and the Poisson term in the weight. Optional compact mesh for
archive storage.

**Method.** Default is SEP's SExtractor-style mesh background
(`sep.Background`) with iterative object masking (`sep_background.iterations`);
zero skips the object-mask refinement while retaining preliminary and final sky
estimation.
The explicit median-filter method fills excluded pixels from their nearest
valid neighbors before filtering. Fallbacks when SEP cannot run: crowded frames (`mask_threshold` is a masked
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
contrast cut on connected components measured above the sky, and an optional fainter second pass
that keeps only elongated multi-pixel “worms”. `sigclip` is a dimensionless
standardized-residual threshold and remains fixed when image units or the ADU
background RMS change. The former raw-ADU RMS adjustment was removed because it
made the threshold more permissive under a pure unit rescaling. The canonical
8.5 is the historical shipped conservative setting; the scale-control evidence
establishes dimensional consistency, not that 8.5 is an optimized threshold.

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
two extractors (order does not matter; they OR together), then a
trail-aligned strip is refined on the full-resolution image:

- Binned Hough-peak search.
- Elongated contour morphology.
- Strip profile growth and geometric gates (`mask_params`).
- Optional sparse RANSAC on residual bright pixels for dashed trails.

Refits must retain the required along-trail support before a narrower strip can
replace the current fit. The supplied config groups Hough support across gaps
of at most 24 observed pixels and discards runs shorter than 8 pixels. Masked
strip samples are omitted from Hough run selection. After confirmation, the
fitted band crosses an interior source mask when the same candidate's surviving
support still passes the coherence and occupancy gates and brackets the mask,
and the science and RMS are finite there. Those pixels do not supply detection
evidence. Unknown-noise regions and unmasked gaps are not filled.
Contour candidates cannot bridge gaps; broad dashed-trail recovery remains a
known limitation.

Residual RANSAC searches continue after rejected clutter models. Only accepted
trails count toward `sparse_ransac_params.max_trails`, and consumed inliers are
removed before the next search. The supplied 128-pixel minimum length rejects a
persistent 116-pixel detector feature measured at the same coordinates in three
epochs; the two curated long linear features and the focused synthetic sparse
case are unchanged relative to the former 100-pixel floor.

The following removal record is historical engineering evidence, not current
release qualification. Two further extractors were removed after measurement,
not preference. A
multi-scale Canny/Hough stage accepted nothing on 56 of 56 real amps. An
angle-binned Radon rescue accepted on 3 of 83, all false positives, and changed
`recall_line` by +0.000 across 8 of 8 injected-trail cells spanning 4-12 sigma,
two lengths and two seeds, while costing 121.7 s/amp of a 125.7 s/amp stage.
Those measurements predate the 0.2.1 object-mask fixes. The frozen figures are
retained in [`detector_audit.md`](detector_audit.md) and the corresponding
historical changelog section; current benchmark commands are not reproductions
of that chronology.

Frangi-ridge comparison code is not in the package; it lives in
`benchmarks/frangi_legacy.py`.

**Config.** `streak_masking`.

## Inverse variance, weight, and confidence

**What.** A per-pixel weight and a normalized confidence map. The inverse-variance
FITS product is the same plane after sanitizing non-finite or non-positive
values (`INVALID_VARIANCE`).

**Method.** Default `variance.method: theoretical`, in ADU⁻²:

```
ivar = g² F² / (S g F + r²)
```

`S` is sky in ADU, `F` the flat, `g` gain in e⁻/ADU, `r` read noise in e⁻.
The sky term is `S g F` because a star sitting on a spatially varying flat
(`tests/test_photometry_bias.py`) shifts the weighted aperture by more than
that fixture's read-noise floor. At `F = 1` the two denominators agree.
That expression is the core plane. Canonical `weightmask.yml` then adds
`flat_rel_noise`, the dimensionless fractional flat uncertainty. It contributes
`(S g F · rel)²` before flat division, or `(S · rel)²` to the calibrated ADU²
variance, with `rel` increased where the flat is below its median.
`rescale_variance` then scales the plane so background SNR has robust standard
deviation 1. Omit those keys and the in-code fallbacks leave both off. All three
variance methods estimate background-only variance.

Weight is masked inverse variance. Confidence is that weight divided by its
configured percentile (default 99th), clipped to `[0, 1]` unless
`confidence_params.scale_to_100` is true. `normalize_scope: per_exposure` is
applied once to unclipped weights by the MEF CLI only when the primary map is confidence
(`output_map_format: confidence`); the canonical YAML writes weight, so that
rescale is a no-op. Single-array `process_image` does not apply it. The in-code
fallback is `per_hdu`. Exposure normalization samples at most 20,000 positive
weights per HDU; it approximates the exposure percentile and gives similar
influence to differently sized HDUs.

Gain and read noise are one scalar per HDU (first present header keyword,
recorded as `GAIN_SRC`) unless both `GAINA` and `GAINB` exist and a section
keyword splits the HDU. Then the variance plane uses that gain map.

**Config.** `variance`, `confidence_params`, `output_params`.

## Out of scope

Not modelled: persistence, CTI trails, IPC, amplifier crosstalk, ghosts,
scattered light as a separate class, fringing, correlated read-noise
templates, or a learned detector.

## References

- Bertin, E., & Arnouts, S. 1996, A&AS, 117, 393 (SExtractor background and
  extraction; SEP is a Python implementation of that model).
- Barbary, K. 2016, JOSS, 1, 58, [sep](https://github.com/kbarbary/sep).
- van Dokkum, P. G. 2001, PASP, 113, 1420 (L.A.Cosmic).
- McCully, C., et al., [astroscrappy](https://github.com/astropy/astroscrappy)
  (ASCL:1609.012).
- Duda, R. O., & Hart, P. E. 1972, Commun. ACM, 15, 11 (Hough transform).
- Lorensen, W. E., & Cline, H. E. 1987, SIGGRAPH, 21, 163 (marching contours;
  scikit-image `find_contours`).
- Fischler, M. A., & Bolles, R. C. 1981, Commun. ACM, 24, 381 (RANSAC).
- STScI `acstools.satdet` (HST/ACS satellite-trail tools; weightmask's Hough
  path is inspired by this style of detector, not a verbatim port).
