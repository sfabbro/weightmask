# The CR component filters — what room there actually is

> **DO NOT IMPLEMENT — measured to be worth ~nothing.** The fresh profile
> (`cprofile_hdu_fresh.txt`, 2026-09-27) shows `detect_cosmic_rays` is 26.6s
> *tottime*, i.e. astroscrappy's C code, and neither `_post_filter_components`,
> `_filter_faint_components` nor `regionprops` reaches the top 35. The
> per-component cost measured below is real; there simply are not enough
> components for it to matter. Kept as a reference in case the CR stage grows,
> and because the exactness argument is reusable. The only real CR lever is
> the science knobs (`sigclip`, `niter`, `min_component_area`).
>
> Scope: `weightmask/cosmics.py` `_post_filter_components` and
> `_filter_faint_components`. `streaks.py` is out of scope this round by decision.

## Why this is the target

Both functions are the same shape: label the CR mask, then loop
`for region in regionprops(labeled, intensity_image=sci_data)` and apply scalar
gates per component. Every gate is a per-label reduction, so the whole loop is a
missed vectorization.

Measured on a synthetic CR-like mask (1940-3256 components), cost of building
regionprops objects and touching properties:

| access pattern | time | delta |
| --- | --- | --- |
| `region.area` | 187 ms | — |
| `+ region.coords` | 288 ms | +0.03 ms/comp |
| `+ axis_major_length, axis_minor_length` | 2453 ms | **+0.67 ms/comp** |
| `.perimeter` alone (not used by this code) | 718 ms | +0.22 ms/comp |

## Why the axis lengths are so expensive (skimage 0.26)

`region.axis_major_length` = `4*sqrt(l1)`, `axis_minor_length` = `4*sqrt(l2)`, where
`l1,l2` are eigenvalues of the region's inertia tensor. Per component that runs:

1. `moments_central(image, order=3)` — two `einsum(optimize='greedy')` calls, i.e.
   a **path-planning cost per component**, and order 3 when inertia needs only 2.
2. `np.linalg.eigvalsh(T)` — a **LAPACK dispatch per component** on a 2x2 matrix.
3. a Python `sorted()` for the descending order.

So ~0.67 ms is almost entirely per-component framework overhead, not arithmetic.
Note this also corrects the Sep-9 profile, which blamed 139k `perimeter` calls on
this code: in skimage 0.26 the axis lengths come from the inertia tensor, and
`perimeter` is never touched here (verified by counting accesses: 0).

## The option space

Ranked by (measured or derivable win) / risk. A = trivial and exact,
D = the real fix.

### A. Drop the unused `intensity_image`
`_post_filter_components` and `_filter_faint_components` pass
`intensity_image=sci_data` but never read skimage's intensity statistics — they
compute their own SNR from `coords`. Passing it makes skimage compute per-region
intensity stats that are then discarded. Removing the argument is a strict
reduction in work with no effect on any returned value.

### B. Replace the per-region SNR with one segmented max
Both loops do, per region:
`np.nanmax(sci_data[coords] / np.maximum(safe_rms[coords], 1e-6))`.
That is a tiny numpy call (several µs of dispatch) repeated once per component —
tens of thousands of times. The identical quantity for all regions at once is
`scipy.ndimage.maximum(ratio_image, labels=labeled, index=range(1, n+1))`, one C
call. Needs care to reproduce the `nanmax` (skip-NaN) and `1e-6` floor semantics
exactly.

### C. Compare the eigenvalue ratio, not the axis lengths
The caller only wants `major / minor`, and `major/minor == sqrt(l1/l2)`. So the two
`sqrt`s and the `4*` factors are round-trips that can be dropped, and the gate
becomes `l1 >= min_elongation**2 * l2`. Strictly fewer float ops and no
sqrt-domain edge cases. `_axis_lengths` would stop returning lengths nobody uses.

### D. Full vectorization, with exact re-measure near the cut  ← recommended
Compute per-label raw moments with `np.bincount` over `np.nonzero(labelled)`,
convert to central moments with the same algebra skimage uses, build the 2x2
inertia tensor, and take its eigenvalues in closed form:
`half ± sqrt(((a-d)/2)^2 + b^2)`. No per-component Python object, no einsum path
planning, no LAPACK call.

Pixel coordinates are integers, so the raw sums are exact in float64 and the
inertia entries are bit-comparable to skimage's. Only the eigenvalue step differs,
by the usual last-ulp from solving a 2x2 in closed form rather than via LAPACK.

**Validated on 1940 components: max relative difference 5.65e-11, 100% of
components within 1e-9, and exactly 1 of 1940 keep/drop decisions flipped at
`min_elongation = 2.0`.**

So wrap it in the idiom already proven in this repo (`streaks._prune_small_edges`
+ `_PERIMETER_ORDER_EPS`): decide with the fast path, and for any component whose
ratio sits within a relative epsilon of the threshold, fall back to skimage for
that component only. With a band of 1e-6 — three orders above the observed
5.65e-11 — the decision becomes *exactly* skimage's, and the band will contain
essentially nothing.

### E. Cheaper still: bound elongation from area and perimeter
Elongation is bounded by area and perimeter alone (Cauchy-Schwarz on the region
boundary). `area` is a `bincount`; `perimeter` for all labels at once is
`streaks._region_perimeters`, which already exists and is already validated
against skimage. Components whose bound interval excludes `min_elongation` are
decided with no eigenvalue math at all. Strictly less work than D and composable
with it, at the cost of a second bound to reason about. Worth doing only if D
lands and the stage is still hot.

### F. Move a science knob, not code
Component count is set by `sigclip`, `niter` and `min_component_area`. Loosening
any of them makes everything downstream faster — and changes what gets detected.
This is likely the single largest lever and it is a *science* decision, not an
engineering one. The useful engineering contribution is to measure the
time/recall curve so the choice is informed; the choice itself belongs to
whoever owns the CR science requirements.

### G. Ruled out
- `regionprops(..., cache=True)`: each region is touched once, so there is no
  reuse to cache.
- Parallelising the loop: `label()` is global, so the component decomposition
  can't be split without a different labelling strategy.
- Replacing croc/moment axis lengths with a different elongation *definition*:
  fast, but it silently changes which components qualify. Only defensible as an
  explicit science change, not as an optimization.

## Recommended sequencing

1. A, B, C together — small, local, exact, and they also collapse two
   near-duplicate loops into one shared "component statistics" helper.
2. D behind the epsilon re-measure band, with a test asserting the keep/drop set
   is identical to skimage's on a fixture with components straddling the
   threshold.
3. Re-measure. Only then consider E.
4. F as a separate, explicitly scientific discussion with the curve attached.

## Verification bar

`pixi run lint`, `pixi run test` (and do not edit while it runs --
`test_parallel.py` re-reads `weightmask.yml`), plus `pixi run science-gate` and a
MegaCam product-identity comparison, since this touches which pixels are masked.
