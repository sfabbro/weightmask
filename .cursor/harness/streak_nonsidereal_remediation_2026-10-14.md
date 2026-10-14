# STREAK on non-sidereal frames — acceptance test and remediation plan (2026-10-14)

The `ns-review` campaign processed seven non-sidereal exposures and shipped
1,767,221 STREAK pixels in which the audit
(`release_remediation_plan_2026-10-06.md`, "Seven non-sidereal exposures processed
in pods") found no satellite trail: axis-aligned detector bands recurring at
identical coordinates across exposures, chains of near-collinear star trails at
the field's own trail angle with sky between their heads, and a few thick flat
bright features. This document is the plan, and the test that decides whether a
change to the plan worked.

## 1. The test

`benchmarks/streak_nonsidereal_acceptance.py` builds the frame the stage has
never been measured on and scores four injected classes:

    pixi run streak-nonsidereal                      # synthetic fixture, quick matrix
    pixi run python benchmarks/streak_nonsidereal_acceptance.py \
        --frame real --exposure <1851960p.fits.fz> --hdu 17 --crop 3700,400 --size 1024

- **positive** satellite trail: PSF-width, uniform along its length, at an angle
  the star-trail family does not share;
- **negative** star trail: the family's own angle and length, bright head, fading
  tail;
- **negative** chain: four star trails placed collinearly with gaps -- the
  geometry the campaign's diagonal groups actually had;
- **negative** column / row band: axis-aligned, constant width, full span, faint.

Inputs come from the production path (`capture_detector_inputs`), not from a bare
background pass, because the flat-less input path is part of the mechanism under
test. The control run -- a frame with *no* satellite injected -- is the number
that matters; a non-sidereal frame must produce an empty STREAK mask, and that is
what fails today. `tests/test_streak_nonsidereal_acceptance.py` pins the harness's
structure and its mask-classification helpers; it does not pin the 1024-pixel
baseline below, which is remediation work rather than suite runtime.

## 2. Baseline, measured 2026-10-14

| run | existing mask | **control footprint** | largest control component | injected satellites found |
|---|---|---|---|---|
| synthetic 1024^2, seed 0, no flat | -- | **4,526 px** | 3,906 px, 17 x 327 at y123 | 1/6 |
| real 1851960 HDU 17, y3700 x400, no flat | 56,520 px | **4,379 px** | 1,216 px, 117 x 124 | 4/6 |
| real 1851960 HDU 17, y3700 x400, with flat | 84,467 px | **0 px** | none | 2/6 |

Where the synthetic control's pixels sit: 3,452 px on the injected row bands,
189 on the column bands, 127 on star trails, 758 unattributed. The largest
component is the mask grown 17 rows deep around the 8-row band at y128, of which
2,568 px are exactly on the band -- across 327 of the frame's 1,024 columns, so
the stage accepted a third of a detector band as a trail, not the whole of it.
The other three bands (column 553 width 3, column 432 width 11, row 580 width 3)
were rejected; the 11-px-wide column was rejected by the width gate that the
8-px row passes. No negative was accepted as a body: the injected chain scored
0.00 coverage, the bands 0.00-0.03.

Three caveats, stated rather than buried. The synthetic fixture's noise is
Gaussian -- no Poisson, no cosmic rays -- and it has no flat, so it is the *weak*
input path by construction and it overstates the band class relative to a
production run with a flat. Each injection cell is one placement per seed, so
per-cell recall differences between rows of that table are placement luck, not
sensitivity: the full matrix is what would settle recall. And no genuine
satellite trail has been found in any of these seven fields, so recall on real
non-sidereal data remains unmeasured -- the injections are the only denominator
there is.

## 3. What the baseline changed about the plan

The three-way comparison is the whole story, and it moved the first item from
"write new vetoes" to "pass the flat that is already downloaded".

The campaign ran `process_exposure.py <obsid> --out ... --workers ...`
(`benchmarks/canfar/run_exposures.sh:68`) with no `--flat`. The wrapper that
stages each job *does* download a flat when the group lists one
(`run_one.sh:184-198`) and links it for the benchmark path
(`run_one.sh:231-241`); the campaign's command line simply never passes it, and
`process_exposure.py` documents that it applies no other prior either: "No
persistence prior is applied. That prior is defined across *other* science frames
of the same detectors in the same run, and one observation id has none"
(`benchmarks/canfar/process_exposure.py:32-34`).

On the one crop measured both ways, the flat changes the answer completely: the
upstream mask grows from 56,520 to 84,467 px (the chip-fixed structure the flat
identifies), and the streak stage's footprint on a frame containing no satellite
goes from 4,379 px to **zero**. Nothing in the stage changed between those two
runs.

## 4. Plan

**Status 2026-10-14, after P3 landed.** The chain class is fixed: the flat-less
`1851960` HDU 17 crop's STREAK footprint is 0 px on all three seeds with the veto
on, and nothing else moved (no recall cell, no sidereal mask). What that leaves is
the *band* class, which no star-trail veto can touch and which the crop cannot
show because its bands are not the ones the campaign's tables count. So P0 and P2
are still the open items, and the choice between them is now the only open
decision: mandate the flat (P0), or teach the stage to drop a full-span
axis-aligned band (P2) for the runs where no flat exists. P1 is what asks that
question out loud. P4 remains conditional, with the 2- and 3-segment chains --
the ones a length-only condition cannot separate -- as its first probe.

**P0 — re-run the campaign with the flat it already has.** No stage change. Add
`--flat <staged flat>` to the pod command in `run_exposures.sh:68`; the flat is
already downloaded and `process_exposure.py:199` already matches it to each
science HDU by index. Verified here: `flat_08Bm01_r.fits.fz`, the file the crate
stages, against HDU 17 of `1851960`. Cost is one campaign re-run -- 19,702 s of
processing across seven pods by the campaign's own `seconds` column -- plus the
one decision the wrapper already forces: `run_one.sh:111-112` refuses more than
one flat without an explicit per-exposure mapping, so a group with several flats
needs that mapping written down.

**P1 — make a flat-less science run a decision rather than an accident.** The
same frame identifies itself: 56,520 masked pixels without a flat against 84,467
with one, both recorded in `run.json`. Either `process_exposure.py` refuses to
process science frames without a flat unless `--allow-no-flat` is given, or the
flat-less path is treated as supported and P2/P3 become its defence. Decide
before P2, because that decision is what justifies the vetoes.

**P2 — axis-aligned span veto (only if P1 keeps the flat-less path alive).**
Insertion point is the single negative veto every source's mask already passes
through: the `_veto` closure in `detect_streaks` (`weightmask/streaks.py:1242`),
which composes `_drop_bright_components` and `_drop_premasked_components` and is
applied to the Hough, contour, RANSAC and prediction masks alike. A new
`_drop_axis_aligned_components(mask, shape, max_width, min_span_fraction)` drops a
component whose bounding box spans most of one axis at a narrow transverse
extent. It needs its own knob rather than `max_support_width: 12`, for the reason
already documented: that gate measures the transverse MAD of the refined strip,
which for the campaign's 10-px bands is ~4.5 px, and it runs after the mask is
built where the defect's full span is visible. Measured target: the synthetic
control's 4,526 px, of which 3,641 sit on or within 2 px of an axis-aligned band.

**P3 — star-trail family veto. DONE 2026-10-14.** This is the class a flat
cannot fix: chains of near-collinear star trails at the field's own trail angle,
10-36% of whose masked pixels are above sky, against 86% for the one genuinely
continuous feature in the campaign. It landed as
`frame_trail_family(data_sub, rms, existing, config)` plus
`_drop_star_trail_family_components(mask, family, config)`, composed into the
same `_veto` closure P2 would use, with the estimate recorded in `debug_info`
(`star_trail_family`, `star_trail_dropped`) and printed per HDU.

Two conditions, not three: orientation within `angle_tol_deg` of the *frame's own*
trail angle, and a span no longer than `length_factor` times the family's median
length. The length condition is the one that protects a satellite flying parallel
to the field's motion (`1774908` HDU 19 was 1.8 degrees off), because a chain of
star trails cannot outgrow the trails it is made of while a satellite's length
comes from its own angular rate. The third condition in the draft -- brightness
relative to the family's heads -- was dropped rather than shipped unmeasured; if
P2/P3 leave a residue, that is P4's specification.

The family is the *dominant orientation mode*, not a frame-wide spread: elongated
sources are binned in overlapping 10-degree windows (`mode_window_deg`) and the
densest window that holds `min_sources` members wins. This was measured, not
designed in the abstract. A first version took the MAD of every elongated source
and failed open on a 1024-crop of `1851960` HDU 17: 26 field trails plus the
harness's six injected satellites pushed the spread past its limit and the family
disappeared *exactly when a satellite was present*. The mode keeps the family
while foreign angles stay outside the window, and it is what keeps a sidereal
frame inert, since cosmic rays, bleed and galaxies scatter over every orientation
and no narrow window holds 20 of them.

Measured on the full acceptance matrix (24 injected cells per frame per state,
three seeds, `pixi run python benchmarks/streak_nonsidereal_acceptance.py` with and
without `--set streak_masking.star_trail_veto.enable=false`):

| frame | control, veto off | control, veto on | injected cells found, off -> on |
|---|---|---|---|
| synthetic 1024^2 (no flat) | 4,526 / 21,229 / 6,430 px | 4,203 / 21,229 / 6,044 px | 3/72 -> 3/72 |
| real `1851960` HDU 17 crop, no flat | 4,379 / 4,379 / 4,379 px | **0 / 0 / 0 px** | 5/72 -> 5/72 |
| real `1851960` HDU 17 crop, with flat | 0 px | 0 px | 1/24 -> 1/24 |

The veto removes every one of the 13,137 flat-less control pixels across the three
seeds and changes no injected cell at any sigma or offset, including the
10-degree-offset cells, which is where the design tradeoff would show up first.
On the synthetic fixture it removes 2% (323 px) because that fixture's dominant
control class is the axis-aligned band, which is P2's class and deliberately not
this item's: the fixture supplies no flat, so its bands reach the stage.
The residual recall numbers are the honest state of the positive side: 0 of 24
cells at 4 sigma on both frames, and the same before this change as after it.

Cost and cross-check, on the full (4644 x 2112) HDU rather than a crop: the
estimate takes **0.58 s** against a stage that runs 1-4 s/HDU, and it reproduces
the audit's independent measurement of the same frame -- 40.76 deg and 55.06 px
over 248 of 270 elongated sources, against the audit's 40.2 deg, median 55 px,
measured by a separate payload with its own source criterion. Two methods, one
number, which is the only kind of agreement that makes a per-frame prior
worth trusting.

**P4 — brightness continuity.** Only if P2/P3 leave a residue. A "trail must be
bright along its length" gate is measurable (fraction of the fitted line above
sky, in windows) but it risks a real, dashed trail; it must not be built before
there is a false positive left that needs it. Concretely, the residue to look for
is a chain of two or three star trails: its span is 2-3 times the family length,
inside the `length_factor: 4.0` band, so the length condition that catches the
campaign's four-head chain (5.8x) does not catch it. The acceptance harness's
chain fixture is four segments; parameterising it to 2 and 3 is the honest first
measurement, before any code.

**P5 — the acceptance test becomes the regression gate.** Before any stage
change: run the full matrix (both frames, 4 offsets x 2 lengths x 3 sigmas x 3
seeds) and freeze the report. Every stage change must then (a) leave
`pixi run streak-recall-floor` and `pixi run science-gate` unchanged -- the
existing recall floors must never fall to buy a cleaner control -- (b) hold the
control footprint at 0 on a flat-corrected real crop, (c) reduce the synthetic
flat-less control, and (d) keep injected-trail recall at or above the frozen
baseline. The P3 matrix above is that frozen baseline for the non-sidereal side;
`tests/test_star_trail_family.py` holds the parts of it that belong in CI, and
`--set streak_masking.star_trail_veto.enable=false` reproduces every pre-P3
number on the same code.

**P6 — the product stance for what has already shipped.** Until P0 runs, the
campaign's STREAK numbers are a false-positive census. Either re-run and
re-publish under P0, or label the STREAK product on these fields as trail
*candidates* in the campaign record. That is a decision about published artifacts
and belongs to the owner of the campaign, not to this plan.

## 5. Verification protocol

1. P0 first, because it is cheap, reversible and needs no code: one exposure
   processed with and without the flat, per-HDU STREAK counts compared. If the
   bands survive the flat on any CCD, the residue is P2's specification.
2. Only then the stage items, in the order P2, P3, P4, each with the P5 gates
   green before the next one starts.
3. A stage change that reduces the control footprint but moves a recall floor is
   a regression, not a fix; the floors exist for exactly this.

## 6. Open questions this plan cannot answer from the data

- Does the flat remove the bands on **all** detectors, or only on the CCD probed
  here (HDU 17 of `1851960`, the one non-sidereal HDU whose chain geometry was
  measured in detail)? One crop is one crop.
- Is `flat_08Bm01_r.fits.fz` the right flat for all seven exposures, or do some
  groups need a different one? `run_one.sh` stages flats per group, and the
  campaign's groups are visible in its staging records.
- ~~What happens on a frame where the field's trails and a satellite share a
  direction? P3's length condition is a design choice; no measurement yet exists
  to set it.~~ **Answered, §7.**

## 7. P3's length window, measured 2026-10-14

`benchmarks/star_trail_veto_window.py` (`pixi run streak-veto-window`), 1024-px
synthetic non-sidereal fixture, 60 trailed stars, 55 px star trails through the
production input path. The family is estimated from the frame and comes back at
40.07 deg / 47.43 px / 59 of 86 elongated sources, so a rendered 55 px trail
measures 0.86 of itself -- the window is quoted in *measured* family lengths and
the geometric lengths are alongside.

Both classes were swept at the family's own angle, which is where the length
condition is the only thing deciding:

| cell | geometric | measured | outcome |
|---|---|---|---|
| parallel satellite, 2.0x | 110 px | 2.32x | DROPPED, whole component (375 px) |
| parallel satellite, 3.0x | 165 px | 3.48x | DROPPED |
| parallel satellite, 3.5x | 192 px | 4.06x | kept |
| parallel satellite, 8.0x | 440 px | 9.28x | kept |

Bisecting the bracket puts the decision flip at **189.0 px = 3.985x the measured
family length**, which is the configured `length_factor` 4.0 to within 0.4%: the
knob sets exactly the boundary it claims to. Length follows the trail's own
angular rate, so a satellite flying parallel to the field's motion inside ~3.4
star-trail lengths is a false negative by design, and `length_factor` is the dial
for it. The angle tolerance measures 9.13 deg inside and 11.14 deg outside its
10 deg setting.

The chain side, which P4 called the honest first measurement (2 to 12 segments,
gap factors 1.0, 1.6 and 2.5 star-trail lengths):

- **Gapped chains are dropped at every span measured, to 21.5x the family
  length.** Each gap leaves a separate component, every one of them measuring
  1.14-1.17x, so the veto removes all of it -- and on the 8-segment/1.6L case
  through the stage, the 778 px of chain in the mask falls to 0 and the frame's
  total STREAK footprint goes 14,123 -> 7,370 px.
- **A contiguous chain is not.** With the segments touching it is one component
  and grows without limit, so it is dropped inside the same window and survives
  outside it: 3.47x dropped, 4.64x kept. Length alone cannot separate a touching
  chain from the parallel satellite the window protects, so P3's docstring
  argument -- "a chain of star trails cannot outgrow the trails it is made of" --
  holds for the gapped geometry only. That is the limitation now recorded in the
  CHANGELOG and in the config comment, not a defect to fix by turning the dial:
  raising `length_factor` buys contiguous-chain coverage with parallel-satellite
  recall on a one-for-one basis.
- The injected parallel trail does not move the family it is judged against
  (47.20-47.31 px, angle within 0.02 deg, at 2x, 4x and 8x), so the comparison
  cannot close on the object it is meant to pass.

Still open, and not answerable locally. The campaign's HDU 17 chain was recorded
as a 931 px *contiguous* component against a 55 px family, i.e. ~20x -- outside
the window -- so if that line reached the veto as one component, the length
condition did not remove it, and the recorded 4,379 -> 0 px drop on that crop
came from the other components the veto does catch. The competing reading is that
the 4,379 px footprint was mostly per-segment trails (each ~1.15x, dropped) and
detector bands, and the 931 px figure describes the mask *before* the fix rather
than anything that survived into that number. Both readings fit the arithmetic
above; only the crop separates them, the flat-less numbers come from the
campaign's pods, and the real exposure is not on this machine. This belongs with
§6's first question: re-measure it where the data is.
