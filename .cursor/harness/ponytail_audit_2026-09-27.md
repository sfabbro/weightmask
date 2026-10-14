# weightmask — ponytail-audit round, 2026-09-27

Shipped in `a086557` (298 passed). Scope: every module except `streaks.py`.
Read alongside `perf_plan_2026-09-27.md` (the performance round) and
`cr_filter_options.md` (closed, do not implement).

## Fixed

1. **`objects.py` — `UnboundLocalError` on every zero-detection image.** Mine,
   from the previous round: I dedented `return obj_add_mask` out of the
   `keep_objects` branch and left the assignment inside. Zero detections →
   unbound local → swallowed by the broad handler → all-False mask plus a
   spurious `ERROR`. The mask was always correct, which is exactly why
   `test_detect_objects_empty_input` passed throughout: it asserts the mask,
   not the log. **Lesson: a test can cover the path and still miss the bug if
   it asserts the return value but not the side effect.**

2. **`mef.py` — short flat MEF silently degraded to a unit flat.** Every
   affected CCD got unflat-fielded weights, one INFO line, exit 0. Now fails
   the HDU. This exposed a flaw in an existing test: science written with an
   empty primary (data at 1,2) against a single-image flat (data at 0) had been
   passing *because* of the bug. **Lesson: a green test can be depending on the
   defect it fails to notice.**

3. **`utils.py` — `"--5"` aborted config loading.** Mine, from the previous
   round: `lstrip("+-").isdigit()` accepts it, then `int()` raises out of
   `clean_config_dict`. **Lesson: two rounds of tightening a parser and it
   still crashed on malformed input.**

Two of the three were regressions from the immediately preceding round, found
only because this round scrutinised that diff specifically. Nothing flagged
them by failing.

## Refuted — do not re-audit (measured, not reasoned)

- **"`ellipse_k: 3.0` makes the DETECTED mask effectively empty"** (rated HIGH,
  with SEP `analyse.c` geometry worked out). A bright round star gives **629**
  DETECTED px at 3.0 vs 308 at 2.0. More, not fewer. Seed-pass geometry holds.
- **`amplifier_gain_map` broken for `fitsio.FITSHDR`.** `FITSHDR` has `.get`,
  and the per-amplifier map is correct. I got this wrong twice before
  verifying (wrong signature; then a transposed FITS `[x1:x2,y1:y2]` section —
  the code is right and my test string was not).
- **`bad.py` vignetting false positives.** 0.00% on mild and strong vignettes,
  1 bad pixel found, 1 false positive. An earlier apparent 14.6% was my own
  synthetic flat going negative.
- **`_bounded_percentile` / `_rescale_variance_robust` stride cliff.** The
  arithmetic is real (`size=100000` → `step=1`, `100001` → `step=2`) but both
  are approximations with documented ceilings, not defects.

Three auditors each produced a "high severity" claim that measurement
demolished. Two of the three I initially believed. **Trusting any of them would
have meant "fixing" working code.**

## Deferred, with reasons

- **`_unbias_variance` disagrees with LSST by a factor of `gain`.** Code
  subtracts `signal/gain`; LSST's ADU² convention gives `signal/gain²`; the
  suggested fix `signal*gain` is a *third* value. My first check was circular
  (`sig/g ≡ sig·g/g²` reproduced the code). The frozen formula's own units are
  ambiguous — `S·g` is e⁻ but `r²` is e⁻². It is **off by default** and
  labelled "frozen 0.1", so guessing is worse than the bug. **Needs the science
  owner, not an engineer.**
- **`satur` `effective_full_scale` can land below `data_max`** (40000 vs
  45000), contradicting its own comment. Mechanism confirmed; I could not
  demonstrate wrong output — harm needs data in the gap band. Not fixed on
  speculation.
- **`bad.py` cache-write failure is silent.** Reproduced: an unwritable cache
  dir (read-only VOSpace, the CANFAR target) yields no output at all, so a
  ~25 min/run tax is invisible. Cheap fix, not yet applied.
- **`mef.py` flat guard reason string.** The failure is loud and no products
  are written, but the reported reason is a downstream `TypeError` rather than
  the intended message.
- **`print()` → `logging`** (previously D8). Unchanged; still a design decision.
  Note the false `ERROR` from finding 1 is an argument *for* it.

## Performance: nothing left outside streaks

Measured with a full stage table (no cProfile top-N cutoff — that once made a
1.3 s/amp stage look absent). With streaks off, of 58.9s: cosmics 14.0s,
bgobj_loop 2.1s, everything else under 1s each, `bad_flat` 41s **cold** and
0.002s **warm** (the per-flat disk cache is a ~20,000x win and does hit).

`cosmics` is astroscrappy's C; its only lever is `sigclip`/`niter`, which is a
science decision. So: no code-level speedup remains outside `streaks.py`.
