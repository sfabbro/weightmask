# weightmask — thermo-nuclear code-quality review, 2026-10-14

Whole repo at `67820bd`. Read-only; nothing fixed yet. Read with
`ponytail_audit_2026-09-27.md` (its Refuted list still stands).

## Verified by measurement

- **Dark frame strips flat bad pixels (science bug).** `mef._compute_one`
  passes `_merge_dark_mask(...)` as `precomputed_bad_mask`; `process_hdu`
  treats any non-None value as the complete bad mask and skips
  `compute_flat_bad_mask_cached` (`mef.py:76-86`, `:1232`, `:1150`, `:1258`).
  Repro: 100-px flat defect + 1 hot dark px -> 100 bad px without dark, 1 with
  dark, flat mask computed 0 times. `test_bad.py:303` passes on the bug
  (flat=None). Root cause: one parameter means dead-CCD veto, dark mask, and
  flat cache.
- **Code defaults diverge from the packaged YAML.** `cli.load_configuration`
  does not merge user config over `default_config_text()`; ~200 `.get(k, d)`
  sites carry their own defaults (e.g. `cosmics.py:301` sigclip 4.5 vs YAML
  8.5; streaks `strip_width` 96 vs 80; `config._cross_field_errors` a third
  copy). Only partial configs are affected.
- **Streak debug record is discarded.** `process.py:162` shallow-copies the
  section; `streaks.py:1594` writes `_last_run` into the copy.
- Root `weightmask.yml` is a byte copy of the packaged one, pinned equal by
  `test_release_machinery.py:596`; exists for `cli._find_default_config` cwd
  lookup and benchmark `open("weightmask.yml")` calls.
- Benchmarks with zero non-notes refs: `elixir_parity`, `race_poloka`,
  `canfar/{pod_build_env,probe_image_env,probe_pod_env,probe_dns_outage,
  submit_gate.sh,gate_one.sh}`, `tests/benchmarks/wm16_hough_starvation`.
  `probe_pod_env` calls a missing `pod_exec.sh`.

## Structural plan

1. DONE (2026-10-14): `process_hdu(extra_bad_mask=...)` always computes the
   cached flat mask and ORs dark/veto into `badpix_mask`; regression test
   `test_extra_bad_mask_adds_to_flat_and_keep_map_bad_pixels` (fails on old
   code). Degenerate flats (zeros/NaN) on vetoed CCDs do not raise. Still
   open: `process_image`'s own uncached flat-mask branch is now reached only
   by direct API callers; delete it when `process_image` takes `bad_mask` as
   given.
2. Merge user config over packaged YAML at load; stages index keys; delete
   inline defaults, re-validation inside stages, `_OBSOLETE_*` keys,
   single-value `streak_masking.mode`.
3. Stages return typed results; stop writing results into config dicts
   (`objects.py` `_elongated_*`, background `_diagnostics`); raise
   `StageFailure` instead of six-`None` tuples.
4. Split `mef.py` -> `mef` / `products` / `replicas`; split `streaks.py` ->
   `persistence` + `streaks/{geometry,refine,proposers,vetoes}`.
5. Delete dead benchmarks and CANFAR launcher generations A/B; move
   `tests/benchmarks` + `simulate_and_test.py` into `benchmarks/`; make
   `benchmarks` a package; drop it from ruff exclude.
6. Tests: add `testpaths`, a `conftest.py`, merge by module, drop
   text-grepping CI/doc tests, stop pinning private helpers.

## Outcome (2026-10-15, branch wip/thermo-cleanup)

Done: 1 (3d27bd6); 2 and 3 (f5fa2de; StageFailure plus raising
`process_image`/`process_hdu` in b9c7273); 4 for mef (products/replicas) and
the streak refiner dedupe; 5 (dead scripts deleted, suite/simulator moved to
`benchmarks/`, benchmarks linted); 6 `testpaths`. Golden: every label since p2
is bit-identical on syn/rep/trail; syn_partial differs from 3d27bd6 by design
(partial configs now get packaged defaults).

Refuted or not worth it after checking: "dead candidate fields" (they feed the
report and benchmarks); veto "label once" (each stage's mask is vetoed on its
own components, merging changes semantics); `_SEGMENT_SCORE` is a constant
term the confidence thresholds were tuned with; the text-reading CLI tests
guard against dead config keys and flags, so they stay.

Deferred: `streaks.py` package split (32 test/benchmark modules patch its
private names; 1.4k lines of one stage), `conftest.py` (no shared fixture
duplication worth it), merging test modules, print -> logging, satur
`effective_full_scale`, `_unbias_variance` naming, version bump (recommend
0.3.0: config semantics changed).
