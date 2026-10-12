# weightmask — release remediation plan (2026-10-06)

Status: **hold — remediated work committed to a PR; 0.2.1 not released**

## Re-measurement 2026-10-12 on `b2e4da6` (PR head, clean tree)

Nothing below changes the hold. It replaces estimates with measurements taken on
the exact PR head commit, and records the release-path state that the earlier
entries assumed.

- **WM-16 still red, floors untouched.** `python -m tests.benchmarks.run --suite
  synthetic_v2 --with-baselines` exited `1`: `average streak F1 0.149 < 0.200`.
  All five per-case gates report `ok`; four cases contribute streak F1 `0.000`,
  so the suite aggregate rests on `synthetic_sparse` alone. Evidence:
  `test_outputs/benchmarks/synthetic_v2/summary.md`.
- **Science gate re-run at HEAD completed: still `failed`, and the stale-evidence
  problem is closed.** The run finished `2026-10-12T17:23:05Z` and wrote
  `test_outputs/harness/science-gate.json` with `commit_sha=b2e4da6` (HEAD),
  `version 0.2.1`, `status: failed`, `real_trail_recall: "n/a"`, and
  `scope_exclusions: ["real_trail_recall"]`. That is a real qualification artifact
  rather than the earlier `skipped` `0.0.0` temp-dir file: not stale, not skipped,
  correctly versioned, with its only disqualifier being that the gates fail. The
  four failures are unchanged from the `9b5231a` run — continuous recall
  0.720 < 0.75, recall@5px 0.800 < 0.90, worm recall 0.207 (audit 0.275),
  single-pixel 0.054 < 0.15 — so the 2026-10-10 measurement is confirmed on the
  PR head, not superseded by it. The pre-run copies stay preserved as
  `release-evidence.9b5231a-2026-10-10.json` and
  `science-gate.9b5231a-2026-10-10.md`.
- **The gate's default evidence path is not the path the release path reads.**
  `benchmarks/science_gate.py` defaults its manifest to
  `test_outputs/harness/science-gate.json`, while `docs/releasing.md` (lines 22,
  24, 174-179) and `.github/workflows/release.yml` (lines 53, 109, 167) all read
  `test_outputs/harness/release-evidence.json`. Following the documented command
  is consistent because it passes `--manifest` explicitly; running the bare
  `pixi run science-gate` silently writes evidence that no release path reads,
  which is how a stale `release-evidence.json` survived next to a fresh
  `science-gate.json`. One of the two names should become canonical — the choice
  is a release-surface decision, so it is recorded here rather than changed on a
  candidate branch.
- **The HEAD-pinned evidence belongs to `wip/wm-remediation-0.2.1` at
  `b2e4da6`.** The CANFAR runner lives on a separate branch
  (`wip/canfar-gate-harness`, now `063f15e`) precisely so that stays true; on
  that harness branch `HEAD` has moved past the evidence and a `release-check`
  there would report a SHA mismatch for reasons that have nothing to do with the
  gate.
- **`release-check` sections 1–3 pass; only the evidence section fails.**
  `release-check --version 0.2.1 --require-qualified --skip-build` on `b2e4da6`
  reported exactly two problems against the pre-existing manifest: *status is
  `failed`, not `passed`* and *evidence commit SHA does not match HEAD*. Version
  agreement, packaged-configuration hash match, the changelog section, the clean
  tree, and the tag guard all reported `ok`.
- **Upstream still carries the old publisher.** `astroai/weightmask:main` is
  `48b5a74`, declares `0.1.0`, has no `weightmask/_version.py`, and has
  `publish.yml` (triggered by `release: [published]` and `workflow_dispatch`, no
  validation) instead of `release.yml`. PR #75 replaces it. **Do not push a
  `v0.2.1` tag before that merge:** `release.yml` fails closed when the tag
  already exists, so an early tag poisons the only sanctioned release path.
- **`0.2.0` was tagged but never released.** PyPI lists only `0.1.0`, neither
  remote has a `v0.2.0` GitHub release, and `v0.2.0` exists only on the fork
  (`6a09de4`); upstream has only `v0.1.0`. The changelog nevertheless carries a
  `## 0.2.0 - 2026-10-02` section. Either complete that release or state
  explicitly that 0.2.0 was never published; the tag guard and the documentation
  links depend on which choice is made.
- **Release staging path unblocked.** `release-check --outdir dist` refuses while
  any entry exists in `dist/`; the five superseded 0.2.1 generations were moved
  (not deleted) to `test_outputs/harness/dist-archive/` with their hashes. The
  operator then approved and Tier B of the artifact purge was executed:
  `test_outputs/` went from 128 GB to 689 MB, keeping every manifest, report, log,
  hash, and pinned input. See `artifact_retention_2026-10-12.md` for the inventory,
  the provenance map, and exactly what was removed and retained.
- **Qualification is blocked by container egress, not by host load.** That
  distinction matters: headless CANFAR jobs get isolated per-pod resources, so
  "no concurrent benchmark load" is a statement about a shared machine and does
  not apply to them. The Mac-side serialization, not the load rule, was the real
  limit. A gate harness now exists (`benchmarks/canfar/gate_one.sh` and
  `submit_gate.sh`, branch `wip/canfar-gate-harness`, runner id `wm-gate-v1`) and
  was exercised against the live platform on 2026-10-12: jobs create, run, and
  report, `pixi 0.81.0` is already in `images.canfar.net/astroai/base:latest`, and
  the project mount `/arc/projects/mlao/cfhtcast/weightmask-perf` is writable by
  `sfabbro` and already carries `repos/`, `results/`, and `data/`.
- **Pod egress is not "CADC only"; the pod resolver answers for a handful of
  names.** Sampling resolution from inside a pod 36 times over six minutes
  (`benchmarks/canfar/probe_dns_outage.py`) gives:

  | host | resolved |
  |---|---|
  | `ws-uv.canfar.net` | 24/36 |
  | `pypi.org` | 27/36 |
  | `files.pythonhosted.org` | 19/36 |
  | `www.cadc-ccda.hia-iha.nrc-cnrc.gc.ca` | 0/36 |
  | `cadc-ccda.hia-iha.nrc-cnrc.gc.ca` | 0/36 |
  | `ws.cadc-ccda.hia-iha.nrc-cnrc.gc.ca` | 0/36 |
  | `cadc.nrc.ca` | 0/36 |
  | `conda.anaconda.org` | 0/36 |
  | `github.com` | 0/36 |
  | `google.com` | 0/36 |

  `google.com` failing rules out a CADC outage, and `/etc/resolv.conf` names the
  cluster's NodeLocal cache (`nameserver 169.254.25.10`). Only the platform API
  and the PyPI hosts answer, so *both* transports the campaign scripts assume are
  dead: `run_one.sh` cannot git-clone GitHub, and a CADC/VOSpace transport is not
  available either — the earlier `200` from `www.cadc-ccda…` no longer
  reproduces. The pod does still carry a per-session proxy certificate at
  `/arc/home/sfabbro/.ssl/cadcproxy.pem`.
- **The CANFAR control plane is degraded by roughly half, and its errors name
  the dependency.** 8 sequential `GET /skaha/v1/session` calls gave 5 x `200` and
  3 x `500`; 20 `canfar create` attempts gave 10 accepted. The failures are
  `IllegalStateException: Error calling /ac to augment subject` and
  `RuntimeException: unexpected failure download certificate`, plus one
  `UnknownHostException` naming the service's own hostname — the control plane
  fails on precisely the calls that reach outside itself. So every submission
  *and* retrieval needs a retry loop, `canfar info` and `canfar logs` included;
  `canfar delete` also blocks on a confirmation prompt unless given `--force`.
  `astroai/improc-webterm:latest` is gone from the registry, and
  `GET /skaha/v1/image` is the authority for what is public.
- **The local pixi environment is a relocated copy.** Its console scripts carry
  a stale shebang (`/Users/fabbros/src/weightmask/.pixi/...`), so
  `.pixi/envs/default/bin/pre-commit` fails with `bad interpreter` while
  `.pixi/envs/default/bin/python -m pre_commit` works. Run `pixi install` to
  rebuild it; see the "Rebuild copied or moved environments" section of
  `docs/releasing.md`.

### CANFAR transport and environment (2026-10-12)

Code now reaches a pod with neither GitHub nor CADC, so the *transport* blocker
is solved. The *input data* blocker is not.

- **`benchmarks/canfar/pod_ship.py`** ships a script through the session API.
  Two independent limits had to be worked around. The `canfar` CLI sends
  `cmd`/`args` as *query parameters*, and Tomcat rejects a request line above its
  header limit: an 11.9 KB payload returns `400 Bad Request` before the app sees
  it, so about 6 KB of args is usable. The same parameters in an
  `application/x-www-form-urlencoded` **body** are accepted (a 920 KB body
  creates a session). The body then meets argv limits instead — a 920 KB payload
  reaches the pod as `exec … argument list too long`, because a single argv
  string is capped at `MAX_ARG_STRLEN` (128 KiB). The payload is therefore
  lzma+base64 compressed and split across space-separated argv tokens that a
  bootstrap expression rejoins from `sys.argv`; the whole repository at a commit
  is six tokens.
- **`benchmarks/canfar/stage_source.py`** unpacks that tarball on the shared
  mount at `…/weightmask-perf/repos/cand-<sha12>`, verifying the tarball sha256
  in the pod before extracting. Verified at `313cb43`: 168 files, 3,443,875
  bytes, sha256 `1e8bfd3c…` matched.
- **`benchmarks/canfar/pod_run.py`** is the in-pod runner, parameterised by
  `WM_TREE` / `WM_COMMAND`. A `$` in any env value is rejected with `400 Illegal
  group reference` (the platform runs env values through a regex expansion), so
  commands use a `{py}` placeholder for the interpreter instead.
- **No public image carries the stack.** Probed per image: `astroai/improc`,
  `astroai/improc-notebook` and `astroai/base` have numpy, scipy, astropy, sep
  and yaml but lack **fitsio, scikit-image and astroscrappy**; `astroai/ray-worker`
  and `cadc/webterm:26.04` have essentially none of them. `pixi install` cannot
  repair that while `conda.anaconda.org` is at 0/36 — but the September
  campaign's package cache survives at
  `/arc/projects/mlao/cfhtcast/weightmask-perf/pixi-cache`, and solving against
  it completes the conda half; only the PyPI step for the editable `weightmask`
  dependency then fails. Because `weightmask` is pure Python, the built env's
  interpreter is reused directly with `PYTHONPATH=<tree>`, so the install cost is
  paid once per revision and later jobs need no network at all.

Measured in pods at `313cb43`:

| check | result |
|---|---|
| `benchmark-synthetic` (`synthetic_v2 --with-baselines`) | **ran and reproduces WM-16** — `F1 0.000 (TP:0, FP:0, FN:1390)` and `average streak F1 0.149 < 0.200` |
| `pytest -q` | started; still running when this was written, no result captured |
| `science-gate` | **blocked** — its inputs are three 343 MB CADC exposures and `www.cadc-ccda…` is at 0/36 |

A pod-side smoke run of the gate is possible against
`…/cfhtcast-bench/downloads/2240997p.fits.fz`, which the mount already carries,
but that exposure is not the fixture input set, so its numbers would not
reproduce or contradict the release gate.

### WM-16 root cause (2026-10-12)

The four zero-F1 cases are not a near miss: the production full-span trail
detector returns **nothing at all**. `pred_area` is `0` against ground-truth
areas of 1390-2476 px, and the stage logs `0 peaks` in both strata.

Instrumenting `hough_line_peaks` inside the real detection path
(`_detect_streaks_houghpeaks`) across all five `synthetic_v2` cases gives the
reason:

| case | max accumulator votes | min_votes | peaks found |
| --- | --- | --- | --- |
| `synthetic_sparse` | 33 | 100 | 0 |
| `synthetic_complex` | 70 | 100 | 0 |
| `synthetic_gradient_step` | 47 | 100 | 0 |
| `synthetic_variable_width` | 72 | 100 | 0 |
| `synthetic_elongated_sources` | 9 | 100 | 0 |

A line can collect at most one accumulator vote per binned pixel it crosses, so
on a 384 px case (96 binned at `bin: 4`) the whole accumulator tops out between
9 and 72 votes. `min_votes=100` therefore sits above the stage's own ceiling and
the Hough path can never accept anything at this image size.
`synthetic_sparse` reaches F1 0.746 with **zero** Hough peaks, so its trail is
found by the sparse-RANSAC stage, which is gated to sparse fields; the crowded,
gradient, variable-width, and elongated regimes have no equivalent and score 0.
`min_votes` is a scale-dependent absolute gating a quantity (trail length in
binned pixels) that is not scale-free, so the aggregate gate is measuring a
parameter cliff rather than the detector.

**A one-line fix was attempted and rejected.** Scaling the floor to the
retrievable span (`min_votes * min(bh, bw) / 512`, a no-op at the reference
2048 px MegaCam amp scale) was implemented and then reverted: it broke six
existing tests in `tests/test_streaks.py` and `tests/test_streak_fixes.py`, each
going from a detected trail to an *empty* mask. Admitting weaker peaks also lets
one claim a `(rho, theta)` slot in the `used` suppression list in iteration
order, pre-empting the true trail. Those tests are right to fail, so the floor
alone is not the fix. A correct change needs a size-aware floor **and** an
order-independent acceptance step (rank candidates by score before claiming a
slot) — a detector change that must not be smuggled into a release candidate.
The tree was reverted clean; the patch is kept at `/tmp/wm/streaks_scaling.patch`
for that work.

Measured 2026-10-09 on the dirty candidate tree:

- `pixi run lint` passed. `pixi run test` passed: 808 passed, 1 skipped, 1 xfailed.
- `pixi run benchmark-synthetic` failed: average streak F1 0.149 < 0.200. The CI-sized `synthetic_sparse` case passes its own gates. The other four cases stay at streak F1 0. Raising their flux or opening elongated-object handoff pixels did not pass the aggregate gate without also failing the sparse overmask gate. Thresholds were not lowered.
- Real-trail recall stays `n/a` (zero confirmed trails in the local labels). 0.2.1 states that exclusion in the changelog, and a passed evidence manifest may carry `scope_exclusions: ["real_trail_recall"]`.
- `pixi run science-gate` on this tree failed (2026-10-10T05:03:45Z): artefact FP rate 0.000 and real-trail recall `n/a`, but continuous recall 0.720 < 0.75, recall5 0.800 < 0.90, worm recall 0.207, single-pixel recall 0.054. Evidence status is `failed`. Do not publish.

Goal: resolve the 2026-10-06 repository audit without mixing unrelated changes, weakening scientific gates, or treating green unit tests as release qualification.

Release state: **NO-GO** until every item in the release checklist at the end of this file is satisfied.

## Decision 2026-10-09 — hold and open a PR without releasing

The 2026-10-09 A/B on an identical harness showed the red gates are the corrected
instrument, not a detector regression: on the bumped injection revisions
(`truth-pixel-recall-v3`, `additive-finite-injection-v3`) the candidate scores
continuous recall 0.720 / recall5 0.800 against HEAD `main` at 0.506 / 0.561,
and HEAD `main` also fails `synthetic_v2`. The candidate is better than `main`,
but the floors (0.75 / 0.90, worm drop, single-pixel 0.15, synthetic F1 0.200)
were calibrated on the superseded harness, so the earlier 0.753 / 0.903 pass in
`test_outputs/harness/0.2.1-release/` is not valid evidence under the corrected
instrument.

Owner decision: **do not release and do not lower a gate to pass.** Commit the
remediated work to a `wip/` branch and open the upstream PR with the evidence
left failing and labelled. WM-16 stays red; the release notes state the failure
and the real-trail recall exclusion. Re-baselining is deferred and, if taken up,
must re-derive the injection floors as a documented regression guard against the
HEAD-`main` corrected score (0.506 / 0.561) rather than to just below the
candidate score.

## How to use this plan

- Keep this file as the durable tracker. Change `[ ]` to `[x]` only after the listed acceptance criteria and verification pass.
- Implement one PR below per branch. Do not combine PRs merely because they touch the same module.
- Branches use `wip/wm-<id>-<topic>` and target `astroai/weightmask:main` through the fork workflow in `AGENTS.md`.
- Every defect starts with a regression test that fails against the current implementation.
- For subtle fixes, revert the implementation temporarily and confirm the new regression test fails before restoring the fix.
- Do not lower a benchmark threshold to make a failing detector pass.
- Do not delete code called through headers, configuration, generated products, benchmarks, examples, or historical evidence without proving the side channel is unused.
- Record each completed PR under `.cursor/harness/trajectories/` with verifier and outcome.

## Model assignment and review protocol

- **Sol:** `AI-Zone-GPT-5.6-Sol-US`, provider `azure-responses`, variant `high`.
  - Owns scientific formulas, detector semantics, FITS I/O, transactions, WCS geometry, memory, and performance-sensitive changes.
- **Luna:** `AI-Zone-GPT-5.6-Luna-US`, provider `azure-responses`, variant `high`.
  - Owns isolated boundary tests, deterministic fixtures, documentation, packaging, CI, workflow, manifests, and release evidence.
- The non-authoring model reviews each PR before it is considered complete.
- Scientific changes require an independent analytic/reference implementation, not only expected values copied from production code.
- Release and workflow changes require a clean exported-checkout run, not only an in-place run.

## Default decisions

These defaults avoid blocking implementation while preserving conservative behavior. Change them only with a recorded owner decision.

1. Ambiguous or missing-WCS chip matches are preserved, never cleared.
2. Detector stages are disabled only through explicit `enable: false`; zero iteration counts are validated according to each documented algorithm rather than silently interpreted as disablement.
3. Multi-file publication uses same-directory temporaries, generation IDs, validation, ordered atomic replacement, and rollback. This protects ordinary failures but does not claim crash-proof multi-file atomicity.
4. A valid saturation header takes precedence over an ambiguous sparse-tail estimate. An unvalidated header remains advisory.
5. Unknown nested configuration keys fail immediately; no deprecation period is planned before the unreleased `0.2.1` release.
6. Builders may accept non-negative signed integers after bounds checks; direct `WeightMaskProduct` construction requires canonical `uint32` masks.
7. Exposure normalization uses deterministic proportional bounded sampling, checked against exact pooled percentiles on small fixtures.
8. Legacy sky meshes with incomplete or inconsistent dimension cards are rejected rather than guessed.
9. No automatic deletion of ignored benchmark data, outputs, environments, or release artifacts will be added.
10. Dependency upper bounds are added only for reproduced incompatibilities.

## Baseline and global gates

Baseline evidence from the audit:

- `pixi run lint`: passed.
- Local suite: 524 passed, 1 skipped.
- Clean exported checkout: 516 passed, 9 skipped.
- Release artifact check: passed.
- Science gate: reported passed, but real-trail recall was `n/a` because the corpus contains zero confirmed real trails.
- Synthetic benchmark: failed average streak F1 `0.091 < 0.200` and object recall `0.697 < 0.900`.

Run after every PR:

```bash
pixi run lint
pixi run test
```

Run after every wave:

```bash
pixi run ci-local
pixi run python -m pre_commit run --all-files
```

The module form is intentional until the local stale Pixi launcher shebang is rebuilt; the repository command remains `pre-commit run --all-files` in a clean environment.

Run after changes affecting masks, variance, detector inputs, output I/O, or normalization:

```bash
pixi run benchmark-synthetic
pixi run science-gate
```

A pre-existing benchmark failure must remain visible. It cannot be attributed to a change without a before/after run using the same commit inputs, config, environment, and metric revision.

---

# Wave 0 — Tracking and release freeze

## [ ] WM-00 — Freeze the candidate release

Owner: Luna
Reviewer: Sol

Actions:

- Do not tag or publish `0.2.1` while this plan is open.
- Keep release notes marked as candidate/unreleased until upstream contains the fixes.
- Record the exact base commit, config SHA, benchmark input identifiers, and metric revisions used for every science comparison.
- Open one tracking issue containing the IDs in this file and link each remediation PR back to it.

Acceptance:

- No workflow can publish the current audited commit as `0.2.1`.
- Every implementation PR has one primary WM ID and an explicit dependency list.

---

# Wave 1 — Small, independently testable correctness boundaries

These PRs are independent and may run in parallel.

## [x] WM-01 — Correct flat-uncertainty propagation

Owner: Sol
Reviewer: Luna
Files: `weightmask/variance.py`, `tests/test_variance.py`, `weightmask.yml`, algorithm documentation

Work:

- Replace the circular test formula with output-ADU uncertainty propagation over flats `0.25, 0.5, 1, 2`.
- Pin `S=1000`, `gain=2`, `read_noise=0`, `flat_rel_noise=0.01`, `flat=0.5` to variance `1100 ADU²` and inverse variance `1/1100`.
- Include the flat factor in the pre-division electron-space uncertainty term.
- Clarify the parameter definition and units.

Acceptance:

- The calibrated flat-noise term is `(S * rel_map)^2`.
- `flat_rel_noise=0` is unchanged.
- Invalid flat pixels still produce zero valid inverse variance.
- The new test fails when the old formula is restored.

Targeted gate:

```bash
pixi run pytest tests/test_variance.py -q
```

## [x] WM-02 — Enforce quality-mask integer semantics

Owner: Luna
Reviewer: Sol
Files: `weightmask/contract.py`, `tests/test_contract.py`, `tests/test_release_io.py`

Work:

- Reject negative signed values before conversion.
- Enforce `uint32` on direct product construction.
- Test non-negative signed inputs, bools, floats, object arrays, oversized integers, byte order, non-contiguous inputs, future bits, and caller mutation.

Acceptance:

- No negative mask becomes `0xffffffff`.
- Future valid `uint32` bits survive.
- Inputs are not mutated.
- The regression test fails when the bounds check is removed.

Targeted gate:

```bash
pixi run pytest tests/test_contract.py tests/test_release_io.py -q
```

## [x] WM-03 — Validate sky-mesh metadata and shape

Owner: Sol
Reviewer: Luna
Files: `weightmask/background.py`, `weightmask/reconstruct_sky.py`, `tests/test_background.py`, `tests/test_release_io.py`

Work:

- Validate both mesh box dimensions, output dimensions, rank, finiteness, and implied node count before interpolation or writing.
- Reject missing, unequal, zero, negative, truncated, or inconsistent products.

Acceptance:

- Malformed meshes fail before output creation.
- Valid one-node dimensions and current round trips remain supported.

Targeted gate:

```bash
pixi run pytest tests/test_background.py tests/test_release_io.py -q
```

## [x] WM-04 — Repair sparse saturation fallback

Owner: Sol
Reviewer: Luna
Files: `weightmask/satur.py`, `tests/test_satur.py`

Work:

- Add the reproduced eight-at-50k plus one-at-65k case.
- Implement a repeated upper-tail plateau/change-point estimate rather than an extreme percentile.
- Add header-precedence, smooth-tail, isolated-outlier, and sparse-source controls.

Acceptance:

- One outlier cannot move a repeated plateau threshold to the maximum.
- A genuine smooth upper tail is not declared saturated without adequate evidence.
- Header behavior is explicit and tested.

Targeted gate:

```bash
pixi run pytest tests/test_satur.py tests/test_bad.py -q
```

## [x] WM-05 — Make test randomness deterministic

Owner: Luna
Reviewer: Sol
Files: stochastic tests in `tests/test_objects.py`, `tests/test_robust_features.py`, `tests/test_cosmics.py`, `tests/test_satur.py`, `tests/test_cli.py`, `tests/test_variance.py`

Work:

- Replace global RNG calls and global seeding with local `default_rng` instances.
- Preserve current scientific values and tolerances.

Acceptance:

- Affected tests do not mutate global NumPy RNG state.
- Repeated isolated runs produce identical outcomes.

Targeted gate:

```bash
pixi run pytest tests/test_objects.py tests/test_robust_features.py tests/test_cosmics.py tests/test_satur.py tests/test_cli.py tests/test_variance.py -q
```

---

# Wave 2 — Detector inputs, calibration masks, and validation

## [x] WM-06 — Make flat masking tile-invariant and column-specific

Owner: Sol
Reviewer: Luna
Files: `weightmask/bad.py`, `tests/test_bad.py`, `tests/test_flat_mask_cache.py`, `tests/test_fix_regressions.py`

Work:

- Add halo-expanded tile processing and crop filtered results back to tile cores.
- Compute column-level statistics across the full HDU.
- Attribute derivative discontinuities to the side that deviates from the local baseline.
- Test gradients, defects near every tile boundary, isolated hot/dead columns, adjacent defects, and detector edges.

Acceptance:

- Tested masks are invariant across tile sizes.
- An isolated bad column masks exactly that column.
- Peak working memory remains bounded by tile size plus halo.
- Cache keys change if semantics require it.

Targeted gate:

```bash
pixi run pytest tests/test_bad.py tests/test_flat_mask_cache.py tests/test_fix_regressions.py -q
```

## [x] WM-07 — Stream finite-aware persistence profiles

Owner: Sol
Reviewer: Luna
Files: `weightmask/streaks.py`, `weightmask/cli.py`, `tests/test_persistent_axis.py`, `tests/test_release_io.py`

Work:

- Use finite-aware row/column statistics and finite-only robust centers/scatters.
- Treat all-invalid axes explicitly.
- Accumulate profiles/hit counts while streaming and release each full CCD array immediately.
- Add a bounded-memory benchmark or instrumentation test.

Acceptance:

- An unrelated NaN or Inf cannot suppress a repeated hot axis.
- Invalid axes are not falsely flagged.
- Persistence storage scales with profile lengths, not total image pixels.

Targeted gate:

```bash
pixi run pytest tests/test_persistent_axis.py tests/test_release_io.py -q
```

## [x] WM-08 — Complete preflight configuration validation

Owner: Sol
Reviewer: Luna
Files: `weightmask/process.py`, `weightmask/cli.py`, `weightmask.yml`, `tests/test_cli.py`, `tests/test_fix_regressions.py`, `tests/test_release_io.py`

Work:

- Inventory every consumed configuration key.
- Validate types, finiteness, ranges, enums, and cross-field constraints.
- Reject unknown nested keys.
- Preserve `process.validate_config` as the public wrapper.
- Validate before FITS handles or output writers open.

Acceptance:

- The canonical YAML passes.
- Wrong types, NaN/Inf, invalid sizes, inverted thresholds, negative counts/epsilon/areas, booleans-as-numbers, and misspellings fail deterministically before I/O.

Targeted gate:

```bash
pixi run pytest tests/test_cli.py tests/test_fix_regressions.py tests/test_release_io.py -q
```

## [x] WM-09 — Fix detector-prior contract coverage

Owner: Luna
Reviewer: Sol
Files: `weightmask/process.py`, `tests/test_fix_regressions.py`, `tests/test_streaks.py`

Work:

- Test prior shape mismatch, non-contiguous input, mutation, exclusion propagation, prior-only pixels, and overlap with confirmed streaks.
- Make shape errors explicit rather than silently ignoring a prior.

Acceptance:

- Caller priors are unchanged.
- Prior-only pixels do not acquire STREAK bits.
- Overlap semantics are documented and pinned.

---

# Wave 3 — Fail-closed processing and consistent publication

## [x] WM-10 — Fail closed on required detector errors

Owner: Sol
Reviewer: Luna
Depends on: WM-08
Files: `weightmask/cosmics.py`, `weightmask/objects.py`, `weightmask/process.py`, `weightmask/mef.py`, detector and release-I/O tests

Work:

- Reverse tests that currently require exceptions to become empty masks.
- Introduce a typed stage failure propagated through image and HDU orchestration.
- Cover primary CR, enabled faint CR, SEP seed/main, object post-processing, missing backends, and genuine zero detections.

Acceptance:

- Required backend/runtime failure produces a nonzero CLI result and no scientifically complete HDU.
- Explicitly disabled stages remain successful and empty.
- Genuine zero detections remain distinguishable from failure.

Targeted gate:

```bash
pixi run pytest tests/test_cosmics.py tests/test_objects.py tests/test_cli_process_hdu.py tests/test_release_io.py -q
```

## [x] WM-11 — Transactional product publication

Owner: Sol
Reviewer: Luna
Depends on: WM-03, WM-10
Files: `weightmask/mef.py`, `weightmask/cli.py`, `weightmask/reconstruct_sky.py`, `tests/test_release_io.py`, `tests/test_cli.py`, `tests/test_parallel.py`, `tests/test_hdu_naming.py`

Work:

- Prepopulate destinations and inject failures before first write, between products, during post-processing, during promotion, and during reconstruction.
- Write every requested product to a same-directory temporary path.
- Validate HDU count, shape, `EXTNAME`, contract metadata, and shared generation ID before promotion.
- Use ordered `os.replace` with backups and rollback; remove temporaries on failure.

Acceptance:

- Failure leaves all previous outputs byte-identical or leaves no outputs when none existed.
- No mixed-generation product set is exposed after handled errors.
- Successful output files share a generation ID and synchronized HDU structure.

Targeted gate:

```bash
pixi run pytest tests/test_release_io.py tests/test_cli.py tests/test_parallel.py tests/test_hdu_naming.py -q
```

## [x] WM-12 — Pixel-weighted exposure normalization

Owner: Sol
Reviewer: Luna
Depends on: WM-11
Files: `weightmask/mef.py`, `weightmask/contract.py`, `tests/test_release_io.py`, `tests/test_parallel.py`, `tests/test_contract.py`

Work:

- Add unequal-HDU-size and unequal-valid-fraction fixtures.
- Count positive populations and sample deterministically over the logical pooled population.

Acceptance:

- Small fixtures equal the exact pooled percentile.
- Large workloads remain bounded by the documented sample cap.
- Results are deterministic and independent of HDU partitioning.

---

# Wave 4 — Component-safe streak handling

## [x] WM-13 — Replace aggregate chip-replica clearing

Owner: Sol
Reviewer: Luna
Depends on: WM-07, WM-11
Files: `weightmask/mef.py`, `tests/test_chip_replica.py`, `tests/test_release_io.py`, `tests/test_trail_truth.py`; geometry reference in `benchmarks/curate_trail_truth.py`

Work:

- Label and catalog connected streak components separately.
- Add fixtures containing one repeated and one unique component on the same HDU.
- Add genuine WCS-consistent cross-chip trails with equal local coordinates.
- Transform component lines into a common tangent-plane/focal-plane frame.
- Clear only independently matched detector-fixed components; preserve ambiguous/no-WCS components.

Acceptance:

- Matching one component never clears another.
- WCS-consistent sky trails survive.
- Detector-fixed replicas are removed only with sufficient independent evidence.
- Mask, weight, raw weight, confidence, and individual streak products stay synchronized in the transaction.

Targeted gate:

```bash
pixi run pytest tests/test_chip_replica.py tests/test_release_io.py tests/test_trail_truth.py -q
```

## [x] WM-14 — Recompute scatter for every fan candidate

Owner: Sol
Reviewer: Luna
Files: `weightmask/streaks.py`, streak profile/fan tests

Work:

- Recompute sideband values, robust center, and scatter for each fan geometry.
- Store the winning scatter with the winning geometry.
- Add a spatially varying-noise regression.

Acceptance:

- Acceptance and mask width use the candidate’s own noise estimate.

## [x] WM-15 — Resolve dynamic CR significance semantics

Owner: Sol
Reviewer: Luna
Files: `weightmask/cosmics.py`, `weightmask.yml`, `tests/test_cosmics.py`, CR benchmark code

Work:

- Keep the shipped default off while measuring fixed versus adaptive dimensionless thresholds on controlled normalized residuals.
- Remove or replace ADU-RMS scaling only after curves show the intended false-positive/recall behavior.

Acceptance:

- No threshold rule becomes more permissive merely because data units or ADU RMS increase.
- Any retained adaptive rule is dimensionless and benchmarked.

This is an evidence task, not permission to tune until the suite turns green.

---

# Wave 5 — Benchmark and test integrity

## [ ] WM-16 — Enforce synthetic numerical gates in CI

The CI-sized `synthetic_sparse` gates are in `tests/benchmarks/run.py` and pass under `pixi run test`. The full-suite aggregate does not: average streak F1 0.149 < 0.200 on 2026-10-09. Do not lower 0.200.

Owner: Luna
Reviewer: Sol
Files: `tests/benchmarks/run.py`, `tests/test_benchmarks.py`, synthetic fixtures/config

Work:

- Make the CI-sized case assert meaningful case-specific precision, recall, F1, overmask, and coverage floors.
- Preserve the full-suite aggregate gate.
- Investigate the current aggregate failures; fix detector/generator/metric defects rather than lowering thresholds.

Acceptance:

- Zero-recall and gross-overmask mutations fail ordinary CI.
- Full `pixi run benchmark-synthetic` passes its existing justified gates before release.

## [x] WM-17 — Replace stale stage evidence with current-code fixtures

Owner: Luna
Reviewer: Sol
Files: `tests/test_streak_dead_stages.py`, `tests/test_benchmark_evidence.py`, `benchmarks/streak_recall_floor.py`, `benchmarks/streak_stage_sweep.py`

Work:

- Add small deterministic contour-only and RANSAC-only recovery cases executed in CI.
- Require archived evidence to contain input, config, source, metric, and code hashes.

Acceptance:

- CI no longer skips the only proof that each enabled stage earns its cost.
- Source/config changes invalidate historical evidence rather than silently passing it.

## [x] WM-18 — Make streak and CR injection production-faithful

Owner: Luna
Reviewer: Sol
Depends on: WM-10
Files: `tests/test_streak_recall_floor.py`, `benchmarks/streak_inject.py`, `benchmarks/cr_faint_curves.py`, `benchmarks/production_inputs.py`, `weightmask/process.py`

Work:

- Inject into raw science before recomputing every upstream stage.
- Add source Poisson noise using gain.
- Ensure injected truth does not overlap itself or pre-existing invalid pixels unless scored separately.
- Capture and compare each detector call’s inputs with production.

Acceptance:

- Benchmarks and production pass demonstrably equivalent detector inputs.
- No current zero-positive corpus is described as real-trail recall evidence.

## [x] WM-19 — Strengthen scientific behavior tests

Owner: Luna
Reviewer: Sol
Files: `tests/test_photometry_bias.py`, `tests/test_streaks.py`, `tests/test_streak_recall_floor.py`, production integration tests

Work:

- Replace local-array-only photometry tests with `process_image` integration tests.
- Add nonzero supported dashed-trail recall floors.
- Replace 35%-frame false-positive allowance with precision, false pixels/Mpix, largest component, and clean-source flux-loss limits.

Acceptance:

- Breaking production masking or uncertainty behavior fails these tests.
- Empty streak output cannot satisfy a supported-regime recall test.

## [x] WM-20 — Correct benchmark product and statistical comparisons

Owner: Luna
Reviewer: Sol
Files: `benchmarks/perf_megacam.py`, `benchmarks/weight_pull_compare.py`, `benchmarks/mine_trail_candidates.py`, `tests/test_perf_report.py`, `tests/test_benchmark_evidence.py`

Work:

- Compare expected file sets symmetrically.
- Match HDUs/products through `EXTNAME`/CCD identity.
- Compare quality mask, inverse variance, normalized weight, confidence, sky, individual masks, metadata, and HDU ordering.
- Use physical inverse variance for Gaussian pulls.

Acceptance:

- A missing product fails.
- Positional mispairing fails.
- A known Gaussian fixture recovers unit-width pulls.

## [x] WM-21 — Harden remaining benchmark edge cases

Owner: Luna
Reviewer: Sol
Files: `benchmarks/curate_trail_truth.py`, `benchmarks/score_trail_truth.py`, `benchmarks/cr_faint_curves.py`, associated tests

Work:

- Make curation finite-aware.
- Use robust finite RMS for Poloka comparisons.
- Sample CR origins without replacement and reject truth overlap.
- Ensure streak placement bands do not intersect.
- Scale nominal streak sigma by the local valid RMS map.

Acceptance:

- NaN/Inf cannot silently suppress proposal generation.
- Sentinel-heavy RMS either yields a finite robust estimate or an explicit failure.
- Truth events remain distinct and locally calibrated.

## [x] WM-22 — Establish a positive real-trail fixture

Owner: science owner + Luna implementation
Reviewer: Sol
External dependency: independently labelled, redistributable data

No redistributable positive trail was available. The release-blocking clause is met by exclusion: 0.2.1 does not claim validated real-trail recall, and a passed manifest records `scope_exclusions: ["real_trail_recall"]`. A future fixture still needs independent labels, checksums, and a numeric recall gate.

## [x] WM-23 — Make performance evidence statistically defensible

Owner: Luna
Reviewer: Sol
Depends on: WM-20
Files: `benchmarks/exposure_time.py`, `benchmarks/perf_megacam.py`, benchmark documentation/tests

Work:

- Randomize/interleave A/B order.
- Use at least five timed repetitions.
- Report median and IQR/MAD, cold and warm cache separately, fixed thread environment, and input hashes.
- Require correctness equivalence before reporting speed comparisons.

Acceptance:

- No best-of-two headline is used as release evidence.
- Timing reports are reproducible and identify their instrument/configuration.

---

# Wave 6 — Packaging, compatibility, documentation, and release machinery

## [x] WM-24 — Deliver a version-compatible canonical configuration

Owner: Luna
Reviewer: Sol
Files: `pyproject.toml`, `MANIFEST.in`, `weightmask.yml`, package resources, installation/usage docs, release workflow/checks

Work:

- Package one canonical YAML under `weightmask/` and expose a documented copy/resource mechanism.
- If a root convenience copy remains, hash-check it against the packaged copy.
- Verify wheel, sdist, and GitHub release contents.
- Remove instructions to fetch mutable `main` configuration for an installed release.

Acceptance:

- Installed wheel and sdist can locate/copy the matching configuration outside the source tree.
- GitHub release attaches validated artifacts and configuration/evidence as designed.

## [x] WM-25 — Define supported Python, platform, and dependency ranges

Owner: Luna
Reviewer: Sol
Files: `pyproject.toml`, `pixi.toml`, `pixi.lock`, CI workflow

Work:

- Measure minimum supported dependency versions.
- Add evidence-based lower bounds.
- Add Python 3.10–3.13 test/package-smoke coverage or narrow metadata.
- Keep Linux as the explicit tested platform unless macOS/Windows jobs are added.
- Add real optional-torchfits coverage where feasible.

Acceptance:

- Every advertised Python version has CI evidence.
- Minimum-dependency environment passes its supported subset.
- Metadata does not imply untested platform support.

## [x] WM-26 — Define and test the public API and output contract

Owner: Luna
Reviewer: Sol
Files: `docs/api.md`, `weightmask/__init__.py`, `weightmask/background.py`, `weightmask/torchfits_adapter.py`, CLI/usage docs, API smoke tests

Work:

- Decide and document exact supported import paths.
- Recommended: document background reconstruction helpers and torchfits adapter as supported module APIs without adding package-root re-exports.
- Document derived mask/inverse-variance/sky filenames.
- Document or reject `sky_format: mesh` fallback to full sky.

Acceptance:

- Every documented import and CLI behavior has a smoke test.
- Internal modules are not presented as stable APIs.

## [x] WM-27 — Correct current documentation and historical labeling

Owner: Luna
Reviewer: Sol
Files: `README.md`, `CHANGELOG.md`, `docs/usage.md`, `docs/algorithms.md`, `docs/releasing.md`, `benchmarks/canfar_experiments/manifest.json`, CANFAR scripts/tests

Work:

- Correct stale trail pixel counts.
- Move superseded MRT/Radon chronology out of final release notes or label it historical.
- Mark the old CANFAR manifest pinned/historical.
- Refuse current submission of removed parameters and unresolved E8/TBD groups.
- Fix upstream links only after the target files exist upstream.

Acceptance:

- Current docs agree with tested behavior.
- Historical evidence remains reproducible but cannot be mistaken for a current campaign config.

## [x] WM-28 — Add release evidence enforcement

`real_trail_recall: n/a` is qualified only together with `scope_exclusions: ["real_trail_recall"]` and the same sentence in the changelog. A skipped MegaCam report is still unqualified. Publishing requires that JSON via `evidence_b64` because the GitHub runner has no exposures.

Owner: Luna
Reviewer: Sol
Depends on: WM-16, WM-20, WM-22, WM-24, WM-25, WM-26
Files: `benchmarks/science_gate.py`, `benchmarks/release_check.py`, `.github/workflows/release.yml`, release tests/docs

Work:

- Emit a small tracked evidence manifest with status, version, commit SHA, config SHA, input/manifest SHA, metric revisions, command, timestamp, and data IDs.
- Require a matching passed report for non-dry-run publishing.
- Treat skipped, stale, mismatched, and `trail_recall: n/a` reports as unqualified.
- Attach validated wheel, sdist, configuration, and evidence before tagging.
- Pin third-party publishing actions to reviewed commit SHAs and narrow job permissions.

Acceptance:

- A dry run may build without private data but remains visibly unqualified.
- Non-dry-run publishing fails before tagging if evidence is absent or mismatched.
- The exact artifacts qualified are the artifacts published.

## [x] WM-29 — Document artifact retention and cleanup

Owner: Luna
Reviewer: Sol
Files: `.gitignore`, `docs/releasing.md`, benchmark/CANFAR documentation

Work:

- Document dry-run inspection commands for ignored `test_outputs/`, `benchmark_data/`, duplicate `dist/`, caches, and stale local environments.
- Define retention for small JSON/Markdown provenance versus large FITS/cache products.
- Document that copied/moved Pixi environments must be rebuilt; stale executable shebangs are not valid evidence.

Acceptance:

- No automatic destructive cleanup is added.
- Operators can distinguish current evidence from stale outputs before deletion or release.

---

# Wave 7 — Final integration and upstream release

## [ ] WM-30 — Full release-candidate qualification

Owners: Sol + Luna
Depends on: all release-blocking items above

Run on the exact candidate commit in a clean environment with no concurrent benchmark load:

```bash
pixi install
pixi run lint
pixi run test
pixi run ci-local
pixi run python -m pre_commit run --all-files
pixi run benchmark-synthetic
pixi run science-gate
pixi run release-check -- --version 0.2.1
```

Additional requirements:

- Run product equivalence and performance protocols on the same pinned MegaCam inputs and config.
- Inspect the wheel, sdist, packaged configuration, evidence manifest, and GitHub-release attachment set.
- Confirm all advertised Python-version CI jobs pass.
- Review every commit included relative to `upstream/main`.
- Merge through a PR into `astroai/weightmask:main`; do not publish from the fork or from the current 49-commit-ahead local branch.

Release acceptance checklist:

- [ ] WM-01 through WM-21 complete or explicitly rejected with reproduced counterevidence.
- [ ] WM-22 supplies numeric positive real-trail evidence, or release scope explicitly removes that claim.
- [ ] WM-24 through WM-29 complete.
- [ ] Synthetic benchmark has no gate failures.
- [ ] Science evidence is numeric, current, and tied to the candidate commit/config/data.
- [ ] No required real-data/stage test skips in the release qualification environment.
- [ ] Full tests, clean-checkout CI, hooks, package builds, isolated installs, and entry points pass.
- [ ] Product set is transactional and internally generation-consistent.
- [ ] Release notes describe final shipped behavior only.
- [ ] Upstream `main`, tag, PyPI artifacts, GitHub release assets, and documentation agree on the same version.

---

# Findings intentionally tracked as decisions, not blind edits

These audit findings remain covered but must be resolved with evidence rather than automatic code changes:

- Dynamic CR significance (`WM-15`): measurement precedes algorithm change.
- Positive real-trail qualification (`WM-22`): requires independent labels and redistribution rights.
- Dependency upper bounds (`WM-25`): only reproduced incompatibilities justify caps.
- Multi-file crash atomicity (`WM-11`): generation IDs and rollback are the default; a manifest/directory pointer is a separate consumer-contract decision.
- Automatic deletion of large ignored data (`WM-29`): documentation only.
- No production module is currently classified as provably dead. Historical manifests and superseded changelog material are retained as labelled evidence rather than silently deleted.

# Completion record

When all release-blocking work is finished, replace the status at the top with:

```text
Status: closed — released <version> from <commit>
```

Then record the final release URL, tag, artifact hashes, evidence-manifest hash, and the commands above with their exact outcomes.
