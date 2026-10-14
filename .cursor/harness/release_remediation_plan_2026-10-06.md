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
  `test_outputs/benchmarks/synthetic_v2/summary.md`. The cause recorded below was
  incomplete, however: the accumulator is starved *upstream* of the vote floor.
  Between 17.6% and 86.4% of each trail's own ground-truth pixels are already
  inside the pre-mask when the streak stage runs, and only 0-6 binned cells on
  the trail's line clear `thresh_sig` even unmasked. See the corrected WM-16
  root cause below.
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
- **Standing image decision: use `images.canfar.net/astroai/base:latest` for
  headless sessions; `astroai/improc` is not needed.** It is public, it is
  labelled `headless`, it carries `pixi 0.81.0`, and probing it showed the same
  package set as `improc` (numpy, scipy, astropy, sep, yaml, pandas, matplotlib —
  and the same three missing packages), so `improc` buys nothing.
  `benchmarks/canfar/pod_ship.py` defaults to it and `CANFAR_IMAGE` overrides it.
  The `improc-webterm` name in
  `benchmarks/canfar_experiments/manifest.json` is deliberately left alone: that
  manifest is pinned and historical, and rewriting it would falsify the record of
  what the September campaign actually ran.
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

**The transport commit broke the campaign validator pin, and only a full-suite
run caught it.** `3d7ecca` added one blank line to
`benchmarks/canfar/manifest.py` — ruff `I001` requires it after the import block,
and without it `pixi run lint` fails, which is why the change was made — but that
changed the validator's sha256 from `ba5bd984…` to `85ff9d4a…`. The historical
campaign manifest pins the validator digest at `runner.validator_sha256`, so the
guard in `submit_all.sh` began hard-failing before it ever reached `canfar
create` (`validator SHA mismatch: manifest=ba5bd984… script=85ff9d4a…`), and five
tests in `tests/test_canfar_scripts.py` failed, including
`test_manifest_is_pinned_historical_and_hash_is_content_hash`.

**Fixed by refreshing the pin, not by reverting the blank line.** The whitespace
change is required by lint and is behaviour-neutral, so the *pin* was the stale
side: `runner.validator_sha256` now carries `85ff9d4a…`. `git grep` confirms no
other tracked file referenced the old digest. Verified with `pixi run python -m
pytest tests/test_canfar_scripts.py -q` (20 passed) and a full `pixi run test`.
Recorded because it is this plan's own failure mode: a commit that looks clean,
passes lint, and whose collateral damage appears only when the *whole* suite runs.

Measured in pods at `313cb43`:

| check | result |
|---|---|
| `benchmark-synthetic` (`synthetic_v2 --with-baselines`) | **ran and reproduces WM-16** — `F1 0.000 (TP:0, FP:0, FN:1390)` and `average streak F1 0.149 < 0.200` |
| `pytest -q -v`, bound by `timeout 1500` | **did not finish**: reached 71% (through `tests/test_rms_validity.py`) before the bound hit, `rc=124`. Three failures appeared first — two in `tests/test_perf_protocol.py`, one in `tests/test_release_machinery.py` — but the run was cut off before the failure summary, so the assertions were never captured. The same two modules pass locally in 45 s (81 passed, 12 subtests), and a re-run aimed at just those modules failed with `FATAL: WM_TREE and WM_COMMAND are both required`: the session was created but its env never arrived, which is a control-plane flake rather than a test result. **Treat the pod failure count as unexplained, not as product failures.** Resolved 2026-10-13: it did not reproduce at the tip, and the `FATAL` blamed on a flake was a transport bug of ours — see the next section. |
| `science-gate` | **blocked** — its inputs are three 343 MB CADC exposures and `www.cadc-ccda…` is at 0/36. Superseded 2026-10-13: egress is back, the inputs were staged in-pod in seconds, and what remains is a *budget* question, not an availability one — the gate's score stage is 89 real detector runs over the labelled HDUs and has no scope knob. See "The science gate in a pod" below. |

A pod-side smoke run of the gate is possible against
`…/cfhtcast-bench/downloads/2240997p.fits.fz`, which the mount already carries,
but that exposure is not the fixture input set, so its numbers would not
reproduce or contradict the release gate.

### The science gate in a pod: available, but over budget (2026-10-13)

With egress restored the gate's inputs stop being the blocker, so the pod run
was attempted twice. The inputs themselves are cheap: three exposures of
343,817,280 / 343,776,960 / 343,676,160 B plus the 389,885,760 B flat, each
fetched in 5-7 s, and `stage_inputs` is idempotent — a later job found all four
already present in the staged tree and downloaded nothing.

| run | inputs | budget | result |
|---|---|---|---|
| `wm-gate` (session `dw3b82r2`) | shared mount | none set | reaped at the session cutoff after ~80 min, **no stage output** |
| `wm-gate-scratch` (session `hnqi7hey`) | node-local `/scratch` | `WM_TIMEOUT=5400` | `!! timed out after 5400s`, `rc=124`, no gate artifacts — only a stale `benchmarks` dir was copied |

The scratch run settles the question the mount run left open: the gate does not
fit in 90 minutes even with its inputs node-local, so **I/O locality is not the
constraint** — the score stage is. What that stage costs is not an accident of
configuration: the fixture labels **89 (exposure, HDU) jobs** (25 + 26 + 27 for
the three exposures, 11 more for `megacam_streak_case`), and `score_trail_truth`
re-runs the production streak stage on each one. The gate exposes no scope knob
for it either — `--hdu` bounds only the injection and cosmic-ray stages, while
`_score` always walks the whole label set. So the gate's pod cost is
89 detector runs plus three cheap stages, and the runner needs a budget sized
from that rather than from the 90-minute default that was tried.

Both attempts were also *unobservable*, which is the reason the first one can
only be described as "reaped": every long step in the harness captured its
child's output and printed it after the child finished, so a stage that runs for
hours produced no lines at all, and a timeout discarded whatever it had
printed. That is now fixed rather than worked around — `pod_run.py`,
`science_gate._run` and the scorer's own pool loop all stream as they go (see
"Streaming progress" below).

### Streaming progress, so a long stage cannot be a black box (2026-10-13)

Three changes, all in the direction of "a run that is killed still tells you what
it did":

- **`benchmarks/canfar/pod_run.py`**: `sh(..., stream=True)` hands the child the
  pod's own stdout instead of a pipe, and the step has no output to lose at all.
  It is used for the four steps that can outlast a coffee break — the tree copy,
  the `pixi install`, the input downloads, and `WM_COMMAND` itself. The default
  is unchanged (capture), because several callers still log `rc` from short
  commands where interleaving would only add noise.
- **`benchmarks/science_gate.py`**: `_run` streams each stage line and still
  accumulates the same text the notes are parsed from, so the gate's own output
  keeps identical content while `canfar logs` gains per-stage progress.
- **`benchmarks/score_trail_truth.py`**: the pool loop prints one line per HDU as
  it lands (`[17/89] 1013720p HDU 14: scored in 41s`), which is what makes the
  89-job cost model visible while it is being paid rather than after.

Verified with `pixi run lint` and 105 tests across `test_science_gate.py`,
`test_benchmark_evidence.py`, `test_wm21_benchmarks.py` and
`test_canfar_scripts.py`.

### Seven non-sidereal exposures processed in pods (2026-10-13)

Asked for: launch a headless CANFAR session per observation id, process on
`/scratch`, and leave the result somewhere fetchable so it can be reviewed. The
observation ids are non-sidereal-tracking MegaCam fields, i.e. the case the
streak stage exists for: `1729091`, `1774908`, `1851368`, `1851960`, `1853217`,
`1853221`, `1853389`.

Contract, in `benchmarks/canfar/process_exposure.py`:

- the exposure is downloaded to `/scratch` (275-345 MB per file) and processed
there, because a pod's scratch is fast and everything on it dies with the pod;
- the reviewable artifacts go to the shared mount **as the run proceeds**, not at
the end: `<obsid>.mask.fits.fz` (one HDU per science HDU, written in index
order), `<obsid>.hdu_stats.tsv` (per-bit pixel counts), `<obsid>.run.json`
(what ran, with what digest, for how long). Full-resolution weight/invvar/sky
maps stay on scratch: they are large, reproducible from the exposure plus the
config, and no review of "what did the mask do" needs them beside the mask;
- destination root is `/arc/projects/mlao/cfhtcast/weightmask-perf/results`,
which survives the session (`pod_run.py` exports it as `WM_RESULTS_ROOT`).

Result of the first fan-out, **4 of 7 complete**:

| obsid | HDUs | holes | unreadable slots | STREAK px | HDUs with streaks | run.json->mask digest | stats->pixels |
|---|---|---|---|---|---|---|---|
| `1729091` | 36/36 | 0 | 0 | 1,902 | 2 | ok | ok |
| `1853217` | 40/40 | 0 | 0 | 210,479 | 7 | ok | ok |
| `1853221` | 40/40 | 0 | 0 | 385,059 | 12 | ok | ok |
| `1853389` | 40/40 | 0 | 0 | 446,842 | 21 | ok | ok |

The three that did not finish (`1774908`, `1851368`, `1851960`) were left with an
unrenamed `.wm-tmp-*` mask and no stats at all, and **the deletion was ours, not
the platform's**: `pod_ship.py` deleted the session when its own `--wait`
deadline expired, whether or not the pod had finished. A harness that kills the
job it is waiting for is indistinguishable from a cluster fault on the mount, so
the default is inverted — a session that is still running at the deadline is now
left alone (`--delete-unfinished` restores the old behaviour), pinned by
`test_pod_ship_does_not_delete_a_job_its_own_deadline_missed`. The three were
re-run with a 4 h deadline and the interrupted `run.json` kept as
`<obsid>.run.attempt1.json`.

**Cost model, measured after the fact from each run's own `run.json`** (all four
are "ok", 16 workers, and the numbers are the processing stage alone, download
and environment excluded): `1729091` 36 HDUs in 851.6 s, `1853217` 40 in
3087.6 s, `1853221` 40 in 5129.0 s, `1853389` 40 in 2828.4 s. That is 380-2050
CPU-seconds per HDU — a 5x spread across fields at identical settings, so
**per-HDU cost is set by field content, not by the pipeline's fixed overhead**,
and a budget sized from one exposure does not carry to another. (An earlier note
in this file quoted 586-1043 s per HDU; that was read off a mid-flight snapshot
of a strained first attempt and is superseded by these whole-run numbers.)

**The three that did not finish are in a different cost regime, and it is
measurable.** They were re-run with a 4 h ship deadline, and were then cut by a
**3 h in-pod `WM_TIMEOUT`** that the launch itself set — a budget chosen before
any of these numbers existed. What the logs and the mount show, 2 h in:

| obsid | HDUs written | mask bytes | mask untouched for | median candidate points |
|---|---|---|---|---|
| `1774908` | ~15 of 36 | 8.74 MB | 55 min | 6,816 |
| `1851368` | ~1.5 of 40 | 0.90 MB | 48 min | 25,240 |
| `1851960` | ~1.7 of 40 | 0.99 MB | 64 min | 154,380 |

**All seven then finished, and the bound is visible in their own logs.**
`1851368` and `1851960` on the bounded tree: 40/40 HDUs, `status=ok`, in 2.76 ks
and 2.82 ks (~46 min each) — the same two exposures that the 3 h cap cut partway
through with 1-2 HDUs written in 2 h. Their logs report fits capped at 20,000
from up to 164,666 candidates and 18 cascades stopped at 200 rejected models,
while STREAK content comparable to the quiet fields is still found. `1774908`
finished attempt 3 in **2,209 s** with 36/36 HDUs and `status=ok`.

against 2,341-4,592 median candidates on the four that finished in 47-85 min.
The mechanism is in `weightmask/streaks.py`: `_detect_trails_sparse_ransac` runs
a **fixed-trial fit over every residual candidate pixel on each pass**, and its
rejected-model path consumes only `min_inliers` pixels before looping, so
per-pass cost and pass count both grow with crowd size. The function's own
comment already says it: "heavily crowded fields need a candidate cap before the
RANSAC search". A non-sidereal frame with long star trails is the crowded case,
and the loop is what made a 40-HDU exposure need more than the 3 h cap.

**Two compounding causes, both measured; one fixed in code and one in the
launcher.** Profiling one real HDU of `1851960` on a quiet machine (cProfile, one
worker) put the whole HDU at 242 s with `streaks` 131 s, of which 72 s was the
sparse RANSAC — 24 passes, each a 300-trial fit over 157k candidates. The pod ran
the same HDU 10-25x slower, and the candidate count alone does not explain it:

1. **The rejection cascade had no bound.** The loop ends only when the candidate
   set drains, one rejected model at a time, and the same HDU took 230+ passes on
   a pod against 24 locally — the cascade length is chaotic in the candidate set,
   so it cannot be budgeted by averaging. The worst case is
   `candidates/min_inliers` (~6,800 passes at ~3.5 s each on one core).
   `weightmask/streaks.py` now bounds it: `sparse_ransac_params.max_candidates:
   20000` caps the set entering each fit (a uniform stride, deterministic and
   spatially even) and the model is re-scored against *every* candidate, so the
   cap bounds the search and never the evidence; `max_rejected: 200` bounds the
   cascade. Both are inert on the fields the stage was tuned on (2.3-4.6k
   candidates, tens of rejected models per HDU) and both are schema-validated.
   Checked on a real crowded HDU: 142,037 candidates reduced to 20,000 fitted,
   with three real trails found (lengths 928-1,278 px, density 0.66-0.69).
2. **Library thread pools were oversubscribed 16x.** `pod_run.py` deliberately
   drops the image's `OMP_NUM_THREADS`/`MKL_NUM_THREADS` (they perturbed the
   parallel-efficiency campaign), but dropping them lets BLAS/OpenMP fall back to
   the *node's* CPU count — 192 on these pods — per worker, so 16 HDUs asked for
   up to 16x192 threads on the 16 cores they were granted. Measured on `1851960`
   with 16 workers: **per-HDU wall 2.6-4.0 ks before, 0.8-1.0 ks with the pools
   pinned to one thread.** `pod_run.py` gains `thread_hygiene()` and
   `WM_THREAD_PIN=1`, pinned by a test.

With both, the crowded HDUs take **320-410 s** now (`streaks` 98-315 s) at 16
workers, against 2,651-5,934 s before; a 40-HDU exposure that the 3 h cap used to
cut partway through reaches ~50% in ~30 minutes. Bounded runs also show the
bound working in situ: fits capped at 20,000 from up to 164,666 candidates, 18
cascades stopped at 200 rejected models, and real trails still found (e.g.
length 1,594.7 px, 1,149 inliers, density 0.721). **The bound is a deliberate
behaviour change on crowded HDUs**, so it is inert on the quiet fields the four
verified exposures came from only where their candidate counts stay under it
(2.3-4.6k median, up to 99k on their busiest HDU); those four artifacts were
produced by the pre-bound code and their `run.json` records still describe
themselves. `benchmarks/canfar/run_exposures.sh` is the committed launcher for
this campaign, carrying `WM_THREAD_PIN=1`, the explicit `WM_TIMEOUT`, and the
attempt-record and partial-mask handling that used to live only in ad-hoc shells. Scaling the pod up was never
the lever: a 32-core / 192 GB request is rejected with HTTP 400 by the create
endpoint while three 16-core pods are running, so the relaunch keeps the proven
16-core shape. `1774908` finished its attempt 3 with `WM_TIMEOUT=21600` (6 h);
`1851368` and `1851960` were re-launched on the re-staged tree (attempt 3,
`WM_THREAD_PIN=1`, `WM_TIMEOUT=10800`), with each cut attempt's record kept as
`<obsid>.run.attempt2.json`. All seven are then re-verified in a separate
session.

**Review path, independent of the processing sessions.**
`benchmarks/canfar/verify_exposures.py` runs in a separate session, trusts
nothing the driver recorded, and recomputes every quality-bit count from the mask
pixels. It re-checks the mask against the digest `run.json` claims, cross-checks
the shipped stats table column by column against the pixels, reports a short or
unreadable mask as a hole rather than as a small clean read, and writes
`results/ns-review/streak_summary.tsv` so a review needs no pod logs. That test
caught a real defect in its own first version: the summary read
`COSMIC_RAY`/`SATURATED`, which are not `QUALITY_BITS` names (`CR`/`SAT` are), so
two columns would have shipped as zeros that look like a detector with no
cosmics in it.

**Verification of all seven, in a separate session (2026-10-14).** Session
`y1cothww` re-derived every number from the mask pixels on the mount and wrote
`results/ns-review/streak_summary.tsv` (7 rows) in 22.5 s, `rc=0`:

| obsid | status | HDUs | holes | STREAK px | HDUs with streaks | digest | stats |
|---|---|---|---|---|---|---|---|
| `1729091` | ok | 36/36 | 0 | 1,902 | 2 | ok | ok |
| `1774908` | ok | 36/36 | 0 | 3,828 | 3 | ok | ok |
| `1851368` | ok | 40/40 | 0 | 452,720 | 15 | ok | ok |
| `1851960` | ok | 40/40 | 0 | 266,391 | 16 | ok | ok |
| `1853217` | ok | 40/40 | 0 | 210,479 | 7 | ok | ok |
| `1853221` | ok | 40/40 | 0 | 385,059 | 12 | ok | ok |
| `1853389` | ok | 40/40 | 0 | 446,842 | 21 | ok | ok |

All seven are `status=ok` with no holes, so digest and stats both match on every
exposure. **The sentence that used to follow here -- that the two crowded fields
"now carry trail content in the same regime as the good ones" -- is wrong and is
retracted; see the audit below.** More STREAK pixels is not more trail.

The artifact itself, read back off the mount by a separate 1-core session rather
than from a processing pod's log: `/arc/projects/mlao/cfhtcast/weightmask-perf/
results/ns-review/streak_summary.tsv`, 977 bytes, 20 columns, `sha256
886ea0c5a4c003a323cfdf47ab1064299e218f03baa623cc5d80b3507e622932`. Its processing
and quality columns:

| obsid | seconds | mask bytes | STREAK px | STREAK px/HDU | CR px | SAT px | flagged px | npix |
|---|---|---|---|---|---|---|---|---|
| `1729091` | 851.6 | 21,346,560 | 1,902 | 52.8 | 29,532 | 216,457 | 33,904,637 | 353,092,608 |
| `1774908` | 2,208.7 | 19,679,040 | 3,828 | 106.3 | 24,822 | 286,925 | 30,109,294 | 353,092,608 |
| `1851368` | 2,768.7 | 19,889,280 | 452,720 | 11,318.0 | 17,089 | 402,453 | 35,603,417 | 392,325,120 |
| `1851960` | 2,828.1 | 30,254,400 | 266,391 | 6,659.8 | 27,627 | 827,494 | 63,938,067 | 392,325,120 |
| `1853217` | 3,087.6 | 24,013,440 | 210,479 | 5,262.0 | 23,331 | 347,433 | 42,548,727 | 392,325,120 |
| `1853221` | 5,129.0 | 24,053,760 | 385,059 | 9,626.5 | 23,546 | 327,748 | 42,939,591 | 392,325,120 |
| `1853389` | 2,828.4 | 27,918,720 | 446,842 | 11,171.0 | 44,524 | 618,744 | 51,271,984 | 392,325,120 |

**Audit: on these fields the STREAK product is detector defects and star trails,
not trails (2026-10-14).** The review images prompted the question, and the
answer is that essentially none of the 1,767,221 STREAK pixels in this campaign
is a satellite trail. Auditing the shipped masks by shape (`/tmp/wm/
audit_streaks_payload.py`, run in session `zofvhumg` against the mount's own
masks) splits them three ways:

| obsid | STREAK px | axis-aligned row bands | axis-aligned column groups | diagonal groups / star-trail angle | thin lines |
|---|---|---|---|---|---|
| `1729091` | 1,902 | 0 | 0 | 1,375 | 524 |
| `1774908` | 3,828 | 0 | 0 | 3,023 | 805 |
| `1851368` | 452,720 | 65,940 | 13,832 (plus 9-16 px bands in the group column) | 368,515 | 4,433 |
| `1851960` | 266,391 | 0 | 41,262 | 204,151 | 20,978 |
| `1853217` | 210,479 | 55,224 | 13,745 | 129,261 | 12,249 |
| `1853221` | 385,059 | 67,549 | 0 | 311,146 | 6,364 |
| `1853389` | 446,842 | 82,790 | 0 | 362,238 | 1,761 |

(The "diagonal groups" column is the audit's catch-all bucket: its largest
members are bands 9-16 px wide running 1,900-4,600 rows down the detector, i.e.
the same defect class as the narrow ones, plus the star-trail angle structures.)

What each class is, and the evidence:

- **Axis-aligned bands that recur in independent exposures.** Horizontal bands of
  exactly 3, 6, 7 or 8 rows spanning 2,050 of the 2,112 columns, and vertical
  groups 3-16 px wide spanning 2,400-4,612 of the 4,644 rows. They appear at the
  *same detector coordinates* in different exposures of the same CCD -- mask HDU 1
  is flagged at column 1,624 in both `1851368` and `1851960`, and HDU 6/16/17/23/38
  carry bands at rows 3 and 7 in three or four of the seven exposures each. Sky
  cannot do that and a trail cannot either; a detector defect must.
- **The star trails.** `1851960` HDU 17 is a non-sidereal frame, so every star is
  trailed: 134 of its 137 elongated sources sit at 30-45 degrees (mean 40.2,
  median length 55 px). The flagged component in that HDU is at **40.2 degrees**,
  602x712 px over 931 px of length -- and its pixels are at sky level: median
  +9 ADU over a sky of 571 with rms 22, with four bright heads along it (values
  1,044 / 952 / 882 / 770) and sky everywhere between. Two or more star trails
  that happen to be near-collinear are enough for the RANSAC pass (which draws a
  line between the segment endpoints), so the mask paints a 931-px line across
  931 px of empty sky. The campaign's "thin lines" are this, at the field's trail
  angle: 20,978 px at 40 degrees in `1851960`, 13,136 px at 40 degrees in its
  thin-line histogram.
- **The edge band.** The largest single object in the local 1851960 HDU 17 mask is
  7,055 px: rows 0-7 across 952 columns, at sky level (median 572 against 574 in
  the neighbouring rows). It is neither a trail nor a defect class any stage owns;
  it is the frame edge being detected.

The 20-sigma brightness veto cannot catch any of this: `_drop_bright_components`
drops what is *too bright* (a star's head, a saturated column), and every false
positive here is at or below the sky. Nor is it a tuning problem: the gates in
`detect_streaks` test geometry (thin, long, straight) and these are all thin,
long and straight. The two properties that separate them are absent from the
stage: alignment with the *detector axes* (a defect runs the full span of one
axis at constant narrow width, which no sky feature does), and alignment with the
*field's own star-trail direction* (a satellite trail has its own angular rate, so
on a non-sidereal frame it is the one direction the star trails do not share --
unless it happens to be parallel, which the length check then has to settle).

**The two fields that were questioned, measured (session `ynjdznld`).** For
`1729091` and `1774908` the field's own star-trail angle and length were measured
from the frame, and every flagged component compared against them:

| obsid | mask HDU | field star trails | flagged component | shape | brightness | verdict |
|---|---|---|---|---|---|---|
| `1729091` | 4 | 16.0 deg, 110 sources, median 21 px | 1,375 px, 174x187, 43.5 deg (**27.5 off**) | 10.7 px thick, profile flat (868-938, one 1,403 end) | +142 ADU = **5.2 rms** | not a trail and not a star trail: a bright, thick, flat feature (bleed/ghost class), unresolved |
| `1729091` | 12 | 17.3 deg, 87 sources | 524 px, 38x176, 11.5 deg (5.8 off) | 2.9 px thick, profile 1,473 at one end then 774-792 | +14 ADU = 0.56 rms | a line drawn from a bright spot across 176 px of sky |
| `1774908` | 19 | 31.7 deg, 89 sources, median 27 px | 404 px, 91x136, 33.5 deg (**1.8 off = parallel**) | 2.5 px thick, bright ends, sky between | +11 ADU = 0.93 rms | star-trail chain |
| `1774908` | 29 | 34.9 deg, 65 sources | 3,023 px, 175x305, 28.9 deg (6.0 off) | 10.1 px thick, profile flat (231-253) | +51 ADU = 2.3 rms | thick flat band, not trail-shaped |
| `1774908` | 34 | 32.4 deg, 95 sources | 401 px, 127x135, 43.2 deg (10.8 off) | 2.2 px thick, bright knots with sky between | +16 ADU = 1.26 rms | star-trail chain |

The satellite signature -- one line at the PSF's width, roughly uniform in flux,
at an angle the field's trail family does not share -- appears in none of them.
Every flagged component is either aligned with a detector axis, or within 6 deg of
the field's own trail angle with sky between its heads, or 10 px thick and flat
(which a satellite trail is not). The star-trail length itself is a measurable
per-frame quantity (21 px here, 27 px there, 55 px on `1851960`) and is the
natural length scale a veto needs, since a satellite's trail length is set by its
own angular rate rather than by the telescope's.

Two config facts explain why none of the existing gates bite. `max_support_width:
12` admits a 10 px band: the profile gate measures the transverse MAD of the
component, which for these bands is ~4.5 px, not the 10 px an eye measures.
And `min_length: 128` was set to "reject a measured persistent 116 px detector
feature" -- so a persistent detector feature was already known, and the campaign's
axis-aligned ones are 1,700-4,612 px long, far past any length floor.

**Consequences, stated plainly.** The `ns-review` campaign delivered what it
promised mechanically -- seven exposures processed, artifacts digest-verified,
no holes -- but the STREAK numbers in the tables above are a false-positive
census, not a trail detection, and `1851960`'s "trail content" was the artifact of
this audit rather than evidence that the stage works on these fields. The bound
comparison is affected too: on the one HDU measured both ways, the bounded run
*added* the 2,132-px 40-degree false positive (7,544 px -> 9,676 px), so "inert on
the quiet fields" was too strong -- capping the candidate set changes what the
RANSAC accepts, and on this field it changed the answer.

`seconds` is each run's own processing figure and confirms the cost model: seven
exposures, 36-40 HDUs each, now land in 852-5,129 s (**19,702 s of processing
for the campaign**), where the three crowded fields previously exceeded a 3 h cap
with a handful of HDUs written. `bad_px` is 0 on all seven and `no_data_px` is
one full HDU per exposure (13,059,072 = 36-CCD fields, 14,510,080 = 40-CCD), and
`digest_match`/`stats_match` are `True` throughout. Verification ran on the same bounded tree as processing, so it
clears the artifacts, not the bound: the bound's own evidence is the in-log
capping above plus the local HDU check (142,037 candidates -> 20,000 fitted,
three real trails).

**Review images: what the streak stage did, at the pixel level (2026-10-14).**
A table of STREAK pixel counts says something happened but not whether it happened
on a real trail, so `benchmarks/canfar/plot_trail_cutouts.py` renders one sheet
per exposure: up to three cutouts of 256x256, each the *real* frame on the left,
zscale stretched on its own pixels (contrast 0.25, so the crop is read at the
contrast a reviewer would pick by hand rather than at min/max where a few
saturated stars flatten the trail), beside the same crop of `<obsid>.mask.fits.fz`
on the right with one colour per defect class and unpainted pixels white. Both
panels use `origin="lower"` so a feature sits at the same place in both. Masks
come from the mount, the exposure is re-downloaded to scratch, and the sheet plus
an `index.json` go to `results/ns-review/cutouts/` (pod session `xv5g50kx`,
8 sheets, 0 failures).

The one non-obvious decision is that a panel is only drawn when the masked line
is genuinely brighter than the sky, because the naive probe is wrong and was
measured to be wrong: an intermittent trail is dashed, the mask closes the gaps
between segments, so about half the masked pixels are sky and the *masked median
sits on the sky median* (1851960 HDU 17: 574.3 vs 571.2). What the eye sees is
the ridge, so the test is the 90th percentile of the masked pixels against the
window's brightest clean pixels (its 99.9th percentile): a candidate below 1.0 is
reported under `skipped` instead of being published as a trail. Across the seven
exposures three candidates were skipped that way (one each in `1774908`,
`1851960`, `1853217`) and every drawn panel passed with 1.009-1.669.

| obsid | cutouts drawn | ridge (masked p90 / clean p99.9) | candidate skipped |
|---|---|---|---|
| `1729091` | 2 (ccd03, ccd11) | 1.159, 1.383 | 0 |
| `1774908` | 2 (ccd18, ccd33) | 1.669, 1.642 | 1 (line on sky) |
| `1851368` | 3 (ccd00) | 1.009, 1.182, 1.198 | 0 |
| `1851960` | 3 (ccd18, ccd22, ccd29) | 1.160, 1.172, 1.196 | 1 |
| `1853217` | 3 (ccd00, ccd31, ccd37) | 1.164, 1.224, 1.228 | 1 |
| `1853221` | 3 (ccd00) | 1.163, 1.185, 1.185 | 0 |
| `1853389` | 3 (ccd00, ccd22) | 1.021, 1.104, 1.139 | 0 |

Three caveats, stated rather than left to be discovered. The exposures were
re-downloaded for this, so the sheets are the same observation ids and the same
CADC files by byte count (all seven match the processing runs' `exposure_bytes`
exactly) but their digests were not compared against `run.json`. Windows are
clamped inside the frame, so an edge candidate cannot be centred exactly --
`1774908` HDU 19 is at `y0=0` and its trail runs out of the top. And selection
prefers the longest segments, so one CCD can fill a sheet: `1851368`, `1853221`
and `1853389` are all streaks in `ccd00`.

One thing to look at rather than accept: `1729091` is nearly streak-free (1,902 px
in 2 of 36 HDUs) while the other six have thousands of pixels across 3-21 HDUs.
Same instrument, same tracking mode, so either that field genuinely has far less
trail or the stage is missing it there; the mask is on the mount for exactly that
kind of check.

**The acceptance test, and the first number it produced (2026-10-14).**
`benchmarks/streak_nonsidereal_acceptance.py` measures the one thing the campaign
did not: a frame with a known star-trail family, a control run with *no* satellite
injected, and injected star trails, chains and axis-aligned bands as negatives.
On the control, where the answer must be an empty mask, the stage flags **4,526
px** in a synthetic 1024-squared fixture and **4,379 px** in a real crop of
`1851960` HDU 17. Repeating the real crop **with the flat the campaign's own crate
downloaded** (`flat_08Bm01_r.fits.fz`) takes that to **0 px**, with no stage
change: the upstream mask grows from 56,520 to 84,467 px and the defect class
never reaches the stage. The campaign's pod command passes no `--flat`
(`run_exposures.sh:68`) and `process_exposure.py` applies no persistence prior
either (its own docstring), so the dominant class in the tables above is a
configuration omission before it is a stage bug. Plan, ranked, with the veto
insertion points, in `streak_nonsidereal_remediation_2026-10-14.md`.

### CANFAR egress restored, and the "flake" that was our own transport (2026-10-13)

**The pod-side DNS outage is over.** Re-probed from a pod
(`benchmarks/canfar/probe_dns_outage.py`, 15 samples at 6 s intervals) the same
hosts that returned 0/36 on 2026-10-12 now answer: `www.cadc-ccda…` **15/15**,
`ws.cadc-ccda…` **15/15**, `github.com` **15/15**, `conda.anaconda.org`
**15/15**, `pypi.org` and `files.pythonhosted.org` **15/15**. Two names still
give 0/15 — `cadc-ccda.hia-iha.nrc-cnrc.gc.ca` and `cadc.nrc.ca` — and that is
not a fault: neither is a hostname, so they were never evidence of the outage
even when they were read alongside it. The control plane is healthy this
session too: `canfar ps` answered, and every create below was accepted on
**attempt 1**, against 10 of 20 on 2026-10-12. Both transports the campaign
scripts assume therefore work again — `git clone` and CADC — so the earlier
conclusion that "pod egress is not CADC only" describes that week's fault, not
the platform.

**The `FATAL: WM_TREE and WM_COMMAND are both required` recorded above as a
control-plane flake is a defect in `pod_ship.py`.** It reproduced on the first
attempt of the next job, which is what a flake should not do, so it was
measured instead of retried. A payload that dumps `os.environ` shows the
platform's `env` parameter delivering `WM_TREE`, `WM_TIMEOUT` and
`WM_RESULTS_TAG` while dropping `WM_COMMAND` entirely:

| env value sent | what the pod saw |
|---|---|
| `/arc/projects/…/cand-ce16410c5cae` | arrived |
| `7200` | arrived |
| `pytest -q` | arrived |
| `a b`, `hello\tworld`, `x--y`, `--help`, `--`, `---x` | arrived (spaces, tabs and dashes are not the problem) |
| `-p no:cacheprovider` | arrived |
| `{py} -m pytest -q -rf --tb=short -p no:cacheprovider` | **absent** |
| `--tb=short` | **absent** |
| `a --tb=short b` | **absent** |
| `k=v`, `--foo=bar`, `=a` | **absent** |
| `a=` | arrived as `a` |

So the loss tracks `=` in the *value*, and every shell command acquires one
sooner or later. `pod_ship.py` no longer sends environment through that
parameter at all: the map is serialised into the payload as argv token 1
(`base64(json(env))`) and an `__import__`-only bootstrap applies it before the
script runs, which is why the bootstrap must contain no space — a space is an
argument boundary once the server has re-split `cmd` and `args`. argv was
already proven lossless for ~900 KB of base64 chunks by the staged-source run,
so this adds no new trust. Verified end to end against the live platform: the
exact `WM_COMMAND` that used to vanish now arrives byte-identical.

**A second trap, found by trying to falsify the first.** In that same dump a pod
set `WM_SHORTCODE=a:b:c` and printed `a:b:c`, while the log showed `a🅱c` — and
an earlier bisect had "proved" an emoji-shortcode expander rewriting values. It
is the *log* that rewrites: printing the map as base64 taken **inside the pod**
yields `a:b:c` exactly. `canfar logs` is therefore not a faithful channel for
exact strings, and it also hard-wraps long lines. The docstrings say so, and the
regression test splits the real args string the way the server does rather than
trusting a reading of the log. Two corrections came out of this: the emoji
expansion is a rendering artifact, not a transport effect, and the `=` rule
above is a working correlation — proven for the case that mattered (pod
behaviour, not log text: the runner genuinely did not see `WM_COMMAND`) but
inferred for the rest from log output.

**The pod-side suite is green at the tip, and the three failures were
environmental.** `repos/cand-ce16410c5cae` built its own environment (`pixi
install`, 58 s, attempt 1, shared package cache) and then ran
`{py} -m pytest -q -rf --tb=short -p no:cacheprovider` for 26:04: **1306 passed,
4 skipped, 1 xfailed, 852 subtests, rc=0** — with nothing failing. The
`WM_COMMAND` that used to vanish arrived intact, so the transport fix is
validated in the real workload and not only in the harness.

The counts are fully accounted for against the local run (1309 passed, 1
skipped), which matters more than the green colour: `tests/test_contract.py`
skips `torchfits is optional` in both, and exactly three tests —
`test_both_features_are_recovered_at_every_veto_setting`,
`test_known_hot_column_does_not_overlap_the_streak_mask`, and
`test_production_input_roi_stays_out_of_streaks` — call `_production_inputs`,
which raises `SkipTest` when `benchmark_data/megacam` is absent, and a freshly
staged pod tree carries no fixture inputs. 1309 − 3 = 1306 passed with 1 + 3 = 4
skipped, so the pod runs the same suite rather than a quietly smaller one.

Why the three 2026-10-12 failures happened is still an *inference*, and is
recorded as one. Both `tests/test_perf_protocol.py` failures are `_pixi_argv`
tests that shell out to `pixi run` **inside the tree**, which cannot work until
an environment exists there; that run also predates `442bfc8`, the fix that made
the reuse path (`WM_TREE_ORIGIN`) actually execute, so a copied tree would have
had no environment at all while DNS was down. The original assertion text was
never captured and the run is not reproducible now that the env is built and
egress works, so this is the best-supported explanation rather than a
measurement.

**The mount, inventoried before spending jobs.** `repos/` holds `bootstrap`
(`124df7f`, stale `M` on two files), `bootstrap2` (`d84163e`) — both without a
built env — and `cand-313cb436e7f9`, which has no `.git` (it was extracted from a
tarball) but whose `.pixi/envs/default/bin/python` satisfies every required
import, so a job can reuse that interpreter instead of paying for a solve. A new
tree was staged at the candidate tip: `repos/cand-ce16410c5cae`, tarball
663,756 B with sha256 `c69e40d0…` verified in the pod, 176 files, 3,535,750
bytes. One thing the inventory does **not** show is the gate's fixture inputs:
the only CFHT exposure on the mount is
`cfhtcast-bench/downloads/2240997p.fits.fz`, which is not the fixture set, so a
pod gate run has to fetch its inputs.

`/usr/local/bin/pixi` is `0.81.0` in `images.canfar.net/astroai/base:latest`, so
the image needs no pixi install either.

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

**Correction 2026-10-12: the vote floor is the third cause, not the first.** The
table above is not wrong, but it reports the accumulator's maximum *after* the
pre-mask zeroed the binned SNR image, which makes the floor look like the whole
story. Re-deriving the stage's own binary images from the exact
`(data_sub, bkg_rms_map, existing_mask)` it receives shows the accumulator is not
what starves. With the pre-mask ignored there are 741-1850 lit binned cells per
case; with it applied there are 7-21. A floor cannot bind on an image that is
already almost empty.

`tests/benchmarks/wm16_hough_starvation.py` (new, committed, no network) measures
the trail-specific quantities. `cells` is how many binned cells the trail's line
crosses — the trail's vote ceiling even if every one were lit and unmasked:

| case | trail px | trail premasked | frame premasked | cells | lit unmasked (strict/nominal) | lit but masked (strict/nominal) |
| --- | --- | --- | --- | --- | --- | --- |
| `synthetic_sparse` | 937 | 17.6% | 20.6% | 76 | 1 / 3 | 0 / 0 |
| `synthetic_complex` | 1390 | 69.0% | 63.8% | 91 | 0 / 6 | 20 / 28 |
| `synthetic_gradient_step` | 1390 | 49.5% | 62.3% | 91 | 0 / 1 | 31 / 48 |
| `synthetic_variable_width` | 2476 | 38.3% | 48.5% | 74 | 0 / 2 | 15 / 22 |
| `synthetic_elongated_sources` | 1390 | 86.4% | 63.8% | 91 | 0 / 0 | 25 / 36 |

The order of causes is therefore:

1. **The pre-mask is what starves the stage.** At 384 px with 60-80 stars the
   saturation, cosmic-ray and object masks together cover 48.5-63.8% of the
   frame and, more to the point, 17.6-86.4% of the trail's own pixels. The
   trail's lit cells are almost entirely on the masked side (`synthetic_elongated_sources`:
   0 lit unmasked against 36 lit masked). `_drop_premasked_components` then
   encodes the same fact as policy — a candidate more than 25% premasked is
   dropped as "not a new finding" — so the pipeline deliberately declines to
   report a trail the earlier stages already covered.
2. **Releasing the mask alone would still not find it.** Only 0-6 unmasked
   binned cells on the trail's line clear `thresh_sig: 2.5`, against 74-91 that
   the line crosses. So the trail is largely below the detector's significance
   floor at this scale as well.
3. **The vote ceiling is real but latent.** The trail's line crosses only 74-91
   binned cells at 384 px, so its absolute maximum is ~91 votes against
   `min_votes: 100` — the arithmetic above is right, and the floor would become
   binding the moment causes 1 and 2 were removed. This is precisely why the
   rejected one-line scaling could not have fixed the aggregate: it raises the
   ceiling above a binary image that contains at most 6 trail cells, not the ~90
   the argument assumes. Order of repair is mask and significance first, floor
   third.

The instrument asserts the generator is reproducible from the case parameters
before it reports, so a fixture change cannot silently invalidate these numbers.

**A background-absorption hypothesis was tested and refuted.** An exploratory
2x2 crossing trail flux with frame size measured the contrast the pipeline's sky
subtraction retains: 99.9-100.4%. SEP is not eating the trail. That 2x2 is
*not* used to choose a fix, because its columns are confounded — the generator's
background scales with frame size, so the same `streak` flux is not the same
significance at a different size (median-subtracted 90th-percentile trail SNR:
62.4 at 384 px versus 3.08 at 1536 px), and the ground-truth trail mask includes
off-core pixels from the square dilation in `mark()`. One observation survives as
a lead rather than proof: at 1536 px the same regime and seed reaches F1 0.426
with a 6x flux where 384 px reaches 0.000 at both fluxes, so the detector does
recover a trail once the frame is large enough for it — the fixture at 384 px
puts 60-80 stars into 1/28th of a reference amp's area, at 28x the surface
density for the same star count.

**Consequence for the work list.** Both "fix the generator" and "fix the
detector" remain open, and neither is a one-liner: the generator must be
scale-consistent (star density and trail significance at 384 px), and the
detector must stop treating an upstream mask as proof of absence. Choosing
between them changes the injection floors, which the 2026-10-09 decision
explicitly defers to an owner decision that re-derives them against the
HEAD-`main` corrected score. So this correction changes the *diagnosis* only;
WM-16 stays open and the floors stay untouched.

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

Root cause corrected by measurement on 2026-10-12: the vote floor is the third
cause, not the first. The pre-mask starves the accumulator (17.6-86.4% of each
trail's own pixels are already masked when the streak stage runs) and only 0-6
unmasked binned cells clear `thresh_sig`. Re-measure with
`pixi run python tests/benchmarks/wm16_hough_starvation.py`; do not re-derive a
floor fix from the accumulator-maximum table alone.

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
