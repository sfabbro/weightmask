# Ignored-artifact inventory and retention record (2026-10-12)

Status: **inspection complete — Tier B purge executed (2026-10-12); one reversible relocation performed**

Candidate commit inspected: `b2e4da6` (`wip/wm-remediation-0.2.1`, PR
[astroai/weightmask#75](https://github.com/astroai/weightmask/pull/75)).

Method: read-only inventory (`du -sh`, `find`, `cat`), plus one `mv` of already
superseded build artifacts out of the release staging path. No benchmark input,
report, log, or manifest was deleted. This record is the small provenance the
retention policy in `docs/releasing.md` requires before any large product is
purged.

## 1. Change made: `dist/` emptied by relocation, not deletion

`release_check.py` refuses `--outdir dist` while *any* entry exists in `dist/`
(it tests `any(outdir.iterdir())`, so nested directories count). The five
superseded generations below therefore blocked the documented publishing
command. They are all referenced by trajectory records, so the retention policy
requires keeping them; they were **moved** to
`test_outputs/harness/dist-archive/` (still on disk, outside the staging path)
rather than removed.

| Moved from | Moved to | `SHA256SUMS` recorded |
|---|---|---|
| `dist/0.2.1-all-files/` | `test_outputs/harness/dist-archive/0.2.1-all-files/` | wheel `25aed797…`, sdist `5cdfef6f…` |
| `dist/0.2.1-candidate/` | `test_outputs/harness/dist-archive/0.2.1-candidate/` | wheel `c45d496c…`, sdist `94a8b4fb…` |
| `dist/0.2.1-no-compat/` | `test_outputs/harness/dist-archive/0.2.1-no-compat/` | none present |
| `dist/0.2.1-release/` | `test_outputs/harness/dist-archive/0.2.1-release/` | wheel `db1450d3…`, sdist `2ebe9bf0…` |
| `dist/0.2.1-verified/` | `test_outputs/harness/dist-archive/0.2.1-verified/` | wheel `84fa34fc…`, sdist `be02764a…` |

All five are stale generations of version `0.2.1` built 2026-10-03…10-05, before
the corrected injection/recall metrics invalidated their science evidence. They
are **not** evidence for `b2e4da6`. `dist/` itself no longer exists; the staging
directory must be absent or empty, and it is now absent. Full hashes remain in
the per-directory `SHA256SUMS` inside the archive.

Trajectory records that reference these paths, and which therefore justify
retaining them: `2026-10-03-release-audit.md` (`0.2.1-verified`),
`2026-10-04-all-files-release-audit.md` (`0.2.1-all-files`),
`2026-10-04-release-qualification.md` (`0.2.1-no-compat`),
`2026-10-05-release-finalization.md` (`0.2.1-candidate`),
`2026-10-05-release-ready.md` and `2026-10-05-qualified-release.md`
(`0.2.1-release`).

## 2. `test_outputs/` inventory (measured before the purge: 128 GB total)

| Path | Size | Kind |
|---|---|---|
| `test_outputs/perf/` | 57 GB | 63 `.weight.fits` products + 31 small files |
| `test_outputs/cadc_megacam_p_gt30_20261005/results/` | 48 GB | per-exposure weight/ivar/sky products |
| `test_outputs/cadc_megacam_p_gt30_20261005/inputs/` | 2.9 GB | CADC cutouts (re-downloadable) |
| `test_outputs/megacam_real/` | 5.9 GB | products + JSON/MD provenance |
| `test_outputs/megacam_real_flat/` | 5.9 GB | products + JSON/MD provenance |
| `test_outputs/megacam_real_long/` | 2.1 GB | products + JSON/MD provenance |
| `test_outputs/megacam_real_long_nostreak{,_n1faint,_residual}/` | 6.0 GB | products + provenance |
| `test_outputs/harness/` | 655 MB | current + historical release evidence, `trail4.npz` (58 MB), `dist-archive/` |
| `test_outputs/benchmarks/` | 12 MB | synthetic suite metrics/summary |
| `test_outputs/streaks/`, `mask_*.fits`, `simulated_raw.fits` | ~8 MB | small fixtures |

## 3. Retention verdict (analysis that the decision in §4 was made from)

Per `docs/releasing.md` ("Retention policy for benchmark and release
evidence"), a large product is a purge candidate **only** once its small
provenance has been checked and the result is reproducible or explicitly
labelled a retained one-off. Every large directory here has that provenance:

- `test_outputs/perf/` → `megacam_perf.json`/`.md`, `megacam_perf_after.json`,
  `exposures*.json`, `flats.json`, `cprofile_hdu*.txt`, `*_run.log`,
  `checkpoint_after.json`. Reproducible with `pixi run exposure-time` on the
  pinned MegaCam inputs.
- `…/cadc_megacam_p_gt30_20261005/` → `manifest.json`, `run_config.yml`,
  `summary.json`, `summary.csv`, `summary.md`, `timings.json`, `sample.csv`,
  `logs/`. Reproducible from `inputs/` (or by re-fetching from CADC).
- `test_outputs/megacam_real*/` → `megacam_perf.json`/`.md`,
  `real_output_metrics.json`, `streak_geometry.json`, `records.json`.
- `test_outputs/harness/` → manifests, gates, scores, `SHA256SUMS`.

| Item | Verdict | Reason |
|---|---|---|
| `perf/*.weight.fits` (57 GB) | purge candidate | provenance retained; regenerable by re-running the documented benchmark |
| `cadc…/results/` (48 GB) | purge candidate | `manifest.json` + `summary.*` + `timings.json` retained; inputs still present |
| `cadc…/inputs/` (2.9 GB) | retain for now | re-downloading costs CADC time; keep while the campaign result is live |
| `megacam_real*/` products (~14 GB) | purge candidate | per-directory provenance retained |
| `test_outputs/harness/0.2.1-release/` | **retain** | explicitly referenced by the remediation plan as invalid-but-historical evidence |
| `test_outputs/harness/release-evidence.9b5231a-2026-10-10.json` and `science-gate.9b5231a-2026-10-10.md` | **retain** | superseded copies preserved before the 2026-10-12 re-run overwrote the current paths |
| `test_outputs/harness/0.2.1-qualified/`, `0.2.1-final/` | **retain (empty)** | referenced by `2026-10-05-qualified-release.md`; both are empty, so a report the trajectory names is missing — historical gap, not a purge target |
| `test_outputs/perf/.thumbs/` (4 KB) | purge candidate | macOS thumbnail cache, no evidentiary value |

## 4. Decision taken and executed: Tier B

Large deletion is not automatic by policy, so the operator chose the tier on
2026-10-12. **Tier B** was executed: regenerable products *and* the CADC cutouts
were removed; every manifest, report, log, JSON, hash, and pinned input was
retained. Commands actually run, in this order:

```bash
rm -f  test_outputs/perf/*.weight.fits
rm -rf test_outputs/perf/.thumbs
rm -rf test_outputs/cadc_megacam_p_gt30_20261005/results
rm -rf test_outputs/cadc_megacam_p_gt30_20261005/inputs
find test_outputs/megacam_real test_outputs/megacam_real_flat \
     test_outputs/megacam_real_long test_outputs/megacam_real_long_nostreak \
     test_outputs/megacam_real_long_nostreak_n1faint \
     test_outputs/megacam_real_long_nostreak_residual \
     -name '*.fits' -delete
rm -f  test_outputs/perf/*.fits      # second pass: the remaining .mask/.weight.<stage> products
```

Measured outcome: `test_outputs/` went from **128 GB to 689 MB** (≈127 GB
reclaimed). Final state:

```text
test_outputs            689M
test_outputs/harness    656M
test_outputs/benchmarks  12M
test_outputs/perf       8.2M   (ten .fits.fz inputs + 42 provenance files)
test_outputs/streaks    3.6M
test_outputs/cadc_…     1.2M   (summary, manifest, config, logs only)
```

`git status --short` after the purge shows only this record and the tracker edit
— no tracked file was touched, and `benchmark_data/` was not touched at all.

Retained deliberately, with the reason:

- `test_outputs/perf/*.fits.fz` — the ten pinned exposures the benchmark *reads*.
  The `rm` patterns used `*.fits` and `*.weight.fits`, so these `.fits.fz` inputs
  survived; the second pass was checked against that mistake before running.
- `test_outputs/perf/` reports — `megacam_perf.json`/`.md`,
  `megacam_perf_after.json`, `exposures*.json`, `flats.json`,
  `cprofile_hdu*.txt`, `*_run.log`, `checkpoint_after.json`.
- `test_outputs/cadc_megacam_p_gt30_20261005/` — `manifest.json`,
  `run_config.yml`, `summary.json`, `summary.csv`, `summary.md`,
  `timings.json`, `sample.csv`, `logs/`.
- `test_outputs/megacam_real*/` — `megacam_perf.json`/`.md`,
  `real_output_metrics.json`, `streak_geometry.json`, `records.json`.
- `test_outputs/harness/` untouched, including the superseded `0.2.1-release/`
  evidence and the preserved `9b5231a` manifest and report.
- `benchmark_data/` untouched: real, non-redistributed inputs; not a purge
  candidate under the release documentation.

Reproducing what was deleted, if it is ever needed again:

```bash
pixi run exposure-time        # rebuilds test_outputs/perf products and reports
pixi run benchmark-megacam    # rebuilds the megacam_real* products
```

The CADC campaign is reproduced by re-fetching its cutouts with the campaign
configuration in `benchmarks/canfar/`; `run_config.yml` and `manifest.json`
under `test_outputs/cadc_megacam_p_gt30_20261005/` record what to re-fetch and
which commit produced the summary.

## 5. Consequences to keep in mind

- Deleting `test_outputs/perf/*.fits` removes the products the stage-cost tests
  in `tests/` read optionally; those tests skip when the files are absent
  (already the documented behaviour), so the suite result stays meaningful but
  loses that coverage. Recorded in
  `.cursor/harness/trajectories/2026-10-04-all-files-tests.md` (finding L02).
- `cadc_megacam_p_gt30_20261005` is a single local campaign run, not pinned
  historical evidence; if its result is ever cited in release notes, Tier A
  keeps the summary that supports the citation but deletes the pixel products
  behind it. Decide accordingly before purging.
- No automatic cleanup was added to any release path, and no current report was
  replaced by a similarly named file from another commit.
