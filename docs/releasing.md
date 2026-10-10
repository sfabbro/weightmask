# Releasing

The version in `pyproject.toml`, `weightmask/_version.py` and `CHANGELOG.md`
must agree before anything is tagged. Everything below exists to make sure that
happens before a tag exists, not after.

## Do the check first

```bash
pixi run lint
pixi run test
pixi run release-check -- --version 0.2.1
```

The science gate emits `test_outputs/harness/release-evidence.json`. The small
JSON manifest records its status, version, commit SHA, configuration SHA,
input/fixture SHA, metric revisions, command, UTC timestamp, data IDs, and
real-trail recall. `release-check` validates those identities against the
candidate before it builds:

```bash
pixi run science-gate -- --version 0.2.1 --manifest test_outputs/harness/release-evidence.json
pixi run release-check -- --version 0.2.1 \
  --evidence test_outputs/harness/release-evidence.json
```

The current WM-22 fixture contains no independently confirmed positive trail,
so its recall is `n/a`. Version 0.2.1 does not claim validated real-trail
recall. A passed manifest records that limit as
`scope_exclusions: ["real_trail_recall"]`. A dry run may still build, but a
skipped, failed, stale, or mismatched report is explicitly **UNQUALIFIED**.
A non-dry-run validation adds `--require-qualified` and refuses to continue
for missing, skipped, failed, stale, or mismatched evidence, or for
`real_trail_recall: n/a` without that exclusion. The changelog section must
state the exclusion. This refusal happens before the tag step.

For 0.2.1, also qualify the complete MegaCam fixture with `pixi run science-gate`
and require a passed report before publication. Missing data produces a skip,
which is not qualification. This data-backed gate remains an explicit local
check; ordinary CI does not download or run the campaign.

The release workflow runs the same `release-check` script. It refuses to proceed on:

| Check | What it catches |
|---|---|
| Version > newest numeric release tag across all branches | reusing or lowering a tagged version |
| Version agreement across `pyproject.toml`, `_version.py`, and installed package | a bumped `pyproject.toml` with a stale `_version.py` |
| `CHANGELOG.md` has a `## <version>` section, with content | a release with no notes |
| Clean working tree | tagging something other than what was reviewed |
| `python -m build` succeeds, `twine check` passes | broken packaging metadata |
| Root and packaged YAML copies match; wheel and sdist each contain the canonical resource | a release whose configuration differs by artifact |
| Wheel contains `_version.py`; wheel and sdist each install in separate environments, import outside the source tree, and report the right version | a broken artifact hidden by the source tree or the other artifact |
| Evidence manifest is current, passed, and identity-matched. Real-trail recall is numeric, or `n/a` with `scope_exclusions: ["real_trail_recall"]` and the same statement in the changelog | publishing a dry run, skipped report, stale report, or an unexplained WM-22 `n/a` recall |

`--skip-build` skips artifact validation and the installed-package import. It
still checks declared versions, tags, changelog, and the clean working tree.

Run it on the exact commit you intend to tag. A clean tree is part of the check.
`--outdir dist` exports the validated wheel and sdist. The destination must be
absent or empty; existing files are never removed. The tag guard is local:
fetch tags and check the project's published PyPI versions before dispatching.

## Inspect ignored artifacts before a dry run

Ignored paths are not release evidence by themselves. Before inspecting a dry
run or deciding what to retain, record the candidate commit and inventory local
state. These commands only report paths, sizes, and ignore rules:

```bash
git rev-parse HEAD
git status --short --ignored
git check-ignore -v --no-index -- \
  test_outputs benchmark_data dist \
  .pixi/ '.venv*/' .uv-cache/ .env venv/ env/ \
  __pycache__/ .pytest_cache/ .ruff_cache/ .mypy_cache/ .cache \
  .coverage '.coverage.*' coverage.xml htmlcov/ || true
for path in test_outputs benchmark_data dist .pixi .uv-cache .env venv env \
  __pycache__ .pytest_cache .ruff_cache .mypy_cache .cache .coverage \
  coverage.xml htmlcov; do
  if [ -e "$path" ]; then du -sk "$path"; fi
done
for path in .venv*/; do
  if [ -d "$path" ]; then du -sk "$path"; fi
done
git ls-files --others --ignored --exclude-standard
```

`du -sk` is available on both declared platforms and reports allocated KiB
recursively for either a file or a directory. The shell variable is quoted, so
the same read-only loop remains safe when an inspected path contains spaces.

For `test_outputs/` and `benchmark_data/`, inspect report names, timestamps,
input IDs, and hashes before looking at large FITS products. For duplicate `dist/`
artifacts, compare filenames, sizes, and hashes, then inspect the wheel
and sdist metadata. Only the wheel, sdist, and configuration produced by the
same successful validation invocation are release artifacts; an older or
duplicate file is not evidence for the current commit. A non-empty `dist/`
causes the release validation to refuse to proceed, rather than mixing artifact
generations.

## Retention policy for benchmark and release evidence

Keep the small provenance needed to identify and reproduce a result:

- retain the current JSON or Markdown report, manifest, and any checksum file;
- retain the commit SHA, configuration SHA, input or data IDs, command,
  timestamp, package version, and metric revision in that provenance; and
- retain historical reports when another report, changelog entry, or release
  decision refers to them. Mark them as historical rather than presenting them
  as current evidence.

Large FITS/cache products, profiling output, detector caches, and
duplicate build artifacts are purge candidates only after their small
provenance has been checked and the result is either reproducible or explicitly
labelled as a retained one-off. `benchmark_data/` contains local,
non-redistributed inputs; do not attach those inputs to a release. Keep them
only for an active or explicitly archived analysis. A CANFAR session's scratch
output is not durable evidence: copy its small manifest and report to
persistent project storage before the session ends, and retain large products
there only when the analysis requires them.

There is no automatic cleanup in the release process. An operator must make a
separate, explicit retention decision after this inspection. In particular,
never replace a current report with a similarly named file from another commit
and never treat a cache as proof that a benchmark ran.

## Rebuild copied or moved environments

`.pixi/` and `.venv*/` are stale local environments when they come from another
checkout; neither is release evidence. Their executable
scripts contain absolute shebangs, so copying or moving an environment can make
`ruff`, `pytest`, or a benchmark invoke an interpreter from the old checkout.
Inspect a suspected launcher without executing it:

```bash
for executable in .pixi/envs/default/bin/ruff .pixi/envs/default/bin/pytest \
  .venv*/bin/ruff .venv*/bin/pytest venv/bin/pytest env/bin/pytest; do
  if [ -f "$executable" ]; then
    printf '%s: ' "$executable"
    file "$executable"
    readlink "$executable" || true
    IFS= read -r shebang < "$executable"
    printf 'shebang: %s\n' "$shebang"
  fi
done
```

From the current checkout, rebuild the environment from the lock file before
running any release or benchmark command:

```bash
pixi reinstall --locked
pixi run lint
pixi run test
```

If the environment was copied or moved, stale executable shebangs are not valid
evidence; rerun the command after the rebuild and retain the new report's
provenance.

## The button

**Actions → Release → Run workflow**, type the version, leave `dry_run` true.

A dry run runs lint/tests, emits whatever science gate the runner can produce,
validates, builds, and saves a downloadable workflow artifact containing the
wheel, sdist, canonical `weightmask.yml`, and evidence manifest. GitHub-hosted
runners do not have the MegaCam exposures, so that emitted report is a skip and
the dry run stays **UNQUALIFIED**. It creates no tag or PyPI release.

Publishing needs evidence from a local science gate on the same commit:

```bash
pixi run science-gate -- --version 0.2.1 --manifest test_outputs/harness/release-evidence.json
pixi run release-check -- --version 0.2.1 \
  --evidence test_outputs/harness/release-evidence.json --require-qualified --skip-build
gh workflow run release.yml -R astroai/weightmask \
  -f version=0.2.1 -f dry_run=false \
  -f evidence_b64="$(base64 < test_outputs/harness/release-evidence.json | tr -d '\n')"
```

The workflow decodes that manifest and refuses to tag if it does not match the
checked-out commit. Publishing is allowed only from `astroai/weightmask` on
`main`; fork and branch dry runs are allowed. Merge the workflow into upstream
`main` first for the Release button to exist there.

That run then, in order: runs lint/tests, re-validates and saves the artifacts,
fails if the tag already exists, creates
and pushes the annotated tag, publishes to PyPI, and opens a GitHub release
whose notes are the `CHANGELOG.md` section for that version.
The GitHub release includes the same wheel and sdist plus the validated canonical
configuration resource and evidence manifest. The wheel and sdist uploaded to
PyPI are the exact files produced by the successful validation invocation.

## One-time setup on the repository

Publishing uses [trusted publishing](https://docs.pypi.org/trusted-publishers/),
so there is no API token to configure. On PyPI, add a publisher for the repo
`astroai/weightmask`, workflow `release.yml`, environment `pypi`. The workflow
declares that `environment:`, and PyPI enforces it — a run without it is rejected.

## Why there is no `release: published` trigger

A workflow triggered by both `workflow_dispatch` and `release: published`
publishes twice: once when you run it, once when the release it just created is
observed. The second upload is rejected, after the tag exists. `tests/
test_release_machinery.py` asserts the single trigger.

## After the tag

If a fix matters to users, cut a new version rather than moving a published
tag. PyPI forbids
re-uploading a version, and a moved tag makes the sdist on PyPI disagree with
the git history.

Tagging precedes PyPI publication. If publication fails after the tag is pushed,
the workflow cannot simply be rerun for that version: the existing-tag guard
will refuse. Preserve the saved validated artifacts, fix the publisher problem,
and complete the missing upload/release from those exact artifacts. Check PyPI
first to distinguish a failed upload from a successful upload followed by a
GitHub-release failure. Never move the tag or rebuild different artifacts under
the same version. Partial-release recovery is manual.

## Historical CANFAR campaign

`benchmarks/canfar_experiments/manifest.json` is a pinned historical manifest,
not a current campaign configuration. It remains in the tree for reproducible
analysis. Setup stages the versioned, SHA-256-pinned current runner
(`runner.sha256`) and validator (`runner.validator_sha256`) in the bootstrap
checkout first; each job then clones the pinned science checkout in a
separate working directory. The CANFAR submission wrappers validate each selected group against
the canonical configuration schema, reject removed overrides, unresolved E0
inputs, multi-flat E7 submissions without an explicit mapping, and the
unfinished E8 winner group. Bootstrap setup also requires the versioned
provenance-bearing `run_one.sh`. Each run logs the manifest content hash, the
manifest's pinned code SHA, the requested checkout ref, and the checked-out
commit SHA in its result metrics, failing on mismatches.
