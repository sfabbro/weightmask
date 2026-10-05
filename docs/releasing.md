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
| Wheel contains `_version.py`; wheel and sdist each install in separate environments, import outside the source tree, and report the right version | a broken artifact hidden by the source tree or the other artifact |

`--skip-build` skips artifact validation and the installed-package import. It
still checks declared versions, tags, changelog, and the clean working tree.

Run it on the exact commit you intend to tag. A clean tree is part of the check.
`--outdir dist` exports the validated wheel and sdist. The destination must be
absent or empty; existing files are never removed. The tag guard is local:
fetch tags and check the project's published PyPI versions before dispatching.

## The button

**Actions → Release → Run workflow**, type the version, leave `dry_run` true.

A dry run runs lint/tests, validates, builds, and saves a downloadable workflow
artifact. It creates no tag or PyPI release. Read the log and inspect the
artifacts; if it is green, re-run with `dry_run: false` on the same commit.
Publishing is allowed only from `astroai/weightmask` on `main`; fork and branch
dry runs are allowed. Merge the workflow into upstream `main` first for the
Release button to exist there.

That run then, in order: runs lint/tests, re-validates and saves the artifacts,
fails if the tag already exists, creates
and pushes the annotated tag, publishes to PyPI, and opens a GitHub release
whose notes are the `CHANGELOG.md` section for that version.

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
