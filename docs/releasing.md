# Releasing

The version in `pyproject.toml`, `weightmask/_version.py` and `CHANGELOG.md`
must agree before anything is tagged. Everything below exists to make sure that
happens before a tag exists, not after.

## Do the check first

```bash
pixi run release-check -- --version 0.2.1
```

This is the same script the release workflow runs. It refuses to proceed on:

| Check | What it catches |
|---|---|
| Version > latest tag | re-releasing a version PyPI already has |
| Version agreement across three files | a bumped `pyproject.toml` with a stale `_version.py` |
| `CHANGELOG.md` has a `## <version>` section, with content | a release with no notes |
| Clean working tree | tagging something other than what was reviewed |
| `python -m build` succeeds, `twine check` passes | broken packaging metadata |
| Wheel contains `_version.py`, installs outside the source tree, imports, reports the right version | the artifact that actually gets uploaded is broken |

`--skip-build` runs everything except section 5, for when you only want the
version and changelog consistency check.

Run it on the exact commit you intend to tag. A clean tree is part of the check.

## The button

**Actions → Release → Run workflow**, type the version, leave `dry_run` true.

A dry run validates and builds. Nothing is tagged, nothing is uploaded. Read the
log; if it is green, re-run with `dry_run: false`.

That run then, in order: re-validates, fails if the tag already exists, creates
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

`v0.2.0` was tagged before the CI fixes landed, so a tag can predate the
commits that made the release safe. If a fix matters to users — not just to the
build — cut a new version rather than moving a published tag. PyPI forbids
re-uploading a version, and a moved tag makes the sdist on PyPI disagree with
the git history.
