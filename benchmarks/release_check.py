#!/usr/bin/env python3
"""Validate a release before anything is tagged or uploaded.

Everything here runs locally against a version string, so the workflow that tags
and publishes to PyPI is a thin wrapper around code that has already been
exercised. The failure this avoids: a workflow that builds, tags and uploads
without ever checking that the artifacts work, or that the version agrees across
four files.

Checks, in order:

1. the version is well formed and greater than the latest tag;
2. ``pyproject.toml``, ``weightmask/_version.py`` and the installed package all
   declare it;
3. ``CHANGELOG.md`` has a ``## <version>`` section with real content under it;
4. a clean tree, so a tagged commit cannot differ from what was reviewed;
5. an sdist and a wheel build, pass ``twine check``, and both install and
   imports *outside* the source tree with the right version and entry points.

Exit 0 means releasable. Exit 1 prints what failed and stops.

Usage:
    pixi run release-check -- --version 0.2.1
    pixi run release-check -- --version 0.2.1 --skip-build   # checks 1-4 only
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import tomllib

REPO = Path(__file__).resolve().parents[1]
PYPROJECT = REPO / "pyproject.toml"
VERSION_MODULE = REPO / "weightmask" / "_version.py"
CHANGELOG = REPO / "CHANGELOG.md"

FAILURES: list[str] = []
CURRENT = [""]


def check(condition, message):
    if condition:
        print(f"  ok    {message}")
    else:
        print(f"  FAIL  {message}")
        FAILURES.append(message)
    return bool(condition)


def section(title):
    print(f"\n{title}\n{'-' * len(title)}")


def declared_versions():
    pyproject = tomllib.loads(PYPROJECT.read_text())["project"]["version"]
    module = re.search(r'__version__\s*=\s*"([^"]+)"', VERSION_MODULE.read_text()).group(1)
    return pyproject, module


def latest_tag():
    result = subprocess.run(["git", "tag", "--list", "v*"], cwd=REPO, capture_output=True, text=True)
    if result.returncode != 0:
        return None
    versions = [tag[1:] for tag in result.stdout.splitlines() if re.fullmatch(r"v\d+\.\d+\.\d+", tag)]
    return max(versions, key=parse_version) if versions else None


def parse_version(text):
    match = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", text or "")
    return tuple(map(int, match.groups())) if match else None


def changelog_section(version):
    """The text under ``## <version>``, up to the next ``## `` heading.

    The remainder of the heading line is dropped, so ``## 0.2.1 - 2026-10-03``
    yields the body rather than a leading ``- 2026-10-03`` fragment. That
    fragment was being carried straight into the GitHub release notes by the
    workflow's own extraction.
    """
    text = CHANGELOG.read_text()
    match = re.search(rf"^##[ \t]+{re.escape(version)}(?:[ \t]+[^\n]*)?\n(.*?)(?=^##[ \t]|\Z)", text, re.M | re.S)
    return match.group(1).strip() if match else None


def run(command, cwd=None, env=None):
    return subprocess.run(command, cwd=cwd or REPO, capture_output=True, text=True, env=env)


def finish():
    """Print the verdict and return it, for early exit from the build section."""
    print()
    if FAILURES:
        print(f"NOT RELEASABLE -- {len(FAILURES)} problem(s):")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print(f"releasable: {CURRENT[0]}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--version", help="version to release, e.g. 0.2.1")
    parser.add_argument("--skip-build", action="store_true", help="run checks 1-4 only")
    parser.add_argument(
        "--outdir",
        help="copy the validated wheel and sdist into this directory (e.g. dist) after all checks pass",
    )
    args = parser.parse_args(argv)

    outdir = None
    if args.outdir:
        outdir = Path(args.outdir).absolute()
        if outdir.is_symlink() or (outdir.exists() and (not outdir.is_dir() or any(outdir.iterdir()))):
            parser.error("--outdir must be absent or an empty directory; existing files will not be removed")
        if args.skip_build:
            parser.error("--outdir requires building artifacts; omit --skip-build")

    FAILURES.clear()
    version = args.version
    if not version:
        pyproject, _module = declared_versions()
        version = pyproject
        print(f"no --version given; using the version in pyproject.toml ({version})")
    CURRENT[0] = version

    section(f"1. version {version}")
    parsed = parse_version(version)
    check(bool(parsed) and version == ".".join(str(p) for p in parsed), f"{version} is well formed")
    latest = latest_tag()
    if latest:
        check(
            parsed is not None and parse_version(latest) is not None and parsed > parse_version(latest),
            f"{version} is greater than the latest tag v{latest}",
        )
    else:
        print(f"  note  no tags found; treating v{version} as the first release")

    section("2. version agreement")
    pyproject, module = declared_versions()
    check(pyproject == version, f"pyproject.toml declares {pyproject}")
    check(module == version, f"weightmask/_version.py declares {module}")
    if not args.skip_build:
        installed = run([sys.executable, "-c", "import weightmask; print(weightmask.__version__)"])
        check(
            installed.stdout.strip() == version,
            f"the installed package reports {installed.stdout.strip() or '<none>'}",
        )

    section("3. changelog")
    body = changelog_section(version)
    check(body is not None, f"CHANGELOG.md has a '## {version}' section")
    if body is not None:
        check(len(body) > 200, f"that section has real content ({len(body)} chars)")
        check(
            not re.search(r"^##\s+Unreleased", CHANGELOG.read_text()[:400], re.M),
            "no 'Unreleased' heading is left above the release",
        )

    section("4. tree")
    status = run(["git", "status", "--porcelain"])
    check(status.returncode == 0, "git status succeeds")
    dirty = [line for line in status.stdout.splitlines() if line.strip()]
    check(
        not dirty,
        "the working tree is clean" if not dirty else f"uncommitted changes: {[line[3:] for line in dirty]}",
    )
    head = run(["git", "rev-parse", "--short", "HEAD"])
    check(head.returncode == 0, "HEAD is a commit")
    print(f"  note  would tag HEAD at {head.stdout.strip()}")

    if FAILURES:
        return finish()

    if args.skip_build:
        section("build skipped")
    else:
        section("5. artifacts")
        workdir = Path(tempfile.mkdtemp(prefix="wm_release_check_"))
        try:
            # Build tools are isolated from runtime dependencies and cleaned
            # together with the temporary artifacts.
            builder = workdir / "build-env"
            created_builder = run([sys.executable, "-m", "venv", str(builder)])
            if not check(created_builder.returncode == 0, "a build venv can be created"):
                return finish()
            builder_python = str(builder / "bin" / "python")
            got_build = run([builder_python, "-m", "pip", "install", "--quiet", "build", "twine"])
            if not check(got_build.returncode == 0, "build can be installed into that venv"):
                print(got_build.stderr[-1200:])
                return finish()
            build = run([builder_python, "-m", "build", "--outdir", str(workdir)])
            if not check(build.returncode == 0, "python -m build succeeds"):
                print(build.stdout[-1500:], build.stderr[-1500:])
            else:
                wheel = next(workdir.glob("*.whl"), None)
                sdist = next(workdir.glob("*.tar.gz"), None)
                check(wheel is not None, "a wheel was produced")
                check(sdist is not None, "an sdist was produced")
                if wheel:
                    check(
                        wheel.name == f"weightmask-{version}-py3-none-any.whl",
                        f"the wheel is named {wheel.name}",
                    )
                    names = zipfile.ZipFile(wheel).namelist()
                    check(
                        "weightmask/_version.py" in names,
                        "_version.py is inside the wheel (it was added after 0.2.0 was tagged)",
                    )
                    check(
                        not any(n.startswith("benchmarks/") or n.startswith("tests/") for n in names),
                        "the wheel contains no benchmarks or tests",
                    )
                artifacts = sorted(a for a in workdir.iterdir() if a.suffix in (".whl", ".gz"))
                check(len(artifacts) == 2, f"exactly two artifacts to check: {[a.name for a in artifacts]}")
                twine = run([builder_python, "-m", "twine", "check", *[str(a) for a in artifacts]])
                check(twine.returncode == 0, "twine check passes on both artifacts")
                if twine.returncode != 0:
                    print(twine.stdout[-1200:], twine.stderr[-800:])

                for artifact in artifacts:
                    venv = workdir / f"venv-{artifact.name}"
                    created = run([builder_python, "-m", "venv", str(venv)])
                    if not check(created.returncode == 0, f"a clean venv can be created for {artifact.name}"):
                        continue
                    pip = [str(venv / "bin" / "python"), "-m", "pip", "install", "--quiet"]
                    installed_artifact = run(pip + [str(artifact)])
                    if not check(installed_artifact.returncode == 0, f"installing {artifact.name}"):
                        print(installed_artifact.stderr[-1200:])
                        continue
                    outside = run(
                        [
                            str(venv / "bin" / "python"),
                            "-I",
                            "-c",
                            "import weightmask;"
                            "from weightmask.contract import ProducerMetadata;"
                            "from weightmask.cli import run_pipeline;"
                            "from weightmask.reconstruct_sky import main;"
                            "assert weightmask.__version__ == weightmask._version.__version__;"
                            "assert ProducerMetadata().version == weightmask.__version__;"
                            "print(weightmask.__version__, ProducerMetadata().version)",
                        ],
                        cwd="/",
                    )
                    check(
                        outside.returncode == 0 and outside.stdout.strip() == f"{version} {version}",
                        f"{artifact.name} imports outside the source tree and reports {version}"
                        if outside.returncode == 0
                        else f"{artifact.name} failed to import outside the source tree",
                    )
                    scripts = [p.name for p in (venv / "bin").glob("weightmask*")]
                    check(
                        set(scripts) >= {"weightmask", "weightmask-reconstruct-sky"},
                        f"console scripts installed: {scripts}",
                    )
                if outdir and not FAILURES:
                    if outdir.is_symlink() or (outdir.exists() and (not outdir.is_dir() or any(outdir.iterdir()))):
                        check(False, "output directory changed during the build; no existing files overwritten")
                        return finish()
                    try:
                        outdir.mkdir(parents=True, exist_ok=True)
                        for artifact in artifacts:
                            with artifact.open("rb") as source, (outdir / artifact.name).open("xb") as destination:
                                shutil.copyfileobj(source, destination)
                    except OSError as error:
                        check(False, f"could not stage validated artifacts: {error}")
                        return finish()
                    staged = sorted(p.name for p in outdir.iterdir())
                    check(
                        staged == [a.name for a in artifacts],
                        f"staged validated artifacts to {args.outdir}: {staged}",
                    )
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    return finish()


if __name__ == "__main__":
    sys.exit(main())
