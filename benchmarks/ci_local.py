#!/usr/bin/env python3
"""Run the GitHub Actions workflow locally, in an environment like the runner's.

`pixi run lint` and `pixi run test` both pass in a working tree and can both fail
on the runner. The reasons are not subtle once seen:

* The runner checks out the repository fresh. Anything you rely on that is
  gitignored -- an editable install, `benchmark_data/`, a warm pixi environment
  -- is simply absent, and tests that depend on it either skip or fail.
* `setup-pixi` materialises `.pixi/` inside the workspace *before* the lint step
  runs, so `ruff check .` walks the environment unless it is excluded. It was,
  and that alone was 3721 errors on the runner.
* A literal in a workflow is a stale literal the moment the version moves.

So this script does not run the steps in place. It exports HEAD to a temporary
directory, runs each step there, and reports which tests skipped rather than
passed -- a green run that silently skipped the real-data assertions is the
failure mode this exists to surface.

Usage:
    pixi run ci-local                 # all steps
    pixi run ci-local -- --step test  # one step: lint | test | smoke | skips
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def export_checkout(destination):
    """A clean export of the committed tree, as actions/checkout would give.

    Uses `git archive HEAD`, so uncommitted work and every gitignored file stay
    behind. That is the point: it is the difference between this tree and the
    runner's.
    """
    archive = subprocess.run(
        ["git", "archive", "--format=tar", "HEAD"], cwd=REPO, check=True, stdout=subprocess.PIPE
    ).stdout
    with open(os.path.join(destination, "tree.tar"), "wb") as handle:
        handle.write(archive)
    subprocess.run(["tar", "-xf", os.path.join(destination, "tree.tar"), "-C", destination], check=True)
    os.remove(os.path.join(destination, "tree.tar"))


def carry_uncommitted(destination):
    """Copy work that is staged or modified but not yet committed.

    A new workflow fix cannot be validated from HEAD alone, since by definition
    it is not in HEAD. Staged and modified files are carried across so the run
    exercises the tree as it stands, while gitignored data and environments are
    deliberately left behind.
    """
    listed = subprocess.run(
        ["git", "status", "--porcelain", "-z", "-uall"], cwd=REPO, check=True, stdout=subprocess.PIPE, text=True
    ).stdout
    carried = []
    records = iter(listed.split("\0"))
    for line in records:
        if not line:
            continue
        path = line[3:]
        if "R" in line[:2] or "C" in line[:2]:
            old_path = next(records)  # -z lists destination first, then source.
            old_target = os.path.join(destination, old_path)
            if "R" in line[:2] and os.path.lexists(old_target):
                os.remove(old_target)
        source = os.path.join(REPO, path)
        target = os.path.join(destination, path)
        if not os.path.isfile(source) and not os.path.islink(source):
            if os.path.lexists(target):
                os.remove(target)
                carried.append(f"-{path}")
            continue
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if os.path.lexists(target):
            os.remove(target)
        shutil.copy2(source, target, follow_symlinks=False)
        carried.append(path)
    return carried


DEST = ""


def run(args, label):
    print(f"\n{'=' * 70}\n  {label}\n{'=' * 70}", flush=True)
    completed = subprocess.run(args, cwd=DEST, text=True, capture_output=True)
    sys.stdout.write(completed.stdout)
    if completed.stderr.strip():
        sys.stderr.write(completed.stderr)
    print(f"  -> exit {completed.returncode}", flush=True)
    return completed.returncode


def main(argv=None):
    global DEST
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--step", default="all", choices=["all", "lint", "test", "smoke", "skips"])
    parser.add_argument("--keep", action="store_true", help="keep the exported checkout")
    args = parser.parse_args(argv)

    root = tempfile.mkdtemp(prefix="wm_ci_local_")
    DEST = os.path.join(root, "checkout")
    os.makedirs(DEST)
    try:
        export_checkout(DEST)
        carried = carry_uncommitted(DEST)
        if carried:
            print("carried uncommitted, non-ignored changes into the export:")
            for path in carried:
                print(f"  {path}")

        has_data = os.path.isdir(os.path.join(DEST, "benchmark_data", "megacam"))
        print(f"\nexported HEAD to {DEST}")
        print(f"benchmark_data present in the export: {has_data}")
        if not has_data:
            print(
                "  NOTE: it is gitignored, so the runner will not have it either.\n"
                "        Every data-backed test will SKIP there. A green run that\n"
                "        skips the real-trail assertions is not a passing run."
            )

        steps = {
            "lint": (["pixi", "run", "lint"], "setup-pixi -> pixi run lint"),
            "test": (["pixi", "run", "test", "--", "-rs"], "pixi run test (including skip reasons)"),
            "smoke": (
                [
                    "pixi",
                    "run",
                    "python",
                    "-c",
                    "from weightmask.cli import run_pipeline; from weightmask.reconstruct_sky import main;"
                    " import weightmask; from weightmask._version import __version__;"
                    " assert weightmask.__version__ == __version__",
                ],
                "entry-point smoke",
            ),
        }
        selected = [
            s for s in ("lint", "test", "smoke") if args.step in ("all", s) or (args.step == "skips" and s == "test")
        ]

        failures = []
        for name in selected:
            command, label = steps[name]
            if run(command, label) != 0:
                failures.append(name)

        print()
        if failures:
            print(f"FAILED: {', '.join(failures)}")
            return 1
        print("all run steps passed")
        return 0
    finally:
        if args.keep:
            print(f"\nexport kept at {DEST}")
        else:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
