"""The CI workflow must not drift from the tree it checks.

Three defects on `.github/workflows/ci.yml` were live on `main` at the 0.2.0 tag
and none of them could be caught by running the workflow's steps in the working
tree, because each one is invisible from inside that tree:

* the entry-point smoke step asserted ``__version__ == '0.1.0'`` as a literal, so
  it failed on every push after the 0.2.0 bump;
* ``setup-pixi`` materialises ``.pixi/`` inside the workspace before the lint step
  runs, and ruff walked it -- 3721 errors, none of them in this repository;
* ``benchmark_data/`` is gitignored, so every real-data assertion skips on the
  runner and the run is green while verifying nothing about trails.

The first is checked here by parsing the workflow. The second is checked by
asserting ruff excludes ``.pixi`` and that ``.pixi`` is gitignored -- the local
config that made ``pixi run lint`` pass while the runner's would not. The third
cannot be asserted away, so it is reported rather than hidden: the point is that
it is a decision someone has to see, not a skip that passes quietly.

``benchmarks/ci_local.py`` re-derives all of this by running the steps in an
export of HEAD.
"""

import re
import subprocess
import unittest
from pathlib import Path

import tomllib

REPO = Path(__file__).resolve().parents[1]
WORKFLOW = REPO / ".github" / "workflows" / "ci.yml"
PYPROJECT = REPO / "pyproject.toml"
GITIGNORE = REPO / ".gitignore"


def _workflow() -> str:
    return WORKFLOW.read_text()


class TestWorkflowHasNoStaleVersionLiteral(unittest.TestCase):
    def test_no_bare_version_assertion(self):
        """The 0.1.0 literal that broke every push after the bump."""
        offenders = re.findall(r"assert\s+\w+(?:\.__version__)?\s*==\s*['\"](\d+\.\d+\.\d+)['\"]", _workflow())
        self.assertEqual(
            offenders,
            [],
            "the workflow asserts a literal version; read it from weightmask._version instead",
        )

    def test_it_checks_the_same_source_the_package_does(self):
        self.assertIn(
            "_version",
            _workflow(),
            "the smoke step should compare against weightmask._version, the single declaration",
        )


class TestLintCannotWalkThePixiEnvironment(unittest.TestCase):
    """3721 errors on the runner, 0 in the working tree, same ruff and same version."""

    def test_pixi_is_excluded_from_ruff(self):
        config = tomllib.loads(PYPROJECT.read_text())
        excluded = config["tool"]["ruff"].get("extend-exclude", [])
        self.assertIn(
            ".pixi",
            excluded,
            "setup-pixi creates .pixi/ in the workspace before lint runs; ruff must skip it",
        )

    def test_pixi_is_gitignored(self):
        lines = [line.strip() for line in GITIGNORE.read_text().splitlines()]
        self.assertIn(
            ".pixi/",
            lines,
            ".pixi/ is a local environment and must not be tracked",
        )

    def test_the_lint_step_passes_here(self):
        """The CI command itself, run as CI runs it.

        Deliberately not a hand-rolled file walk. An earlier version listed
        ``*.py`` itself and filtered by ``extend-exclude``, which silently drops
        ruff's *default* excludes -- ``site-packages`` among them -- and so
        reported 12600 errors from a stale ``.venv-wm`` in the tree that the real
        command never looks at. The clean-checkout property is what
        ``benchmarks/ci_local.py`` exists to test, by exporting HEAD and running
        the real command there.
        """
        result = subprocess.run(
            ["pixi", "run", "lint"], cwd=REPO, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stdout[-2000:])


class TestRealDataAssertionsAreVisible(unittest.TestCase):
    """These skip rather than fail, so nothing else would ever mention them."""

    def test_benchmark_data_is_gitignored_and_therefore_absent_in_ci(self):
        lines = [line.strip() for line in GITIGNORE.read_text().splitlines()]
        self.assertIn(
            "benchmark_data/",
            lines,
            "benchmark_data is gitignored, so the runner has no MegaCam data and the "
            "real-trail assertions in test_brightness_veto.py skip there",
        )

    def test_the_skipping_tests_name_what_they_need(self):
        """A skip must be traceable to a file, not a bare 'if not available'.

        Two kinds of input are missing on the runner: ``benchmark_data`` for the
        MegaCam amps, and ``test_outputs`` for the stage-sweep JSONs. Each test
        must name the one it depends on, so ``pytest -rs`` says which artefact is
        absent instead of just 'not available'.
        """
        expected = {
            "tests/test_brightness_veto.py": "benchmark_data",
            "tests/test_streak_dead_stages.py": "test_outputs",
        }
        for test, needed in expected.items():
            text = (REPO / test).read_text()
            self.assertIn("SkipTest", text, f"{test} should skip explicitly so pytest -rs reports it")
            self.assertIn(
                needed,
                text,
                f"{test} should name {needed} in its skip message so the reason is traceable",
            )


if __name__ == "__main__":
    unittest.main()
