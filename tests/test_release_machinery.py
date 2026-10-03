"""The release machinery must be exercisable before it is trusted.

`benchmarks/release_check.py` is what decides whether a version is releasable, and
the workflow that tags and uploads to PyPI is a thin wrapper around it. That
split only pays off if the script itself is correct, so its individual checks are
tested here rather than being discovered on a release run.

Also pinned: the release workflow has exactly one trigger, and it is not
`release: published`. With both a manual dispatch and a release trigger, a
release created from the GitHub UI would publish to PyPI a second time and
trusted publishing would reject it -- an inconvenient failure discovered after
the fact. And the old `publish.yml` is gone for the same reason.

These tests never build, tag or publish anything.
"""

import importlib.util
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO / ".github" / "workflows"
RELEASE_WORKFLOW = WORKFLOWS / "release.yml"


def _load_release_check():
    """Import the script by path; it lives in benchmarks/, not on the package path."""
    spec = importlib.util.spec_from_file_location("release_check", REPO / "benchmarks" / "release_check.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


release_check = _load_release_check()


class TestVersionParsing(unittest.TestCase):
    def test_parse_version(self):
        self.assertEqual(release_check.parse_version("0.2.10"), (0, 2, 10))
        self.assertEqual(release_check.parse_version("v1.0.0"), (1, 0, 0))

    def test_malformed_versions_are_rejected(self):
        for bad in ("0.2", "zero.point.two", "", None, "0.2.x"):
            self.assertIsNone(release_check.parse_version(bad), f"{bad!r} must not parse")

    def test_ordering_is_numeric_not_lexicographic(self):
        """0.2.10 is newer than 0.2.9; string comparison says otherwise."""
        self.assertGreater(release_check.parse_version("0.2.10"), release_check.parse_version("0.2.9"))


class TestChangelogSectionExtraction(unittest.TestCase):
    def _with_changelog(self, text):
        """Point the module at a temporary CHANGELOG and extract from it."""
        original = release_check.CHANGELOG
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as handle:
            handle.write(text)
            temp = Path(handle.name)
        release_check.CHANGELOG = temp
        try:
            return release_check.changelog_section("0.2.1")
        finally:
            release_check.CHANGELOG = original
            temp.unlink(missing_ok=True)

    def test_returns_the_body_under_the_heading(self):
        body = self._with_changelog(
            "# Changelog\n\n## 0.2.1 - 2026-10-03\n\nSomething happened.\n\n## 0.2.0\n\nOlder.\n"
        )
        self.assertEqual(body, "Something happened.")

    def test_stops_at_the_next_heading(self):
        body = self._with_changelog("## 0.2.1\n\nNew text.\n\n## 0.2.0\n\nOld text.\n")
        self.assertNotIn("Old text", body)

    def test_missing_section_is_none_not_an_exception(self):
        self.assertIsNone(self._with_changelog("## 0.1.0\n\nOnly this.\n"))

    def test_a_prefix_match_is_not_a_match(self):
        """0.2.1 must not pick up 0.2.10's section."""
        body = self._with_changelog("## 0.2.10\n\nTen.\n")
        self.assertIsNone(body)


class TestReleaseWorkflowShape(unittest.TestCase):
    def test_it_exists_and_is_valid_yaml(self):
        import yaml

        config = yaml.safe_load(RELEASE_WORKFLOW.read_text())
        self.assertIn("jobs", config)
        self.assertIn("release", config["jobs"])

    def test_it_is_triggered_only_by_manual_dispatch(self):
        """One trigger. A release-event trigger would double-publish."""
        text = RELEASE_WORKFLOW.read_text()
        triggers = re.search(r"^on:\n(.*?)^jobs:", text, re.M | re.S)
        self.assertIsNotNone(triggers, "could not locate the on: block")
        block = triggers.group(1)
        self.assertIn("workflow_dispatch", block, "the release must be triggerable by hand")
        self.assertNotIn(
            "release:",
            block,
            "a `release: published` trigger would publish to PyPI a second time when a "
            "release is created from the GitHub UI",
        )

    def test_it_dry_runs_by_default(self):
        """Parsed, not grepped. A `dry_run:[\\s\\S]*?default: true` regex is
        satisfied by any later `default: true` anywhere in the file, so it
        cannot tell this input's default from the next block's."""
        import yaml

        config = yaml.safe_load(RELEASE_WORKFLOW.read_text())
        dry_run = config[True]["workflow_dispatch"]["inputs"]["dry_run"]
        self.assertIs(
            dry_run["default"],
            True,
            "a release workflow must not publish on its first accidental click",
        )

    def test_the_old_publish_workflow_is_gone(self):
        """Two workflows reaching PyPI is the double-publish hazard."""
        self.assertFalse(
            (WORKFLOWS / "publish.yml").exists(),
            "publish.yml would publish on `release: published`; release.yml supersedes it",
        )

    def test_it_validates_before_it_tags(self):
        text = RELEASE_WORKFLOW.read_text()
        validate = text.index("Validate the release")
        tag = text.index("name: Tag")
        self.assertLess(validate, tag, "nothing may be tagged before the release is validated")

    def test_every_tag_and_publish_step_is_guarded_by_dry_run(self):
        """One dry-run condition, used everywhere that has an effect."""
        text = RELEASE_WORKFLOW.read_text()
        for step in ("name: Tag", "gh-action-pypi-publish", "action-gh-release"):
            index = text.index(step)
            window = text[max(0, index - 400) : index]
            self.assertIn(
                "!inputs.dry_run",
                window,
                f"'{step}' is not guarded by `if: ${{{{ !inputs.dry_run }}}}`",
            )

    def test_it_requires_the_right_publish_permission(self):
        config_text = RELEASE_WORKFLOW.read_text()
        self.assertIn("id-token: write", config_text, "trusted publishing needs id-token: write")
        self.assertIn("contents: write", config_text, "creating the tag and release needs it")

    def test_it_checks_out_the_tags(self):
        """Otherwise the newest-tag guard is skipped without saying so.

        ``latest_tag()`` shells out to ``git describe --tags --abbrev=0``. The
        actions/checkout default is ``fetch-depth: 1``, which fetches no tags:
        the command fails, ``latest_tag()`` returns None, and check 1 quietly
        turns into "no tags found; treating this as the first release". The
        workflow would then tag and upload a version that is already on PyPI,
        and the only symptom would be a rejected upload.
        """
        config_text = RELEASE_WORKFLOW.read_text()
        self.assertRegex(
            config_text,
            r"fetch-depth:\s*0",
            "the checkout must fetch tags, or the 'greater than the latest tag' check is silently skipped",
        )

    def test_it_does_not_rely_on_a_bare_python(self):
        """The job installs pixi but no setup-python, so `python` is not on PATH.

        The notes step ran `python - <<PY` and failed with "command not found"
        before any notes were written -- the failure mode being that the release
        is published and then the notes step dies.
        """
        config_text = RELEASE_WORKFLOW.read_text()
        notes = config_text[config_text.index("Extract the changelog") :]
        self.assertIn("pixi run python", notes, "must call the pixi env's interpreter")
        # `^` needs re.M here: `notes` is a slice of the file, so without it
        # the pattern can only ever match at offset 0 and the assertion is dead.
        self.assertIsNone(
            re.search(r"^\s+python -", notes, re.M),
            "a bare `python` is not on PATH in a job that only installs pixi",
        )

    def test_the_notes_reuse_the_tested_extraction(self):
        """Two copies of the changelog regex is where they drifted.

        The workflow carried its own inline copy, which kept the heading's date
        fragment and emitted `- 2026-10-03` as the first line of the release
        body. It now calls the function the check called.
        """
        config_text = RELEASE_WORKFLOW.read_text()
        notes = config_text[config_text.index("Extract the changelog") :]
        self.assertIn("module.changelog_section", notes, "must call release_check.changelog_section")
        self.assertNotIn("re.search", notes, "an inline copy of the extraction can drift from the check")


class TestCheckTaskIsWiredUp(unittest.TestCase):
    def test_pixi_exposes_release_check(self):
        import tomllib

        tasks = tomllib.loads((REPO / "pixi.toml").read_text())["tasks"]
        self.assertIn("release-check", tasks, "the workflow calls `pixi run release-check`")

    def test_the_check_runs_and_reports_on_this_tree(self):
        """It must not crash. It is expected to refuse: the tree is dirty and
        0.2.0 is already tagged, and a refusal is the correct answer here."""
        result = subprocess.run(
            [sys.executable, "benchmarks/release_check.py", "--skip-build"],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        self.assertIn(result.returncode, (0, 1), result.stdout + result.stderr)
        self.assertTrue(
            re.search(r"^(releasable:|NOT RELEASABLE)", result.stdout, re.M),
            "the script must end with an explicit verdict:\n" + result.stdout[-1500:],
        )


if __name__ == "__main__":
    unittest.main()
