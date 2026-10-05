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
import os
import re
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

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
        for bad in ("0.2", "zero.point.two", "", None, "0.2.x", "0.2.1rc1", "0.2.1.2", "release-0.2.1"):
            self.assertIsNone(release_check.parse_version(bad), f"{bad!r} must not parse")

    def test_ordering_is_numeric_not_lexicographic(self):
        """0.2.10 is newer than 0.2.9; string comparison says otherwise."""
        self.assertGreater(release_check.parse_version("0.2.10"), release_check.parse_version("0.2.9"))

    def test_latest_tag_includes_releases_on_other_branches(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            for args in (
                ("init", "-b", "main"),
                ("config", "user.email", "audit@example.invalid"),
                ("config", "user.name", "Audit"),
                ("commit", "--allow-empty", "-m", "base"),
                ("tag", "v0.1.0"),
                ("checkout", "-b", "other"),
                ("commit", "--allow-empty", "-m", "newer release"),
                ("tag", "v0.3.0"),
                ("tag", "v9.0.0rc1"),
                ("checkout", "main"),
            ):
                subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
            with patch.object(release_check, "REPO", repo):
                self.assertEqual(release_check.latest_tag(), "0.3.0")


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

    def test_a_prerelease_heading_is_not_the_stable_release(self):
        self.assertIsNone(self._with_changelog("## 0.2.1-rc1\n\nPreview only.\n"))


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

        ``latest_tag()`` shells out to ``git tag --list``. The
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

    def test_validate_step_exports_dist_for_pypi_publish(self):
        """gh-action-pypi-publish uploads from dist/ by default; release_check builds
        in a tempdir unless --outdir dist is passed."""
        config_text = RELEASE_WORKFLOW.read_text()
        validate = config_text[config_text.index("Validate the release") : config_text.index("name: Tag")]
        self.assertRegex(
            validate,
            r"--outdir\s+dist\b",
            "Validate step must pass `--outdir dist` so gh-action-pypi-publish has wheels/sdists in dist/",
        )

    def test_publishing_requires_upstream_main(self):
        import yaml

        steps = yaml.safe_load(RELEASE_WORKFLOW.read_text())["jobs"]["release"]["steps"]
        guard = next((step for step in steps if step.get("name") == "Check release source"), None)
        self.assertIsNotNone(guard, "publishing must reject fork repositories and unmerged branches")
        self.assertIn("!inputs.dry_run", guard.get("if", ""), "forks and branches must still allow dry runs")
        self.assertLess(steps.index(guard), next(i for i, step in enumerate(steps) if step.get("name") == "Tag"))
        for repository, ref, code in (
            ("astroai/weightmask", "refs/heads/main", 0),
            ("sfabbro/weightmask", "refs/heads/main", 1),
            ("astroai/weightmask", "refs/heads/wip/unreviewed", 1),
            ("astroai/weightmask", "refs/tags/v0.2.1", 1),
        ):
            result = subprocess.run(
                ["bash", "-c", guard["run"]],
                env={**os.environ, "GITHUB_REPOSITORY": repository, "GITHUB_REF": ref},
                capture_output=True,
            )
            self.assertEqual(result.returncode, code, (repository, ref, result.stdout, result.stderr))

    def test_lint_and_tests_pass_before_tagging(self):
        import yaml

        steps = yaml.safe_load(RELEASE_WORKFLOW.read_text())["jobs"]["release"]["steps"]
        tag = next(i for i, step in enumerate(steps) if step.get("name") == "Tag")
        for command in ("pixi run lint", "pixi run test"):
            index = next((i for i, step in enumerate(steps) if step.get("run") == command), None)
            self.assertIsNotNone(index, f"{command} must gate a release")
            self.assertLess(index, tag)
            self.assertNotIn("continue-on-error", steps[index])

    def test_version_inputs_are_data_instead_of_script_text(self):
        import yaml

        job = yaml.safe_load(RELEASE_WORKFLOW.read_text())["jobs"]["release"]
        self.assertEqual(job.get("env", {}).get("RELEASE_VERSION"), "${{ inputs.version }}")
        for step in job["steps"]:
            self.assertNotIn("${{ inputs.version }}", step.get("run", ""))

    def test_failed_runs_do_not_claim_a_release_or_successful_build(self):
        import yaml

        steps = yaml.safe_load(RELEASE_WORKFLOW.read_text())["jobs"]["release"]["steps"]
        summary = next(step for step in steps if step.get("name") == "Summary")
        self.assertEqual(summary.get("env", {}).get("RELEASE_STATUS"), "${{ job.status }}")
        for dry_run in ("true", "false"):
            with tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp) / "summary.md"
                subprocess.run(
                    ["bash", "-c", summary["run"]],
                    check=True,
                    env={
                        **os.environ,
                        "GITHUB_STEP_SUMMARY": str(output),
                        "RELEASE_STATUS": "failure",
                        "RELEASE_VERSION": "0.2.1",
                        "RELEASE_DRY_RUN": dry_run,
                    },
                    capture_output=True,
                )
                text = output.read_text()
                self.assertIn("failure", text)
                self.assertNotIn("Released", text)
                self.assertNotIn("Validated and built", text)

    def test_validated_artifacts_are_saved_before_tagging(self):
        import yaml

        steps = yaml.safe_load(RELEASE_WORKFLOW.read_text())["jobs"]["release"]["steps"]
        upload = next(
            (i for i, step in enumerate(steps) if step.get("uses", "").startswith("actions/upload-artifact@")), None
        )
        self.assertIsNotNone(upload, "dry runs and failed publications must retain validated artifacts")
        self.assertLess(upload, next(i for i, step in enumerate(steps) if step.get("name") == "Tag"))
        self.assertNotIn("if", steps[upload], "dry runs also need reviewable artifacts")
        self.assertEqual(steps[upload]["with"]["path"], "dist/*")
        self.assertEqual(steps[upload]["with"]["if-no-files-found"], "error")


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
            re.search(r"^(releasable: \S+|NOT RELEASABLE)", result.stdout, re.M),
            "the script must end with an explicit verdict and version:\n" + result.stdout[-1500:],
        )

    def test_output_directory_does_not_delete_existing_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "keep"
            output.mkdir()
            sentinel = output / "important.txt"
            sentinel.write_text("preserve me")
            with patch.object(release_check, "run", return_value=subprocess.CompletedProcess([], 1, "", "")):
                with self.assertRaises(SystemExit) as raised:
                    release_check.main(["--version", "0.2.1", "--outdir", str(output)])
            self.assertEqual(raised.exception.code, 2)
            self.assertEqual(sentinel.read_text(), "preserve me")

    def test_failed_validation_does_not_start_building(self):
        def command(args, **kwargs):
            if "venv" in args or "pip" in args or "build" in args:
                self.fail("failed preflight must stop before creating a venv or downloading packages")
            return subprocess.CompletedProcess(args, 0, "", "")

        with patch.object(release_check, "run", side_effect=command):
            self.assertEqual(release_check.main(["--version", "invalid"]), 1)

    def test_git_failure_is_not_treated_as_a_clean_tree(self):
        def command(args, **kwargs):
            if "venv" in args:
                self.fail("git failures must stop before building")
            if args[0] == "git":
                return subprocess.CompletedProcess(args, 128, "", "not a git repository")
            return subprocess.CompletedProcess(args, 0, "0.2.1\n", "")

        with (
            patch.object(release_check, "run", side_effect=command),
            patch.object(release_check, "latest_tag", return_value=None),
            patch.object(release_check, "declared_versions", return_value=("0.2.1", "0.2.1")),
        ):
            self.assertEqual(release_check.main(["--version", "0.2.1"]), 1)

    def test_each_artifact_installs_in_its_own_isolated_environment_and_is_exported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workdir = root / "work"
            workdir.mkdir()
            output = root / "dist"
            changelog = root / "CHANGELOG.md"
            changelog.write_text("## 0.2.1\n\n" + "Fixture release notes. " * 20)
            installed = []

            def command(args, **kwargs):
                text = ""
                if "build" in args and "--outdir" in args:
                    artifacts = Path(args[-1])
                    with zipfile.ZipFile(artifacts / "weightmask-0.2.1-py3-none-any.whl", "w") as wheel:
                        wheel.writestr("weightmask/_version.py", '__version__ = "0.2.1"')
                    (artifacts / "weightmask-0.2.1.tar.gz").write_bytes(b"test sdist")
                elif "venv" in args:
                    scripts = Path(args[-1]) / "bin"
                    scripts.mkdir(parents=True, exist_ok=True)
                    for name in ("weightmask", "weightmask-reconstruct-sky"):
                        (scripts / name).touch()
                elif "install" in args and Path(args[-1]).suffix in (".whl", ".gz"):
                    installed.append((args[0], args[-1]))
                elif "-c" in args:
                    if "ProducerMetadata" in args[-1]:
                        self.assertIn("-I", args, "artifact imports must ignore PYTHONPATH and the user site")
                        text = "0.2.1 0.2.1\n"
                    else:
                        text = "0.2.1\n"
                return subprocess.CompletedProcess(args, 0, text, "")

            with (
                patch.object(release_check.tempfile, "mkdtemp", return_value=str(workdir)),
                patch.object(release_check, "run", side_effect=command),
                patch.object(release_check, "latest_tag", return_value=None),
                patch.object(release_check, "declared_versions", return_value=("0.2.1", "0.2.1")),
                patch.object(release_check, "CHANGELOG", changelog),
            ):
                self.assertEqual(release_check.main(["--version", "0.2.1", "--outdir", str(output)]), 0)
            self.assertEqual(len(installed), 2)
            self.assertEqual(len({python for python, _ in installed}), 2, "sdist cannot shadow the wheel smoke check")
            self.assertEqual({p.name for p in output.iterdir()}, {Path(artifact).name for _, artifact in installed})
            self.assertFalse(workdir.exists(), "build tools and test environments must be cleaned up")


class TestDocsMatchCurrentSurface(unittest.TestCase):
    def test_user_docs_do_not_reference_deleted_pipeline_class(self):
        for rel in ("docs/api.md", "docs/usage.md", "docs/algorithms.md", "docs/installation.md", "README.md"):
            text = (REPO / rel).read_text()
            self.assertNotIn("WeightMapGenerator", text, f"{rel} still references deleted WeightMapGenerator")
            self.assertNotIn("weightmask.pipeline", text, f"{rel} still references deleted weightmask.pipeline")

    def test_user_docs_list_all_quality_bits(self):
        from weightmask.contract import QUALITY_BITS

        for rel in ("docs/api.md", "docs/usage.md", "docs/algorithms.md"):
            text = (REPO / rel).read_text()
            for bit_name in QUALITY_BITS:
                self.assertIn(bit_name, text, f"{rel} is missing QUALITY_BITS[{bit_name!r}]")


if __name__ == "__main__":
    unittest.main()
