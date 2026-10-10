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
import io
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
WORKFLOWS = REPO / ".github" / "workflows"
RELEASE_WORKFLOW = WORKFLOWS / "release.yml"


def _evidence_fixture(**changes):
    evidence = {
        "schema": "weightmask.release-evidence.v1",
        "status": "passed",
        "version": "0.2.1",
        "commit_sha": "a" * 40,
        "config_sha": "b" * 64,
        "input_manifest_sha": "c" * 64,
        "metric_revisions": ["truth-band-recall-v2"],
        "command": "pixi run science-gate -- --manifest evidence.json",
        "timestamp": "2026-10-07T23:00:00Z",
        "data_ids": ["fixture:1013719p"],
        "real_trail_recall": 1.0,
    }
    evidence.update(changes)
    return evidence


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


class TestReleaseEvidence(unittest.TestCase):
    def test_manifest_validates_all_release_identity_fields(self):
        from benchmarks.release_evidence import validate_manifest

        self.assertEqual(
            validate_manifest(
                _evidence_fixture(),
                version="0.2.1",
                commit_sha="a" * 40,
                config_sha="b" * 64,
                input_manifest_sha="c" * 64,
            ),
            [],
        )

    def test_manifest_mutations_are_unqualified(self):
        from benchmarks.release_evidence import validate_manifest

        for mutation in (
            {"status": "skipped"},
            {"status": "failed"},
            {"commit_sha": "d" * 40},
            {"config_sha": "d" * 64},
            {"input_manifest_sha": "d" * 64},
            {"real_trail_recall": "n/a"},
            {"real_trail_recall": "n/a", "scope_exclusions": []},
            {"real_trail_recall": 1.0, "scope_exclusions": ["real_trail_recall"]},
            {"scope_exclusions": ["not-a-claim"]},
            {"metric_revisions": ["old-revision"]},
            {"metric_revisions": None},
            {"config_sha": ""},
        ):
            with self.subTest(mutation=mutation):
                errors = validate_manifest(
                    {**_evidence_fixture(), **mutation},
                    version="0.2.1",
                    commit_sha="a" * 40,
                    config_sha="b" * 64,
                    input_manifest_sha="c" * 64,
                    metric_revisions=["truth-band-recall-v2"],
                    require_qualified=True,
                )
                self.assertTrue(errors)

    def test_excluded_real_trail_recall_is_qualified(self):
        from benchmarks.release_evidence import validate_manifest

        errors = validate_manifest(
            _evidence_fixture(real_trail_recall="n/a", scope_exclusions=["real_trail_recall"]),
            version="0.2.1",
            commit_sha="a" * 40,
            config_sha="b" * 64,
            input_manifest_sha="c" * 64,
            metric_revisions=["truth-band-recall-v2"],
            require_qualified=True,
        )
        self.assertEqual(errors, [])

    def test_missing_manifest_is_rejected_only_for_qualified_release(self):
        from benchmarks.release_evidence import validate_manifest

        self.assertTrue(validate_manifest(None, version="0.2.1", require_qualified=True))
        self.assertTrue(validate_manifest(None, version="0.2.1", require_qualified=False))


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

    def test_changelog_states_the_real_trail_exclusion(self):
        body = release_check.changelog_section("0.2.1")
        self.assertIn("does not claim validated real-trail recall", body)


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

    def test_release_actions_are_pinned_to_commit_shas(self):
        config_text = RELEASE_WORKFLOW.read_text()
        for action in (
            "actions/checkout",
            "prefix-dev/setup-pixi",
            "actions/upload-artifact",
            "pypa/gh-action-pypi-publish",
            "softprops/action-gh-release",
        ):
            self.assertRegex(config_text, rf"{re.escape(action)}@[0-9a-f]{{40}}")

    def test_release_evidence_is_checked_and_attached_before_tagging(self):
        import yaml

        steps = yaml.safe_load(RELEASE_WORKFLOW.read_text())["jobs"]["release"]["steps"]
        evidence = next(i for i, step in enumerate(steps) if step.get("name") == "Validate release evidence")
        tag = next(i for i, step in enumerate(steps) if step.get("name") == "Tag")
        self.assertLess(evidence, tag)
        upload = next(i for i, step in enumerate(steps) if step.get("uses", "").startswith("actions/upload-artifact@"))
        self.assertLess(upload, tag)
        self.assertIn("evidence", steps[upload]["with"]["path"])
        release = next(step for step in steps if step.get("name") == "Create the GitHub release")
        self.assertIn("evidence", release["with"]["files"])

    def test_publish_requires_supplied_evidence_and_does_not_inline_it(self):
        import yaml

        config = yaml.safe_load(RELEASE_WORKFLOW.read_text())
        evidence_input = config[True]["workflow_dispatch"]["inputs"]["evidence_b64"]
        self.assertEqual(evidence_input["default"], "")
        steps = config["jobs"]["release"]["steps"]
        install = next(step for step in steps if step.get("name") == "Install supplied release evidence")
        emit = next(step for step in steps if step.get("name") == "Emit release evidence")
        self.assertIn("!inputs.dry_run", install.get("if", ""))
        self.assertIn("inputs.dry_run", emit.get("if", ""))
        self.assertNotIn("inputs.evidence_b64", install.get("run", ""))
        self.assertEqual(install["env"]["RELEASE_EVIDENCE_B64"], "${{ inputs.evidence_b64 }}")
        self.assertIn("base64 -d", install["run"])

    def test_release_permissions_are_narrow_and_publish_is_not_tagged_by_dry_run(self):
        import yaml

        job = yaml.safe_load(RELEASE_WORKFLOW.read_text())["jobs"]["release"]
        self.assertEqual(set(job["permissions"]), {"contents", "id-token"})
        self.assertEqual(job["permissions"]["contents"], "write")
        self.assertEqual(job["permissions"]["id-token"], "write")

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
        self.assertIn("dist/*", steps[upload]["with"]["path"])
        self.assertIn("weightmask/weightmask.yml", steps[upload]["with"]["path"])
        self.assertEqual(steps[upload]["with"]["if-no-files-found"], "error")

    def test_github_release_attaches_the_validated_config(self):
        import yaml

        steps = yaml.safe_load(RELEASE_WORKFLOW.read_text())["jobs"]["release"]["steps"]
        release = next(step for step in steps if step.get("name") == "Create the GitHub release")
        self.assertIn("dist/*", release["with"]["files"])
        self.assertIn("weightmask/weightmask.yml", release["with"]["files"])


class TestCheckTaskIsWiredUp(unittest.TestCase):
    def test_pixi_exposes_release_check(self):
        import tomllib

        tasks = tomllib.loads((REPO / "pixi.toml").read_text())["tasks"]
        self.assertIn("release-check", tasks, "the workflow calls `pixi run release-check`")

    def test_the_check_runs_and_reports_on_this_tree(self):
        """It must not crash and must end with one explicit verdict for this
        tree's version. All three are legitimate depending on state: a dirty
        tree is NOT RELEASABLE, a clean tree without a qualified evidence
        manifest is an unqualified dry run, and a clean tree with matching
        passed evidence is releasable. This ran green only while the tree was
        dirty, so the unqualified verdict -- the state a clean release tree
        reaches without evidence -- was never actually exercised."""
        import tomllib

        declared = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]["version"]
        result = subprocess.run(
            [sys.executable, "benchmarks/release_check.py", "--skip-build"],
            cwd=REPO,
            capture_output=True,
            text=True,
        )
        self.assertIn(result.returncode, (0, 1), result.stdout + result.stderr)
        verdict = re.search(r"^(releasable|unqualified dry run): (\S+)|^NOT RELEASABLE", result.stdout, re.M)
        self.assertTrue(
            verdict,
            "the script must end with an explicit verdict and version:\n" + result.stdout[-1500:],
        )
        if verdict.group(2):
            self.assertEqual(verdict.group(2), declared, "the verdict must name the version under check")

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
                        wheel.writestr("weightmask/weightmask.yml", release_check.PACKAGED_CONFIG.read_bytes())
                    with tarfile.open(artifacts / "weightmask-0.2.1.tar.gz", "w:gz") as archive:
                        data = release_check.PACKAGED_CONFIG.read_bytes()
                        member = tarfile.TarInfo("weightmask-0.2.1/weightmask/weightmask.yml")
                        member.size = len(data)
                        archive.addfile(member, io.BytesIO(data))
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
                        self.assertEqual(args[-1], release_check.artifact_smoke_script())
                        self.assertIn("default_config_path", args[-1])
                        self.assertIn("copy_default_config", args[-1])
                        self.assertIn("hashlib.sha256(config_data).hexdigest()", args[-1])
                        self.assertIn("copied.read_bytes() == config_data", args[-1])
                        self.assertEqual(
                            kwargs["env"]["WEIGHTMASK_EXPECTED_CONFIG_SHA256"],
                            release_check.config_sha256(release_check.PACKAGED_CONFIG),
                        )
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


class TestPackagedConfiguration(unittest.TestCase):
    def test_root_copy_matches_packaged_resource(self):
        from weightmask.config import default_config_path, default_config_resource

        packaged = default_config_resource().read_bytes()
        self.assertEqual(packaged, (REPO / "weightmask.yml").read_bytes())
        with default_config_path() as path:
            self.assertEqual(path.read_bytes(), packaged)

    def test_isolated_artifact_smoke_script_is_valid_and_executes(self):
        script = release_check.artifact_smoke_script()
        compile(script, "<artifact-smoke>", "exec")
        result = subprocess.run(
            [sys.executable, "-I", "-c", script],
            cwd="/",
            env={
                **os.environ,
                "WEIGHTMASK_EXPECTED_CONFIG_SHA256": release_check.config_sha256(release_check.PACKAGED_CONFIG),
            },
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), "0.2.1 0.2.1")

    def test_default_config_can_be_copied_without_source_tree(self):
        from weightmask.config import copy_default_config

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "config.yml"
            self.assertEqual(copy_default_config(destination), destination)
            self.assertEqual(destination.read_bytes(), (REPO / "weightmask.yml").read_bytes())
            with self.assertRaises(FileExistsError):
                copy_default_config(destination)
            copy_default_config(destination, overwrite=True)

    def test_copy_rejects_existing_and_dangling_symlinks_even_when_overwriting(self):
        from weightmask.config import copy_default_config

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            target = root / "real.yml"
            target.write_text("do not clobber")
            for name, link_target in (("existing.yml", target), ("dangling.yml", root / "missing.yml")):
                link = root / name
                link.symlink_to(link_target)
                for overwrite in (False, True):
                    with self.assertRaises(FileExistsError):
                        copy_default_config(link, overwrite=overwrite)
                self.assertTrue(link.is_symlink())
            self.assertEqual(target.read_text(), "do not clobber")

    def test_concurrent_no_overwrite_copy_has_one_winner(self):
        from concurrent.futures import ThreadPoolExecutor

        from weightmask.config import copy_default_config, default_config_bytes

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "config.yml"
            barrier = threading.Barrier(8)

            def copy():
                barrier.wait()
                try:
                    copy_default_config(destination)
                except FileExistsError:
                    return False
                return True

            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(executor.map(lambda _: copy(), range(8)))
            self.assertEqual(sum(results), 1)
            self.assertEqual(destination.read_bytes(), default_config_bytes())

    def test_overwrite_failure_preserves_existing_file_and_cleans_temporary(self):
        from unittest.mock import patch

        from weightmask.config import copy_default_config

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "config.yml"
            destination.write_text("keep this generation")
            with patch("weightmask.config.os.replace", side_effect=OSError("replace failed")):
                with self.assertRaises(OSError):
                    copy_default_config(destination, overwrite=True)
            self.assertEqual(destination.read_text(), "keep this generation")
            self.assertEqual(list(Path(tmp).glob(".config.yml.*")), [])

    def test_overwrite_preserves_existing_regular_file_mode(self):
        from weightmask.config import copy_default_config, default_config_bytes

        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "config.yml"
            destination.write_bytes(b"old generation")
            destination.chmod(0o640)
            copy_default_config(destination, overwrite=True)
            self.assertEqual(destination.read_bytes(), default_config_bytes())
            self.assertEqual(destination.stat().st_mode & 0o777, 0o640)

    def test_overwrite_rejects_fifo_and_device_targets(self):
        from weightmask.config import copy_default_config

        with tempfile.TemporaryDirectory() as tmp:
            fifo = Path(tmp) / "config.fifo"
            os.mkfifo(fifo)
            with self.assertRaises(FileExistsError):
                copy_default_config(fifo, overwrite=True)
            with self.assertRaises(FileExistsError):
                copy_default_config(Path("/dev/null"), overwrite=True)

    def test_config_docs_do_not_fetch_a_mutable_main_copy(self):
        for rel in ("README.md", "docs/installation.md", "docs/usage.md"):
            text = (REPO / rel).read_text()
            self.assertNotIn("blob/main/weightmask.yml", text)
            self.assertNotIn("not bundled", text.lower())

    def test_readme_doc_links_use_the_release_tag(self):
        import re

        version = re.search(r'^version = "([^"]+)"$', (REPO / "pyproject.toml").read_text(), re.MULTILINE).group(1)
        prefix = f"https://github.com/astroai/weightmask/blob/v{version}/"
        text = (REPO / "README.md").read_text()
        for target in (
            "docs/installation.md",
            "docs/usage.md",
            "docs/algorithms.md",
            "docs/api.md",
            "docs/releasing.md",
            "CHANGELOG.md",
            "LICENSE",
        ):
            self.assertIn(prefix + target, text)
        metadata = (REPO / "pyproject.toml").read_text()
        self.assertIn(f'Changelog = "{prefix}CHANGELOG.md"', metadata)


class TestArtifactRetentionDocumentation(unittest.TestCase):
    DOCUMENT = REPO / "docs" / "releasing.md"
    INSPECTION_PATTERNS = (
        "__pycache__/",
        ".pixi/",
        ".venv*/",
        ".uv-cache/",
        ".env",
        "venv/",
        "env/",
        ".pytest_cache/",
        ".ruff_cache/",
        ".mypy_cache/",
        ".cache",
        ".coverage",
        ".coverage.*",
        "coverage.xml",
        "htmlcov/",
    )

    @classmethod
    def _inspection_block(cls):
        text = cls.DOCUMENT.read_text()
        start = text.index("## Inspect ignored artifacts before a dry run")
        end = text.index("## Retention policy for benchmark and release evidence")
        return text[start:end]

    @classmethod
    def _launcher_inspection_block(cls):
        text = cls.DOCUMENT.read_text()
        start = text.index("Inspect a suspected launcher without executing it:")
        end = text.index("From the current checkout, rebuild the environment")
        return text[start:end]

    def test_release_docs_cover_safe_ignored_artifact_inventory(self):
        text = self.DOCUMENT.read_text()
        for term in (
            "git status --short --ignored",
            "git check-ignore -v",
            "test_outputs/",
            "benchmark_data/",
            "duplicate",
            "`dist/`",
            ".pytest_cache",
            ".ruff_cache",
            ".mypy_cache",
            "stale local environments",
            "stale executable shebangs",
        ):
            self.assertIn(term, text, f"release docs are missing the WM-29 inspection guidance: {term}")

    def test_release_docs_define_provenance_and_large_data_retention(self):
        text = self.DOCUMENT.read_text()
        for term in (
            "JSON or Markdown report",
            "commit SHA",
            "configuration SHA",
            "FITS/cache products",
            "detector caches",
            "CANFAR session",
            "There is no automatic cleanup",
            "pixi reinstall --locked",
        ):
            self.assertIn(term, text, f"release docs are missing the WM-29 retention rule: {term}")

    def test_every_ignored_environment_and_cache_pattern_is_in_inventory(self):
        gitignore = (REPO / ".gitignore").read_text()
        block = self._inspection_block()
        for pattern in self.INSPECTION_PATTERNS:
            self.assertIn(pattern, gitignore, f".gitignore is missing the expected pattern: {pattern}")
            self.assertIn(pattern, block, f"WM-29 inventory omits .gitignore pattern: {pattern}")

    def test_inspection_blocks_forbid_environment_execution(self):
        for block in (self._inspection_block(), self._launcher_inspection_block()):
            self.assertNotRegex(
                block,
                r"(?m)^\s*(?:pixi|python(?:\d+(?:\.\d+)*)?|pip|uv|conda|pytest|ruff|weightmask|release-check)\b",
            )
            self.assertNotRegex(block, r"(?m)^\s*(?:source|activate|exec|rm|mv|cp|unlink|clean)\b")

    def test_launcher_inspection_reads_only_metadata_and_first_line(self):
        block = self._launcher_inspection_block()
        for command in ("file", "readlink", "read -r", "shebang"):
            self.assertIn(command, block, f"launcher inspection must use read-only {command}: {block}")

    def test_inventory_uses_portable_recursive_size_for_a_temp_tree(self):
        block = self._inspection_block()
        self.assertIn('du -sk "$path"', block)
        self.assertNotIn("stat -f", block)
        self.assertNotRegex(block, r"du\s+[^\n]*\s--(?:\s|\")")
        loop = re.search(r"for path in test_outputs benchmark_data.*?^done$", block, re.M | re.S)
        self.assertIsNotNone(loop, "the documented inventory loop must remain executable")

        with tempfile.TemporaryDirectory(prefix="wm29 tree ") as tmp:
            root = Path(tmp)
            payload = root / "test_outputs" / "nested" / "payload.bin"
            payload.parent.mkdir(parents=True)
            payload.write_bytes(b"x" * 8192)
            result = subprocess.run(
                ["bash", "-eu", "-c", loop.group(0)],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            )
            reported = next(line for line in result.stdout.splitlines() if line.endswith("test_outputs"))
            directory_kib = int(reported.split()[0])
            file_kib = int(
                subprocess.run(["du", "-sk", str(payload)], capture_output=True, text=True, check=True).stdout.split()[
                    0
                ]
            )
            self.assertGreaterEqual(directory_kib, file_kib)

    def test_retention_guidance_has_no_destructive_cleanup_command(self):
        text = self.DOCUMENT.read_text()
        self.assertNotRegex(text, r"(?m)^\s*(?:rm|git\s+clean|pixi\s+clean)\b")
        self.assertNotRegex(text, r"(?m)^\s*find\b.*(?:-delete|-exec\s+rm)\b")


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
