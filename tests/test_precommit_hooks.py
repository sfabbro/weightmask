"""The pre-push hooks must actually deliver CI parity, not just claim it.

The original failure that cost four red CI pushes was invisible to the local
``pre-push`` hook: it ran ``pixi run lint && pixi run test`` ``in place``, where
gitignored data and a warm environment mask exactly the divergences that fail on
the runner. ``benchmarks/ci_local.py`` exists to close that gap by exporting
HEAD and running the steps there. These tests pin the hook contract so the
clean-export half cannot silently disappear again, and that both hooks route
through ``benchmarks/run_with_pixi.sh`` so a push from a shell without pixi on
its PATH yields an install hint instead of a bare exit 127.
"""

import unittest
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / ".pre-commit-config.yaml"
WRAPPER = REPO / "benchmarks" / "run_with_pixi.sh"


class TestPrePushHooks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(CONFIG.read_text())
        cls.hooks = {h["id"]: h for repo in cls.config["repos"] for h in repo.get("hooks", [])}

    def test_both_halves_of_ci_parity_run_before_a_push(self):
        _inplace, _inplace_tests = (self.hooks.get(k) for k in ("ci-parity-inplace", "ci-parity-inplace-tests"))
        _clean = self.hooks.get("ci-parity-clean-export")
        self.assertIsNotNone(_inplace, "in-place lint must stay a pre-push gate")
        self.assertIsNotNone(_inplace_tests, "in-place tests must stay a pre-push gate")
        self.assertIsNotNone(
            _clean,
            "the clean-export run (benchmarks/ci_local.py) must gate the push; "
            "without it the hook repeats the failure mode that kept CI red",
        )
        for hook in (_inplace, _inplace_tests, _clean):
            self.assertEqual(hook.get("stages"), ["pre-push"], f"{hook['id']} must run at pre-push")

    def test_no_hook_entry_chains_with_shell_operators(self):
        """pre-commit shlex-splits entries; `&&` becomes literal arguments.

        A chained entry once made the wrapper receive `lint`, `&&`, and the rest
        as one argument vector, so pixi's lint task ran `ruff check . && bash
        benchmarks/run_with_pixi.sh test -q` -- ruff parsed the shell command
        as Python and failed with E902 syntax errors instead of running either
        half of the gate cleanly.
        """
        for repo in self.config["repos"]:
            for hook in repo.get("hooks", []):
                entry = str(hook.get("entry", ""))
                self.assertNotIn(
                    "&&",
                    entry,
                    f"{hook['id']} chains with &&; pre-commit shlex-splits entries, making those literal args",
                )
                self.assertNotRegex(
                    entry,
                    r"\|[;&]",
                    f"{hook['id']} uses a shell operator pre-commit will not interpret",
                )

    def test_the_clean_export_hook_runs_the_real_replica(self):
        entry = str(self.hooks["ci-parity-clean-export"]["entry"])
        self.assertIn(
            "ci-local",
            entry,
            "the clean-export hook must run benchmarks/ci_local.py (via the pixi wrapper), not an approximation",
        )
        self.assertIn(
            "run_with_pixi.sh",
            entry,
            "the clean-export hook must survive a shell where pixi is not on PATH",
        )

    def test_the_inplace_hooks_keep_the_literal_ci_commands(self):
        lint_entry = str(self.hooks["ci-parity-inplace"]["entry"])
        test_entry = str(self.hooks["ci-parity-inplace-tests"]["entry"])
        self.assertIn("lint", lint_entry)
        self.assertIn("test", test_entry)
        # and nothing else: each hook runs exactly one CI command
        self.assertNotIn("test", lint_entry)
        self.assertNotIn("lint", test_entry)

    def test_both_hooks_go_through_the_pixi_wrapper(self):
        """The wrapper is what turns a pixi-less push into an actionable message.

        Its first release initially ran `pixi run …` directly, which fails with a
        bare exit 127 in pre-commit's non-login shell because ~/.pixi/bin is not
        on PATH there.
        """
        for hook_id in ("ci-parity-inplace", "ci-parity-inplace-tests", "ci-parity-clean-export"):
            entry = str(self.hooks[hook_id]["entry"])
            self.assertIn("run_with_pixi.sh", entry, f"{hook_id} must degrade gracefully without pixi on PATH")

    def test_the_pixi_wrapper_is_executable_and_handles_missing_home(self):
        self.assertTrue(WRAPPER.exists(), "the wrapper script must exist at its hook entry path")
        import os
        import subprocess
        import tempfile

        self.assertTrue(os.access(WRAPPER, os.X_OK), "the wrapper must be executable to be usable from a hook")
        # Neither PATH nor HOME need to be visible for a clean error, not a
        # bash "unbound variable" crash swallowed by the hook's exit code.
        result = subprocess.run(
            ["bash", str(WRAPPER), "--version"],
            env={"PATH": "/usr/bin:/bin"},
            capture_output=True,
            text=True,
        )
        self.assertEqual(
            result.returncode,
            127,
            "a pixi-less environment must exit 127 like a missing command",
        )
        self.assertIn("pixi is not installed", result.stderr)
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                ["bash", str(WRAPPER), "--version"],
                env={"PATH": "/usr/bin:/bin", "HOME": tmp},
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 127)
            self.assertIn("pixi is not installed", result.stderr)


if __name__ == "__main__":
    unittest.main()
