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
        _inplace = self.hooks.get("ci-parity-inplace")
        _clean = self.hooks.get("ci-parity-clean-export")
        self.assertIsNotNone(_inplace, "in-place lint+tests must stay a pre-push gate")
        self.assertIsNotNone(
            _clean,
            "the clean-export run (benchmarks/ci_local.py) must gate the push; "
            "without it the hook repeats the failure mode that kept CI red",
        )
        for hook in (_inplace, _clean):
            self.assertEqual(hook.get("stages"), ["pre-push"], f"{hook['id']} must run at pre-push")

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

    def test_the_inplace_hook_keeps_the_literal_ci_commands(self):
        entry = str(self.hooks["ci-parity-inplace"]["entry"])
        self.assertIn("lint", entry)
        self.assertIn("test", entry)

    def test_both_hooks_go_through_the_pixi_wrapper(self):
        """The wrapper is what turns a pixi-less push into an actionable message.

        Its first release initially ran `pixi run …` directly, which fails with a
        bare exit 127 in pre-commit's non-login shell because ~/.pixi/bin is not
        on PATH there.
        """
        for hook_id in ("ci-parity-inplace", "ci-parity-clean-export"):
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
