"""The pre-push hooks must actually deliver CI parity, not just claim it.

The original failure that cost four red CI pushes was invisible to the local
``pre-push`` hook: it ran ``pixi run lint && pixi run test`` ``in place``, where
gitignored data and a warm environment mask exactly the divergences that fail on
the runner. ``benchmarks/ci_local.py`` exists to close that gap by exporting
HEAD and running the steps there. These tests pin the hook contract so the
clean-export half cannot silently disappear again.
"""

import unittest
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
CONFIG = REPO / ".pre-commit-config.yaml"


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
            "pixi run ci-local",
            entry,
            "the clean-export hook must invoke benchmarks/ci_local.py, not an approximation",
        )

    def test_the_inplace_hook_keeps_the_literal_ci_commands(self):
        entry = str(self.hooks["ci-parity-inplace"]["entry"])
        self.assertIn("pixi run lint", entry)
        self.assertIn("pixi run test", entry)


if __name__ == "__main__":
    unittest.main()
