"""Streak behavior regression tests (kept with the streak overhaul commit).

Unknown modes must fail fast instead of silently returning empty masks;
the method alias must keep resolving after the mode cleanup.
"""

import unittest

import numpy as np


class TestStreakUnknownMode(unittest.TestCase):
    def test_unknown_mode_raises(self):
        from weightmask.streaks import detect_streaks

        data = np.zeros((64, 64), dtype=np.float32)
        with self.assertRaises(ValueError):
            detect_streaks(data, None, None, {"enable": True, "mode": "bogus"})

    def test_method_alias_still_resolves(self):
        from weightmask.streaks import _resolve_streak_mode

        self.assertEqual(_resolve_streak_mode({"mode": "satdet"}), "satdet_only")
        self.assertEqual(_resolve_streak_mode({"method": "mrt_only"}), "mrt_only")


if __name__ == "__main__":
    unittest.main()
