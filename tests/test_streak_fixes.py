"""Streak behavior regression tests (kept with the streak overhaul commit).

Unknown modes must fail fast instead of silently returning empty masks;
production accepts only auto_ground (method is a legacy alias for mode).
"""

import unittest

import numpy as np


class TestStreakUnknownMode(unittest.TestCase):
    def test_unknown_mode_raises(self):
        from weightmask.streaks import detect_streaks

        data = np.zeros((64, 64), dtype=np.float32)
        with self.assertRaises(ValueError):
            detect_streaks(data, None, None, {"enable": True, "mode": "bogus"})

    def test_only_auto_ground_accepted(self):
        from weightmask.streaks import _resolve_streak_mode

        self.assertEqual(_resolve_streak_mode({"mode": "auto_ground"}), "auto_ground")
        self.assertEqual(_resolve_streak_mode({"method": "auto_ground"}), "auto_ground")
        self.assertEqual(_resolve_streak_mode({}), "auto_ground")
        with self.assertRaises(ValueError):
            _resolve_streak_mode({"mode": "satdet"})
        with self.assertRaises(ValueError):
            _resolve_streak_mode({"method": "mrt_only"})


if __name__ == "__main__":
    unittest.main()
