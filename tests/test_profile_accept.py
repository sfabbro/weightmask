"""A voted streak mask is kept only where the pixels form a thin line."""

import unittest

import numpy as np

from weightmask.streaks import _gate_mask_by_profile


class TestProfileAccept(unittest.TestCase):
    def test_thin_line_is_kept_and_a_wide_band_is_dropped(self):
        line = np.zeros((80, 80), dtype=bool)
        for i in range(10, 70):
            line[i, i] = True
            line[i, min(i + 1, 79)] = True
        wide = np.zeros((80, 80), dtype=bool)
        wide[20:60, 10:50] = True
        self.assertGreater(int(np.count_nonzero(_gate_mask_by_profile(line))), 40)
        self.assertEqual(int(np.count_nonzero(_gate_mask_by_profile(wide))), 0)


if __name__ == "__main__":
    unittest.main()
