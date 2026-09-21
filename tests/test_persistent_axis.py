"""A column that repeats on other exposures is a detector prior, not a trail."""

import unittest

import numpy as np

from weightmask.streaks import persistent_axis_mask


class TestPersistentAxisMask(unittest.TestCase):
    def test_two_frames_flag_a_column_and_one_frame_does_not(self):
        rng = np.random.default_rng(0)
        frames = [rng.normal(0.0, 1.0, (32, 24)) for _ in range(3)]
        for frame in frames:
            frame[:, 5] += 20.0
            frame[7, :] += 20.0
        frames[0][:, 8] += 20.0
        mask = persistent_axis_mask(frames, min_other=2)
        self.assertTrue(bool(np.all(mask[:, 5])))
        self.assertTrue(bool(np.all(mask[7, :])))
        self.assertFalse(bool(np.all(mask[:, 8])))

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            persistent_axis_mask([])


if __name__ == "__main__":
    unittest.main()
