"""A column that repeats on other exposures is a detector prior, not a trail."""

import unittest
import weakref

import numpy as np

from weightmask.streaks import _profile_outliers, persistent_axis_mask


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

    def test_nonfinite_pixels_do_not_suppress_repeated_hot_axes(self):
        frames = [np.zeros((32, 24), dtype=np.float32) for _ in range(2)]
        for frame in frames:
            frame[:, 5] = 20.0
        frames[0][3, 5] = np.nan
        frames[1][4, 5] = np.inf

        mask = persistent_axis_mask(frames, min_other=2)

        self.assertTrue(bool(np.all(mask[:, 5])))

    def test_finite_profile_preserves_overflow_fallback(self):
        profile = np.array([-1e308, -1e308, 1e308, 1e308])

        with np.errstate(over="ignore"):
            outliers = _profile_outliers(profile, sigma=3.0)

        np.testing.assert_array_equal(outliers, [False, False, True, True])

    def test_all_invalid_axes_are_not_flagged(self):
        frames = [np.zeros((32, 24), dtype=np.float32) for _ in range(2)]
        for frame in frames:
            frame[:, 5] = 20.0
            frame[:, 9] = np.nan
            frame[11, :] = np.inf

        mask = persistent_axis_mask(frames, min_other=2)

        self.assertTrue(bool(np.all(mask[:, 5])))
        self.assertFalse(bool(np.all(mask[:, 9])))
        self.assertFalse(bool(np.all(mask[11, :])))

    def test_frames_are_released_while_profiles_stream(self):
        references = []

        def frames():
            for _ in range(3):
                frame = np.zeros((32, 24), dtype=np.float32)
                references.append(weakref.ref(frame))
                yield frame
                del frame
                self.assertTrue(all(reference() is None for reference in references))

        mask = persistent_axis_mask(frames(), min_other=2)

        self.assertEqual(mask.shape, (32, 24))


if __name__ == "__main__":
    unittest.main()
