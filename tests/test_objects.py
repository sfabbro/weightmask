import unittest
import numpy as np
import sep
from weightmask.objects import detect_objects

class TestObjects(unittest.TestCase):
    def test_detect_objects_exception_handling(self):
        """
        Test that detect_objects returns an empty mask when an exception
        (e.g., ValueError from sep.extract due to 1D arrays) occurs.
        """
        # SEP requires 2D arrays, providing 1D will cause sep.extract to raise ValueError
        data_sub = np.ones(10, dtype=np.float32)
        bkg_rms_map = np.ones(10, dtype=np.float32)

        mask = detect_objects(data_sub, bkg_rms_map, None, {})

        self.assertEqual(mask.shape, data_sub.shape)
        self.assertFalse(np.any(mask))
        self.assertEqual(mask.dtype, bool)

    def test_detect_objects_happy_path(self):
        """
        Basic smoke test for the happy path.
        """
        data_sub = np.zeros((100, 100), dtype=np.float32)
        bkg_rms_map = np.ones((100, 100), dtype=np.float32)

        # Add a fake object
        data_sub[45:55, 45:55] = 100.0

        mask = detect_objects(data_sub, bkg_rms_map, None, {'extract_thresh': 3.0})

        self.assertEqual(mask.shape, data_sub.shape)
        self.assertTrue(np.any(mask))
        self.assertEqual(mask.dtype, bool)

if __name__ == '__main__':
    unittest.main()
