"""On-disk flat bad-mask cache (Phase 2): keying, reuse and fallbacks.

``compute_flat_bad_mask_cached`` reuses one flat HDU's bad-pixel mask across
every exposure that shares the flat. Products must never depend on the cache
being present, current or writable, so these tests pin the key sensitivity and
each fallback path.
"""

import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from weightmask import bad


def _flat(shape=(64, 96), seed=1):
    rng = np.random.default_rng(seed)
    flat = rng.uniform(0.8, 1.2, shape).astype(np.float32)
    flat[5, 5] = 0.0  # a dead pixel for the local-median test to find
    return flat


class TestFlatBadMaskCache(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.flat_path = os.path.join(self.dir, "flat.fits.fz")
        with open(self.flat_path, "wb") as handle:
            handle.write(b"flat-bytes")
        self.cfg = {"local_filter_size": 15, "local_low_thresh": 0.5, "local_high_thresh": 2.0}

    def tearDown(self):
        self._tmp.cleanup()

    def _cache_file(self, **overrides):
        options = {
            "flat_cfg": self.cfg,
            "flat_path": self.flat_path,
            "hdu_index": 3,
            "shape": (64, 96),
            "tile_size": 1024,
        }
        options.update(overrides)
        return bad.flat_bad_mask_cache_file(**options)

    def test_cached_mask_equals_direct_computation(self):
        flat = _flat()
        cached = bad.compute_flat_bad_mask_cached(flat, self.cfg, 1024, flat_path=self.flat_path, hdu_index=3)
        np.testing.assert_array_equal(cached, bad.compute_flat_bad_mask(flat, self.cfg, 1024))
        self.assertEqual(cached.dtype, np.bool_)

    def test_key_is_stable_and_sensitive_to_everything_that_matters(self):
        base = self._cache_file()
        self.assertEqual(base, self._cache_file())
        self.assertNotEqual(base, self._cache_file(hdu_index=4))
        self.assertNotEqual(base, self._cache_file(tile_size=512))
        self.assertNotEqual(base, self._cache_file(shape=(64, 97)))
        self.assertNotEqual(base, self._cache_file(flat_cfg={**self.cfg, "local_low_thresh": 0.6}))
        self.assertIsNone(self._cache_file(flat_path=None))
        self.assertIsNone(self._cache_file(flat_cfg={**self.cfg, "bad_mask_cache": False}))
        with open(self.flat_path, "wb") as handle:
            handle.write(b"flat-bytes-changed")  # replaced flat must invalidate
        self.assertNotEqual(base, self._cache_file())

    def test_cache_control_keys_do_not_change_the_key(self):
        base = self._cache_file()
        self.assertEqual(base, self._cache_file(flat_cfg={**self.cfg, "bad_mask_cache": True}))
        self.assertEqual(base, self._cache_file(flat_cfg={**self.cfg, "bad_mask_cache_dir": None}))

    def test_second_call_reuses_the_stored_mask(self):
        flat = _flat()
        with mock.patch.object(bad, "compute_flat_bad_mask", side_effect=bad.compute_flat_bad_mask) as spy:
            first = bad.compute_flat_bad_mask_cached(flat, self.cfg, 1024, flat_path=self.flat_path, hdu_index=3)
            second = bad.compute_flat_bad_mask_cached(flat, self.cfg, 1024, flat_path=self.flat_path, hdu_index=3)
        self.assertEqual(spy.call_count, 1)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(os.path.exists(self._cache_file()))

    def test_disabled_cache_recomputes(self):
        flat = _flat()
        cfg = {**self.cfg, "bad_mask_cache": False}
        with mock.patch.object(bad, "compute_flat_bad_mask", side_effect=bad.compute_flat_bad_mask) as spy:
            bad.compute_flat_bad_mask_cached(flat, cfg, 1024, flat_path=self.flat_path, hdu_index=3)
            bad.compute_flat_bad_mask_cached(flat, cfg, 1024, flat_path=self.flat_path, hdu_index=3)
        self.assertEqual(spy.call_count, 2)
        self.assertFalse(os.path.exists(self._cache_file()))

    def test_without_a_flat_path_nothing_is_cached(self):
        flat = _flat()
        with mock.patch.object(bad, "compute_flat_bad_mask", side_effect=bad.compute_flat_bad_mask) as spy:
            bad.compute_flat_bad_mask_cached(flat, self.cfg, 1024)
        self.assertEqual(spy.call_count, 1)

    def test_corrupt_entry_is_recomputed_and_repaired(self):
        flat = _flat()
        good = bad.compute_flat_bad_mask_cached(flat, self.cfg, 1024, flat_path=self.flat_path, hdu_index=3)
        path = self._cache_file()
        with open(path, "wb") as handle:
            handle.write(b"not an npy file")
        again = bad.compute_flat_bad_mask_cached(flat, self.cfg, 1024, flat_path=self.flat_path, hdu_index=3)
        np.testing.assert_array_equal(good, again)
        np.testing.assert_array_equal(np.load(path), good)

    def test_empty_entry_is_recomputed(self):
        """A zero-byte entry must fall back: numpy raises EOFError, not OSError."""
        flat = _flat()
        path = self._cache_file()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb"):
            pass
        mask = bad.compute_flat_bad_mask_cached(flat, self.cfg, 1024, flat_path=self.flat_path, hdu_index=3)
        np.testing.assert_array_equal(mask, bad.compute_flat_bad_mask(flat, self.cfg, 1024))

    def test_shape_mismatch_in_entry_is_ignored(self):
        flat = _flat()
        path = self._cache_file()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.save(path, np.zeros((4, 4), dtype=bool))
        mask = bad.compute_flat_bad_mask_cached(flat, self.cfg, 1024, flat_path=self.flat_path, hdu_index=3)
        self.assertEqual(mask.shape, flat.shape)
        np.testing.assert_array_equal(mask, bad.compute_flat_bad_mask(flat, self.cfg, 1024))

    def test_unwritable_cache_directory_falls_back(self):
        flat = _flat()
        blocker = os.path.join(self.dir, "blocker")
        with open(blocker, "wb") as handle:
            handle.write(b"x")
        cfg = {**self.cfg, "bad_mask_cache_dir": os.path.join(blocker, "cache")}
        mask = bad.compute_flat_bad_mask_cached(flat, cfg, 1024, flat_path=self.flat_path, hdu_index=3)
        np.testing.assert_array_equal(mask, bad.compute_flat_bad_mask(flat, self.cfg, 1024))


if __name__ == "__main__":
    unittest.main()
