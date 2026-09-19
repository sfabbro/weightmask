"""Equivalence guarantees for the streak throughput work.

The Phase 1 speedups are meant to be byte-identical:

* ``_prune_small_edges`` replaced a per-region ``regionprops`` loop with a
  vectorized reproduction of skimage's perimeter (with an exactness fallback on
  the keep/drop threshold), and
* ``_StreakImageCache`` shares the mask-independent streak preparation across
  the satdet primary pass, the unmasked retry, and the corridor build.

These tests pin both down against their own reference implementation.
"""

import unittest
from unittest import mock

import numpy as np
from skimage.measure import label, regionprops

from weightmask import streaks
from weightmask.streaks import _prune_small_edges, detect_streaks


def _prune_small_edges_reference(edge_mask, min_perimeter):
    """The pre-optimization implementation, verbatim, as the reference."""
    if min_perimeter <= 0:
        return edge_mask
    labeled = label(edge_mask, connectivity=2)
    cleaned = np.zeros_like(edge_mask, dtype=bool)
    for region in regionprops(labeled):
        if region.perimeter >= min_perimeter:
            coords = region.coords
            cleaned[coords[:, 0], coords[:, 1]] = True
    return cleaned


def _cluttered_edge_mask(seed=11, shape=(160, 160)):
    """Isolated specks, touching blocks and a long diagonal in one mask."""
    rng = np.random.default_rng(seed)
    mask = rng.random(shape) < 0.025
    for _ in range(120):
        row = int(rng.integers(0, shape[0] - 8))
        col = int(rng.integers(0, shape[1] - 8))
        height = int(rng.integers(1, 8))
        width = int(rng.integers(1, 8))
        mask[row : row + height, col : col + width] = True
    mask |= np.eye(shape[0], shape[1], dtype=bool)
    return mask


def _streak_config():
    return {
        "enable": True,
        "mode": "auto_ground",
        "retry_without_existing_mask": True,
        "satdet_params": {
            "bin_factor": 2,
            "rescale_percentiles": [1.0, 99.5],
            "gaussian_sigmas": [1.0, 1.5],
            "canny_low_threshold": 0.05,
            "canny_high_threshold": 0.2,
            "small_edge_perimeter": 5,
            "hough_threshold": 5,
            "hough_min_line_length": 20,
            "hough_max_line_gap": 6,
            "cluster_angle_tol_deg": 4.0,
            "cluster_rho_tol_px": 16.0,
            "min_cluster_segments": 3,
            "edge_buffer": 16,
            "min_edge_touches": 0,
            "min_interior_span": 40.0,
            "min_segment_density": 0.015,
            "candidate_corridor_radius": 8,
            "max_existing_mask_fraction": 0.8,
            "confidence_threshold": 0.2,
        },
        "mrt_rescue_params": {"theta_step_deg": 4.0, "peak_threshold_sig": 3.0, "max_candidates": 2},
        "mask_params": {
            "strip_length": 120,
            "strip_width": 48,
            "profile_sigma_threshold": 1.0,
            "profile_percentile": 75.0,
            "padding": 2,
            "min_mask_pixels": 10,
            "min_row_hits": 4,
            "min_row_hit_fraction": 0.2,
            "max_support_width": 12,
        },
        "enable_sparse_ransac": False,
    }


def _streak_scene(shape=(256, 256)):
    """Noise with a masked star field, plus a real trail for the retry to find."""
    rng = np.random.default_rng(5)
    data = rng.normal(0.0, 1.0, shape).astype(np.float32)
    rms = np.full(shape, 5.0, dtype=np.float32)
    existing_mask = np.zeros(shape, dtype=bool)
    ys = rng.integers(10, shape[0] - 10, size=60)
    xs = rng.integers(10, shape[1] - 10, size=60)
    data[ys, xs] += 25.0
    for y, x in zip(ys, xs):
        existing_mask[max(0, y - 2) : y + 3, max(0, x - 2) : x + 3] = True
    return data, rms, existing_mask


class TestPruneSmallEdgesEquivalence(unittest.TestCase):
    def test_matches_regionprops_loop(self):
        edge_mask = _cluttered_edge_mask()
        # 0.2071067811865476 is one of skimage's perimeter weights, so cuts at
        # that value land exactly on regions' measured perimeters.
        for cut in (0.2071067811865476, 1.0, 4.0, 8.0, 24.0, 40.0, 100.0):
            with self.subTest(cut=cut):
                np.testing.assert_array_equal(
                    _prune_small_edges(edge_mask, cut),
                    _prune_small_edges_reference(edge_mask, cut),
                )

    def test_matches_every_perimeter_in_a_realistic_mask(self):
        edge_mask = _cluttered_edge_mask(seed=3, shape=(120, 120))
        perimeters = sorted({region.perimeter for region in regionprops(label(edge_mask, connectivity=2))})
        for cut in perimeters[:25]:  # a cut sitting exactly on a region's perimeter
            with self.subTest(cut=cut):
                np.testing.assert_array_equal(
                    _prune_small_edges(edge_mask, float(cut)),
                    _prune_small_edges_reference(edge_mask, float(cut)),
                )

    def test_single_pixel_regions_are_dropped(self):
        edge_mask = np.eye(40, dtype=bool)
        np.testing.assert_array_equal(_prune_small_edges(edge_mask, 1.0), _prune_small_edges_reference(edge_mask, 1.0))

    def test_disabled_threshold_returns_input(self):
        edge_mask = _cluttered_edge_mask(shape=(32, 32))
        self.assertIs(_prune_small_edges(edge_mask, 0.0), edge_mask)

    def test_empty_mask(self):
        edge_mask = np.zeros((16, 16), dtype=bool)
        cleaned = _prune_small_edges(edge_mask, 40.0)
        self.assertEqual(cleaned.shape, edge_mask.shape)
        self.assertFalse(cleaned.any())


class TestStreakImageCache(unittest.TestCase):
    def test_shared_core_equals_recomputed_preparation(self):
        rng = np.random.default_rng(3)
        data = rng.normal(size=(96, 96)).astype(np.float32)
        mask = rng.random((96, 96)) < 0.1
        cache = streaks._StreakImageCache()
        core = cache.core(data)
        for exclusion in (None, mask):
            shared = streaks._prepare_streak_image(data, exclusion, core=core)
            recomputed = streaks._prepare_streak_image(data, exclusion)
            np.testing.assert_array_equal(shared, recomputed)
            self.assertEqual(shared.dtype, recomputed.dtype)

    def test_binned_entry_is_memoized_and_paired_with_its_core(self):
        rng = np.random.default_rng(4)
        data = rng.normal(size=(64, 64)).astype(np.float32)
        cache = streaks._StreakImageCache()
        binned, binned_core = cache.binned(data, 2)
        again, again_core = cache.binned(data, 2)
        self.assertIs(again, binned)
        self.assertIs(again_core, binned_core)
        self.assertEqual(binned.shape, (32, 32))
        np.testing.assert_array_equal(
            streaks._prepare_streak_image(binned, None, core=binned_core),
            streaks._prepare_streak_image(binned, None),
        )

    def test_cache_rebinds_when_the_source_array_changes(self):
        cache = streaks._StreakImageCache()
        first = np.zeros((16, 16), dtype=np.float32)
        second = np.zeros((16, 16), dtype=np.float32)
        second[8, 8] = 100.0  # a compact source the preparation keeps
        self.assertFalse(cache.core(first).any())
        self.assertTrue(cache.core(second).any())

    def test_satdet_pair_prepares_each_array_once(self):
        data, rms, existing_mask = _streak_scene()
        config = _streak_config()
        original = streaks._streak_image_core

        with mock.patch.object(streaks, "_streak_image_core", side_effect=original) as shared_spy:
            cache = streaks._StreakImageCache()
            primary, _, _ = streaks._detect_streaks_satdet(data, rms, existing_mask, config, cache)
            retry, _, _ = streaks._detect_streaks_satdet(data, rms, None, config, cache)
        # One binned prescreen image + one full-resolution corridor image.
        self.assertEqual(shared_spy.call_count, 2)

        with mock.patch.object(streaks, "_streak_image_core", side_effect=original) as fresh_spy:
            fresh_primary, _, _ = streaks._detect_streaks_satdet(
                data, rms, existing_mask, config, streaks._StreakImageCache()
            )
            fresh_retry, _, _ = streaks._detect_streaks_satdet(data, rms, None, config, streaks._StreakImageCache())
        self.assertEqual(fresh_spy.call_count, 2 * shared_spy.call_count)

        np.testing.assert_array_equal(primary, fresh_primary)
        np.testing.assert_array_equal(retry, fresh_retry)

    def test_detect_streaks_is_unchanged_when_cores_are_never_shared(self):
        data, rms, existing_mask = _streak_scene()
        config = _streak_config()
        original = streaks._prepare_streak_image
        shared = detect_streaks(data, rms, existing_mask, config)
        # Dropping the ``core`` argument reproduces the pre-cache call pattern
        # (every preparation recomputed in full).
        with mock.patch.object(streaks, "_prepare_streak_image", lambda d, m, core=None: original(d, m)):
            recomputed = detect_streaks(data, rms, existing_mask, config)
        self.assertTrue(shared.any())
        np.testing.assert_array_equal(shared, recomputed)


def _quiet_scene(shape=(256, 256)):
    """Noise well below every streak threshold: nothing is accepted up front."""
    rng = np.random.default_rng(13)
    data = rng.normal(0.0, 0.4, shape).astype(np.float32)
    return data, np.full(shape, 5.0, dtype=np.float32), None


class TestMrtRescueGate(unittest.TestCase):
    def test_rescue_runs_by_default_and_can_be_disabled(self):
        data, rms, existing_mask = _quiet_scene()
        config = _streak_config()
        # Make the primary and retry stages reject every candidate, which is the
        # low-confidence state the rescue exists for.
        config["satdet_params"]["confidence_threshold"] = 1.5
        original = streaks._detect_streaks_mrt_like
        calls = []

        def counting(*args, **kwargs):
            calls.append(1)
            return original(*args, **kwargs)

        with mock.patch.object(streaks, "_detect_streaks_mrt_like", side_effect=counting):
            detect_streaks(data, rms, existing_mask, config)
        self.assertEqual(len(calls), 1, "scene must reach the low-confidence rescue pass")

        calls.clear()
        config["mrt_rescue_params"] = {**config["mrt_rescue_params"], "enable": False}
        with mock.patch.object(streaks, "_detect_streaks_mrt_like", side_effect=counting):
            detect_streaks(data, rms, existing_mask, config)
        self.assertEqual(len(calls), 0)


if __name__ == "__main__":
    unittest.main()
