"""The benchmark comparators must be instruments, not labels.

A comparator earns its name by being a *different* algorithm reached through no
weightmask code. Three comparators here failed that and none of them said so:

* ``_simple_hough_baseline`` imported ``_detect_streaks_satdet`` and raised
  ImportError, so ``pixi run benchmark-synthetic`` exited non-zero;
* ``rubin_compatible_kht`` called ``detect_streaks`` -- a "comparator" scored
  against the thing it was meant to be compared to;
* ``simple_radon`` sat in the MegaCam manifest's ``comparators`` with no
  implementation anywhere, so requesting it silently produced no baseline;
* the two ``acstools`` handlers were ``try:`` around a dict literal, so the
  ``except`` was unreachable and both always answered "available" having run
  nothing.

These tests check the substitutable property, not just that the functions
import: each baseline must find a real line and must not manufacture one out of
a single hot pixel or pure noise.
"""

import unittest

import numpy as np

from tests.benchmarks.run import _mask_stats, _radon_baseline, _simple_hough_baseline

SHAPE = (300, 300)
RMS = np.full(SHAPE, 5.0, dtype=np.float32)
XS = np.arange(20, 280)


def _line_image(slope=0.5, intercept=40.0, value=60.0):
    data = np.zeros(SHAPE, dtype=np.float32)
    for x in XS:
        y = int(intercept + slope * x)
        if 0 <= y < SHAPE[0]:
            data[y - 1 : y + 2, x] = value
    return data


def _line_pixels(mask, slope=0.5, intercept=40.0):
    on = 0
    for x in XS:
        y = int(intercept + slope * x)
        if 0 <= y < SHAPE[0] and mask[y, x]:
            on += 1
    return on


class TestComparatorsAreRealDetectors(unittest.TestCase):
    def test_simple_hough_finds_a_real_line(self):
        on = _line_pixels(_simple_hough_baseline(_line_image(), RMS))
        self.assertGreater(on / len(XS), 0.9, "a plain Hough transform must recover a clean 260px line")

    def test_radon_finds_a_real_line(self):
        on = _line_pixels(_radon_baseline(_line_image(), RMS))
        self.assertGreater(on / len(XS), 0.9, "the matched-filter Radon search must recover the same line")

    def test_comparators_import_no_weightmask_detector(self):
        """The point of a comparator: different code, not the same code twice."""
        import ast
        import inspect
        import textwrap

        from tests.benchmarks import run as bench

        for baseline in (_simple_hough_baseline, _radon_baseline):
            # Walk the AST, not the text: these docstrings *name* the stages they
            # replaced, and matching on source would trip over their own history.
            tree = ast.parse(textwrap.dedent(inspect.getsource(baseline)))
            called = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    func = node.func
                    called.add(func.id if isinstance(func, ast.Name) else getattr(func, "attr", ""))
                if isinstance(node, ast.ImportFrom):
                    called.update(alias.name for alias in node.names)
                if isinstance(node, ast.Import):
                    called.update(alias.name for alias in node.names)
            self.assertNotIn("detect_streaks", called, f"{baseline.__name__} routes through the detector under test")
            for stage in ("_detect_streaks_houghpeaks", "_detect_streaks_contours", "_detect_streaks_mrt_like"):
                self.assertNotIn(stage, called, f"{baseline.__name__} reuses a production stage")
            self.assertIs(inspect.getmodule(baseline), bench)


class TestComparatorsDoNotManufactureTrails(unittest.TestCase):
    def test_a_single_hot_pixel_is_not_a_trail(self):
        data = np.zeros(SHAPE, dtype=np.float32)
        data[150, 50] = 60.0  # 12 sigma, but it is one pixel
        for name, baseline in (("simple_hough", _simple_hough_baseline), ("radon", _radon_baseline)):
            self.assertLess(int(baseline(data, RMS).sum()), 100, f"{name} turned one hot pixel into a trail")

    def test_pure_noise_produces_nothing(self):
        data = np.random.default_rng(0).normal(0.0, 5.0, SHAPE).astype(np.float32)
        for name, baseline in (("simple_hough", _simple_hough_baseline), ("radon", _radon_baseline)):
            self.assertEqual(int(baseline(data, RMS).sum()), 0, f"{name} masked pure noise")

    def test_mask_stats_reports_a_precision_of_one_on_a_clean_line(self):
        """Guard the scorer too: an all-hit mask must not be reported as 0.0."""
        predicted = np.zeros(SHAPE, dtype=bool)
        truth = np.zeros(SHAPE, dtype=bool)
        truth[100:103, 40:200] = True
        predicted[100:103, 40:200] = True
        stats = _mask_stats(predicted, truth)
        self.assertAlmostEqual(stats["precision"], 1.0)
        self.assertAlmostEqual(stats["recall"], 1.0)
        self.assertAlmostEqual(stats["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
