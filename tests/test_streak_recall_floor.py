"""CI recall floor for streak detection on a small synthetic HDU.

The swept grid lives in ``benchmarks/streak_inject.py``; this is the fast subset
that has to hold on every commit, so that a threshold or gate change cannot
quietly trade recall for speed.

Every assertion here is a *lower bound* (or, for false positives and coverage, an
upper bound), so improving the detector never turns this red. The one number that
is deliberately not asserted is the dashed-trail recall: it is 0 today, and
pinning it at 0 would make fixing it a test failure. It is recorded in
``docs/detector_audit.md`` instead.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.streak_inject import inject_grid, score_mask, trail_recall  # noqa: E402
from weightmask.background import estimate_background  # noqa: E402
from weightmask.streaks import detect_streaks  # noqa: E402

SHAPE = (700, 700)
CONTINUOUS_SPECS = [(400, 8.0, False), (400, 4.0, False)]
DASHED_SPECS = [(200, 5.0, True)]
ALL_SPECS = CONTINUOUS_SPECS + DASHED_SPECS

# Floors measured on this fixture before the assertion was written; margins are
# deliberately loose enough to absorb a different BLAS/skimage minor version.
FLOOR_CONTINUOUS_RECALL = 0.75
FLOOR_CONTINUOUS_RECALL5 = 0.90
FLOOR_ALL_RECALL5 = 0.60
CEILING_FP5_PX = 400
CEILING_COVERAGE = 0.05


def synthetic_hdu(shape=SHAPE, seed=0, noise=5.0, sky=1000.0, n_stars=14):
    """A quiet CCD-like frame: sky + read noise + a scatter of stars for clutter."""
    rng = np.random.default_rng(seed)
    data = rng.normal(sky, noise, shape).astype(np.float32)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    for _ in range(n_stars):
        cy, cx = rng.uniform(30, shape[0] - 30), rng.uniform(30, shape[1] - 30)
        amplitude = rng.uniform(50, 4000)
        sigma = rng.uniform(1.5, 3.0)
        data += (amplitude * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2)))).astype(np.float32)
    return data


class TestStreakRecallFloor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        config = yaml.safe_load(open("weightmask.yml"))
        cls.streak_cfg = dict(config["streak_masking"])
        cls.streak_cfg["enable"] = True

        cls.sci = synthetic_hdu()
        cls.empty_mask = np.zeros(cls.sci.shape, dtype=bool)
        sky, rms = estimate_background(cls.sci, cls.empty_mask, config["sep_background"])
        data_sub = cls.sci - sky
        finite = data_sub[np.isfinite(data_sub)]
        noise = float(np.median(np.abs(finite - np.median(finite))) * 1.4826)

        rng = np.random.default_rng(0)
        flux, cls.truth, cls.trails = inject_grid(cls.sci.shape, ALL_SPECS, rng, min_separation=150.0)
        cls.n_continuous = sum(1 for trail in cls.trails if not trail["dashed"])

        cls.clean_mask = detect_streaks(data_sub, rms, cls.empty_mask, dict(cls.streak_cfg))
        cls.mask = detect_streaks(data_sub + flux * noise, rms, cls.empty_mask, dict(cls.streak_cfg))
        cls.fp_px, cls.fp5_px, _novel = score_mask(cls.mask, cls.clean_mask, cls.truth)

        cls.per_trail = [trail_recall(cls.mask, trail["truth"]) for trail in cls.trails]
        cls.continuous = [trail_recall(cls.mask, trail["truth"]) for trail in cls.trails if not trail["dashed"]]
        assert cls.n_continuous >= 2, "fixture must place the continuous trails"

    def test_the_quiet_frame_alone_yields_no_streak_pixels(self):
        """A clean synthetic field must not produce a streak mask at all."""
        self.assertEqual(int(np.count_nonzero(self.clean_mask)), 0)

    def test_continuous_trails_are_recalled(self):
        recalls = [exact for exact, _ in self.continuous]
        tolerant = [loose for _, loose in self.continuous]
        mean_recall = float(np.mean(recalls))
        mean_tolerant = float(np.mean(tolerant))
        self.assertGreaterEqual(
            mean_recall,
            FLOOR_CONTINUOUS_RECALL,
            f"continuous-trail recall fell to {mean_recall:.3f} ({recalls})",
        )
        self.assertGreaterEqual(
            mean_tolerant,
            FLOOR_CONTINUOUS_RECALL5,
            f"5px-tolerant continuous recall fell to {mean_tolerant:.3f} ({tolerant})",
        )

    def test_overall_tolerant_recall_floor(self):
        mean_tolerant = float(np.mean([loose for _, loose in self.per_trail]))
        self.assertGreaterEqual(mean_tolerant, FLOOR_ALL_RECALL5, f"grid mean recall5 fell to {mean_tolerant:.3f}")

    def test_false_positives_stay_bounded(self):
        self.assertLessEqual(self.fp5_px, CEILING_FP5_PX, f"novel false-positive pixels rose to {self.fp5_px}")

    def test_the_mask_does_not_run_away(self):
        coverage = float(np.mean(self.mask))
        self.assertLessEqual(coverage, CEILING_COVERAGE, f"mask covers {coverage:.2%} of the frame")

    def test_dashed_gap_is_recorded_not_asserted(self):
        """Documents today's known gap without freezing it in place."""
        dashed = [trail for trail in self.trails if trail["dashed"]]
        for trail in dashed:
            exact, tolerant = trail_recall(self.mask, trail["truth"])
            self.assertGreaterEqual(exact, 0.0)
            self.assertGreaterEqual(tolerant, exact)
            print(
                f"[knowngap] dashed len={trail['length']} sig={trail['peak_sig']}: "
                f"recall={exact:.3f} recall5={tolerant:.3f}"
            )


if __name__ == "__main__":
    unittest.main()
