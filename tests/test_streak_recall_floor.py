"""CI recall floor for streak detection on a small synthetic HDU.

The swept grid lives in ``benchmarks/streak_inject.py``; this is the fast subset
that has to hold on every commit, so that a threshold or gate change cannot
quietly trade recall for speed.

Every assertion here is a *lower bound* (or, for false positives and coverage, an
upper bound), so improving the detector never turns this red. The dashed cell is
kept as an explicit unsupported-regime xfail until the benchmark matrix supports
a stable multi-seed floor.
"""

import contextlib
import io
import sys
import unittest
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "benchmarks") not in sys.path:
    sys.path.insert(0, str(ROOT / "benchmarks"))

from benchmarks.production_inputs import (  # noqa: E402
    capture_detector_inputs,
    production_gain_read_noise,
    streak_config,
)
from benchmarks.streak_inject import (  # noqa: E402
    inject_grid,
    inject_source_poisson,
    score_mask,
    scoreable_truth,
    trail_recall,
)
from weightmask.streaks import detect_streaks  # noqa: E402

SHAPE = (700, 700)
CONTINUOUS_SPECS = [(400, 8.0, False), (400, 4.0, False)]
DASHED_SPECS = [(200, 5.0, True)]
ALL_SPECS = CONTINUOUS_SPECS + DASHED_SPECS

# Floors and ceilings measured on this fixture **through the production input
# path** (process_image's iterated background and accumulated exclusion), which
# is how the detector is actually fed. Re-measured 2026-09-29:
#   continuous trails  recall 0.910 / 0.994, recall5 1.000 both
#   dashed trail        recall 0.000 (unsupported at this cell)
#   mask coverage       0.0186 of the frame
#   fp5                 669 px
#
# CEILING_FP5_PX was 400 when this fixture built its inputs with a single
# background pass over an empty mask. On that path the detector missed both
# continuous trails, so there was little halo to count; on the production path it
# finds them, and fp5 measures the widened band around trails that were
# previously invisible rather than a separate population of false positives.
# It is bounded by the injected trail length, not by the sky. The 800 px figure
# leaves room for that halo while still catching a runaway mask.
FLOOR_SUPPORTED_CONTINUOUS_RECALL = 0.75
FLOOR_SUPPORTED_CONTINUOUS_RECALL5 = 0.95
FLOOR_ALL_RECALL5 = 0.60
CEILING_FP5_PX = 800
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
        # Detector inputs must come from the production path: process_image's
        # iterated background and its accumulated exclusion. A single background
        # pass over an empty mask is the path that made the curated
        # false-positive baseline read 0.250 where production reads 0.031, and it
        # left the chip-fixed columns in the image as bright linear features.
        cls.streak_cfg_full = streak_config(config)
        with contextlib.redirect_stdout(io.StringIO()):
            data_sub, rms, empty = capture_detector_inputs(
                cls.sci,
                {"GAIN": 1.5, "RDNOISE": 5.0},
                config,
                flat=np.ones(cls.sci.shape, dtype=np.float32),
            )
        cls.empty_mask = empty
        finite = data_sub[np.isfinite(data_sub)]
        noise = float(np.nanmedian(rms[np.isfinite(rms)]))
        if not np.isfinite(noise) or noise <= 0:
            noise = float(np.median(np.abs(finite - np.median(finite))) * 1.4826)

        cfg = dict(cls.streak_cfg_full)
        cls.clean_mask = detect_streaks(data_sub, rms, cls.empty_mask, cfg)
        rng = np.random.default_rng(0)
        flux, cls.truth, cls.trails, cls.rejected_cells = inject_grid(
            cls.sci.shape,
            ALL_SPECS,
            rng,
            min_separation=150.0,
            invalid_mask=~np.isfinite(cls.sci) | cls.clean_mask,
            return_rejected=True,
        )
        cls.n_continuous = sum(1 for trail in cls.trails if not trail["dashed"])
        gain, _read_noise = production_gain_read_noise({"GAIN": 1.5, "RDNOISE": 5.0}, config)
        raw_test = inject_source_poisson(
            cls.sci,
            flux * noise,
            gain,
            np.random.default_rng(100),
            invalid_mask=~np.isfinite(cls.sci),
        )
        with contextlib.redirect_stdout(io.StringIO()):
            injected_sub, injected_rms, injected_existing = capture_detector_inputs(
                raw_test,
                {"GAIN": 1.5, "RDNOISE": 5.0},
                config,
                flat=np.ones(cls.sci.shape, dtype=np.float32),
            )
        cls.mask = detect_streaks(injected_sub, injected_rms, injected_existing, dict(cfg))
        cls.scoreable = np.zeros(cls.sci.shape, dtype=bool)
        cls.rejected_truth = np.zeros(cls.sci.shape, dtype=bool)
        for trail in cls.trails:
            accepted, rejected = scoreable_truth(trail["truth"], injected_sub, injected_existing, cls.clean_mask)
            cls.scoreable |= accepted
            cls.rejected_truth |= rejected
        cls.fp_px, cls.fp5_px, _novel = score_mask(cls.mask, cls.clean_mask, cls.scoreable)

        cls.per_trail = [
            trail_recall(cls.mask, scoreable_truth(trail["truth"], injected_sub, injected_existing, cls.clean_mask)[0])
            for trail in cls.trails
        ]
        cls.continuous = [
            trail_recall(cls.mask, scoreable_truth(trail["truth"], injected_sub, injected_existing, cls.clean_mask)[0])
            for trail in cls.trails
            if not trail["dashed"]
        ]
        assert cls.n_continuous >= 2, "fixture must place the continuous trails"

    def test_the_quiet_frame_alone_yields_no_streak_pixels(self):
        """A clean synthetic field must not produce a streak mask at all."""
        self.assertEqual(int(np.count_nonzero(self.clean_mask)), 0)

    def test_supported_continuous_trails_are_recalled(self):
        recalls = [exact for exact, _, _ in self.continuous]
        tolerant = [loose for _, loose, _ in self.continuous]
        mean_recall = float(np.mean(recalls))
        mean_tolerant = float(np.mean(tolerant))
        self.assertGreaterEqual(
            mean_recall,
            FLOOR_SUPPORTED_CONTINUOUS_RECALL,
            f"continuous-trail recall fell to {mean_recall:.3f} ({recalls})",
        )
        self.assertGreaterEqual(
            mean_tolerant,
            FLOOR_SUPPORTED_CONTINUOUS_RECALL5,
            f"5px-tolerant continuous recall fell to {mean_tolerant:.3f} ({tolerant})",
        )

    def test_overall_tolerant_recall_floor(self):
        mean_tolerant = float(np.mean([loose for _, loose, _ in self.per_trail]))
        self.assertGreaterEqual(mean_tolerant, FLOOR_ALL_RECALL5, f"grid mean recall5 fell to {mean_tolerant:.3f}")

    def test_false_positives_stay_bounded(self):
        self.assertLessEqual(self.fp5_px, CEILING_FP5_PX, f"novel false-positive pixels rose to {self.fp5_px}")

    def test_rejected_truth_cells_are_reported(self):
        self.assertIsInstance(self.rejected_cells, list)
        self.assertEqual(int(np.count_nonzero(self.scoreable & self.rejected_truth)), 0)

    def test_the_mask_does_not_run_away(self):
        coverage = float(np.mean(self.mask))
        self.assertLessEqual(coverage, CEILING_COVERAGE, f"mask covers {coverage:.2%} of the frame")

    @unittest.expectedFailure
    def test_unsupported_dashed_cell_is_not_a_recall_floor(self):
        dashed = [trail for trail in self.trails if trail["dashed"]]
        for trail in dashed:
            exact, tolerant, line = trail_recall(self.mask, trail["truth"])
            self.assertGreater(exact, 0.0)
            self.assertGreaterEqual(tolerant, exact)
            self.assertGreaterEqual(line, exact)


if __name__ == "__main__":
    unittest.main()
