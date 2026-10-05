"""The Radon rescue is gone, and the deletions are pinned.

The rescue was the *sensitive* stage -- the one meant to find what the cheap
prescreen misses. Before removal it was measured, not assumed:

* over 83 real amps it accepted on 3, and all three were false positives, while
  houghpeaks explained all 23,035 px of the real detections;
* on injected trails at 4/6/8/12 sigma, two lengths, two seeds, it moved
  ``recall_line`` by +0.000 in 8 of 8 cells;
* it cost 121.7 s/amp of a 125.7 s/amp stage.

So this file no longer tests a gate. It asserts the stage is *absent* -- deleted
functions, deleted config, and a detector that still runs and still finds real
trails without it. The recall curve that justified the removal lives in
``benchmarks/streak_recall_floor.py``; the cost attribution in
``benchmarks/streak_stage_sweep.py``. A future proposal to restore a faint-trail
stage must beat the recall curve it can now measure, not argue from this file.
"""

import contextlib
import io
import json
import unittest
from pathlib import Path

import numpy as np
import yaml

from weightmask import streaks
from weightmask.streaks import detect_streaks

REPO = Path(__file__).resolve().parents[1]


def _shipped_streak_config():
    return yaml.safe_load(open(REPO / "weightmask.yml"))["streak_masking"]


def _quiet_scene(shape=(256, 256)):
    rng = np.random.default_rng(5)
    data = rng.normal(0.0, 1.0, shape).astype(np.float32)
    return data, np.full(shape, 5.0, dtype=np.float32), np.zeros(shape, dtype=bool)


class TestRescueStageIsGone(unittest.TestCase):
    def test_streaks_module_has_no_radon_stage(self):
        """Deleted, not merely disabled: no dead code left behind."""
        for gone in (
            "_detect_streaks_mrt_like",
            "_radon_projections",
            "_radon_pad_offsets",
            "_valid_rho_range",
            "_subtract_hot_axis_bands",
            "_PERIMETER_4CONN_CORNER_WEIGHT",
            "_PERIMETER_HV_STEP_WEIGHT",
            "_PERIMETER_DIAG_STEP_WEIGHT",
        ):
            self.assertFalse(hasattr(streaks, gone), f"{gone} should be removed")

    def test_debug_flag_populates_last_run_without_mrt(self):
        data, rms, existing_mask = _quiet_scene((64, 64))
        config = dict(_shipped_streak_config())
        config["debug"] = True
        with contextlib.redirect_stdout(io.StringIO()):
            detect_streaks(data, rms, existing_mask, config)
        self.assertIn("_last_run", config, "debug: True must record diagnostics in config['_last_run']")
        self.assertNotIn("mrt", config["_last_run"], "obsolete 'mrt' key must not appear in _last_run")
        self.assertEqual(set(config["_last_run"].keys()), {"mode", "houghpeaks", "contours", "sparse_ransac"})

    def test_shipped_config_carries_no_rescue_or_satdet_keys(self):
        streak_masking = _shipped_streak_config()
        for gone in (
            "mrt_rescue_params",
            "satdet_params",
            "retry_without_existing_mask",
            "retry_if_area_fraction_exceeds",
            "retry_if_support_width_exceeds",
        ):
            self.assertNotIn(gone, streak_masking, f"{gone} is dead config")

    def test_a_stale_rescue_key_fails_before_detection(self):
        data, rms, existing_mask = _quiet_scene()
        config = dict(_shipped_streak_config())
        config["mrt_rescue_params"] = {"enable": True, "max_candidates": 4}
        with self.assertRaisesRegex(ValueError, "mrt_rescue_params"):
            detect_streaks(data, rms, existing_mask, config)


class TestDetectorStillWorksWithoutTheRescue(unittest.TestCase):
    """Removing the rescue must not have removed the detector."""

    def test_a_long_bright_line_is_still_detected(self):
        data, rms, existing_mask = _quiet_scene((400, 400))
        data[195:199, 20:380] = 40.0
        with contextlib.redirect_stdout(io.StringIO()):
            mask = detect_streaks(data, rms, existing_mask, dict(_shipped_streak_config()))
        self.assertGreater(int(mask[195:199, 20:380].sum()), 0, "houghpeaks must still find a real trail")

    def test_a_quiet_frame_still_produces_nothing(self):
        data, rms, existing_mask = _quiet_scene()
        with contextlib.redirect_stdout(io.StringIO()):
            mask = detect_streaks(data, rms, existing_mask, dict(_shipped_streak_config()))
        self.assertEqual(int(np.count_nonzero(mask)), 0)


class TestSurvivingStagesEarnTheirCost(unittest.TestCase):
    """The complement of the tests above.

    Deleting a stage needs evidence, and so does keeping one that costs more than
    it appears to. ``contours`` accepted on 0 of 224 real amps and is 44% of the
    remaining stage -- the same two signals that condemned the rescue. Measured on
    injected trails it is the opposite: it adds recall in 2 of 8 cells and costs
    recall in none. This pins that asymmetry so a future retune cannot quietly
    turn contours into dead weight.

    Full grid: ``pixi run streak-recall-floor`` and the same with
    ``--disable contours``.
    """

    #: 1013719p:9 is where the whole difference lands: at seed 0 and 6 sigma,
    #: houghpeaks reaches 0.000 recall and contours reaches 1.000.
    CLEAN_AMP = ("1013719p", 9)
    #: Both were measured; only 6 sigma is asserted, so this stays a claim about
    #: contours being the sole finder rather than about a specific recall value.
    SIGMA = 6.0
    SEED = 0
    LENGTH = 800

    @classmethod
    def setUpClass(cls):
        root = REPO / "test_outputs" / "perf"
        with_ = root / "recall_floor_dashed.json"
        without = root / "recall_no_contours.json"
        if not (with_.exists() and without.exists()):
            raise unittest.SkipTest(
                "run: pixi run streak-recall-floor && pixi run streak-recall-floor -- --disable contours"
            )
        cls.on = json.load(open(with_))
        cls.off = json.load(open(without))

    def _cell(self, records):
        """Best recall for this amp/length/sigma/seed across trail kinds.

        Filters on every key including ``seed``: two seeds place the trail
        differently and contours recovers only one of them, so a lookup that ignored
        seed would report whichever row came last.
        """
        values = [
            row["recall_line"]
            for record in records
            if (record["exposure"], record["hdu"]) == self.CLEAN_AMP
            for row in record["rows"]
            if row["length"] == self.LENGTH and row["sigma"] == self.SIGMA and row["seed"] == self.SEED
        ]
        if not values:
            self.fail(f"no cell for {self.CLEAN_AMP} len={self.LENGTH} sigma={self.SIGMA} seed={self.SEED}")
        return max(values)

    def test_contours_is_the_only_stage_that_finds_a_trail_houghpeaks_misses(self):
        self.assertEqual(self._cell(self.off), 0.0, "houghpeaks alone is expected to miss this cell entirely")
        self.assertGreater(
            self._cell(self.on), 0.5, "contours is the sole finder here; if this fails, contours is dead weight"
        )

    def test_no_real_amp_is_detected_by_the_contour_stage(self):
        """Why the real-amp sweep alone would have got this deletion wrong."""
        sweep = REPO / "test_outputs" / "perf" / "streak_stage_sweep.json"
        if not sweep.exists():
            raise unittest.SkipTest("run: pixi run streak-sweep")
        records = [r for r in json.load(open(sweep)) if not r["error"]]
        self.assertTrue(records, "no successful real-amp records in the sweep")
        accepted = [
            r
            for r in records
            if r["stages"].get("_detect_streaks_contours", {}).get("accepted")
            and any(n > 0 for n in r["stages"]["_detect_streaks_contours"]["accepted"])
        ]
        self.assertEqual(accepted, [], "contours accepted nothing; that is why the injected-trail grid decides this")


class TestRansacEarnsItsCost(unittest.TestCase):
    """Historical sparse RANSAC evidence, before unconditional residual detection.

    The archived real-amp sweep measured 0 acceptances on 222 amps and 0.82 s/amp.

    Same shape of question as contours, same instrument. It is the only finder on
    1500 px dashed trails at 12 sigma, and it costs recall on none of the 16
    cells. The grid must inject *dashed* trails: a solid-only grid measures RANSAC
    on inputs it was never built for and returns a clean +0.000 for the wrong
    reason. These artifacts record the earlier qualification, not current cost.

    Full grid: ``pixi run streak-recall-floor`` and the same with
    ``--disable ransac``.
    """

    #: RANSAC uniquely recovers long dashed trails at high sigma. Which *seed*
    #: it works for depends on where the trail lands, so this names the cell and
    #: not a particular row.
    DASHED_CELL = (1500, 12.0, True)

    @classmethod
    def setUpClass(cls):
        root = REPO / "test_outputs" / "perf"
        with_ = root / "recall_floor_dashed.json"
        off = root / "recall_no_ransac.json"
        if not (with_.exists() and off.exists()):
            raise unittest.SkipTest(
                "run: pixi run streak-recall-floor && pixi run streak-recall-floor -- --disable ransac"
            )
        cls.on = json.load(open(with_))
        cls.off = json.load(open(off))

    @staticmethod
    def _rows(records):
        """Key on exposure, HDU and seed as well as cell.

        Two seeds place the trail differently and RANSAC recovers only one of them,
        so a key without ``seed`` silently lets the last row overwrite the first --
        which is how an earlier version of this test asserted on a 0.000 recall that
        belonged to a different placement than the one being claimed.
        """
        return {
            (r["length"], r["sigma"], r["dashed"], r["seed"], rec["exposure"], rec["hdu"]): r["recall_line"]
            for rec in records
            for r in rec["rows"]
        }

    def _matching(self, rows):
        return {k: v for k, v in rows.items() if (k[0], k[1], k[2]) == self.DASHED_CELL}

    def test_ransac_is_the_sole_finder_of_a_long_dashed_trail(self):
        """Some placement of a long dashed trail is found only with RANSAC."""
        on, off = self._matching(self._rows(self.on)), self._matching(self._rows(self.off))
        self.assertTrue(on, "no dashed cell at this length/sigma in the grid")
        self.assertEqual(set(on), set(off), "stage-on and stage-off artifacts must contain the same cells")
        self.assertTrue(
            any(v > 0.5 and off[k] == 0.0 for k, v in on.items()),
            "expected at least one placement of a 1500px dashed 12-sigma trail that "
            f"RANSAC finds and nothing else does; got {on}",
        )

    def test_ransac_never_costs_recall_anywhere(self):
        on, off = self._rows(self.on), self._rows(self.off)
        self.assertTrue(on, "no injected trail cells in the artifacts")
        self.assertEqual(set(on), set(off), "stage-on and stage-off artifacts must contain the same cells")
        hurt = {k: (on[k], off[k]) for k in on if on[k] < off[k] - 0.02}
        self.assertEqual(hurt, {}, "RANSAC costs recall on these cells")

    def test_the_grid_really_does_inject_dashed_trails(self):
        """Without this, a solid-only grid narrows the check silently.

        The dashed cell above would simply not exist, and ``test_ransac_never_costs_
        recall_anywhere`` would go on passing over solid trails alone -- where
        RANSAC also helps, so nothing would look wrong.
        """
        kinds = {r["dashed"] for rec in self.on for r in rec["rows"]}
        self.assertIn(True, kinds, "the grid must inject dashed trails")

    def test_the_grid_cannot_price_ransac_and_the_suite_says_so(self):
        """The historical grid predates unconditional residual detection.

        Its primary-acceptance shortcut skipped RANSAC in 94 of 96 injected
        runs, so its timing cannot establish the current detector's cost.
        """
        on, off = self._rows(self.on), self._rows(self.off)
        self.assertTrue(on, "no injected trail cells in the artifacts")
        self.assertEqual(set(on), set(off), "stage-on and stage-off artifacts must contain the same cells")
        differing = sum(1 for k in on if on[k] != off[k])
        total = len(on)
        self.assertLess(
            differing * 4,
            total,
            "RANSAC now changes recall in a quarter of injected runs; the gate measured here "
            "is not the one that shipped, and 0.82 s/amp needs re-deriving",
        )


if __name__ == "__main__":
    unittest.main()
