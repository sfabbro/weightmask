"""Regression tests for the MegaCam perf-report aggregation.

The aggregator is pure, so these run in milliseconds and need no CCD data. The
resumed-exposure case is the important one: a fully-resumed run used to divide
this process's wall clock by the pixel count of *all* exposures, and reported
6.4e7 Mpix/s against a true 0.073.
"""

import importlib.util
import pathlib
import unittest

_PERF = pathlib.Path(__file__).resolve().parents[1] / "benchmarks" / "perf_megacam.py"
_spec = importlib.util.spec_from_file_location("perf_megacam", _PERF)
perf_megacam = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(perf_megacam)
_aggregate_report = perf_megacam._aggregate_report


def _exposure(safe_id, wall_s, hdu_total, mpix, *, resumed=False):
    rec = {
        "safe_id": safe_id,
        "wall_s": wall_s,
        "nhdus_processed": 36,
        "mpix": mpix,
        "stage_totals": {"hdu_total": hdu_total},
    }
    if resumed:
        rec["resumed"] = True
    return rec


class TestAggregateReport(unittest.TestCase):
    def test_unresumed_run_reports_throughput(self):
        per_exp = [_exposure("a", 100.0, 99.0, 353.0), _exposure("b", 200.0, 198.0, 353.0)]
        report = _aggregate_report(per_exp, 300.0)

        self.assertAlmostEqual(report["mpix_per_s"], 706.0 / 300.0, places=9)
        self.assertEqual(report["n_resumed_exposures"], 0)
        self.assertEqual(report["warnings"], [])

    def test_fully_resumed_run_refuses_to_report_throughput(self):
        # total_wall times only the checkpoint walk, not the exposures it replays.
        per_exp = [
            _exposure("a", 2250.0, 2310.0, 353.0, resumed=True),
            _exposure("b", 2300.0, 2360.0, 353.0, resumed=True),
        ]
        report = _aggregate_report(per_exp, 5.479e-05)

        self.assertIsNone(report["mpix_per_s"])
        self.assertEqual(report["n_resumed_exposures"], 2)
        self.assertTrue(any("resumed from checkpoint" in w for w in report["warnings"]))
        # Stage totals still describe the resumed work and must survive.
        self.assertAlmostEqual(report["hdu_wall_s"], 4670.0)
        self.assertAlmostEqual(report["mpix_total"], 706.0)

    def test_partially_resumed_run_refuses_to_report_throughput(self):
        per_exp = [_exposure("a", 100.0, 99.0, 353.0, resumed=True), _exposure("b", 200.0, 198.0, 353.0)]
        report = _aggregate_report(per_exp, 200.0)

        self.assertIsNone(report["mpix_per_s"])
        self.assertEqual(report["n_resumed_exposures"], 1)

    def test_warns_when_stage_totals_disagree_with_process_wall(self):
        # Sequential workers=1: per-HDU stage time should track the process wall.
        per_exp = [_exposure("a", 100.0, 900.0, 353.0)]
        report = _aggregate_report(per_exp, 100.0)

        self.assertTrue(any("disagree" in w for w in report["warnings"]))
        self.assertIsNotNone(report["mpix_per_s"])

    def test_zero_wall_does_not_divide(self):
        report = _aggregate_report([_exposure("a", 0.0, 10.0, 353.0)], 0.0)
        self.assertIsNone(report["mpix_per_s"])


if __name__ == "__main__":
    unittest.main()
