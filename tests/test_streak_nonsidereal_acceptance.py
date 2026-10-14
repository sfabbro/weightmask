"""The non-sidereal acceptance harness must be runnable, and its two
mask-reading helpers must classify a defect the way the audit does.

The harness's whole point is a number (the control's footprint) that is
currently non-zero on a 1024-pixel frame. These tests pin the harness, not that
number: the 1024-pixel baseline is recorded in
``.cursor/harness/streak_nonsidereal_remediation_2026-10-14.md`` and belongs to
the remediation work, not to the suite's runtime budget.
"""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from benchmarks.streak_nonsidereal_acceptance import (
    attribute,
    band,
    main,
    render_trail,
    summarize_components,
)


class TestNonSiderealAcceptanceHarness(unittest.TestCase):
    def test_control_is_empty_and_negatives_are_scored_on_a_small_frame(self):
        """A 256-pixel frame with no injected satellite must yield no STREAK
        pixels at all. This is the property the campaign failed on full frames;
        if it ever fails here first, the harness or the stage changed shape."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "report.json"
            self.assertEqual(main(["--quick", "--size", "256", "--stars", "8", "--seed", "3", "--out", str(out)]), 0)
            report = json.loads(out.read_text())
        self.assertEqual(report["control_px"], 0, "no satellite was injected, so nothing should be flagged")
        self.assertEqual(report["control_attribution"]["total"], report["control_px"])
        positives = [row for row in report["rows"] if row["kind"] == "satellite"]
        self.assertTrue(positives, "the harness must score the injected trail cells")
        self.assertTrue(
            any(row["kind"].startswith("row-band") for row in report["rows"]),
            "the detector bands must be scored as negatives",
        )
        self.assertTrue(all("body_coverage" in row for row in report["rows"]))

    def test_a_full_span_narrow_component_is_axis_aligned(self):
        mask = np.zeros((512, 400), dtype=bool)
        mask[200:204, :] = True
        mask[10:60, 300:340] = True
        mask[400:460, 100:112] = True  # diagonal-ish blob, neither axis
        components = summarize_components(mask)
        by_px = {item["px"]: item for item in components}
        band_component = by_px[4 * 400]
        self.assertEqual((band_component["y_span"], band_component["x_span"]), (4, 400))
        self.assertTrue(band_component["axis_aligned"], "four rows across every column is a detector band")
        self.assertFalse(by_px[50 * 40]["axis_aligned"], "a 50x40 blob is not a defect line")
        self.assertFalse(by_px[60 * 12]["axis_aligned"])

    def test_attribution_accounts_for_every_pixel_once(self):
        shape = (200, 200)
        rms = np.ones(shape, dtype=np.float32)
        _flux, band_body = band(shape, axis="row", start=50, width=4, amplitude=2.0, rms_map=rms)
        _flux, trail_body = render_trail(
            shape, angle_deg=40.0, length=60.0, cx=100.0, cy=120.0, peak_sig=10.0, rms_map=rms
        )
        truth = {
            "star_trails": [{"body": trail_body}],
            "columns": [],
            "rows": [{"body": band_body}],
        }
        mask = band_body | trail_body
        counts = attribute(mask, truth)
        self.assertEqual(counts["total"], int(mask.sum()))
        self.assertEqual(counts["total"], counts["rows"] + counts["star_trails"] + counts["unexplained"])
        self.assertGreater(
            counts["rows"], counts["star_trails"], "a 4-row full-width band has more pixels than a 60-px trail"
        )
        # A pixel on the band must be attributed to it, not to the dilated trail.
        self.assertTrue(counts["rows"] >= int(band_body.sum()))
        empty = attribute(mask, {"star_trails": [], "columns": [], "rows": []})
        self.assertEqual(empty["unexplained"], int(mask.sum()))


if __name__ == "__main__":
    unittest.main()
