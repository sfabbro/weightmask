"""The MegaCam science gate skips when its data directory is empty.

A skip is not a pass, and this test does not run the MegaCam benches.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import fitsio
import numpy as np
import pytest

from benchmarks import science_gate
from benchmarks import score_trail_truth as scorer
from benchmarks.science_gate import main


class TestScienceGateSkip(unittest.TestCase):
    def test_missing_fits_is_a_skip(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "science-gate.md")
            code = main(["--data-dir", tmp, "--out", out])
            self.assertEqual(code, 0)
            text = open(out).read()
        self.assertIn("status: skipped", text)
        self.assertIn("not a pass", text)
        self.assertNotIn("status: passed", text)

    def test_missing_fits_emits_release_evidence_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = os.path.join(tmp, "science-gate.md")
            manifest = os.path.join(tmp, "science-gate.json")
            code = main(["--data-dir", tmp, "--out", out, "--manifest", manifest, "--version", "0.2.1"])
            self.assertEqual(code, 0)
            evidence = json.loads(open(manifest).read())
        self.assertEqual(evidence["status"], "skipped")
        self.assertEqual(evidence["version"], "0.2.1")
        self.assertIn("commit_sha", evidence)
        self.assertIn("config_sha", evidence)
        self.assertIn("input_manifest_sha", evidence)
        self.assertIn("metric_revisions", evidence)
        self.assertIn("command", evidence)
        self.assertIn("timestamp", evidence)
        self.assertIn("data_ids", evidence)
        self.assertEqual(evidence["real_trail_recall"], "n/a")
        self.assertEqual(evidence["scope_exclusions"], ["real_trail_recall"])

    def test_injection_rejects_empty_malformed_or_nonfinite_metrics(self):
        header = "dashed\trecall\trecall5\n"
        for body in ("", "false\tnan\t1\n", "false\t1\tinf\n", "false\t1\n", "false\tbad\t1\n", "false\t2\t1\n"):
            with self.subTest(body=body), tempfile.TemporaryDirectory() as tmp:
                with open(os.path.join(tmp, "streak_inject.tsv"), "w") as handle:
                    handle.write(header + body)
                failures = []
                with patch.object(science_gate, "_run", return_value=(0, "")):
                    science_gate._inject("fixture.fits", 1, tmp, failures, [])
                self.assertTrue(failures)

    def test_cosmics_rejects_nonfinite_or_malformed_metrics(self):
        for body in ("main+faint\tnan\t1\n", "main+faint\t1\tbad\n", "main+faint\t1\n"):
            with self.subTest(body=body), tempfile.TemporaryDirectory() as tmp:
                with open(os.path.join(tmp, "cr_faint.tsv"), "w") as handle:
                    handle.write("variant\tworm_recall\tsingle_recall\n" + body)
                failures = []
                with patch.object(science_gate, "_run", return_value=(0, "")):
                    science_gate._cosmics("fixture.fits", 1, tmp, failures, [])
                self.assertTrue(failures)


@pytest.mark.parametrize("relocate", [False, True])
def test_gate_uses_fixture_paths_unless_data_directory_is_explicit(tmp_path, relocate):
    class LocalPool:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def map(self, fn, jobs, **kwargs):
            return map(fn, jobs)

    data_dir = tmp_path / science_gate.DEFAULT_DATA
    data_dir.mkdir(parents=True)
    data = np.ones((8, 9), np.float32)
    fitsio.write(str(data_dir / "1013719p.fits"), data)
    original = data_dir.parent / "megacam_streak_case.fits"
    actual = data_dir / original.name if relocate else original
    fitsio.write(str(actual), data)
    fixture = {
        "schema": "small-test",
        "data_requirements": {"data_root": str(data_dir), "exposures": {"case": {"path": str(original)}}},
        "entries": [
            {
                "exposure": "case",
                "hdu": 0,
                "ccdname": "A",
                "label": "artefact",
                "geometry": {"point": [0, 4], "direction": [1, 0], "offset_px": 4},
            }
        ],
        "summary": {"by_label": {"trail": 0, "artefact": 1}},
    }
    fixture_path = tmp_path / "labels.json"
    fixture_path.write_text(json.dumps(fixture))
    config = tmp_path / "config.yml"
    config.write_text("{}\n")
    out = tmp_path / "output/science-gate.md"

    def run_score(command):
        return scorer.main(["--fixture", str(fixture_path), "--config", str(config), *command[2:]]), ""

    with (
        patch.object(science_gate, "ROOT", str(tmp_path)),
        patch.object(science_gate, "_run", side_effect=run_score),
        patch.object(science_gate, "_inject"),
        patch.object(science_gate, "_cosmics"),
        patch.object(science_gate, "_perf"),
        patch.object(scorer, "ProcessPoolExecutor", LocalPool),
        patch.object(scorer, "capture_detector_inputs", return_value=(data, data, np.zeros(data.shape, bool))),
        patch.object(scorer, "run_detector", return_value=np.zeros(data.shape, bool)),
    ):
        assert main(["--out", str(out), *(["--data-dir", str(data_dir)] if relocate else [])]) == 0
    manifest = tmp_path / "test_outputs" / "harness" / "science-gate.json"
    assert manifest.is_file()
    assert json.loads(manifest.read_text())["scope_exclusions"] == ["real_trail_recall"]
    report = json.loads((out.parent / "trail_truth.score.json").read_text())
    assert report["totals"]["streaks"]["artefact_entries"] == 1
    assert not report["skipped"]


if __name__ == "__main__":
    unittest.main()
