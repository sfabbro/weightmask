import unittest

from tests.benchmarks.download_data import validate_case_file
from tests.benchmarks.run import ROOT, load_manifest, run_suite


class TestBenchmarks(unittest.TestCase):
    def test_load_manifest(self):
        manifest = load_manifest("megacam_real")
        self.assertEqual(manifest["suite"], "megacam_real")
        self.assertTrue(len(manifest["cases"]) > 0)

    def test_run_synthetic_v2_smoke(self):
        summary = run_suite("synthetic_v2", with_baselines=False, selected_cases={"synthetic_sparse"})
        self.assertEqual(summary["suite"], "synthetic_v2")
        self.assertIn("synthetic_sparse", summary["results"])
        self.assertIn("bad_pixel_stats", summary["results"]["synthetic_sparse"])

    def test_validate_case_file_rejects_wrong_instrument(self):
        manifest = load_manifest("megacam_real")
        sparse_case = next(case for case in manifest["cases"] if case["case_id"] == "megacam_sparse_control")
        local_path = ROOT / sparse_case["local_path"]
        if local_path.exists():
            valid, reason = validate_case_file(sparse_case, local_path)
            self.assertFalse(valid)
            self.assertIn("MegaCam", reason)

    def test_run_real_suite_reports_concrete_status(self):
        summary = run_suite("acs_compare", with_baselines=False)
        self.assertEqual(summary["suite"], "acs_compare")
        self.assertTrue(
            all(
                result["status"] in {"missing_data", "loaded", "invalid_instrument"}
                for result in summary["results"].values()
            )
        )


if __name__ == "__main__":
    unittest.main()
