import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from astropy.io import fits

from tests.benchmarks.download_data import validate_case_file
from tests.benchmarks.run import (
    ROOT,
    _load_case_label,
    _load_quality_reference,
    _quality_gate_failures,
    _quality_validation_metrics,
    load_manifest,
    main,
    run_suite,
)
from weightmask.contract import INVERSE_VARIANCE_SEMANTICS, MASK_POLARITY


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
            bad_case = sparse_case.copy()
            bad_case["expected_detector"] = "WrongDetector"
            valid, reason = validate_case_file(bad_case, local_path)
            self.assertFalse(valid)
            self.assertIn("WrongDetector", reason)

    def test_run_real_suite_reports_concrete_status(self):
        summary = run_suite("acs_compare", with_baselines=False)
        self.assertEqual(summary["suite"], "acs_compare")
        self.assertTrue(
            all(
                result["status"]
                in {
                    "missing_data",
                    "missing_labels",
                    "invalid_labels",
                    "invalid_label_provenance",
                    "invalid_science_provenance",
                    "loaded",
                    "invalid_instrument",
                    "invalid_quality_reference",
                }
                for result in summary["results"].values()
            )
        )

    def test_real_suite_and_cli_fail_closed_without_inputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            with (
                patch("tests.benchmarks.run.ROOT", root),
                patch("tests.benchmarks.run.OUTPUT_ROOT", root / "outputs"),
            ):
                summary = run_suite("acs_compare", with_baselines=False)
                self.assertEqual(len(summary["gate_failures"]), 2)
                self.assertTrue(all(result["status"] == "missing_data" for result in summary["results"].values()))
                self.assertEqual(main(["--suite", "acs_compare"]), 1)

    def test_acs_err_and_dq_planes_preserve_native_semantics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "acs.fits"
            primary = fits.PrimaryHDU()
            science = fits.ImageHDU(np.ones((4, 5), dtype=np.float32), name="SCI")
            error = fits.ImageHDU(np.full((4, 5), 2.0, dtype=np.float32), name="ERR")
            dq_values = np.zeros((4, 5), dtype=np.uint16)
            dq_values[1, 2] = 4
            dq = fits.ImageHDU(dq_values, name="DQ")
            for hdu in (science, error, dq):
                hdu.header["EXTVER"] = 2
            fits.HDUList([primary, science, error, dq]).writeto(path)

            case = {"quality_reference": {"inverse_variance": "ERR", "quality_mask": "DQ"}}
            crop = {"y0": 0, "x0": 0, "height": 4, "width": 5}
            inverse_variance, quality_mask, reason = _load_quality_reference(case, path, science.header, crop)

            self.assertIsNone(reason)
            np.testing.assert_allclose(inverse_variance, 0.25)
            self.assertEqual(np.count_nonzero(quality_mask), 1)
            self.assertTrue(quality_mask[1, 2])

    def test_manual_label_requires_exact_hash_full_frame_and_binary_values(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            label_path = root / "label.fits"
            label = np.zeros((6, 8), dtype=np.uint8)
            label[2, 3:6] = 1
            fits.PrimaryHDU(label).writeto(label_path)
            digest = hashlib.sha256(label_path.read_bytes()).hexdigest()
            case = {
                "label_artifact": {
                    "path": "label.fits",
                    "sha256": digest,
                    "science_sha256": "a" * 64,
                    "coordinate_frame": "science_full_frame",
                    "polarity": "one_means_trail",
                }
            }
            crop = {"y0": 1, "x0": 2, "height": 4, "width": 5}
            with patch("tests.benchmarks.run.ROOT", root):
                loaded, evidence, status, reason = _load_case_label(case, label.shape, crop)
                self.assertIsNone(status)
                self.assertIsNone(reason)
                self.assertEqual(np.count_nonzero(loaded), 3)
                self.assertEqual(evidence["sha256"], digest)

                case["label_artifact"]["sha256"] = "0" * 64
                loaded, evidence, status, reason = _load_case_label(case, label.shape, crop)
                self.assertIsNone(loaded)
                self.assertIsNone(evidence)
                self.assertEqual(status, "invalid_label_provenance")
                self.assertIn("SHA-256 mismatch", reason)

                case["label_artifact"]["sha256"] = digest
                loaded, evidence, status, reason = _load_case_label(case, (3, 4), crop)
                self.assertIsNone(loaded)
                self.assertEqual(status, "invalid_labels")
                self.assertIn("does not match science shape", reason)

                label[0, 0] = 2
                fits.PrimaryHDU(label).writeto(label_path, overwrite=True)
                case["label_artifact"]["sha256"] = hashlib.sha256(label_path.read_bytes()).hexdigest()
                loaded, evidence, status, reason = _load_case_label(case, label.shape, crop)
                self.assertIsNone(loaded)
                self.assertEqual(status, "invalid_labels")
                self.assertIn("finite binary 0/1", reason)

    def test_quality_metrics_cover_w1_acceptance_semantics(self):
        rng = np.random.default_rng(12)
        data = rng.normal(0.0, 1.0, (64, 64)).astype(np.float32)
        data[10, 10] = 20.0
        prediction = np.zeros(data.shape, dtype=bool)
        prediction[10, 10] = True
        truth = np.zeros(data.shape, dtype=bool)
        truth[30, 8:56] = True
        native_quality = np.zeros(data.shape, dtype=bool)
        native_quality[4, 4] = True
        inverse_variance = np.ones(data.shape, dtype=np.float32)
        inverse_variance[0, 0] = 0.0

        metrics = _quality_validation_metrics(data, prediction, truth, inverse_variance, native_quality)

        self.assertEqual(metrics["mask_polarity"], MASK_POLARITY)
        self.assertEqual(metrics["inverse_variance_semantics"], INVERSE_VARIANCE_SEMANTICS)
        self.assertTrue(metrics["flagged_pixels_have_zero_weight"])
        self.assertTrue(metrics["clean_weight_matches_inverse_variance"])
        self.assertTrue(metrics["invalid_variance_is_flagged"])
        self.assertAlmostEqual(metrics["overmask_fraction"], 0.0)
        self.assertGreater(metrics["flux_bias_fraction"], 0.9)
        self.assertGreater(metrics["noise_scale"], 0.8)
        self.assertLess(metrics["noise_scale"], 1.2)

        failures = _quality_gate_failures(
            "fixture",
            metrics,
            {
                "max_overmask_fraction": 0.05,
                "max_flux_bias_fraction": 0.02,
                "noise_scale_min": 0.8,
                "noise_scale_max": 1.2,
            },
        )
        self.assertTrue(any("flux_bias_fraction" in failure for failure in failures))


if __name__ == "__main__":
    unittest.main()
