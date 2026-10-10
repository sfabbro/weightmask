"""Exercise CANFAR orchestration locally, without submissions or benchmarks."""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "benchmarks" / "canfar"


class TestCanfarScripts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.bin = self.work / "bin"
        self.bin.mkdir()
        (self.bin / "python3").symlink_to(sys.executable)
        self.env = dict(
            os.environ,
            PATH=f"{self.bin}:{os.environ['PATH']}",
            PROJECT_MOUNT=str(self.work),
            LOG=str(self.work / "calls"),
        )

    def executable(self, name, body):
        path = self.bin / name
        path.write_text("#!/bin/bash\n" + body)
        path.chmod(0o755)

    def test_thread_twins_use_the_manifest_settings(self):
        self.executable("pixi", "echo 'pixi 0.44.0'\n")
        prefix = (SCRIPTS / "run_one.sh").read_text().split('git clone "$REPO_URL"')[0]
        # Keep the original location so BOOTSTRAP resolves the real manifest.
        prefix = prefix.replace('BOOTSTRAP="$(cd "$(dirname "$0")/../.." && pwd)"', f'BOOTSTRAP="{ROOT}"')
        prefix = prefix.replace(
            'JOB_DIR="$SCR_BASE/wm-$EXP_ID-$JOB_TAG-$$"', 'JOB_DIR="$PROJECT_MOUNT/wm-$EXP_ID-$JOB_TAG-$$"'
        )
        for tag, expected in (("unset", ":"), ("pinned", "1:1")):
            manifest = json.loads((ROOT / "benchmarks/canfar_experiments/manifest.json").read_text())
            job = next(
                j
                for g in manifest["groups"]
                if g["exp_id"] == "E5"
                for j in g["jobs"]
                if (bool(j.get("env")) == (tag == "pinned"))
            )
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    prefix + '\nprintf "THREADS=%s:%s\\n" "${OMP_NUM_THREADS-}" "${MKL_NUM_THREADS-}"',
                    "run_one.sh",
                    "E5",
                    job["tag"],
                ],
                env=dict(
                    self.env,
                    MANIFEST_SHA=manifest["code"]["pinned_sha"],
                    MANIFEST_FILE_SHA=hashlib.sha256(
                        (ROOT / "benchmarks/canfar_experiments/manifest.json").read_bytes()
                    ).hexdigest(),
                    OMP_NUM_THREADS="9",
                    MKL_NUM_THREADS="9",
                ),
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("THREADS=" + expected, result.stdout)

    def test_waits_for_every_job_and_no_wait_skips_polling(self):
        self.executable(
            "canfar",
            """echo "$*" >> "$LOG"
case "$1" in
  config) ;;
  create) echo "ID: $(printf '%s' "$*" | sed -n 's/.*--name \\([^ ]*\\).*/\\1/p' | tr -d '-')";;
  info) echo 'Status: Completed';;
esac
""",
        )
        for no_wait in (False, True):
            Path(self.env["LOG"]).unlink(missing_ok=True)
            result = subprocess.run(
                ["bash", str(SCRIPTS / "submit_all.sh"), "e0"] + (["--no-wait"] if no_wait else []),
                env=self.env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            calls = Path(self.env["LOG"]).read_text().splitlines()
            self.assertEqual(sum(line.startswith("create ") for line in calls), 2)
            self.assertEqual(sum(line.startswith("info ") for line in calls), 0 if no_wait else 2)

    def test_setup_dry_run_keeps_current_runner_separate_from_science_checkout(self):
        self.executable(
            "canfar",
            """echo "$*" >> "$LOG"
case "$1" in
  create)
    case "$*" in
      *"git clone"*)
        if [ -f "$LOG.state" ]; then exit 1; fi
        printf '%s' stale > "$LOG.state";;
      *"git fetch origin"*) printf '%s' fetched > "$LOG.state";;
      *"checkout -B runner origin/main"*)
        test "$(cat "$LOG.state")" = fetched
        printf '%s' runner-origin-main > "$LOG.state";;
      *"rev-parse HEAD"*) test "$(cat "$LOG.state")" = runner-origin-main;;
    esac
    echo "ID: setup$(date +%s%N)";;
  info) echo 'Status: Completed';;
esac
""",
        )
        bootstrap = self.work / "bootstrap"
        setup_env = dict(self.env, PROJECT_MOUNT=str(self.work), BOOTSTRAP=str(bootstrap))
        for _ in range(2):
            result = subprocess.run(
                ["bash", str(SCRIPTS / "submit_all.sh"), "setup"],
                env=setup_env,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
        calls = Path(self.env["LOG"]).read_text().splitlines()
        creates = [line for line in calls if line.startswith("create ")]
        self.assertIn("git clone https://github.com/astroai/weightmask.git", creates[0])
        self.assertIn("git -C " + str(bootstrap) + " checkout -B runner origin/main", creates[2])
        self.assertIn("git -C " + str(bootstrap) + " grep -F wm27-provenance-v2", creates[4])
        self.assertIn(
            "git -C "
            + str(bootstrap)
            + " ls-files --error-unmatch benchmarks/canfar/run_one.sh benchmarks/canfar/manifest.py",
            creates[5],
        )
        self.assertNotIn("cc1d7caf", "\n".join(creates))
        self.assertEqual((Path(self.env["LOG"] + ".state")).read_text(), "runner-origin-main")
        self.assertIn("bootstrap exists; reusing", result.stdout)

    def test_mask_identity_covers_all_hdus_and_matches_exposures(self):
        repo = self.work / "repo"
        perf = repo / "test_outputs/perf"
        perf.mkdir(parents=True)
        job = self.work / "job"
        job.mkdir()
        result = self.work / "results/E5-pinned"
        result.mkdir(parents=True)
        control = result.parent / "E0-w8"
        control.mkdir()
        (job / "group.json").write_text(
            json.dumps(
                {
                    "job": {"workers": 1},
                    "group": {"safe_ids": ["a", "b"]},
                    "manifest_sha256": "m" * 64,
                    "manifest_status": "historical",
                    "code_pinned_sha": "c" * 40,
                }
            )
        )
        (perf / "megacam_perf_E5-pinned.json").write_text(json.dumps({"per_exposure": [{"wall_s": 1, "cpu_s": 1}]}))
        zero = np.zeros((3, 4), np.uint8)
        for exposure in ("a", "b"):
            changed = zero.copy()
            changed[0, 0] = 1
            fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(zero), fits.ImageHDU(changed)]).writeto(
                perf / f"{exposure}.w1.mask.fits"
            )
            fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(zero), fits.ImageHDU(zero)]).writeto(
                control / f"{exposure}.w8.mask.fits"
            )
        source = (SCRIPTS / "run_one.sh").read_text()
        block = (
            source.split('"$PIXI" run python - "$JOB_DIR" "$RESULTS_DIR"')[1]
            .split("<<'EOF'\n", 1)[1]
            .split("\nEOF", 1)[0]
        )
        run = subprocess.run(
            [sys.executable, "-c", block, str(job), str(result), "E5", "pinned", "0"],
            cwd=repo,
            env=dict(
                self.env,
                CHECKOUT_REF="fixture-ref",
                CHECKOUT_SHA="c" * 40,
                RUN_ONE_VERSION="wm27-provenance-v2",
            ),
            capture_output=True,
            text=True,
        )
        self.assertEqual(run.returncode, 0, run.stderr)
        metrics = json.loads((result / "metrics.json").read_text())
        self.assertEqual(metrics["mask_diff"]["gained"], 2)
        self.assertFalse(metrics["mask_diff"]["identical"])
        self.assertEqual(set(metrics["mask_checksums"]), {"a", "b"})
        self.assertEqual(metrics["mask_checksum_scope"], "all-image-hdus-v1")
        self.assertEqual(metrics["manifest_sha256"], "m" * 64)
        self.assertEqual(metrics["manifest_status"], "historical")
        self.assertEqual(metrics["code_pinned_sha"], "c" * 40)
        self.assertEqual(metrics["run_one_version"], "wm27-provenance-v2")
        self.assertEqual(metrics["checkout_ref"], "fixture-ref")
        self.assertEqual(metrics["checkout_sha"], "c" * 40)

    def test_aggregate_does_not_claim_legacy_or_swapped_identity(self):
        spec = importlib.util.spec_from_file_location("aggregate", SCRIPTS / "aggregate.py")
        aggregate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(aggregate)
        for scope in (None, "all-image-hdus-v1"):
            for tag, checksums in (("pinned", {"a": "aaaa", "b": "bbbb"}), ("unset", {"a": "bbbb", "b": "aaaa"})):
                path = self.work / ("E5-" + tag)
                path.mkdir(exist_ok=True)
                metrics = {"mask_checksums": checksums}
                if scope:
                    metrics["mask_checksum_scope"] = scope
                (path / "metrics.json").write_text(json.dumps(metrics))
            aggregate.main([str(self.work)])
            self.assertNotIn("mask-identical: True", (self.work / "table.md").read_text())

    def test_partial_download_is_not_a_successful_cached_variant(self):
        from unittest.mock import patch

        source = (SCRIPTS / "run_one.sh").read_text()
        fetch_source = source.split("def fetch(url, out):", 1)[1].split("\nPUB = ", 1)[0]
        scope = {}
        exec("import os, urllib.request\ndef fetch(url, out):" + fetch_source, scope)
        destination = self.work / "flat.fits"

        def interrupted(url, path):
            Path(path).write_bytes(b"partial")
            raise OSError("interrupted")

        with patch("urllib.request.urlretrieve", side_effect=interrupted):
            with self.assertRaises(OSError):
                scope["fetch"]("fixture", str(destination))
        self.assertFalse(destination.exists())

    def test_multi_flat_group_fails_before_running(self):
        source = (SCRIPTS / "run_one.sh").read_text()
        block = (
            source.split('python3 - "$MANIFEST" "$EXP_ID" "$JOB_TAG" "$GROUP_JSON"', 1)[1]
            .split("<<'EOF'\n", 1)[1]
            .split("\nEOF", 1)[0]
        )
        manifest = ROOT / "benchmarks/canfar_experiments/manifest.json"
        result = subprocess.run(
            [sys.executable, "-c", block, str(manifest), "E7", "w8", str(self.work / "group.json")],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("per-exposure mapping", result.stderr)

    def test_manifest_rejects_removed_parameters(self):
        from benchmarks.canfar.manifest import validate_manifest

        manifest = json.loads((ROOT / "benchmarks/canfar_experiments/manifest.json").read_text())
        with pytest.raises(ValueError, match="Unsupported configuration key") as raised:
            validate_manifest(manifest, "E3")
        self.assertIn("mrt_rescue_params", str(raised.value))
        self.assertIn("satdet_params", str(raised.value))

    def test_manifest_accepts_canonical_override_keys(self):
        from benchmarks.canfar.manifest import validate_manifest

        manifest = json.loads((ROOT / "benchmarks/canfar_experiments/manifest.json").read_text())
        manifest["groups"][0]["config_set"] = {
            "cosmic_ray.dilation_radius": 2,
            "variance.method": "rms_map",
        }
        validate_manifest(manifest, "E0")

    def test_manifest_rejects_unresolved_e0_and_multiflat_e7(self):
        from benchmarks.canfar.manifest import validate_manifest

        manifest = json.loads((ROOT / "benchmarks/canfar_experiments/manifest.json").read_text())
        manifest["groups"][0]["safe_ids"] = ["TBD"]
        with pytest.raises(ValueError, match="E0 group has unresolved inputs"):
            validate_manifest(manifest, "E0")
        with pytest.raises(ValueError, match="E7 multiple flats"):
            validate_manifest(manifest, "E7")

    def test_manifest_rejects_paired_id_cardinality_mismatch(self):
        from benchmarks.canfar.manifest import validate_manifest

        manifest = json.loads((ROOT / "benchmarks/canfar_experiments/manifest.json").read_text())
        manifest["groups"][0]["publisherIDs"] = []
        with pytest.raises(ValueError, match="safe_ids/publisherIDs cardinality mismatch"):
            validate_manifest(manifest, "E0")

    def test_download_checks_cardinality_before_pairing_ids(self):
        source = (SCRIPTS / "run_one.sh").read_text()
        self.assertLess(
            source.index("safe_ids/publisherIDs cardinality mismatch before download"),
            source.index('zip(grp["publisherIDs"], grp["safe_ids"])'),
        )

    def test_manifest_rejects_unresolved_e8(self):
        from benchmarks.canfar.manifest import validate_manifest

        manifest = json.loads((ROOT / "benchmarks/canfar_experiments/manifest.json").read_text())
        with pytest.raises(ValueError, match="E8.*unresolved"):
            validate_manifest(manifest, "E8")

    def test_manifest_accepts_resolved_e8(self):
        from benchmarks.canfar.manifest import validate_manifest

        manifest = json.loads((ROOT / "benchmarks/canfar_experiments/manifest.json").read_text())
        group = next(group for group in manifest["groups"] if group["exp_id"] == "E8")
        group.update(
            {
                "safe_ids": ["1030000p"],
                "publisherIDs": ["ivo://cadc.nrc.ca/CFHT?1030000/1030000p"],
                "flat_safe_ids": ["08Bm02.flat.g.36.02"],
                "flat_publisherIDs": ["ivo://cadc.nrc.ca/CFHT?08Bm02.flat.g.36.02/08Bm02.flat.g.36.02"],
                "flat_paths": ["08Bm02.flat.g.36.02.fits.fz"],
            }
        )
        validate_manifest(manifest, "E8")

    def test_manifest_is_pinned_historical_and_hash_is_content_hash(self):
        from benchmarks.canfar.manifest import manifest_sha256

        path = ROOT / "benchmarks/canfar_experiments/manifest.json"
        manifest = json.loads(path.read_text())
        self.assertEqual(manifest["status"], "historical")
        self.assertTrue(manifest["historical"])
        self.assertTrue(manifest["pinned"])
        self.assertEqual(manifest_sha256(path), hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(
            manifest["runner"]["sha256"], hashlib.sha256((SCRIPTS / "run_one.sh").read_bytes()).hexdigest()
        )
        self.assertEqual(
            manifest["runner"]["validator_sha256"], hashlib.sha256((SCRIPTS / "manifest.py").read_bytes()).hexdigest()
        )

    def test_run_one_and_setup_pin_provenance_version(self):
        run_one = (SCRIPTS / "run_one.sh").read_text()
        submit_all = (SCRIPTS / "submit_all.sh").read_text()
        self.assertIn('RUN_ONE_VERSION="wm27-provenance-v2"', run_one)
        self.assertIn("manifest_sha256=", run_one)
        self.assertIn("checkout_sha=", run_one)
        self.assertIn("VALIDATOR_SHA256", run_one)
        self.assertIn("wm27-provenance-v2", submit_all)

    def test_run_one_rejects_code_or_manifest_hash_mismatch(self):
        prefix = (SCRIPTS / "run_one.sh").read_text().split('git clone "$REPO_URL"')[0]
        prefix = prefix.replace('BOOTSTRAP="$(cd "$(dirname "$0")/../.." && pwd)"', f'BOOTSTRAP="{ROOT}"')
        manifest = json.loads((ROOT / "benchmarks/canfar_experiments/manifest.json").read_text())
        for env_updates, expected in (
            ({"MANIFEST_SHA": "wrong"}, "manifest code SHA mismatch"),
            (
                {
                    "MANIFEST_SHA": manifest["code"]["pinned_sha"],
                    "MANIFEST_FILE_SHA": "0" * 64,
                },
                "manifest file SHA mismatch",
            ),
            (
                {
                    "MANIFEST_SHA": manifest["code"]["pinned_sha"],
                    "VALIDATOR_SHA256": "0" * 64,
                },
                "validator SHA mismatch",
            ),
        ):
            result = subprocess.run(
                ["bash", "-c", prefix, "run_one.sh", "E0", "w8"],
                env=dict(self.env, **env_updates),
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(expected, result.stderr)

    def test_submit_rejects_mutated_manifest_before_canfar_create(self):
        self.executable("canfar", 'echo called >> "$LOG"\n')
        path = self.work / "manifest.json"
        manifest = json.loads((ROOT / "benchmarks/canfar_experiments/manifest.json").read_text())
        manifest["groups"][-1]["safe_ids"] = ["TBD"]
        path.write_text(json.dumps(manifest))
        result = subprocess.run(
            ["bash", str(SCRIPTS / "submit_all.sh"), "e8"],
            env=dict(self.env, MANIFEST_PATH=str(path)),
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("E8 group has unresolved inputs", result.stderr + result.stdout)
        self.assertFalse(Path(self.env["LOG"]).exists())

    def test_submit_rejects_historical_removed_parameter_group(self):
        self.executable("canfar", 'echo called >> "$LOG"\n')
        result = subprocess.run(
            ["bash", str(SCRIPTS / "submit_all.sh"), "submit", "E3"],
            env=self.env,
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Unsupported configuration key", result.stderr + result.stdout)
        self.assertFalse(Path(self.env["LOG"]).exists())

    def test_speed_floor_requires_the_same_input_cell(self):
        spec = importlib.util.spec_from_file_location("aggregate", SCRIPTS / "aggregate.py")
        aggregate = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(aggregate)
        for name, exposure, total in (("E0-w8", "a", 20), ("E2-w8", "b", 1)):
            path = self.work / name
            path.mkdir()
            (path / "metrics.json").write_text(
                json.dumps(
                    {"input_key": {"safe_ids": [exposure]}, "per_stage": {"hdu_total": {"mean_per_hdu_s": total}}}
                )
            )
        aggregate.main([str(self.work)])
        table = (self.work / "table.md").read_text()
        self.assertNotIn("MET", table)
        self.assertIn("incomparable inputs", table)


if __name__ == "__main__":
    unittest.main()
