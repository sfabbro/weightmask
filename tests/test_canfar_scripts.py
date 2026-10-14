"""Exercise CANFAR orchestration locally, without submissions or benchmarks."""

import gzip
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import types
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

    def test_exposure_driver_probes_cadc_and_reads_its_own_mask(self):
        """The per-exposure driver: URL candidates, and stats from a written mask.

        Both halves are what a reviewer reads back later (the mask file and the
        per-bit counts), so they are checked here rather than only in a pod.
        """
        from benchmarks.canfar import process_exposure

        urls = process_exposure.candidate_urls("1729091")
        self.assertTrue(urls[0].endswith("/1729091p.fits.fz"), urls)
        self.assertTrue(urls[1].endswith("/1729091.fits.fz"), urls)

        mask = self.work / "mask.fits"
        data = np.zeros((4, 4), dtype=np.uint16)
        data[0, 0] = int(process_exposure.QUALITY_BITS["BAD"])
        data[1, 1] = int(process_exposure.QUALITY_BITS["STREAK"])
        data[2, 2] = int(process_exposure.QUALITY_BITS["STREAK"]) | int(process_exposure.QUALITY_BITS["CR"])
        fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data=data, name="SCI")]).writeto(mask)

        rows = process_exposure.mask_bit_counts(mask)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["npix"], 16)
        self.assertEqual(row["flagged"], 3)
        self.assertEqual(row["BAD"], 1)
        self.assertEqual(row["STREAK"], 2)
        self.assertEqual(row["CR"], 1)
        self.assertEqual(row["DETECTED"], 0)
        self.assertEqual(row["extname"], "SCI")

    def test_verifier_recomputes_from_pixels_and_catches_a_bad_stats_table(self):
        """The reviewer's summary must come from the mask, not from the driver.

        A run that was cut off mid-HDU leaves a mask that opens but is short, and
        a stats file that does not describe its pixels is worse than none: both
        have to fail here rather than be reported as a clean exposure.
        """
        import json

        from benchmarks.canfar import verify_exposures
        from weightmask.contract import QUALITY_BITS

        directory = self.work / "ns-9999999"
        directory.mkdir()
        good = np.zeros((4, 4), dtype=np.uint16)
        good[0, 0] = int(QUALITY_BITS["STREAK"])
        good[1, 1] = int(QUALITY_BITS["CR"])
        fits.HDUList(
            [fits.PrimaryHDU(), fits.ImageHDU(data=good, name="SCI"), fits.ImageHDU(data=good, name="SCI")]
        ).writeto(directory / "9999999.mask.fits.fz")
        # Every column is right except CR, so the mismatch set is exactly the
        # one wrong column rather than every column the table never had.
        (directory / "9999999.hdu_stats.tsv").write_text(
            "hdu\textname\tnpix\tflagged\tSTREAK\tCR\n1\tSCI\t16\t2\t1\t0\n2\tSCI\t16\t2\t1\t0\n"
        )
        (directory / "9999999.run.json").write_text(json.dumps({"status": "ok", "hdus": [1, 2]}) + "\n")

        record = verify_exposures.summarise("9999999", directory, QUALITY_BITS)
        self.assertEqual(record["recomputed_rows"], 2)
        self.assertEqual(record["holes"], 0)
        self.assertEqual(record["STREAK"], 2)
        self.assertEqual(record["streak_hdus"], 2)
        # `QUALITY_BITS` names the bits `CR`/`SAT`, not `COSMIC_RAY`/`SATURATED`:
        # the summary reads these keys, and a prose name would write a zero column.
        self.assertEqual(record["CR"], 2)
        self.assertEqual(record["SAT"], 0)
        summary_source = (SCRIPTS / "verify_exposures.py").read_text()
        for prose in ('record.get("COSMIC_RAY"', 'record.get("SATURATED"'):
            self.assertNotIn(prose, summary_source)
        # No mask digest was recorded, so the file cannot be vouched for.
        self.assertFalse(record["digest_match"])
        # The shipped table understates CR on both HDUs; the verifier must say so.
        self.assertFalse(record["stats_match"])
        self.assertTrue(all(mismatch[1] == "CR" for mismatch in record["stats_mismatches"]))

        # A truncated mask is reported, never read as a clean short exposure.
        for name, payload in (
            ("truncated", (directory / "9999999.mask.fits.fz").read_bytes()[:64]),
            ("garbage", b"not a fits file at all"),
        ):
            broken = self.work / f"ns-{name}"
            broken.mkdir()
            obsid = "7777777" if name == "truncated" else "6666666"
            (broken / f"{obsid}.mask.fits.fz").write_bytes(payload)
            record = verify_exposures.summarise(obsid, broken, QUALITY_BITS)
            self.assertTrue(
                record.get("error") or record.get("holes", 0) >= 1,
                f"{name} mask reported as clean: {record}",
            )
            self.assertNotIn("STREAK", [key for key, value in record.items() if value == 2])

        # End to end over a results root: the summary must describe the one
        # exposure present, and must not pick up its own output directory, which
        # matches the `ns-*` glob it scans.
        root = self.work / "results"
        root.mkdir()
        entry = root / "ns-9999999"
        entry.mkdir()
        for name in ("9999999.mask.fits.fz", "9999999.hdu_stats.tsv", "9999999.run.json"):
            (entry / name).write_bytes((directory / name.replace("9999999", "9999999")).read_bytes())
        (root / "ns-review").mkdir()
        (root / "ns-review" / "streak_summary.tsv").write_text("obsid\n")
        out = root / "written.tsv"
        self.assertEqual(verify_exposures.main(["--root", str(root), "--out", str(out)]), 0)
        rows = out.read_text().strip().splitlines()
        self.assertEqual(rows[0].split("\t")[0], "obsid")
        self.assertEqual(len(rows), 2, rows)
        self.assertTrue(rows[1].startswith("9999999\t"), rows[1])
        self.assertNotIn("review", out.read_text())
        self.assertIn("cr_px", rows[0])
        self.assertEqual(rows[1].split("\t")[rows[0].split("\t").index("cr_px")], "2")

    def test_worktree_staging_carries_uncommitted_edits_and_skips_ignored_files(self):
        """A pod can only run what was staged, and a candidate is usually dirty.

        `git archive` would silently stage the last commit instead, which is how
        a measurement ends up describing code that was not the thing under test.
        """
        from benchmarks.canfar import stage_source

        repo = self.work / "repo"
        repo.mkdir()

        def git(*args):
            subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)

        git("init", "-q")
        git("config", "user.email", "test@example.invalid")
        git("config", "user.name", "test")
        (repo / "tracked.py").write_text("committed = True\n")
        (repo / ".gitignore").write_text("ignored.txt\n")
        git("add", "tracked.py", ".gitignore")
        git("commit", "-qm", "init")
        (repo / "tracked.py").write_text("uncommitted = True\n")
        (repo / "untracked.py").write_text("new = True\n")
        (repo / "ignored.txt").write_text("never staged\n")

        def extract(data):
            directory = tempfile.mkdtemp(dir=self.work)
            with tarfile.open(fileobj=io.BytesIO(gzip.decompress(data))) as archive:
                try:
                    archive.extractall(directory, filter="data")
                except TypeError:  # Python < 3.12 has no extraction filters
                    archive.extractall(directory)
            return Path(directory)

        staged = extract(stage_source.git_archive_worktree(repo))
        self.assertIn("uncommitted", (staged / "tracked.py").read_text())
        self.assertTrue((staged / "untracked.py").exists())
        self.assertFalse((staged / "ignored.txt").exists())

        committed = extract(stage_source.git_archive("HEAD", repo))
        self.assertNotIn("uncommitted", (committed / "tracked.py").read_text())
        self.assertFalse((committed / "untracked.py").exists())

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

    def _pod_ship(self):
        spec = importlib.util.spec_from_file_location("pod_ship", SCRIPTS / "pod_ship.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _pod_run(self):
        spec = importlib.util.spec_from_file_location("pod_run", SCRIPTS / "pod_run.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def test_pod_ship_carries_env_in_the_payload_not_the_platform_parameter(self):
        """The platform's `env` parameter did not deliver `WM_COMMAND`.

        Measured in a pod 2026-10-13: a command containing `--flag=value` never
        arrived, which surfaced as `FATAL: WM_TREE and WM_COMMAND are both
        required` and was recorded as a flaky control plane. Bisecting
        correlated the loss with `=` in the value (`KEY=a=b` absent, `KEY=a=`
        arriving stripped to `a`). The environment now rides in the payload, and
        this pins both halves of that: values survive the round trip, and
        nothing is sent through the lossy parameter.
        """
        pod_ship = self._pod_ship()
        env = {
            "WM_TREE": "/tmp",
            "WM_COMMAND": "{py} -m pytest -q -rf --tb=short -p no:cacheprovider",
            "WM_EQUALS": "k=v",
            "WM_DOLLAR": "$HOME/out",
            "WM_SHORTCODE": "a:b:c",
        }
        source = (
            "import json, os\n"
            f"print('PAYLOAD', json.dumps({{k: os.environ.get(k) for k in {sorted(env)!r}}}), __name__, flush=True)\n"
        ).encode()

        args, info = pod_ship.build_args(source, env)

        self.assertNotIn(" ", pod_ship.BOOTSTRAP, "a space would split the bootstrap into several arguments")
        self.assertEqual(info["env_keys"], sorted(env))
        options = types.SimpleNamespace(name=None, image="img", cpu=1, ram=4)
        self.assertNotIn(
            "env",
            pod_ship.build_params(options, args),
            "the platform env parameter loses any value containing '='",
        )

        # Reproduce the platform's own handling: it joins cmd and args and splits
        # on spaces again, so splitting here is the faithful simulation.
        result = subprocess.run(
            [sys.executable, *args.split(" ")],
            # Scrub the WM_ keys so a pass cannot come from the test's own env.
            env={key: value for key, value in os.environ.items() if not key.startswith("WM_")},
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        line = next(line for line in result.stdout.splitlines() if line.startswith("PAYLOAD "))
        # rpartition, not split: the JSON itself contains spaces.
        payload, _, name = line[len("PAYLOAD ") :].rpartition(" ")
        self.assertEqual(json.loads(payload), env)
        self.assertEqual(name, "__main__", "the payload must run as a script, not an import")

    def test_pod_ship_does_not_delete_a_job_its_own_deadline_missed(self):
        """A deadline that expires under a running pod must not delete it.

        Measured 2026-10-13: seven exposure jobs were fanned out with a fixed
        `--wait`; the three slowest were deleted mid-run by this harness, which
        left them with an unrenamed `.wm-tmp-*` mask and no stats, and the
        symptom on the mount was indistinguishable from a cluster crash. The
        session is now left in place unless it actually finished (or the caller
        asks for the old behaviour).
        """
        pod_ship = self._pod_ship()
        running = "Status        Running"
        done = "Status        Completed"
        self.assertEqual(pod_ship.deletion_plan(running, keep=False, delete_unfinished=False)[0], False)
        self.assertEqual(pod_ship.deletion_plan(done, keep=False, delete_unfinished=False)[0], True)
        self.assertEqual(pod_ship.deletion_plan(running, keep=False, delete_unfinished=True)[0], True)
        self.assertEqual(pod_ship.deletion_plan(done, keep=True, delete_unfinished=False)[0], False)
        for text, expected in ((running, False), ("Status        Failed", True), ("Succeeded", True)):
            self.assertEqual(pod_ship.status_text_is_finished(text), expected)
        # The rule has to be wired into main, not merely available.
        source = (SCRIPTS / "pod_ship.py").read_text()
        self.assertIn("deletion_plan(status_text, options.keep, options.delete_unfinished)", source)
        self.assertNotIn("if not options.keep:\n        # --force", source)

    def test_exposure_launcher_encodes_the_two_cost_lessons(self):
        """The committed launcher must pin threads and state the budget explicitly.

        Both were measured on 2026-10-14: without the pin a 16-worker pod asks for
        up to 16x192 library threads and per-HDU wall goes 0.8-1.0 ks -> 2.6-4.0 ks;
        without an explicit WM_TIMEOUT, or with one sized from a quiet field, a
        crowded exposure is killed mid-HDU and keeps only an unrenamed partial
        mask.
        """
        source = (SCRIPTS / "run_exposures.sh").read_text()
        self.assertIn('--env "WM_THREAD_PIN=1"', source)
        self.assertIn('--env "WM_TIMEOUT=$WM_TIMEOUT"', source)
        self.assertIn('--env "WM_COMMAND=$CMD"', source)
        self.assertIn(': "${WM_TIMEOUT:=', source)
        self.assertIn("process_exposure.py", source)
        self.assertIn("run.attempt2.json", source)
        self.assertIn("wm-tmp-*", source)

    def test_pod_run_thread_hygiene_drops_defaults_and_can_pin_them(self):
        """Image thread defaults are never trusted; WM_THREAD_PIN pins all pools to 1.

        Dropping them is not enough for a job that runs many workers: the libraries
        then fall back to the node's CPU count (192 per worker on a MegaCam pod), so
        16 HDUs on 16 granted cores asked for thousands of threads. Measured on
        1851960 with 16 workers, per-HDU wall was 2.6-4.0 ks with the defaults and
        0.8-1.0 ks with the pools pinned.
        """
        pod_run = self._pod_run()
        env = {
            "OMP_NUM_THREADS": "192",
            "MKL_NUM_THREADS": "192",
            "OPENBLAS_NUM_THREADS": "8",
            "OTHER": "keep",
        }
        pod_run.thread_hygiene(env)
        self.assertNotIn("OMP_NUM_THREADS", env)
        self.assertNotIn("MKL_NUM_THREADS", env)
        self.assertEqual(env["OPENBLAS_NUM_THREADS"], "8")  # untouched unless pinning
        self.assertEqual(env["OTHER"], "keep")
        pinned = dict(env)
        pod_run.thread_hygiene(pinned, pin=True)
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            self.assertEqual(pinned[key], "1")
        # The flag has to be wired into main, not merely available.
        source = (SCRIPTS / "pod_run.py").read_text()
        self.assertIn('thread_hygiene(env, pin=os.environ.get("WM_THREAD_PIN") == "1")', source)

    def test_cutout_sheet_paints_each_class_and_rejects_a_sky_window(self):
        """The review image has to be checkable, not just pretty.

        Three properties, all of which were wrong in a first draft: only the
        window's STREAK pixels may be painted as a trail, a candidate whose
        masked line is only sky must be skipped rather than published, and the
        mask HDU has to be matched to the exposure HDU by detector name (the
        recorded order is the fallback).
        """
        from benchmarks.canfar import plot_trail_cutouts as cutouts
        from weightmask.contract import QUALITY_BITS

        mask = np.zeros((64, 64), dtype=np.uint16)
        # A trail across the window, closed by the mask (which is what the
        # detector does to a dashed trail: the segments, not the gaps, are
        # flagged but the mask runs the line between endpoints), plus one CR and
        # one saturated pixel that must keep their own colours.
        mask[28:31, 6:60] = int(QUALITY_BITS["STREAK"])
        mask[5, 5] = int(QUALITY_BITS["CR"])
        mask[6, 6] = int(QUALITY_BITS["SAT"])
        mask[7, 7] = int(QUALITY_BITS["STREAK"]) | int(QUALITY_BITS["CR"])

        rgb = cutouts.paint_classes(mask)
        self.assertEqual(rgb[7, 7].tolist(), list(cutouts.CLASS_COLORS["STREAK"]))
        self.assertEqual(rgb[5, 5].tolist(), list(cutouts.CLASS_COLORS["CR"]))
        self.assertEqual(rgb[6, 6].tolist(), list(cutouts.CLASS_COLORS["SAT"]))
        self.assertEqual(rgb[0, 0].tolist(), list(cutouts.BACKGROUND))
        # DETECTED is a source catalogue, not a defect, so it stays unpainted
        # unless it is asked for.
        detected = np.zeros((2, 2), dtype=np.uint16)
        detected[0, 0] = int(QUALITY_BITS["DETECTED"])
        self.assertEqual(cutouts.paint_classes(detected)[0, 0].tolist(), list(cutouts.BACKGROUND))
        self.assertEqual(
            cutouts.paint_classes(detected, detected=True)[0, 0].tolist(),
            list(cutouts.CLASS_COLORS["DETECTED"]),
        )

        # Ranking: the named window holds the trail, and the span-1 speck is
        # below min_pixels rather than a trail of its own.
        candidates = cutouts.trail_candidates(mask, min_pixels=3)
        self.assertEqual(len(candidates), 1, candidates)
        self.assertEqual(candidates[0]["span"], 54)
        self.assertEqual(candidates[0]["npix"], int(np.count_nonzero(mask[28:31])))
        self.assertEqual(cutouts.trail_candidates(mask, min_pixels=10_000), [])
        self.assertEqual(cutouts.trail_candidates(np.zeros((8, 8), np.uint16)), [])

        # The ridge test is what separates a trail from a masked line on sky.
        sky = np.full((64, 64), 500.0)
        rng = np.random.default_rng(0)
        sky += rng.normal(0.0, 3.0, sky.shape)
        on_sky = cutouts.signal_prominence(sky, mask)
        self.assertLess(on_sky["ridge"], 1.05)
        with_trail = sky.copy()
        trail = (mask & int(QUALITY_BITS["STREAK"])) != 0
        with_trail[trail] += 120.0
        bright = cutouts.signal_prominence(with_trail, mask)
        self.assertGreater(bright["ridge"], 1.2)
        self.assertGreater(bright["bright_fraction"], 0.5)

        # The window never leaves the array, even when the centroid is on the edge.
        self.assertEqual(cutouts.window((0.0, 0.0), 32, (64, 64)), (0, 0, 32, 32))
        self.assertEqual(cutouts.window((63.0, 63.0), 32, (64, 64)), (32, 32, 64, 64))

        # Mask HDUs are matched by detector name when the mask carries one, and
        # fall back to the recorded order when it does not.
        self.assertEqual(cutouts.exposure_hdu_for("MASK_8351-11-4", [9], {"8351-11-4": 17}, 0), (17, "ccdname"))
        self.assertEqual(cutouts.exposure_hdu_for("", [17, 18], {}, 1), (18, "order"))
        self.assertEqual(cutouts.exposure_hdu_for("", [], {}, 0), (None, "none"))

    def test_cutout_sheet_renders_a_trail_component_to_a_png(self):
        """End to end on a synthetic exposure and mask: PNG, index, and mapping."""
        from benchmarks.canfar import plot_trail_cutouts as cutouts
        from weightmask.contract import QUALITY_BITS

        exposure = self.work / "0000001p.fits.fz"
        rng = np.random.default_rng(1)
        plane = (500.0 + rng.normal(0.0, 3.0, (64, 64))).astype(np.float32)
        streak = np.zeros((64, 64), dtype=np.uint16)
        # An intermittent trail: bright dashes with sky in the gaps, and a mask
        # that runs the whole line. That is the real shape on 1851960, and it is
        # why a median cannot be the test for "is there a trail here".
        streak[30:32, 6:58] = int(QUALITY_BITS["STREAK"])
        for x in range(6, 58, 3):
            plane[30:32, x : x + 2] += 200.0
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(data=plane, name="ccd00", header=fits.Header({"CCDNAME": "8351-11-4"})),
            ]
        ).writeto(exposure)
        mask = self.work / "0000001.mask.fits.fz"
        fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data=streak, name="MASK_8351-11-4")]).writeto(mask)

        png = self.work / "sheet.png"
        entry = cutouts.render("0000001", mask, exposure, [1], png, count=3)

        self.assertEqual(entry["cutouts"][0]["hdu"], 1)
        self.assertEqual(entry["cutouts"][0]["route"], "ccdname")
        self.assertEqual(entry["mapping"]["1"]["hdu"], 1)
        self.assertGreater(entry["cutouts"][0]["signal"]["ridge"], 1.2)
        self.assertEqual(entry["cutouts"][0]["classes"]["STREAK"], int(np.count_nonzero(streak)))
        self.assertTrue(png.exists() and png.stat().st_size > 1000, png)

        # A mask whose STREAK pixels all lie on sky yields no panel at all, and
        # the reason names the exposure rather than raising a bare error.
        empty = self.work / "0000002.mask.fits.fz"
        fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(data=np.zeros((64, 64), np.uint16), name="MASK_X")]).writeto(
            empty
        )
        with self.assertRaises(ValueError):
            cutouts.render("0000002", empty, exposure, [1], self.work / "none.png", count=1)

        # The driver writes the index next to the PNGs and survives one bad
        # exposure: a missing download must not cost the other exposures.
        root = self.work / "results"
        entry_dir = root / "ns-0000001"
        entry_dir.mkdir(parents=True)
        (entry_dir / "0000001.mask.fits.fz").write_bytes(mask.read_bytes())
        (entry_dir / "0000001.run.json").write_text(json.dumps({"hdus": [1], "url": ""}) + "\n")
        cache = self.work / "cache" / "0000001"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.mkdir()
        (cache / "0000001p.fits.fz").write_bytes(exposure.read_bytes())
        # The cache floor is a real 1 MB (a partial download must not be read as
        # an exposure); the synthetic frame here is a few hundred bytes.
        cutouts.MIN_EXPOSURE_BYTES = 0
        out = self.work / "out"
        self.assertEqual(cutouts.main(["--root", str(root), "--out", str(out), "--work", str(self.work / "cache")]), 0)
        index = json.loads((out / "index.json").read_text())
        self.assertEqual([item["obsid"] for item in index["exposures"]], ["0000001"])
        self.assertTrue((out / "ns-0000001.cutouts.png").exists())
        self.assertEqual(index["failures"], [])

    def test_pod_ship_env_rejects_malformed_pairs(self):
        pod_ship = self._pod_ship()
        for pairs, expected in (
            (["NOEQUALS"], "KEY=VALUE"),
            (["1BAD=x"], "not a valid environment name"),
            (["A=1", "A=2"], "given twice"),
        ):
            with self.assertRaises(SystemExit) as caught:
                pod_ship.parse_env(pairs)
            self.assertIn(expected, str(caught.exception))

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
