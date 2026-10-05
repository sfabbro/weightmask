"""Small evidence checks; no data-backed detectors or full benchmarks run."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from astropy.io import fits

from benchmarks import (
    cr_faint_curves,
    exposure_time,
    perf_megacam,
    poloka_tracks,
    streak_inject,
    streak_stage_sweep,
    weight_pull_compare,
)
from benchmarks import curate_trail_truth as curator
from benchmarks import score_trail_truth as scorer


class LocalPool:
    def __init__(self, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def map(self, fn, jobs, **kwargs):
        return map(fn, jobs)


def test_tolerant_recall_counts_truth_hits_not_mask_area():
    truth = np.zeros((30, 30), bool)
    truth[10, 5:25] = True
    mask = np.zeros_like(truth)
    mask[11:16, 5:15] = True
    _, tolerant, _ = streak_inject.trail_recall(mask, truth)
    assert tolerant < 0.8  # mask area is large, but it misses the right end.


def test_empty_mask_has_no_phantom_distance_hits():
    truth = np.zeros((8, 8), bool)
    truth[:2, :2] = True
    assert streak_inject.trail_recall(np.zeros_like(truth), truth) == (0.0, 0.0, 0.0)


def test_cache_key_includes_inputs_config_and_detector_source(tmp_path):
    data = np.zeros((4, 5), np.float32)
    rms = np.ones_like(data)
    existing = np.zeros_like(data, bool)
    source = tmp_path / "streaks.py"
    source.write_text("version A")
    with patch.object(streak_inject, "SOURCE_DIR", tmp_path):
        original = streak_inject.baseline_key(data, rms, existing, {"upstream": 1})
        assert original != streak_inject.baseline_key(data + 1, rms, existing, {"upstream": 1})
        assert original != streak_inject.baseline_key(data, rms + 1, existing, {"upstream": 1})
        assert original != streak_inject.baseline_key(data, rms, ~existing, {"upstream": 1})
        assert original != streak_inject.baseline_key(data, rms, existing, {"upstream": 2})
        source.write_text("version B")
        assert original != streak_inject.baseline_key(data, rms, existing, {"upstream": 1})


def scoring_fixture(tmp_path):
    fixture = json.loads((Path(__file__).resolve().parents[1] / scorer.DEFAULT_FIXTURE).read_text())
    fixture["entries"] = fixture["entries"][:1]
    fixture["summary"]["by_label"] = {"trail": 0, "artefact": 1, "uncertain": 0}
    source = tmp_path / "labels.json"
    source.write_text(json.dumps(fixture))
    return fixture, source


def test_data_root_relocates_fixture_files(tmp_path):
    fixture, path = scoring_fixture(tmp_path)
    exposure = fixture["entries"][0]["exposure"]
    relocated = tmp_path / (exposure + ".fits.fz")
    relocated.touch()
    jobs = []

    def fake_score(job):
        jobs.append(job)
        return {"exposure": exposure, "hdu": 2, "results": {}, "error": "fixture stop"}

    with patch.object(scorer, "ProcessPoolExecutor", LocalPool), patch.object(scorer, "score_hdu", fake_score):
        scorer.main(["--fixture", str(path), "--data-root", str(tmp_path), "--report", str(tmp_path / "score.json")])
    assert jobs and jobs[0][0] == str(relocated)


def test_incomplete_scoring_fails_the_gate(tmp_path):
    fixture, path = scoring_fixture(tmp_path)
    exposure = fixture["entries"][0]["exposure"]
    data = tmp_path / "data.fits"
    data.touch()
    fixture["data_requirements"]["exposures"][exposure]["path"] = str(data)
    path.write_text(json.dumps(fixture))
    record = {"exposure": exposure, "hdu": 2, "results": {}, "error": "detector failed"}
    with patch.object(scorer, "ProcessPoolExecutor", LocalPool), patch.object(scorer, "score_hdu", return_value=record):
        code = scorer.main(["--fixture", str(path), "--report", str(tmp_path / "score.json")])
    assert code == 1
    assert json.loads((tmp_path / "score.json").read_text())["gate_failures"]


def test_curator_review_retains_uncertain_entries(tmp_path):
    fixture, _ = scoring_fixture(tmp_path)
    artefact = fixture["entries"][0]
    uncertain = copy.deepcopy(artefact)
    uncertain["label"] = "uncertain"
    inventory = {
        "fixture": {
            "path": "fixture.fits",
            "hdus": [],
            "object": "test",
            "crval": [0, 0],
            "date_obs": "test",
            "exptime": 1,
            "filter": "r",
            "nhdu": 0,
        }
    }
    reviewed = []
    with (
        patch.object(curator, "exposure_inventory", return_value=(inventory, [])),
        patch.object(curator, "ProcessPoolExecutor", LocalPool),
        patch.object(curator, "curate", return_value=[artefact, uncertain]),
        patch.object(curator, "review_sheet", side_effect=lambda f: reviewed.extend(f["entries"]) or "review"),
    ):
        curator.main(
            [
                "--exposures",
                "fixture",
                "--out",
                str(tmp_path / "labels.json"),
                "--review-sheet",
                str(tmp_path / "review.md"),
            ]
        )
    assert len(reviewed) == 2
    written = json.loads((tmp_path / "labels.json").read_text())
    assert len(written["entries"]) == written["summary"]["n_entries"] == 1


def test_poloka_exclusions_stay_excluded_below_zero_sky():
    data = np.full((30, 100), -100.0)
    excluded = np.zeros(data.shape, bool)
    excluded[:, 45:50] = True
    mask, _, _ = poloka_tracks.poloka_satellite_mask(data, data, 1, existing_mask=excluded)
    assert not mask.any()


def test_cr_injection_adds_to_image_and_rejects_unknown_rms():
    data = np.full((40, 40), 1000.0)
    sky = np.zeros_like(data)
    out, singles, _ = cr_faint_curves.inject(data, sky, np.ones_like(data), np.random.default_rng(0), 1, 0)
    assert np.all(out[singles] > data[singles])
    rms = np.full_like(data, np.inf)
    rms[15:25, 15:25] = 1
    out, singles, worms = cr_faint_curves.inject(data, sky, rms, np.random.default_rng(0), 10, 5)
    assert np.all(np.isfinite(out[singles | worms]))


def test_pull_scatter_is_centered():
    values = np.arange(1000, dtype=float) / 1000
    assert weight_pull_compare.pull_stats(values)["rsig"] == weight_pull_compare.pull_stats(values + 100)["rsig"]


def test_scaling_requires_single_worker_reference():
    with patch.object(exposure_time, "run_once", side_effect=AssertionError("must validate before timing")):
        try:
            exposure_time.scaling(SimpleNamespace(scaling=[2, 4], repeats=1), "fixture", "/tmp", 36)
        except ValueError as exc:
            assert "1" in str(exc)
        else:
            raise AssertionError("missing reference accepted")


def test_known_different_ccds_do_not_match_by_extver():
    assert not curator._same_ccd({"ccdname": "A", "extver": 1}, {"ccdname": "B", "extver": 1})


def test_component_brightness_ignores_unknown_rms():
    mask = np.zeros((20, 20), bool)
    mask[5:15, :10] = True
    data = np.full(mask.shape, 20.0)
    rms = np.ones(mask.shape)
    rms[:, 10:] = np.inf
    assert streak_stage_sweep.component_table(mask, data, rms)[0]["bright_blobs_on_line"] > 0


def test_perf_comparison_rejects_same_output_directory(tmp_path):
    with patch.object(perf_megacam, "resolve_exposures", side_effect=AssertionError("must validate first")):
        try:
            perf_megacam.main(["--out-dir", str(tmp_path), "--compare-baseline", str(tmp_path)])
        except SystemExit as exc:
            assert "same" in str(exc).lower()
        else:
            raise AssertionError("self comparison accepted")


def test_cr_off_truth_pixels_ignore_preexisting_detections():
    truth = np.zeros((20, 20), bool)
    truth[10, 10] = True
    baseline = np.zeros_like(truth)
    baseline[0, 0] = True
    flag = baseline | truth
    assert cr_faint_curves.score(flag, truth, np.zeros_like(truth), baseline)[2] == 0


def test_perf_cpu_seconds_only_cover_the_timed_processing_region(tmp_path):
    import resource

    from weightmask import cli, mef

    image = SimpleNamespace(get_info=lambda: {"dims": [4, 5]})

    class Handle:
        def __getitem__(self, index):
            return image

        def close(self):
            pass

    rec = {"file": str(tmp_path / "input.fits"), "safe_id": "fixture", "publisherID": "fixture"}
    rusage = [
        SimpleNamespace(ru_utime=100, ru_stime=0, ru_maxrss=1000),
        SimpleNamespace(ru_utime=105, ru_stime=0, ru_maxrss=1000),
    ]
    with (
        patch("fitsio.FITS", return_value=Handle()),
        patch.object(cli, "get_hdus_to_process", return_value=[0]),
        patch.object(cli, "determine_output_paths", return_value={}),
        patch.object(mef, "process_all_hdus", return_value=1),
        patch.object(perf_megacam, "_install_timing_collector", return_value=([], None, None, None)),
        patch.object(perf_megacam, "_uninstall_timing_collector"),
        patch.object(resource, "getrusage", side_effect=rusage),
    ):
        report = perf_megacam._process_one_exposure(rec, 1)
    assert report["cpu_s"] == 5


def test_perf_failed_hdus_are_a_failure_and_sweep_preserves_flat(tmp_path):
    rec = {"safe_id": "fixture", "file": "fixture", "publisherID": "fixture"}
    report = {**rec, "nhdus_expected": 1, "nhdus_processed": 0, "wall_s": 1, "mpix": 1, "stage_totals": {}}
    flat = tmp_path / "flat.fits"
    flat.touch()
    calls = []

    def process(*args, **kwargs):
        calls.append(kwargs)
        return dict(report)

    with (
        patch.object(perf_megacam, "OUT_DIR", tmp_path / "outputs"),
        patch.object(perf_megacam, "resolve_exposures", return_value=[rec]),
        patch.object(perf_megacam, "_process_one_exposure", side_effect=process),
        patch("tests.benchmarks.download_data.validate_case_file", return_value=(True, None)),
    ):
        code = perf_megacam.main(["--flat", str(flat), "--no-cprofile", "--config-set", "variance.default_gain=2"])
    assert code == 1
    assert len(calls) == 4 and all(call.get("flat") == str(flat) for call in calls)
    saved = json.loads((tmp_path / "outputs/megacam_perf.json").read_text())
    assert "flat_publisherID" not in saved["header"]
    assert saved["header"]["config_overrides"] == {"variance": {"default_gain": 2}}
    assert "unmodified" not in saved["header"]["config"]


@pytest.mark.parametrize(
    "target,link", [("science", "symlink"), ("science", "hardlink"), ("other", "symlink"), ("other", "hardlink")]
)
def test_persistence_needs_distinct_physical_other_exposures(tmp_path, target, link):
    data = np.ones((8, 9), np.float32)
    data[:, 4] = 100
    header = fits.Header({"CCDNAME": "A", "OBSTYPE": "OBJECT"})
    science = tmp_path / "science.fits"
    other = tmp_path / "other.fits"
    fits.PrimaryHDU(data, header).writeto(science)
    fits.PrimaryHDU(data, header).writeto(other)
    alias = tmp_path / "alias.fits"
    if link == "symlink":
        alias.symlink_to(science if target == "science" else other)
    else:
        alias.hardlink_to(science if target == "science" else other)
    assert scorer._persistence_prior(science, header, data.shape) is None


def test_persistence_reads_later_compatible_extension(tmp_path):
    data = np.ones((8, 9), np.float32)
    data[:, 4] = 100
    header = fits.Header({"CCDNAME": "A", "OBSTYPE": "OBJECT"})
    science = tmp_path / "science.fits"
    fits.PrimaryHDU(data, header).writeto(science)
    for name in ("other1.fits", "other2.fits"):
        fits.HDUList(
            [
                fits.PrimaryHDU(),
                fits.ImageHDU(np.ones((2, 3), np.float32), header),
                fits.ImageHDU(data, header),
            ]
        ).writeto(tmp_path / name)
    expected = np.zeros(data.shape, bool)
    expected[:, 4] = True
    np.testing.assert_array_equal(scorer._persistence_prior(science, header, data.shape), expected)


@pytest.mark.parametrize(
    "name,key,value",
    [
        ("flat_08Bm01_r.fits.fz", "OBSTYPE", "OBJECT"),
        ("other_calibration.fits", "OBSTYPE", "FLAT"),
        ("other_calibration.fits", "IMAGETYP", "Dark Frame"),
        ("other_calibration.fits", "EXPTYPE", "BIAS"),
    ],
)
def test_persistence_excludes_calibration_frames(tmp_path, name, key, value):
    data = np.ones((8, 9), np.float32)
    data[:, 4] = 100
    header = fits.Header({"CCDNAME": "A", "OBSTYPE": "OBJECT"})
    science = tmp_path / "science.fits"
    fits.PrimaryHDU(data, header).writeto(science)
    fits.PrimaryHDU(data, header).writeto(tmp_path / "other.fits")
    calibration = fits.Header({key: value})
    fits.HDUList([fits.PrimaryHDU(header=calibration), fits.ImageHDU(data, header)]).writeto(tmp_path / name)
    assert scorer._persistence_prior(science, header, data.shape) is None
