"""Tiny checks for benchmark inputs, scoring and failure status; no benchmark runs."""

from types import SimpleNamespace
from unittest.mock import patch

import fitsio
import numpy as np
import pytest
from scipy.ndimage import binary_dilation

from benchmarks import cr_faint_curves, perf_megacam, streak_inject
from tests import simulate_and_test as simulation
from tests.benchmarks import download_data, run
from weightmask.contract import INVERSE_VARIANCE_SEMANTICS, MASK_POLARITY


def test_streak_flux_controls_injected_signal():
    arguments = {"size": 256, "noise_level": 1, "num_stars": 0, "seed": 7}
    faint, _, truth = simulation.create_simulated_data(**arguments, streak_flux=2)
    bright, _, _ = simulation.create_simulated_data(**arguments, streak_flux=20)
    assert np.any(bright[truth["streak"]] > faint[truth["streak"]])
    assert np.array_equal(bright[~truth["streak"]], faint[~truth["streak"]])


def test_streak_source_poisson_uses_gain_and_preserves_invalid_pixels():
    raw = np.full((32, 32), 100.0, dtype=np.float32)
    source = np.full(raw.shape, 20.0, dtype=np.float32)
    invalid = np.zeros(raw.shape, dtype=bool)
    invalid[3, 4] = True
    result = streak_inject.inject_source_poisson(raw, source, 2.0, np.random.default_rng(4), invalid)
    assert result[3, 4] == raw[3, 4]
    assert np.mean(result - raw) == pytest.approx(20.0, abs=2.0)


def test_injected_truth_rejects_overlap_and_invalid_support():
    invalid = np.zeros((120, 120), dtype=bool)
    invalid[45:75, 45:75] = True
    flux, truth, trails = streak_inject.inject_grid(
        invalid.shape,
        [(40, 8.0, False), (40, 8.0, False)],
        np.random.default_rng(9),
        min_separation=0.0,
        invalid_mask=invalid,
    )
    assert not np.any(truth & invalid)
    assert len(trails) <= 2
    assert np.all(np.isfinite(flux))


def test_streak_scoreable_truth_reports_upstream_and_clean_baseline_rejections():
    body = np.zeros((8, 8), dtype=bool)
    body[2, 2] = body[3, 3] = True
    data = np.ones(body.shape, dtype=np.float32)
    existing = np.zeros(body.shape, dtype=bool)
    existing[2, 2] = True
    baseline = np.zeros(body.shape, dtype=bool)
    baseline[3, 3] = True
    scoreable, rejected = streak_inject.scoreable_truth(body, data, existing, baseline)
    assert not scoreable.any()
    assert np.array_equal(rejected, body)


def test_cr_truth_pixels_are_unique_and_valid():
    sci = np.full((40, 40), 100.0, dtype=np.float32)
    sky = np.full_like(sci, 100.0)
    rms = np.full_like(sci, 2.0)
    invalid = np.zeros(sci.shape, dtype=bool)
    invalid[10:20, 10:20] = True
    injected, singles, worms = cr_faint_curves.inject(
        sci, sky, rms, np.random.default_rng(3), n_single=30, n_worm=20, gain=1.5, invalid_mask=invalid
    )
    assert not np.any((singles | worms) & invalid)
    assert not np.any(singles & worms)
    assert np.all(np.isfinite(injected))
    _, _, _, rejected = cr_faint_curves.inject(
        sci,
        sky,
        rms,
        np.random.default_rng(3),
        n_single=30,
        n_worm=20,
        gain=1.5,
        invalid_mask=invalid,
        return_rejected=True,
    )
    assert all(item["reason"] in {"truth_overlap", "placement_failed"} for item in rejected)


def test_cr_scoreable_truth_reports_upstream_rejections():
    single = np.zeros((8, 8), dtype=bool)
    worm = np.zeros_like(single)
    single[2, 2] = True
    worm[3, 3] = True
    data = np.ones_like(single, dtype=np.float32)
    existing = np.zeros_like(single)
    existing[2, 2] = True
    baseline = np.zeros_like(single)
    baseline[3, 3] = True
    score_single, score_worm, rejected_single, rejected_worm = cr_faint_curves.scoreable_truth(
        single, worm, (data, existing), baseline
    )
    assert not score_single[2, 2]
    assert rejected_single[2, 2]
    assert not score_worm[3, 3]
    assert rejected_worm[3, 3]


def test_synthetic_streak_truth_only_marks_injected_support():
    data = np.zeros((256, 256), dtype=np.float32)
    truth = {name: np.zeros(data.shape, bool) for name in ("sat", "cr", "stars", "streak", "defects")}
    simulation._add_streaks(data, truth, 256, 40.0, "complex")
    supported = binary_dilation(data > 0, iterations=2)
    assert np.all(~truth["streak"] | supported)


def test_synthetic_object_recall_excludes_prior_masks():
    args = SimpleNamespace(size=384, noise=5.0, stars=20, streak=50.0, mask_pct=0.0, regime_type="normal", seed=11)
    metrics = simulation.run_masking_test("weightmask.yml", args, save_fits=False)
    assert metrics["Objects"][1] >= 0.90


def test_complex_component_labels_record_upstream_object_exclusion():
    args = SimpleNamespace(size=256, noise=1.0, stars=0, streak=80.0, mask_pct=0.0, regime_type="complex", seed=7)
    _, products = simulation.run_masking_test("weightmask.yml", args, save_fits=False, return_products=True)
    component = products["ground_truth"]["streak_components"] == 1
    exclusion = products["streak_exclusion_mask"]
    prediction = products["masks"]["streaks"]
    assert np.any(component)
    assert np.any(component & exclusion)
    assert not np.any(prediction & exclusion)


def test_complex_mode_reaches_simulator(tmp_path):
    config = tmp_path / "config.yml"
    config.write_text("{}\n")
    args = SimpleNamespace(size=256, noise=1, stars=0, streak=2, complex_mode=True, seed=7)
    # Stop at the simulator boundary: no detector work or output files needed.
    with patch.object(simulation, "create_simulated_data", side_effect=RuntimeError("stop")) as generate:
        with pytest.raises(RuntimeError, match="stop"):
            simulation.run_masking_test(str(config), args, save_fits=False)
    assert generate.call_args.kwargs["regime_type"] == "complex"


def test_simulated_read_noise_excludes_poisson_noise(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = tmp_path / "config.yml"
    config.write_text("{}\n")
    args = SimpleNamespace(size=4, noise=5, stars=0, streak=2, regime_type="complex", seed=7)
    science = np.full((4, 4), 100, np.float32)
    rms = np.full_like(science, np.sqrt(100 + args.noise**2))
    mask = np.zeros(science.shape, bool)
    truth = {name: mask for name in ("sat", "cr", "stars", "streak")}
    with (
        patch.object(simulation, "create_simulated_data", return_value=(science, rms, truth)),
        patch.object(simulation, "detect_saturated_pixels", return_value=(65535, "test", mask)),
        patch.object(simulation, "grow_bleed_trails", return_value=mask),
        patch.object(simulation, "detect_cosmic_rays", return_value=mask) as cosmics,
        patch.object(simulation, "detect_objects", return_value=mask),
        patch.object(simulation, "detect_streaks", return_value=mask),
        patch.object(simulation, "estimate_background", return_value=(science, rms)),
    ):
        simulation.run_masking_test(str(config), args, save_fits=False)
    assert cosmics.call_args.kwargs["read_noise"] == args.noise


def test_halo_only_prediction_has_zero_core_recall():
    truth = np.zeros((9, 9), bool)
    truth[4, 4] = True
    halo = binary_dilation(truth, iterations=2)
    prediction = halo & ~truth
    stats = run._mask_stats(prediction, truth, eval_gt_mask=halo)
    assert stats["precision"] == pytest.approx(1)
    assert stats["recall"] == 0
    assert stats["f1"] == 0
    precision, recall = simulation.evaluate_mask(prediction, truth, "Streaks")
    assert precision == pytest.approx(1)
    assert recall == 0


@pytest.mark.parametrize("suite", ["synthetic_v2", "acs_compare", "all"])
def test_unknown_case_exits_nonzero_without_running_detectors(tmp_path, suite):
    with (
        patch.object(run, "OUTPUT_ROOT", tmp_path),
        patch.object(run, "run_masking_test", side_effect=AssertionError("must not evaluate")),
    ):
        assert run.main(["--suite", suite, "--case", "not-a-case"]) == 1


def test_quality_gates_reject_nonfinite_metrics():
    metrics = {
        "mask_polarity": MASK_POLARITY,
        "inverse_variance_semantics": INVERSE_VARIANCE_SEMANTICS,
        "flagged_pixels_have_zero_weight": True,
        "clean_weight_matches_inverse_variance": True,
        "invalid_variance_is_flagged": True,
        "overmask_fraction": np.nan,
        "flux_bias_fraction": np.nan,
        "noise_scale": np.nan,
    }
    failures = run._quality_gate_failures("fixture", metrics, {})
    for key in ("overmask_fraction", "flux_bias_fraction", "noise_scale"):
        assert any(key in failure for failure in failures)


def test_interrupted_download_is_not_a_successful_cache_entry(tmp_path):
    destination = tmp_path / "science.fits"

    def interrupted(_url, path):
        with open(path, "wb") as handle:
            handle.write(b"partial FITS")
        raise OSError("connection lost")

    with patch.object(download_data, "urlretrieve", side_effect=interrupted) as fetch:
        assert not download_data.download_http("https://example.invalid/data", destination)
        assert not destination.exists()
        assert not download_data.download_http("https://example.invalid/data", destination)
        assert fetch.call_count == 2


@pytest.mark.parametrize("quarantine", [download_data._quarantine_invalid_file, perf_megacam._quarantine_invalid])
def test_repeated_quarantine_preserves_existing_archives(tmp_path, quarantine):
    source = tmp_path / "science.fits"
    usual = tmp_path / "science.fits.invalid"
    usual.write_bytes(b"original archive")
    for payload in (b"first invalid input", b"second invalid input"):
        source.write_bytes(payload)
        quarantine(source)
        assert not source.exists()
        assert usual.read_bytes() == b"original archive"
    assert {path.read_bytes() for path in tmp_path.iterdir()} == {
        b"original archive",
        b"first invalid input",
        b"second invalid input",
    }


def test_download_failure_exits_nonzero(tmp_path):
    manifest = {
        "cases": [
            {"case_id": "fixture", "download_source": "http", "download_url": "unused", "local_path": "missing.fits"}
        ]
    }
    with (
        patch.object(download_data, "ROOT", tmp_path),
        patch.object(download_data, "load_manifest", return_value=manifest),
        patch.object(download_data, "download_http", return_value=False),
    ):
        assert download_data.main(["--suite", "acs_compare"]) == 1


def test_sweep_failure_has_nonzero_status_and_deterministic_seeds():
    seen = []

    def failed_case(_config, args, **_kwargs):
        seen.append(getattr(args, "seed", None))
        return {"Streaks": (0, 0)}

    with patch.object(simulation, "run_masking_test", side_effect=failed_case):
        assert simulation.run_auto_sweep() == 1
    assert seen and all(seed is not None for seed in seen)


def test_reference_plane_must_cover_nonzero_crop_origin():
    crop = {"y0": 4, "x0": 4, "height": 4, "width": 4}
    with pytest.raises(ValueError, match="does not cover science crop"):
        run._crop_reference(np.ones((4, 4)), crop)


def test_dark_injection_uses_production_dark_detector(tmp_path):
    dark = tmp_path / "dark.fits"
    residual = np.full((8, 8), 5, np.float32)
    residual[3, 3] = 20
    fitsio.write(str(dark), residual)
    with (
        patch.object(run, "ROOT", tmp_path),
        patch.object(run, "_load_repo_config", return_value={"dark_masking": {"hot_sigma": 8}}),
        patch.object(run, "detect_cosmic_rays", return_value=np.zeros((8, 8), bool)),
    ):
        _, products, baselines = run._evaluate_dark_injection(
            {"case_id": "fixture", "dark_local_path": "dark.fits"}, np.ones((8, 8), np.float32), {}, True
        )
    assert products["pred_bad"][3, 3]
    assert products["pred_bad"].sum() == 1
    assert "astroscrappy_only" not in baselines, "the postfiltered weightmask CR result is not an independent baseline"
    assert "dark_threshold_baseline" not in baselines, "truth scored against itself is not a comparator"


def test_dark_manifest_selects_science_and_does_not_claim_fake_comparators():
    case = next(
        case for case in run.load_manifest("megacam_real")["cases"] if case["label_recipe"] == "dark_injection_v1"
    )
    assert "Observation.type = 'OBJECT'" in case["cadc_query"]
    assert not case["comparators"]


def test_sweep_gates_reject_nonfinite_scores():
    metrics = {name: (np.nan, np.nan) for name in ("Saturation", "Cosmics", "Objects", "Streaks")}
    results = {
        name: metrics for name in ("Complex Ideal (Var Bkg/PSF)", "Extreme Poisson Crowded", "Galactic Plane (Crowded)")
    }
    assert simulation._benchmark_gate_failures(results)


@pytest.mark.parametrize("metric", ["streak_f1", "object_recall", "bad_pixel_f1"])
def test_synthetic_gates_reject_nonfinite_scores_without_benchmark(tmp_path, metric):
    shape = (4, 4)
    mask = np.zeros(shape, bool)
    products = {
        "science": np.zeros(shape),
        "bkg_rms": np.ones(shape),
        "ground_truth": {"streak": mask},
        "masks": {"streaks": mask},
    }
    metrics = {"Objects": (1, np.nan if metric == "object_recall" else 1)}
    with (
        patch.object(run, "OUTPUT_ROOT", tmp_path),
        patch.object(run, "run_masking_test", return_value=(metrics, products)),
        patch.object(run, "_mask_stats", return_value={"f1": np.nan if metric == "streak_f1" else 1}),
        patch.object(
            run, "_benchmark_synthetic_bad_pixels", return_value={"f1": np.nan if metric == "bad_pixel_f1" else 1}
        ),
    ):
        summary = run.run_synthetic_v2(selected_cases={"synthetic_sparse"})
    assert summary["gate_failures"]


def test_perf_explicit_local_input_ignores_unrelated_resolved_cache(tmp_path):
    science = tmp_path / "1013719p.fits"
    fitsio.write(
        str(science),
        np.ones((4, 5), np.float32),
        header={"INSTRUME": "MegaPrime", "DETECTOR": "MegaCam", "EXPTIME": 560},
    )
    with (
        patch.object(perf_megacam, "OUT_DIR", tmp_path / "outputs"),
        patch.object(perf_megacam, "resolve_exposures", side_effect=AssertionError("unrelated cached records used")),
    ):
        assert perf_megacam.main(["--exposure-file", str(science), "--resolve-only"]) == 0


def test_perf_explicit_input_validates_instrument(tmp_path, capsys):
    science = tmp_path / "wrong.fits"
    fitsio.write(str(science), np.ones((4, 5), np.float32), header={"INSTRUME": "Other", "DETECTOR": "Other"})
    with pytest.raises(SystemExit):
        perf_megacam.main(["--exposure-file", str(science), "--resolve-only"])
    assert "validation failed" in capsys.readouterr().err


def test_perf_explicit_input_does_not_count_links_twice(tmp_path, capsys):
    science = tmp_path / "science.fits"
    fitsio.write(str(science), np.ones((4, 5), np.float32), header={"INSTRUME": "MegaPrime", "DETECTOR": "MegaCam"})
    alias = tmp_path / "alias.fits"
    alias.hardlink_to(science)
    with pytest.raises(SystemExit):
        perf_megacam.main(["--exposure-file", str(science), "--exposure-file", str(alias), "--resolve-only"])
    assert "duplicate" in capsys.readouterr().err


def test_perf_rejects_obsolete_all_hdus_flag_before_resolving():
    with patch.object(perf_megacam, "resolve_exposures", side_effect=AssertionError("must reject before resolving")):
        with pytest.raises(SystemExit):
            perf_megacam.main(["--all-hdus", "--resolve-only"])
