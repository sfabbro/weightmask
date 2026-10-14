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
    mine_trail_candidates,
    perf_megacam,
    poloka_tracks,
    streak_inject,
    streak_recall_floor,
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


def test_archived_stage_evidence_requires_current_provenance(tmp_path):
    source = tmp_path / "streaks.py"
    metric = tmp_path / "streak_inject.py"
    source.write_text("version A")
    metric.write_text("metric A")
    config = {"stage": "contours", "threshold": 2.5}
    input_data = np.arange(16, dtype=np.float32).reshape(4, 4)
    with patch.object(streak_recall_floor, "_metric_code_paths", return_value=(metric,)):
        artifact = streak_recall_floor.build_stage_evidence(
            records=[{"stage": "contours", "recall": 1.0}],
            input_data=input_data,
            config=config,
            source_paths=[source],
        )
        assert streak_recall_floor.verify_stage_evidence(
            artifact, input_data=input_data, config=config, source_paths=[source]
        )

        changed_records = copy.deepcopy(artifact)
        changed_records["records"][0]["recall"] = 0.0
        with pytest.raises(ValueError, match="records hash"):
            streak_recall_floor.verify_stage_evidence(
                changed_records, input_data=input_data, config=config, source_paths=[source]
            )

        metric.write_text("metric B")
        with pytest.raises(ValueError, match="code hash"):
            streak_recall_floor.verify_stage_evidence(
                artifact, input_data=input_data, config=config, source_paths=[source]
            )
        metric.write_text("metric A")

    source.write_text("version B")
    with pytest.raises(ValueError, match="source hash"):
        streak_recall_floor.verify_stage_evidence(artifact, input_data=input_data, config=config, source_paths=[source])

    source.write_text("version A")
    changed_config = {**config, "threshold": 3.0}
    with pytest.raises(ValueError, match="config hash"):
        streak_recall_floor.verify_stage_evidence(
            artifact, input_data=input_data, config=changed_config, source_paths=[source]
        )

    with pytest.raises(ValueError, match="input hash"):
        streak_recall_floor.verify_stage_evidence(
            artifact, input_data=input_data + 1, config=config, source_paths=[source]
        )

    changed = copy.deepcopy(artifact)
    changed["evidence"]["metric_revision"] = "old-metric"
    with pytest.raises(ValueError, match="metric revision"):
        streak_recall_floor.verify_stage_evidence(changed, input_data=input_data, config=config, source_paths=[source])


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


def test_controlled_cr_sigclip_curve_is_unit_invariant_only_for_fixed_rule():
    shipped_sigclip = cr_faint_curves.load_shipped_sigclip()
    assert shipped_sigclip == 8.5
    rows = cr_faint_curves.normalized_residual_sigclip_curve(
        fixed_sigclip=shipped_sigclip, seed=7, n_noise=200_000, n_cosmic=2_000
    )
    fixed = [row for row in rows if row["variant"] == "fixed_dimensionless"]
    legacy = [row for row in rows if row["variant"] == "legacy_adu_rms"]

    assert {row["sigclip"] for row in fixed} == {shipped_sigclip}
    assert len({(row["recall"], row["false_positive_rate"]) for row in fixed}) == 1
    assert [row["sigclip"] for row in legacy] == sorted((row["sigclip"] for row in legacy), reverse=True)
    assert legacy[-1]["sigclip"] < legacy[0]["sigclip"]
    assert legacy[-1]["recall"] > legacy[0]["recall"]
    assert legacy[-1]["false_positive_rate"] > legacy[0]["false_positive_rate"]


def test_controlled_cr_sigclip_curve_requires_explicit_fixed_threshold():
    with pytest.raises(TypeError, match="fixed_sigclip"):
        cr_faint_curves.normalized_residual_sigclip_curve(seed=7)


def test_normalized_cr_cli_loads_requested_config_sigclip(tmp_path):
    config = tmp_path / "weightmask.yml"
    config.write_text("cosmic_ray:\n  sigclip: 7.25\n")

    with patch.object(cr_faint_curves, "normalized_residual_sigclip_curve", return_value=[]) as curve:
        assert cr_faint_curves.main(["--normalized-residuals", "--config", str(config)]) == 0

    curve.assert_called_once_with(seed=0, fixed_sigclip=7.25)


def test_pull_scatter_is_centered():
    values = np.arange(1000, dtype=float) / 1000
    assert weight_pull_compare.pull_stats(values)["rsig"] == weight_pull_compare.pull_stats(values + 100)["rsig"]


def _named_product(data, extname="WEIGHT", ccd="A"):
    hdu = fits.PrimaryHDU(data)
    hdu.header["EXTNAME"] = extname
    hdu.header["CCDNAME"] = ccd
    lower = extname.lower()
    if lower == "mask":
        artifact, semantics = "quality_mask", "named_quality_bits"
    elif "ivar" in lower or "inverse" in lower:
        artifact, semantics = "inverse_variance", "inverse_variance_adu^-2"
    elif "sky" in lower:
        artifact, semantics = "sky", "background_adu"
    elif "confidence" in lower:
        artifact, semantics = "confidence", "normalized_weight_0_to_1"
    else:
        artifact, semantics = "weight", "masked_inverse_variance"
    hdu.header["WMART"] = artifact
    hdu.header["WMSEM"] = semantics
    return hdu


FLOAT32_EIGHT_ULP = 8 * np.finfo(np.float32).eps
DOCUMENTED_FLOAT_TOLERANCES = {
    "inverse_variance": (2e-6, max(1e-7, FLOAT32_EIGHT_ULP)),
    "normalized_weight": (2e-6, 1e-6),
    "confidence": (2e-6, 1e-6),
    "sky": (2e-6, 1e-4),
}


def test_product_comparison_is_symmetric_over_expected_files(tmp_path):
    current = tmp_path / "current"
    baseline = tmp_path / "baseline"
    current.mkdir()
    baseline.mkdir()
    _named_product(np.ones((2, 2), dtype=np.float32)).writeto(current / "run.weight.fits")
    _named_product(np.ones((2, 2), dtype=np.float32)).writeto(baseline / "run.weight.fits")
    _named_product(np.ones((2, 2), dtype=np.uint32), "MASK").writeto(baseline / "run.mask.fits")

    report = perf_megacam._compare_products(current, baseline)

    assert report["missing_in_this"] == ["run.mask.fits"]
    assert report["missing_in_baseline"] == []
    assert report["different"] == 0
    assert report["ok"] is False


def test_product_comparison_requires_explicit_expected_product_manifest(tmp_path):
    current = tmp_path / "current"
    baseline = tmp_path / "baseline"
    current.mkdir()
    baseline.mkdir()
    for directory in (current, baseline):
        _named_product(np.ones((2, 2), dtype=np.float32)).writeto(directory / "run.weight.fits")

    report = perf_megacam._compare_products(current, baseline, expected_files=["run.weight.fits", "run.mask.fits"])

    assert report["missing_in_this"] == ["run.mask.fits"]
    assert report["missing_in_baseline"] == ["run.mask.fits"]
    assert report["ok"] is False


def test_product_comparison_rejects_unrequested_new_current_product(tmp_path):
    current = tmp_path / "current"
    baseline = tmp_path / "baseline"
    current.mkdir()
    baseline.mkdir()
    _named_product(np.ones((2, 2), dtype=np.float32)).writeto(current / "run.weight.fits")
    _named_product(np.ones((2, 2), dtype=np.float32), "MASK").writeto(current / "run.mask.fits")
    _named_product(np.ones((2, 2), dtype=np.float32)).writeto(baseline / "run.weight.fits")

    report = perf_megacam._compare_products(current, baseline, expected_files=["run.weight.fits"])

    assert report["unexpected_in_this"] == ["run.mask.fits"]
    assert report["ok"] is False


def test_product_comparison_rejects_positional_hdu_mispairing(tmp_path):
    current = tmp_path / "current"
    baseline = tmp_path / "baseline"
    current.mkdir()
    baseline.mkdir()

    def write(path, order):
        hdus = [fits.PrimaryHDU()]
        for ccd in order:
            hdu = fits.ImageHDU(np.full((2, 2), ord(ccd), dtype=np.float32), name=f"WEIGHT_{ccd}")
            hdu.header["CCDNAME"] = ccd
            hdu.header["WMART"] = "weight"
            hdu.header["WMSEM"] = "masked_inverse_variance"
            hdus.append(hdu)
        fits.HDUList(hdus).writeto(path)

    write(current / "run.weight.fits", ("A", "B"))
    write(baseline / "run.weight.fits", ("B", "A"))

    report = perf_megacam._compare_products(current, baseline)

    assert report["different"] == 1
    assert any("ordering" in problem for problem in report["details"][0]["problems"])


def test_product_comparison_checks_contract_metadata(tmp_path):
    current = tmp_path / "current"
    baseline = tmp_path / "baseline"
    current.mkdir()
    baseline.mkdir()

    for directory, semantics in ((current, "normalized_weight_0_to_1"), (baseline, "inverse_variance_adu^-2")):
        hdu = _named_product(np.ones((2, 2), dtype=np.float32))
        hdu.header["WMVERS"] = "1"
        hdu.header["WMART"] = "weight"
        hdu.header["WMSEM"] = semantics
        hdu.writeto(directory / "run.weight.fits")

    report = perf_megacam._compare_products(current, baseline)

    assert report["different"] == 1
    assert any("metadata" in problem for problem in report["details"][0]["problems"])


def test_product_comparison_rejects_incompatible_semantics_even_when_headers_match(tmp_path):
    current = tmp_path / "current"
    baseline = tmp_path / "baseline"
    current.mkdir()
    baseline.mkdir()
    for directory in (current, baseline):
        hdu = _named_product(np.ones((2, 2), dtype=np.float32))
        hdu.header["WMART"] = "weight"
        hdu.header["WMSEM"] = "normalized_weight_0_to_1"
        hdu.writeto(directory / "run.ivar.fits")

    report = perf_megacam._compare_products(current, baseline, expected_files=["run.ivar.fits"])

    assert report["ok"] is False
    assert "incompatible" in report["details"][0]["problems"][0]


def test_product_comparison_requires_declared_contract_semantics(tmp_path):
    current = tmp_path / "current"
    baseline = tmp_path / "baseline"
    current.mkdir()
    baseline.mkdir()
    for directory in (current, baseline):
        hdu = _named_product(np.ones((2, 2), dtype=np.float32))
        hdu.header.remove("WMART")
        hdu.header.remove("WMSEM")
        hdu.writeto(directory / "run.weight.fits")

    report = perf_megacam._compare_products(current, baseline, expected_files=["run.weight.fits"])

    assert report["ok"] is False
    assert "missing WMART/WMSEM" in report["details"][0]["problems"][0]


@pytest.mark.parametrize(
    "mutation,needle",
    [
        ("shape", "shape"),
        ("mask_dtype", "dtype"),
        ("mask_pixel", "mask"),
    ],
)
def test_product_comparison_rejects_shape_dtype_and_pixel_mutations(tmp_path, mutation, needle):
    current = tmp_path / "current"
    baseline = tmp_path / "baseline"
    current.mkdir()
    baseline.mkdir()
    baseline_data = np.zeros((2, 2), dtype=np.uint32)
    current_data = baseline_data.copy()
    if mutation == "shape":
        current_data = np.zeros((3, 2), dtype=np.uint32)
    elif mutation == "mask_dtype":
        current_data = current_data.astype(np.int16)
    else:
        current_data[0, 0] = 1
    _named_product(baseline_data, "MASK").writeto(baseline / "run.mask.fits")
    _named_product(current_data, "MASK").writeto(current / "run.mask.fits")

    report = perf_megacam._compare_products(current, baseline, expected_files=["run.mask.fits"])

    assert report["ok"] is False
    assert any(needle in problem.lower() for problem in report["details"][0]["problems"])


def test_product_comparison_rejects_duplicate_hdu_identity(tmp_path):
    current = tmp_path / "current"
    baseline = tmp_path / "baseline"
    current.mkdir()
    baseline.mkdir()

    def write(path):
        hdus = [fits.PrimaryHDU()]
        for value in (1.0, 2.0):
            hdu = fits.ImageHDU(np.full((2, 2), value, dtype=np.float32), name="WEIGHT")
            hdu.header["CCDNAME"] = "A"
            hdu.header["WMART"] = "weight"
            hdu.header["WMSEM"] = "masked_inverse_variance"
            hdus.append(hdu)
        fits.HDUList(hdus).writeto(path)

    write(current / "run.weight.fits")
    write(baseline / "run.weight.fits")

    report = perf_megacam._compare_products(current, baseline)

    assert report["different"] == 1
    assert "duplicate" in report["details"][0]["problems"][0].lower()


def test_product_comparison_rejects_duplicate_extname_with_distinct_ccd_ids(tmp_path):
    current = tmp_path / "current"
    baseline = tmp_path / "baseline"
    current.mkdir()
    baseline.mkdir()

    def write(path):
        hdus = [fits.PrimaryHDU()]
        for ccd in ("A", "B"):
            hdu = fits.ImageHDU(np.ones((2, 2), dtype=np.float32), name="WEIGHT")
            hdu.header["CCDNAME"] = ccd
            hdu.header["WMART"] = "weight"
            hdu.header["WMSEM"] = "masked_inverse_variance"
            hdus.append(hdu)
        fits.HDUList(hdus).writeto(path)

    write(current / "run.weight.fits")
    write(baseline / "run.weight.fits")

    report = perf_megacam._compare_products(current, baseline)

    assert report["ok"] is False
    assert "duplicate" in report["details"][0]["problems"][0].lower()


def test_product_comparison_rejects_empty_product_files(tmp_path):
    current = tmp_path / "current"
    baseline = tmp_path / "baseline"
    current.mkdir()
    baseline.mkdir()
    fits.PrimaryHDU().writeto(current / "run.weight.fits")
    fits.PrimaryHDU().writeto(baseline / "run.weight.fits")

    report = perf_megacam._compare_products(current, baseline, expected_files=["run.weight.fits"])

    assert report["ok"] is False
    assert "nonempty" in report["details"][0]["problems"][0]


def test_product_comparison_retains_blank_primary_contract_metadata(tmp_path):
    current = tmp_path / "current"
    baseline = tmp_path / "baseline"
    current.mkdir()
    baseline.mkdir()

    def write(path, version):
        primary = fits.PrimaryHDU(header=fits.Header({"WMVERS": version, "WMART": "weight"}))
        image = fits.ImageHDU(np.ones((2, 2), dtype=np.float32), name="WEIGHT")
        image.header["CCDNAME"] = "A"
        image.header["WMART"] = "weight"
        image.header["WMSEM"] = "masked_inverse_variance"
        fits.HDUList([primary, image]).writeto(path)

    write(current / "run.weight.fits", "1")
    write(baseline / "run.weight.fits", "2")

    report = perf_megacam._compare_products(current, baseline)

    assert report["ok"] is False
    assert "primary metadata" in report["details"][0]["problems"][0]


@pytest.mark.parametrize(
    "dtype,extname,artifact,semantics",
    [
        (np.float32, "WEIGHT", "weight", "masked_inverse_variance"),
        (np.uint16, "MASK", "quality_mask", "named_quality_bits"),
        (np.uint32, "MASK", "quality_mask", "named_quality_bits"),
    ],
)
def test_product_comparison_ignores_compression_transport_cards_for_supported_dtypes(
    tmp_path, dtype, extname, artifact, semantics
):
    current = tmp_path / "current"
    baseline = tmp_path / "baseline"
    current.mkdir()
    baseline.mkdir()
    data = np.arange(4, dtype=dtype).reshape(2, 2)
    compressed = fits.CompImageHDU(data, name=extname)
    compressed.header["CCDNAME"] = "A"
    compressed.header["WMART"] = artifact
    compressed.header["WMSEM"] = semantics
    plain = fits.ImageHDU(data, name=extname)
    plain.header["CCDNAME"] = "A"
    plain.header["WMART"] = artifact
    plain.header["WMSEM"] = semantics
    fits.HDUList([fits.PrimaryHDU(), compressed]).writeto(current / "run.weight.fits.fz")
    fits.HDUList([fits.PrimaryHDU(), plain]).writeto(baseline / "run.weight.fits.fz")
    import fitsio

    with fitsio.FITS(str(current / "run.weight.fits.fz"), "rw") as handle:
        handle[1].write_key("ZSIMPLE", True)

    report = perf_megacam._compare_products(current, baseline)

    assert report["ok"] is True


def test_product_comparison_ignores_structural_comment_and_extend_cards(tmp_path):
    current = tmp_path / "current"
    baseline = tmp_path / "baseline"
    current.mkdir()
    baseline.mkdir()
    for directory, comment, extend in ((current, "current layout", True), (baseline, "old layout", False)):
        hdu = _named_product(np.ones((2, 2), dtype=np.float32))
        hdu.header["EXTEND"] = extend
        hdu.header.add_comment(comment)
        hdu.writeto(directory / "run.weight.fits")

    report = perf_megacam._compare_products(current, baseline)

    assert report["ok"] is True


def test_product_comparison_does_not_hide_domain_zeropt_metadata(tmp_path):
    current = tmp_path / "current"
    baseline = tmp_path / "baseline"
    current.mkdir()
    baseline.mkdir()
    for directory, zeropt in ((current, 1.0), (baseline, 2.0)):
        hdu = _named_product(np.ones((2, 2), dtype=np.float32))
        hdu.header["ZEROPT"] = zeropt
        hdu.writeto(directory / "run.weight.fits")

    report = perf_megacam._compare_products(current, baseline)

    assert report["ok"] is False
    assert "ZEROPT" in report["details"][0]["problems"][0]


def test_product_comparison_rejects_float_dtype_change(tmp_path):
    current = tmp_path / "current"
    baseline = tmp_path / "baseline"
    current.mkdir()
    baseline.mkdir()
    _named_product(np.ones((2, 2), dtype=np.float32)).writeto(baseline / "run.weight.fits")
    _named_product(np.ones((2, 2), dtype=np.float64)).writeto(current / "run.weight.fits")

    report = perf_megacam._compare_products(current, baseline)

    assert report["ok"] is False
    assert "dtype" in report["details"][0]["problems"][0]


def test_expected_product_manifest_matches_cli_individual_mask_names():
    records = [{"safe_id": "run"}]
    names = perf_megacam._expected_product_names(records, workers=1, flat=None, write_mask=True, all_products=True)
    assert "run.w1.weight.bad.fits" in names
    assert "run.w1.bad.fits" not in names
    assert "run.w1.mask.fits" in names


def test_expected_product_manifest_respects_mask_write_flag():
    records = [{"safe_id": "run"}]
    without = perf_megacam._expected_product_names(records, workers=1, flat=None, write_mask=False, all_products=False)
    with_mask = perf_megacam._expected_product_names(records, workers=1, flat=None, write_mask=True, all_products=False)
    assert "run.w1.mask.fits" not in without
    assert "run.w1.mask.fits" in with_mask


def test_requested_worker_products_exist_before_baseline_comparison(tmp_path):
    source = tmp_path / "science.fits"
    baseline = tmp_path / "baseline"
    out = tmp_path / "out"
    source.touch()
    baseline.mkdir()
    events = []
    record = {
        "publisherID": None,
        "safe_id": "science",
        "file": str(source),
        "nhdus_expected": 1,
        "nhdus_processed": 1,
        "nhdus_timed": 1,
        "wall_s": 0.1,
        "mpix": 1.0,
        "stage_totals": {},
        "peak_rss_kb": 0.0,
        "cpu_s": 0.0,
    }

    def process(rec, workers, out_suffix="", **kwargs):
        events.append(("process", workers, out_suffix))
        return dict(record)

    def compare(*args, **kwargs):
        events.append(("compare",))
        return {
            "expected_file_count": 1,
            "file_count": 1,
            "identical": 1,
            "different": 0,
            "missing": [],
            "missing_in_baseline": [],
            "missing_in_this": [],
            "unexpected_in_this": [],
            "unexpected_in_baseline": [],
            "details": [],
            "ok": True,
        }

    with (
        patch.object(perf_megacam, "_validate_megacam", return_value=(True, None)),
        patch.object(perf_megacam, "_process_one_exposure", side_effect=process),
        patch.object(perf_megacam, "_compare_products", side_effect=compare),
        patch("weightmask.mef._resolve_max_workers", side_effect=lambda workers, _: workers),
    ):
        assert (
            perf_megacam.main(
                [
                    "--exposure-file",
                    str(source),
                    "--out-dir",
                    str(out),
                    "--compare-baseline",
                    str(baseline),
                    "--workers",
                    "2",
                    "--no-cprofile",
                ]
            )
            == 0
        )

    assert events[-1] == ("compare",)
    assert events[-2][0:2] == ("process", 2)


def test_product_comparison_ignores_blank_compressed_primary(tmp_path):
    current = tmp_path / "current"
    baseline = tmp_path / "baseline"
    current.mkdir()
    baseline.mkdir()

    def write(path):
        primary = fits.PrimaryHDU()
        image = fits.CompImageHDU(np.ones((2, 2), dtype=np.float32), name="WEIGHT")
        image.header["CCDNAME"] = "A"
        image.header["WMART"] = "weight"
        image.header["WMSEM"] = "masked_inverse_variance"
        fits.HDUList([primary, image]).writeto(path)

    write(current / "run.weight.fits.fz")
    write(baseline / "run.weight.fits.fz")

    report = perf_megacam._compare_products(current, baseline)

    assert report["ok"] is True


@pytest.mark.parametrize("kind", ["inverse_variance", "normalized_weight", "confidence", "sky"])
def test_product_comparison_numeric_tolerance_is_dtype_aware_and_bounded(tmp_path, kind):
    current = tmp_path / "current"
    baseline = tmp_path / "baseline"
    current.mkdir()
    baseline.mkdir()
    filename = f"run.{kind}.fits"
    base = np.ones((2, 2), dtype=np.float32)
    _named_product(base, kind.upper()).writeto(baseline / filename)
    rtol, atol = DOCUMENTED_FLOAT_TOLERANCES[kind]
    _named_product(base + np.float32(0.5 * (atol + rtol)), kind.upper()).writeto(current / filename)
    assert perf_megacam._compare_products(current, baseline)["ok"] is True
    _named_product(base + np.float32(4 * (atol + rtol)), kind.upper()).writeto(current / filename, overwrite=True)
    assert perf_megacam._compare_products(current, baseline)["ok"] is False


def test_mine_trail_candidates_rejects_missing_or_ambiguous_mask_match():
    target = fits.Header({"CCDNAME": "A"})
    with pytest.raises(ValueError):
        mine_trail_candidates._matching_hdu(fits.HDUList([fits.PrimaryHDU()]), target)
    duplicate = fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(np.zeros((2, 2)), name="MASK"),
            fits.ImageHDU(np.zeros((2, 2)), name="MASK"),
        ]
    )
    duplicate[1].header["CCDNAME"] = "A"
    duplicate[2].header["CCDNAME"] = "A"
    with pytest.raises(ValueError):
        mine_trail_candidates._matching_hdu(duplicate, target)


def test_weight_pull_rejects_normalized_weight_as_inverse_variance():
    with pytest.raises(ValueError, match="inverse_variance"):
        weight_pull_compare._validate_inverse_variance_header({"WMART": "weight", "WMSEM": "normalized_weight_0_to_1"})


def test_weight_pull_main_wiring_uses_inverse_variance_not_normalized_weight():
    science = np.array([100.0, 104.0])
    sky = np.full(2, 100.0)
    common = np.ones(2, dtype=bool)
    old_ivar = np.full(2, 0.25)
    new_ivar = np.full(2, 0.25)
    normalized_weight = np.full(2, 1.0)
    old_pulls = weight_pull_compare.gaussian_pulls(science, sky, old_ivar, mask=~common)
    new_pulls = weight_pull_compare.gaussian_pulls(science, sky, new_ivar, mask=~common)
    wrong_pulls = weight_pull_compare.gaussian_pulls(science, sky, normalized_weight, mask=~common)
    np.testing.assert_array_equal(old_pulls, [0.0, 2.0])
    np.testing.assert_array_equal(new_pulls, [0.0, 2.0])
    np.testing.assert_array_equal(wrong_pulls, [0.0, 4.0])


def test_weight_pull_series_wires_old_and_new_physical_inverse_variance(monkeypatch):
    calls = []

    def capture(science, sky, inverse_variance, mask=None):
        calls.append(inverse_variance.copy())
        return np.array([1.0])

    monkeypatch.setattr(weight_pull_compare, "gaussian_pulls", capture)
    old = np.array([0.25])
    new = np.array([0.0625])
    result = weight_pull_compare._pull_series(
        np.array([1.0]), np.array([True]), old, np.array([0.0]), new, np.array([0.0])
    )

    assert [tag for tag, _ in result] == ["old", "new"]
    np.testing.assert_array_equal(calls, [old, new])


def test_gaussian_pulls_use_physical_inverse_variance():
    rng = np.random.default_rng(13)
    sigma = 4.0
    science = rng.normal(100.0, sigma, size=100_000)
    sky = np.full_like(science, 100.0)
    inverse_variance = np.full_like(science, 1.0 / sigma**2)
    normalized_weight = np.full_like(science, 0.25)

    pulls = weight_pull_compare.gaussian_pulls(science, sky, inverse_variance)
    wrong_pulls = weight_pull_compare.gaussian_pulls(science, sky, normalized_weight)

    assert np.std(pulls) == pytest.approx(1.0, abs=0.02)
    assert np.std(wrong_pulls) == pytest.approx(2.0, abs=0.04)


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
