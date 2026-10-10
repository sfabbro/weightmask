import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks import exposure_time, perf_megacam, perf_protocol


def test_interleaved_schedule_is_seeded_and_balanced():
    first = perf_protocol.interleaved_schedule(5, seed=17)
    second = perf_protocol.interleaved_schedule(5, seed=17)

    assert first == second
    assert len(first) == 10
    assert [arm for _, arm in first].count("A") == 5
    assert [arm for _, arm in first].count("B") == 5
    assert all({first[index][1], first[index + 1][1]} == {"A", "B"} for index in range(0, 10, 2))


def test_summary_reports_median_iqr_and_mad():
    assert perf_protocol.summarize([1, 2, 3, 4, 5]) == {
        "n": 5,
        "median": 3.0,
        "iqr": 2.0,
        "mad": 1.0,
    }


def test_protocol_requires_five_repetitions():
    with pytest.raises(ValueError, match="at least 5"):
        perf_protocol.validate_repeats(4)


def test_release_eligibility_requires_repetitions_and_wm20_baseline():
    assert not perf_protocol.release_eligibility(1, True, True)["eligible"]
    assert not perf_protocol.release_eligibility(5, False, True)["eligible"]
    assert perf_protocol.release_eligibility(5, True, True, full_products=True, distinct_arms=True)["eligible"]


def test_release_eligibility_requires_full_products_and_distinct_arms():
    assert not perf_protocol.release_eligibility(5, True, True, full_products=False, distinct_arms=True)["eligible"]
    assert not perf_protocol.release_eligibility(5, True, True, full_products=True, distinct_arms=False)["eligible"]
    assert perf_protocol.release_eligibility(5, True, True, full_products=True, distinct_arms=True)["eligible"]


def test_identical_arm_definitions_are_not_distinct():
    assert not perf_protocol.distinct_arm_definitions({"baseline": {"command": "run"}, "treatment": {"command": "run"}})
    assert perf_protocol.distinct_arm_definitions(
        {"baseline": {"command": "run"}, "treatment": {"command": "run --new"}}
    )


def test_release_eligibility_is_reachable_with_one_distinct_protocol_and_descriptive_path():
    result = perf_protocol.protocols_release_eligibility(
        [
            {"distinct_arms": False, "correctness_equivalent": True},
            {"distinct_arms": True, "correctness_equivalent": True},
        ],
        5,
        True,
        full_products=True,
    )

    assert result["eligible"] is True


def test_schedule_can_add_a_separate_cold_sample_to_five_warm_repetitions():
    schedule = perf_protocol.interleaved_schedule(5, seed=17, cold_repetitions=1)

    assert len(schedule) == 12
    assert [arm for _, arm in schedule].count("A") == 6
    assert [arm for _, arm in schedule].count("B") == 6


def test_report_rejects_fewer_than_five_warm_samples(tmp_path):
    config = tmp_path / "config.yml"
    input_path = tmp_path / "input.fits"
    config.write_text("threshold: 2\n")
    input_path.write_bytes(b"input")
    with pytest.raises(ValueError, match="warm samples"):
        perf_protocol.make_report(
            instrument="fixture",
            input_paths=[input_path],
            config_paths=[config],
            repeats=5,
            seed=0,
            schedule=perf_protocol.interleaved_schedule(5, seed=0, cold_repetitions=1),
            samples={"cold": {"A": [1], "B": [1]}, "warm": {"A": [1, 2, 3, 4], "B": [1, 2, 3, 4]}},
            correctness={"A": True, "B": True},
            thread_environment=perf_protocol.thread_environment(),
        )


def test_logical_config_hash_ignores_ephemeral_cache_paths(tmp_path):
    first = tmp_path / "first.yml"
    second = tmp_path / "second.yml"
    first.write_text("flat_masking:\n  bad_mask_cache_dir: /tmp/one/cache\ntemp_dir: /tmp/one/run\n")
    second.write_text("flat_masking:\n  bad_mask_cache_dir: /tmp/two/cache\ntemp_dir: /tmp/two/run\n")

    assert perf_protocol.logical_config_hash(first) == perf_protocol.logical_config_hash(second)


def test_report_contains_hashes_thread_environment_and_cache_summaries(tmp_path):
    input_path = tmp_path / "input.fits"
    config_path = tmp_path / "config.yml"
    input_path.write_bytes(b"input")
    config_path.write_text("threshold: 2\n")

    report = perf_protocol.make_report(
        instrument="fixture",
        input_paths=[input_path],
        config_paths=[config_path],
        repeats=5,
        seed=17,
        schedule=perf_protocol.interleaved_schedule(5, seed=17, cold_repetitions=1),
        samples={"cold": {"A": [1], "B": [2]}, "warm": {"A": [3, 4, 5, 6, 7], "B": [4, 5, 6, 7, 8]}},
        correctness={"A": True, "B": True},
        thread_environment={"OMP_NUM_THREADS": "1"},
    )

    assert report["protocol"]["repeats"] == 5
    assert report["protocol"]["schedule"] == [[n, arm] for n, arm in report["protocol"]["schedule"]]
    assert report["correctness"]["equivalent"] is True
    assert report["inputs"][0]["sha256"]
    assert report["configuration"][0]["sha256"]
    assert report["thread_environment"] == {"OMP_NUM_THREADS": "1"}
    assert report["results"]["cold"]["A"]["median"] == 1.0
    assert "best" not in json.dumps(report).lower()


def test_thread_environment_ignores_inherited_values():
    assert perf_protocol.thread_environment({name: "99" for name in perf_protocol.THREAD_VARIABLES}) == {
        name: "1" for name in perf_protocol.THREAD_VARIABLES
    }


def test_cache_directories_isolate_cold_samples_and_share_warm_arm_cache(tmp_path):
    cold_a = perf_protocol.cache_directory(tmp_path, "cold", "A", 0)
    cold_b = perf_protocol.cache_directory(tmp_path, "cold", "A", 1)
    warm_a = perf_protocol.cache_directory(tmp_path, "warm", "A", 0)
    warm_b = perf_protocol.cache_directory(tmp_path, "warm", "A", 1)

    assert cold_a != cold_b
    assert warm_a == warm_b


def test_exposure_config_records_private_cold_cache(tmp_path):
    cold = perf_protocol.cache_directory(tmp_path, "cold", "A", 0)
    path = exposure_time.make_config(tmp_path, True, config_id="cold", cache_dir=cold)

    assert str(cold) in Path(path).read_text()


def test_exposure_time_script_help_works_in_pixi():
    completed = subprocess.run(
        ["pixi", "run", "exposure-time", "--", "--help"],
        cwd=os.fspath(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--repeats" in completed.stdout


def test_megacam_cli_help_exposes_release_protocol_controls():
    completed = subprocess.run(
        ["pixi", "run", "python", "benchmarks/perf_megacam.py", "--help"],
        cwd=os.fspath(Path(__file__).resolve().parents[1]),
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--repeats" in completed.stdout
    assert "--seed" in completed.stdout


def test_perf_markdown_separates_descriptive_headline_from_release_eligibility(tmp_path):
    output = tmp_path / "perf.md"
    report = {
        "header": {"arch": "fixture", "cpu_count": 1, "platform": "test", "config": "weightmask.yml"},
        "total_wall_s": 3.0,
        "hdu_wall_s": 2.0,
        "mpix_total": 1.0,
        "mpix_per_s": 0.333,
        "stages": {},
        "per_exposure": [],
        "sequential_descriptive": {"wall_s": 3.0, "speedup_eligible": False},
        "timing_protocol": {"release_evidence_eligible": True},
    }

    perf_megacam._write_markdown(report, {"1": 3.0}, "skipped", out_md=output)
    text = output.read_text()
    headline = next(line for line in text.splitlines() if line.startswith("exposures="))

    assert "release_evidence_eligible" not in headline
    assert "sequential_descriptive.speedup_eligible=false" in text
    assert "release_evidence_eligible=true" in text


def test_usage_distinguishes_sequential_description_from_distinct_scaling_protocol():
    usage = (Path(__file__).resolve().parents[1] / "docs/usage.md").read_text()

    assert "sequential" in usage and "descriptive" in usage
    assert "distinct" in usage and "worker-scaling" in usage


def test_correctness_gate_rejects_mismatch_before_timing_claims():
    with pytest.raises(ValueError, match="correctness equivalence"):
        perf_protocol.require_correctness_equivalence({"A": True, "B": False})


def test_exposure_comparison_uses_interleaved_repetitions_and_writes_report(tmp_path, monkeypatch):
    exposure = tmp_path / "exposure.fits"
    exposure.write_bytes(b"fixture")
    calls = []
    cache_dirs = []

    def fake_run_once(_exposure, enable, outdir, **kwargs):
        calls.append((enable, kwargs["run_id"]))
        cache_dirs.append(Path(kwargs["cache_dir"]))
        Path(kwargs["cache_dir"]).mkdir(parents=True, exist_ok=True)
        Path(outdir, kwargs["run_id"]).mkdir(parents=True, exist_ok=True)
        return 10.0 + len(calls), "--- Image processed in 1.0 seconds --- top=streaks:0.2s\n"

    monkeypatch.setattr(exposure_time, "run_once", fake_run_once)
    monkeypatch.setattr(exposure_time, "_arm_output_equivalent", lambda _paths: True)
    # `comparison` hashes `[exposure, FLAT]` into the report. The shipped FLAT is
    # a gitignored benchmark_data file, so the real one is absent on the runner
    # and the test would fail there while passing on a developer's tree. Nothing
    # reads its contents here -- `run_once` is faked and the report only sha256s
    # the path -- so a fixture stands in.
    flat = tmp_path / "flat.fits.fz"
    flat.write_bytes(b"flat")
    monkeypatch.setattr(exposure_time, "FLAT", str(flat))
    report_path = tmp_path / "report.json"
    report = exposure_time.comparison(
        SimpleNamespace(repeats=5, seed=17, keep=False),
        str(exposure),
        str(tmp_path / "runs"),
        1,
        report_path=report_path,
    )

    assert len(calls) == 14
    assert {run_id for _, run_id in calls[:2]} == {"prewarm-baseline", "prewarm-treatment"}
    assert not (tmp_path / "runs/prewarm-baseline").exists()
    assert not (tmp_path / "runs/prewarm-treatment").exists()
    assert report["protocol"]["schedule"] == [
        [n, arm]
        for n, arm in perf_protocol.interleaved_schedule(5, seed=17, arms=("baseline", "treatment"), cold_repetitions=1)
    ]
    assert report["results"]["warm"]["baseline"]["n"] == 5
    assert report["results"]["cold"]["treatment"]["n"] == 1
    assert all(not path.exists() for path in cache_dirs if "/cold/" in str(path))
    assert json.loads(report_path.read_text()) == report
    assert "best" not in report_path.read_text().lower()


def test_megacam_protocol_reports_warm_stats_and_correctness_gate(tmp_path, monkeypatch):
    record = {
        "safe_id": "fixture",
        "file": "fixture.fits",
        "publisherID": None,
        "nhdus_expected": 1,
        "nhdus_processed": 1,
        "nhdus_timed": 1,
        "wall_s": 1.0,
        "mpix": 1.0,
        "stage_totals": {"hdu_total": 1.0},
        "peak_rss_kb": 1.0,
        "cpu_s": 1.0,
    }
    calls = []

    def process(*args, **kwargs):
        calls.append(kwargs)
        Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
        cache_dir = kwargs["config_overrides"]["flat_masking"]["bad_mask_cache_dir"]
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        return dict(record)

    monkeypatch.setattr(perf_megacam, "OUT_DIR", tmp_path / "outputs")
    monkeypatch.setattr(perf_megacam, "_process_one_exposure", process)
    monkeypatch.setattr(
        perf_megacam,
        "_compare_products",
        lambda *_args, **_kwargs: {"ok": True},
    )
    protocol = perf_megacam._run_protocol(
        [record],
        1,
        treatment_workers=2,
        repeats=5,
        seed=3,
        flat=None,
        write_mask=False,
        config_overrides={},
        hdu_limit=0,
        all_products=False,
    )

    assert len(calls) == 14
    assert {Path(call["output_dir"]).name for call in calls[:2]} == {"prewarm-baseline", "prewarm-treatment"}
    assert all("prewarm" not in str(call.get("output_dir")) for call in calls[2:])
    assert protocol["protocol"]["results"]["warm"]["treatment"]["n"] == 5
    assert protocol["protocol"]["results"]["cold"]["baseline"]["n"] == 1
    assert protocol["protocol"]["correctness_equivalent"] is True
    assert protocol["protocol"]["distinct_arms"] is True
    assert protocol["protocol"]["arms"]["baseline"]["workers"] != protocol["protocol"]["arms"]["treatment"]["workers"]
    cold_root = tmp_path / "outputs/protocol/workers-1-vs-2/cache/cold"
    assert not cold_root.exists() or not list(cold_root.iterdir())
    assert not (tmp_path / "outputs/protocol/workers-1-vs-2/prewarm-baseline").exists()
    assert not (tmp_path / "outputs/protocol/workers-1-vs-2/prewarm-treatment").exists()
