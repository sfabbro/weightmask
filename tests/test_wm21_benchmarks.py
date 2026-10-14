from unittest.mock import patch

import numpy as np
import pytest

from benchmarks import cr_faint_curves, curate_trail_truth, poloka_tracks, score_trail_truth, streak_inject


def test_curator_bright_proposals_ignore_nonfinite_pixels():
    data = np.zeros((40, 40), dtype=np.float32)
    data[18:22, 5:35] = 50.0
    data[0, 0] = np.inf
    data[5, 5] = np.nan

    proposals = curate_trail_truth._bright_components(data, 5.0, 20, 2.0, 4)

    assert proposals
    assert proposals[0]["n_mask_px"] >= 20


def test_curator_line_stats_ignore_nonfinite_background_samples():
    data = np.zeros((120, 120), dtype=np.float32)
    data[60, :] = 10.0
    data[35, :] = np.nan
    data[85, :] = 1.0

    stats = curate_trail_truth._independent_line_stats(data, (60.0, 60.0), (1.0, 0.0), data.shape)

    assert stats is not None
    assert np.isfinite(stats["support_px"])
    assert stats["support_px"] > 0


def test_poloka_rms_uses_finite_positive_values():
    rms = np.full((8, 8), np.inf, dtype=np.float32)
    rms[2:4, 2:4] = 3.0

    assert poloka_tracks.finite_robust_rms(rms) == pytest.approx(3.0)


def test_poloka_rms_fails_when_no_valid_values_exist():
    with pytest.raises(ValueError, match="finite positive RMS"):
        poloka_tracks.finite_robust_rms(np.full((4, 4), np.nan))


def test_poloka_comparison_rejects_all_invalid_rms():
    science = np.zeros((40, 40), dtype=np.float32)
    with pytest.raises(ValueError, match="finite positive RMS"):
        score_trail_truth.run_detector(
            "poloka",
            science,
            science,
            np.full(science.shape, np.inf),
            np.zeros(science.shape, dtype=bool),
            {"SATURATE": 1000.0},
            {},
        )


def test_poloka_comparison_uses_finite_rms_for_sentinel_heavy_map():
    science = np.zeros((40, 40), dtype=np.float32)
    science[20, 10:35] = 10.0
    rms = np.full(science.shape, np.inf, dtype=np.float32)
    rms[20, 10:35] = 2.0

    mask = score_trail_truth.run_detector(
        "poloka",
        science,
        science,
        rms,
        np.zeros(science.shape, dtype=bool),
        {"SATURATE": 1000.0},
        {},
    )

    assert np.any(mask[20, 10:35])


def test_streak_placement_rejects_crossing_bands():
    class CrossingRng:
        def __init__(self):
            self.angles = iter((0.0, np.pi / 2.0))
            self.centres = iter((300.0, 500.0, 500.0, 300.0))

        def uniform(self, low, high):
            if high - low == np.pi:
                return next(self.angles)
            return next(self.centres)

    placements = list(
        streak_inject.place_trails(
            (1000, 1000),
            [(800, 8.0, False), (800, 8.0, False)],
            CrossingRng(),
            min_separation=0.0,
            margin=0.15,
            attempts=1,
        )
    )

    assert len(placements) == 1


def test_streak_flux_scales_by_local_rms_map():
    rms = np.ones((30, 30), dtype=np.float32)
    rms[:, 15:] = 7.0

    flux, body = streak_inject.trail_flux(rms.shape, 20, 5.0, False, 0.0, 15.0, 15.0, half_width=1.0, rms_map=rms)

    assert np.all(np.isfinite(flux))
    assert np.any(body[:, :15]) and np.any(body[:, 15:])
    assert float(np.max(flux[:, 15:])) == pytest.approx(7.0 * float(np.max(flux[:, :15])))


def test_streak_flux_does_not_multiply_invalid_rms_outside_support():
    rms = np.full((30, 30), np.inf, dtype=np.float32)
    rms[14:17, 5:25] = 2.0

    flux, body = streak_inject.trail_flux(rms.shape, 20, 5.0, False, 0.0, 15.0, 15.0, half_width=1.0, rms_map=rms)

    assert np.all(np.isfinite(flux))
    assert np.any(body)


def test_curator_all_invalid_axes_have_explicit_finite_veto_values():
    data = np.zeros((12, 12), dtype=np.float32)
    data[:, 3] = np.nan
    data[4, :] = np.nan

    column_sigma, row_sigma = curate_trail_truth._column_and_row_noise(data)

    assert np.isinf(column_sigma[3])
    assert np.isinf(row_sigma[4])
    assert not np.any(np.isnan(column_sigma))
    assert not np.any(np.isnan(row_sigma))


def test_curator_zero_noise_axes_remain_in_veto_baseline():
    data = np.zeros((20, 20), dtype=np.float32)
    data[:, 7] = np.where(np.arange(20) % 2, -2.0, 2.0)

    column_sigma, _ = curate_trail_truth._column_and_row_noise(data)

    assert curate_trail_truth._axis_noise_baseline(column_sigma) == 0.0
    assert column_sigma[7] > 0.0


def test_cr_variant_scores_share_one_truth_denominator():
    truth_single = np.zeros((8, 8), dtype=bool)
    truth_worm = np.zeros_like(truth_single)
    truth_single[2, 2] = True
    truth_worm[3, 3] = True
    baseline = np.zeros_like(truth_single)
    baseline[2, 2] = True
    detector_input = (np.ones_like(truth_single, dtype=np.float32), np.zeros_like(truth_single))
    with patch.object(cr_faint_curves, "scoreable_truth", wraps=cr_faint_curves.scoreable_truth) as denominator:
        rows = cr_faint_curves.score_variants(
            {"strict": truth_single | truth_worm, "loose": np.zeros_like(truth_single)},
            truth_single,
            truth_worm,
            detector_input,
            baseline,
        )

    assert denominator.call_count == 1
    assert rows["strict"]["single_scoreable"] == rows["loose"]["single_scoreable"] == 0
    assert rows["strict"]["worm_scoreable"] == rows["loose"]["worm_scoreable"] == 1


def test_cr_truth_overlap_is_rejected_instead_of_truncated():
    class OverlapRng:
        def __init__(self):
            self.origins = iter((15 * 30 + 15, 15 * 30 + 14))

        def choice(self, available):
            wanted = next(self.origins, None)
            if wanted is not None and wanted in available:
                return wanted
            return available[0]

        def uniform(self, low, high):
            return 0.0

        def integers(self, low, high):
            return 3

        def poisson(self, mean):
            return max(1, int(mean))

    science = np.full((30, 30), 100.0, dtype=np.float32)
    sky = np.zeros_like(science)
    rms = np.ones_like(science)
    injected, singles, worms, rejected = cr_faint_curves.inject(
        science,
        sky,
        rms,
        OverlapRng(),
        n_single=1,
        n_worm=1,
        return_rejected=True,
    )

    assert np.all(np.isfinite(injected))
    assert not np.any(singles & worms)
    assert all(item["reason"] in {"truth_overlap", "placement_failed"} for item in rejected)


def test_cr_rejection_count_counts_completed_events_not_attempts():
    class RetryRng:
        def __init__(self):
            self.origins = iter((15 * 30 + 15, 15 * 30 + 14, 15 * 30 + 10))

        def choice(self, available):
            wanted = next(self.origins, None)
            if wanted is not None and wanted in available:
                return wanted
            return available[0]

        def uniform(self, low, high):
            return 0.0

        def integers(self, low, high):
            return 3

        def poisson(self, mean):
            return max(1, int(mean))

    _, singles, worms, rejected = cr_faint_curves.inject(
        np.full((30, 30), 100.0, dtype=np.float32),
        np.zeros((30, 30), dtype=np.float32),
        np.ones((30, 30), dtype=np.float32),
        RetryRng(),
        n_single=1,
        n_worm=1,
        return_rejected=True,
    )

    assert singles.sum() == 1
    assert worms.sum() > 0
    assert rejected == []


def test_cr_single_event_accounting_counts_every_unplaced_request():
    _, singles, worms, rejected = cr_faint_curves.inject(
        np.full((22, 22), 100.0, dtype=np.float32),
        np.zeros((22, 22), dtype=np.float32),
        np.ones((22, 22), dtype=np.float32),
        np.random.default_rng(5),
        n_single=7,
        n_worm=0,
        return_rejected=True,
    )

    single_rejected = [item for item in rejected if item["kind"] == "single"]
    assert int(singles.sum()) + len(single_rejected) == 7
    assert not worms.any()
