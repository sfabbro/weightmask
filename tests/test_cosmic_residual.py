import numpy as np
import pytest

import weightmask.cosmics as cosmics


@pytest.mark.parametrize("gate", [cosmics._post_filter_components, cosmics._filter_faint_components])
def test_component_contrast_gates_are_invariant_to_sky_level(gate):
    residual = np.zeros((8, 8), dtype=np.float32)
    flag = np.zeros(residual.shape, dtype=bool)
    flag[3, 2:5] = True
    residual[flag] = 2.0
    rms = np.ones_like(residual)
    without_sky = gate(flag, residual, rms, {})
    with_sky = gate(flag, residual + 100.0, rms, {})
    assert not without_sky.any()
    assert np.array_equal(with_sky, without_sky)


def test_residual_faint_detection_requires_measured_rms():
    sky = np.full((32, 32), 100.0, dtype=np.float32)
    data = sky.copy()
    data[10, 10:17] += 8.0
    rms = np.ones_like(sky)
    measured = cosmics._detect_residual_faint_components(
        data, np.zeros(data.shape, dtype=bool), sky, rms, {}, psf_aware=False
    )
    assert measured[10, 10:17].all()
    rms[10, 10:17] = np.inf
    unmeasured = cosmics._detect_residual_faint_components(
        data, np.zeros(data.shape, dtype=bool), sky, rms, {}, psf_aware=False
    )
    assert not unmeasured.any()


def test_residual_faint_enhancement_recovers_linear_component_and_rejects_compact_source():
    detector = getattr(cosmics, "_detect_residual_faint_components", None)
    assert detector is not None, "residual faint-CR enhancement is not implemented"

    shape = (96, 96)
    sky = np.full(shape, 100.0, dtype=np.float32)
    data = sky.copy()
    rms = np.ones(shape, dtype=np.float32)

    worm = np.array([(30 + i, 40 + i) for i in range(10)], dtype=int)
    data[worm[:, 0], worm[:, 1]] += 8.0

    star = np.array([(65 + dy, 65 + dx) for dy in range(3) for dx in range(3)], dtype=int)
    data[star[:, 0], star[:, 1]] += 8.0

    config = {
        "threshold_sig": 4.0,
        "min_component_area": 4,
        "min_elongation": 4.0,
        "min_contrast_sigma": 4.0,
    }
    flag = detector(data, np.zeros(shape, dtype=bool), sky, rms, config, header={})

    assert flag[worm[:, 0], worm[:, 1]].mean() > 0.8
    assert not flag[star[:, 0], star[:, 1]].any()


def test_residual_mode_skips_second_full_astroscrappy_call(monkeypatch):
    shape = (64, 64)
    data = np.full(shape, 100.0, dtype=np.float32)
    data[20:25, 30:35] += 8.0
    sky = np.full(shape, 100.0, dtype=np.float32)
    rms = np.ones(shape, dtype=np.float32)
    existing = np.zeros(shape, dtype=bool)
    calls = []

    def fake_detect_cosmics(image, **kwargs):
        calls.append(kwargs)
        return np.zeros(image.shape, dtype=bool), image.copy()

    monkeypatch.setattr(cosmics, "detect_cosmics", fake_detect_cosmics)
    config = {
        "niter": 1,
        "faint_cr": {
            "enable": True,
            "enhancement": "residual",
            "residual": {
                "threshold_sig": 4.0,
                "min_component_area": 4,
                "min_elongation": 4.0,
                "min_contrast_sigma": 4.0,
            },
        },
    }

    cosmics.detect_cosmic_rays(data, existing, 65000.0, 1.5, 5.0, config, bkg_rms_map=rms, sky_map=sky, header={})

    assert len(calls) == 1
