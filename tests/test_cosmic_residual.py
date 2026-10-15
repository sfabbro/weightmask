import numpy as np
import pytest

import weightmask.cosmics as cosmics
from weightmask.config import section_config

CR = section_config("cosmic_ray")


def _post_filter(flag, sci, rms):
    return cosmics._filter_components(
        flag, sci, rms, None, max_area=CR["max_component_area"], min_contrast=CR["min_component_contrast_sigma"]
    )


def _faint_filter(flag, sci, rms):
    return cosmics._filter_faint_components(flag, sci, rms, CR["faint_cr"])


def _residual(data, sky, rms, psf_aware=True, header=None, **residual):
    cfg = section_config("cosmic_ray", {"psf_aware": psf_aware, "faint_cr": {"residual": residual}})
    return cosmics._detect_residual_faint_components(
        data, np.zeros(data.shape, dtype=bool), sky, rms, cfg, gain=1.5, read_noise=5.0, header=header
    )


@pytest.mark.parametrize("gate", [_post_filter, _faint_filter])
def test_component_contrast_gates_are_invariant_to_sky_level(gate):
    residual = np.zeros((8, 8), dtype=np.float32)
    flag = np.zeros(residual.shape, dtype=bool)
    flag[3, 2:5] = True
    residual[flag] = 2.0
    rms = np.ones_like(residual)
    without_sky = gate(flag, residual, rms)
    with_sky = gate(flag, residual + 100.0, rms)
    assert not without_sky.any()
    assert np.array_equal(with_sky, without_sky)


def test_residual_faint_detection_requires_measured_rms():
    sky = np.full((32, 32), 100.0, dtype=np.float32)
    data = sky.copy()
    data[10, 10:17] += 8.0
    rms = np.ones_like(sky)
    assert _residual(data, sky, rms, psf_aware=False)[10, 10:17].all()
    rms[10, 10:17] = np.inf
    assert not _residual(data, sky, rms, psf_aware=False).any()


def test_residual_faint_enhancement_recovers_linear_component_and_rejects_compact_source():
    shape = (96, 96)
    sky = np.full(shape, 100.0, dtype=np.float32)
    data = sky.copy()
    rms = np.ones(shape, dtype=np.float32)

    worm = np.array([(30 + i, 40 + i) for i in range(10)], dtype=int)
    data[worm[:, 0], worm[:, 1]] += 8.0

    star = np.array([(65 + dy, 65 + dx) for dy in range(3) for dx in range(3)], dtype=int)
    data[star[:, 0], star[:, 1]] += 8.0

    flag = _residual(data, sky, rms, header={}, threshold_sig=4.0, min_component_area=4, min_elongation=4.0)

    assert flag[worm[:, 0], worm[:, 1]].mean() > 0.8
    assert not flag[star[:, 0], star[:, 1]].any()


def test_residual_psf_gate_uses_the_cosmic_ray_fwhm(monkeypatch):
    seen = []

    def spy(crmask, sci, fwhm_guess, *args, **kwargs):
        seen.append(fwhm_guess)
        return crmask

    monkeypatch.setattr(cosmics, "_apply_psf_protection", spy)
    shape = (32, 32)
    sky = np.full(shape, 100.0, dtype=np.float32)
    cfg = section_config("cosmic_ray", {"psf_fwhm_guess": 4.5})
    cosmics._detect_residual_faint_components(
        sky + 8.0, np.zeros(shape, dtype=bool), sky, np.ones(shape), cfg, gain=1.5, read_noise=5.0, header=None
    )
    assert seen == [4.5]


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
            "residual": {"threshold_sig": 4.0, "min_component_area": 4, "min_elongation": 4.0},
        },
    }

    cosmics.detect_cosmic_rays(data, existing, 65000.0, 1.5, 5.0, config, bkg_rms_map=rms, sky_map=sky, header={})

    assert len(calls) == 1
