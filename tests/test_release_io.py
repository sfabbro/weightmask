"""Small release regressions for IO safety and contract boundaries."""

import warnings
import weakref
from unittest.mock import patch

import fitsio
import numpy as np
import pytest

from weightmask.background import sky_to_mesh
from weightmask.cli import build_persistence_priors, load_configuration, run_pipeline
from weightmask.contract import ArtifactMetadata, ProducerMetadata, QualityBit, WeightMaskProduct, build_weight_product
from weightmask.mef import _clear_chip_replicas, _streak_catalog, _StreamingMapWriter, process_all_hdus, process_hdu
from weightmask.process import validate_config
from weightmask.reconstruct_sky import reconstruct_sky_fits


def _fake_product(*args, **kwargs):
    shape = (8, 8)
    mask = np.zeros(shape, np.uint32)
    plane = np.ones(shape, np.float32)
    components = {name: np.zeros(shape, bool) for name in ("bad", "sat", "cr", "obj", "streak", "nodata")}
    return mask, plane, plane, plane, plane, {"individual_masks": components}


@pytest.mark.parametrize("text", ["- a\n- b\n", "output_params: null\n", "output_params: []\n"])
def test_invalid_config_shapes_return_failure(tmp_path, text):
    config = tmp_path / "config.yml"
    config.write_text(text)
    assert load_configuration(str(config)) is None


def test_float16_fits_output_is_rejected():
    assert not validate_config({"output_params": {"ivar_bitpix": 16}})


@pytest.mark.parametrize("default,key", [(np.nan, "default_gain"), (0, "default_gain"), (-1, "default_rdnoise")])
def test_invalid_calibration_defaults_are_rejected(default, key):
    assert not validate_config({"variance": {key: default}})


@pytest.mark.parametrize("bad", [np.nan, np.inf, -1.0])
def test_invalid_calibration_headers_fall_back_before_cosmics(bad):
    from weightmask.process import process_image

    shape = (32, 32)
    image = np.random.default_rng(6).normal(100, 5, shape).astype(np.float32)
    config = {
        "variance": {"default_gain": 2.0, "default_rdnoise": 3.0},
        "saturation": {"effective_full_scale": 10000, "mask_bleed_trails": False},
    }
    with patch("weightmask.process.detect_cosmic_rays", return_value=np.zeros(shape, bool)) as cosmic:
        process_image(image, {"GAIN": bad, "RDNOISE": bad}, np.ones(shape, np.float32), config)
    assert cosmic.call_args.args[3:5] == (2.0, 3.0)


def test_output_failure_exits_nonzero(tmp_path):
    source = tmp_path / "science.fits"
    fitsio.write(str(source), np.full((8, 8), 1234, np.float32))
    config = tmp_path / "config.yml"
    config.write_text("{}\n")
    with patch("weightmask.mef.process_hdu", side_effect=_fake_product):
        code = run_pipeline([str(source), "--config", str(config), "-o", str(tmp_path / "missing" / "map.fits")])
    assert code != 0


@pytest.mark.parametrize("fault", ["shape", "nan", "nonbinary", "unreadable"])
def test_invalid_keep_map_fails_hdu_without_dropping_calibration(tmp_path, fault):
    from unittest.mock import Mock

    science, keep = tmp_path / "science.fits", tmp_path / "keep.fits"
    fitsio.write(str(science), np.ones((8, 8), np.float32))
    data = np.ones((4, 4) if fault == "shape" else (8, 8), np.float32)
    if fault == "nan":
        data[0, 0] = np.nan
    if fault == "nonbinary":
        data[0, 0] = 2
    fitsio.write(str(keep), data)
    with fitsio.FITS(str(science)) as sci, fitsio.FITS(str(keep)) as bpm:
        hdu = Mock() if fault == "unreadable" else bpm[0]
        if fault == "unreadable":
            hdu.read.side_effect = OSError("corrupt keep-map")
        with patch("weightmask.mef.process_image", side_effect=_fake_product) as process:
            result = process_hdu(sci[0], None, {}, 0, hdu_badpix=hdu)
        assert process.call_count == 0
        assert result == (None,) * 6


@pytest.mark.parametrize("alias", ["input", "symlink", "hardlink", "product"])
def test_output_aliases_are_rejected_without_clobbering(tmp_path, alias):
    source = tmp_path / "science.fits"
    fitsio.write(str(source), np.full((8, 8), 1234, np.float32))
    config = tmp_path / "config.yml"
    config.write_text("{}\n")
    out = source if alias == "input" else tmp_path / "map.fits"
    if alias == "symlink":
        out.symlink_to(source)
    if alias == "hardlink":
        out.hardlink_to(source)
    args = [str(source), "--config", str(config), "-o", str(out)]
    if alias == "product":
        args += ["--output_mask", str(out)]
    with patch("weightmask.mef.process_hdu", side_effect=_fake_product) as process:
        assert run_pipeline(args) != 0
    assert process.call_count == 0
    assert np.all(fitsio.read(str(source)) == 1234)


def test_missing_provenance_chunk_warns():
    header = ArtifactMetadata("weight", ProducerMetadata(), {"k": "v"}).to_header()
    del header["WMPV01"]
    with pytest.warns(RuntimeWarning, match="could not be decoded"):
        assert ArtifactMetadata.from_header(header).producer.version == "unknown"


def test_float32_overflow_and_underflow_are_invalid_variance():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        product = build_weight_product(np.array([1e300, 1e-300, 1.0]))
    assert np.array_equal(product.inverse_variance, [0, 0, 1])
    assert np.array_equal(product.quality_mask, [QualityBit.INVALID_VARIANCE, QualityBit.INVALID_VARIANCE, 0])


@pytest.mark.parametrize("ivar", [np.array([-1], np.int16), np.array([-1.0]), np.array([np.nan]), np.array([np.inf])])
def test_product_rejects_invalid_inverse_variance(ivar):
    with pytest.raises((TypeError, ValueError), match="inverse_variance"):
        WeightMaskProduct(np.zeros(1, np.uint32), ivar, np.zeros(1, np.float32), np.zeros(1, np.float32), {})


def test_reconstruct_scaled_mesh_does_not_scale_twice(tmp_path):
    mesh, cards = sky_to_mesh(np.full((32, 32), 123.4, np.float32), 8)
    source, output = tmp_path / "mesh.fits", tmp_path / "sky.fits"
    fitsio.write(str(source), mesh, header={**cards, "BZERO": 100, "BSCALE": 2})
    expected = fitsio.read(str(source))[0, 0]
    assert reconstruct_sky_fits(str(source), str(output)) == 0
    assert np.allclose(fitsio.read(str(output)), expected)


def test_reconstruction_streams_full_planes(tmp_path):
    mesh, cards = sky_to_mesh(np.ones((32, 32), np.float32), 8)
    source, output = tmp_path / "mesh.fits", tmp_path / "sky.fits"
    fitsio.write(str(source), None)
    with fitsio.FITS(str(source), "rw") as handle:
        handle.write(mesh, header=cards)
        handle.write(mesh, header=cards)
    previous = []

    def rebuild(*args):
        assert not any(reference() is not None for reference in previous)
        plane = np.ones((32, 32), np.float32)
        previous.append(weakref.ref(plane))
        return plane

    with patch("weightmask.reconstruct_sky.reconstruct_sky_from_header", side_effect=rebuild):
        assert reconstruct_sky_fits(str(source), str(output)) == 0


def test_reconstruction_write_failure_returns_exit_code(tmp_path):
    mesh, cards = sky_to_mesh(np.ones((32, 32), np.float32), 8)
    source, output = tmp_path / "mesh.fits", tmp_path / "sky.fits"
    fitsio.write(str(source), mesh, header=cards)
    with patch("weightmask.reconstruct_sky.fitsio.write", side_effect=OSError("disk full")):
        assert reconstruct_sky_fits(str(source), str(output)) != 0


def test_persistence_requires_distinct_exposures_and_accepts_later_compatible_hdu(tmp_path):
    source, other, mismatch = (tmp_path / name for name in ("science.fits", "other.fits", "mismatch.fits"))
    shape = (32, 24)
    image = np.zeros(shape, np.float32)
    image[:, 7] = 20
    fitsio.write(str(source), np.zeros(shape, np.float32), header={"CCDNAME": "C1"})
    fitsio.write(str(other), image, header={"CCDNAME": "C1"})
    fitsio.write(str(mismatch), np.zeros((12, 12), np.float32), header={"CCDNAME": "C1"})
    with fitsio.FITS(str(mismatch), "rw") as handle:
        handle.write(image, header={"CCDNAME": "C1"})
    assert not build_persistence_priors(str(source), [0], [str(other), str(other)])
    assert not build_persistence_priors(str(source), [0], [str(source), str(other)])
    prior = build_persistence_priors(str(source), [0], [str(mismatch), str(other)])
    assert np.all(prior[0][:, 7])


def test_compressed_writer_tracks_real_image_position(tmp_path):
    source, output = tmp_path / "science.fits", tmp_path / "map.fits.fz"
    image = np.ones((8, 8), np.float32)
    fitsio.write(str(source), image)
    with fitsio.FITS(str(source)) as hdul:
        writer = _StreamingMapWriter(str(output), hdul, {}, compress=True)
        writer.write(0, image, {}, "HDU0")
    with fitsio.FITS(str(output)) as handle:
        assert np.array_equal(handle[writer.positions[0]].read(), image)


@pytest.mark.parametrize("output_format", ["weight", "confidence"])
@pytest.mark.parametrize("mask_dtype", [np.uint16, np.uint64])
def test_chip_replica_restores_outputs_and_keeps_detected_exclusion(tmp_path, output_format, mask_dtype):
    shape = (32, 32)
    source = tmp_path / "science.fits"
    fitsio.write(str(source), np.ones(shape, np.float32))
    writers = {}
    catalogs = []
    with fitsio.FITS(str(source)) as hdul:
        for key in ("mask", "map", "invvar", "weight_raw"):
            writers[key] = _StreamingMapWriter(
                str(tmp_path / f"{key}.fits"), hdul, {}, dtype=mask_dtype if key == "mask" else np.float32
            )
        for index in (0, 1):
            mask = np.zeros(shape, np.uint32)
            mask[:, 12] = int(QualityBit.STREAK)
            mask[:, 14] = int(QualityBit.STREAK | QualityBit.DETECTED)
            ivar = np.ones(shape, np.float32)
            weight = ivar.copy()
            weight[:, [12, 14]] = 0
            catalogs.append(_streak_catalog(index, mask))
            for key, data in (("mask", mask), ("map", weight), ("weight_raw", weight), ("invvar", ivar)):
                writers[key].write(index, data, {}, str(index))
    _clear_chip_replicas(
        catalogs, writers, {"output_params": {"output_map_format": output_format, "mask_detected_in_weight": True}}
    )
    for key in ("map", "weight_raw"):
        with fitsio.FITS(writers[key].out_path) as handle:
            result = handle[writers[key].positions[0]].read()
        assert np.all(result[:, 12] == 1)
        assert np.all(result[:, 14] == 0)


@pytest.mark.parametrize("compress,scale", [(False, False), (True, True)])
def test_exposure_confidence_uses_unclipped_weights_and_configured_percentile(tmp_path, compress, scale):
    from argparse import Namespace

    source, output = tmp_path / "science.fits", tmp_path / "confidence.fits"
    shape = (32, 32)
    weights = [np.arange(1, 1025, dtype=np.float32).reshape(shape) * multiplier for multiplier in (1, 10)]
    fitsio.write(str(source), None)
    with fitsio.FITS(str(source), "rw") as handle:
        for _ in weights:
            handle.write(np.ones(shape, np.float32))
    paths = {
        "out_map_path": str(output),
        "out_mask_path": None,
        "out_invvar_path": None,
        "out_sky_path": None,
        "out_weight_raw_path": None,
    }
    config = {
        "output_params": {"output_map_format": "confidence", "compress": compress},
        "confidence_params": {"normalize_scope": "per_exposure", "normalize_percentile": 50, "scale_to_100": scale},
    }

    def product(_science, _flat, _config, index, **kwargs):
        weight = weights[index - 1]
        contract = build_weight_product(weight, confidence_percentile=50)
        return contract.quality_mask, contract.inverse_variance, weight, contract.confidence, weight, {}

    with fitsio.FITS(str(source)) as hdul, patch("weightmask.mef.process_hdu", side_effect=product):
        assert (
            process_all_hdus(
                [1, 2],
                hdul,
                None,
                config,
                paths,
                Namespace(individual_masks=False, max_workers=1),
                input_path=str(source),
            )
            == 2
        )
    normalization = np.percentile(np.concatenate([weight.ravel() for weight in weights]), 50)
    with fitsio.FITS(str(output)) as handle:
        for index, weight in enumerate(weights, 1):
            expected = np.clip(weight / normalization, 0, 1) * (100 if scale else 1)
            assert np.allclose(handle[index].read(), expected, atol=0.01 if compress else 1e-6)
