"""Small release regressions for IO safety and contract boundaries."""

import os
import warnings
import weakref
from unittest.mock import patch

import fitsio
import numpy as np
import pytest

from weightmask.background import sky_to_mesh
from weightmask.cli import build_persistence_priors, load_configuration, run_pipeline
from weightmask.contract import ArtifactMetadata, ProducerMetadata, QualityBit, WeightMaskProduct, build_weight_product
from weightmask.errors import StageFailure
from weightmask.mef import (
    _clear_chip_replicas,
    _ProductPublication,
    _rescale_confidence_to_global,
    _streak_catalog,
    _StreamingMapWriter,
    process_all_hdus,
    process_hdu,
)
from weightmask.process import validate_config
from weightmask.reconstruct_sky import reconstruct_sky_fits


def _fake_product(*args, **kwargs):
    shape = (8, 8)
    mask = np.zeros(shape, np.uint32)
    plane = np.ones(shape, np.float32)
    components = {name: np.zeros(shape, bool) for name in ("bad", "sat", "cr", "obj", "streak", "nodata")}
    return mask, plane, plane, plane, plane, {"individual_masks": components}


def _transaction_paths(tmp_path):
    return {
        "out_map_path": str(tmp_path / "weight.fits"),
        "out_mask_path": str(tmp_path / "mask.fits"),
        "out_invvar_path": str(tmp_path / "invvar.fits"),
        "out_sky_path": str(tmp_path / "sky.fits"),
        "out_weight_raw_path": None,
        "individual_mask_paths": {},
    }


def _prepopulate_products(paths):
    snapshots = {}
    for index, path in enumerate(path for path in paths.values() if isinstance(path, str)):
        fitsio.write(path, np.full((3, 3), index + 11, np.float32), header={"OLDGEN": index}, clobber=True)
        snapshots[path] = open(path, "rb").read()
    return snapshots


def _assert_products_unchanged(snapshots):
    for path, expected in snapshots.items():
        assert open(path, "rb").read() == expected
    directory = os.path.dirname(next(iter(snapshots)))
    assert not [name for name in os.listdir(directory) if ".wm-tmp-" in name or ".wm-bak-" in name]


def _run_fake_product_set(source, paths, *, config=None):
    from argparse import Namespace

    with fitsio.FITS(str(source)) as hdul:
        with patch("weightmask.mef.process_hdu", side_effect=_fake_product):
            return process_all_hdus(
                [0],
                hdul,
                None,
                config or {},
                paths,
                Namespace(individual_masks=False, max_workers=1),
                input_path=str(source),
            )


@pytest.mark.parametrize("text", ["- a\n- b\n", "output_params: null\n", "output_params: []\n"])
def test_invalid_config_shapes_return_failure(tmp_path, text):
    config = tmp_path / "config.yml"
    config.write_text(text)
    assert load_configuration(str(config)) is None


def test_invalid_config_is_rejected_before_fits_input_validation(tmp_path):
    config = tmp_path / "config.yml"
    config.write_text("sep_objects:\n  min_are: 3\n")

    with (
        patch("weightmask.cli.validate_input_files") as validate_inputs,
        patch("weightmask.cli.fitsio.FITS") as open_fits,
    ):
        assert run_pipeline(["science.fits", "--config", str(config)]) != 0

    validate_inputs.assert_not_called()
    open_fits.assert_not_called()


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


def test_failure_before_first_product_write_preserves_existing_generation(tmp_path):
    source = tmp_path / "science.fits"
    fitsio.write(str(source), np.ones((8, 8), np.float32))
    paths = _transaction_paths(tmp_path)
    snapshots = _prepopulate_products(paths)

    with patch("weightmask.mef.process_hdu", side_effect=StageFailure("objects.main", "failed before write")):
        with fitsio.FITS(str(source)) as hdul:
            with pytest.raises(StageFailure):
                process_all_hdus(
                    [0],
                    hdul,
                    None,
                    {},
                    paths,
                    __import__("argparse").Namespace(individual_masks=False, max_workers=1),
                    input_path=str(source),
                )

    _assert_products_unchanged(snapshots)


def test_failure_between_product_writes_preserves_existing_generation(tmp_path):
    source = tmp_path / "science.fits"
    fitsio.write(str(source), np.ones((8, 8), np.float32))
    paths = _transaction_paths(tmp_path)
    snapshots = _prepopulate_products(paths)
    real_write = _StreamingMapWriter.write
    calls = 0

    def fail_second_write(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full between products")
        return real_write(self, *args, **kwargs)

    with patch.object(_StreamingMapWriter, "write", new=fail_second_write):
        assert _run_fake_product_set(source, paths) == 0

    _assert_products_unchanged(snapshots)


def test_postprocessing_failure_preserves_existing_generation(tmp_path):
    source = tmp_path / "science.fits"
    fitsio.write(str(source), np.ones((8, 8), np.float32))
    paths = _transaction_paths(tmp_path)
    snapshots = _prepopulate_products(paths)
    config = {
        "output_params": {"output_map_format": "confidence"},
        "confidence_params": {"normalize_scope": "per_exposure"},
    }

    with patch("weightmask.mef._rescale_confidence_to_global", return_value=False):
        assert _run_fake_product_set(source, paths, config=config) == 0

    _assert_products_unchanged(snapshots)


def test_promotion_failure_rolls_back_every_existing_product(tmp_path):
    source = tmp_path / "science.fits"
    fitsio.write(str(source), np.ones((8, 8), np.float32))
    paths = _transaction_paths(tmp_path)
    snapshots = _prepopulate_products(paths)
    real_replace = os.replace
    real_unlink = os.unlink
    calls = 0

    def fail_during_promotion(source_path, destination_path):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("promotion interrupted")
        return real_replace(source_path, destination_path)

    def reject_unlink_of_published_path(path):
        if os.fspath(path) in snapshots:
            raise PermissionError("published destination cannot be unlinked")
        real_unlink(path)

    with (
        patch("weightmask.mef.os.replace", side_effect=fail_during_promotion),
        patch("weightmask.mef.os.unlink", side_effect=reject_unlink_of_published_path),
    ):
        assert _run_fake_product_set(source, paths) == 0

    assert calls >= 4
    _assert_products_unchanged(snapshots)


def test_backup_cleanup_failure_keeps_successful_product_set_published(tmp_path, capsys):
    source = tmp_path / "science.fits"
    fitsio.write(str(source), np.ones((8, 8), np.float32))
    paths = _transaction_paths(tmp_path)
    snapshots = _prepopulate_products(paths)
    real_unlink = os.unlink

    def reject_backup_unlink(path):
        if ".wm-bak-" in os.fspath(path):
            raise PermissionError("backup is busy")
        real_unlink(path)

    with patch("weightmask.mef.os.unlink", side_effect=reject_backup_unlink):
        assert _run_fake_product_set(source, paths) == 1

    assert "backup is busy" in capsys.readouterr().out
    assert all(open(path, "rb").read() != previous for path, previous in snapshots.items())
    assert not [path for path in tmp_path.iterdir() if ".wm-tmp-" in path.name]
    assert len([path for path in tmp_path.iterdir() if ".wm-bak-" in path.name]) == len(snapshots)


def test_incomplete_rollback_keeps_promotion_error_as_exception_cause(tmp_path):
    paths = {
        "out_map_path": str(tmp_path / "weight.fits"),
        "out_mask_path": str(tmp_path / "mask.fits"),
    }
    publication = _ProductPublication(paths)
    for path in paths.values():
        open(path, "wb").write(b"old")
    for key in paths:
        open(publication.paths[key], "wb").write(b"new")
    real_replace = os.replace
    promotion_error = OSError("promotion interrupted")
    rollback_error = OSError("rollback interrupted")
    calls = 0

    def fail_promotion_and_rollback(source_path, destination_path):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise promotion_error
        if calls == 5:
            raise rollback_error
        return real_replace(source_path, destination_path)

    with patch("weightmask.mef.os.replace", side_effect=fail_promotion_and_rollback):
        with pytest.raises(OSError, match="rollback was incomplete") as raised:
            publication.promote()

    assert raised.value.__cause__ is promotion_error


def test_successful_product_set_has_one_generation_and_synchronized_structure(tmp_path):
    from argparse import Namespace

    source = tmp_path / "science.fits"
    fitsio.write(str(source), None)
    with fitsio.FITS(str(source), "rw") as handle:
        handle.write(np.ones((8, 8), np.float32), header={"CCDNAME": "C1"})
        handle.write(np.ones((8, 8), np.float32), header={"CCDNAME": "C2"})
    paths = _transaction_paths(tmp_path)

    with fitsio.FITS(str(source)) as hdul, patch("weightmask.mef.process_hdu", side_effect=_fake_product):
        count = process_all_hdus(
            [1, 2],
            hdul,
            None,
            {},
            paths,
            Namespace(individual_masks=False, max_workers=1),
            input_path=str(source),
        )

    assert count == 2
    generations = set()
    structures = []
    for path in (value for value in paths.values() if isinstance(value, str)):
        with fitsio.FITS(path) as handle:
            images = [item for item in handle if item.get_info().get("ndims") == 2]
            structures.append([(item.read().shape, item.read_header().get("EXTNAME")) for item in images])
            for item in images:
                header = item.read_header()
                generations.add(header.get("WMGENID"))
                assert header.get("WMVERS")
                assert header.get("WMART")
    assert None not in generations
    assert len(generations) == 1
    assert all([shape for shape, _name in structure] == [(8, 8), (8, 8)] for structure in structures)
    assert all(all(name for _shape, name in structure) for structure in structures)


def test_detector_failure_exits_nonzero_without_output(tmp_path, capsys):
    source = tmp_path / "science.fits"
    output = tmp_path / "weight.fits"
    config = tmp_path / "config.yml"
    fitsio.write(str(source), np.full((32, 32), 1234, np.float32))
    config.write_text(
        "cosmic_ray:\n"
        "  niter: 1\n"
        "  faint_cr:\n"
        "    enable: false\n"
        "sep_background:\n"
        "  iterations: 0\n"
        "saturation:\n"
        "  mask_bleed_trails: false\n"
    )

    with patch("weightmask.cosmics.detect_cosmics", side_effect=OSError("broken backend")):
        code = run_pipeline([str(source), "--config", str(config), "-o", str(output)])

    log = capsys.readouterr().out
    assert code != 0
    assert not output.exists()
    assert "HDU 0" in log
    assert "cosmic_ray.primary" in log
    assert "broken backend" in log
    assert "Pipeline finished" not in log


def test_cli_converts_propagated_stage_failure_to_nonzero(tmp_path):
    source = tmp_path / "science.fits"
    output = tmp_path / "weight.fits"
    config = tmp_path / "config.yml"
    fitsio.write(str(source), np.ones((8, 8), np.float32))
    config.write_text("{}\n")

    with patch("weightmask.cli.process_all_hdus", side_effect=StageFailure("objects.main", "broken SEP")):
        code = run_pipeline([str(source), "--config", str(config), "-o", str(output)])

    assert code != 0
    assert not output.exists()


def test_failed_detector_hdu_propagates_after_prior_output(tmp_path):
    from argparse import Namespace

    source = tmp_path / "science.fits"
    output = tmp_path / "weight.fits"
    fitsio.write(str(source), None)
    with fitsio.FITS(str(source), "rw") as handle:
        handle.write(np.ones((8, 8), np.float32), extname="GOOD")
        handle.write(np.ones((8, 8), np.float32), extname="FAILED")
    paths = {
        "out_map_path": str(output),
        "out_mask_path": None,
        "out_invvar_path": None,
        "out_sky_path": None,
        "out_weight_raw_path": None,
    }

    cause = OSError("broken SEP")
    failure = StageFailure("objects.main", cause)
    failure.__cause__ = cause
    with fitsio.FITS(str(source)) as hdul:
        with patch(
            "weightmask.mef.process_hdu",
            side_effect=[_fake_product(), failure],
        ):
            with pytest.raises(StageFailure) as raised:
                process_all_hdus(
                    [1, 2],
                    hdul,
                    None,
                    {},
                    paths,
                    Namespace(individual_masks=False, max_workers=1),
                    input_path=str(source),
                )

    assert raised.value is failure
    assert raised.value.stage == "objects.main"
    assert raised.value.detail == "broken SEP"
    assert raised.value.__cause__ is cause
    assert raised.value.hdu_index == 2
    assert not output.exists()


def test_parallel_stage_failure_cancels_unsubmitted_hdus_and_writes_nothing(tmp_path):
    from argparse import Namespace

    source = tmp_path / "science.fits"
    output = tmp_path / "weight.fits"
    fitsio.write(str(source), None)
    with fitsio.FITS(str(source), "rw") as handle:
        for index in range(1, 4):
            handle.write(np.ones((8, 8), np.float32), extname=f"SCI{index}")
    paths = {
        "out_map_path": str(output),
        "out_mask_path": None,
        "out_invvar_path": None,
        "out_sky_path": None,
        "out_weight_raw_path": None,
    }
    calls = []
    failure = StageFailure("cosmic_ray.primary", "parallel failure")

    def product(_science, _flat, _config, index, **kwargs):
        calls.append(index)
        if index == 1:
            raise failure
        return _fake_product()

    with fitsio.FITS(str(source)) as hdul, patch("weightmask.mef.process_hdu", side_effect=product):
        with pytest.raises(StageFailure) as raised:
            process_all_hdus(
                [1, 2, 3],
                hdul,
                None,
                {},
                paths,
                Namespace(individual_masks=False, max_workers=2),
                input_path=str(source),
            )

    assert raised.value is failure
    assert raised.value.hdu_index == 1
    assert 3 not in calls
    assert not output.exists()


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


def test_future_uint32_quality_bits_survive_release_contract_boundary():
    mask = np.array([np.uint32(1 << 31), np.uint32((1 << 31) | 1)], dtype=np.uint32)

    product = build_weight_product(np.ones(2, dtype=np.float32), mask)

    np.testing.assert_array_equal(product.quality_mask, mask)
    assert product.quality_mask.dtype == np.uint32


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


@pytest.mark.parametrize(
    "malformation",
    ["missing_box_height", "unequal_boxes", "zero_box", "negative_output", "truncated", "extra", "nonfinite", "rank"],
)
def test_malformed_sky_mesh_fails_before_output_creation(tmp_path, malformation):
    mesh, cards = sky_to_mesh(np.ones((32, 32), np.float32), 8)
    if malformation == "missing_box_height":
        del cards["MESHBH"]
    elif malformation == "unequal_boxes":
        cards["MESHBH"] = 4
    elif malformation == "zero_box":
        cards["MESHBW"] = cards["MESHBH"] = 0
    elif malformation == "negative_output":
        cards["SKYH"] = -1
    elif malformation == "truncated":
        mesh = mesh[:-1]
    elif malformation == "extra":
        mesh = np.pad(mesh, ((0, 1), (0, 0)))
    elif malformation == "nonfinite":
        mesh[0, 0] = np.nan
    else:
        mesh = mesh.reshape(-1)
    source, output = tmp_path / "mesh.fits", tmp_path / "sky.fits"
    fitsio.write(str(source), mesh, header=cards)

    assert reconstruct_sky_fits(str(source), str(output)) != 0
    assert not output.exists()


def test_all_sky_mesh_hdus_are_validated_before_output_creation(tmp_path):
    mesh, cards = sky_to_mesh(np.ones((32, 32), np.float32), 8)
    source, output = tmp_path / "mesh.fits", tmp_path / "sky.fits"
    fitsio.write(str(source), None)
    with fitsio.FITS(str(source), "rw") as handle:
        handle.write(mesh, header=cards)
        handle.write(mesh[:-1], header=cards)

    assert reconstruct_sky_fits(str(source), str(output)) != 0
    assert not output.exists()


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


def test_reconstruction_failure_preserves_existing_destination(tmp_path):
    mesh, cards = sky_to_mesh(np.ones((32, 32), np.float32), 8)
    source, output = tmp_path / "mesh.fits", tmp_path / "sky.fits"
    fitsio.write(str(source), None)
    with fitsio.FITS(str(source), "rw") as handle:
        handle.write(mesh, header=cards, extname="SKY_C1")
        handle.write(mesh, header=cards, extname="SKY_C2")
    fitsio.write(str(output), np.full((4, 4), 19, np.float32), header={"OLDGEN": 1})
    snapshot = output.read_bytes()

    with patch(
        "weightmask.reconstruct_sky.reconstruct_sky_from_header",
        side_effect=[np.ones((32, 32), np.float32), OSError("reconstruction failed")],
    ):
        assert reconstruct_sky_fits(str(source), str(output)) != 0

    assert output.read_bytes() == snapshot
    assert not [path for path in tmp_path.iterdir() if ".wm-tmp-" in path.name or ".wm-bak-" in path.name]


def test_reconstruction_backup_cleanup_failure_reports_success(tmp_path, capsys):
    mesh, cards = sky_to_mesh(np.ones((32, 32), np.float32), 8)
    source, output = tmp_path / "mesh.fits", tmp_path / "sky.fits"
    fitsio.write(str(source), mesh, header=cards)
    fitsio.write(str(output), np.full((4, 4), 19, np.float32), header={"OLDGEN": 1})
    snapshot = output.read_bytes()
    real_unlink = os.unlink

    def reject_backup_unlink(path):
        if ".wm-bak-" in os.fspath(path):
            raise PermissionError("reconstruction backup is busy")
        real_unlink(path)

    with patch("weightmask.mef.os.unlink", side_effect=reject_backup_unlink):
        assert reconstruct_sky_fits(str(source), str(output)) == 0

    assert "reconstruction backup is busy" in capsys.readouterr().out
    assert output.read_bytes() != snapshot
    assert fitsio.read(str(output)).shape == (32, 32)
    assert not [path for path in tmp_path.iterdir() if ".wm-tmp-" in path.name]
    assert len([path for path in tmp_path.iterdir() if ".wm-bak-" in path.name]) == 1


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


def test_persistence_construction_releases_each_ccd_after_profiling(tmp_path, monkeypatch):
    source = tmp_path / "science.fits"
    shape = (32, 24)
    fitsio.write(str(source), np.zeros(shape, np.float32), header={"CCDNAME": "C1"})
    others = []
    for index in range(3):
        other = tmp_path / f"other-{index}.fits"
        fitsio.write(str(other), np.zeros(shape, np.float32), header={"CCDNAME": "C1"})
        others.append(str(other))

    references = []
    ascontiguousarray = np.ascontiguousarray

    def track(array, *args, **kwargs):
        assert all(reference() is None for reference in references)
        result = ascontiguousarray(array, *args, **kwargs)
        references.append(weakref.ref(result))
        return result

    monkeypatch.setattr(np, "ascontiguousarray", track)
    priors = build_persistence_priors(str(source), [0], others)

    assert 0 in priors
    assert all(reference() is None for reference in references)


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
    shape = (64, 64)
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
            mask[:, 48] = int(QualityBit.STREAK | QualityBit.DETECTED)
            ivar = np.ones(shape, np.float32)
            weight = ivar.copy()
            weight[:, [12, 48]] = 0
            header = {
                "CTYPE1": "RA---TAN",
                "CTYPE2": "DEC--TAN",
                "CRVAL1": 150.0,
                "CRVAL2": 2.0,
                "CRPIX1": 1.0 + 100.0 * index,
                "CRPIX2": 1.0,
                "CD1_1": 1.0e-4,
                "CD1_2": 0.0,
                "CD2_1": 0.0,
                "CD2_2": 1.0e-4,
            }
            catalogs.extend(_streak_catalog(index, mask, header))
            for key, data in (("mask", mask), ("map", weight), ("weight_raw", weight), ("invvar", ivar)):
                writers[key].write(index, data, {}, str(index))
    _clear_chip_replicas(
        catalogs, writers, {"output_params": {"output_map_format": output_format, "mask_detected_in_weight": True}}
    )
    for key in ("map", "weight_raw"):
        with fitsio.FITS(writers[key].out_path) as handle:
            result = handle[writers[key].positions[0]].read()
        assert np.all(result[:, 12] == 1)
        assert np.all(result[:, 48] == 0)


@pytest.mark.parametrize("output_format,restored_map", [("weight", 4.0), ("confidence", 1.0)])
def test_chip_replica_transaction_synchronizes_every_affected_product(tmp_path, output_format, restored_map):
    from argparse import Namespace

    shape = (64, 64)
    source = tmp_path / "science.fits"
    header = {
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRVAL1": 150.0,
        "CRVAL2": 2.0,
        "CRPIX1": 1.0,
        "CRPIX2": 1.0,
        "CD1_1": 1.0e-4,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 1.0e-4,
    }
    fitsio.write(str(source), None)
    with fitsio.FITS(str(source), "rw") as handle:
        handle.write(np.ones(shape, np.float32), header=header, extname="C0")
        handle.write(np.ones(shape, np.float32), header={**header, "CRPIX1": 101.0}, extname="C1")
    paths = {
        "out_map_path": str(tmp_path / "map.fits"),
        "out_mask_path": str(tmp_path / "mask.fits"),
        "out_invvar_path": str(tmp_path / "ivar.fits"),
        "out_sky_path": None,
        "out_weight_raw_path": str(tmp_path / "raw.fits"),
        "individual_mask_paths": {"streak": str(tmp_path / "streak.fits")},
    }
    masks = []
    for index in range(2):
        mask = np.zeros(shape, np.uint32)
        mask[:, 12] = int(QualityBit.STREAK)
        if index == 0:
            mask[np.arange(30), np.arange(30) + 30] = int(QualityBit.STREAK)
        masks.append(mask)

    def product(_science, _flat, _config, index, **_kwargs):
        mask = masks[index - 1]
        ivar = np.full(shape, 4.0, np.float32)
        weight = ivar.copy()
        weight[mask != 0] = 0
        confidence = (weight > 0).astype(np.float32)
        individual = {name: np.zeros(shape, bool) for name in ("bad", "sat", "cr", "obj", "nodata")}
        individual["streak"] = mask != 0
        return mask, ivar, weight, confidence, np.zeros(shape, np.float32), {"individual_masks": individual}

    config = {
        "output_params": {"output_map_format": output_format},
        "confidence_params": {"normalize_scope": "per_hdu", "normalize_percentile": 99.0},
    }
    with fitsio.FITS(str(source)) as hdul, patch("weightmask.mef.process_hdu", side_effect=product):
        assert (
            process_all_hdus(
                [1, 2],
                hdul,
                None,
                config,
                paths,
                Namespace(individual_masks=True, max_workers=1),
                input_path=str(source),
            )
            == 2
        )

    products = {}
    for name, path in (
        ("map", paths["out_map_path"]),
        ("mask", paths["out_mask_path"]),
        ("ivar", paths["out_invvar_path"]),
        ("raw", paths["out_weight_raw_path"]),
        ("streak", paths["individual_mask_paths"]["streak"]),
    ):
        with fitsio.FITS(path) as handle:
            products[name] = [item.read() for item in handle if item.get_info().get("ndims") == 2]
    for index in range(2):
        assert np.all(products["mask"][index][:, 12] & int(QualityBit.STREAK) == 0)
        assert np.all(products["map"][index][:, 12] == restored_map)
        assert np.all(products["raw"][index][:, 12] == 4.0)
        assert np.all(products["ivar"][index][:, 12] == 4.0)
        assert np.all(products["streak"][index][:, 12] == 0)
    unique = (np.arange(30), np.arange(30) + 30)
    assert np.all(products["mask"][0][unique] & int(QualityBit.STREAK))
    assert np.all(products["map"][0][unique] == 0)
    assert np.all(products["raw"][0][unique] == 0)
    assert np.all(products["streak"][0][unique] == 1)


def test_missing_individual_streak_position_aborts_transaction(tmp_path):
    from argparse import Namespace

    shape = (32, 32)
    source = tmp_path / "science.fits"
    header = {
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRVAL1": 150.0,
        "CRVAL2": 2.0,
        "CRPIX1": 1.0,
        "CRPIX2": 1.0,
        "CD1_1": 1.0e-4,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 1.0e-4,
    }
    fitsio.write(str(source), None)
    with fitsio.FITS(str(source), "rw") as handle:
        handle.write(np.ones(shape, np.float32), header=header, extname="C0")
        handle.write(np.ones(shape, np.float32), header={**header, "CRPIX1": 101.0}, extname="C1")
    paths = {
        "out_map_path": str(tmp_path / "map.fits"),
        "out_mask_path": str(tmp_path / "mask.fits"),
        "out_invvar_path": str(tmp_path / "ivar.fits"),
        "out_sky_path": None,
        "out_weight_raw_path": str(tmp_path / "raw.fits"),
        "individual_mask_paths": {"streak": str(tmp_path / "streak.fits")},
    }

    def product(_science, _flat, _config, _index, **_kwargs):
        mask = np.zeros(shape, np.uint32)
        mask[:, 12] = int(QualityBit.STREAK)
        ivar = np.full(shape, 4.0, np.float32)
        weight = ivar.copy()
        weight[:, 12] = 0
        individual = {name: np.zeros(shape, bool) for name in ("bad", "sat", "cr", "obj", "nodata")}
        individual["streak"] = mask != 0
        return mask, ivar, weight, weight, np.zeros(shape, np.float32), {"individual_masks": individual}

    real_clear = _clear_chip_replicas

    def desynchronize(catalogs, writers, config):
        writers["ind_streak"].positions.pop(1)
        return real_clear(catalogs, writers, config)

    with (
        fitsio.FITS(str(source)) as hdul,
        patch("weightmask.mef.process_hdu", side_effect=product),
        patch("weightmask.mef._clear_chip_replicas", side_effect=desynchronize),
    ):
        assert (
            process_all_hdus(
                [1, 2],
                hdul,
                None,
                {"output_params": {"output_map_format": "weight"}},
                paths,
                Namespace(individual_masks=True, max_workers=1),
                input_path=str(source),
            )
            == 0
        )

    requested = [path for key, path in paths.items() if key != "individual_mask_paths" and path]
    requested.extend(paths["individual_mask_paths"].values())
    assert not any(os.path.exists(path) for path in requested)


@pytest.mark.parametrize("fault", ["missing_position", "shape_mismatch"])
def test_chip_replica_rejects_unsynchronized_individual_streak_before_mutation(tmp_path, fault):
    shape = (32, 32)
    source = tmp_path / "science.fits"
    fitsio.write(str(source), np.ones(shape, np.float32))
    writers = {}
    catalogs = []
    header = {
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRVAL1": 150.0,
        "CRVAL2": 2.0,
        "CRPIX1": 1.0,
        "CRPIX2": 1.0,
        "CD1_1": 1.0e-4,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 1.0e-4,
    }
    with fitsio.FITS(str(source)) as hdul:
        for key in ("mask", "map", "invvar", "weight_raw", "ind_streak"):
            dtype = np.uint32 if key == "mask" else np.uint8 if key == "ind_streak" else np.float32
            writers[key] = _StreamingMapWriter(str(tmp_path / f"{key}.fits"), hdul, {}, dtype=dtype)
        for index in (0, 1):
            mask = np.zeros(shape, np.uint32)
            mask[:, 12] = int(QualityBit.STREAK)
            ivar = np.full(shape, 4.0, np.float32)
            weight = ivar.copy()
            weight[:, 12] = 0
            individual_streak = (mask != 0).astype(np.uint8)
            if fault == "shape_mismatch" and index == 0:
                individual_streak = individual_streak[:-1]
            catalogs.extend(_streak_catalog(index, mask, {**header, "CRPIX1": 1.0 + 100.0 * index}))
            for key, data in (
                ("mask", mask),
                ("map", weight),
                ("invvar", ivar),
                ("weight_raw", weight),
                ("ind_streak", individual_streak),
            ):
                writers[key].write(index, data, {}, str(index))
    if fault == "missing_position":
        writers["ind_streak"].positions.pop(0)
    before = {}
    for key, writer in writers.items():
        with fitsio.FITS(writer.out_path) as handle:
            before[key] = [item.read() for item in handle if item.get_info().get("ndims") == 2]

    assert not _clear_chip_replicas(catalogs, writers, {"output_params": {"output_map_format": "weight"}})

    for key, writer in writers.items():
        with fitsio.FITS(writer.out_path) as handle:
            after = [item.read() for item in handle if item.get_info().get("ndims") == 2]
        assert len(after) == len(before[key])
        for actual, expected in zip(after, before[key]):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "population_case,compress,scale",
    [
        ("balanced", False, False),
        ("balanced", True, True),
        ("unequal_sizes", False, False),
        ("unequal_valid_fractions", False, False),
    ],
)
def test_exposure_confidence_matches_exact_pooled_percentile(tmp_path, population_case, compress, scale):
    from argparse import Namespace

    source, output = tmp_path / "science.fits", tmp_path / "confidence.fits"
    if population_case == "balanced":
        weights = [np.arange(1, 1025, dtype=np.float32).reshape(32, 32) * multiplier for multiplier in (1, 10)]
    elif population_case == "unequal_sizes":
        weights = [
            np.arange(1, 24_001, dtype=np.float32).reshape(120, 200),
            np.full((10, 10), 100_000, dtype=np.float32),
        ]
    else:
        dense = np.arange(1, 22_501, dtype=np.float32).reshape(150, 150)
        sparse = np.zeros((150, 150), dtype=np.float32)
        sparse.ravel()[:100] = 100_000
        weights = [dense, sparse]
    fitsio.write(str(source), None)
    with fitsio.FITS(str(source), "rw") as handle:
        for weight in weights:
            handle.write(np.ones(weight.shape, np.float32))
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
    normalization = np.percentile(np.concatenate([weight[weight > 0] for weight in weights]), 50)
    with fitsio.FITS(str(output)) as handle:
        for index, weight in enumerate(weights, 1):
            expected = np.clip(weight / normalization, 0, 1) * (100 if scale else 1)
            assert np.allclose(handle[index].read(), expected, atol=0.01 if compress else 1e-6)


def test_exposure_confidence_reads_each_hdu_once_for_sampling_and_once_for_writing(monkeypatch):
    class ImageHDU:
        def __init__(self, data):
            self.data = data
            self.read_count = 0

        def read(self):
            self.read_count += 1
            return self.data.copy()

        def write(self, data):
            self.data = data

    class FitsHandle:
        def __init__(self, hdus):
            self.hdus = hdus

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __getitem__(self, position):
            return self.hdus[position]

    hdus = {
        1: ImageHDU(np.arange(1, 120_001, dtype=np.float32).reshape(300, 400)),
        2: ImageHDU(np.array([[0, np.nan, np.inf, 2]], dtype=np.float32)),
    }
    handle = FitsHandle(hdus)
    writer = type(
        "Writer",
        (),
        {
            "positions": {9: 1, 4: 2},
            "identities": {9: "MAP_CCD-A", 4: "MAP_CCD-B"},
            "dtype": np.dtype(np.float32),
        },
    )()
    config = {
        "output_params": {"output_map_format": "confidence"},
        "confidence_params": {"normalize_percentile": 50},
    }

    monkeypatch.setattr("weightmask.mef.fitsio.FITS", lambda *_args, **_kwargs: handle)

    assert _rescale_confidence_to_global({"out_map_path": "unused.fits"}, {"map": writer}, config)
    assert [hdus[position].read_count for position in (1, 2)] == [2, 2]
    assert np.isfinite(hdus[1].data).all()


def test_exposure_confidence_empty_population_needs_no_write_pass(monkeypatch):
    class EmptyHDU:
        def __init__(self):
            self.read_count = 0

        def read(self):
            self.read_count += 1
            return np.zeros((4, 4), dtype=np.float32)

        def write(self, _data):
            raise AssertionError("empty confidence population must not be rewritten")

    class FitsHandle:
        def __init__(self, hdu):
            self.hdu = hdu

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __getitem__(self, _position):
            return self.hdu

    hdu = EmptyHDU()
    writer = type(
        "Writer",
        (),
        {"positions": {1: 0}, "identities": {1: "MAP_EMPTY"}, "dtype": np.dtype(np.float32)},
    )()
    config = {
        "output_params": {"output_map_format": "confidence"},
        "confidence_params": {"normalize_percentile": 50},
    }

    monkeypatch.setattr("weightmask.mef.fitsio.FITS", lambda *_args, **_kwargs: FitsHandle(hdu))

    assert _rescale_confidence_to_global({"out_map_path": "unused.fits"}, {"map": writer}, config)
    assert hdu.read_count == 1


def test_exposure_confidence_is_invariant_to_reordered_hdus_above_cap():
    class ImageHDU:
        def __init__(self, data):
            self.data = data.copy()

        def read(self):
            return self.data.copy()

        def write(self, data):
            self.data = data

    class FitsHandle:
        def __init__(self, hdus):
            self.hdus = hdus

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __getitem__(self, position):
            return self.hdus[position]

    populations = {
        "MAP_CCD-A": np.ones(50_500, dtype=np.float32),
        "MAP_CCD-B": np.full(50_501, 100, dtype=np.float32),
    }
    config = {
        "output_params": {"output_map_format": "confidence"},
        "confidence_params": {"normalize_percentile": 50},
    }

    def normalize(order):
        hdus = {position: ImageHDU(populations[identity]) for position, identity in enumerate(order, 1)}
        writer = type(
            "Writer",
            (),
            {
                "positions": {hdu_index: position for hdu_index, position in enumerate(hdus, 1)},
                "identities": {hdu_index: identity for hdu_index, identity in enumerate(order, 1)},
                "dtype": np.dtype(np.float32),
            },
        )()
        with patch("weightmask.mef.fitsio.FITS", return_value=FitsHandle(hdus)):
            assert _rescale_confidence_to_global({"out_map_path": "unused.fits"}, {"map": writer}, config)
        return {identity: hdus[position].data for position, identity in enumerate(order, 1)}

    forward = normalize(("MAP_CCD-A", "MAP_CCD-B"))
    reversed_order = normalize(("MAP_CCD-B", "MAP_CCD-A"))

    np.testing.assert_array_equal(forward["MAP_CCD-A"], reversed_order["MAP_CCD-A"])
    np.testing.assert_array_equal(forward["MAP_CCD-B"], reversed_order["MAP_CCD-B"])


def test_exposure_confidence_rejects_duplicate_hdu_identities_before_sampling(monkeypatch):
    class ImageHDU:
        def read(self):
            raise AssertionError("duplicate identities must fail before sampling")

    class FitsHandle:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def __getitem__(self, _position):
            return ImageHDU()

    writer = type(
        "Writer",
        (),
        {
            "positions": {1: 1, 2: 2},
            "identities": {1: "MAP_DUPLICATE", 2: "MAP_DUPLICATE"},
            "dtype": np.dtype(np.float32),
        },
    )()
    config = {
        "output_params": {"output_map_format": "confidence"},
        "confidence_params": {"normalize_percentile": 50},
    }
    monkeypatch.setattr("weightmask.mef.fitsio.FITS", lambda *_args, **_kwargs: FitsHandle())

    with pytest.raises(ValueError, match="duplicate HDU identity"):
        _rescale_confidence_to_global({"out_map_path": "unused.fits"}, {"map": writer}, config)
