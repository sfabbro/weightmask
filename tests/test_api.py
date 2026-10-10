from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import yaml

import weightmask
import weightmask.background as background
import weightmask.config as config_module
import weightmask.reconstruct_sky as reconstruct_sky
import weightmask.torchfits_adapter as torchfits_adapter

ROOT = Path(__file__).resolve().parents[1]


def test_root_all_is_exact_and_does_not_grow_with_module_apis():
    assert weightmask.__all__ == ["MASK_BITS", "MASK_DTYPE", "QUALITY_BITS", "__version__"]
    assert set(weightmask.__all__) == {"MASK_BITS", "MASK_DTYPE", "QUALITY_BITS", "__version__"}
    assert config_module.__all__ == [
        "clean_config_dict",
        "copy_default_config",
        "default_config_bytes",
        "default_config_path",
        "default_config_resource",
        "default_config_sha256",
        "default_config_text",
    ]


def test_supported_reconstruction_and_adapter_exports_are_explicit():
    assert background.__all__ == [
        "sky_to_mesh",
        "reconstruct_sky_mesh",
        "parse_sky_mesh_header",
        "reconstruct_sky_from_header",
    ]
    assert torchfits_adapter.__all__ == [
        "TorchfitsArrayHeaderIO",
        "TorchfitsUnavailableError",
        "torchfits_available",
    ]
    assert reconstruct_sky.__all__ == ["reconstruct_sky_fits", "main"]
    assert not hasattr(weightmask, "sky_to_mesh")
    assert not hasattr(weightmask, "reconstruct_sky_mesh")
    assert not hasattr(weightmask, "TorchfitsArrayHeaderIO")


def test_supported_reconstruction_import_paths_are_importable():
    from weightmask.background import (
        parse_sky_mesh_header,
        reconstruct_sky_from_header,
        reconstruct_sky_mesh,
        sky_to_mesh,
    )
    from weightmask.config import (
        clean_config_dict,
        copy_default_config,
        default_config_bytes,
        default_config_path,
        default_config_resource,
        default_config_sha256,
        default_config_text,
    )
    from weightmask.contract import (
        INVERSE_VARIANCE_SEMANTICS,
        MASK_POLARITY,
        QUALITY_BITS,
        ArrayHeaderIO,
        WeightMaskProduct,
        build_weight_product,
    )
    from weightmask.process import process_image, validate_config
    from weightmask.reconstruct_sky import main, reconstruct_sky_fits
    from weightmask.torchfits_adapter import (
        TorchfitsArrayHeaderIO,
        TorchfitsUnavailableError,
        torchfits_available,
    )

    assert callable(parse_sky_mesh_header)
    assert callable(reconstruct_sky_from_header)
    assert callable(reconstruct_sky_mesh)
    assert callable(sky_to_mesh)
    assert callable(reconstruct_sky_fits)
    assert callable(main)
    assert ArrayHeaderIO is not None
    assert callable(clean_config_dict)
    assert callable(copy_default_config)
    assert callable(default_config_bytes)
    assert callable(default_config_path)
    assert callable(default_config_resource)
    assert callable(default_config_sha256)
    assert callable(default_config_text)
    assert callable(process_image)
    assert callable(validate_config)
    assert callable(build_weight_product)
    assert WeightMaskProduct is not None
    assert QUALITY_BITS is not None
    assert MASK_POLARITY
    assert INVERSE_VARIANCE_SEMANTICS
    assert callable(TorchfitsArrayHeaderIO)
    assert issubclass(TorchfitsUnavailableError, ImportError)
    assert isinstance(torchfits_available(), bool)


def test_api_docs_match_supported_import_paths():
    text = (ROOT / "docs" / "api.md").read_text()
    assert "weightmask.background" in text
    assert "weightmask.torchfits_adapter" in text
    assert "TorchfitsArrayHeaderIO" in text
    assert "weightmask.config" in text
    assert "weightmask.process" in text
    assert "weightmask.contract" in text
    assert "weightmask.reconstruct_sky" in text
    assert "weightmask.sky_to_mesh` and" in text
    assert "not supported paths" in text
    assert "weightmask.utils" not in text


def test_usage_docs_define_derived_output_names_and_mesh_fallback():
    text = (ROOT / "docs" / "usage.md").read_text()
    normalized = " ".join(text.split())
    assert "<primary_stem>.mask.fits" in text
    assert "<primary_stem>.ivar.fits" in text
    assert "<primary_stem>.sky.fits" in text
    assert "<primary_stem>.bad.fits" in text
    assert "falls back to a full-resolution sky map" in text
    assert "sky_format: mesh" in text
    assert "previous generation remains untouched" in normalized
    assert "weightmask.utils" not in text


def test_derived_output_paths_follow_the_primary_map_stem(tmp_path):
    from weightmask.cli import determine_output_paths

    cases = (
        ("science.fits", False, None, "science.weight.fits"),
        ("science.fits.fz", True, None, "science.weight.fits.fz"),
        ("science.fits", False, "custom.map.fits", "custom.map.fits"),
        ("science.fits", False, "custom", "custom"),
    )
    suffixes = {
        "out_mask_path": "mask",
        "out_invvar_path": "ivar",
        "out_sky_path": "sky",
    }
    individual_suffixes = ("bad", "sat", "cr", "obj", "streak", "nodata")
    for input_name, compress, explicit_map, primary_name in cases:
        args = Namespace(
            output_map=str(tmp_path / explicit_map) if explicit_map else None,
            output_mask=None,
            output_invvar=None,
            output_sky=None,
            output_weight_raw=None,
            individual_masks=True,
        )
        paths = determine_output_paths(
            args,
            str(tmp_path / input_name),
            {"output_params": {"compress": compress}},
        )
        assert Path(paths["out_map_path"]).name == primary_name
        stem = Path(primary_name).stem
        for key, suffix in suffixes.items():
            assert Path(paths[key]).name == f"{stem}.{suffix}.fits"
        for suffix in individual_suffixes:
            assert Path(paths["individual_mask_paths"][suffix]).name == f"{stem}.{suffix}.fits"
        assert paths["out_weight_raw_path"] is None


def test_cli_help_describes_the_bundled_config_copy_path():
    from io import StringIO
    from unittest.mock import patch

    from weightmask.cli import parse_arguments

    output = StringIO()
    with patch("sys.stdout", output), pytest.raises(SystemExit) as raised:
        parse_arguments(["--help"])
    assert raised.value.code == 0
    text = output.getvalue()
    assert "weightmask.yml is bundled" in text
    assert "weightmask.config.copy_default_config" in text
    assert "weightmask.utils" not in text


def test_mesh_output_without_sep_geometry_falls_back_to_full_sky():
    from weightmask.process import _format_sky_output

    sky = np.arange(20, dtype=np.float32).reshape(4, 5)
    output, cards = _format_sky_output(sky, {"sky_format": "mesh"}, None)
    np.testing.assert_array_equal(output, sky)
    assert cards == {}


@pytest.mark.parametrize(
    ("method", "mesh_expected"),
    (("sep", True), ("median_filter", False), ("robust_median_fallback", False)),
)
def test_sky_format_mesh_is_consistent_for_every_valid_background_method(method, mesh_expected):
    from weightmask.process import process_image

    config = yaml.safe_load((ROOT / "weightmask.yml").read_text())
    config["sep_background"].update({"method": method, "iterations": 0})
    config["cosmic_ray"].update({"niter": 0})
    config["cosmic_ray"]["faint_cr"]["enable"] = False
    config["sep_objects"]["extract_thresh"] = 1.0e6
    config["streak_masking"]["enable"] = False
    config["output_params"]["sky_format"] = "mesh"
    data = np.full((64, 64), 1000.0, dtype=np.float32)

    _mask, _ivar, _weight, _confidence, sky, info = process_image(data, {}, None, config, tile_size=32)

    if mesh_expected:
        assert sky.shape == (2, 2)
        assert info["sky_cards"] == {"SKYMESH": True, "MESHBW": 32, "MESHBH": 32, "SKYH": 64, "SKYW": 64}
    else:
        assert sky.shape == data.shape
        assert info["sky_cards"] == {}
