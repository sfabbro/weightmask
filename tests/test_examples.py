"""Examples must report failures and preserve existing science downloads."""

import io
from http.client import IncompleteRead
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

import numpy as np
import pytest
from astropy.io import fits

from examples import cfht_megaprime_example as cfht
from examples import real_world_robustness as robustness
from examples import test_real_mef as real_mef


def test_download_http_failure_preserves_existing_science(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = Path("2079618p.fits.fz")
    target.write_bytes(b"existing FITS science")

    with patch("urllib.request.urlopen", side_effect=HTTPError("https://example.invalid", 404, "not found", {}, None)):
        result = cfht.download_cfht_image()
    assert result is None
    assert target.read_bytes() == b"existing FITS science"


@pytest.mark.parametrize("compressed", [False, True])
def test_successful_download_replaces_target_only_after_transfer(tmp_path, monkeypatch, compressed):
    monkeypatch.chdir(tmp_path)
    target = Path("2079618p.fits.fz")
    target.write_bytes(b"existing FITS science")
    response = io.BytesIO()
    data = np.ones((2, 2), np.float32)
    if compressed:
        fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(data)]).writeto(response)
    else:
        fits.PrimaryHDU(data).writeto(response)
    science = response.getvalue()
    response.seek(0)
    with patch("urllib.request.urlopen", return_value=response):
        result = cfht.download_cfht_image()
    assert result == target.name
    assert target.read_bytes() == science
    assert sorted(path.name for path in tmp_path.iterdir()) == [target.name]


@pytest.mark.parametrize("transfer_error", [OSError("transfer interrupted"), IncompleteRead(b"partial", 100)])
def test_interrupted_download_preserves_existing_science(tmp_path, monkeypatch, transfer_error):
    monkeypatch.chdir(tmp_path)
    target = Path("2079618p.fits.fz")
    target.write_bytes(b"existing FITS science")

    class InterruptedResponse(io.BytesIO):
        def read(self, size=-1):
            assert target.read_bytes() == b"existing FITS science"
            if self.tell():
                raise transfer_error
            return super().read(size)

    with patch("urllib.request.urlopen", return_value=InterruptedResponse(b"partial")):
        assert cfht.download_cfht_image() is None
    assert target.read_bytes() == b"existing FITS science"
    assert sorted(path.name for path in tmp_path.iterdir()) == [target.name]


def test_http_success_with_nonfits_body_preserves_existing_science(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = Path("2079618p.fits.fz")
    target.write_bytes(b"existing FITS science")
    with patch("urllib.request.urlopen", return_value=io.BytesIO(b"<html>error</html>")):
        assert cfht.download_cfht_image() is None
    assert target.read_bytes() == b"existing FITS science"
    assert sorted(path.name for path in tmp_path.iterdir()) == [target.name]


def test_http_success_with_truncated_fits_preserves_existing_science(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = Path("2079618p.fits.fz")
    target.write_bytes(b"existing FITS science")
    response = io.BytesIO()
    fits.PrimaryHDU(np.ones((2, 2), np.float32)).writeto(response)
    complete_header_only = io.BytesIO(response.getvalue()[:2880])
    with patch("urllib.request.urlopen", return_value=complete_header_only):
        assert cfht.download_cfht_image() is None
    assert target.read_bytes() == b"existing FITS science"
    assert sorted(path.name for path in tmp_path.iterdir()) == [target.name]


def test_robustness_example_finds_config_outside_repo(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    def run(config, *args, **kwargs):
        assert Path(config).is_file()
        return {"Streaks": (0.0, 0.0)}

    with patch.object(robustness, "run_masking_test", side_effect=run):
        robustness.demonstrate_robustness()


def test_real_mef_example_reports_missing_input(tmp_path, monkeypatch):
    monkeypatch.setattr(real_mef, "REPO_ROOT", str(tmp_path))
    assert real_mef.evaluate_real_mef() == 1


def test_real_mef_example_reports_processing_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(real_mef, "REPO_ROOT", str(tmp_path))
    path = tmp_path / "benchmark_data" / "megacam" / "megacam_streak_case.fits"
    path.parent.mkdir(parents=True)
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.ones((2, 2))), fits.ImageHDU(np.ones((2, 2)))]).writeto(path)
    (tmp_path / "weightmask.yml").write_text("{}")
    with patch.object(real_mef, "process_image", return_value=(None,) * 6):
        assert real_mef.evaluate_real_mef() == 1
