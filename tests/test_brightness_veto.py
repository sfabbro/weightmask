"""The brightness veto is optional, and it must never cost a real trail.

``max_component_sigma`` can be set to any number or to ``null``, which disables
the veto. The two facts worth pinning are separate and both have been true here:

* the veto suppresses real artefact classes -- a hot column group and a bright
  star arm, 10734 and 4385 px, which the pre-masked veto cannot catch because
  both sit below its 0.25 threshold;
* it does not touch the two real satellite trails, at any setting including
  ``null``, because they measure 3.0 and 3.2 sigma against a threshold of 20.

The second is the one that matters for correctness. A deletion of the Hough
stage passed the whole suite while dropping one of those trails to zero, because
nothing asserted that the pipeline still finds them. These tests do.

The veto is also uncalibrated at the high end: 20 is a 6x margin below a sample
of two trails. That is recorded in weightmask.yml and is a deliberate
provisional setting, not a measured constant.

Data-backed cases skip when benchmark_data/megacam is absent rather than passing
vacuously.
"""

import contextlib
import copy
import io
import sys
import unittest
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weightmask.streaks import _drop_bright_components, detect_streaks  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MEGACAM = REPO / "benchmark_data" / "megacam"
FLAT = MEGACAM / "perf" / "flat_08Bm01_r.fits.fz"

#: The two exposures carrying a confirmed satellite trail, and the pixel counts the
#: detector produces on them. These are the regression this file exists for.
REAL_TRAILS = [("long/996195p.fits.fz", 35, 12001), ("long/996195p.fits.fz", 36, 11034)]


def _shipped():
    return yaml.safe_load(open(REPO / "weightmask.yml"))["streak_masking"]


def _production_inputs(relative, hdu):
    """(data_sub, rms, existing_mask) as process_image builds them, or skip."""
    if not MEGACAM.is_dir():
        raise unittest.SkipTest("benchmark_data/megacam not present")
    path = MEGACAM / relative
    if not path.exists():
        raise unittest.SkipTest(f"{path} not present")
    import fitsio

    from benchmarks.production_inputs import capture_detector_inputs, streak_config

    config = yaml.safe_load(open(REPO / "weightmask.yml"))
    with fitsio.FITS(str(path)) as handle:
        science = np.ascontiguousarray(handle[hdu].read().astype(np.float32))
        header = handle[hdu].read_header()
    flat = None
    if FLAT.exists():
        with fitsio.FITS(str(FLAT)) as handle:
            if hdu < len(handle) and handle[hdu].read().shape == science.shape:
                flat = np.ascontiguousarray(handle[hdu].read().astype(np.float32))
    with contextlib.redirect_stdout(io.StringIO()):
        data_sub, rms, existing = capture_detector_inputs(science, header, config, flat=flat)
    return data_sub, rms, existing, streak_config(config)


def _detect(data_sub, rms, existing, scfg, **mask_params):
    config = copy.deepcopy(scfg)
    config["mask_params"] = {**config["mask_params"], **mask_params}
    with contextlib.redirect_stdout(io.StringIO()):
        return detect_streaks(data_sub, rms, existing, config)


class TestBrightnessVetoIsOptional(unittest.TestCase):
    """null must disable the veto, and nothing else may."""

    def test_none_disables_the_veto(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[20, 5:35] = True
        data = np.full((40, 40), 1000.0)
        rms = np.ones((40, 40))
        dropped = _drop_bright_components(mask, data, rms, 20.0, 24)
        self.assertEqual(int(dropped.sum()), 0, "a 1000-sigma component must be dropped at threshold 20")
        kept = _drop_bright_components(mask, data, rms, None, 24)
        self.assertEqual(int(kept.sum()), int(mask.sum()), "None must disable the veto, not merely relax it")

    def test_the_threshold_is_honoured_between_the_extremes(self):
        mask = np.zeros((40, 40), dtype=bool)
        mask[20, 5:35] = True
        data = np.full((40, 40), 50.0)
        rms = np.ones((40, 40))
        self.assertEqual(int(_drop_bright_components(mask, data, rms, 20.0, 24).sum()), 0)
        self.assertEqual(
            int(_drop_bright_components(mask, data, rms, 100.0, 24).sum()), int(mask.sum()), "50 < 100 must be kept"
        )

    def test_shipped_config_declares_it_and_the_key_is_readable(self):
        self.assertIn("max_component_sigma", _shipped()["mask_params"])
        self.assertIsNotNone(_shipped()["mask_params"]["max_component_sigma"])


class TestRealTrailsSurviveEveryVetoSetting(unittest.TestCase):
    """The guard against the mistake this file was written for.

    Deleting a proposal stage left one real trail at zero pixels and the suite
    green, because no test asserted the pipeline still finds it.
    """

    def test_both_trails_are_found_at_every_veto_setting(self):
        settings = {
            "shipped default": {},
            "veto off": {"max_component_sigma": None},
            "veto wide": {"max_component_sigma": 5000.0},
        }
        for relative, hdu, expected in REAL_TRAILS:
            data_sub, rms, existing, scfg = _production_inputs(relative, hdu)
            for label, override in settings.items():
                mask = _detect(data_sub, rms, existing, scfg, **override)
                pixels = int(mask.sum())
                self.assertEqual(
                    pixels,
                    expected,
                    f"{relative}:{hdu} with {label} produced {pixels} px, expected {expected}. "
                    "A real satellite trail must survive every veto setting.",
                )

    def test_the_veto_really_is_load_bearing(self):
        """Guard the opposite direction: if the veto stopped mattering, this file's
        other assertions would pass for the wrong reason."""
        path, hdu, _expected = ("perf/1013719p.fits.fz", 5, 0)
        data_sub, rms, existing, scfg = _production_inputs(path, hdu)
        with_veto = int(_detect(data_sub, rms, existing, scfg).sum())
        without = int(_detect(data_sub, rms, existing, scfg, max_component_sigma=None).sum())
        self.assertEqual(with_veto, 0, "the hot column group should be suppressed by default")
        self.assertGreater(without, 1000, "disabling the veto should let the column group back in")


if __name__ == "__main__":
    unittest.main()
