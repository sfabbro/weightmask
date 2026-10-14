"""The brightness veto is optional, and it must never cost an unconfirmed linear feature.

``max_component_sigma`` can be set to any number or to ``null``, which disables
the veto. The two facts worth pinning are separate and both have been true here:

* the known hot column is removed by the shipped veto and appears when disabled;
* it does not touch the two known linear features, at any setting including
  ``null``, because they measure 3.0 and 3.2 sigma against a threshold of 20.

The second is the one that matters for correctness. A deletion of the Hough
stage passed the whole suite while dropping one of those features to zero, because
nothing asserted that the pipeline still finds them. These tests do.

The veto is also uncalibrated at the high end: 20 is a 6x margin above a sample
of two features. That is recorded in weightmask.yml and is a deliberate
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
from unittest.mock import patch

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from weightmask.streaks import _drop_bright_components, detect_streaks  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MEGACAM = REPO / "benchmark_data" / "megacam"
FLAT = MEGACAM / "perf" / "flat_08Bm01_r.fits.fz"

LINEAR_FEATURES = (
    ("long/996195p.fits.fz", 35, ((3176, 307), (3207, 2078))),
    ("long/996195p.fits.fz", 36, ((3228, 1106), (3244, 2078))),
)
CORRIDOR_HALF_WIDTH = 16.0
MIN_CENTERLINE_COVERAGE = 0.75
MIN_RECOVERED_SPAN = 750.0
MIN_INSIDE_FRACTION = 0.95
MAX_FEATURE_AREA = 30000
HOT_COLUMN_ROI = (slice(None), slice(1922, 1951))
REPEATED_STRUCTURE_ROI = (slice(2436, 2447), slice(1560, 1687))
REPEATED_STRUCTURE_CASES = ("perf/1013719p.fits.fz",)


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


def _detect(data_sub, rms, existing, scfg, sparse_min_length=None, **mask_params):
    config = copy.deepcopy(scfg)
    config["mask_params"] = {**config["mask_params"], **mask_params}
    if sparse_min_length is not None:
        config["sparse_ransac_params"] = {
            **config["sparse_ransac_params"],
            "min_length": sparse_min_length,
        }
    with contextlib.redirect_stdout(io.StringIO()):
        return detect_streaks(data_sub, rms, existing, config)


def _corridor_metrics(mask, endpoints):
    from scipy.ndimage import distance_transform_edt

    (y0, x0), (y1, x1) = endpoints
    dy = float(y1 - y0)
    dx = float(x1 - x0)
    length = float(np.hypot(dy, dx))
    samples = int(np.ceil(length)) + 1
    center_y = np.rint(np.linspace(y0, y1, samples)).astype(int)
    center_x = np.rint(np.linspace(x0, x1, samples)).astype(int)
    pad = int(np.ceil(CORRIDOR_HALF_WIDTH)) + 1
    y_start = max(0, min(y0, y1) - pad)
    y_stop = min(mask.shape[0], max(y0, y1) + pad + 1)
    x_start = max(0, min(x0, x1) - pad)
    x_stop = min(mask.shape[1], max(x0, x1) + pad + 1)
    distance = distance_transform_edt(~mask[y_start:y_stop, x_start:x_stop])
    centerline_coverage = float(np.mean(distance[center_y - y_start, center_x - x_start] <= CORRIDOR_HALF_WIDTH))

    recovered_y, recovered_x = np.where(mask)
    perpendicular = np.abs(dx * (recovered_y - y0) - dy * (recovered_x - x0)) / length
    along = (dx * (recovered_x - x0) + dy * (recovered_y - y0)) / length
    inside = (
        (perpendicular <= CORRIDOR_HALF_WIDTH)
        & (along >= -CORRIDOR_HALF_WIDTH)
        & (along <= length + CORRIDOR_HALF_WIDTH)
    )
    recovered_span = float(np.ptp(along[inside])) if np.any(inside) else 0.0
    return {
        "centerline_coverage": centerline_coverage,
        "recovered_span": recovered_span,
        "inside_fraction": float(np.mean(inside)) if inside.size else 0.0,
        "area": int(mask.sum()),
    }


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
        self.assertEqual(_shipped()["sparse_ransac_params"]["min_length"], 128)


class TestRealLinearFeaturesSurviveEveryVetoSetting(unittest.TestCase):
    """Guard the two curated real linear features against veto regressions.

    Deleting a proposal stage left one real feature at zero pixels and the suite
    green, because no test asserted the pipeline still finds it.
    """

    def test_both_features_are_recovered_at_every_veto_setting(self):
        settings = {
            "shipped default": {},
            "veto off": {"max_component_sigma": None},
            "veto wide": {"max_component_sigma": 5000.0},
        }
        for relative, hdu, endpoints in LINEAR_FEATURES:
            data_sub, rms, existing, scfg = _production_inputs(relative, hdu)
            for label, override in settings.items():
                mask = _detect(data_sub, rms, existing, scfg, **override)
                metrics = _corridor_metrics(mask, endpoints)
                self.assertGreaterEqual(metrics["centerline_coverage"], MIN_CENTERLINE_COVERAGE, (hdu, label, metrics))
                self.assertGreaterEqual(metrics["recovered_span"], MIN_RECOVERED_SPAN, (hdu, label, metrics))
                if label == "shipped default":
                    self.assertGreaterEqual(metrics["inside_fraction"], MIN_INSIDE_FRACTION, (hdu, label, metrics))
                    self.assertLessEqual(metrics["area"], MAX_FEATURE_AREA, (hdu, label, metrics))

            at_100 = _detect(data_sub, rms, existing, scfg, sparse_min_length=100)
            metrics = _corridor_metrics(at_100, endpoints)
            self.assertGreaterEqual(metrics["centerline_coverage"], MIN_CENTERLINE_COVERAGE, (hdu, 100, metrics))
            self.assertGreaterEqual(metrics["recovered_span"], MIN_RECOVERED_SPAN, (hdu, 100, metrics))
            self.assertGreaterEqual(metrics["inside_fraction"], MIN_INSIDE_FRACTION, (hdu, 100, metrics))
            self.assertLessEqual(metrics["area"], MAX_FEATURE_AREA, (hdu, 100, metrics))

    def test_known_hot_column_does_not_overlap_the_streak_mask(self):
        path, hdu = ("perf/1013719p.fits.fz", 5)
        data_sub, rms, existing, scfg = _production_inputs(path, hdu)
        shipped = _detect(data_sub, rms, existing, scfg)
        disabled = _detect(data_sub, rms, existing, scfg, max_component_sigma=None)
        self.assertFalse(shipped[HOT_COLUMN_ROI].any(), "shipped brightness veto")
        self.assertTrue(disabled[HOT_COLUMN_ROI].any(), "None must disable the brightness veto")
        params = scfg["mask_params"]
        restored = _drop_bright_components(
            disabled,
            data_sub,
            rms,
            params["max_component_sigma"],
            params["min_mask_pixels"],
        )
        self.assertFalse(restored[HOT_COLUMN_ROI].any(), "the shipped veto must remove the disabled component")

    def test_production_input_roi_stays_out_of_streaks(self):
        cases = [(relative, _production_inputs(relative, 5)) for relative in REPEATED_STRUCTURE_CASES]
        for relative, inputs in cases:
            with self.subTest(relative=relative):
                _data_sub, _rms, _existing, _scfg = inputs

        at_128 = _detect(*cases[0][1], sparse_min_length=128)
        self.assertFalse(at_128[REPEATED_STRUCTURE_ROI].any())

    def test_sparse_length_test_skips_cleanly_when_required_data_missing(self):
        with patch(f"{__name__}._production_inputs", side_effect=unittest.SkipTest("fixture unavailable")):
            with self.assertRaises(unittest.SkipTest):
                self.test_production_input_roi_stays_out_of_streaks()


if __name__ == "__main__":
    unittest.main()
