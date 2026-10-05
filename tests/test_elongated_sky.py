"""Elongated detections stay out of DETECTED and out of the sky mesh."""

import unittest
from unittest.mock import patch

import numpy as np
import yaml

from weightmask.contract import QualityBit
from weightmask.process import process_image


class TestElongatedSkyHandoff(unittest.TestCase):
    def test_bar_is_in_the_sky_mask_and_not_detected(self):
        from weightmask.background import estimate_background

        shape = (120, 120)
        rng = np.random.default_rng(1)
        data = rng.normal(0.0, 1.0, shape).astype(np.float32)
        data[58:62, 20:100] = 40.0
        seen = []

        def record(sci, mask, config):
            seen.append(np.array(mask, copy=True))
            return estimate_background(sci, mask, config)

        cfg = yaml.safe_load(open("weightmask.yml"))
        cfg["streak_masking"]["enable"] = False
        cfg["sep_objects"]["extract_thresh"] = 2.0
        cfg["sep_objects"]["min_area"] = 5
        cfg["sep_objects"]["max_elongation"] = 2.0
        cfg["sep_objects"]["spike_enable"] = False
        with (
            patch("weightmask.process.detect_cosmic_rays", return_value=np.zeros(shape, dtype=bool)),
            patch("weightmask.process.estimate_background", side_effect=record),
        ):
            mask, *_rest = process_image(
                data, {"GAIN": 1.5, "RDNOISE": 5.0}, np.ones(shape, dtype=np.float32), cfg, tile_size=32
            )
        self.assertTrue(seen)
        bar = np.zeros(shape, dtype=bool)
        bar[58:62, 20:100] = True
        self.assertGreater(int(np.count_nonzero(seen[-1] & bar)), 0)
        self.assertEqual(int(np.count_nonzero(((mask & int(QualityBit.DETECTED)) != 0) & bar)), 0)


if __name__ == "__main__":
    unittest.main()
