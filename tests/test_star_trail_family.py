"""The star-trail-family estimator and the veto built on it.

The veto's whole argument rests on two claims that need to be tested rather than
asserted: a non-sidereal frame carries a measurable family of trails, and a
sidereal frame does not. If the second claim were false, enabling the veto by
default would put every sidereal recall number in the project at risk.
"""

import contextlib
import io
import unittest
from pathlib import Path

import numpy as np
import yaml

from benchmarks.production_inputs import capture_detector_inputs, streak_config
from benchmarks.streak_inject import trail_recall
from weightmask.config import configuration_errors, default_config, section_config
from weightmask.streaks import (
    _angular_distance_deg,
    _axis_aligned_band,
    _veto_components,
    detect_streaks,
    frame_trail_family,
)

ROOT = Path(__file__).resolve().parents[1]
SHAPE = (360, 360)
# Foreign trails have to be long to be foreign, and in 360 px they cross enough
# of the grid to merge with it; the estimator tests that inject them use this.
BIG_SHAPE = (700, 700)
FAMILY_ANGLE = 40.0
FAMILY_LENGTH = 60.0


def _trail(shape, *, angle_deg, length, cx, cy, peak, head=0.0, width=1.2):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float64)
    theta = np.radians(angle_deg)
    along = (xx - cx) * np.cos(theta) + (yy - cy) * np.sin(theta)
    across = np.abs(-(xx - cx) * np.sin(theta) + (yy - cy) * np.cos(theta))
    on_trail = (np.abs(along) <= length / 2.0) & (across <= 3.0)
    flux = np.where(on_trail, peak * np.exp(-0.5 * (across / width) ** 2), 0.0)
    if head:
        flux = flux + head * np.exp(-0.5 * (((along + length / 2.0) / 1.6) ** 2 + (across / 1.6) ** 2))
    return flux


def _star(shape, *, cx, cy, peak, sigma=1.4):
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float64)
    return peak * np.exp(-0.5 * (((xx - cx) / sigma) ** 2 + ((yy - cy) / sigma) ** 2))


def _non_sidereal_frame(shape=SHAPE, *, angle=FAMILY_ANGLE, length=FAMILY_LENGTH, count=36, seed=0, jitter=1.0):
    """Trailed stars on a grid, one angle, bright enough to be measured.

    A grid rather than random placement: at 60 px spacing in a 360 px frame,
    random placement merges neighbours into blobs whose orientation is noise, and
    the estimator is then being judged on the fixture rather than on itself.
    """
    rng = np.random.default_rng(seed)
    frame = rng.normal(0.0, 1.0, size=shape)
    side = int(np.ceil(np.sqrt(count)))
    step = shape[0] / (side + 1.0)
    placed = 0
    for row in range(side):
        for column in range(side):
            if placed >= count:
                break
            frame += _trail(
                shape,
                angle_deg=angle + float(rng.normal(0.0, jitter)),
                length=length,
                cx=(column + 1.0) * step,
                cy=(row + 1.0) * step,
                peak=60.0,
                head=150.0,
            )
            placed += 1
    return frame, np.ones(shape)


def _sidereal_frame(shape=SHAPE, *, stars=30, seed=0):
    rng = np.random.default_rng(seed)
    frame = rng.normal(0.0, 1.0, size=shape)
    for _ in range(stars):
        frame += _star(
            shape,
            cx=float(rng.uniform(30, shape[1] - 30)),
            cy=float(rng.uniform(30, shape[0] - 30)),
            peak=float(rng.uniform(60.0, 300.0)),
        )
    return frame, np.ones(shape)


def _coverage(mask, body):
    if not np.any(body):
        return 0.0
    return float(np.count_nonzero(mask & body)) / float(np.count_nonzero(body))


def _line_mask(shape, *, angle_deg, length, cx, cy, width=2):
    body = _trail(shape, angle_deg=angle_deg, length=length, cx=cx, cy=cy, peak=1.0) > 0.4
    return body


class TestFrameTrailFamily(unittest.TestCase):
    def test_family_is_recovered_from_a_non_sidereal_frame(self):
        frame, rms = _non_sidereal_frame(BIG_SHAPE, count=30)
        family = frame_trail_family(frame, rms, None, {"enable": True})
        self.assertIsNotNone(family, "40 trails at one angle are a family")
        self.assertLess(abs(_angular_distance_deg(family["angle_deg"], FAMILY_ANGLE)), 3.0)
        self.assertGreaterEqual(family["count"], 20)
        self.assertLess(family["angle_spread_deg"], 6.0)
        # The length statistic is sqrt(12 * variance), so a rendered trail with a
        # bright head measures somewhere under its geometric length; the number
        # only has to be a stable scale for the ratio the veto compares.
        self.assertGreater(family["length_px"], 0.4 * FAMILY_LENGTH)
        self.assertLess(family["length_px"], 1.5 * FAMILY_LENGTH)

    def test_round_stars_are_not_a_family(self):
        frame, rms = _sidereal_frame()
        self.assertIsNone(frame_trail_family(frame, rms, None, {"enable": True}))

    def test_random_orientations_are_not_a_family(self):
        rng = np.random.default_rng(7)
        frame = rng.normal(0.0, 1.0, size=SHAPE)
        for _ in range(30):
            frame += _trail(
                SHAPE,
                angle_deg=float(rng.uniform(0.0, 180.0)),
                length=40.0,
                cx=float(rng.uniform(40, SHAPE[1] - 40)),
                cy=float(rng.uniform(40, SHAPE[0] - 40)),
                peak=25.0,
            )
        self.assertIsNone(
            frame_trail_family(frame, np.ones(SHAPE), None, {"enable": True}),
            "cosmic rays and bleed share no angle, so their spread is not a family",
        )

    def test_full_span_axis_bands_are_excluded_from_the_family(self):
        self.assertTrue(_axis_aligned_band(0.95 * SHAPE[1], 0.0, SHAPE))
        self.assertTrue(_axis_aligned_band(0.95 * SHAPE[0], 90.0, SHAPE))
        self.assertFalse(_axis_aligned_band(0.4 * SHAPE[1], 0.0, SHAPE))
        self.assertFalse(_axis_aligned_band(0.95 * SHAPE[1], 45.0, SHAPE))

    def test_foreign_trails_do_not_erase_the_family(self):
        """A handful of long trails at other angles must not hide the family.

        Measured on a 1024-crop of ``1851960`` HDU 17: 26 field trails plus the
        harness's six injected satellites pushed a whole-frame spread test past
        its limit and the family disappeared -- failing open exactly when a
        satellite was present. The dominant orientation window keeps it.
        """
        frame, rms = _non_sidereal_frame(BIG_SHAPE, count=30)
        rng = np.random.default_rng(5)
        for offset in (0.0, 90.0, 30.0, 60.0, 90.0, 15.0):
            frame += _trail(
                frame.shape,
                angle_deg=(FAMILY_ANGLE + offset + 20.0) % 180.0,
                length=300.0,
                cx=float(rng.uniform(80, frame.shape[1] - 80)),
                cy=float(rng.uniform(80, frame.shape[0] - 80)),
                peak=12.0,
            )
        family = frame_trail_family(frame, rms, None, {"enable": True})
        self.assertIsNotNone(family, "six foreign trails must not erase a 30-source family")
        self.assertLess(abs(_angular_distance_deg(family["angle_deg"], FAMILY_ANGLE)), 4.0)
        self.assertGreaterEqual(family["count"], 20)

    def test_min_sources_is_a_floor_not_a_formality(self):
        frame, rms = _non_sidereal_frame(count=8)
        self.assertIsNone(frame_trail_family(frame, rms, None, {"enable": True}))
        self.assertIsNotNone(frame_trail_family(frame, rms, None, {"enable": True, "min_sources": 5}))


def _family_veto(mask, family, **veto):
    config = section_config("streak_masking", {"star_trail_veto": veto})
    return _veto_components(mask, np.zeros(mask.shape, dtype=np.float32), None, None, config, family)


class TestStarTrailVeto(unittest.TestCase):
    family = {"angle_deg": FAMILY_ANGLE, "length_px": FAMILY_LENGTH, "angle_spread_deg": 2.0, "count": 40}

    def test_a_chain_parallel_to_the_family_is_dropped(self):
        chain = _line_mask(SHAPE, angle_deg=FAMILY_ANGLE + 1.0, length=8.0 * FAMILY_LENGTH, cx=180, cy=180)
        kept, dropped = _family_veto(chain, self.family)
        self.assertEqual(dropped["components"], 0, "an 8x family length is a satellite, not a chain")
        self.assertTrue(np.array_equal(kept, chain))

        chain = _line_mask(SHAPE, angle_deg=FAMILY_ANGLE + 1.0, length=2.5 * FAMILY_LENGTH, cx=180, cy=180)
        kept, dropped = _family_veto(chain, self.family)
        self.assertEqual(dropped["components"], 1)
        self.assertEqual(dropped["px"], int(np.count_nonzero(chain)))
        self.assertFalse(np.any(kept))

    def test_a_trail_at_its_own_angle_survives(self):
        trail = _line_mask(SHAPE, angle_deg=FAMILY_ANGLE + 40.0, length=2.0 * FAMILY_LENGTH, cx=180, cy=180)
        kept, dropped = _family_veto(trail, self.family)
        self.assertEqual(dropped["components"], 0)
        self.assertTrue(np.array_equal(kept, trail))

    def test_angle_and_length_are_both_required(self):
        long_parallel = _line_mask(SHAPE, angle_deg=FAMILY_ANGLE, length=10.0 * FAMILY_LENGTH, cx=180, cy=180)
        kept, _dropped = _family_veto(long_parallel, self.family)
        self.assertTrue(np.array_equal(kept, long_parallel), "a satellite parallel to the field keeps its trail")

        short_angled = _line_mask(SHAPE, angle_deg=FAMILY_ANGLE + 60.0, length=FAMILY_LENGTH, cx=180, cy=180)
        kept, _dropped = _family_veto(short_angled, self.family)
        self.assertTrue(np.array_equal(kept, short_angled))

    def test_small_components_are_left_to_the_profile_gate(self):
        tiny = np.zeros(SHAPE, dtype=bool)
        tiny[10, 10:16] = True
        kept, dropped = _family_veto(tiny, self.family)
        self.assertEqual(dropped["components"], 0)
        self.assertTrue(np.array_equal(kept, tiny))

    def test_no_family_means_no_drop(self):
        chain = _line_mask(SHAPE, angle_deg=FAMILY_ANGLE, length=2.0 * FAMILY_LENGTH, cx=180, cy=180)
        kept, dropped = _family_veto(chain, None)
        self.assertEqual(dropped["components"], 0)
        self.assertTrue(np.array_equal(kept, chain))

    def test_the_angle_tolerance_is_the_knob_that_moves(self):
        near = _line_mask(SHAPE, angle_deg=FAMILY_ANGLE + 15.0, length=2.0 * FAMILY_LENGTH, cx=180, cy=180)
        _kept, dropped = _family_veto(near, self.family)
        self.assertEqual(dropped["components"], 0, "15 degrees off is outside the 10-degree tolerance")
        _kept, dropped = _family_veto(near, self.family, angle_tol_deg=20.0)
        self.assertEqual(dropped["components"], 1)


STAGE_SHAPE = (700, 700)


def _sidereal_hdu(shape=STAGE_SHAPE, *, sky=1000.0, noise=5.0, stars=14, seed=0):
    """A quiet CCD-like frame with round stars, as the recall floor builds one.

    The stage is fed through the production input path below, not with a bare
    array: a single background pass over an unmasked frame manufactures linear
    features of its own, which is the same trap the campaign fell into.
    """
    rng = np.random.default_rng(seed)
    data = rng.normal(sky, noise, shape).astype(np.float32)
    yy, xx = np.mgrid[0 : shape[0], 0 : shape[1]]
    for _ in range(stars):
        cy, cx = rng.uniform(30, shape[0] - 30), rng.uniform(30, shape[1] - 30)
        amplitude = rng.uniform(50, 4000)
        sigma = rng.uniform(1.5, 3.0)
        data += (amplitude * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma**2)))).astype(np.float32)
    return data


class TestVetoThroughTheStage(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trail_recall = staticmethod(trail_recall)
        config = default_config()
        cls.streak_cfg = streak_config(config)
        cls.raw = _sidereal_hdu()
        # 8 sigma of this fixture's 5 ADU noise, the same regime as the recall
        # floor's continuous cells. Brighter is *not* safer here: the brightness
        # veto deletes a trail above max_component_sigma (20), which is the risk
        # its own config comment names for a bright satellite.
        peak = 8.0 * 5.0
        cls.body = (
            _trail(
                STAGE_SHAPE,
                angle_deg=20.0,
                length=400.0,
                cx=0.55 * STAGE_SHAPE[1],
                cy=0.5 * STAGE_SHAPE[0],
                peak=peak,
            )
            > 0.4
        )
        injected = cls.raw + _trail(
            STAGE_SHAPE,
            angle_deg=20.0,
            length=400.0,
            cx=0.55 * STAGE_SHAPE[1],
            cy=0.5 * STAGE_SHAPE[0],
            peak=peak,
        ).astype(np.float32)
        header = {"GAIN": 1.5, "RDNOISE": 5.0}
        with contextlib.redirect_stdout(io.StringIO()):
            cls.clean = capture_detector_inputs(cls.raw, header, config, flat=np.ones(STAGE_SHAPE, np.float32))
            cls.inputs = capture_detector_inputs(injected, header, config, flat=np.ones(STAGE_SHAPE, np.float32))

    def _run(self, *, enable):
        cfg = dict(self.streak_cfg)
        cfg["star_trail_veto"] = dict(cfg["star_trail_veto"], enable=enable)
        with contextlib.redirect_stdout(io.StringIO()):
            return detect_streaks(self.inputs[0], self.inputs[1], self.inputs[2], cfg).mask

    def test_a_sidereal_trail_is_detected_identically_with_and_without_the_veto(self):
        """The estimator must return no family on a frame of round stars, so the
        veto cannot reach a sidereal detection. This is the claim that makes
        ``enable: true`` in the shipped config safe, and the trail has to be
        found in both runs for the comparison to mean anything."""
        with_veto = self._run(enable=True)
        without_veto = self._run(enable=False)
        self.assertTrue(np.array_equal(with_veto, without_veto), "the family veto moved a sidereal mask")
        _exact, tolerant, line = self.trail_recall(with_veto, self.body, dilation=5, line_half_width=2.0)
        self.assertGreater(line, 0.5, f"the injected trail must be found (tolerant {tolerant:.2f})")

    def test_the_quiet_frame_alone_yields_no_streak_pixels(self):
        cfg = dict(self.streak_cfg)
        with contextlib.redirect_stdout(io.StringIO()):
            mask = detect_streaks(self.clean[0], self.clean[1], self.clean[2], cfg).mask
        self.assertEqual(int(np.count_nonzero(mask)), 0, "the fixture must be clean before injection")


class TestShippedConfiguration(unittest.TestCase):
    def test_both_canonical_configs_validate_and_carry_the_veto(self):
        blocks = []
        for path in (ROOT / "weightmask" / "weightmask.yml", ROOT / "weightmask" / "weightmask.yml"):
            with open(path) as handle:
                config = yaml.safe_load(handle)
            self.assertEqual(configuration_errors(config), [], f"{path} does not validate")
            block = config["streak_masking"]["star_trail_veto"]
            self.assertTrue(block["enable"])
            self.assertGreater(block["length_factor"], 1.0)
            blocks.append(block)
        self.assertEqual(blocks[0], blocks[1], "the published and packaged configs must agree")

    def test_an_unknown_veto_key_is_rejected(self):
        config = default_config()
        config["streak_masking"]["star_trail_veto"]["length_factorr"] = 4.0
        self.assertTrue(configuration_errors(config), "a typo in a config key must be an error")


if __name__ == "__main__":
    unittest.main()
