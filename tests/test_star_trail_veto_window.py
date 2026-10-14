"""The veto's length window, measured rather than asserted.

``streak_masking.star_trail_veto`` decides on two conditions, and the shipped
config justifies the length one as a trade: "raising length_factor trades these
false positives for that sensitivity". Neither side of that trade had a number
on it. ``benchmarks/star_trail_veto_window.py`` measures both on one frame, and
these tests pin what it measures so a change to the veto or to the family
estimator has to move a recorded boundary instead of retiring in silence.

Two of these tests record limitations rather than guarantees. A gapped chain --
the campaign's own geometry -- is dropped at every span up to 21x the family
length. A *contiguous* chain is not: once its segments touch it is one component
of the same length as a satellite trail, so it is dropped inside the window and
kept outside it, exactly like the satellite the window is trying to protect. The
length condition cannot separate those two classes, and the test says so.
"""

import unittest
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits

from benchmarks.production_inputs import capture_detector_inputs
from benchmarks.star_trail_veto_window import (
    CHAIN_GAP_FACTORS,
    CHAIN_SEGMENTS,
    chain_cells,
    contamination_cells,
    parallel_cells,
    probe,
    refine_boundary,
    render_trail,
    synthetic_frame,
    window,
)
from weightmask.streaks import frame_trail_family

ROOT = Path(__file__).resolve().parents[1]
SHAPE = (1024, 1024)
ANGLE = 40.2
TRAIL_LENGTH = 55.0
RMS = 21.8
HEADER = fits.Header({"NAXIS": 2, "SATURATE": 65535.0, "EXPTIME": 60.0, "GAIN": 1.6, "RDNOISE": 5.0})
# Multiples of the *rendered* star-trail length, and the measured family length
# they come to on this fixture: the estimator reports a 55 px trail as 47.43 px,
# so a 3.48x measured multiple is 165 px of trail and a 4.06x one is 192 px.
DROPPED_THROUGH = 3.48
KEPT_FROM = 4.06


class TestStarTrailVetoWindow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(ROOT / "weightmask.yml") as handle:
            cls.config = yaml.safe_load(handle)
        cls.veto_cfg = cls.config["streak_masking"]["star_trail_veto"]
        cls.rms_map = np.full(SHAPE, RMS, dtype=np.float32)
        frame, _truth = synthetic_frame(
            SHAPE, angle_deg=ANGLE, trail_length=TRAIL_LENGTH, stars=60, sky=571.0, rms=RMS, seed=0
        )
        cls.frame = frame
        data_sub, rms, existing = capture_detector_inputs(frame, HEADER, cls.config, flat=None)
        cls.inputs = (data_sub, rms, existing)
        cls.family = frame_trail_family(data_sub, rms, existing, cls.veto_cfg)
        assert cls.family is not None, "the fixture must carry a family for the veto to have a window"

    def _parallel(self):
        return parallel_cells(
            SHAPE,
            angle_deg=ANGLE,
            trail_length=TRAIL_LENGTH,
            family=self.family,
            veto_cfg=self.veto_cfg,
            rms_map=self.rms_map,
        )

    def test_the_family_length_is_measured_not_nominal(self):
        """The comparison is in measured units, so the bias has to be recorded.

        A trail rendered with a bright head measures under its geometric length,
        and the veto compares that estimate against itself -- which is why the
        window is quoted as a multiple of the family's length and not of the
        star trails a camera operator would measure on the frame.
        """
        fraction = self.family["length_px"] / TRAIL_LENGTH
        self.assertGreater(fraction, 0.75, "the estimator must not lose a quarter of the trail")
        self.assertLess(fraction, 1.0, "a rendered trail with a head measures under its span")
        self.assertEqual(self.family["angle_deg"], 40.07)
        self.assertEqual(self.family["length_px"], 47.43)
        self.assertGreaterEqual(self.family["count"], int(self.veto_cfg["min_sources"]))

    def test_the_size_floor_never_decides_a_cell(self):
        """The sweep measures the length and angle conditions, not the size gate.

        ``min_mask_pixels`` is hard-coded at 24 in the probe because every body
        it is handed clears it by roughly four times -- the smallest cell is a
        half-length trail at 93 px. If a future cell got near the floor the
        window would be measuring the size rule instead, and this is where that
        shows up.
        """
        smallest = min(row["size_px"] for row in self._parallel())
        with open(ROOT / "weightmask.yml") as handle:
            min_mask_pixels = yaml.safe_load(handle)["streak_masking"]["mask_params"]["min_mask_pixels"]
        self.assertEqual(smallest, 93)
        self.assertGreater(smallest, 3 * int(min_mask_pixels))

    def test_a_parallel_trail_is_dropped_inside_the_window_and_kept_outside_it(self):
        rows = self._parallel()
        by_multiple = {row["geometric_multiple"]: row for row in rows}
        short = by_multiple[2.32]
        long = by_multiple[6.96]
        self.assertTrue(short["dropped"], f"a satellite {short['length_multiple']}x the family is dropped")
        self.assertEqual(short["dropped_px"], short["size_px"], "the whole component goes, not part of it")
        self.assertFalse(long["dropped"], f"a satellite {long['length_multiple']}x the family is kept")
        self.assertEqual(long["dropped_px"], 0)

    def test_the_boundary_sits_on_the_configured_length_factor(self):
        """The measured window, in family-length multiples, is where the config puts it.

        The sweep is coarse -- half a star-trail length apart -- so the boundary
        is pinned to within one step, and the step itself is asserted so a
        fixture that quietly changed shape is noticed rather than tolerated.
        """
        measure = window(self._parallel())
        length_factor = float(self.veto_cfg["length_factor"])
        self.assertAlmostEqual(measure["dropped_through"], DROPPED_THROUGH, places=2)
        self.assertAlmostEqual(measure["kept_from"], KEPT_FROM, places=2)
        self.assertLess(measure["dropped_through"], length_factor)
        self.assertGreater(measure["kept_from"], length_factor)
        self.assertLess(measure["kept_from"] - measure["dropped_through"], 1.0)

    def test_the_crossing_bisects_onto_the_configured_length_factor(self):
        """The crossing itself, not just the bracket: 189.01 px against a 4x factor.

        The sweep is half a star-trail length apart, so bisection is what turns
        "somewhere between 3.48x and 4.06x" into a number. It lands at 0.996 of the
        configured factor, which is what makes "the boundary is ``length_factor``"
        a measurement instead of a restatement of the code -- and in the units a
        camera operator would measure, that is 3.44 of the 55 px star trails the
        fixture renders.
        """
        rows = self._parallel()
        parallel = [row for row in rows if row["kind"] == "satellite-parallel"]
        boundary = refine_boundary(
            SHAPE,
            angle_deg=ANGLE,
            family=self.family,
            veto_cfg=self.veto_cfg,
            rms_map=self.rms_map,
            lower_px=max(row["geometric_length_px"] for row in parallel if row["dropped"]),
            upper_px=min(row["geometric_length_px"] for row in parallel if not row["dropped"]),
        )
        length_factor = float(self.veto_cfg["length_factor"])
        self.assertAlmostEqual(boundary["crossing_px"], 189.01, places=2)
        self.assertAlmostEqual(boundary["crossing_multiple"], 3.985, places=3)
        self.assertLess(abs(boundary["crossing_multiple"] - length_factor), 0.05)
        self.assertAlmostEqual(boundary["crossing_px"] / TRAIL_LENGTH, 3.44, places=2)
        measure = window(rows)
        self.assertLess(measure["dropped_through"], boundary["crossing_multiple"])
        self.assertGreater(measure["kept_from"], boundary["crossing_multiple"])

    def test_raising_length_factor_moves_the_window(self):
        """The knob the config calls a trade has to actually move the boundary."""
        rows = parallel_cells(
            SHAPE,
            angle_deg=ANGLE,
            trail_length=TRAIL_LENGTH,
            family=self.family,
            veto_cfg=dict(self.veto_cfg, length_factor=8.0),
            rms_map=self.rms_map,
        )
        by_multiple = {row["geometric_multiple"]: row for row in rows}
        self.assertTrue(by_multiple[6.96]["dropped"], "at 8x the same 330 px trail is inside the window")
        self.assertFalse(by_multiple[9.28]["dropped"], "and a 440 px one is still outside it")
        rows = parallel_cells(
            SHAPE,
            angle_deg=ANGLE,
            trail_length=TRAIL_LENGTH,
            family=self.family,
            veto_cfg=dict(self.veto_cfg, length_factor=2.0),
            rms_map=self.rms_map,
        )
        by_multiple = {row["geometric_multiple"]: row for row in rows}
        self.assertFalse(by_multiple[2.32]["dropped"], "at 2x the 110 px trail is now outside the window")
        self.assertTrue(by_multiple[1.16]["dropped"], "and a star-trail-length one is still inside it")

    def test_the_angle_condition_binds_outside_the_tolerance(self):
        """Length alone would veto a parallel satellite's neighbour; angle is what spares it.

        The tolerance is 10 degrees, and the fixture measures it to within a
        degree: 9.13 degrees off is inside, 11.14 degrees off is not.
        """
        for offset, dropped in ((5.0, True), (9.0, True), (11.0, False), (15.0, False), (90.0, False)):
            object_angle = (ANGLE + offset) % 180.0
            _flux, body = render_trail(
                SHAPE,
                angle_deg=object_angle,
                length=2.0 * TRAIL_LENGTH,
                cx=0.5 * SHAPE[1],
                cy=0.5 * SHAPE[0],
                peak_sig=8.0,
                rms_map=self.rms_map,
            )
            row = probe(
                body,
                self.family,
                self.veto_cfg,
                angle_deg=object_angle,
                kind="satellite-offset",
                label=f"offset {offset}",
                length=2.0 * TRAIL_LENGTH,
            )
            self.assertEqual(
                row["dropped"],
                dropped,
                f"2x parallel, {offset} deg off: measured offset {row['angle_offset_deg']} deg",
            )

    def test_a_gapped_chain_is_dropped_at_every_span(self):
        """The campaign's geometry: the chain breaks into per-trail components.

        Each gap isolates its own star trail, so every component measures about
        1.15x the family's median length however long the chain runs, and the
        veto leaves none of it behind -- up to 21x the family length.
        """
        rows = chain_cells(
            SHAPE,
            angle_deg=ANGLE,
            trail_length=TRAIL_LENGTH,
            family=self.family,
            veto_cfg=self.veto_cfg,
            rms_map=self.rms_map,
        )
        gapped = [row for row in rows if row["gap_factor"] > 1.0]
        self.assertEqual(len(gapped), len(CHAIN_GAP_FACTORS[1:]) * len(CHAIN_SEGMENTS))
        for row in gapped:
            self.assertTrue(row["dropped"], f"{row['label']} (span {row['span_multiple']}x) must be dropped")
            self.assertLess(row["length_multiple"], 1.25, "each segment is a star trail, not a chain")
            self.assertEqual(row["dropped_components"], row["components"], "every segment goes")
        self.assertGreater(max(row["span_multiple"] for row in gapped), 20.0)

    def test_a_contiguous_chain_is_indistinguishable_from_a_satellite(self):
        """Where the veto's stated argument stops holding, pinned on purpose.

        The docstring argues that "a chain of star trails cannot outgrow the
        trails it is made of". Touching segments falsify that: a row of them is
        one component and grows without limit, so it is dropped inside the same
        window as a satellite and survives outside it. Until something other
        than length separates the two, a contiguous chain past the window is a
        false positive the veto cannot catch -- and this test is what makes that
        limitation a recorded measurement instead of an assumption.
        """
        rows = chain_cells(
            SHAPE,
            angle_deg=ANGLE,
            trail_length=TRAIL_LENGTH,
            family=self.family,
            veto_cfg=self.veto_cfg,
            rms_map=self.rms_map,
        )
        touching = [row for row in rows if row["gap_factor"] == 1.0]
        dropped = [row for row in touching if row["dropped"]]
        kept = [row for row in touching if not row["dropped"]]
        self.assertTrue(dropped and kept, "the contiguous chain must straddle the window")
        self.assertLess(max(row["length_multiple"] for row in dropped), float(self.veto_cfg["length_factor"]))
        self.assertGreater(min(row["length_multiple"] for row in kept), float(self.veto_cfg["length_factor"]))
        self.assertEqual([row["components"] for row in touching], [1] * len(touching), "touching means one component")

    def test_the_parallel_trail_does_not_move_the_family_it_is_judged_against(self):
        """A single long object must not re-scale the scale it is measured on.

        The family length is a median over its members, so one injected trail --
        even at 8x -- should leave both the length and the angle where they were.
        If that ever stops being true the window would close on the very object
        it is meant to pass.
        """
        rows = contamination_cells(
            SHAPE,
            angle_deg=ANGLE,
            trail_length=TRAIL_LENGTH,
            frame=self.frame,
            header=HEADER,
            config=self.config,
            veto_cfg=self.veto_cfg,
            rms_map=self.rms_map,
            flat=None,
        )
        for row in rows:
            self.assertIsNotNone(row["family"], f"injecting {row['injected_multiple']}x must not erase the family")
            self.assertLess(abs(row["family"]["length_px"] - self.family["length_px"]) / self.family["length_px"], 0.02)
            self.assertLess(abs(row["family"]["angle_deg"] - self.family["angle_deg"]), 0.1)


if __name__ == "__main__":
    unittest.main()
