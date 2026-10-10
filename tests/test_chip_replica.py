"""Chip-replica veto: the same CCD-local line on two chips is not a sky trail."""

import contextlib
import io
import os
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch

import fitsio
import numpy as np
import yaml

from weightmask.contract import QualityBit
from weightmask.mef import (
    _replica_component_indices,
    _streak_catalog,
    line_geometry,
    process_all_hdus,
    replica_indices,
)


def _column(shape, x):
    mask = np.zeros(shape, dtype=bool)
    mask[:, x] = True
    return mask


def _diagonal(shape):
    mask = np.zeros(shape, dtype=bool)
    for i in range(shape[0]):
        mask[i, min(shape[1] - 1, i + 3)] = True
    return mask


def _tan_header(*, crpix1=1.0, crpix2=1.0):
    return {
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRVAL1": 150.0,
        "CRVAL2": 2.0,
        "CRPIX1": crpix1,
        "CRPIX2": crpix2,
        "CD1_1": 1.0e-4,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 1.0e-4,
    }


def _quality_mask(*components):
    mask = np.zeros((64, 64), np.uint32)
    for component in components:
        mask[component] |= int(QualityBit.STREAK)
    return mask


class TestChipReplicaGeometry(unittest.TestCase):
    def test_shared_column_is_cleared_and_a_diagonal_is_kept(self):
        shape = (64, 48)
        column = _column(shape, 20)
        diagonal = _diagonal(shape)
        geoms = [
            line_geometry(*np.nonzero(column), shape),
            line_geometry(*np.nonzero(column), shape),
            line_geometry(*np.nonzero(diagonal), shape),
        ]
        cleared = replica_indices(geoms)
        self.assertEqual(cleared, {0, 1})

    def test_catalog_keeps_disconnected_components_separate(self):
        repeated = _column((64, 64), 12)
        unique = np.zeros((64, 64), bool)
        unique[np.arange(30), np.arange(30) + 30] = True

        catalog = _streak_catalog(3, _quality_mask(repeated, unique), _tan_header())

        self.assertEqual(len(catalog), 2)
        self.assertEqual(sorted(component["ys"].size for component in catalog), [30, 64])

    def test_one_replicated_component_does_not_clear_unique_component_on_hdu(self):
        repeated = _column((64, 64), 12)
        unique = np.zeros((64, 64), bool)
        unique[np.arange(30), np.arange(30) + 30] = True
        catalogs = [
            *_streak_catalog(0, _quality_mask(repeated, unique), _tan_header(crpix1=1.0)),
            *_streak_catalog(1, _quality_mask(repeated), _tan_header(crpix1=101.0)),
        ]

        cleared = _replica_component_indices(catalogs)

        self.assertEqual({catalogs[index]["hdu"] for index in cleared}, {0, 1})
        self.assertEqual(len(cleared), 2)
        kept = [catalogs[index] for index in range(len(catalogs)) if index not in cleared]
        self.assertEqual(len(kept), 1)
        self.assertEqual(kept[0]["ys"].size, 30)

    def test_wcs_consistent_cross_chip_trail_survives_equal_local_coordinates(self):
        trail = _column((64, 64), 12)
        catalogs = [
            *_streak_catalog(0, _quality_mask(trail), _tan_header(crpix2=1.0)),
            *_streak_catalog(1, _quality_mask(trail), _tan_header(crpix2=101.0)),
        ]

        self.assertEqual(_replica_component_indices(catalogs), set())

    def test_detector_local_replicas_clear_when_common_wcs_lines_are_incompatible(self):
        replica = _column((64, 64), 12)
        catalogs = [
            *_streak_catalog(0, _quality_mask(replica), _tan_header(crpix1=1.0)),
            *_streak_catalog(1, _quality_mask(replica), _tan_header(crpix1=101.0)),
        ]

        self.assertEqual(_replica_component_indices(catalogs), {0, 1})

    def test_multiple_detector_fixed_peers_are_preserved_as_ambiguous(self):
        replica = _column((64, 64), 12)
        catalogs = []
        for index in range(3):
            catalogs.extend(_streak_catalog(index, _quality_mask(replica), _tan_header(crpix1=1.0 + 100.0 * index)))

        self.assertEqual(_replica_component_indices(catalogs), set())

    def test_conflicting_sky_and_detector_peer_evidence_preserves_group(self):
        replica = _column((64, 64), 12)
        catalogs = [
            *_streak_catalog(0, _quality_mask(replica), _tan_header(crpix1=1.0)),
            *_streak_catalog(1, _quality_mask(replica), _tan_header(crpix1=101.0)),
            *_streak_catalog(2, _quality_mask(replica), _tan_header(crpix1=1.0)),
        ]

        self.assertEqual(_replica_component_indices(catalogs), set())

    def test_missing_wcs_preserves_ambiguous_replicas(self):
        replica = _column((64, 64), 12)
        catalogs = [
            *_streak_catalog(0, _quality_mask(replica), _tan_header()),
            *_streak_catalog(1, _quality_mask(replica), {}),
        ]

        self.assertEqual(_replica_component_indices(catalogs), set())

    def test_noncelestial_wcs_preserves_detector_local_replicas(self):
        replica = _column((64, 64), 12)
        left = {**_tan_header(crpix1=1.0), "CTYPE1": "PIXEL", "CTYPE2": "PIXEL"}
        right = {**_tan_header(crpix1=101.0), "CTYPE1": "PIXEL", "CTYPE2": "PIXEL"}
        catalogs = [
            *_streak_catalog(0, _quality_mask(replica), left),
            *_streak_catalog(1, _quality_mask(replica), right),
        ]

        self.assertEqual(_replica_component_indices(catalogs), set())

    def test_invalid_celestial_wcs_preserves_detector_local_replicas(self):
        replica = _column((64, 64), 12)
        invalid = {**_tan_header(crpix1=101.0), "CD2_2": 0.0}
        catalogs = [
            *_streak_catalog(0, _quality_mask(replica), _tan_header(crpix1=1.0)),
            *_streak_catalog(1, _quality_mask(replica), invalid),
        ]

        self.assertIsNone(catalogs[1]["common"])
        self.assertEqual(_replica_component_indices(catalogs), set())

    def test_incomplete_celestial_wcs_preserves_detector_local_replicas(self):
        replica = _column((64, 64), 12)
        incomplete = _tan_header(crpix1=101.0)
        for key in ("CD1_1", "CD1_2", "CD2_1", "CD2_2"):
            incomplete.pop(key)
        catalogs = [
            *_streak_catalog(0, _quality_mask(replica), _tan_header(crpix1=1.0)),
            *_streak_catalog(1, _quality_mask(replica), incomplete),
        ]

        self.assertIsNone(catalogs[1]["common"])
        self.assertEqual(_replica_component_indices(catalogs), set())


class TestChipReplicaOnDisk(unittest.TestCase):
    def test_two_chips_sharing_a_column_write_no_streak_bit(self):
        shape = (48, 48)

        def column_streak(data_sub, _rms, _existing, _cfg):
            return _column(data_sub.shape, data_sub.shape[1] // 2)

        with tempfile.TemporaryDirectory() as tmp:
            science = os.path.join(tmp, "sci.fits")
            flat = os.path.join(tmp, "flat.fits")
            rng = np.random.default_rng(0)
            image = (1000 + 10 * rng.standard_normal(shape)).astype(np.float32)
            fitsio.write(science, image, header=_tan_header(crpix1=1.0), clobber=True)
            with fitsio.FITS(science, "rw") as handle:
                handle.write(image.copy(), header=_tan_header(crpix1=101.0))
            fitsio.write(flat, np.ones(shape, dtype=np.float32), clobber=True)
            with fitsio.FITS(flat, "rw") as handle:
                handle.write(np.ones(shape, dtype=np.float32))
            paths = {
                "out_map_path": os.path.join(tmp, "o.weight.fits"),
                "out_mask_path": os.path.join(tmp, "o.mask.fits"),
                "out_invvar_path": os.path.join(tmp, "o.ivar.fits"),
                "out_sky_path": None,
                "out_weight_raw_path": None,
                "individual_mask_paths": {},
            }
            args = Namespace(tile_size=32, individual_masks=False, max_workers=1)
            cfg = yaml.safe_load(open("weightmask.yml"))
            cfg["streak_masking"]["enable"] = True
            with patch("weightmask.process.detect_streaks", side_effect=column_streak):
                with fitsio.FITS(science) as hdul, fitsio.FITS(flat) as hdul_flat:
                    n = process_all_hdus([0, 1], hdul, hdul_flat, cfg, paths, args, flat_path=flat, input_path=science)
            self.assertEqual(n, 2)
            with fitsio.FITS(paths["out_mask_path"]) as handle:
                masks = [handle[0].read(), handle[1].read()]
            generations = set()
            for key, prefix in (("out_map_path", "MAP"), ("out_mask_path", "MASK"), ("out_invvar_path", "INVVAR")):
                with fitsio.FITS(paths[key]) as handle:
                    headers = [handle[index].read_header() for index in range(len(handle))]
                self.assertEqual([header.get("EXTNAME") for header in headers], [f"{prefix}_HDU0", f"{prefix}_HDU1"])
                generations.update(header.get("WMGENID") for header in headers)
            self.assertEqual(len(generations), 1)
            self.assertNotIn(None, generations)
            bit = int(QualityBit.STREAK)
            self.assertFalse(bool(np.any(masks[0] & bit)))
            self.assertFalse(bool(np.any(masks[1] & bit)))

    def test_weight_map_without_inverse_variance_does_not_crash_the_veto(self):
        """No inverse-variance product means no ivar writer, hence no positions.

        Looking up `ivar_writer.positions` unconditionally raised AttributeError,
        which escaped `_clear_chip_replicas` (its handler catches only OSError)
        after the mask had already been rewritten -- so the run aborted with the
        mask saying "good" and the weight map still zeroed at the same pixels.
        """
        shape = (48, 48)

        def column_streak(data_sub, _rms, _existing, _cfg):
            return _column(data_sub.shape, data_sub.shape[1] // 2)

        with tempfile.TemporaryDirectory() as tmp:
            science = os.path.join(tmp, "sci.fits")
            flat = os.path.join(tmp, "flat.fits")
            rng = np.random.default_rng(2)
            fitsio.write(science, None, clobber=True)
            with fitsio.FITS(science, "rw") as handle:
                for i in range(2):
                    handle.write(
                        (1000 + 15 * rng.standard_normal(shape)).astype(np.float32),
                        header={**_tan_header(crpix1=1.0 + 100.0 * i), "CCDID": f"CCD{i}"},
                    )
            fitsio.write(flat, None, clobber=True)
            with fitsio.FITS(flat, "rw") as handle:
                for _ in range(2):
                    handle.write(np.ones(shape, dtype=np.float32))
            paths = {
                "out_map_path": os.path.join(tmp, "o.weight.fits"),
                "out_mask_path": os.path.join(tmp, "o.mask.fits"),
                "out_invvar_path": None,  # weight map but no inverse variance
                "out_sky_path": None,
                "out_weight_raw_path": None,
                "individual_mask_paths": {},
            }
            args = Namespace(tile_size=32, individual_masks=False, max_workers=1)
            cfg = yaml.safe_load(open("weightmask.yml"))
            cfg["streak_masking"]["enable"] = True
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                with patch("weightmask.process.detect_streaks", side_effect=column_streak):
                    with fitsio.FITS(science) as hdul, fitsio.FITS(flat) as hdul_flat:
                        n = process_all_hdus(
                            [1, 2], hdul, hdul_flat, cfg, paths, args, flat_path=flat, input_path=science
                        )
            self.assertEqual(n, 2)
            # The mask was rewritten before the restore, so the inability to
            # restore must be reported rather than passing silently.
            self.assertIn("could not be restored", buf.getvalue())

    def test_raw_weight_restore_does_not_depend_on_the_weight_map(self):
        """The raw-weight branch must resolve its own ivar position.

        With ``out_weight_raw_path`` set but no weight map, the raw branch used
        to read an ``ipos`` that was only ever assigned inside the weight-map
        branch: a NameError on the first HDU, or a restore from the previous
        HDU's inverse variance on later ones.
        """
        shape = (48, 48)

        def column_streak(data_sub, _rms, _existing, _cfg):
            return _column(data_sub.shape, data_sub.shape[1] // 2)

        with tempfile.TemporaryDirectory() as tmp:
            science = os.path.join(tmp, "sci.fits")
            flat = os.path.join(tmp, "flat.fits")
            rng = np.random.default_rng(1)
            images = [(1000 + 10 * rng.standard_normal(shape)).astype(np.float32) for _ in range(3)]
            fitsio.write(science, images[0], header=_tan_header(crpix1=1.0), clobber=True)
            with fitsio.FITS(science, "rw") as handle:
                for index, image in enumerate(images[1:], 1):
                    handle.write(image, header=_tan_header(crpix1=1.0 + 100.0 * index))
            fitsio.write(flat, np.ones(shape, dtype=np.float32), clobber=True)
            with fitsio.FITS(flat, "rw") as handle:
                for _ in images:
                    handle.write(np.ones(shape, dtype=np.float32))
            paths = {
                "out_map_path": None,  # no weight map: the raw branch stands alone
                "out_mask_path": os.path.join(tmp, "o.mask.fits"),
                "out_invvar_path": os.path.join(tmp, "o.ivar.fits"),
                "out_sky_path": None,
                "out_weight_raw_path": os.path.join(tmp, "o.raw.fits"),
                "individual_mask_paths": {},
            }
            args = Namespace(tile_size=32, individual_masks=False, max_workers=1)
            cfg = yaml.safe_load(open("weightmask.yml"))
            cfg["streak_masking"]["enable"] = True
            with patch("weightmask.process.detect_streaks", side_effect=column_streak):
                with fitsio.FITS(science) as hdul, fitsio.FITS(flat) as hdul_flat:
                    n = process_all_hdus(
                        list(range(3)),
                        hdul,
                        hdul_flat,
                        cfg,
                        paths,
                        args,
                        flat_path=flat,
                        input_path=science,
                    )
            self.assertEqual(n, 3)
            with fitsio.FITS(paths["out_weight_raw_path"]) as handle:
                raw = [handle[i].read() for i in range(len(handle))]
            self.assertEqual(len(raw), 3)
            generations = set()
            for key, prefix in (
                ("out_mask_path", "MASK"),
                ("out_invvar_path", "INVVAR"),
                ("out_weight_raw_path", "WEIGHT"),
            ):
                with fitsio.FITS(paths[key]) as handle:
                    headers = [handle[index].read_header() for index in range(len(handle))]
                self.assertEqual(
                    [header.get("EXTNAME") for header in headers],
                    [f"{prefix}_HDU{index}" for index in range(3)],
                )
                generations.update(header.get("WMGENID") for header in headers)
            self.assertEqual(len(generations), 1)
            self.assertNotIn(None, generations)
            # Each HDU's raw weight must be its own, not a copy of its neighbour's.
            for i in range(1, 3):
                self.assertFalse(np.array_equal(raw[i], raw[0]))


if __name__ == "__main__":
    unittest.main()
