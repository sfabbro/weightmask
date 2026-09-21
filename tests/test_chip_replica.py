"""Chip-replica veto: the same CCD-local line on two chips is not a sky trail."""

import os
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import patch

import fitsio
import numpy as np
import yaml

from weightmask.contract import QualityBit
from weightmask.mef import apply_chip_replica_veto, process_all_hdus


def _column(shape, x):
    mask = np.zeros(shape, dtype=bool)
    mask[:, x] = True
    return mask


def _diagonal(shape):
    mask = np.zeros(shape, dtype=bool)
    for i in range(shape[0]):
        mask[i, min(shape[1] - 1, i + 3)] = True
    return mask


class TestChipReplicaGeometry(unittest.TestCase):
    def test_shared_column_is_cleared_and_a_diagonal_is_kept(self):
        shape = (64, 48)
        column = _column(shape, 20)
        diagonal = _diagonal(shape)

        def _quality(streak):
            quality = np.zeros(shape, dtype=np.uint32)
            quality[streak] = np.uint32(QualityBit.STREAK)
            return quality

        records = [
            {"streak": column.copy(), "quality": _quality(column)},
            {"streak": column.copy(), "quality": _quality(column)},
            {"streak": diagonal.copy(), "quality": _quality(diagonal)},
        ]
        cleared = apply_chip_replica_veto(records)
        self.assertEqual(cleared, {0, 1})
        self.assertFalse(bool(np.any(records[0]["streak"])))
        self.assertFalse(bool(np.any(records[1]["quality"] & int(QualityBit.STREAK))))
        self.assertTrue(bool(np.any(records[2]["streak"])))


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
            fitsio.write(science, image, clobber=True)
            with fitsio.FITS(science, "rw") as handle:
                handle.write(image.copy())
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
            bit = int(QualityBit.STREAK)
            self.assertFalse(bool(np.any(masks[0] & bit)))
            self.assertFalse(bool(np.any(masks[1] & bit)))


if __name__ == "__main__":
    unittest.main()
