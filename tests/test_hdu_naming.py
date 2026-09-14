"""Output HDU naming and the bounded per-HDU memory window (Phase 3).

Products used to be named from HDU position alone (``MAP_HDU1``) because the
input EXTNAME is a compression artifact; they now carry the CCD identifier the
header provides. ``process_all_hdus`` also streams each HDU out instead of
holding the whole MEF, which these tests pin directly.
"""

import os
import tempfile
import threading
import unittest
from argparse import Namespace
from unittest.mock import patch

import fitsio
import numpy as np

from tests.test_parallel import _load_cfg
from weightmask import mef
from weightmask.mef import _hdu_identifier, process_all_hdus


def _write_mef_with_ccd_ids(path, ccd_ids, shape=(32, 32), seed=0):
    rng = np.random.default_rng(seed)
    fitsio.write(path, None, clobber=True)
    with fitsio.FITS(path, "rw") as handle:
        for ccd in ccd_ids:
            data = (1000 + 30 * rng.standard_normal(shape)).astype(np.float32)
            handle.write(data, header={"CCDNAME": ccd})


def _write_flat(path, n_hdus, shape=(32, 32)):
    fitsio.write(path, None, clobber=True)
    with fitsio.FITS(path, "rw") as handle:
        for _ in range(n_hdus):
            handle.write(np.ones(shape, dtype=np.float32))


def _paths(out_dir, tag):
    return {
        "out_map_path": os.path.join(out_dir, f"{tag}.weight.fits"),
        "out_mask_path": os.path.join(out_dir, f"{tag}.mask.fits"),
        "out_invvar_path": None,
        "out_sky_path": None,
        "out_weight_raw_path": None,
        "individual_mask_paths": {},
    }


class TestHduIdentifier(unittest.TestCase):
    def test_prefers_the_ccd_id(self):
        self.assertEqual(_hdu_identifier(None, {"CCDNAME": "8341-7-5"}, 1), "8341-7-5")
        self.assertEqual(_hdu_identifier(None, {"CCDNAME": " 8341-7-5 "}, 1), "8341-7-5")
        self.assertEqual(_hdu_identifier(None, {"CCDNAME": b"8341-7-5"}, 1), "8341-7-5")

    def test_falls_back_through_the_known_keys(self):
        self.assertEqual(_hdu_identifier(None, {"CCDNAME": "  ", "CCDNAM": "amp-2"}, 1), "amp-2")
        self.assertEqual(_hdu_identifier(None, {"CCDNAM": " 8341-7-5 "}, 1), "8341-7-5")

    def test_the_detector_model_is_not_an_identifier(self):
        """``CCD`` is the detector model on MegaCam, identical for every HDU.

        Naming products from it would give all 36 HDUs one EXTNAME, so it is
        not part of the chain and such headers fall back to the index.
        """
        self.assertEqual(_hdu_identifier(None, {"CCD": "Marconi/EEV CCD42-90"}, 1), "HDU1")
        self.assertEqual(_hdu_identifier(None, {"CCDNAME": " ", "CCD": "model-x"}, 5), "HDU5")

    def test_falls_back_to_the_hdu_name_then_the_index(self):
        class Named:
            name = "SCI"

        self.assertEqual(_hdu_identifier(Named(), {}, 0), "SCI")
        self.assertEqual(_hdu_identifier(None, {"EXTNAME": "SCI"}, 1), "HDU1")
        self.assertEqual(_hdu_identifier(None, None, 7), "HDU7")
        self.assertEqual(_hdu_identifier(None, MagicHeader(), 2), "HDU2")


class MagicHeader:
    """A header stand-in whose ``get`` raises, as fitsio's does on bad cards."""

    def get(self, _key, _default=None):
        raise RuntimeError("boom")


class TestOutputHduNaming(unittest.TestCase):
    def test_products_are_named_after_the_ccd_id(self):
        ccd_ids = ["8341-7-1", "8341-7-2", "8341-7-3"]
        with tempfile.TemporaryDirectory() as tmp:
            in_path = os.path.join(tmp, "science.fits")
            flat_path = os.path.join(tmp, "flat.fits")
            _write_mef_with_ccd_ids(in_path, ccd_ids)
            _write_flat(flat_path, len(ccd_ids))
            paths = _paths(tmp, "named")
            args = Namespace(tile_size=1024, individual_masks=False, max_workers=2)
            with fitsio.FITS(in_path) as hdul_in, fitsio.FITS(flat_path) as hdul_flat:
                n = process_all_hdus(
                    list(range(1, len(ccd_ids) + 1)),
                    hdul_in,
                    hdul_flat,
                    _load_cfg(),
                    paths,
                    args,
                    flat_path=flat_path,
                    max_workers=2,
                    input_path=in_path,
                )
            self.assertEqual(n, len(ccd_ids))
            with fitsio.FITS(paths["out_map_path"]) as handle:
                names = [handle[i].read_header().get("EXTNAME") for i in range(len(handle))]
            self.assertEqual(names, [None] + [f"MAP_{ccd}" for ccd in ccd_ids])
            with fitsio.FITS(paths["out_mask_path"]) as handle:
                names = [handle[i].read_header().get("EXTNAME") for i in range(len(handle))]
            self.assertEqual(names, [None] + [f"MASK_{ccd}" for ccd in ccd_ids])


class TestBoundedHduWindow(unittest.TestCase):
    def test_products_are_released_per_hdu_not_accumulated(self):
        """At most one worker-window of HDUs may be held at once.

        The previous implementation filled a ``results`` dict for every HDU and
        only wrote afterwards, which held the whole MEF's products (~8 GB for
        36 MegaPrime CCDs).
        """
        n_hdus = 24
        max_workers = 3
        with tempfile.TemporaryDirectory() as tmp:
            in_path = os.path.join(tmp, "bounded.fits")
            flat_path = os.path.join(tmp, "bounded_flat.fits")
            _write_mef_with_ccd_ids(in_path, [f"ccd-{i}" for i in range(1, n_hdus + 1)], shape=(24, 24))
            _write_flat(flat_path, n_hdus, shape=(24, 24))
            paths = _paths(tmp, "bounded")
            args = Namespace(tile_size=1024, individual_masks=False, max_workers=max_workers)

            live = {"now": 0, "peak": 0}
            lock = threading.Lock()

            class Held:
                """Stands in for one HDU's product arrays."""

                def __init__(self):
                    with lock:
                        live["now"] += 1
                        live["peak"] = max(live["peak"], live["now"])

                def __del__(self):
                    with lock:
                        live["now"] -= 1

            def fake_process_hdu(_hdu_sci, _hdu_flat, _config, _hdu_index, tile_size=1024, **_kwargs):
                shape = (8, 8)
                held = Held()
                header_info = {"individual_masks": {}, "_held": held, "timings": {}}
                ones = np.ones(shape, dtype=np.float32)
                return (
                    np.zeros(shape, dtype=bool),
                    ones,
                    ones,
                    ones,
                    ones,
                    header_info,
                )

            held_at_flush: list = []

            with (
                patch.object(mef, "process_hdu", side_effect=fake_process_hdu),
                patch.object(mef, "_store_output_maps"),
                patch.object(mef, "_make_output_writers", return_value={}),
                patch.object(mef, "_flush_hdu_output", side_effect=lambda *a, **k: held_at_flush.append(live["now"])),
            ):
                with fitsio.FITS(in_path) as hdul_in, fitsio.FITS(flat_path) as hdul_flat:
                    n = process_all_hdus(
                        list(range(1, n_hdus + 1)),
                        hdul_in,
                        hdul_flat,
                        _load_cfg(),
                        paths,
                        args,
                        flat_path=flat_path,
                        max_workers=max_workers,
                        input_path=in_path,
                    )

            self.assertEqual(n, n_hdus)
            self.assertEqual(len(held_at_flush), n_hdus)
            self.assertGreater(live["peak"], 0)
            self.assertLessEqual(
                live["peak"],
                max_workers + 1,
                "products accumulate across HDUs instead of streaming out",
            )


if __name__ == "__main__":
    unittest.main()
