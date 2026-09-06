"""Parallel vs sequential equivalence (Step 1).

Threaded 36/40-HDU MEFs must match single-thread output byte-identically;
dead-CCD MEFs must not assume 36; single-HDU files use the per-file path.
"""

import os
import tempfile
import unittest
from argparse import Namespace

import fitsio
import numpy as np
import yaml

from weightmask.cli import _resolve_max_workers, process_all_hdus


def _load_cfg():
    cfg = yaml.safe_load(open("weightmask.yml"))
    cfg["streak_masking"]["enable"] = False
    cfg["output_params"]["output_map_format"] = "weight"
    return cfg


def _write_mef(path, n_hdus, shape=(48, 48), seed=0, dead_hdu=None, flat_dead_val=0.01):
    rng = np.random.default_rng(seed)
    # Primary empty (MegaCam layout: primary not an image)
    fitsio.write(path, None, clobber=True)
    with fitsio.FITS(path, "rw") as f:
        for _ in range(n_hdus):
            data = (1000 + 30 * rng.standard_normal(shape)).astype(np.float32)
            f.write(data)


def _write_flat(path, n_hdus, shape=(48, 48), dead_hdu=None, flat_dead_val=0.01):
    fitsio.write(path, None, clobber=True)
    with fitsio.FITS(path, "rw") as f:
        for i in range(n_hdus):
            if dead_hdu is not None and i == dead_hdu:
                f.write(np.full(shape, flat_dead_val, dtype=np.float32))
            else:
                f.write(np.ones(shape, dtype=np.float32))


def _run(in_path, flat_path, out_dir, tag, max_workers):
    paths = {
        "out_map_path": os.path.join(out_dir, f"{tag}.weight.fits"),
        "out_mask_path": os.path.join(out_dir, f"{tag}.mask.fits"),
        "out_invvar_path": None,
        "out_sky_path": None,
        "out_weight_raw_path": None,
        "individual_mask_paths": {},
    }
    args = Namespace(tile_size=1024, individual_masks=False, max_workers=max_workers)
    cfg = _load_cfg()
    with fitsio.FITS(in_path) as hi, fitsio.FITS(flat_path) as hf:
        idx = list(range(len(hi)))[1:]  # skip empty primary
        # hdus_to_process are 1-based for empty-primary MEFs
        n = process_all_hdus(
            idx, hi, hf, cfg, paths, args, flat_path=flat_path, max_workers=max_workers,
            input_path=in_path,
        )
    assert n == len(idx), f"{tag}: processed {n} != {len(idx)}"
    with fitsio.FITS(paths["out_map_path"]) as f:
        w = [f[i].read() for i in range(len(f)) if i > 0 or len(f) == 1]
    with fitsio.FITS(paths["out_mask_path"]) as f:
        m = [f[i].read() for i in range(len(f)) if i > 0 or len(f) == 1]
    return w, m


class TestParallelEquivalence(unittest.TestCase):
    def test_36hdu_threaded_matches_sequential(self):
        with tempfile.TemporaryDirectory() as tmp:
            sp = os.path.join(tmp, "s36.fits")
            fp = os.path.join(tmp, "f36.fits")
            _write_mef(sp, 36)
            _write_flat(fp, 36)
            w_seq, m_seq = _run(sp, fp, tmp, "seq36", max_workers=1)
            w_par, m_par = _run(sp, fp, tmp, "par36", max_workers=4)
            self.assertEqual(len(w_seq), 36)
            for a, b in zip(w_seq, w_par):
                np.testing.assert_array_equal(a, b)
            for a, b in zip(m_seq, m_par):
                np.testing.assert_array_equal(a, b)

    def test_40hdu_threaded_matches_sequential(self):
        with tempfile.TemporaryDirectory() as tmp:
            sp = os.path.join(tmp, "s40.fits")
            fp = os.path.join(tmp, "f40.fits")
            _write_mef(sp, 40, seed=1)
            _write_flat(fp, 40)
            w_seq, m_seq = _run(sp, fp, tmp, "seq40", max_workers=1)
            w_par, m_par = _run(sp, fp, tmp, "par40", max_workers=8)
            self.assertEqual(len(w_seq), 40)
            for a, b in zip(w_seq, w_par):
                np.testing.assert_array_equal(a, b)
            for a, b in zip(m_seq, m_par):
                np.testing.assert_array_equal(a, b)

    def test_dead_ccd_no_hardcoded_36(self):
        with tempfile.TemporaryDirectory() as tmp:
            sp = os.path.join(tmp, "sdead.fits")
            fp = os.path.join(tmp, "fdead.fits")
            _write_mef(sp, 36, seed=2)
            _write_flat(fp, 36, dead_hdu=5)
            w_seq, m_seq = _run(sp, fp, tmp, "seqdead", max_workers=1)
            w_par, m_par = _run(sp, fp, tmp, "pardead", max_workers=4)
            self.assertEqual(len(w_seq), 36)
            for a, b in zip(w_seq, w_par):
                np.testing.assert_array_equal(a, b)
            for a, b in zip(m_seq, m_par):
                np.testing.assert_array_equal(a, b)
            # Dead CCD veto flags the whole HDU BAD (bit 0).
            dead_mask = m_seq[5]
            self.assertTrue(bool(((dead_mask & 1) != 0).all()))

    def test_single_hdu_per_file_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            sp = os.path.join(tmp, "single.fits")
            rng = np.random.default_rng(3)
            sci = (1000 + 30 * rng.standard_normal((48, 48))).astype(np.float32)
            fitsio.write(sp, sci, clobber=True)
            fp = os.path.join(tmp, "fsingle.fits")
            fitsio.write(fp, np.ones((48, 48), dtype=np.float32), clobber=True)
            paths = {
                "out_map_path": os.path.join(tmp, "o.weight.fits"),
                "out_mask_path": os.path.join(tmp, "o.mask.fits"),
                "out_invvar_path": None,
                "out_sky_path": None,
                "out_weight_raw_path": None,
                "individual_mask_paths": {},
            }
            args = Namespace(tile_size=1024, individual_masks=False, max_workers=8)
            cfg = _load_cfg()
            with fitsio.FITS(sp) as hi, fitsio.FITS(fp) as hf:
                n = process_all_hdus([0], hi, hf, cfg, paths, args, flat_path=fp, input_path=sp)
            self.assertEqual(n, 1)
            self.assertTrue(os.path.exists(paths["out_map_path"]))

    def test_resolve_workers(self):
        import os as _os

        self.assertEqual(_resolve_max_workers(0, 36), 1)
        self.assertEqual(_resolve_max_workers(1, 36), 1)
        self.assertEqual(_resolve_max_workers(4, 2), 2)
        self.assertEqual(_resolve_max_workers(None, 100), min(8, _os.cpu_count() or 4))


if __name__ == "__main__":
    unittest.main()
