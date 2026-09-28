"""Parallel vs sequential equivalence (Step 1).

Threaded 36/40-HDU MEFs must match single-thread output byte-identically;
dead-CCD MEFs must not assume 36; single-HDU files use the per-file path.
"""

import contextlib
import io
import os
import tempfile
import unittest
from argparse import Namespace

import fitsio
import numpy as np
import yaml

from weightmask.mef import _resolve_max_workers, process_all_hdus


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
            idx,
            hi,
            hf,
            cfg,
            paths,
            args,
            flat_path=flat_path,
            max_workers=max_workers,
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

    def test_short_flat_mef_fails_loudly_instead_of_using_a_unit_flat(self):
        """A flat shorter than the science MEF must not degrade silently.

        With `hdu_flat_obj = None`, process_image substitutes a unit flat and
        prints only an INFO line, so every affected CCD gets unflat-fielded
        weights and the run still reports success.
        """
        with tempfile.TemporaryDirectory() as tmp:
            sp = os.path.join(tmp, "sci.fits")
            fp = os.path.join(tmp, "flat_one.fits")
            shape = (48, 48)
            _write_mef(sp, 3, shape=shape)
            fitsio.write(fp, np.full(shape, 0.5, dtype=np.float32), clobber=True)
            paths = {
                "out_map_path": os.path.join(tmp, "o.weight.fits"),
                "out_mask_path": os.path.join(tmp, "o.mask.fits"),
                "out_invvar_path": None,
                "out_sky_path": None,
                "out_weight_raw_path": None,
                "individual_mask_paths": {},
            }
            args = Namespace(tile_size=1024, individual_masks=False, max_workers=1)
            cfg = _load_cfg()
            with fitsio.FITS(sp) as hi, fitsio.FITS(fp) as hf:
                n = process_all_hdus([1, 2, 3], hi, hf, cfg, paths, args, flat_path=fp, input_path=sp)
            self.assertEqual(n, 0)

    def test_short_flat_mef_reports_why(self):
        """The failure must name the length mismatch, not a downstream error."""
        with tempfile.TemporaryDirectory() as tmp:
            sp = os.path.join(tmp, "sci.fits")
            fp = os.path.join(tmp, "flat_one.fits")
            shape = (48, 48)
            _write_mef(sp, 3, shape=shape)
            fitsio.write(fp, np.full(shape, 0.5, dtype=np.float32), clobber=True)
            paths = {
                "out_map_path": os.path.join(tmp, "o.weight.fits"),
                "out_mask_path": os.path.join(tmp, "o.mask.fits"),
                "out_invvar_path": None,
                "out_sky_path": None,
                "out_weight_raw_path": None,
                "individual_mask_paths": {},
            }
            args = Namespace(tile_size=1024, individual_masks=False, max_workers=1)
            cfg = _load_cfg()
            buf = io.StringIO()
            with fitsio.FITS(sp) as hi, fitsio.FITS(fp) as hf:
                with contextlib.redirect_stdout(buf):
                    process_all_hdus([1, 2, 3], hi, hf, cfg, paths, args, flat_path=fp, input_path=sp)
            log = buf.getvalue()
            self.assertIn("flat", log)
            self.assertIn("HDU(s), need index", log)
            # The old bug: the reason was a downstream TypeError from calling
            # len() on an already-closed fitsio handle.
            self.assertNotIn("NoneType", log)

    def test_short_dark_mef_warns_instead_of_silently_skipping(self):
        with tempfile.TemporaryDirectory() as tmp:
            sp = os.path.join(tmp, "sci.fits")
            fp = os.path.join(tmp, "flat.fits")
            dp = os.path.join(tmp, "dark_one.fits")
            shape = (48, 48)
            _write_mef(sp, 3, shape=shape)
            _write_flat(fp, 3, shape=shape)
            fitsio.write(dp, None, clobber=True)
            with fitsio.FITS(dp, "rw") as handle:
                handle.write(np.full(shape, 100.0, dtype=np.float32))
            paths = {
                "out_map_path": os.path.join(tmp, "o.weight.fits"),
                "out_mask_path": os.path.join(tmp, "o.mask.fits"),
                "out_invvar_path": None,
                "out_sky_path": None,
                "out_weight_raw_path": None,
                "individual_mask_paths": {},
            }
            args = Namespace(tile_size=1024, individual_masks=False, max_workers=1)
            cfg = _load_cfg()
            buf = io.StringIO()
            # The *_path arguments are filename overrides for an already-open
            # handle, not standalone inputs: the handle must be passed too, as
            # the CLI does.
            with fitsio.FITS(sp) as hi, fitsio.FITS(fp) as hf, fitsio.FITS(dp) as hd:
                with contextlib.redirect_stdout(buf):
                    process_all_hdus(
                        [1, 2, 3],
                        hi,
                        hf,
                        cfg,
                        paths,
                        args,
                        flat_path=fp,
                        input_path=sp,
                        hdul_dark=hd,
                        dark_path=dp,
                    )
            log = buf.getvalue()
            self.assertIn("dark frame", log)
            self.assertIn("No dark hot-pixel mask", log)

    def test_scale_to_100_survives_rescale_and_is_labelled_honestly(self):
        """A 0-100 confidence map must not be clipped to 1.0 or labelled 0-1.

        `normalize_scope: per_exposure` is the shipped default, so with
        `scale_to_100: true` the global rescale used to run
        `np.clip(data*factor, 0, 1)` over a 0-100 file, flattening everything
        above 1% of the normalisation to 1.0, while the header still claimed
        `normalized_weight_0_to_1`.
        """
        from weightmask.contract import CONFIDENCE_SEMANTICS_SCALED

        shape = (48, 48)
        with tempfile.TemporaryDirectory() as tmp:
            sp = os.path.join(tmp, "sci.fits")
            fp = os.path.join(tmp, "flat.fits")
            rng = np.random.default_rng(5)
            fitsio.write(sp, None, clobber=True)
            with fitsio.FITS(sp, "rw") as handle:
                for level in (1000.0, 4000.0):  # different p99 per HDU
                    handle.write((level + 30 * rng.standard_normal(shape)).astype(np.float32))
            _write_flat(fp, 2, shape=shape)
            paths = {
                "out_map_path": os.path.join(tmp, "o.conf.fits"),
                "out_mask_path": None,
                "out_invvar_path": None,
                "out_sky_path": None,
                "out_weight_raw_path": None,
                "individual_mask_paths": {},
            }
            args = Namespace(tile_size=1024, individual_masks=False, max_workers=1)
            cfg = _load_cfg()
            cfg["streak_masking"]["enable"] = False
            cfg["output_params"]["output_map_format"] = "confidence"
            cfg["confidence_params"]["scale_to_100"] = True
            with fitsio.FITS(sp) as hi, fitsio.FITS(fp) as hf:
                n = process_all_hdus([1, 2], hi, hf, cfg, paths, args, flat_path=fp, input_path=sp)
            self.assertEqual(n, 2)
            with fitsio.FITS(paths["out_map_path"]) as f:
                maps = [f[1].read(), f[2].read()]
                semantics = f[1].read_header().get("WMSEM")
            for m in maps:
                self.assertGreater(float(np.max(m)), 1.5, "0-100 map was clipped flat")
            self.assertEqual(semantics, CONFIDENCE_SEMANTICS_SCALED)

    def test_resolve_workers(self):
        import os as _os

        self.assertEqual(_resolve_max_workers(0, 36), 1)
        self.assertEqual(_resolve_max_workers(1, 36), 1)
        self.assertEqual(_resolve_max_workers(4, 2), 2)
        self.assertEqual(_resolve_max_workers(None, 100), min(8, _os.cpu_count() or 4))

    def test_small_format_generic_defaults(self):
        # 512x512 vignetted flat + dead column, GAINA-only headers (no MegaPrime flat/GAIN).
        shape = (512, 512)
        with tempfile.TemporaryDirectory() as tmp:
            sp = os.path.join(tmp, "small.fits")
            fp = os.path.join(tmp, "small_flat.fits")
            rng = np.random.default_rng(7)
            fitsio.write(sp, None, clobber=True)
            with fitsio.FITS(sp, "rw") as f:
                for _ in range(4):
                    d = (1000 + 30 * rng.standard_normal(shape)).astype(np.float32)
                    f.write(d, header={"GAINA": 2.0, "RDNOIS": 6.0, "SATUR": 50000.0})
            yy, xx = np.mgrid[0:512, 0:512]
            flat1 = (0.8 + 0.2 * np.cos((xx - 256) / 256 * np.pi / 2) ** 2).astype(np.float32)
            flat1[:, 100] = 0.0
            fitsio.write(fp, None, clobber=True)
            with fitsio.FITS(fp, "rw") as f:
                for _ in range(4):
                    f.write(flat1)
            w_seq, m_seq = _run(sp, fp, tmp, "seqsmall", max_workers=1)
            w_par, m_par = _run(sp, fp, tmp, "parsmall", max_workers=4)
            self.assertEqual(len(w_seq), 4)
            for a, b in zip(w_seq, w_par):
                np.testing.assert_array_equal(a, b)
            for a, b in zip(m_seq, m_par):
                np.testing.assert_array_equal(a, b)
            # Dead column flagged BAD; wire dtypes uint16/float32.
            self.assertTrue(bool(((m_seq[0][:, 100] & 1) != 0).all()))
            with fitsio.FITS(os.path.join(tmp, "seqsmall.mask.fits")) as f:
                for i in range(len(f)):
                    if f[i].get_info().get("ndims") == 2:
                        self.assertEqual(f[i].read().dtype, np.uint16)
                        break
            with fitsio.FITS(os.path.join(tmp, "seqsmall.weight.fits")) as f:
                for i in range(len(f)):
                    if f[i].get_info().get("ndims") == 2:
                        self.assertEqual(f[i].read().dtype, np.float32)
                        break

    def test_parallel_different_gain_does_not_crosstalk(self):
        shape = (48, 48)
        with tempfile.TemporaryDirectory() as tmp:
            sp = os.path.join(tmp, "sgain.fits")
            fp = os.path.join(tmp, "fgain.fits")
            rng = np.random.default_rng(4)
            fitsio.write(sp, None, clobber=True)
            with fitsio.FITS(sp, "rw") as f:
                for gain in (1.0, 4.0):
                    d = (1000 + 30 * rng.standard_normal(shape)).astype(np.float32)
                    f.write(d, header={"GAIN": gain, "RDNOISE": 5.0})
            fitsio.write(fp, None, clobber=True)
            with fitsio.FITS(fp, "rw") as f:
                for _ in range(2):
                    f.write(np.ones(shape, dtype=np.float32))
            paths = {
                "out_map_path": os.path.join(tmp, "g.weight.fits"),
                "out_mask_path": os.path.join(tmp, "g.mask.fits"),
                "out_invvar_path": os.path.join(tmp, "g.ivar.fits"),
                "out_sky_path": None,
                "out_weight_raw_path": None,
                "individual_mask_paths": {},
            }
            args = Namespace(tile_size=32, individual_masks=False, max_workers=2)
            cfg = _load_cfg()
            cfg["variance"]["rescale_variance"] = False
            with fitsio.FITS(sp) as hi, fitsio.FITS(fp) as hf:
                n = process_all_hdus([1, 2], hi, hf, cfg, paths, args, flat_path=fp, max_workers=2, input_path=sp)
            self.assertEqual(n, 2)
            with fitsio.FITS(paths["out_invvar_path"]) as f:
                ivars = [f[i].read() for i in range(len(f)) if i > 0 or len(f) == 1]
            med = [float(np.median(a[a > 0])) for a in ivars]
            self.assertEqual(len(med), 2)
            ratio = med[1] / med[0]
            self.assertGreater(ratio, 3.0)
            self.assertLess(ratio, 5.5)


if __name__ == "__main__":
    unittest.main()
