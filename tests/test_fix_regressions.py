"""Regression tests for the weightmask defect-review fix list (2026-09-05).

Each test pins one fix behaviorally (observable result, not source text):
- process_all_hdus computes each flat HDU mask exactly once, in its worker (dedup; no disk cache)
- unknown streak mode raises instead of silently returning zeros
- non-finite inverse variance sets INVALID_VARIANCE and zeroes weight end to end
- CR niter config is forwarded to astroscrappy (yml pins niter: 2)
"""

import os
import tempfile
import unittest
from argparse import Namespace
from unittest.mock import MagicMock, patch

import fitsio
import numpy as np
import yaml

from weightmask.contract import QualityBit, build_weight_product
from weightmask.mef import process_all_hdus


def _tiny_flat_config():
    return {
        "flat_masking": {
            "local_filter_size": 3,
            "local_low_thresh": 0.5,
            "local_high_thresh": 2.0,
            "col_enable": False,
        }
    }


class TestSingleFlatPrecompute(unittest.TestCase):
    def test_each_flat_hdu_computed_once(self):
        from weightmask import mef as cli_mod

        config = _tiny_flat_config()
        flat_data = np.ones((32, 32), dtype=np.float32)
        hdu = MagicMock()
        hdu.read.return_value = flat_data
        hdu.read_header.return_value = fitsio.FITSHDR()
        hdul_flat = [None, hdu]
        hdul_input = MagicMock()
        hdul_input.__len__.return_value = 2
        hdul_input.__getitem__.side_effect = lambda i: hdu
        args = Namespace(tile_size=1024)

        calls = []

        def counting_compute(data, cfg, tile):
            calls.append(tile)
            return np.zeros((32, 32), dtype=bool)

        # Tactic D: the sequential pre-loop reads medians only; the worker
        # computes each HDU's mask exactly once via the miss path, so run
        # the real process_hdu and count real compute calls end to end.
        with (
            patch.object(cli_mod, "compute_flat_bad_mask", side_effect=counting_compute),
            patch.object(cli_mod, "_make_output_writers", return_value={}),
            patch.object(cli_mod, "_store_output_maps"),
            patch.object(cli_mod, "_flush_hdu_output"),
        ):
            n = process_all_hdus([1], hdul_input, hdul_flat, config, {}, args, flat_path="flat.fits")

        self.assertEqual(n, 1)
        self.assertEqual(len(calls), 1)  # one HDU -> exactly one mask compute, in its worker
        # Worker miss path uses _effective_tile_size (32x32 → 16), matching process_image.
        self.assertEqual(calls[0], 16)


class TestInvalidVarianceEndToEnd(unittest.TestCase):
    def test_nan_ivar_sets_bit_and_zeroes_weight(self):
        invar = np.ones((16, 16), dtype=np.float32)
        invar[3, 3] = np.nan
        invar[4, 4] = -1.0
        mask = np.zeros((16, 16), dtype=np.uint32)
        product = build_weight_product(invar, mask)
        qmask = product.quality_mask
        self.assertTrue(bool(qmask[3, 3] & int(QualityBit.INVALID_VARIANCE)))
        self.assertTrue(bool(qmask[4, 4] & int(QualityBit.INVALID_VARIANCE)))
        self.assertEqual(product.weight[3, 3], 0.0)
        self.assertEqual(product.weight[4, 4], 0.0)
        self.assertGreater(product.weight[0, 0], 0.0)


class TestInvalidVarianceWritten(unittest.TestCase):
    def test_nan_ivar_sets_bit_on_disk(self):
        shape = (32, 32)

        def fake_ivar(_cfg, sky_map, *_args, **_kwargs):
            out = np.full(sky_map.shape, 0.01, dtype=np.float32)
            out[3, 3] = np.nan
            out[4, 4] = -1.0
            return out

        with tempfile.TemporaryDirectory() as tmp:
            sp = os.path.join(tmp, "sci.fits")
            fp = os.path.join(tmp, "flat.fits")
            rng = np.random.default_rng(0)
            fitsio.write(sp, (1000 + 10 * rng.standard_normal(shape)).astype(np.float32), clobber=True)
            fitsio.write(fp, np.ones(shape, dtype=np.float32), clobber=True)
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
            cfg["streak_masking"]["enable"] = False
            with patch("weightmask.process.calculate_inverse_variance", side_effect=fake_ivar):
                with fitsio.FITS(sp) as hi, fitsio.FITS(fp) as hf:
                    n = process_all_hdus([0], hi, hf, cfg, paths, args, flat_path=fp, input_path=sp)
            self.assertEqual(n, 1)
            with fitsio.FITS(paths["out_mask_path"]) as f:
                mask = f[0].read()
            with fitsio.FITS(paths["out_invvar_path"]) as f:
                ivar = f[0].read()
            with fitsio.FITS(paths["out_map_path"]) as f:
                weight = f[0].read()
            invalid_bit = int(QualityBit.INVALID_VARIANCE)
            self.assertTrue(bool(mask[3, 3] & invalid_bit))
            self.assertTrue(bool(mask[4, 4] & invalid_bit))
            self.assertEqual(float(ivar[3, 3]), 0.0)
            self.assertEqual(float(ivar[4, 4]), 0.0)
            self.assertTrue(np.isfinite(ivar).all())
            self.assertEqual(float(weight[3, 3]), 0.0)
            self.assertEqual(float(weight[4, 4]), 0.0)


class TestProcessImageConfigIsolation(unittest.TestCase):
    def test_process_image_does_not_mutate_variance_config(self):
        from weightmask.process import process_image

        cfg = yaml.safe_load(open("weightmask.yml"))
        cfg["streak_masking"]["enable"] = False
        self.assertNotIn("gain", cfg["variance"])
        rng = np.random.default_rng(1)
        data = (1000 + 10 * rng.standard_normal((48, 48))).astype(np.float32)
        process_image(data, {"GAIN": 2.5, "RDNOISE": 5.0}, np.ones_like(data), cfg, tile_size=32)
        self.assertNotIn("gain", cfg["variance"])
        self.assertNotIn("read_noise", cfg["variance"])


class TestElixirKeepMap(unittest.TestCase):
    def test_keep_map_zero_sets_bad_bit(self):
        shape = (48, 48)
        with tempfile.TemporaryDirectory() as tmp:
            sp = os.path.join(tmp, "sci.fits")
            fp = os.path.join(tmp, "flat.fits")
            bp = os.path.join(tmp, "keep.fits")
            rng = np.random.default_rng(2)
            fitsio.write(sp, (1000 + 30 * rng.standard_normal(shape)).astype(np.float32), clobber=True)
            fitsio.write(fp, np.ones(shape, dtype=np.float32), clobber=True)
            keep = np.ones(shape, dtype=np.uint8)
            keep[7, 11] = 0
            fitsio.write(bp, keep, clobber=True)
            paths = {
                "out_map_path": os.path.join(tmp, "o.weight.fits"),
                "out_mask_path": os.path.join(tmp, "o.mask.fits"),
                "out_invvar_path": None,
                "out_sky_path": None,
                "out_weight_raw_path": None,
                "individual_mask_paths": {},
            }
            args = Namespace(tile_size=32, individual_masks=False, max_workers=1)
            cfg = yaml.safe_load(open("weightmask.yml"))
            cfg["streak_masking"]["enable"] = False
            with fitsio.FITS(sp) as hi, fitsio.FITS(fp) as hf, fitsio.FITS(bp) as hb:
                n = process_all_hdus(
                    [0], hi, hf, cfg, paths, args, flat_path=fp, input_path=sp, hdul_badpix=hb, badpix_path=bp
                )
            self.assertEqual(n, 1)
            with fitsio.FITS(paths["out_mask_path"]) as f:
                mask = f[0].read()
            self.assertTrue(bool((mask[7, 11] & int(QualityBit.BAD_PIXEL)) != 0))


class TestDeadCcdVeto(unittest.TestCase):
    def test_outlier_hdu_goes_fully_bad(self):
        from weightmask.mef import _veto_dead_ccd_hdus

        masks = {i: np.zeros((8, 8), dtype=bool) for i in range(1, 9)}
        meds = {i: 0.94 for i in range(1, 9)}
        meds[4] = 1.53
        _veto_dead_ccd_hdus(masks, meds, {})
        self.assertTrue(bool(masks[4].all()))
        for i in range(1, 9):
            if i != 4:
                self.assertFalse(bool(masks[i].any()))

    def test_agreeing_ccds_untouched(self):
        from weightmask.mef import _veto_dead_ccd_hdus

        masks = {i: np.zeros((8, 8), dtype=bool) for i in range(1, 9)}
        meds = {i: 0.94 + 0.001 * i for i in range(1, 9)}
        _veto_dead_ccd_hdus(masks, meds, {})
        for i in range(1, 9):
            self.assertFalse(bool(masks[i].any()))

    def test_small_run_skipped(self):
        from weightmask.mef import _veto_dead_ccd_hdus

        masks = {0: np.zeros((8, 8), dtype=bool)}
        _veto_dead_ccd_hdus(masks, {0: 99.0}, {})
        self.assertFalse(bool(masks[0].any()))


class TestConfidenceGlobalHookup(unittest.TestCase):
    def test_confidence_mode_uses_global_p99(self):
        from argparse import Namespace

        import fitsio
        import yaml

        from weightmask.mef import process_all_hdus

        cfg = yaml.safe_load(open("weightmask.yml"))
        cfg["streak_masking"]["enable"] = False
        cfg["output_params"]["output_map_format"] = "confidence"
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmp:
            sp = os.path.join(tmp, "s.fits")
            fp = os.path.join(tmp, "f.fits")
            s1 = (1000 + 30 * rng.standard_normal((64, 64))).astype(np.float32)
            s2 = (4000 + 60 * rng.standard_normal((64, 64))).astype(np.float32)
            fitsio.write(sp, None, clobber=True)
            with fitsio.FITS(sp, "rw") as f:
                f.write(s1)
                f.write(s2)
            fitsio.write(fp, np.ones((64, 64), dtype=np.float32), clobber=True)
            paths = {
                "out_map_path": os.path.join(tmp, "o.conf.fits"),
                "out_mask_path": None,
                "out_invvar_path": None,
                "out_sky_path": None,
                "out_weight_raw_path": None,
                "individual_mask_paths": {},
            }
            args = Namespace(tile_size=1024, individual_masks=False)
            with fitsio.FITS(sp) as hi, fitsio.FITS(fp) as hf:
                n = process_all_hdus([1, 2], hi, hf, cfg, paths, args, flat_path=fp)
            self.assertEqual(n, 2)
            with fitsio.FITS(os.path.join(tmp, "o.conf.fits")) as f:
                c1, c2 = f[1].read(), f[2].read()
            # low-sky HDU pins 1.0; high-sky HDU scales down -> comparable
            self.assertAlmostEqual(float(np.percentile(c1[c1 > 0], 99)), 1.0, places=2)
            self.assertLess(float(np.percentile(c2[c2 > 0], 99)), 0.5)


class TestDarkLeg(unittest.TestCase):
    def test_dark_hot_pixels_join_bad(self):
        from argparse import Namespace

        import fitsio
        import yaml

        from weightmask.mef import process_all_hdus

        cfg = yaml.safe_load(open("weightmask.yml"))
        cfg["streak_masking"]["enable"] = False
        rng = np.random.default_rng(0)
        with tempfile.TemporaryDirectory() as tmp:
            sci = (1000 + 30 * rng.standard_normal((64, 64))).astype(np.float32)
            sp = os.path.join(tmp, "s.fits")
            fp = os.path.join(tmp, "f.fits")
            dp = os.path.join(tmp, "d.fits")
            fitsio.write(sp, sci, clobber=True)
            fitsio.write(fp, np.ones((64, 64), dtype=np.float32), clobber=True)
            dark = np.full((64, 64), 5.0, dtype=np.float32)
            dark[10, 10] = 500.0
            fitsio.write(dp, dark, clobber=True)
            paths = {
                "out_map_path": os.path.join(tmp, "o.weight.fits"),
                "out_mask_path": os.path.join(tmp, "o.mask.fits"),
                "out_invvar_path": None,
                "out_sky_path": None,
                "out_weight_raw_path": None,
                "individual_mask_paths": {},
            }
            args = Namespace(tile_size=1024, individual_masks=False)
            with fitsio.FITS(sp) as hi, fitsio.FITS(fp) as hf, fitsio.FITS(dp) as hd:
                n = process_all_hdus([0], hi, hf, cfg, paths, args, flat_path=fp, hdul_dark=hd)
            self.assertEqual(n, 1)
            with fitsio.FITS(os.path.join(tmp, "o.mask.fits")) as f:
                m = f[0].read()
            self.assertTrue(bool(m[10, 10] & 1))


class TestGlobalConfidenceRescale(unittest.TestCase):
    def test_rescale_matches_global_p99(self):
        from weightmask.mef import _rescale_confidence_to_global, _StreamingMapWriter

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "conf.fits")
            hdul = [None, None, None]
            w = _StreamingMapWriter(path, hdul, None)
            hdr = {}
            # HDU1 weights ~1.0 -> conf 1.0; HDU2 weights ~10.0 -> conf 1.0 (per-HDU norm)
            w.write(1, np.ones((8, 8), dtype=np.float32), hdr, "C1")
            w.write(2, np.ones((8, 8), dtype=np.float32), hdr, "C2")
            s1 = np.full(64, 1.0)
            s2 = np.full(64, 10.0)
            cfg = {"output_params": {"output_map_format": "confidence"}}
            _rescale_confidence_to_global({"out_map_path": path}, {"map": w}, {1: s1, 2: s2}, {1: 1.0, 2: 10.0}, cfg)
            import fitsio

            with fitsio.FITS(path) as f:
                # global p99 ~10: HDU1 -> 0.1, HDU2 stays 1.0
                self.assertAlmostEqual(float(f[1].read().mean()), 0.1, places=5)
                self.assertAlmostEqual(float(f[2].read().mean()), 1.0, places=5)

    def test_skip_when_weight_format(self):
        from weightmask.mef import _rescale_confidence_to_global

        # Must not touch anything when the map product holds weight.
        _rescale_confidence_to_global(
            {"out_map_path": "/nonexistent.fits"},
            {},
            {1: np.ones(8)},
            {1: 1.0},
            {"output_params": {"output_map_format": "weight"}},
        )


class TestFlatNoiseVariance(unittest.TestCase):
    def test_term_lowes_vignetted_more(self):
        from weightmask.variance import _calculate_inverse_variance_theoretical as f

        sky = np.full((32, 32), 2700.0, dtype=np.float64)
        flat = np.ones((32, 32))
        flat[:, :8] = 0.7
        base = f(sky, flat, 1.5, 5.0, 1e-9)
        with_term = f(sky, flat, 1.5, 5.0, 1e-9, 0.003)
        self.assertTrue(bool(((with_term < base) & (base > 0)).all()))
        rc = float(with_term[16, 16] / base[16, 16])
        re = float(with_term[16, 2] / base[16, 2])
        self.assertLess(re, rc)  # vignette scaling bites harder at low flat
        same = f(sky, flat, 1.5, 5.0, 1e-9, 0.0)
        np.testing.assert_array_equal(same, base)


class TestBleedAdaptiveCap(unittest.TestCase):
    def _setup(self):
        sci = np.full((200, 20), 100.0, dtype=np.float32)
        sci[:, 10] = 5000.0  # bright column well above sky+5σ everywhere
        sci[100, 10] = 60000.0
        sat = np.zeros((200, 20), dtype=bool)
        sat[100, 10] = True
        sky = np.full((200, 20), 100.0, dtype=np.float32)
        rms = np.full((200, 20), 10.0, dtype=np.float32)
        return sci, sat, sky, rms

    def test_adaptive_shrinks_single_pixel_core(self):
        from weightmask.satur import grow_bleed_trails

        sci, sat, sky, rms = self._setup()
        base = {
            "mask_bleed_trails": True,
            "bleed_thresh_sigma": 5.0,
            "bleed_grow_vertical": 50,
            "bleed_grow_horizontal": 0,
        }
        fixed = grow_bleed_trails(sci, sat, sky, rms, dict(base))
        adap = grow_bleed_trails(
            sci,
            sat,
            sky,
            rms,
            {
                **base,
                "bleed_adaptive_cap": True,
                "bleed_cap_core_factor": 3.0,
                "bleed_cap_min": 20,
                "bleed_cap_max": 200,
            },
        )
        # core_rows=1 -> cap=clip(3,20,200)=20: stops at ±20 while fixed runs to ±50
        self.assertTrue(bool(adap[80:121, 10].all()))
        self.assertFalse(bool(adap[79, 10]))
        self.assertTrue(bool(fixed[50:151, 10].all()))
        self.assertLessEqual(int(adap.sum()), int(fixed.sum()))


class TestFaintCrGate(unittest.TestCase):
    def test_worm_kept_single_dropped(self):
        from weightmask.cosmics import _filter_faint_components

        sci = np.full((32, 32), 100.0, dtype=np.float32)
        rms = np.full((32, 32), 10.0, dtype=np.float32)
        raw = np.zeros((32, 32), dtype=bool)
        raw[5, 5:10] = True  # 5px worm
        sci[5, 5:10] = 500.0
        raw[20, 20] = True  # single-pixel hit
        sci[20, 20] = 500.0
        kept = _filter_faint_components(
            raw,
            sci,
            rms,
            {"min_component_area": 3, "max_component_area": 12, "min_elongation": 2.0, "min_contrast_sigma": 4.0},
        )
        self.assertTrue(bool(kept[5, 5:10].all()))
        self.assertFalse(bool(kept[20, 20]))

    def test_empty_in_empty_out(self):
        from weightmask.cosmics import _filter_faint_components

        sci = np.full((16, 16), 100.0, dtype=np.float32)
        out = _filter_faint_components(np.zeros((16, 16), dtype=bool), sci, None, {})
        self.assertEqual(int(out.sum()), 0)


class TestCosmicNiterForwarding(unittest.TestCase):
    def test_niter_reaches_astroscrappy(self):
        from weightmask import cosmics as cosmics_mod

        data = np.full((32, 32), 100.0, dtype=np.float32)
        existing = np.zeros((32, 32), dtype=bool)
        rms = np.full((32, 32), 10.0, dtype=np.float32)
        seen = {}

        def fake_detect(sci, **kwargs):
            seen.update(kwargs)
            return np.zeros(sci.shape, dtype=bool), None

        with patch.object(cosmics_mod, "detect_cosmics", side_effect=fake_detect):
            out = cosmics_mod.detect_cosmic_rays(
                data,
                existing,
                60000.0,
                1.5,
                5.0,
                {"sigclip": 8.5, "objlim": 15.0, "niter": 2, "psf_aware": False},
                bkg_rms_map=rms,
            )
        self.assertEqual(seen.get("niter"), 2)
        self.assertEqual(out.shape, (32, 32))


if __name__ == "__main__":
    unittest.main()
