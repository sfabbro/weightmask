import unittest

import numpy as np

from weightmask.background import estimate_background


class TestBackground(unittest.TestCase):
    def test_estimate_background(self):
        """Test background estimation."""
        # Create test science data
        sci_data = np.random.poisson(100, (100, 100)).astype(np.float32)

        # Add some sources
        sci_data[30:40, 30:40] = 1000.0

        # Create a mask for the sources
        mask = np.zeros((100, 100), dtype=bool)
        mask[30:40, 30:40] = True

        config = {"box_size": 32, "filter_size": 3}

        bkg_map, bkg_rms_map = estimate_background(sci_data, mask, config)

        # Check that we got results
        self.assertIsNotNone(bkg_map)
        self.assertIsNotNone(bkg_rms_map)

        # Check that results have correct shape
        self.assertEqual(bkg_map.shape, sci_data.shape)
        self.assertEqual(bkg_rms_map.shape, sci_data.shape)

        # Check that results are finite
        self.assertTrue(np.isfinite(bkg_map).all())
        self.assertTrue(np.isfinite(bkg_rms_map).all())

        # Check that RMS values are positive
        self.assertTrue(np.all(bkg_rms_map > 0))

    def test_mask_threshold_is_crowding_fraction_not_object_cut(self):
        sci_data = np.full((64, 64), 100.0, dtype=np.float32)
        mask = np.ones((64, 64), dtype=bool)
        mask[:4] = False
        diagnostics = {}
        estimate_background(
            sci_data,
            mask,
            {"method": "sep", "mask_threshold": 0.8, "_diagnostics": diagnostics},
        )
        self.assertEqual(diagnostics.get("fallback"), "global_sep")

    def test_estimate_background_no_mask(self):
        """Test background estimation with no masked pixels."""
        # Create test science data
        sci_data = np.random.poisson(100, (100, 100)).astype(np.float32)

        # Create an empty mask
        mask = np.zeros((100, 100), dtype=bool)

        config = {"box_size": 32, "filter_size": 3}

        bkg_map, bkg_rms_map = estimate_background(sci_data, mask, config)

        # Check that we got results
        self.assertIsNotNone(bkg_map)
        self.assertIsNotNone(bkg_rms_map)

    def test_median_filter_ignores_masked_source_values(self):
        data = np.full((40, 40), 100.0, dtype=np.float32)
        mask = np.zeros(data.shape, dtype=bool)
        mask[12:28, 12:28] = True
        config = {"method": "median_filter", "median_kernel_size": 9}
        expected, _ = estimate_background(data, mask, config)
        data[mask] = 10000.0
        actual, _ = estimate_background(data, mask, config)
        np.testing.assert_array_equal(actual, expected)

    def test_estimate_background_all_masked(self):
        """Test background estimation with all pixels masked."""
        # Create test science data
        sci_data = np.random.poisson(100, (100, 100)).astype(np.float32)

        # Create a mask with all pixels masked
        mask = np.ones((100, 100), dtype=bool)

        config = {"box_size": 32, "filter_size": 3}

        bkg_map, bkg_rms_map = estimate_background(sci_data, mask, config)

        # Check that we got results (fallback to global)
        self.assertIsNotNone(bkg_map)
        self.assertIsNotNone(bkg_rms_map)

    def test_big_endian_fits_array_uses_sep_without_fallback(self):
        """astropy.io.fits returns big-endian (>f4) arrays; SEP requires native byte order."""
        from astropy.io import fits

        from weightmask.process import process_image

        rng = np.random.default_rng(7)
        native = (1000.0 + rng.normal(0.0, 5.0, (128, 128))).astype(np.float32)
        big_endian = native.astype(">f4")
        mask = np.zeros((128, 128), dtype=bool)
        diag = {}
        bkg_be, rms_be = estimate_background(
            big_endian,
            mask,
            {"method": "sep", "box_size": 32, "filter_size": 3, "_diagnostics": diag},
        )
        bkg_ne, rms_ne = estimate_background(
            native,
            mask,
            {"method": "sep", "box_size": 32, "filter_size": 3},
        )
        self.assertEqual(diag.get("effective_method"), "sep")
        self.assertIsNone(diag.get("fallback"))
        np.testing.assert_allclose(bkg_be, bkg_ne, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(rms_be, rms_ne, rtol=1e-6, atol=1e-6)

        hdr = fits.Header({"GAIN": 1.5, "RDNOISE": 5.0, "SATURATE": 60000.0})
        cfg = {"streak_masking": {"enable": False}}
        mask_be, ivar_be, w_be, _, sky_be, _ = process_image(big_endian, hdr, big_endian / 1000.0, dict(cfg))
        mask_ne, ivar_ne, w_ne, _, sky_ne, _ = process_image(native, hdr, native / 1000.0, dict(cfg))
        np.testing.assert_array_equal(mask_be, mask_ne)
        np.testing.assert_allclose(ivar_be, ivar_ne, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(w_be, w_ne, rtol=1e-6, atol=1e-6)
        np.testing.assert_allclose(sky_be, sky_ne, rtol=1e-6, atol=1e-6)


class TestSkyMeshRoundtrip(unittest.TestCase):
    def test_sep_back_roundtrip_sub_adu(self):
        """Codec recovers SEP back() to sub-ADU (wrong phase / Keys cubic fails this)."""
        import sep

        from weightmask.background import (
            parse_sky_mesh_header,
            reconstruct_sky_from_header,
            reconstruct_sky_mesh,
            sky_to_mesh,
        )

        rng = np.random.default_rng(0)
        data = (1200.0 + rng.normal(0.0, 5.0, (200, 150))).astype(np.float64)
        sky = sep.Background(data, bw=40, bh=40, fw=3, fh=3).back().astype(np.float32)
        mesh, cards = sky_to_mesh(sky, 40)
        self.assertEqual(mesh.shape, ((200 - 1) // 40 + 1, (150 - 1) // 40 + 1))
        shape, box = parse_sky_mesh_header(cards)
        self.assertEqual((shape, box), (sky.shape, 40))
        rec = reconstruct_sky_from_header(mesh, cards)
        diff = np.abs(rec.astype(np.float64) - sky.astype(np.float64))
        self.assertLess(float(np.max(diff)), 0.05)
        self.assertTrue(bool(np.allclose(reconstruct_sky_mesh(np.full((2, 2), 7.0), (64, 64), 32), 7.0)))
        with self.assertRaises(ValueError):
            parse_sky_mesh_header({"SKYH": 10, "SKYW": 10})

    def test_mesh_header_requires_complete_consistent_positive_dimensions(self):
        from weightmask.background import parse_sky_mesh_header

        valid = {"SKYMESH": True, "MESHBW": 8, "MESHBH": 8, "SKYH": 32, "SKYW": 24}
        invalid = (
            {key: value for key, value in valid.items() if key != "MESHBH"},
            {**valid, "MESHBW": 4},
            {**valid, "MESHBH": 0},
            {**valid, "SKYH": -1},
            {**valid, "SKYW": 0},
        )
        for header in invalid:
            with self.subTest(header=header), self.assertRaises(ValueError):
                parse_sky_mesh_header(header)

    def test_mesh_data_requires_rank_finiteness_and_implied_shape(self):
        from weightmask.background import reconstruct_sky_mesh

        invalid = (
            np.ones(16, dtype=np.float32),
            np.ones((4, 4, 1), dtype=np.float32),
            np.ones((3, 4), dtype=np.float32),
            np.ones((5, 4), dtype=np.float32),
            np.full((4, 4), np.nan, dtype=np.float32),
            np.full((4, 4), np.inf, dtype=np.float32),
        )
        for mesh in invalid:
            with self.subTest(shape=mesh.shape), self.assertRaises(ValueError):
                reconstruct_sky_mesh(mesh, (32, 32), 8)

    def test_one_node_mesh_dimensions_roundtrip(self):
        from weightmask.background import reconstruct_sky_from_header, sky_to_mesh

        for shape in ((1, 1), (1, 7), (7, 1)):
            with self.subTest(shape=shape):
                sky = np.full(shape, 1234.0, dtype=np.float32)
                mesh, cards = sky_to_mesh(sky, 8)
                rebuilt = reconstruct_sky_from_header(mesh, cards)
                self.assertEqual(mesh.shape, (1, 1))
                self.assertEqual(rebuilt.shape, shape)
                np.testing.assert_array_equal(rebuilt, sky)

    def test_single_node_column_mesh_reconstructs_at_full_width(self):
        """A mesh narrower than one box has one node column, not one output column.

        Reachable whenever the HDU is narrower than the mesh box (a 30 px wide
        HDU with box 32 gives a (125, 1) mesh). The reconstruction used to
        return (h, 1), silently writing a wrong-shaped sky FITS product.
        """
        from weightmask.background import reconstruct_sky_mesh, sky_to_mesh

        for shape, box in (((4000, 30), 32), ((4000, 200), 128), ((300, 200), 128)):
            sky = np.full(shape, 1234.0, dtype=np.float32)
            mesh, cards = sky_to_mesh(sky, box)
            rec = reconstruct_sky_mesh(mesh, shape, box)
            self.assertEqual(rec.shape, shape, f"shape={shape} box={box} mesh={mesh.shape}")
            np.testing.assert_allclose(rec, 1234.0, atol=1e-3)

    def test_megaprime_scale_roundtrip_sub_adu(self):
        """Pin MegaPrime CCD geometry in CI (SEP back -> mesh -> reconstruct)."""
        import sep

        from weightmask.background import reconstruct_sky_mesh, sky_to_mesh

        rng = np.random.default_rng(0)
        h, w, box = 4612, 2048, 128
        data = (
            1200.0
            + 0.02 * np.arange(w, dtype=np.float64)
            + 0.01 * np.arange(h, dtype=np.float64)[:, None]
            + rng.normal(0.0, 5.0, (h, w))
        )
        sky = sep.Background(data, bw=box, bh=box, fw=3, fh=3).back().astype(np.float32)
        mesh, cards = sky_to_mesh(sky, box)
        self.assertEqual(mesh.shape, (37, 16))
        self.assertEqual((cards["SKYH"], cards["SKYW"]), (h, w))
        rec = reconstruct_sky_mesh(mesh, sky.shape, box)
        diff = np.abs(rec.astype(np.float64) - sky.astype(np.float64))
        self.assertLess(float(np.max(diff)), 0.5)
        self.assertLess(float(np.sqrt(np.mean(diff**2))), 0.02)
        self.assertLess(float(np.percentile(diff, 99)), 0.1)

    def test_gradient_roundtrip_uses_clipped_node_abscissae(self):
        """Encode and decode must share clipped (k+0.5)*box nodes."""
        from weightmask.background import reconstruct_sky_mesh, sky_to_mesh

        h, w, box = 300, 200, 128
        sky = (1200.0 + 0.5 * np.arange(w) + 0.3 * np.arange(h)[:, None]).astype(np.float32)
        mesh, _cards = sky_to_mesh(sky, box)
        rec = reconstruct_sky_mesh(mesh, sky.shape, box)
        diff = np.abs(rec.astype(np.float64) - sky.astype(np.float64))
        self.assertLess(float(diff[-16:].max()), 0.5)
        self.assertLess(float(diff.max()), 0.5)

    def test_diagnostics_records_actual_sep_box(self):
        from weightmask.background import estimate_background

        rng = np.random.default_rng(0)
        data = (100.0 + rng.normal(0.0, 1.0, (128, 128))).astype(np.float32)
        mask = np.zeros((128, 128), dtype=bool)
        diag = {}
        estimate_background(
            data,
            mask,
            {"box_size": 32, "auto_box_scaling": False, "filter_size": 3, "_diagnostics": diag},
        )
        self.assertEqual(diag.get("box_size"), 32)
        diag2 = {}
        estimate_background(
            data,
            np.ones((128, 128), dtype=bool),
            {"mask_threshold": 0.5, "_diagnostics": diag2},
        )
        self.assertNotIn("box_size", diag2)


class TestDipRepair(unittest.TestCase):
    def test_unmeasured_rms_dip_is_not_repaired(self):
        from weightmask.background import _repair_negative_dips

        data = np.full((20, 20), 1000.0, dtype=np.float32)
        sky = data.copy()
        sky[10, 10] = -100.0
        rms = np.full_like(sky, 10.0)
        rms[10, 10] = np.inf
        out = _repair_negative_dips(sky, data, rms, np.zeros_like(sky, dtype=bool), {})
        self.assertEqual(float(out[10, 10]), -100.0)

    def test_overshoot_filled_blanks_untouched(self):
        from weightmask.background import _repair_negative_dips

        rng = np.random.default_rng(0)
        sky = np.full((200, 200), 1000.0)
        data = (1000 + 10 * rng.standard_normal((200, 200))).astype(np.float32)
        yy, xx = np.mgrid[0:200, 0:200]
        dip = (xx - 150) ** 2 + (yy - 150) ** 2 < 15**2
        sky[dip] = -200.0  # certain overshoot: data ~1000, map deeply negative
        blank = (xx - 40) ** 2 + (yy - 40) ** 2 < 10**2
        data[blank] = 0.0
        sky[blank] = -5.0  # dark blank: filling with 1000 would be catastrophic
        rms = np.full((200, 200), 10.0, dtype=np.float32)
        out = _repair_negative_dips(sky, data, rms, np.zeros((200, 200), bool), {})
        self.assertTrue(bool((out[dip] > 900).all()))  # overshoot filled from neighbor sky
        self.assertTrue(bool((out[blank] == -5.0).all()))  # blanks never touched

    def test_cap_and_disable(self):
        from weightmask.background import _repair_negative_dips

        cap_sky = np.full((50, 50), -10.0)
        cap_data = np.full((50, 50), 1000.0, dtype=np.float32)
        cap_rms = np.full((50, 50), 10.0, dtype=np.float32)
        out = _repair_negative_dips(cap_sky, cap_data, cap_rms, np.zeros((50, 50), bool), {})
        self.assertTrue(bool((out == -10.0).all()))  # over cap: untouched
        tiny_sky = np.full((10, 10), -10.0)
        tiny_data = np.full((10, 10), 1000.0, dtype=np.float32)
        tiny_rms = np.full((10, 10), 10.0, dtype=np.float32)
        out2 = _repair_negative_dips(
            tiny_sky,
            tiny_data,
            tiny_rms,
            np.zeros((10, 10), bool),
            {"dip_repair_enable": False},
        )
        self.assertTrue(bool((out2 == -10.0).all()))  # disabled: untouched


if __name__ == "__main__":
    unittest.main()
