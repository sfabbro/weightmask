"""Direct coverage for the per-HDU pipeline helpers in ``weightmask.mef``.

``_merge_dark_mask`` was a closure inside ``_process_all_hdus_to_paths`` and
``_open_hdu_handles`` was only reachable through a full MEF run, so both were
only exercised end-to-end. They carry the fail-open rules that decide whether an
HDU is processed at all: an absent, short or unreadable auxiliary input must
degrade to "no auxiliary mask", while a flat or keep-map MEF shorter than the
science MEF must refuse the HDU rather than silently emit unflat-fielded
weights that look successful.
"""

import tempfile
import unittest
from pathlib import Path

import fitsio
import numpy as np
from numpy.testing import assert_array_equal

from weightmask.mef import _merge_dark_mask, _open_hdu_handles


class _StubHdu:
    """Just enough of a fitsio image HDU for the dark merge."""

    def __init__(self, data, dims=None):
        self._data = np.asarray(data)
        self._dims = tuple(self._data.shape) if dims is None else dims
        self.read_calls = 0

    def read(self):
        self.read_calls += 1
        return self._data

    def get_dims(self):
        return self._dims


class _StubHdul:
    """An indexable, sized container of HDUs."""

    def __init__(self, hdus):
        self._hdus = list(hdus)

    def __len__(self):
        return len(self._hdus)

    def __getitem__(self, index):
        return self._hdus[index]


class _RaisingHdul:
    def __len__(self):
        return 2

    def __getitem__(self, index):
        raise OSError("truncated dark")


def _flat_dark(shape=(8, 8), pedestal=100.0):
    return np.full(shape, pedestal, dtype=np.float32)


def _dark_with_hot_pixel(shape=(8, 8), pedestal=100.0, hot=10000.0, hot_index=(3, 4)):
    dark = np.full(shape, pedestal, dtype=np.float32)
    dark[hot_index] = hot
    return dark


class TestMergeDarkMask(unittest.TestCase):
    def _merge(self, i, sci, pre, dark=None, cfg=None, **kwargs):
        return _merge_dark_mask(
            i,
            sci,
            pre,
            dark_path=kwargs.pop("dark_path", None),
            hdul_dark=dark,
            dark_cfg={} if cfg is None else cfg,
        )

    def test_no_dark_input_at_all_leaves_the_mask_untouched(self):
        pre = np.zeros((8, 8), dtype=bool)
        result = self._merge(0, _StubHdu(np.zeros((8, 8))), pre)
        self.assertIs(result, pre)

    def test_a_hot_pixel_is_merged_into_an_existing_mask(self):
        pre = np.zeros((8, 8), dtype=bool)
        pre[0, 0] = True
        dark = _dark_with_hot_pixel()
        result = self._merge(0, _StubHdu(np.zeros((8, 8))), pre, dark=_StubHdul([_StubHdu(dark)]))
        expected = pre.copy()
        expected[3, 4] = True
        assert_array_equal(result, expected)
        self.assertIsNot(result, pre)

    def test_a_missing_precomputed_mask_starts_from_the_dark_mask(self):
        dark = _dark_with_hot_pixel()
        result = self._merge(0, _StubHdu(np.zeros((8, 8))), None, dark=_StubHdul([_StubHdu(dark)]))
        self.assertIsInstance(result, np.ndarray)
        self.assertTrue(result[3, 4])

    def test_a_clean_dark_adds_nothing(self):
        pre = np.zeros((8, 8), dtype=bool)
        result = self._merge(0, _StubHdu(np.zeros((8, 8))), pre, dark=_StubHdul([_StubHdu(_flat_dark())]))
        assert_array_equal(result, pre)

    def test_a_dark_whose_shape_does_not_match_the_science_hdu_is_skipped(self):
        pre = np.zeros((8, 8), dtype=bool)
        dark = _StubHdul([_StubHdu(_dark_with_hot_pixel(shape=(4, 4), hot_index=(1, 1)))])
        result = self._merge(0, _StubHdu(np.zeros((8, 8))), pre, dark=dark)
        self.assertIs(result, pre)

    def test_a_precomputed_mask_of_the_wrong_shape_is_replaced_not_kept(self):
        # The dark HDU decides the shape; a stale mask of another shape would
        # otherwise be OR-ed with it and fail the shape contract downstream.
        pre = np.zeros((5, 5), dtype=bool)
        dark = _StubHdul([_StubHdu(_dark_with_hot_pixel(shape=(8, 8)))])
        result = self._merge(0, _StubHdu(np.zeros((8, 8))), pre, dark=dark)
        self.assertEqual(result.shape, (8, 8))
        self.assertTrue(result[3, 4])

    def test_a_dark_mef_shorter_than_the_index_is_skipped(self):
        pre = np.zeros((8, 8), dtype=bool)
        dark = _StubHdul([_StubHdu(_dark_with_hot_pixel())])
        result = self._merge(3, _StubHdu(np.zeros((8, 8))), pre, dark=dark)
        self.assertIs(result, pre)

    def test_an_unreadable_dark_container_is_skipped(self):
        pre = np.zeros((8, 8), dtype=bool)
        result = self._merge(0, _StubHdu(np.zeros((8, 8))), pre, dark=_RaisingHdul())
        self.assertIs(result, pre)

    def test_an_unreadable_dark_hdu_is_skipped_rather_than_raising(self):
        class Boom(_StubHdu):
            def read(self):
                raise OSError("bad dark data")

        pre = np.zeros((8, 8), dtype=bool)
        result = self._merge(0, _StubHdu(np.zeros((8, 8))), pre, dark=_StubHdul([Boom(_flat_dark())]))
        self.assertIs(result, pre)

    def test_an_invalid_dark_configuration_does_not_fail_the_hdu(self):
        pre = np.zeros((8, 8), dtype=bool)
        dark = _StubHdul([_StubHdu(_dark_with_hot_pixel())])
        # detect_dark_hot_pixels refuses an unknown key; the merge must absorb it.
        result = self._merge(0, _StubHdu(np.zeros((8, 8))), pre, dark=dark, cfg={"bogus": 1})
        self.assertIs(result, pre)

    def test_a_reopened_dark_path_takes_precedence_over_the_open_container(self):
        pre = np.zeros((8, 8), dtype=bool)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "dark.fits")
            fitsio.write(path, _dark_with_hot_pixel(), clobber=True)
            # A container that would add nothing: prove the path was used.
            decoy = _StubHdul([_StubHdu(_flat_dark())])
            result = self._merge(0, _StubHdu(np.zeros((8, 8))), pre, dark=decoy, dark_path=path)
        self.assertTrue(result[3, 4])

    def test_an_unopenable_dark_path_is_skipped(self):
        pre = np.zeros((8, 8), dtype=bool)
        result = self._merge(0, _StubHdu(np.zeros((8, 8))), pre, dark_path="/nonexistent/dark.fits")
        self.assertIs(result, pre)

    def test_a_dark_path_without_the_index_is_skipped(self):
        pre = np.zeros((8, 8), dtype=bool)
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "dark.fits")
            fitsio.write(path, _dark_with_hot_pixel(), clobber=True)
            result = self._merge(4, _StubHdu(np.zeros((8, 8))), pre, dark_path=path)
        self.assertIs(result, pre)


class TestOpenHduHandles(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.science = str(self.tmp / "sci.fits")
        self.flat = str(self.tmp / "flat.fits")
        self.badpix = str(self.tmp / "badpix.fits")
        self.short = str(self.tmp / "short.fits")

    def tearDown(self):
        self._tmp.cleanup()

    def _write(self, path, n_hdus, name="8341-7-5"):
        # fitsio.FITS.write rather than fitsio.write: the latter ignores
        # ``append`` (deprecated, and a future error), which silently produced
        # a one-HDU file and made every i > 0 case look like a short MEF.
        with fitsio.FITS(path, "rw", clobber=True) as handle:
            for _ in range(n_hdus):
                handle.write(np.ones((4, 4), dtype=np.float32), header={"CCDNAME": name})

    def test_a_path_reopen_yields_every_handle_and_a_working_close(self):
        self._write(self.science, 3)
        self._write(self.flat, 3)
        self._write(self.badpix, 3)
        # Production always has the open handles as well as the paths; the
        # reopen branch is gated on both being present.
        with fitsio.FITS(self.flat) as hdul_flat, fitsio.FITS(self.badpix) as hdul_badpix:
            sci, flat, badpix, header, name, close, err = _open_hdu_handles(
                1,
                in_path=self.science,
                fl_path=self.flat,
                bp_path=self.badpix,
                hdul_input=None,
                hdul_flat=hdul_flat,
                hdul_badpix=hdul_badpix,
            )
            try:
                self.assertIsNone(err)
                self.assertIsNotNone(sci)
                self.assertIsNotNone(flat)
                self.assertIsNotNone(badpix)
                self.assertEqual(name, "8341-7-5")
                self.assertEqual(tuple(sci.get_dims()), (4, 4))
                self.assertIsNotNone(header)
                self.assertEqual(header.get("CCDNAME"), "8341-7-5")
            finally:
                close()
            # close() is idempotent and must not raise on a second call.
            close()

    def test_the_index_identifies_which_ccd_was_opened(self):
        with fitsio.FITS(self.science, "rw", clobber=True) as handle:
            handle.write(np.ones((4, 4), dtype=np.float32), header={"CCDNAME": "first"})
            handle.write(np.ones((4, 4), dtype=np.float32), header={"CCDNAME": "mid"})
            handle.write(np.full((4, 4), 2.0, dtype=np.float32), header={"CCDNAME": "second"})
        _sci, _flat, _bad, _header, name, close, err = _open_hdu_handles(
            2, in_path=self.science, fl_path=None, bp_path=None, hdul_input=None, hdul_flat=None, hdul_badpix=None
        )
        close()
        self.assertIsNone(err)
        self.assertEqual(name, "second")

    def test_an_index_past_the_science_mef_is_refused(self):
        self._write(self.science, 2)
        sci, flat, badpix, header, name, close, err = _open_hdu_handles(
            5, in_path=self.science, fl_path=None, bp_path=None, hdul_input=None, hdul_flat=None, hdul_badpix=None
        )
        close()
        self.assertIsNone(sci)
        self.assertIsNone(header)
        self.assertEqual(name, "HDU5")
        self.assertIn("range", err)

    def test_a_flat_mef_shorter_than_the_science_mef_refuses_the_hdu(self):
        # Substituting a unit flat here would emit unflat-fielded weights that
        # look like a successful run, so this must be an error, not a default.
        self._write(self.science, 3)
        self._write(self.short, 1)
        with fitsio.FITS(self.short) as hdul_flat:
            sci, _flat, _bad, _header, name, close, err = _open_hdu_handles(
                2,
                in_path=self.science,
                fl_path=self.short,
                bp_path=None,
                hdul_input=None,
                hdul_flat=hdul_flat,
                hdul_badpix=None,
            )
        close()
        self.assertIsNone(sci)
        self.assertEqual(name, "HDU2")
        self.assertIn("flat", err)
        self.assertIn("need index 2", err)

    def test_a_keep_map_mef_shorter_than_the_science_mef_refuses_the_hdu(self):
        self._write(self.science, 3)
        self._write(self.short, 1)
        with fitsio.FITS(self.short) as hdul_badpix:
            sci, _flat, _bad, _header, _name, close, err = _open_hdu_handles(
                2,
                in_path=self.science,
                fl_path=None,
                bp_path=self.short,
                hdul_input=None,
                hdul_flat=None,
                hdul_badpix=hdul_badpix,
            )
        close()
        self.assertIsNone(sci)
        self.assertIn("keep-map", err)
        self.assertIn("need index 2", err)

    def test_absent_auxiliary_paths_yield_no_auxiliary_handles(self):
        self._write(self.science, 1)
        sci, flat, badpix, _header, _name, close, err = _open_hdu_handles(
            0, in_path=self.science, fl_path=None, bp_path=None, hdul_input=None, hdul_flat=None, hdul_badpix=None
        )
        close()
        self.assertIsNone(err)
        self.assertIsNotNone(sci)
        self.assertIsNone(flat)
        self.assertIsNone(badpix)

    def test_open_handles_are_used_when_no_path_is_given(self):
        self._write(self.science, 2)
        self._write(self.flat, 2)
        with fitsio.FITS(self.science) as hdul_input, fitsio.FITS(self.flat) as hdul_flat:
            sci, flat, badpix, header, name, close, err = _open_hdu_handles(
                1,
                in_path=None,
                fl_path=None,
                bp_path=None,
                hdul_input=hdul_input,
                hdul_flat=hdul_flat,
                hdul_badpix=None,
            )
            close()
            self.assertIsNone(err)
            self.assertIsNotNone(sci)
            self.assertIsNotNone(flat)
            self.assertIsNone(badpix)
            self.assertEqual(name, "8341-7-5")
            self.assertEqual(header.get("CCDNAME"), "8341-7-5")

    def test_an_auxiliary_mef_shorter_than_the_index_yields_no_auxiliary_handle(self):
        # Without a path to reopen there is no fail-closed check; the aux input
        # is simply absent for this HDU, which process_image must then decide on.
        self._write(self.science, 3)
        self._write(self.short, 1)
        with fitsio.FITS(self.science) as hdul_input, fitsio.FITS(self.short) as hdul_flat:
            sci, flat, _bad, _header, _name, close, err = _open_hdu_handles(
                2,
                in_path=None,
                fl_path=None,
                bp_path=None,
                hdul_input=hdul_input,
                hdul_flat=hdul_flat,
                hdul_badpix=None,
            )
            close()
            self.assertIsNone(err)
            self.assertIsNotNone(sci)
            self.assertIsNone(flat)

    def test_an_index_past_the_open_handles_is_reported_rather_than_raised(self):
        self._write(self.science, 2)
        with fitsio.FITS(self.science) as hdul_input:
            sci, _flat, _bad, header, name, close, err = _open_hdu_handles(
                9,
                in_path=None,
                fl_path=None,
                bp_path=None,
                hdul_input=hdul_input,
                hdul_flat=None,
                hdul_badpix=None,
            )
            close()
        self.assertIsNone(sci)
        self.assertIsNotNone(err)
        self.assertEqual(name, "HDU9")
        self.assertIsNone(header)


if __name__ == "__main__":
    unittest.main()
