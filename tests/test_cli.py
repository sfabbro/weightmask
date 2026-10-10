import contextlib
import copy
import io
import os
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import fitsio
import numpy as np
import yaml

from weightmask.cli import run_pipeline, validate_config, validate_fits_file, validate_input_files


class TestValidateFitsFile(unittest.TestCase):
    @patch("weightmask.cli.fitsio.FITS")
    def test_validate_fits_file_valid(self, mock_fits):
        mock_fits_instance = MagicMock()
        mock_fits_instance.__len__.return_value = 2
        mock_fits.return_value.__enter__.return_value = mock_fits_instance

        result = validate_fits_file("dummy.fits")
        self.assertTrue(result)
        mock_fits.assert_called_once_with("dummy.fits", "r")

    @patch("weightmask.cli.fitsio.FITS")
    def test_validate_fits_file_empty(self, mock_fits):
        mock_fits_instance = MagicMock()
        mock_fits_instance.__len__.return_value = 0
        mock_fits.return_value.__enter__.return_value = mock_fits_instance

        result = validate_fits_file("empty.fits")
        self.assertFalse(result)
        mock_fits.assert_called_once_with("empty.fits", "r")

    @patch("weightmask.cli.fitsio.FITS")
    def test_validate_fits_file_oserror(self, mock_fits):
        mock_fits.side_effect = OSError("Invalid file format")

        result = validate_fits_file("invalid.fits")
        self.assertFalse(result)
        mock_fits.assert_called_once_with("invalid.fits", "r")


class TestValidateConfig(unittest.TestCase):
    @staticmethod
    def _canonical_config():
        path = Path(__file__).resolve().parents[1] / "weightmask.yml"
        return yaml.safe_load(path.read_text())

    @staticmethod
    def _replace(config, path, value):
        changed = copy.deepcopy(config)
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        return changed

    @staticmethod
    def _leaves(config, prefix=()):
        for key, value in config.items():
            path = (*prefix, key)
            if isinstance(value, dict):
                yield from TestValidateConfig._leaves(value, path)
            else:
                yield path, value

    def test_validate_config_valid(self):
        """Test with a fully valid configuration."""
        valid_config = {
            "flat_masking": {},
            "saturation": {},
            "sep_background": {},
            "cosmic_ray": {},
            "sep_objects": {},
            "streak_masking": {},
            "variance": {"method": "theoretical"},
            "confidence_params": {},
            "output_params": {},
        }
        self.assertTrue(validate_config(valid_config))

    def test_validate_config_missing_sections(self):
        """Test with missing sections. It should print warnings but return True."""
        empty_config = {}
        self.assertTrue(validate_config(empty_config))

    def test_validate_config_invalid_variance_not_dict(self):
        """Test with an invalid variance section (not a dictionary)."""
        invalid_config = {"variance": "not_a_dict"}
        self.assertFalse(validate_config(invalid_config))

    def test_validate_config_invalid_variance_method(self):
        """Test with an invalid variance method."""
        invalid_config = {"variance": {"method": "invalid_method"}}
        self.assertFalse(validate_config(invalid_config))

    def test_validate_config_valid_variance_methods(self):
        """Test with all valid variance methods."""
        for method in ["theoretical", "rms_map", "empirical_fit"]:
            valid_config = {"variance": {"method": method}}
            self.assertTrue(validate_config(valid_config))

    def test_validate_config_invalid_streak_method(self):
        invalid_config = {"streak_masking": {"method": "hough"}}
        self.assertFalse(validate_config(invalid_config))

    def test_validate_config_rejects_stale_top_level_background(self):
        invalid_config = {"background": {}, "variance": {"method": "theoretical"}}
        self.assertFalse(validate_config(invalid_config))

    def test_validate_config_rejects_misplaced_flat_keys(self):
        invalid_config = {
            "variance": {
                "method": "theoretical",
                "local_filter_size": 15,
            }
        }
        self.assertFalse(validate_config(invalid_config))

    def test_every_canonical_leaf_has_declared_type_validation(self):
        config = self._canonical_config()
        for path, value in self._leaves(config):
            if isinstance(value, bool):
                malformed = 1
            elif isinstance(value, (int, float)):
                malformed = True
            elif isinstance(value, str):
                malformed = 1
            elif isinstance(value, list):
                malformed = [1]
            elif value is None:
                malformed = {}
            else:
                self.fail(f"No malformed-value fixture for {'.'.join(path)} ({type(value).__name__})")
            with self.subTest(path=".".join(path)):
                self.assertFalse(validate_config(self._replace(config, path, malformed)))

    def test_numeric_values_must_be_finite(self):
        config = self._canonical_config()
        for path, value in self._leaves(config):
            if isinstance(value, float):
                for malformed in (float("nan"), float("inf")):
                    with self.subTest(path=".".join(path), malformed=malformed):
                        self.assertFalse(validate_config(self._replace(config, path, malformed)))

    def test_every_canonical_leaf_is_consumed_by_its_runtime_section(self):
        package = Path(__file__).resolve().parents[1] / "weightmask"
        consumers = {
            "flat_masking": ("bad.py", "mef.py"),
            "dark_masking": ("bad.py", "mef.py"),
            "saturation": ("satur.py", "process.py"),
            "sep_background": ("background.py", "process.py"),
            "cosmic_ray": ("cosmics.py", "process.py"),
            "sep_objects": ("objects.py", "process.py"),
            "streak_masking": ("streaks.py", "process.py"),
            "variance": ("variance.py", "process.py"),
            "confidence_params": ("weight.py", "mef.py"),
            "output_params": ("weight.py", "mef.py", "process.py", "cli.py"),
        }
        config = self._canonical_config()
        for section, filenames in consumers.items():
            source = "\n".join((package / filename).read_text() for filename in filenames)
            for path, _value in self._leaves({section: config[section]}):
                key = path[-1]
                consumed = f'.get("{key}"' in source or f'["{key}"]' in source
                with self.subTest(path=".".join(path)):
                    self.assertTrue(consumed, f"Canonical key {'.'.join(path)} is not consumed")

    def test_rejects_unknown_keys_at_every_mapping_level(self):
        paths = [
            ("flat_masking",),
            ("dark_masking",),
            ("saturation",),
            ("saturation", "histogram_params"),
            ("sep_background",),
            ("cosmic_ray",),
            ("cosmic_ray", "faint_cr"),
            ("cosmic_ray", "faint_cr", "residual"),
            ("sep_objects",),
            ("streak_masking",),
            ("streak_masking", "houghpeak_params"),
            ("streak_masking", "contour_params"),
            ("streak_masking", "mask_params"),
            ("streak_masking", "sparse_ransac_params"),
            ("variance",),
            ("confidence_params",),
            ("output_params",),
        ]
        config = self._canonical_config()
        for path in paths:
            changed = copy.deepcopy(config)
            target = changed
            for key in path:
                target = target[key]
            target["misspelled_setting"] = 1
            with self.subTest(path=".".join(path)):
                self.assertFalse(validate_config(changed))

    def test_rejects_invalid_ranges_and_cross_field_combinations(self):
        cases = [
            (("flat_masking", "local_filter_size"), 0),
            (("flat_masking", "dead_ccd_badpix_fraction"), 1.1),
            (("saturation", "bleed_grow_vertical"), -1),
            (("sep_background", "iterations"), -1),
            (("sep_background", "mask_threshold"), 1.1),
            (("cosmic_ray", "niter"), -1),
            (("cosmic_ray", "max_component_area"), 0),
            (("cosmic_ray", "faint_cr", "min_component_area"), -1),
            (("sep_objects", "min_area"), 0),
            (("sep_objects", "deblend_cont"), 1.1),
            (("streak_masking", "houghpeak_params", "min_votes"), -1),
            (("streak_masking", "contour_params", "area_cut"), -1.0),
            (("streak_masking", "mask_params", "rotation_interpolation_order"), 6),
            (("streak_masking", "mask_params", "min_mask_pixels"), -1),
            (("streak_masking", "sparse_ransac_params", "max_trials"), -1),
            (("variance", "epsilon"), 0.0),
            (("variance", "flat_rel_noise"), -1.0),
            (("confidence_params", "normalize_percentile"), 0.0),
        ]
        config = self._canonical_config()
        for path, value in cases:
            with self.subTest(path=".".join(path), value=value):
                self.assertFalse(validate_config(self._replace(config, path, value)))

        cross_field_cases = [
            (("flat_masking", "local_low_thresh"), 2.0, ("flat_masking", "local_high_thresh"), 1.0),
            (
                ("saturation", "histogram_params", "hist_min_adu"),
                50000.0,
                ("saturation", "histogram_params", "hist_max_adu"),
                40000.0,
            ),
            (("saturation", "bleed_cap_min"), 201, ("saturation", "bleed_cap_max"), 200),
            (("sep_background", "box_size"), 2048, ("sep_background", "max_box_size"), 1024),
            (
                ("cosmic_ray", "faint_cr", "min_component_area"),
                13,
                ("cosmic_ray", "faint_cr", "max_component_area"),
                12,
            ),
            (
                ("cosmic_ray", "faint_cr", "residual", "min_component_area"),
                13,
                ("cosmic_ray", "faint_cr", "residual", "max_component_area"),
                12,
            ),
            (
                ("streak_masking", "mask_params", "min_row_hits"),
                257,
                ("streak_masking", "mask_params", "strip_length"),
                256,
            ),
            (
                ("streak_masking", "mask_params", "max_support_width"),
                81,
                ("streak_masking", "mask_params", "strip_width"),
                80,
            ),
        ]
        for first_path, first_value, second_path, second_value in cross_field_cases:
            changed = self._replace(config, first_path, first_value)
            changed = self._replace(changed, second_path, second_value)
            with self.subTest(first=".".join(first_path), second=".".join(second_path)):
                self.assertFalse(validate_config(changed))

    def test_background_max_box_default_tracks_explicit_box_size(self):
        self.assertTrue(validate_config({"sep_background": {"box_size": 2048}}))

    def test_iteration_counts_follow_runtime_zero_semantics(self):
        zero_disables = [
            {"sep_background": {"iterations": 0}},
            {"cosmic_ray": {"niter": 0}},
            {"cosmic_ray": {"faint_cr": {"niter": 0}}},
        ]
        for config in zero_disables:
            with self.subTest(config=config):
                self.assertTrue(validate_config(config))

        zero_is_invalid = [
            {"streak_masking": {"sparse_ransac_params": {"max_trials": 0}}},
            {"streak_masking": {"sparse_ransac_params": {"max_trails": 0}}},
        ]
        for config in zero_is_invalid:
            with self.subTest(config=config):
                self.assertFalse(validate_config(config))


class TestRunPipeline(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.workspace = self.test_dir.name

        self.input_file = os.path.join(self.workspace, "test_input.fits")
        rng = np.random.default_rng(0)
        data = rng.normal(100, 10, (100, 100)).astype(np.float32)
        fitsio.write(self.input_file, data, clobber=True)

        self.config_file = os.path.join(self.workspace, "test_config.yml")
        with open(self.config_file, "w") as f:
            f.write(
                """
flat_masking: {}
saturation: {}
sep_background: {}
cosmic_ray: {}
sep_objects: {}
streak_masking: {}
variance:
  method: theoretical
confidence_params: {}
output_params:
  output_map_format: weight
"""
            )

    def tearDown(self):
        self.test_dir.cleanup()

    @patch("sys.argv", ["weightmask"])
    def test_run_pipeline_missing_args(self):
        with self.assertRaises(SystemExit) as cm:
            run_pipeline()
        self.assertEqual(cm.exception.code, 2)

    def test_run_pipeline_missing_input_file(self):
        with patch(
            "sys.argv",
            ["weightmask", "nonexistent.fits", "--config", self.config_file],
        ):
            result = run_pipeline()
            self.assertEqual(result, 1)

    def test_run_pipeline_success(self):
        output_file = os.path.join(self.workspace, "output.weight.fits")
        with patch(
            "sys.argv",
            [
                "weightmask",
                self.input_file,
                "--config",
                self.config_file,
                "-o",
                output_file,
            ],
        ):
            result = run_pipeline()
            self.assertEqual(result, 0)
            self.assertTrue(os.path.exists(output_file))
            with fitsio.FITS(output_file) as f:
                self.assertEqual(len(f), 1)
                data = f[0].read()
                self.assertIsNotNone(data)
                self.assertEqual(data.shape, (100, 100))

    def test_run_pipeline_individual_masks(self):
        output_file = os.path.join(self.workspace, "output2.weight.fits")
        with patch(
            "sys.argv",
            [
                "weightmask",
                self.input_file,
                "--config",
                self.config_file,
                "-o",
                output_file,
                "--individual_masks",
            ],
        ):
            result = run_pipeline()
            self.assertEqual(result, 0)
            self.assertTrue(os.path.exists(output_file))
            self.assertTrue(os.path.exists(os.path.join(self.workspace, "output2.weight.bad.fits")))
            self.assertTrue(os.path.exists(os.path.join(self.workspace, "output2.weight.sat.fits")))

    def test_run_pipeline_missing_config(self):
        with patch(
            "sys.argv",
            ["weightmask", self.input_file, "--config", "nonexistent_config.yml"],
        ):
            result = run_pipeline()
            self.assertEqual(result, 1)

    def test_run_pipeline_bad_config(self):
        bad_config_file = os.path.join(self.workspace, "bad_config.yml")
        with open(bad_config_file, "w") as f:
            f.write("this is not a valid yaml file: [")
        with patch(
            "sys.argv",
            ["weightmask", self.input_file, "--config", bad_config_file],
        ):
            result = run_pipeline()
            self.assertEqual(result, 1)

    def test_run_pipeline_missing_flat(self):
        with patch(
            "sys.argv",
            [
                "weightmask",
                self.input_file,
                "--config",
                self.config_file,
                "--flat_image",
                "nonexistent_flat.fits",
            ],
        ):
            result = run_pipeline()
            self.assertEqual(result, 1)


class TestCLIConfigFallback(unittest.TestCase):
    @patch("weightmask.cli.validate_fits_file")
    @patch("os.path.exists")
    @patch.object(sys, "argv", ["weightmask", "dummy.fits"])
    def test_missing_default_config(self, mock_exists, mock_validate_fits):
        def exists_side_effect(path):
            if path == "dummy.fits":
                return True
            return False

        mock_exists.side_effect = exists_side_effect
        mock_validate_fits.return_value = True

        with patch("builtins.print") as mock_print:
            result = run_pipeline()
            self.assertEqual(result, 1)
            mock_print.assert_any_call("ERROR: Config file not specified and no default found.")

    @patch("weightmask.cli.validate_fits_file")
    @patch("os.path.exists")
    @patch("builtins.open")
    @patch.object(sys, "argv", ["weightmask", "dummy.fits"])
    def test_first_fallback_config_found(self, mock_open, mock_exists, mock_validate_fits):
        def exists_side_effect(path):
            if path == "dummy.fits":
                return True
            if path == "weightmask.yml":
                return True
            return False

        mock_exists.side_effect = exists_side_effect
        mock_validate_fits.return_value = True
        mock_open.side_effect = OSError("Mocked error to stop pipeline")

        with patch("builtins.print") as mock_print:
            result = run_pipeline()
            self.assertEqual(result, 1)
            mock_print.assert_any_call("Using default config file found at: weightmask.yml")

    def test_only_weightmask_yml_is_implicitly_loaded(self):
        from weightmask.cli import load_configuration

        with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp):
            for name in ("config.yml", ".weightmask.yml"):
                with self.subTest(name=name):
                    with open(name, "w") as handle:
                        handle.write("{}\n")
                    self.assertIsNone(load_configuration(None))
                    self.assertIsNotNone(load_configuration(name))
            with open("weightmask.yml", "w") as handle:
                handle.write("{}\n")
            self.assertIsNotNone(load_configuration(None))


class TestCliHelp(unittest.TestCase):
    def test_nproc_is_the_only_worker_option(self):
        from weightmask.cli import parse_arguments

        self.assertEqual(parse_arguments(["in.fits", "--nproc", "4"]).max_workers, 4)
        with self.assertRaises(SystemExit) as cm:
            parse_arguments(["in.fits", "--max-workers", "4"])
        self.assertEqual(cm.exception.code, 2)

    def test_weightmask_help_mentions_config(self):
        from io import StringIO

        from weightmask.cli import parse_arguments

        buf = StringIO()
        with patch("sys.stdout", buf):
            with self.assertRaises(SystemExit) as cm:
                parse_arguments(["--help"])
        self.assertEqual(cm.exception.code, 0)
        text = buf.getvalue()
        self.assertTrue(text.strip())
        self.assertIn("--config", text)
        self.assertIn("Inputs", text)
        self.assertIn("Outputs", text)
        self.assertIn("Run", text)
        self.assertIn("--version", text)
        self.assertIn("bundled", text)
        self.assertNotIn("not installed", text.lower())
        self.assertNotIn("not bundled", text.lower())

    def test_reconstruct_sky_help_mentions_output(self):
        from io import StringIO

        from weightmask.reconstruct_sky import parse_args

        buf = StringIO()
        with patch("sys.stdout", buf):
            with self.assertRaises(SystemExit) as cm:
                parse_args(["--help"])
        self.assertEqual(cm.exception.code, 0)
        text = buf.getvalue()
        self.assertTrue(text.strip())
        self.assertIn("--output", text)

    def test_version_names_the_program_not_the_module_file(self):
        """`python -m weightmask.cli --version` used to print `cli.py 0.2.0`.

        argparse derives prog from sys.argv[0], so the same install reported
        "cli.py" by module and "weightmask" as the console script.
        """
        from io import StringIO

        from weightmask import __version__
        from weightmask.cli import parse_arguments

        buf = StringIO()
        with patch("sys.stdout", buf):
            with self.assertRaises(SystemExit) as cm:
                parse_arguments(["--version"])
        self.assertEqual(cm.exception.code, 0)
        self.assertEqual(buf.getvalue().strip(), f"weightmask {__version__}")

    def test_every_declared_flag_is_actually_read(self):
        """A flag in --help that the pipeline ignores is worse than no flag.

        Captures the real parser rather than re-deriving dests from flag names,
        because --nproc carries an explicit dest="max_workers". Each option's
        dest must be referenced somewhere besides its own declaration, or the
        flag shows in help and silently does nothing.
        """
        import argparse
        import inspect
        import re

        from weightmask import cli

        # Spy on the method rather than rebinding argparse.ArgumentParser to a
        # subclass: Python 3.13's argparse calls super(ArgumentParser, self),
        # so rebinding the module name makes that super() resolve to the
        # subclass and __init__ recurses forever.
        captured = []
        real_parse_known_args = argparse.ArgumentParser.parse_known_args

        def spy(self, args=None, namespace=None):
            captured.append(self)
            return real_parse_known_args(self, args, namespace)

        with patch.object(argparse.ArgumentParser, "parse_known_args", spy):
            cli.parse_arguments(["in.fits"])
        self.assertEqual(len(captured), 1, "failed to capture the parser")
        parser = captured[0]

        source = inspect.getsource(cli)
        options = [action for action in parser._actions if action.option_strings and action.dest != argparse.SUPPRESS]
        self.assertGreater(len(options), 10, "failed to find the declared options")

        for action in options:
            occurrences = len(re.findall(rf"\b{re.escape(action.dest)}\b", source))
            self.assertGreater(
                occurrences,
                1,
                f"{'/'.join(action.option_strings)} (dest {action.dest!r}) is declared "
                "but never read in cli.py: it would show in --help and do nothing",
            )


class TestReconstructSkyCLI(unittest.TestCase):
    def test_reconstruct_sky_roundtrip_fits(self):
        import sep

        from weightmask.background import sky_to_mesh
        from weightmask.reconstruct_sky import main as reconstruct_sky_main

        rng = np.random.default_rng(1)
        data = (1100.0 + rng.normal(0.0, 5.0, (96, 128))).astype(np.float64)
        sky = sep.Background(data, bw=32, bh=32, fw=3, fh=3).back().astype(np.float32)
        mesh, cards = sky_to_mesh(sky, 32)
        with tempfile.TemporaryDirectory() as tmp:
            mesh_path = os.path.join(tmp, "sky_mesh.fits")
            out_path = os.path.join(tmp, "sky_full.fits")
            fitsio.write(mesh_path, mesh, header=cards, clobber=True)
            rc = reconstruct_sky_main([mesh_path, "-o", out_path])
            self.assertEqual(rc, 0)
            with fitsio.FITS(out_path, "r") as f:
                rec = f[0].read()
                hdr = f[0].read_header()
            self.assertEqual(rec.shape, sky.shape)
            self.assertNotIn("SKYMESH", {str(k).upper() for k in hdr.keys()})
            self.assertLess(float(np.max(np.abs(rec - sky))), 0.05)

    def test_reconstruct_sky_missing_output_flag(self):
        from weightmask.reconstruct_sky import main as reconstruct_sky_main

        with self.assertRaises(SystemExit) as cm:
            reconstruct_sky_main(["missing.fits"])
        self.assertEqual(cm.exception.code, 2)

    def test_reconstruct_sky_is_an_ordinary_science_filename(self):
        with tempfile.TemporaryDirectory() as tmp, contextlib.chdir(tmp):
            fitsio.write("reconstruct-sky", np.full((32, 32), 1000.0, dtype=np.float32), clobber=True)
            with open("weightmask.yml", "w") as handle:
                handle.write("{}\n")
            self.assertEqual(run_pipeline(["reconstruct-sky", "-o", "out.weight.fits"]), 0)
            self.assertEqual(fitsio.read("out.weight.fits").shape, (32, 32))


class TestValidateInputFiles(unittest.TestCase):
    """`validate_input_files` must not stat a CFITSIO spec as if it were a path."""

    def test_hdu_spec_in_input_file_is_accepted(self):
        # docs/usage.md advertises `science.fits[1]`. The spec is not part of
        # the path, so os.path.exists("s.fits[1]") is False and the run used to
        # abort with "Input file not found" before the spec was ever parsed.
        with tempfile.TemporaryDirectory() as tmp:
            sp = os.path.join(tmp, "s.fits")
            fitsio.write(sp, np.full((32, 32), 1000.0, dtype=np.float32), clobber=True)
            args = Namespace(input_file=f"{sp}[0]", flat_image=None, badpix_mask=None, dark_image=None)
            self.assertTrue(validate_input_files(args))
            # An out-of-range index is the run's problem, not validation's.
            args.input_file = f"{sp}[7]"
            self.assertTrue(validate_input_files(args))

    def test_partial_run_exits_nonzero_and_says_so(self):
        """A missing CCD must not be reported as a successful run.

        run_pipeline returned 0 whenever at least one HDU succeeded, so a run
        with a failed CCD wrote fewer data extensions than there are science
        HDUs and still exited 0. From the first skipped HDU onward, extension
        position no longer matches the science HDU.
        """
        with tempfile.TemporaryDirectory() as tmp:
            sp = os.path.join(tmp, "s.fits")
            big, odd = (32, 32), (24, 24)
            fitsio.write(sp, None, clobber=True)
            with fitsio.FITS(sp, "rw") as handle:
                for i in range(3):
                    handle.write(np.full(big, 1000.0, dtype=np.float32), header={"CCDID": f"CCD{i}"})
            fp = os.path.join(tmp, "f.fits")
            fitsio.write(fp, None, clobber=True)
            with fitsio.FITS(fp, "rw") as handle:
                handle.write(np.ones(big, dtype=np.float32))
                handle.write(np.ones(odd, dtype=np.float32))  # shape mismatch -> HDU 2 fails
                handle.write(np.ones(big, dtype=np.float32))
            out = os.path.join(tmp, "o.fits")
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                code = run_pipeline([sp, "-o", out, "--flat_image", fp])
            log = buf.getvalue()
            self.assertNotEqual(code, 0, "a partial run must not exit 0")
            self.assertIn("only 2 of 3 HDUs", log)
            self.assertIn("No new product set was published", log)
            self.assertFalse(os.path.exists(out))

    def test_handles_are_released_even_when_the_run_raises(self):
        """_cleanup_hdul used to run only on the success path."""
        with tempfile.TemporaryDirectory() as tmp:
            sp = os.path.join(tmp, "s.fits")
            fitsio.write(sp, np.full((32, 32), 1000.0, dtype=np.float32), clobber=True)
            from weightmask import cli as cli_mod

            calls = []
            real = cli_mod._cleanup_hdul

            def spy(*a, **k):
                calls.append(tuple(x is not None for x in a))
                return real(*a, **k)

            with contextlib.redirect_stdout(io.StringIO()):
                with patch.object(cli_mod, "_cleanup_hdul", side_effect=spy):
                    with patch.object(cli_mod, "process_all_hdus", side_effect=RuntimeError("boom")):
                        with self.assertRaises(RuntimeError):
                            run_pipeline([sp, "-o", os.path.join(tmp, "o.fits")])
            self.assertEqual(len(calls), 1, "handles leaked on the exception path")
            self.assertTrue(calls[0][0], "the input handle was open and must be closed")

    def test_open_fits_files_closes_the_input_when_the_flat_fails(self):
        """Returning (None, None) used to drop an already-open input handle."""
        from weightmask import cli as cli_mod

        with tempfile.TemporaryDirectory() as tmp:
            sp = os.path.join(tmp, "s.fits")
            fitsio.write(sp, np.full((32, 32), 1000.0, dtype=np.float32), clobber=True)
            state = {"closed": False}
            real_fits = cli_mod.fitsio.FITS

            def fake_fits(path, *a, **k):
                if "nonexistent" in str(path):
                    raise OSError("cannot open flat")
                handle = real_fits(path, *a, **k)
                real_close = handle.close

                def close():
                    state["closed"] = True
                    return real_close()

                handle.close = close
                return handle

            with patch.object(cli_mod.fitsio, "FITS", side_effect=fake_fits):
                with contextlib.redirect_stdout(io.StringIO()):
                    got = cli_mod.open_fits_files(sp, "/nonexistent/flat.fits")
            self.assertEqual(got, (None, None))
            self.assertTrue(state["closed"], "input handle was opened then dropped unclosed")

    def test_hdu_spec_on_an_auxiliary_input_is_rejected(self):
        # A flat/dark/keep-map is matched to each science HDU by index, so an
        # explicit [N] has no meaning and used to be parsed into a variable
        # nothing read -- accepted, then silently ignored.
        with tempfile.TemporaryDirectory() as tmp:
            sp = os.path.join(tmp, "s.fits")
            fp = os.path.join(tmp, "f.fits")
            fitsio.write(sp, np.full((32, 32), 1000.0, dtype=np.float32), clobber=True)
            fitsio.write(fp, np.ones((32, 32), dtype=np.float32), clobber=True)
            args = Namespace(input_file=sp, flat_image=f"{fp}[0]", badpix_mask=None, dark_image=None)
            self.assertFalse(validate_input_files(args))
            args.flat_image = fp
            self.assertTrue(validate_input_files(args))

    def test_missing_input_still_reports_the_clean_path(self):
        # CFITSIO puts the spec at the end; `extract_hdu_spec` only strips a
        # trailing "[N]", so this is the form that must be reported cleanly.
        args = Namespace(input_file="/nonexistent/dir.fits[2]", flat_image=None, badpix_mask=None, dark_image=None)
        with contextlib.redirect_stdout(io.StringIO()) as buf:
            self.assertFalse(validate_input_files(args))
        self.assertIn("/nonexistent/dir.fits", buf.getvalue())
        self.assertNotIn("[2]", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
