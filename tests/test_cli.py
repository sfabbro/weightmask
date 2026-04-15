import os
import tempfile
import unittest
from unittest.mock import patch
import fitsio
import numpy as np

from weightmask.cli import run_pipeline

class TestCLI(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.TemporaryDirectory()
        self.workspace = self.test_dir.name

        # Create a basic FITS file
        self.input_file = os.path.join(self.workspace, 'test_input.fits')
        data = np.random.normal(100, 10, (100, 100)).astype(np.float32)
        fitsio.write(self.input_file, data, clobber=True)

        # Create a minimal config file
        self.config_file = os.path.join(self.workspace, 'test_config.yml')
        with open(self.config_file, 'w') as f:
            f.write("""
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
""")

    def tearDown(self):
        self.test_dir.cleanup()

    @patch('sys.argv', ['weightmask'])
    def test_run_pipeline_missing_args(self):
        # argparse will call sys.exit(2) if required args are missing
        with self.assertRaises(SystemExit) as cm:
            run_pipeline()
        self.assertEqual(cm.exception.code, 2)

    def test_run_pipeline_missing_input_file(self):
        with patch('sys.argv', ['weightmask', 'nonexistent.fits', '--config', self.config_file]):
            result = run_pipeline()
            self.assertEqual(result, 1)

    def test_run_pipeline_success(self):
        output_file = os.path.join(self.workspace, 'output.weight.fits')
        with patch('sys.argv', ['weightmask', self.input_file, '--config', self.config_file, '-o', output_file]):
            result = run_pipeline()
            self.assertEqual(result, 0)
            self.assertTrue(os.path.exists(output_file))

    def test_run_pipeline_individual_masks(self):
        output_file = os.path.join(self.workspace, 'output2.weight.fits')
        with patch('sys.argv', ['weightmask', self.input_file, '--config', self.config_file, '-o', output_file, '--individual_masks']):
            result = run_pipeline()
            self.assertEqual(result, 0)
            self.assertTrue(os.path.exists(output_file))
            self.assertTrue(os.path.exists(os.path.join(self.workspace, 'output2.weight.bad.fits')))
            self.assertTrue(os.path.exists(os.path.join(self.workspace, 'output2.weight.sat.fits')))

    def test_run_pipeline_missing_config(self):
        with patch('sys.argv', ['weightmask', self.input_file, '--config', 'nonexistent_config.yml']):
            result = run_pipeline()
            self.assertEqual(result, 1)

    def test_run_pipeline_bad_config(self):
        bad_config_file = os.path.join(self.workspace, 'bad_config.yml')
        with open(bad_config_file, 'w') as f:
            f.write("this is not a valid yaml file: [")
        with patch('sys.argv', ['weightmask', self.input_file, '--config', bad_config_file]):
            result = run_pipeline()
            self.assertEqual(result, 1)

    def test_run_pipeline_missing_flat(self):
        with patch('sys.argv', ['weightmask', self.input_file, '--config', self.config_file, '--flat_image', 'nonexistent_flat.fits']):
            result = run_pipeline()
            self.assertEqual(result, 1)

if __name__ == '__main__':
    unittest.main()
