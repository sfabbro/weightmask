import unittest
from weightmask.cli import validate_config

class TestCLI(unittest.TestCase):
    def test_validate_config_valid(self):
        """Test with a fully valid configuration."""
        valid_config = {
            'flat_masking': {},
            'saturation': {},
            'sep_background': {},
            'cosmic_ray': {},
            'sep_objects': {},
            'streak_masking': {},
            'variance': {'method': 'theoretical'},
            'confidence_params': {},
            'output_params': {}
        }
        self.assertTrue(validate_config(valid_config))

    def test_validate_config_missing_sections(self):
        """Test with missing sections. It should print warnings but return True."""
        # A completely empty config
        empty_config = {}
        self.assertTrue(validate_config(empty_config))

    def test_validate_config_invalid_variance_not_dict(self):
        """Test with an invalid variance section (not a dictionary)."""
        invalid_config = {
            'variance': 'not_a_dict'
        }
        self.assertFalse(validate_config(invalid_config))

    def test_validate_config_invalid_variance_method(self):
        """Test with an invalid variance method."""
        invalid_config = {
            'variance': {'method': 'invalid_method'}
        }
        self.assertFalse(validate_config(invalid_config))

    def test_validate_config_valid_variance_methods(self):
        """Test with all valid variance methods."""
        for method in ['theoretical', 'rms_map', 'empirical_fit']:
            valid_config = {
                'variance': {'method': method}
            }
            self.assertTrue(validate_config(valid_config))

if __name__ == '__main__':
    unittest.main()
