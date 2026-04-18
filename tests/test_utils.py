import unittest

import numpy as np

from weightmask.utils import clean_config_dict, create_binary_mask, extract_hdu_spec


class TestUtils(unittest.TestCase):
    def test_extract_hdu_spec_with_hdu(self):
        """Test extracting HDU specifier from path with HDU."""
        path = "file.fits[1]"
        clean_path, hdu_index = extract_hdu_spec(path)

        self.assertEqual(clean_path, "file.fits")
        self.assertEqual(hdu_index, 1)

    def test_extract_hdu_spec_without_hdu(self):
        """Test extracting HDU specifier from path without HDU."""
        path = "file.fits"
        clean_path, hdu_index = extract_hdu_spec(path)

        self.assertEqual(clean_path, "file.fits")
        self.assertIsNone(hdu_index)

    def test_extract_hdu_spec_none_input(self):
        """Test extracting HDU specifier with None input."""
        clean_path, hdu_index = extract_hdu_spec(None)

        self.assertIsNone(clean_path)
        self.assertIsNone(hdu_index)

    def test_create_binary_mask(self):
        """Test creating binary mask from bitmask."""
        # Create a test bitmask
        mask_data = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=np.uint32)

        # Test with bit flag 1 (first bit)
        binary_mask = create_binary_mask(mask_data, 1)
        expected = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=np.uint8)
        np.testing.assert_array_equal(binary_mask, expected)

        # Test with bit flag 2 (second bit)
        binary_mask = create_binary_mask(mask_data, 2)
        expected = np.array([0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1], dtype=np.uint8)
        np.testing.assert_array_equal(binary_mask, expected)

        # Test with bit flag 4 (third bit)
        binary_mask = create_binary_mask(mask_data, 4)
        expected = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0], dtype=np.uint8)
        np.testing.assert_array_equal(binary_mask, expected)

    def test_clean_config_dict_empty(self):
        """Test clean_config_dict with empty dictionaries."""
        self.assertEqual(clean_config_dict({}), {})
        self.assertEqual(clean_config_dict(None), {})

    def test_clean_config_dict_booleans(self):
        """Test clean_config_dict with boolean strings."""
        config = {
            "t1": "true",
            "t2": "True",
            "t3": "TRUE",
            "t4": "yes",
            "t5": "on",
            "f1": "false",
            "f2": "False",
            "f3": "FALSE",
            "f4": "no",
            "f5": "off",
        }
        cleaned = clean_config_dict(config)
        for i in range(1, 6):
            self.assertTrue(cleaned[f"t{i}"])
            self.assertFalse(cleaned[f"f{i}"])

    def test_clean_config_dict_numbers(self):
        """Test clean_config_dict with numeric strings."""
        config = {"int1": "42", "int2": "-10", "float1": "3.14", "float2": "-0.001", "sci1": "1e-5", "sci2": "2.5E4"}
        cleaned = clean_config_dict(config)
        self.assertEqual(cleaned["int1"], 42)
        self.assertEqual(cleaned["int2"], -10)
        self.assertEqual(cleaned["float1"], 3.14)
        self.assertEqual(cleaned["float2"], -0.001)
        self.assertEqual(cleaned["sci1"], 1e-5)
        self.assertEqual(cleaned["sci2"], 25000.0)

    def test_clean_config_dict_strings(self):
        """Test clean_config_dict with regular strings."""
        config = {
            "s1": "hello",
            "s2": "a longer string",
            "s3": "123a",  # Not fully numeric
        }
        cleaned = clean_config_dict(config)
        self.assertEqual(cleaned["s1"], "hello")
        self.assertEqual(cleaned["s2"], "a longer string")
        self.assertEqual(cleaned["s3"], "123a")

    def test_clean_config_dict_other_types(self):
        """Test clean_config_dict with non-string types."""
        config = {"b1": True, "b2": False, "i1": 100, "f1": 2.718, "n1": None}
        cleaned = clean_config_dict(config)
        self.assertEqual(cleaned, config)

    def test_clean_config_dict_nested_dict(self):
        """Test clean_config_dict with nested dictionaries."""
        config = {"level1": {"val_str": "123", "val_bool": "true", "level2": {"val_float": "1.23"}}}
        cleaned = clean_config_dict(config)
        self.assertEqual(cleaned["level1"]["val_str"], 123)
        self.assertTrue(cleaned["level1"]["val_bool"])
        self.assertEqual(cleaned["level1"]["level2"]["val_float"], 1.23)

    def test_clean_config_dict_lists(self):
        """Test clean_config_dict with lists."""
        config = {
            "list1": ["1", "true", "3.14", "hello", {"nested": "42"}],
            "list2": [1, True, 3.14],  # Already typed
        }
        cleaned = clean_config_dict(config)
        self.assertEqual(cleaned["list1"], [1, True, 3.14, "hello", {"nested": 42}])
        self.assertEqual(cleaned["list2"], [1, True, 3.14])


if __name__ == "__main__":
    unittest.main()
