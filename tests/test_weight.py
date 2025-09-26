import numpy as np
import unittest
from weightmask.weight import generate_weight_and_confidence


class TestWeight(unittest.TestCase):
    
    def test_generate_weight_and_confidence(self):
        """Test weight and confidence map generation."""
        # Create test inverse variance map
        inv_variance_map = np.full((100, 100), 0.01, dtype=np.float32)
        
        # Create test mask with some bad pixels
        final_mask_int = np.zeros((100, 100), dtype=np.uint32)
        final_mask_int[10, 10] = 1  # BAD pixel
        final_mask_int[20, 20] = 2  # SAT pixel
        final_mask_int[30, 30] = 4  # CR pixel
        
        config = {
            'output_params': {
                'mask_detected_in_weight': False
            },
            'confidence_params': {
                'dtype': 'float32',
                'normalize_percentile': 99.0,
                'scale_to_100': False
            }
        }
        
        weight_map, confidence_map = generate_weight_and_confidence(inv_variance_map, final_mask_int, config)
        
        # Check that we got results
        self.assertIsNotNone(weight_map)
        self.assertIsNotNone(confidence_map)
        
        # Check that results are numpy arrays
        self.assertIsInstance(weight_map, np.ndarray)
        self.assertIsInstance(confidence_map, np.ndarray)
        
        # Check that results have correct shape
        self.assertEqual(weight_map.shape, inv_variance_map.shape)
        self.assertEqual(confidence_map.shape, inv_variance_map.shape)
        
        # Check that bad pixels are masked in weight map (if we have a result)
        if weight_map is not None:
            self.assertEqual(weight_map[10, 10], 0.0)  # BAD pixel
            self.assertEqual(weight_map[20, 20], 0.0)  # SAT pixel
            self.assertEqual(weight_map[30, 30], 0.0)  # CR pixel
            
            # Check that good pixels have non-zero weight
            self.assertNotEqual(weight_map[50, 50], 0.0)  # Good pixel
    
    def test_generate_weight_and_confidence_mask_detected(self):
        """Test weight and confidence map generation with detected objects masked."""
        # Create test inverse variance map
        inv_variance_map = np.full((100, 100), 0.01, dtype=np.float32)
        
        # Create test mask with detected objects
        final_mask_int = np.zeros((100, 100), dtype=np.uint32)
        final_mask_int[10, 10] = 8  # DETECTED object
        
        config = {
            'output_params': {
                'mask_detected_in_weight': True  # Mask detected objects
            },
            'confidence_params': {
                'dtype': 'float32',
                'normalize_percentile': 99.0,
                'scale_to_100': False
            }
        }
        
        weight_map, confidence_map = generate_weight_and_confidence(inv_variance_map, final_mask_int, config)
        
        # Check that we got results
        self.assertIsNotNone(weight_map)
        self.assertIsNotNone(confidence_map)
        
        # Check that detected objects are masked when configured to do so (if we have a result)
        if weight_map is not None:
            self.assertEqual(weight_map[10, 10], 0.0)  # DETECTED object should be masked
    
    def test_generate_weight_and_confidence_no_positive_weights(self):
        """Test weight and confidence map generation with no positive weights."""
        # Create test inverse variance map with all zeros
        inv_variance_map = np.zeros((100, 100), dtype=np.float32)
        
        # Create test mask
        final_mask_int = np.zeros((100, 100), dtype=np.uint32)
        
        config = {
            'output_params': {
                'mask_detected_in_weight': False
            },
            'confidence_params': {
                'dtype': 'float32',
                'normalize_percentile': 99.0,
                'scale_to_100': False
            }
        }
        
        weight_map, confidence_map = generate_weight_and_confidence(inv_variance_map, final_mask_int, config)
        
        # Check that we got results
        self.assertIsNotNone(weight_map)
        self.assertIsNotNone(confidence_map)
        
        # Check that results are numpy arrays
        self.assertIsInstance(weight_map, np.ndarray)
        self.assertIsInstance(confidence_map, np.ndarray)
        
        # With all zero weights, confidence map should also be all zeros (if we have a result)
        if confidence_map is not None:
            self.assertTrue(np.all(confidence_map == 0.0))
    
    def test_generate_weight_and_confidence_none_input(self):
        """Test weight and confidence map generation with None input."""
        weight_map, confidence_map = generate_weight_and_confidence(None, None, {})
        
        # Should return None for both when inv_variance_map is None
        self.assertIsNone(weight_map)
        self.assertIsNone(confidence_map)


if __name__ == "__main__":
    unittest.main()