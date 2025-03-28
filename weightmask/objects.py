import numpy as np
import sep

def detect_objects(data_sub, bkg_rms_map, existing_mask, config):
    """
    Detect astronomical objects in the background-subtracted image.
    
    Args:
        data_sub (ndarray): Background-subtracted image data
        bkg_rms_map (ndarray): Background RMS map
        existing_mask (ndarray): Boolean mask of already masked pixels
        config (dict): Configuration dictionary for object detection
        
    Returns:
        ndarray: Boolean mask of newly detected object pixels
    """
    object_mask_bool = np.zeros(data_sub.shape, dtype=bool)
    
    try:
        # Extract objects using SEP
        objects = sep.extract(
            data_sub, 
            thresh=config['extract_thresh'], 
            err=bkg_rms_map, 
            mask=existing_mask, 
            minarea=config['min_area'], 
            segmentation_map=False
        )
        
        print(f"  Detected {len(objects)} objects (thresh={config['extract_thresh']} sigma).")
        
        if len(objects) > 0:
            # Create elliptical masks for each detected object
            sep.mask_ellipse(
                object_mask_bool, 
                objects['x'], objects['y'], 
                objects['a'], objects['b'], 
                objects['theta'], 
                r=config['ellipse_k']
            )
            
            # Only return newly detected pixels (not already in existing_mask)
            obj_add_mask = object_mask_bool & (~existing_mask)
            return obj_add_mask
            
    except Exception as e:
        print(f"  SEP extraction failed: {e}")
    
    return np.zeros(data_sub.shape, dtype=bool)
