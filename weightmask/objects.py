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
    # Use uint8 for SEP compatibility, convert to bool at the end
    object_mask_ui8 = np.zeros(data_sub.shape, dtype=np.uint8)
    
    try:
        # Create a deep copy of config and clean all numeric values
        # This handles cases where YAML or CLI might pass numbers as strings
        clean_config = {}
        if config:
            for k, v in config.items():
                if isinstance(v, (str, bytes)):
                    try:
                        # Try int first for flags/counts, then float
                        if v.lower() in ('true', 'yes', 'on'): clean_config[k] = True
                        elif v.lower() in ('false', 'no', 'off'): clean_config[k] = False
                        else:
                            try:
                                if '.' in v: clean_config[k] = float(v)
                                else: clean_config[k] = int(v)
                            except ValueError:
                                clean_config[k] = float(v)
                    except (ValueError, TypeError):
                        clean_config[k] = v
                else:
                    clean_config[k] = v

        extract_thresh = float(clean_config.get('extract_thresh', 3.0))
        min_area = int(clean_config.get('min_area', 10))
        
        # Force EVERYTHING to be clean, C-contiguous 32-bit floats
        d_sub = np.require(data_sub, dtype=np.float32, requirements=['C', 'A'])
        b_rms = np.require(bkg_rms_map, dtype=np.float32, requirements=['C', 'A'])
        
        m_in = None
        if existing_mask is not None:
            m_in = np.require(existing_mask, dtype=np.bool_, requirements=['C', 'A'])
            
        # Extract objects using SEP
        objects = sep.extract(
            d_sub, 
            thresh=extract_thresh, 
            err=b_rms, 
            mask=m_in, 
            minarea=min_area, 
            segmentation_map=False
        )
        
        print(f"  Detected {len(objects)} objects (thresh={extract_thresh:.1f} sigma).")
        
        if len(objects) > 0:
            base_k = float(clean_config.get('ellipse_k', 2.0))
            alpha = float(clean_config.get('halo_scale_factor', 0.5))
            
            # 1. Calculate semi-axes with optional halo scaling
            if clean_config.get('dynamic_halo_scaling', True):
                print("  Applying dynamic halo masking based on object flux...")
                valid_fluxes = np.clip(objects['flux'], 1e-5, None)
                min_flux = np.percentile(valid_fluxes, 10)
                flux_ratio = np.clip(valid_fluxes / min_flux, 1.0, None)
                scale_multiplier = 1.0 + alpha * np.log10(flux_ratio)
            else:
                scale_multiplier = np.ones(len(objects))
                
            scaled_a = objects['a'] * scale_multiplier
            scaled_b = objects['b'] * scale_multiplier
            
            # Manual ellipse drawing for maximum robustness
            from skimage.draw import ellipse
            h, w = object_mask_ui8.shape
            for i in range(len(objects)):
                if scaled_a[i] > 0 and scaled_b[i] > 0:
                    try:
                        # skimage.draw.ellipse(r, c, ...) - r=row(y), c=col(x)
                        cy, cx = objects['y'][i], objects['x'][i]
                        ry, rx = scaled_b[i] * base_k, scaled_a[i] * base_k
                        
                        rr, cc = ellipse(
                            int(cy + 0.5), int(cx + 0.5), 
                            ry, rx, 
                            shape=(h, w), rotation=-objects['theta'][i]
                        )
                        object_mask_ui8[rr, cc] = 1
                    except Exception:
                        continue
            
            if clean_config.get('dynamic_halo_scaling', True):
                print(f"    Halo scaling multiplier range: 1.0x to {np.max(scale_multiplier):.2f}x")
            
            # --- 2. Diffraction Spike Masking ---
            if clean_config.get('spike_enable', True):
                spike_thresh = float(clean_config.get('spike_flux_thresh', 1e5))
                bright_mask = objects['flux'] > spike_thresh
                
                if np.any(bright_mask):
                    print(f"    Applying diffraction spike masking to {np.sum(bright_mask)} bright stars (Flux > {spike_thresh:.2e})...")
                    spike_length_base = int(clean_config.get('spike_length_base', 100))
                    spike_width = int(clean_config.get('spike_width', 3))
                    
                    h, w = object_mask_ui8.shape
                    for obj in objects[bright_mask]:
                        # Scale spike length slightly by flux
                        s_len = int(spike_length_base * (1.0 + 0.2 * np.log10(obj['flux'] / spike_thresh)))
                        xc, yc = int(obj['x'] + 0.5), int(obj['y'] + 0.5)
                        
                        # Horizontal spike
                        xstart, xend = max(0, xc - s_len), min(w - 1, xc + s_len)
                        for dw in range(-(spike_width//2), spike_width//2 + 1):
                            if 0 <= yc + dw < h:
                                object_mask_ui8[yc + dw, xstart:xend+1] = 1
                                
                        # Vertical spike
                        ystart, yend = max(0, yc - s_len), min(h - 1, yc + s_len)
                        for dw in range(-(spike_width//2), spike_width//2 + 1):
                            if 0 <= xc + dw < w:
                                object_mask_ui8[ystart:yend+1, xc + dw] = 1
            
            # Convert back to boolean for returning
            object_mask_bool = object_mask_ui8.astype(bool)
            
            # Only return newly detected pixels (not already in existing_mask)
            # Use original boolean cast to avoid bitwise issues
            m_orig = existing_mask.astype(bool) if existing_mask is not None else np.zeros_like(object_mask_bool)
            obj_add_mask = object_mask_bool & (~m_orig)
            return obj_add_mask
            
    except Exception as e:
        # print(f"  Object detection failed: {e}")
        # import traceback
        # traceback.print_exc()
        pass
    
    return np.zeros(data_sub.shape, dtype=bool)
