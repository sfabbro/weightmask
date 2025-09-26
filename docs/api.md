# API Documentation

## Module Overview

WeightMask is organized into several modules, each responsible for a specific aspect of mask and weight map generation:

- `weightmask.cli`: Command-line interface
- `weightmask.bad`: Bad pixel detection
- `weightmask.satur`: Saturation detection
- `weightmask.cosmics`: Cosmic ray detection
- `weightmask.objects`: Object detection
- `weightmask.streaks`: Streak detection
- `weightmask.background`: Background estimation
- `weightmask.variance`: Variance calculation
- `weightmask.weight`: Weight and confidence map generation
- `weightmask.utils`: Utility functions

## CLI Module

### run_pipeline()
```python
def run_pipeline() -> int
```
Main function to parse arguments and run the pipeline.

Returns:
- `int`: Exit code (0 for success, 1 for error)

## Bad Pixel Detection

### detect_bad_pixels()
```python
def detect_bad_pixels(flat_data, config, using_unit_flat=False)
```
Detect bad pixels and columns in the flat field.

Args:
- `flat_data` (ndarray): Flat field data array
- `config` (dict): Configuration dictionary for flat masking
- `using_unit_flat` (bool): Whether a unit flat (all 1.0) is being used

Returns:
- `ndarray`: Boolean mask of bad pixels and columns (True = bad)

## Saturation Detection

### detect_saturated_pixels()
```python
def detect_saturated_pixels(sci_data, sci_hdr, config)
```
Detect saturated pixels in the science data using the configured method.

Args:
- `sci_data` (ndarray): Science image data array (float32 recommended)
- `sci_hdr` (fits.Header): Science image header
- `config` (dict): Configuration dictionary for saturation detection

Returns:
- `tuple`: (saturation_level, sat_method_used, sat_mask_bool)
  - `saturation_level` (float): The determined saturation level in ADU
  - `sat_method_used` (str): Method used ('histogram', 'header', 'default')
  - `sat_mask_bool` (ndarray): Boolean mask where True indicates saturated pixels

## Cosmic Ray Detection

### detect_cosmic_rays()
```python
def detect_cosmic_rays(sci_data, existing_mask, saturation_level, gain, read_noise, config, bkg_rms_map=None)
```
Detect cosmic rays in the science data.

Args:
- `sci_data` (ndarray): Science image data array
- `existing_mask` (ndarray): Boolean mask of already masked pixels
- `saturation_level` (float): Saturation level for the detector
- `gain` (float): Gain value in e-/ADU
- `read_noise` (float): Read noise in electrons
- `config` (dict): Configuration dictionary for cosmic ray detection
- `bkg_rms_map` (ndarray, optional): Background RMS map for dynamic sigclip.

Returns:
- `ndarray`: Boolean mask of newly detected cosmic ray pixels

## Object Detection

### detect_objects()
```python
def detect_objects(data_sub, bkg_rms_map, existing_mask, config)
```
Detect astronomical objects in the background-subtracted image.

Args:
- `data_sub` (ndarray): Background-subtracted image data
- `bkg_rms_map` (ndarray): Background RMS map
- `existing_mask` (ndarray): Boolean mask of already masked pixels
- `config` (dict): Configuration dictionary for object detection

Returns:
- `ndarray`: Boolean mask of newly detected object pixels

## Streak Detection

### detect_streaks()
```python
def detect_streaks(data_sub, bkg_rms_map, existing_mask, config)
```
Detect linear streaks using the method specified in the configuration.

Args:
- `data_sub` (ndarray): Background-subtracted image data
- `bkg_rms_map` (ndarray): Background RMS map
- `existing_mask` (ndarray): Boolean mask of already masked pixels
- `config` (dict): Configuration dictionary for streak detection

Returns:
- `ndarray`: Boolean mask of newly detected streak pixels

## Background Estimation

### estimate_background()
```python
def estimate_background(sci_data, mask, config)
```
Estimate background and background RMS using SEP.

Args:
- `sci_data` (ndarray): Science image data array
- `mask` (ndarray): Boolean mask of pixels to exclude from background estimation
- `config` (dict): Configuration dictionary for background estimation

Returns:
- `tuple`: (background_map, background_rms_map)

## Variance Calculation

### calculate_inverse_variance()
```python
def calculate_inverse_variance(variance_cfg, sky_map, flat_map, bkg_rms_map, sci_data=None, obj_mask=None)
```
Calculate inverse variance map using the specified method.

Args:
- `variance_cfg` (dict): Configuration dictionary for variance calculation.
- `sky_map` (ndarray): Background sky map in ADU.
- `flat_map` (ndarray): Flat field response map.
- `bkg_rms_map` (ndarray): Background RMS map in ADU (from SEP, used by 'rms_map').
- `sci_data` (ndarray, optional): Full science data array, required for 'empirical_fit'.
- `obj_mask` (ndarray, optional): Object mask, required for 'empirical_fit'.

Returns:
- `ndarray or None`: Inverse variance map, or None if method is invalid or prerequisites missing.

## Weight and Confidence Maps

### generate_weight_and_confidence()
```python
def generate_weight_and_confidence(inv_variance_map, final_mask_int, config)
```
Calculates the weight map (masked inverse variance) and a confidence map (normalized weight map).

Args:
- `inv_variance_map` (ndarray): The calculated inverse variance map (unmasked)
- `final_mask_int` (ndarray): The final combined integer bitmask
- `config` (dict): Configuration dictionary

Returns:
- `tuple`: (weight_map, confidence_map)

## Utility Functions

### extract_hdu_spec()
```python
def extract_hdu_spec(filepath)
```
Extract HDU specifier from CFITSIO-style filename (e.g., 'file.fits[1]')

Args:
- `filepath` (str): Path with potential HDU specifier

Returns:
- `tuple`: (clean_path, hdu_index) where hdu_index is None if not specified

### create_binary_mask()
```python
def create_binary_mask(mask_data, bit_flag)
```
Create a binary mask (0/1) from a bitmask for a specific flag.

Args:
- `mask_data` (ndarray): Bitmask array
- `bit_flag` (int): Bit flag to extract

Returns:
- `ndarray`: Binary mask (0=not set, 1=set)