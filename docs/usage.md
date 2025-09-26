# WeightMask Usage Guide

WeightMask is a Python toolkit for generating weight maps, confidence maps, and masks for astronomical FITS images. These maps are essential for proper image stacking, coaddition, and source detection in astronomical data processing pipelines.

## Installation

To install WeightMask, clone the repository and install it in development mode:

```bash
git clone https://github.com/sfabbro/weightmask.git
cd weightmask
pip install -e .
```

## Basic Usage

The basic command to run WeightMask is:

```bash
weightmask input.fits
```

This will process the input FITS file and generate a weight map by default.

## Command Line Options

```bash
weightmask [OPTIONS] INPUT_FILE
```

### Required Arguments

- `INPUT_FILE`: Path to the input FITS file

### Optional Arguments

- `--output_map`, `-o`: Path for primary output map (Weight or Confidence). Default: `<input_base>.weight.fits`
- `--config`: Path to YAML configuration file (optional, attempts default locations)
- `--flat_image`: Path to input flat field FITS file (optional)
- `--output_mask`: Path for output bitmask FITS file (optional)
- `--output_invvar`: Path for output inverse variance FITS file (optional)
- `--output_sky`: Path for output sky background map file (optional)
- `--output_weight_raw`: Path for unnormalized weight map (masked inv_var), if different from primary map
- `--hdu`: HDU index to process (e.g., 0, 1). Processes extensions if omitted
- `--individual_masks`: Output individual mask component files

## Configuration

WeightMask uses a YAML configuration file to control its behavior. If not specified with the `--config` option, it will look for `weightmask.yml` in the current directory.

### Configuration Sections

#### Flat Masking
```yaml
flat_masking:
  low_thresh: 0.5
  high_thresh: 2.0
  col_enable: True
  col_low_var_factor: 0.05
  col_median_dev_factor: 0.1
```

#### Saturation Detection
```yaml
saturation:
  method: 'histogram'
  keyword: 'SATURATE'
  fallback_level: 65000.0
```

#### Background Estimation
```yaml
background:
  method: 'sep' # Options: 'sep', 'median_filter'

  # Parameters for method: 'sep'
  box_size: 128
  filter_size: 3
  iterations: 2 # Number of iterative background/object detection loops

  # Parameters for method: 'median_filter'
  median_kernel_size: 31 # Kernel size for the median filter
```

#### Astroscrappy Cosmic Ray Detection
```yaml
cosmic_ray:
  sigclip: 4.5
  objlim: 5.0
  dynamic_sigclip: true # Auto-adjust sigclip based on background noise
```

#### SEP Object Detection
```yaml
sep_objects:
  extract_thresh: 1.5
  min_area: 5
  ellipse_k: 2.5
```

#### Streak Masking
```yaml
streak_masking:
  enable: True
  method: 'ransac' # Options: 'ransac', 'hough'
  dilation_radius: 5

  # Advanced parameters for 'ransac' method
  ransac_params:
    use_canny: True
    canny_sigma: 1.0
    canny_low_threshold: 0.1
    canny_high_threshold: 0.5
    min_elongation: 5.0
    min_inliers: 15
```

#### Variance Calculation
```yaml
variance:
  method: 'empirical_fit' # Options: 'empirical_fit', 'theoretical', 'rms_map'
  gain_keyword: 'GAIN'
  rdnoise_keyword: 'RDNOISE'
  default_gain: 1.5
  default_rdnoise: 5.0
  epsilon: 1.0e-9
  # Parameters for 'empirical_fit' method
  empirical_patch_size: 128
  empirical_clip_sigma: 3.0
```

#### Confidence Map Parameters
```yaml
confidence_params:
  dtype: 'float32'
  normalize_percentile: 99.0
  scale_to_100: False
```

#### Output Parameters
```yaml
output_params:
  output_map_format: 'weight'
  mask_detected_in_weight: False
```

## Output Files

WeightMask can generate several types of output files:

1. **Primary Map**: Either a weight map or confidence map (configured via `output_map_format`)
2. **Bitmask**: Combined mask with different defect types flagged with bit flags
3. **Inverse Variance Map**: Pure inverse variance calculation
4. **Sky Background Map**: Estimated sky background
5. **Raw Weight Map**: Unnormalized weight map
6. **Individual Masks**: Separate masks for each defect type (when `--individual_masks` is used)

## Bit Flags

The bitmask uses the following bit flags:

- `BAD` (1): Bad pixels from flat field
- `SAT` (2): Saturated pixels
- `CR` (4): Cosmic ray hits
- `DETECTED` (8): Detected astronomical objects
- `STREAK` (16): Satellite/artifact streaks

## Examples

### Basic Processing
```bash
weightmask science.fits
```

### Process with Flat Field Correction
```bash
weightmask science.fits --flat_image flat.fits
```

### Generate All Output Types
```bash
weightmask science.fits --output_mask mask.fits --output_invvar invvar.fits --output_sky sky.fits
```

### Process Specific HDU
```bash
weightmask science.fits[1] --hdu 1
```

### Use Custom Configuration
```bash
weightmask science.fits --config my_config.yml
```

### Generate Individual Masks
```bash
weightmask science.fits --individual_masks
```