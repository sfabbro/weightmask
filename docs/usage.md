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
  local_filter_size: 15
  local_low_thresh: 0.5
  local_high_thresh: 2.0
  col_enable: True
  col_deriv_sigma: 10.0
  col_dead_thresh: 0.1
```

#### Saturation Detection
```yaml
saturation:
  method: 'histogram'
  keyword: 'SATURATE'
  effective_full_scale: 65535.0
  histogram_params:
    guard_fraction: 0.75
    max_upper_factor: 1.05
  fallback_level: 65000.0
```

#### Background Estimation
```yaml
sep_background:
  method: 'sep' # Options: 'sep', 'median_filter', 'robust_median_fallback'
  box_size: 128
  auto_box_scaling: true
  filter_size: 3
  iterations: 2 # Number of iterative background/object detection loops
  mask_threshold: 0.8
  max_box_size: 1024
  median_kernel_size: 31 # Kernel size for the median filter
  smooth_surface_fallback: true
```

#### Astroscrappy Cosmic Ray Detection
```yaml
cosmic_ray:
  sigclip: 4.5
  objlim: 5.0
  dynamic_sigclip: true # Auto-adjust sigclip based on background noise
  dynamic_objlim: true
```

#### SEP Object Detection
```yaml
sep_objects:
  extract_thresh: 3.0
  min_area: 10
  deblend_nthresh: 32
  deblend_cont: 0.005
  seed_thresh_factor: 1.25
  ellipse_k: 2.5
  dynamic_halo_scaling: true
  halo_brightness_factor: 0.15
  max_halo_multiplier: 1.8
  handoff_elongated_to_streak: true
```

#### Streak Masking
```yaml
streak_masking:
  enable: True
  mode: 'auto_ground' # Options: 'auto_ground', 'satdet_only', 'mrt_only', 'legacy_compare'
  debug: false
  dilation_radius: 2
  satdet_params:
    rescale_percentiles: [4.5, 93.0]
    gaussian_sigma: 2.0
    gaussian_sigmas: [1.5, 2.0, 3.0]
    canny_low_threshold: 0.1
    canny_high_threshold: 0.35
    small_edge_perimeter: 60
    hough_min_line_length: 120
    hough_max_line_gap: 30
    cluster_angle_tol_deg: 3.0
    cluster_rho_tol_px: 30.0
    edge_buffer: 32
    confidence_threshold: 0.4
  mrt_rescue_params:
    theta_step_deg: 1.0
    peak_threshold_sig: 4.5
    max_candidates: 4
    confidence_threshold: 0.35
  mask_params:
    strip_length: 256
    strip_width: 96
    profile_sigma_threshold: 3.0
    padding: 4
  enable_sparse_ransac: True
  sparse_ransac_params:
    detect_thresh_sig: 5.0
    min_inliers: 10
    min_length: 100
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

## Benchmark Runner

```bash
uv run python -m tests.benchmarks.run --suite synthetic_v2 --with-baselines
uv run python -m tests.benchmarks.run --suite megacam_real
uv run python -m tests.benchmarks.run --suite acs_compare
```
