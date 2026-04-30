# Installation Guide

## System Requirements

WeightMask requires Python 3.10 or higher and the following dependencies:

- numpy
- astropy
- fitsio
- scikit-image
- sep (Source Extractor Python library)
- PyYAML
- astroscrappy

## Installation Methods

### Method 1: Install from Source

```bash
git clone https://github.com/sfabbro/weightmask.git
cd weightmask
pip install -e .
```

### Method 2: Install with Pixi (recommended)

```bash
git clone https://github.com/sfabbro/weightmask.git
cd weightmask
pixi install
```

### Method 3: Install with conda

If you're using conda, you can create a new environment and install the dependencies:

```bash
conda create -n weightmask python=3.10
conda activate weightmask
conda install numpy astropy fitsio scikit-image sep pyyaml
pip install astroscrappy
pip install -e .
```

## Verifying Installation

After installation, you can verify that WeightMask is properly installed by running:

```bash
weightmask --help
```

You can also verify the test suite on a source checkout:

```bash
pixi run test
```

## Testing the Installation

To test the installation with a sample FITS file:

```bash
weightmask sample.fits --config weightmask.yml
```

If you don't have a FITS file, you can create a simple test:

```python
import numpy as np
from astropy.io import fits

# Create a simple test image
data = np.random.poisson(100, (100, 100)).astype(np.float32)
hdu = fits.PrimaryHDU(data)
hdu.writeto('test.fits', overwrite=True)

# Process with weightmask
# weightmask test.fits
```

## Troubleshooting

### Common Issues

1. **Missing Dependencies**: If you encounter import errors, make sure all dependencies are installed:
   ```bash
   pip install numpy astropy fitsio scikit-image sep pyyaml astroscrappy
   ```

2. **Permission Errors**: If you encounter permission errors during installation, try:
   ```bash
   pip install --user weightmask
   ```

3. **Compilation Errors**: Some packages like `sep` require compilation. If you encounter compilation errors, try installing pre-compiled versions:
   ```bash
   conda install sep
   ```

4. **Config Drift**: The supported config surface is the canonical one in `weightmask.yml`. Older sections such as `background` or legacy streak keys are rejected by config validation.

### Getting Help

If you encounter any issues during installation, please check the [GitHub issues](https://github.com/sfabbro/weightmask/issues) or open a new issue if your problem hasn't been reported.
