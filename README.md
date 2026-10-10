# weightmask

[![CI](https://github.com/astroai/weightmask/actions/workflows/ci.yml/badge.svg)](https://github.com/astroai/weightmask/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/weightmask.svg)](https://pypi.org/project/weightmask/)
[![Python](https://img.shields.io/pypi/pyversions/weightmask.svg)](https://pypi.org/project/weightmask/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/astroai/weightmask/blob/v0.2.1/LICENSE)

weightmask reads a detrended FITS or MEF science image and writes a per-pixel
weight, quality mask, inverse-variance map, and sky.

Those products are for any measurement that should ignore bad pixels and
down-weight the rest: coaddition, shape measurement, forced photometry,
profile fitting, and difference imaging.

## Install

```bash
pip install weightmask
```

Python 3.10+. The canonical `weightmask.yml` is bundled in the wheel and sdist.
Copy it into a working directory with:

```bash
python -c "from weightmask.config import copy_default_config; copy_default_config('weightmask.yml')"
```

Pixi (development):

```bash
git clone https://github.com/astroai/weightmask.git
cd weightmask
pixi install
```

Details: [installation](https://github.com/astroai/weightmask/blob/v0.2.1/docs/installation.md).

## Run

```bash
weightmask science.fits --config weightmask.yml --flat_image flat.fits \
  -o weight.fits --output_mask mask.fits
```

## Docs

- [Installation](https://github.com/astroai/weightmask/blob/v0.2.1/docs/installation.md)
- [Usage](https://github.com/astroai/weightmask/blob/v0.2.1/docs/usage.md)
- [Algorithms](https://github.com/astroai/weightmask/blob/v0.2.1/docs/algorithms.md)
- [API](https://github.com/astroai/weightmask/blob/v0.2.1/docs/api.md)
- [Releasing](https://github.com/astroai/weightmask/blob/v0.2.1/docs/releasing.md)
- [CHANGELOG](https://github.com/astroai/weightmask/blob/v0.2.1/CHANGELOG.md)

## License

MIT
