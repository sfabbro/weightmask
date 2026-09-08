# weightmask

[![CI](https://github.com/astroai/weightmask/actions/workflows/ci.yml/badge.svg)](https://github.com/astroai/weightmask/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/weightmask.svg)](https://pypi.org/project/weightmask/)
[![Python](https://img.shields.io/pypi/pyversions/weightmask.svg)](https://pypi.org/project/weightmask/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

weightmask reads a detrended FITS or MEF science image and writes a per-pixel
weight, quality mask, inverse-variance map, and sky.

Those products are for any measurement that should ignore bad pixels and
down-weight the rest: coaddition, shape measurement, forced photometry,
profile fitting, and difference imaging.

## Install

```bash
pip install weightmask
```

Python 3.10+. Copy [`weightmask.yml`](weightmask.yml) into the working directory;
it is not bundled in the wheel.

Pixi (development):

```bash
git clone https://github.com/astroai/weightmask.git
cd weightmask
pixi install
```

Details: [docs/installation.md](docs/installation.md).

## Run

```bash
weightmask science.fits --config weightmask.yml --flat_image flat.fits \
  -o weight.fits --output_mask mask.fits
```

## Docs

- [Installation](docs/installation.md)
- [Usage](docs/usage.md)
- [Algorithms](docs/algorithms.md)
- [API](docs/api.md)
- [CHANGELOG](CHANGELOG.md)

## License

MIT
