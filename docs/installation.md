# Installation

weightmask supports Python 3.10 or newer on Linux. CI runs the package smoke
test on Python 3.10 through 3.13 and the full test suite on Linux with Python
3.13. Runtime dependency floors are numpy 1.25, astropy 5.0, fitsio 1.3.0,
scipy 1.9, scikit-image 0.20, sep 1.2, PyYAML 6.0, and astroscrappy 1.2.

## pip (PyPI)

```bash
pip install weightmask
weightmask --help
```

The optional torchfits adapter can be installed with:

```bash
pip install "weightmask[torchfits]"
```

The default configuration is included in the package. Copy it into a working
directory without depending on the source tree:

```bash
python -c "from weightmask.config import copy_default_config; copy_default_config('weightmask.yml')"
```

The same resource is available to Python callers as
`weightmask.config.default_config_resource()`.

## Pixi (development)

```bash
git clone https://github.com/astroai/weightmask.git
cd weightmask
pixi install
pixi run test
# the CLI entry point, from the environment
pixi run python -m weightmask.cli --help
```

## pip from a clone

```bash
git clone https://github.com/astroai/weightmask.git
cd weightmask
pip install -e .
weightmask --help
```

## conda

```bash
conda create -n weightmask python=3.10
conda activate weightmask
conda install numpy astropy fitsio scipy scikit-image sep pyyaml
pip install astroscrappy
pip install weightmask
weightmask --help
```

A YAML config is required at run time. Unknown top-level YAML sections are
rejected.

If `sep` fails to compile, install a pre-built wheel or `conda install sep`.
Issues: [github.com/astroai/weightmask/issues](https://github.com/astroai/weightmask/issues).
