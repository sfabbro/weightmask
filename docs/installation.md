# Installation

weightmask needs Python 3.10 or newer. CI tests 3.13 on linux-64. Runtime
dependencies: numpy, astropy, fitsio, scipy, scikit-image, sep, PyYAML,
astroscrappy.

## pip (PyPI)

```bash
pip install weightmask
weightmask --help
```

Copy [`weightmask.yml`](../weightmask.yml) from the source tree; it is not
installed with the package.

## Pixi (development)

```bash
git clone https://github.com/astroai/weightmask.git
cd weightmask
pixi install
pixi run weightmask --help
pixi run test
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
pip install -e .
weightmask --help
```

A YAML config is required at run time. Unknown top-level YAML sections are
rejected.

If `sep` fails to compile, install a pre-built wheel or `conda install sep`.
Issues: [github.com/astroai/weightmask/issues](https://github.com/astroai/weightmask/issues).
